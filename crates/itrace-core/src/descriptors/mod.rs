//! Local feature descriptors (Tier 3) and matchers.

pub mod orb;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A detected keypoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Pyramid level the point was found on (0 = base).
    pub level: u8,
    /// Dominant orientation in radians (0 = none).
    pub angle: f32,
    /// Corner/blob response strength.
    pub response: f32,
}

/// A set of descriptors; row-major `data` holds `n * desc_len` bytes.
/// Binary descriptors (ORB/AKAZE) are compared with Hamming distance.
#[derive(Debug, Clone)]
pub struct DescriptorSet {
    pub keypoints: Vec<Keypoint>,
    pub desc_len: usize,
    pub data: Vec<u8>,
}

impl DescriptorSet {
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
    pub fn row(&self, i: usize) -> &[u8] {
        &self.data[i * self.desc_len..(i + 1) * self.desc_len]
    }
}

/// Extractor interface — implementable by ORB, AKAZE, SIFT-backed adapters, etc.
pub trait DescriptorExtractor: Send + Sync {
    fn name(&self) -> &'static str;
    fn detect(&self, gray: &crate::GrayImage, max_features: usize) -> DescriptorSet;
}

/// Available extractor by name.
pub fn extractor_for(algo: &str) -> Option<Box<dyn DescriptorExtractor>> {
    match algo {
        "orb" => Some(Box::new(orb::OrbExtractor)),
        "akaze" => akaze_extractor(),
        "sift" => None,
        _ => None,
    }
}

#[cfg(feature = "akaze")]
fn akaze_extractor() -> Option<Box<dyn DescriptorExtractor>> {
    Some(Box::new(akaze_impl::AkazeExtractor::default()))
}
#[cfg(not(feature = "akaze"))]
fn akaze_extractor() -> Option<Box<dyn DescriptorExtractor>> {
    None
}

/// A cross-checked match (a ↔ b both chose each other).
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub a_idx: usize,
    pub b_idx: usize,
    /// Hamming distance in bits.
    pub distance: u32,
}

/// Hamming distance between two binary descriptors — processed a
/// `u64` word at a time (a 32-byte descriptor is 4 XOR+popcount ops).
#[inline]
pub fn desc_hamming(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut d = 0u32;
    let (ac, arem) = a.as_chunks::<8>();
    let (bc, brem) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        let xa = u64::from_ne_bytes(*x);
        let xb = u64::from_ne_bytes(*y);
        d += (xa ^ xb).count_ones();
    }
    for (x, y) in arem.iter().zip(brem) {
        d += (x ^ y).count_ones();
    }
    d
}

/// Hamming distance with early exit: `Some(d)` iff `d < cap`.
/// Callers pass a running bound (current best / second-best) so
/// rows that can't matter bail after the first chunks.
#[inline]
fn hamming_capped(a: &[u8], b: &[u8], cap: u32) -> Option<u32> {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut d = 0u32;
    let (ac, arem) = a.as_chunks::<8>();
    let (bc, brem) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        let xa = u64::from_ne_bytes(*x);
        let xb = u64::from_ne_bytes(*y);
        d += (xa ^ xb).count_ones();
        if d >= cap {
            return None;
        }
    }
    for (x, y) in arem.iter().zip(brem) {
        d += (x ^ y).count_ones();
    }
    (d < cap).then_some(d)
}

/// Cross-checked nearest-neighbour matching (BFMatcher crossCheck equivalent).
///
/// Two passes, no a×b distance matrix:
/// 1. parallel per-row forward argmin, `hamming_capped` bounded by the
///    running best — `Some(d)` iff `d < best`, so a tie keeps the
///    earlier index, identical to a strict `<` scan over an exact row.
/// 2. backward argmin computed only for the columns some row selected:
///    one capped scan per such column finds the first row attaining the
///    column minimum, bailing entirely as soon as a distance below the
///    candidates' own best appears (then no candidate can be argmin).
pub fn match_cross_check(a: &DescriptorSet, b: &DescriptorSet) -> Vec<Match> {
    if a.desc_len != b.desc_len || a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (an, bn) = (a.len(), b.len());
    // Pass 1 — forward argmin per a-row (first minimum wins).
    let fwd: Vec<(usize, u32)> = (0..an)
        .into_par_iter()
        .map(|i| {
            let ra = a.row(i);
            let mut best = u32::MAX;
            let mut best_j = 0usize;
            for j in 0..bn {
                if let Some(d) = hamming_capped(ra, b.row(j), best) {
                    best = d;
                    best_j = j;
                }
            }
            (best_j, best)
        })
        .collect();
    // For each selected column, the smallest forward distance among its
    // candidate rows — the value a column argmin must attain to match.
    let mut col_min: Vec<u32> = vec![u32::MAX; bn];
    for &(j, d) in &fwd {
        col_min[j] = col_min[j].min(d);
    }
    // Pass 2 — backward argmin, one scan per selected column (parallel).
    // `cap` starts at d_min+1 so the first `Some(d)` is the first row at
    // or below d_min: d < d_min means the column minimum beats every
    // candidate (no match); d == d_min records the argmin and tightens
    // the cap so any later d < d_min still bails. Scan order = row
    // index → first-min-index, same as the old column scan.
    let col_argmin: Vec<Option<usize>> = (0..bn)
        .into_par_iter()
        .map(|j| {
            let d_min = col_min[j];
            if d_min == u32::MAX {
                return None;
            }
            let rb = b.row(j);
            let mut cap = d_min + 1;
            let mut argmin = usize::MAX;
            for k in 0..an {
                if let Some(d) = hamming_capped(a.row(k), rb, cap) {
                    if d < d_min {
                        return None;
                    }
                    argmin = k;
                    cap = d_min;
                }
            }
            Some(argmin)
        })
        .collect();
    let mut out = Vec::new();
    for (i, &(j, d)) in fwd.iter().enumerate() {
        if col_argmin[j] == Some(i) {
            out.push(Match { a_idx: i, b_idx: j, distance: d });
        }
    }
    out.sort_by_key(|m| m.distance);
    out
}

/// kNN match with Lowe ratio test (match_data endpoint).
/// Returns filtered matches a_idx→b_idx.
pub fn match_knn_ratio(a: &DescriptorSet, b: &DescriptorSet, ratio: f64) -> Vec<Match> {
    if a.desc_len != b.desc_len || a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let bn = b.len();
    let mut out: Vec<Match> = (0..a.len())
        .into_par_iter()
        .filter_map(|i| {
            let ra = a.row(i);
            let mut best = u32::MAX;
            let mut second = u32::MAX;
            let mut best_j = 0usize;
            for j in 0..bn {
                // only a distance under the running second-best can
                // still change the outcome — early-exit otherwise
                let Some(d) = hamming_capped(ra, b.row(j), second) else { continue };
                if d < best {
                    second = best;
                    best = d;
                    best_j = j;
                } else {
                    second = d;
                }
            }
            if second < u32::MAX && (best as f64) < ratio * (second as f64) {
                Some(Match { a_idx: i, b_idx: best_j, distance: best })
            } else {
                None
            }
        })
        .collect();
    out.sort_by_key(|m| m.distance);
    out
}

/// Similarity score from cross-checked matches — mirrors the original
/// `calculate_descriptor_similarity`: mean distance of top-k matches,
/// normalized by descriptor bit width — multiplied by a match-density
/// factor so a handful of lucky near-identical descriptors between
/// unrelated images cannot reach a high score.
///
/// `match_cross_check` already returns matches sorted by distance, so
/// the top-k slice needs no further sorting or selection.
pub fn match_score(a: &DescriptorSet, b: &DescriptorSet, top_k: usize) -> f64 {
    let matches = match_cross_check(a, b);
    if matches.is_empty() {
        return 0.0;
    }
    let top = &matches[..matches.len().min(top_k)];
    let avg = top.iter().map(|m| m.distance as f64).sum::<f64>() / top.len() as f64;
    let norm = (a.desc_len * 8) as f64;
    let quality = (1.0 - avg.min(norm) / norm).clamp(0.0, 1.0);
    let density = (matches.len() as f64 / top_k as f64).min(1.0);
    quality * density
}

#[cfg(test)]
mod xcheck_equiv_tests {
    use super::*;

    /// Reference: the old full-matrix implementation.
    fn reference(a: &DescriptorSet, b: &DescriptorSet) -> Vec<Match> {
        if a.desc_len != b.desc_len || a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let (an, bn) = (a.len(), b.len());
        let rows: Vec<(Vec<u32>, usize)> = (0..an)
            .map(|i| {
                let ra = a.row(i);
                let row: Vec<u32> = (0..bn).map(|j| desc_hamming(ra, b.row(j))).collect();
                let mut best = 0usize;
                for (j, &d) in row.iter().enumerate().skip(1) {
                    if d < row[best] {
                        best = j;
                    }
                }
                (row, best)
            })
            .collect();
        let mut col_best: Vec<(u32, usize)> = vec![(u32::MAX, usize::MAX); bn];
        for (k, (row, _)) in rows.iter().enumerate() {
            for (j, &d) in row.iter().enumerate() {
                if d < col_best[j].0 {
                    col_best[j] = (d, k);
                }
            }
        }
        let mut out = Vec::new();
        for (i, (row, j)) in rows.iter().enumerate() {
            if col_best[*j].1 == i {
                out.push(Match { a_idx: i, b_idx: *j, distance: row[*j] });
            }
        }
        out.sort_by_key(|m| m.distance);
        out
    }

    fn set(seed: u64, n: usize, len: usize, dup_every: usize) -> DescriptorSet {
        let mut data = vec![0u8; n * len];
        let mut k = seed;
        for b in data.iter_mut() {
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (k >> 33) as u8;
        }
        // dup_every>1 injects exact-duplicate rows → distance-0 ties
        if dup_every > 1 {
            for r in (dup_every..n).step_by(dup_every) {
                let (dst, src) = (r * len, (r - 1) * len);
                data.copy_within(src..src + len, dst);
            }
        }
        DescriptorSet {
            keypoints: vec![Keypoint { x: 0.0, y: 0.0, level: 0, angle: 0.0, response: 0.0 }; n],
            desc_len: len,
            data,
        }
    }

    #[test]
    fn matches_reference() {
        for (an, bn, len, dup) in [
            (1, 1, 32, 0), (5, 7, 32, 0), (64, 64, 32, 0), (33, 40, 8, 0),
            (50, 50, 32, 2), (40, 40, 32, 3), (17, 9, 33, 0), (200, 180, 32, 4),
        ] {
            let a = set(7, an, len, dup);
            let b = set(13, bn, len, dup);
            let got = match_cross_check(&a, &b);
            let want = reference(&a, &b);
            assert_eq!(got.len(), want.len(), "an={an} bn={bn} len={len} dup={dup}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!((g.a_idx, g.b_idx, g.distance), (w.a_idx, w.b_idx, w.distance),
                    "an={an} bn={bn} len={len} dup={dup}");
            }
        }
    }
}
