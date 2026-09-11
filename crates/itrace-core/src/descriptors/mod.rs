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
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (x, y) in (&mut ac).zip(&mut bc) {
        let xa = u64::from_ne_bytes(x.try_into().unwrap());
        let xb = u64::from_ne_bytes(y.try_into().unwrap());
        d += (xa ^ xb).count_ones();
    }
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
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
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (x, y) in (&mut ac).zip(&mut bc) {
        let xa = u64::from_ne_bytes(x.try_into().unwrap());
        let xb = u64::from_ne_bytes(y.try_into().unwrap());
        d += (xa ^ xb).count_ones();
        if d >= cap {
            return None;
        }
    }
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        d += (x ^ y).count_ones();
    }
    (d < cap).then_some(d)
}

/// Cross-checked nearest-neighbour matching (BFMatcher crossCheck equivalent).
///
/// The a×b distance matrix is computed exactly once (parallel rows);
/// the forward argmin, the column-wise backward argmin and the reported
/// distance all read from it, so each pair distance is computed once.
pub fn match_cross_check(a: &DescriptorSet, b: &DescriptorSet) -> Vec<Match> {
    if a.desc_len != b.desc_len || a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (an, bn) = (a.len(), b.len());
    // matrix rows + per-row forward argmin (first minimum wins)
    let rows: Vec<(Vec<u32>, usize)> = (0..an)
        .into_par_iter()
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
    // backward argmin per column in one sequential pass over the rows
    // (scan order = row index → strict < keeps the first minimum's index)
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
