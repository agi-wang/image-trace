//! Local feature descriptors (Tier 3) and matchers.

pub mod orb;

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

/// Hamming distance between two binary descriptors.
#[inline]
pub fn desc_hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Cross-checked nearest-neighbour matching (BFMatcher crossCheck equivalent).
pub fn match_cross_check(a: &DescriptorSet, b: &DescriptorSet) -> Vec<Match> {
    if a.desc_len != b.desc_len || a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let best_in_b: Vec<usize> = (0..a.len())
        .map(|i| {
            let ra = a.row(i);
            (0..b.len())
                .map(|j| (desc_hamming(ra, b.row(j)), j))
                .min_by_key(|(d, _)| *d)
                .map(|(_, j)| j)
                .unwrap_or(0)
        })
        .collect();
    let mut out = Vec::new();
    for (i, &j) in best_in_b.iter().enumerate() {
        let ra = a.row(i);
        let rb = b.row(j);
        // best match of rb back into a
        let back = (0..a.len())
            .map(|k| (desc_hamming(rb, a.row(k)), k))
            .min_by_key(|(d, _)| *d)
            .map(|(_, k)| k)
            .unwrap_or(usize::MAX);
        if back == i {
            out.push(Match { a_idx: i, b_idx: j, distance: desc_hamming(ra, rb) });
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
    let mut out = Vec::new();
    for i in 0..a.len() {
        let ra = a.row(i);
        let mut best = u32::MAX;
        let mut second = u32::MAX;
        let mut best_j = 0usize;
        for j in 0..b.len() {
            let d = desc_hamming(ra, b.row(j));
            if d < best {
                second = best;
                best = d;
                best_j = j;
            } else if d < second {
                second = d;
            }
        }
        if second < u32::MAX && (best as f64) < ratio * (second as f64) {
            out.push(Match { a_idx: i, b_idx: best_j, distance: best });
        }
    }
    out.sort_by_key(|m| m.distance);
    out
}

/// Similarity score from cross-checked matches — mirrors the original
/// `calculate_descriptor_similarity`: mean distance of top-k matches,
/// normalized by descriptor bit width, mapped to [0,1].
pub fn match_score(a: &DescriptorSet, b: &DescriptorSet, top_k: usize) -> f64 {
    let matches = match_cross_check(a, b);
    if matches.is_empty() {
        return 0.0;
    }
    let top = &matches[..matches.len().min(top_k)];
    let avg = top.iter().map(|m| m.distance as f64).sum::<f64>() / top.len() as f64;
    let norm = (a.desc_len * 8) as f64;
    (1.0 - avg.min(norm) / norm).clamp(0.0, 1.0)
}
