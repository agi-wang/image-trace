//! Image Trace core engine.
//!
//! Pure-Rust image similarity & deduplication primitives:
//! perceptual hashes, pixel metrics, local feature descriptors,
//! orientation / slice robustness, union-find grouping.

pub mod compare;
pub mod descriptors;
pub mod documents;
pub mod features;
pub mod group;
pub mod hashes;
pub mod image_io;
pub mod index;
pub mod metrics;
pub mod ownership;
pub mod slice;

use serde::{Deserialize, Serialize};

/// Hash-family algorithms (Tier 1).
pub const HASH_ALGOS: &[&str] = &["phash", "dhash", "ahash", "whash", "colorhash"];
/// Pixel/structure algorithms (Tier 2).
pub const PIXEL_ALGOS: &[&str] = &["ssim", "histogram", "template"];
/// Local descriptor algorithms (Tier 3).
pub const DESCRIPTOR_ALGOS: &[&str] = &["orb", "akaze", "sift"];
/// Weighted fusion.
pub const FUSION_ALGOS: &[&str] = &["auto"];

pub const SMART_ALGOS: &[&str] = &[
    "phash", "dhash", "ahash", "whash", "ssim", "histogram", "orb",
    "edgehash", "blockhash", "colorlayout", "hu", "orbscale", "sliceprofile",
    "crophash",
];

/// Algorithms that must contribute at least one vote for a confirmed duplicate
/// pair in smart-compare (cheap and robust gate). ahash/colorhash are excluded:
/// 64-bit ahash collides on ~half of unrelated real photos at 0.85, and
/// colorhash's coarse bins similarly over-fire.
pub const HASH_GATE_ALGOS: &[&str] = &["phash", "dhash", "whash", "edgehash"];

/// Crop/slice-robust gate for smart-compare: a pair whose votes all come
/// from geometry-blind features still confirms when a crop-aware algorithm
/// fires. Global gate hashes sit at ~0.6 on a 70% crop or a 2×2 slice tile,
/// so the hash gate alone structurally misses containment duplicates.
/// `crophash` (windowed phash keys) is measured at ~0.9–1.0 on same-source
/// crops/slices vs ≤0.8 on unrelated images; `blockhash` is deliberately
/// NOT a gate — its best-overlap tile matching over-fires (~0.88) on
/// unrelated slice tiles, though it still counts as a normal vote.
pub const CROP_GATE_ALGOS: &[&str] = &["crophash"];

/// Smart-compare pair confirmation, shared by the server and CLI paths:
/// enough distinct algorithm votes AND at least one vote from the hash
/// gate — or, for crop/slice near-dups the global hashes cannot see, one
/// vote from the crop-robust gate.
pub fn smart_pair_confirmed<'a>(
    hits: impl IntoIterator<Item = &'a str>,
    min_agree: usize,
) -> bool {
    let mut n = 0usize;
    let mut gate = false;
    let mut crop_gate = false;
    for a in hits {
        n += 1;
        gate |= HASH_GATE_ALGOS.contains(&a);
        crop_gate |= CROP_GATE_ALGOS.contains(&a);
    }
    n >= min_agree && (gate || crop_gate)
}

/// All comparison algorithm names: every registered extractor's algos,
/// plus descriptor algos handled by the precise path even when no extractor
/// registers them (akaze/sift are optional-feature descriptors), plus fusion.
pub fn all_algorithms() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for e in features::EXTRACTORS {
        v.extend_from_slice(e.algorithms());
    }
    for &a in DESCRIPTOR_ALGOS.iter().chain(FUSION_ALGOS) {
        if !v.contains(&a) {
            v.push(a);
        }
    }
    v
}

pub fn is_known_algorithm(a: &str) -> bool {
    all_algorithms().contains(&a)
}

/// Perceptual-hash bit signature of an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashSet {
    pub phash: u64,
    pub dhash: u64,
    pub ahash: u64,
    pub whash: u64,
    pub colorhash: u64,
}

/// Static features computed once per image at upload time.
#[derive(Debug, Clone)]
pub struct ImageFeatures {
    pub file_hash: String, // blake3 hex of file bytes
    pub file_size: u64,
    pub width: u32,
    pub height: u32,
    pub hashes: HashSet,
}

/// A grayscale image buffer in row-major order.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), (width * height) as usize);
        Self { width, height, data }
    }
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.data[(y * self.width + x) as usize]
    }
    /// Row `y` as a `width`-byte slice — hoists the row offset out of
    /// per-pixel loops.
    #[inline]
    pub fn row(&self, y: u32) -> &[u8] {
        let start = y as usize * self.width as usize;
        &self.data[start..start + self.width as usize]
    }
}

/// A BGR-free RGB buffer (used for histogram / color features).
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), (width * height * 3) as usize);
        Self { width, height, data }
    }
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> (u8, u8, u8) {
        let i = ((y * self.width + x) * 3) as usize;
        (self.data[i], self.data[i + 1], self.data[i + 2])
    }
    /// Row `y` as a `3*width`-byte slice of packed RGB triples.
    #[inline]
    pub fn row(&self, y: u32) -> &[u8] {
        let w = self.width as usize * 3;
        let start = y as usize * w;
        &self.data[start..start + w]
    }
}
