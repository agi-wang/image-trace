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
pub mod metrics;
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
];

/// Algorithms that must contribute at least one vote for a confirmed duplicate
/// pair in smart-compare (cheap and robust gate). ahash/colorhash are excluded:
/// 64-bit ahash collides on ~half of unrelated real photos at 0.85, and
/// colorhash's coarse bins similarly over-fire.
pub const HASH_GATE_ALGOS: &[&str] = &["phash", "dhash", "whash"];

pub fn all_algorithms() -> Vec<&'static str> {
    let mut v = Vec::new();
    v.extend(HASH_ALGOS);
    v.extend(PIXEL_ALGOS);
    v.extend(DESCRIPTOR_ALGOS);
    v.extend(FUSION_ALGOS);
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
}
