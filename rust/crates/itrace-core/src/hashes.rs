//! Perceptual hashes (Tier 1): ahash, dhash, phash, whash, colorhash.
//!
//! All produce 64-bit signatures; similarity = 1 - hamming/64.
//! Semantics follow the classic imagehash definitions closely enough for
//! interchangeable use, but bit-exactness with Python imagehash is not a goal.

use crate::image_io;
use crate::{GrayImage, HashSet, RgbImage};

/// Hamming distance between two 64-bit signatures.
#[inline]
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Similarity in [0,1] from hamming distance over 64 bits.
#[inline]
pub fn hash_similarity(a: u64, b: u64) -> f64 {
    1.0 - hamming(a, b) as f64 / 64.0
}

/// Hash of a single algorithm name applied to a HashSet.
pub fn hash_of(h: &HashSet, algo: &str) -> Option<u64> {
    match algo {
        "phash" => Some(h.phash),
        "dhash" => Some(h.dhash),
        "ahash" => Some(h.ahash),
        "whash" => Some(h.whash),
        "colorhash" => Some(h.colorhash),
        _ => None,
    }
}

/// Convert hash to hex string (DB storage, compatible with old format).
pub fn to_hex(h: u64) -> String {
    format!("{h:016x}")
}

pub fn from_hex(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

/// Compute all five hashes for a decoded image.
pub fn compute_all(gray: &GrayImage, rgb: &RgbImage) -> HashSet {
    HashSet {
        phash: phash(gray),
        dhash: dhash(gray),
        ahash: ahash(gray),
        whash: whash(gray),
        colorhash: colorhash(rgb),
    }
}

// ---------- ahash ----------

/// Average hash: 8×8 mean threshold.
pub fn ahash(gray: &GrayImage) -> u64 {
    let small = image_io::resize_gray_exact(gray, 8, 8);
    let mean: u64 = small.data.iter().map(|&v| v as u64).sum::<u64>() / 64;
    let mut bits = 0u64;
    for (i, &v) in small.data.iter().enumerate() {
        if v as u64 > mean {
            bits |= 1 << i;
        }
    }
    bits
}

// ---------- dhash ----------

/// Difference hash: 9×8 gradient, bit = left > right.
pub fn dhash(gray: &GrayImage) -> u64 {
    let small = image_io::resize_gray_exact(gray, 9, 8);
    let mut bits = 0u64;
    for y in 0..8u32 {
        for x in 0..8u32 {
            if small.get(x, y) > small.get(x + 1, y) {
                bits |= 1 << (y * 8 + x);
            }
        }
    }
    bits
}

// ---------- phash ----------

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Separable 2D DCT-II on an N×N f32 block.
fn dct_2d(input: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    let pi_n = std::f64::consts::PI / n as f64;
    // rows
    let mut tmp = vec![0.0f64; n * n];
    for y in 0..n {
        for u in 0..n {
            let mut s = 0.0;
            for x in 0..n {
                s += input[y * n + x] * ((x as f64 + 0.5) * u as f64 * pi_n).cos();
            }
            tmp[y * n + u] = s * if u == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        }
    }
    // cols
    for v in 0..n {
        for u in 0..n {
            let mut s = 0.0;
            for y in 0..n {
                s += tmp[y * n + u] * ((y as f64 + 0.5) * v as f64 * pi_n).cos();
            }
            out[v * n + u] = s * if v == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        }
    }
    out
}

/// Perceptual hash: 32×32 → DCT-II → top-left 8×8 → median threshold.
pub fn phash(gray: &GrayImage) -> u64 {
    let small = image_io::resize_gray_exact(gray, 32, 32);
    let f: Vec<f64> = small.data.iter().map(|&v| v as f64).collect();
    let dct = dct_2d(&f, 32);
    // low-frequency 8×8
    let mut low = Vec::with_capacity(64);
    for v in 0..8 {
        for u in 0..8 {
            low.push(dct[v * 32 + u]);
        }
    }
    let mut med_src = low.clone();
    let med = median_f64(&mut med_src);
    let mut bits = 0u64;
    for (i, &v) in low.iter().enumerate() {
        if v > med {
            bits |= 1 << i;
        }
    }
    bits
}

// ---------- whash ----------

/// In-place 1D Haar transform step on a row slice.
fn haar_row(data: &mut [f64], stride: usize, len: usize, base: usize) {
    let half = len / 2;
    let mut tmp = vec![0.0f64; len];
    for i in 0..half {
        let a = data[base + (2 * i) * stride];
        let b = data[base + (2 * i + 1) * stride];
        tmp[i] = (a + b) / 2.0;
        tmp[half + i] = (a - b) / 2.0;
    }
    for i in 0..len {
        data[base + i * stride] = tmp[i];
    }
}

/// Full 2D Haar wavelet decomposition of a size×size block (size = power of 2).
fn haar_2d(data: &mut [f64], size: usize) {
    let mut len = size;
    while len > 1 {
        // rows
        for y in 0..len {
            haar_row(data, 1, len, y * size);
        }
        // cols
        for x in 0..len {
            haar_row(data, size, len, x);
        }
        len /= 2;
    }
}

/// Wavelet hash: 32×32 → 2D Haar → zero out DC → low 8×8 of transform → median.
pub fn whash(gray: &GrayImage) -> u64 {
    let small = image_io::resize_gray_exact(gray, 32, 32);
    let mut f: Vec<f64> = small.data.iter().map(|&v| v as f64).collect();
    haar_2d(&mut f, 32);
    // remove DC (top-left) for brightness invariance
    f[0] = 0.0;
    let mut low = Vec::with_capacity(64);
    for v in 0..8 {
        for u in 0..8 {
            low.push(f[v * 32 + u]);
        }
    }
    let mut med_src = low.clone();
    let med = median_f64(&mut med_src);
    let mut bits = 0u64;
    for (i, &v) in low.iter().enumerate() {
        if v > med {
            bits |= 1 << i;
        }
    }
    bits
}

// ---------- colorhash ----------

/// Color signature: HSV histogram (16 hue × 4 sat bins) → bits above median.
/// Robust to mild recolor/compression; complements grayscale hashes.
pub fn colorhash(rgb: &RgbImage) -> u64 {
    let mut bins = [0f64; 64];
    let npix = (rgb.width as usize) * (rgb.height as usize);
    if npix == 0 {
        return 0;
    }
    for i in 0..npix {
        let r = rgb.data[i * 3] as f64 / 255.0;
        let g = rgb.data[i * 3 + 1] as f64 / 255.0;
        let b = rgb.data[i * 3 + 2] as f64 / 255.0;
        let (h, s, v) = rgb_to_hsv(r, g, b);
        // weight by saturation*value so near-gray pixels barely count
        let w = s * v;
        let hb = ((h / 360.0) * 16.0).floor().clamp(0.0, 15.0) as usize;
        let sb = (s * 4.0).floor().clamp(0.0, 3.0) as usize;
        bins[sb * 16 + hb] += w;
    }
    let total: f64 = bins.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    for b in bins.iter_mut() {
        *b /= total;
    }
    let mut med_src: Vec<f64> = bins.to_vec();
    let med = median_f64(&mut med_src);
    let mut bits = 0u64;
    for (i, &v) in bins.iter().enumerate() {
        if v > med {
            bits |= 1 << i;
        }
    }
    bits
}

/// RGB (0-1) → HSV (h: 0-360, s/v: 0-1).
fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max > 0.0 { d / max } else { 0.0 };
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h.rem_euclid(360.0), s, v)
}

/// Compute the file-level features (hashes + metadata) for an image on disk.
pub fn compute_image_features(
    path: &std::path::Path,
) -> anyhow::Result<crate::ImageFeatures> {
    let bytes = std::fs::read(path)?;
    compute_image_features_bytes(&bytes)
}

/// Same, from bytes (works with object-storage backends).
pub fn compute_image_features_bytes(
    bytes: &[u8],
) -> anyhow::Result<crate::ImageFeatures> {
    let file_hash = blake3::hash(bytes).to_hex().to_string();
    let file_size = bytes.len() as u64;
    let img = image_io::decode(bytes)?;
    let (width, height) = image::GenericImageView::dimensions(&img);
    let gray = image_io::to_gray(&img);
    let rgb = image_io::to_rgb(&img);
    let hashes = compute_all(&gray, &rgb);
    Ok(crate::ImageFeatures {
        file_hash,
        file_size,
        width,
        height,
        hashes,
    })
}
