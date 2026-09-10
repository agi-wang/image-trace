//! Precomputed feature vectors + batch N×N similarity matrix engine.
//!
//! One image × 8 orientation variants × N features → compact binary vectors
//! persisted in SQLite BLOBs. Comparison operates on the vectors only —
//! no image decoding in the hot loop.

use crate::descriptors::orb;
use crate::{hashes, image_io, metrics, GrayImage, RgbImage};
use rayon::prelude::*;
use std::collections::HashMap;

/// Feature names stored in feature_store.
pub const HASH_FEATURES: &[&str] =
    &["phash_bits", "dhash_bits", "ahash_bits", "whash_bits", "colorhash_bits"];
pub const PIXEL_FEATURES: &[&str] = &["histogram_hsv", "gray_flat"];
pub const DESCRIPTOR_FEATURES: &[&str] = &["orb_pooled"];
pub const ALL_FEATURES: &[&str] = &[
    "phash_bits", "dhash_bits", "ahash_bits", "whash_bits", "colorhash_bits",
    "histogram_hsv", "gray_flat", "orb_pooled",
];

pub const NUM_VARIANTS: u8 = 8;
pub const GRAY_FLAT_SIZE: u32 = 128;

/// Map comparison algorithm name → stored feature name.
pub fn algo_to_feature(algo: &str) -> Option<&'static str> {
    match algo {
        "phash" => Some("phash_bits"),
        "dhash" => Some("dhash_bits"),
        "ahash" => Some("ahash_bits"),
        "whash" => Some("whash_bits"),
        "colorhash" => Some("colorhash_bits"),
        "histogram" => Some("histogram_hsv"),
        "ssim" => Some("gray_flat"),
        "template" => Some("gray_flat"),
        "orb" => Some("orb_pooled"),
        _ => None,
    }
}

/// One stored feature vector (binary-encoded).
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub algorithm: String,
    pub variant_idx: u8,
    pub dims: usize,
    /// Encoding: u64 LE for *_bits; f32 LE for float features; raw u8 for gray_flat.
    pub data: Vec<u8>,
}

pub fn pack_bits(h: u64) -> Vec<u8> {
    h.to_le_bytes().to_vec()
}

pub fn unpack_bits(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..8].copy_from_slice(&b[..8.min(b.len())]);
    u64::from_le_bytes(buf)
}

fn pack_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn unpack_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Mean-pool a descriptor set into one f32 vector of desc_len dims.
fn pool_descriptors(descs: &crate::descriptors::DescriptorSet) -> Vec<f32> {
    let n = descs.len();
    if n == 0 {
        return vec![0.0; descs.desc_len];
    }
    let mut out = vec![0f32; descs.desc_len];
    for i in 0..n {
        let row = descs.row(i);
        for (k, &b) in row.iter().enumerate() {
            out[k] += b as f32;
        }
    }
    for v in out.iter_mut() {
        *v /= n as f32;
    }
    out
}

/// Compute every feature for one already-decoded variant image.
pub fn compute_variant_features(gray: &GrayImage, rgb: &RgbImage) -> Vec<(String, Vec<u8>, usize)> {
    let mut out = Vec::new();
    let hs = hashes::compute_all(gray, rgb);
    out.push(("phash_bits".to_string(), pack_bits(hs.phash), 64));
    out.push(("dhash_bits".to_string(), pack_bits(hs.dhash), 64));
    out.push(("ahash_bits".to_string(), pack_bits(hs.ahash), 64));
    out.push(("whash_bits".to_string(), pack_bits(hs.whash), 64));
    out.push(("colorhash_bits".to_string(), pack_bits(hs.colorhash), 64));

    let hist = metrics::hsv_histogram(rgb);
    out.push(("histogram_hsv".to_string(), pack_f32(&hist), hist.len()));

    let g = image_io::resize_gray_exact(gray, GRAY_FLAT_SIZE, GRAY_FLAT_SIZE);
    out.push(("gray_flat".to_string(), g.data.clone(), g.data.len()));

    let descs = orb::detect_orb(gray, 512);
    let pooled = pool_descriptors(&descs);
    out.push(("orb_pooled".to_string(), pack_f32(&pooled), pooled.len()));
    out
}

/// Compute features for all 8 orientation variants of a decoded image.
/// Returns (variant_idx, feature_name, bytes, dims) rows.
pub fn compute_all_variants(
    img: &image::DynamicImage,
) -> Vec<(u8, String, Vec<u8>, usize)> {
    let variants = image_io::orientation_variants(img);
    let mut rows = Vec::new();
    for (vi, v) in variants.iter().enumerate() {
        let g = image_io::to_gray(v);
        let r = image_io::to_rgb(v);
        for (name, bytes, dims) in compute_variant_features(&g, &r) {
            rows.push((vi as u8, name, bytes, dims));
        }
    }
    rows
}

// ---------- matrix engine ----------

/// vectors[image_id][variant_idx] = raw bytes for the mapped feature.
pub type FeatureMap = HashMap<i64, HashMap<u8, Vec<u8>>>;

fn is_bits_feature(name: &str) -> bool {
    name.ends_with("_bits")
}

/// N×N hash-similarity matrix (XOR + popcount). variant-max when rot_inv.
pub fn hash_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    rotation_invariant: bool,
) -> Vec<Vec<f64>> {
    let n = ids.len();
    let mut best = vec![vec![0f64; n]; n];
    let variants: Vec<u8> = if rotation_invariant { (0..NUM_VARIANTS).collect() } else { vec![0] };
    for (i, row) in best.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut mx = 0.0f64;
            for &v in &variants {
                let ha = vectors.get(&ids[i]).and_then(|m| m.get(&v)).map(|b| unpack_bits(b));
                let hb = vectors.get(&ids[j]).and_then(|m| m.get(&v)).map(|b| unpack_bits(b));
                if let (Some(a), Some(b)) = (ha, hb) {
                    mx = mx.max(hashes::hash_similarity(a, b));
                }
            }
            (i, j, mx)
        })
        .collect();
    for (i, j, s) in results {
        best[i][j] = s;
        best[j][i] = s;
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na * nb).sqrt()).clamp(0.0, 1.0)
}

fn to_f32(feature: &str, bytes: &[u8]) -> Vec<f32> {
    if feature == "gray_flat" {
        bytes.iter().map(|&b| b as f32).collect()
    } else {
        unpack_f32(bytes)
    }
}

/// N×N cosine-similarity matrix for float/u8 features. variant-max when rot_inv.
pub fn cosine_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    feature: &str,
    rotation_invariant: bool,
) -> Vec<Vec<f64>> {
    let n = ids.len();
    let mut best = vec![vec![0f64; n]; n];
    for (i, row) in best.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let variants: Vec<u8> = if rotation_invariant { (0..NUM_VARIANTS).collect() } else { vec![0] };
    // decode once
    let decoded: Vec<HashMap<u8, Vec<f32>>> = ids
        .iter()
        .map(|id| {
            let mut m = HashMap::new();
            if let Some(vmap) = vectors.get(id) {
                for (&v, b) in vmap {
                    m.insert(v, to_f32(feature, b));
                }
            }
            m
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut mx = 0.0f64;
            for &v in &variants {
                if let (Some(a), Some(b)) = (decoded[i].get(&v), decoded[j].get(&v)) {
                    mx = mx.max(cosine(a, b));
                }
            }
            (i, j, mx)
        })
        .collect();
    for (i, j, s) in results {
        best[i][j] = s;
        best[j][i] = s;
    }
    best
}

/// Entry point: N×N similarity matrix for an algorithm over stored features.
pub fn similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    algorithm: &str,
    rotation_invariant: bool,
) -> Vec<Vec<f64>> {
    let n = ids.len();
    match algo_to_feature(algorithm) {
        None => {
            let mut m = vec![vec![0f64; n]; n];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            m
        }
        Some(feat) if is_bits_feature(feat) => {
            hash_similarity_matrix(vectors, ids, rotation_invariant)
        }
        Some(feat) => cosine_similarity_matrix(vectors, ids, feat, rotation_invariant),
    }
}


