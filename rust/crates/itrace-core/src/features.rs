//! Precomputed feature vectors + batch N×N similarity matrix engine.
//!
//! Modular design: every stored feature is a [`FeatureExtractor`] plugin in
//! [`EXTRACTORS`]. An extractor owns its encoder (`compute`) and its
//! similarity metric (`similarity`), so adding a feature = implementing the
//! trait + one registry entry — no central match statements. Deep-embedding
//! features (e.g. a future DINOv2 ONNX model) slot in the same way.
//!
//! One image × 8 orientation variants × N features → compact binary vectors
//! persisted in SQLite BLOBs. Comparison operates on the vectors only —
//! no image decoding in the hot loop.

use crate::descriptors::orb;
use crate::{hashes, image_io, metrics, GrayImage, RgbImage};
use rayon::prelude::*;
use std::collections::HashMap;

pub const NUM_VARIANTS: u8 = 8;
pub const GRAY_FLAT_SIZE: u32 = 128;

/// How a feature's stored bytes are compared — selects the matrix kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// u64 LE payload, XOR + popcount similarity.
    Bits,
    /// Vector payload (f32 LE or raw u8), cosine similarity.
    Cosine,
}

/// Pluggable feature extractor: computes and compares one stored feature.
pub trait FeatureExtractor: Sync {
    /// `feature_store` key, e.g. `"phash_bits"`.
    fn feature_name(&self) -> &'static str;
    /// Comparison algorithms served by this feature (first = canonical).
    fn algorithms(&self) -> &'static [&'static str];
    fn kind(&self) -> FeatureKind;
    /// Encoded feature bytes for one decoded variant image.
    fn compute(&self, gray: &GrayImage, rgb: &RgbImage) -> Vec<u8>;
    /// Logical dimensionality of an encoded payload (for storage metadata).
    fn dims(&self, data: &[u8]) -> usize;
    /// Similarity in `[0,1]` between two payloads of this feature.
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64;
}

// ---------- encodings ----------

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

// ---------- built-in extractors ----------

/// Any 64-bit perceptual hash (phash/dhash/ahash/whash/colorhash).
struct HashExtractor {
    feature: &'static str,
    algos: &'static [&'static str],
    hash: fn(&GrayImage, &RgbImage) -> u64,
}

impl FeatureExtractor for HashExtractor {
    fn feature_name(&self) -> &'static str {
        self.feature
    }
    fn algorithms(&self) -> &'static [&'static str] {
        self.algos
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Bits
    }
    fn compute(&self, gray: &GrayImage, rgb: &RgbImage) -> Vec<u8> {
        pack_bits((self.hash)(gray, rgb))
    }
    fn dims(&self, _data: &[u8]) -> usize {
        64
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        hashes::hash_similarity(unpack_bits(a), unpack_bits(b))
    }
}

/// 216-dim HSV histogram, cosine distance.
struct HsvHistogramExtractor;

impl FeatureExtractor for HsvHistogramExtractor {
    fn feature_name(&self) -> &'static str {
        "histogram_hsv"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["histogram"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, _gray: &GrayImage, rgb: &RgbImage) -> Vec<u8> {
        pack_f32(&metrics::hsv_histogram(rgb))
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() / 4
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        cosine(&unpack_f32(a), &unpack_f32(b))
    }
}

/// 128×128 raw grayscale raster — the cheap signal behind `ssim`/`template`.
struct GrayFlatExtractor;

impl FeatureExtractor for GrayFlatExtractor {
    fn feature_name(&self) -> &'static str {
        "gray_flat"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["ssim", "template"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        image_io::resize_gray_exact(gray, GRAY_FLAT_SIZE, GRAY_FLAT_SIZE).data
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len()
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        let fa: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let fb: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        cosine(&fa, &fb)
    }
}

/// Mean-pooled ORB descriptor set — coarse global signature for `orb`
/// prefiltering; true geometric matching uses the full descriptor set.
struct OrbPooledExtractor;

impl FeatureExtractor for OrbPooledExtractor {
    fn feature_name(&self) -> &'static str {
        "orb_pooled"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["orb"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        pack_f32(&pool_descriptors(&orb::detect_orb(gray, 512)))
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() / 4
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        cosine(&unpack_f32(a), &unpack_f32(b))
    }
}

// ---------- registry ----------

/// All registered extractors, in canonical order.
pub const EXTRACTORS: &[&dyn FeatureExtractor] = &[
    &HashExtractor { feature: "phash_bits", algos: &["phash"], hash: |g, _| hashes::phash(g) },
    &HashExtractor { feature: "dhash_bits", algos: &["dhash"], hash: |g, _| hashes::dhash(g) },
    &HashExtractor { feature: "ahash_bits", algos: &["ahash"], hash: |g, _| hashes::ahash(g) },
    &HashExtractor { feature: "whash_bits", algos: &["whash"], hash: |g, _| hashes::whash(g) },
    &HashExtractor { feature: "colorhash_bits", algos: &["colorhash"], hash: |_, r| hashes::colorhash(r) },
    &HsvHistogramExtractor,
    &GrayFlatExtractor,
    &OrbPooledExtractor,
];

/// Extractor serving a stored feature name.
pub fn extractor_for_feature(name: &str) -> Option<&'static dyn FeatureExtractor> {
    EXTRACTORS.iter().copied().find(|e| e.feature_name() == name)
}

/// Extractor serving a comparison algorithm (`algo_to_feature` generalized).
pub fn extractor_for_algo(algo: &str) -> Option<&'static dyn FeatureExtractor> {
    EXTRACTORS.iter().copied().find(|e| e.algorithms().contains(&algo))
}

/// All stored feature names, registry order.
pub fn all_features() -> Vec<&'static str> {
    EXTRACTORS.iter().map(|e| e.feature_name()).collect()
}

/// Map comparison algorithm name → stored feature name.
pub fn algo_to_feature(algo: &str) -> Option<&'static str> {
    extractor_for_algo(algo).map(|e| e.feature_name())
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

/// Compute every registered feature for one already-decoded variant image.
pub fn compute_variant_features(gray: &GrayImage, rgb: &RgbImage) -> Vec<(String, Vec<u8>, usize)> {
    EXTRACTORS
        .iter()
        .map(|e| {
            let bytes = e.compute(gray, rgb);
            (e.feature_name().to_string(), bytes.clone(), e.dims(&bytes))
        })
        .collect()
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

/// N×N similarity matrix over stored payloads. variant-max when rot_inv.
/// Pairwise-scored in parallel; `sim` decodes + compares two payloads.
pub fn generic_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    rotation_invariant: bool,
    sim: &(dyn Fn(&[u8], &[u8]) -> f64 + Sync),
) -> Vec<Vec<f64>> {
    let n = ids.len();
    let mut best = vec![vec![0f64; n]; n];
    for (i, row) in best.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let variants: Vec<u8> = if rotation_invariant { (0..NUM_VARIANTS).collect() } else { vec![0] };
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            // cross-variant max: a rotated image's variant list is a
            // permutation of the original's, so the max must range over
            // all (v, w) pairs, not just same-index ones
            let mut mx = 0.0f64;
            let ma = vectors.get(&ids[i]);
            let mb = vectors.get(&ids[j]);
            if let (Some(ma), Some(mb)) = (ma, mb) {
                for &v in &variants {
                    for &w in &variants {
                        if let (Some(a), Some(b)) = (ma.get(&v), mb.get(&w)) {
                            mx = mx.max(sim(a, b));
                        }
                    }
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
/// The registry picks the extractor; its `similarity` drives the matrix.
pub fn similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    algorithm: &str,
    rotation_invariant: bool,
) -> Vec<Vec<f64>> {
    match extractor_for_algo(algorithm) {
        Some(ext) => generic_similarity_matrix(vectors, ids, rotation_invariant, &|a, b| ext.similarity(a, b)),
        None => {
            let n = ids.len();
            let mut m = vec![vec![0f64; n]; n];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            m
        }
    }
}
