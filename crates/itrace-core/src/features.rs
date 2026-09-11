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

pub mod blockhash;
pub mod colorlayout;
pub mod edgehash;
pub mod hu;
pub mod orbscale;
pub mod sliceprofile;

pub const NUM_VARIANTS: u8 = 8;
pub const GRAY_FLAT_SIZE: u32 = 128;
/// Working scale for stored features — the same cap `compare.rs` applies to
/// decoded images, so precomputed vectors match the live-comparison signal.
const MAX_SIDE: u32 = 512;

/// How a feature's stored bytes are compared — selects the matrix kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// u64 LE payload, XOR + popcount similarity.
    Bits,
    /// Vector payload (f32 LE or raw u8), cosine similarity.
    Cosine,
}

/// Typed matrix kernel used by [`similarity_matrix`]: each stored blob is
/// decoded once per (image, variant) into a [`Decoded`] and pair-scoring runs
/// on the decoded forms — `score(decode(a), decode(b))` must equal
/// `similarity(a, b)`. An extractor opts in via
/// [`FeatureExtractor::matrix_kernel`]; custom layouts (blockhash,
/// sliceprofile) stay on the per-pair byte path.
#[derive(Debug, Clone, Copy)]
pub enum MatrixKernel {
    /// u64-LE bit signature → `hashes::hash_similarity`.
    Bits,
    /// f32-LE vector → `cosine`.
    Cosine,
    /// f32-LE vector → mean-centered cosine (`orbscale`).
    CenteredCosine,
    /// f32-LE vector → `exp(-L2)` over log-moment space (`hu_moments`).
    ExpDist,
    /// Raw u8 vector → cosine over integer-upcast values (`gray_flat`).
    U8Cosine,
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
    /// True when every dihedral variant yields the same feature (the
    /// histogram/moment features): the 8×8 variant cross-product in
    /// `similarity_matrix` is then redundant and only variant 0 is compared.
    fn rotation_invariant(&self) -> bool {
        false
    }
    /// Typed kernel for decode-once pair scoring in `similarity_matrix`.
    /// `None` keeps the generic per-pair `similarity` path.
    fn matrix_kernel(&self) -> Option<MatrixKernel> {
        None
    }
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

pub(crate) fn pack_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub(crate) fn unpack_f32(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 4);
    out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
    out
}

pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na * nb).sqrt()).clamp(0.0, 1.0)
}

/// Cosine over raw u8 payloads (gray_flat): u8→f64 is exact per element and
/// the accumulation order matches `cosine` — bit-identical result, no
/// intermediate f32 vectors.
fn cosine_u8(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
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
    /// `colorhash` bins are a pixel-multiset histogram — permutation-invariant,
    /// so all orientation variants hash identically.
    rot_inv: bool,
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
    fn rotation_invariant(&self) -> bool {
        self.rot_inv
    }
    fn matrix_kernel(&self) -> Option<MatrixKernel> {
        Some(MatrixKernel::Bits)
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
    fn rotation_invariant(&self) -> bool {
        true // histogram is a pixel multiset — variants permute it exactly
    }
    fn matrix_kernel(&self) -> Option<MatrixKernel> {
        Some(MatrixKernel::Cosine)
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
        cosine_u8(a, b)
    }
    fn matrix_kernel(&self) -> Option<MatrixKernel> {
        Some(MatrixKernel::U8Cosine)
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
    fn matrix_kernel(&self) -> Option<MatrixKernel> {
        Some(MatrixKernel::Cosine)
    }
}

// ---------- registry ----------

/// All registered extractors, in canonical order.
pub const EXTRACTORS: &[&dyn FeatureExtractor] = &[
    &HashExtractor { feature: "phash_bits", algos: &["phash"], hash: |g, _| hashes::phash(g), rot_inv: false },
    &HashExtractor { feature: "dhash_bits", algos: &["dhash"], hash: |g, _| hashes::dhash(g), rot_inv: false },
    &HashExtractor { feature: "ahash_bits", algos: &["ahash"], hash: |g, _| hashes::ahash(g), rot_inv: false },
    &HashExtractor { feature: "whash_bits", algos: &["whash"], hash: |g, _| hashes::whash(g), rot_inv: false },
    &HashExtractor { feature: "colorhash_bits", algos: &["colorhash"], hash: |_, r| hashes::colorhash(r), rot_inv: true },
    &HsvHistogramExtractor,
    &GrayFlatExtractor,
    &OrbPooledExtractor,
    &blockhash::BlockhashExtractor,
    &colorlayout::ColorLayoutExtractor,
    &edgehash::EdgeHashExtractor,
    &hu::HuExtractor,
    &orbscale::OrbScaleExtractor,
    &sliceprofile::SliceProfileExtractor,
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
pub(crate) fn pool_descriptors(descs: &crate::descriptors::DescriptorSet) -> Vec<f32> {
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

/// One stored-feature row: (variant_idx, feature_name, bytes, dims).
type VariantRow = (u8, String, Vec<u8>, usize);

/// Compute every registered feature for one already-decoded variant image.
/// Extractors run in parallel; rows keep registry order.
pub fn compute_variant_features(gray: &GrayImage, rgb: &RgbImage) -> Vec<(String, Vec<u8>, usize)> {
    EXTRACTORS
        .par_iter()
        .map(|e| {
            let bytes = e.compute(gray, rgb);
            let dims = e.dims(&bytes);
            (e.feature_name().to_string(), bytes, dims)
        })
        .collect()
}

/// Compute features for all 8 orientation variants of a decoded image.
/// The image is first downscaled to the `MAX_SIDE` working scale (matching
/// the live-comparison path in `compare.rs`), so stored payloads for inputs
/// larger than 512px change relative to unscaled extraction. Variants run
/// in parallel; rows stay ordered by variant index, then registry order.
/// Returns (variant_idx, feature_name, bytes, dims) rows.
pub fn compute_all_variants(
    img: &image::DynamicImage,
) -> Vec<(u8, String, Vec<u8>, usize)> {
    let small = image_io::resize_max_side(img, MAX_SIDE);
    let rgb_vars = image_io::orientation_variants(&small);
    // Gray variants of the single to_gray — pixel-identical to
    // `to_gray(orientation_variants(&small)[i])` since luma commutes with
    // exact permutations (same argument as compare.rs).
    let gray_vars = image_io::gray_orientation_variants(&image_io::to_gray(&small));
    let per_variant: Vec<Vec<VariantRow>> = (0..rgb_vars.len())
        .into_par_iter()
        .map(|vi| {
            let r = image_io::to_rgb(&rgb_vars[vi]);
            compute_variant_features(&gray_vars[vi], &r)
                .into_iter()
                .map(|(name, bytes, dims)| (vi as u8, name, bytes, dims))
                .collect()
        })
        .collect();
    per_variant.into_iter().flatten().collect()
}

// ---------- matrix engine ----------

/// vectors[image_id][variant_idx] = raw bytes for the mapped feature.
pub type FeatureMap = HashMap<i64, HashMap<u8, Vec<u8>>>;

/// A stored payload decoded once per (image, variant) for the matrix loop.
enum Decoded<'a> {
    Bits(u64),
    F32(Vec<f32>),
    U8(&'a [u8]),
}

impl MatrixKernel {
    fn decode<'a>(self, b: &'a [u8]) -> Decoded<'a> {
        match self {
            Self::Bits => Decoded::Bits(unpack_bits(b)),
            Self::Cosine | Self::CenteredCosine | Self::ExpDist => {
                Decoded::F32(unpack_f32(b))
            }
            Self::U8Cosine => Decoded::U8(b),
        }
    }

    fn score(self, a: &Decoded<'_>, b: &Decoded<'_>) -> f64 {
        match (self, a, b) {
            (Self::Bits, &Decoded::Bits(x), &Decoded::Bits(y)) => {
                hashes::hash_similarity(x, y)
            }
            (Self::Cosine, Decoded::F32(x), Decoded::F32(y)) => cosine(x, y),
            (Self::CenteredCosine, Decoded::F32(x), Decoded::F32(y)) => {
                orbscale::centered_cosine(x, y)
            }
            (Self::ExpDist, Decoded::F32(x), Decoded::F32(y)) => {
                hu::log_moment_similarity(x, y)
            }
            (Self::U8Cosine, Decoded::U8(x), Decoded::U8(y)) => cosine_u8(x, y),
            _ => 0.0,
        }
    }
}

/// Cross-variant max: a rotated image's variant list is a permutation of the
/// original's, so the max must range over all (v, w) pairs, not just
/// same-index ones. Every kernel scores ≤ 1.0, so once `mx` reaches 0.9999
/// no later pair can move the reported score by more than 1e-4 — stop early.
fn variant_max<T>(
    ma: &HashMap<u8, T>,
    mb: &HashMap<u8, T>,
    variants: &[u8],
    score: impl Fn(&T, &T) -> f64,
) -> f64 {
    let mut mx = 0.0f64;
    'outer: for &v in variants {
        for &w in variants {
            if let (Some(a), Some(b)) = (ma.get(&v), mb.get(&w)) {
                mx = mx.max(score(a, b));
                if mx >= 0.9999 {
                    break 'outer;
                }
            }
        }
    }
    mx
}

/// Symmetric N×N matrix with unit diagonal filled from parallel pair scores.
fn pair_matrix(n: usize, score: &(dyn Fn(usize, usize) -> f64 + Sync)) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0f64; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, score(i, j)))
        .collect();
    for (i, j, s) in results {
        m[i][j] = s;
        m[j][i] = s;
    }
    m
}

/// N×N similarity matrix over stored payloads. variant-max when rot_inv.
/// Pairwise-scored in parallel; `sim` decodes + compares two payloads.
pub fn generic_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    rotation_invariant: bool,
    sim: &(dyn Fn(&[u8], &[u8]) -> f64 + Sync),
) -> Vec<Vec<f64>> {
    let variants: Vec<u8> = if rotation_invariant { (0..NUM_VARIANTS).collect() } else { vec![0] };
    pair_matrix(ids.len(), &|i, j| {
        match (vectors.get(&ids[i]), vectors.get(&ids[j])) {
            (Some(ma), Some(mb)) => variant_max(ma, mb, &variants, |a, b| sim(a, b)),
            _ => 0.0,
        }
    })
}

/// N×N matrix on a [`MatrixKernel`]: decode each stored blob once per
/// (image, variant), then score the decoded forms — identical values to the
/// raw-byte path without per-pair unpack allocations.
fn decoded_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    rotation_invariant: bool,
    kernel: MatrixKernel,
) -> Vec<Vec<f64>> {
    let decoded: HashMap<i64, HashMap<u8, Decoded>> = ids
        .iter()
        .filter_map(|&id| {
            vectors.get(&id).map(|m| {
                (id, m.iter().map(|(&v, b)| (v, kernel.decode(b))).collect())
            })
        })
        .collect();
    let variants: Vec<u8> = if rotation_invariant { (0..NUM_VARIANTS).collect() } else { vec![0] };
    pair_matrix(ids.len(), &|i, j| {
        match (decoded.get(&ids[i]), decoded.get(&ids[j])) {
            (Some(ma), Some(mb)) => variant_max(ma, mb, &variants, |a, b| kernel.score(a, b)),
            _ => 0.0,
        }
    })
}

/// Entry point: N×N similarity matrix for an algorithm over stored features.
/// The registry picks the extractor; extractors whose feature is already
/// orientation-invariant (histogram/colorhash/hu) collapse the variant
/// cross-product to variant 0 — identical payloads make every (v, w) score
/// equal. Typed kernels decode each blob once; custom layouts fall back to
/// the generic per-pair byte path.
pub fn similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    algorithm: &str,
    rotation_invariant: bool,
) -> Vec<Vec<f64>> {
    match extractor_for_algo(algorithm) {
        Some(ext) => {
            let rot_inv = rotation_invariant && !ext.rotation_invariant();
            match ext.matrix_kernel() {
                Some(k) => decoded_similarity_matrix(vectors, ids, rot_inv, k),
                None => generic_similarity_matrix(vectors, ids, rot_inv, &|a, b| ext.similarity(a, b)),
            }
        }
        None => pair_matrix(ids.len(), &|_, _| 0.0),
    }
}
