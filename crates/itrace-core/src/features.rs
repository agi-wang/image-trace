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
use std::sync::OnceLock;

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
/// `similarity(a, b)`. Every registered extractor opts in via
/// [`FeatureExtractor::matrix_kernel`]; the custom layouts decode to their
/// own forms (`blockhash` → `[u64; 32]` tile-row masks, `sliceprofile` →
/// its `Profile` + hoisted means).
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
    /// 128-byte blockhash tile grid → row masks → best-overlap Hamming.
    Blockhash,
    /// 128-byte slice profile → `Profile` + means → transform/correlation.
    SliceProfile,
}

/// Shared per-variant intermediates handed to [`FeatureExtractor::compute_ctx`].
/// Extractors that would otherwise repeat the same sub-computation (phash and
/// whash both start from a 32×32 grayscale downscale) take it from here, so
/// one variant pays for it once. Everything is lazy — extractors that don't
/// need an intermediate never trigger it — and every accessor returns exactly
/// what the corresponding `image_io`/`hashes` call produces.
pub struct VariantCtx<'a> {
    /// The variant's grayscale raster.
    pub gray: &'a GrayImage,
    /// The variant's RGB raster.
    pub rgb: &'a RgbImage,
    small32: OnceLock<GrayImage>,
}

impl<'a> VariantCtx<'a> {
    fn new(gray: &'a GrayImage, rgb: &'a RgbImage) -> Self {
        Self {
            gray,
            rgb,
            small32: OnceLock::new(),
        }
    }

    /// The shared 32×32 grayscale downscale — identical to
    /// `image_io::resize_gray_exact(gray, 32, 32)`, built on first use.
    /// `phash` and `whash` are defined as `_small(resize32(gray))`, so they
    /// consume this instead of resizing privately.
    pub fn small32(&self) -> &GrayImage {
        self.small32
            .get_or_init(|| image_io::resize_gray_exact(self.gray, 32, 32))
    }
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
    /// `compute` with shared per-variant intermediates available; the
    /// default delegates to `compute(ctx.gray, ctx.rgb)`. Overrides must
    /// return byte-identical payloads to `compute`.
    fn compute_ctx(&self, ctx: &VariantCtx<'_>) -> Vec<u8> {
        self.compute(ctx.gray, ctx.rgb)
    }
    /// Logical dimensionality of an encoded payload (for storage metadata).
    fn dims(&self, data: &[u8]) -> usize;
    /// Similarity in `[0,1]` between two payloads of this feature.
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64;
    /// True when every dihedral variant yields the same feature (the
    /// histogram/moment features): the 8×8 variant cross-product in
    /// `similarity_matrix` is then redundant and only variant 0 is compared,
    /// and `compute_all_variants` stores only the variant-0 row.
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
    // zip over the n-slices: same forward accumulation order, no per-element
    // bounds checks.
    for (&x, &y) in a[..n].iter().zip(&b[..n]) {
        let (x, y) = (x as f64, y as f64);
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
    for (&x, &y) in a[..n].iter().zip(&b[..n]) {
        let (x, y) = (x as f64, y as f64);
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
    /// When set, `compute_ctx` evaluates the hash on the variant's shared
    /// 32×32 downscale instead — valid because `phash`/`whash` are defined
    /// as `*_small(resize_gray_exact(gray, 32, 32))`, so the bit output is
    /// identical while the resize happens once per variant, not per hash.
    small32_hash: Option<fn(&GrayImage) -> u64>,
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
    fn compute_ctx(&self, ctx: &VariantCtx<'_>) -> Vec<u8> {
        match self.small32_hash {
            Some(h32) => pack_bits(h32(ctx.small32())),
            None => self.compute(ctx.gray, ctx.rgb),
        }
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
    &HashExtractor {
        feature: "phash_bits",
        algos: &["phash"],
        hash: |g, _| hashes::phash(g),
        small32_hash: Some(hashes::phash_small),
        rot_inv: false,
    },
    &HashExtractor {
        feature: "dhash_bits",
        algos: &["dhash"],
        hash: |g, _| hashes::dhash(g),
        small32_hash: None,
        rot_inv: false,
    },
    &HashExtractor {
        feature: "ahash_bits",
        algos: &["ahash"],
        hash: |g, _| hashes::ahash(g),
        small32_hash: None,
        rot_inv: false,
    },
    &HashExtractor {
        feature: "whash_bits",
        algos: &["whash"],
        hash: |g, _| hashes::whash(g),
        small32_hash: Some(hashes::whash_small),
        rot_inv: false,
    },
    &HashExtractor {
        feature: "colorhash_bits",
        algos: &["colorhash"],
        hash: |_, r| hashes::colorhash(r),
        small32_hash: None,
        rot_inv: true,
    },
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

/// Shared body of [`compute_variant_features`]: extractors run in parallel,
/// rows keep registry order. `include_invariant` gates rotation-invariant
/// extractors — they produce byte-identical output on every variant, so
/// `compute_all_variants` computes them only on variant 0.
///
/// Note on shared HSV: `colorhash` (hashes.rs) bins pixels in f64
/// (`(h/360)*16`, `s*4`, weight `s*v`) while `histogram_hsv`
/// (metrics::hsv_histogram) bins in f32 (`(h/180)*50`, `s*60`, weight 1).
/// Sharing one per-pixel (h,s,v) would force one side through a precision
/// conversion (f64↔f32 double-rounding changes bin indices and stored
/// bits), so the two conversions can't be merged under the bit-identical
/// constraint — only the trivially cheap pixel iteration would be shared.
fn variant_features(
    gray: &GrayImage,
    rgb: &RgbImage,
    include_invariant: bool,
) -> Vec<(String, Vec<u8>, usize)> {
    // One ctx per variant: extractors share its lazy intermediates (the
    // phash/whash 32×32 downscale) across the parallel fan-out.
    let ctx = VariantCtx::new(gray, rgb);
    EXTRACTORS
        .par_iter()
        .filter(move |e| include_invariant || !e.rotation_invariant())
        .map(|e| {
            let bytes = e.compute_ctx(&ctx);
            let dims = e.dims(&bytes);
            (e.feature_name().to_string(), bytes, dims)
        })
        .collect()
}

/// Compute every registered feature for one already-decoded variant image.
/// Extractors run in parallel; rows keep registry order.
pub fn compute_variant_features(gray: &GrayImage, rgb: &RgbImage) -> Vec<(String, Vec<u8>, usize)> {
    variant_features(gray, rgb, true)
}

/// Compute features for the orientation variants of a decoded image.
/// The image is first downscaled to the `MAX_SIDE` working scale (matching
/// the live-comparison path in `compare.rs`), so stored payloads for inputs
/// larger than 512px change relative to unscaled extraction. Variants run
/// in parallel; rows stay ordered by variant index, then registry order.
/// Returns (variant_idx, feature_name, bytes, dims) rows.
///
/// Rotation-invariant extractors ([`FeatureExtractor::rotation_invariant`]
/// — currently `histogram_hsv`, `hu_moments`, `colorhash_bits`) emit
/// byte-identical payloads on all 8 variants, so they are computed and
/// emitted only for variant 0 — 14 rows on variant 0, the 11 remaining
/// extractors on variants 1..8 (91 rows total, down from 112). Readers
/// tolerate missing variant rows: the matrix path for invariant features
/// compares only variant 0 anyway, and variant iteration is driven by the
/// keys present in each image's feature map.
pub fn compute_all_variants(img: &image::DynamicImage) -> Vec<(u8, String, Vec<u8>, usize)> {
    let small = image_io::resize_max_side(img, MAX_SIDE);
    // `to_rgb`/`to_gray` are per-pixel maps — they commute with the dihedral
    // permutations, so a single conversion feeds a pure pixel-triple shuffle
    // for the other 7 variants. Byte-identical to the old
    // `to_rgb(orientation_variants(&small)[i])` /
    // `to_gray(orientation_variants(&small)[i])` pair (proved on real
    // Rgb8/Rgba8/Luma8 inputs by
    // `image_io::tests::rgb_variants_match_dynamicimage_path`), while
    // skipping 8 DynamicImage clones and 7 redundant to_rgb8 conversions.
    let rgb_vars = image_io::rgb_orientation_variants(&image_io::to_rgb(&small));
    let gray_vars = image_io::gray_orientation_variants(&image_io::to_gray(&small));
    let per_variant: Vec<Vec<VariantRow>> = (0..rgb_vars.len())
        .into_par_iter()
        .map(|vi| {
            variant_features(&gray_vars[vi], &rgb_vars[vi], vi == 0)
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
    /// `blockhash` payload unpacked to its 32 tile-row bitmasks.
    Blockhash(Box<[u64; 32]>),
    /// `sliceprofile` payload decoded to its `Profile` (+ means), or the raw
    /// bytes when malformed so the cosine fallback is reproduced exactly.
    SliceProfile(sliceprofile::MatrixDecoded<'a>),
}

impl MatrixKernel {
    fn decode<'a>(self, b: &'a [u8]) -> Decoded<'a> {
        match self {
            Self::Bits => Decoded::Bits(unpack_bits(b)),
            Self::Cosine | Self::CenteredCosine | Self::ExpDist => Decoded::F32(unpack_f32(b)),
            Self::U8Cosine => Decoded::U8(b),
            Self::Blockhash => Decoded::Blockhash(Box::new(blockhash::payload_rows(b))),
            Self::SliceProfile => Decoded::SliceProfile(sliceprofile::decode_for_matrix(b)),
        }
    }

    fn score(self, a: &Decoded<'_>, b: &Decoded<'_>) -> f64 {
        match (self, a, b) {
            (Self::Bits, &Decoded::Bits(x), &Decoded::Bits(y)) => hashes::hash_similarity(x, y),
            (Self::Cosine, Decoded::F32(x), Decoded::F32(y)) => cosine(x, y),
            (Self::CenteredCosine, Decoded::F32(x), Decoded::F32(y)) => {
                orbscale::centered_cosine(x, y)
            }
            (Self::ExpDist, Decoded::F32(x), Decoded::F32(y)) => hu::log_moment_similarity(x, y),
            (Self::U8Cosine, Decoded::U8(x), Decoded::U8(y)) => cosine_u8(x, y),
            (Self::Blockhash, Decoded::Blockhash(x), Decoded::Blockhash(y)) => {
                blockhash::rows_similarity(x, y)
            }
            (Self::SliceProfile, Decoded::SliceProfile(x), Decoded::SliceProfile(y)) => {
                sliceprofile::similarity_decoded(x, y)
            }
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

/// Shared streaming pair core: score every upper-triangle pair (i < j) in
/// parallel and collect the (i, j, score) triples `keep` accepts, in (i, j)
/// order. `pair_matrix` and `similarity_pairs_above` are the two consumers —
/// the score closure decides which payloads/variants feed each comparison.
fn pair_select(
    n: usize,
    score: &(dyn Fn(usize, usize) -> f64 + Sync),
    keep: &(dyn Fn(f64) -> bool + Sync),
) -> Vec<(usize, usize, f64)> {
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
        .filter_map(|(i, j)| {
            let s = score(i, j);
            if keep(s) {
                Some((i, j, s))
            } else {
                None
            }
        })
        .collect()
}

/// Symmetric N×N matrix with unit diagonal filled from parallel pair scores.
fn pair_matrix(n: usize, score: &(dyn Fn(usize, usize) -> f64 + Sync)) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0f64; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for (i, j, s) in pair_select(n, score, &|_| true) {
        m[i][j] = s;
        m[j][i] = s;
    }
    m
}

/// All 8 variant indices — the rotation-aware cross-product domain.
const ALL_VARIANTS: [u8; NUM_VARIANTS as usize] = [0, 1, 2, 3, 4, 5, 6, 7];
/// Variant 0 only — the non-rotation-aware domain.
const BASE_VARIANT: [u8; 1] = [0];

/// The variant list a comparison scans: all 8 for the rotation-aware
/// cross-product, just variant 0 otherwise.
fn compared_variants(rotation_invariant: bool) -> &'static [u8] {
    if rotation_invariant {
        &ALL_VARIANTS
    } else {
        &BASE_VARIANT
    }
}

/// N×N similarity matrix over stored payloads. variant-max when rot_inv.
/// Pairwise-scored in parallel; `sim` decodes + compares two payloads.
pub fn generic_similarity_matrix(
    vectors: &FeatureMap,
    ids: &[i64],
    rotation_invariant: bool,
    sim: &(dyn Fn(&[u8], &[u8]) -> f64 + Sync),
) -> Vec<Vec<f64>> {
    let variants = compared_variants(rotation_invariant);
    // Position-aligned view of `vectors` — pair scoring then indexes a Vec
    // slot per side instead of two `HashMap<i64>` probes per pair.
    let maps: Vec<Option<&HashMap<u8, Vec<u8>>>> =
        ids.iter().map(|id| vectors.get(id)).collect();
    pair_matrix(
        ids.len(),
        &|i, j| match (maps[i], maps[j]) {
            (Some(ma), Some(mb)) => variant_max(ma, mb, variants, |a, b| sim(a, b)),
            _ => 0.0,
        },
    )
}

/// Variant-indexed array of decoded payloads for one image — only the
/// compared variants are decoded (a rotation-blind run never reads slots
/// 1..8), and the inner pair loop indexes arrays instead of probing maps.
type VariantDecoded<'a> = [Option<Decoded<'a>>; NUM_VARIANTS as usize];

/// Decode each stored blob once per (image, compared variant), position-
/// aligned to `ids` — pair scoring then indexes `Vec` slots instead of
/// probing `HashMap`s for both the id and the variant.
fn decoded_rows<'a>(
    vectors: &'a FeatureMap,
    ids: &[i64],
    kernel: MatrixKernel,
    variants: &[u8],
) -> Vec<Option<VariantDecoded<'a>>> {
    ids.iter()
        .map(|&id| {
            vectors.get(&id).map(|m| {
                let mut arr: VariantDecoded<'_> = std::array::from_fn(|_| None);
                for &v in variants {
                    if let Some(b) = m.get(&v) {
                        arr[v as usize] = Some(kernel.decode(b));
                    }
                }
                arr
            })
        })
        .collect()
}

/// `variant_max` over pre-decoded per-variant arrays — direct indexing
/// instead of per-pair HashMap probes. `variants` is always a subset of
/// `0..NUM_VARIANTS` (see `compared_variants`), so indexing is in-range.
fn variant_max_decoded(
    ma: &VariantDecoded<'_>,
    mb: &VariantDecoded<'_>,
    variants: &[u8],
    kernel: MatrixKernel,
) -> f64 {
    let mut mx = 0.0f64;
    'outer: for &v in variants {
        for &w in variants {
            if let (Some(a), Some(b)) = (ma[v as usize].as_ref(), mb[w as usize].as_ref()) {
                mx = mx.max(kernel.score(a, b));
                if mx >= 0.9999 {
                    break 'outer;
                }
            }
        }
    }
    mx
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
    let variants = compared_variants(rotation_invariant);
    let decoded = decoded_rows(vectors, ids, kernel, variants);
    pair_matrix(
        ids.len(),
        &|i, j| match (&decoded[i], &decoded[j]) {
            (Some(ma), Some(mb)) => variant_max_decoded(ma, mb, variants, kernel),
            _ => 0.0,
        },
    )
}

/// Entry point: N×N similarity matrix for an algorithm over stored features.
/// The registry picks the extractor; extractors whose feature is already
/// orientation-invariant (histogram/colorhash/hu) collapse the variant
/// cross-product to variant 0 — identical payloads make every (v, w) score
/// equal. Every registered extractor has a typed kernel, so each blob is
/// decoded once per (image, compared variant); the generic per-pair byte
/// path remains for unknown/custom callers of `generic_similarity_matrix`.
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
                None => {
                    generic_similarity_matrix(vectors, ids, rot_inv, &|a, b| ext.similarity(a, b))
                }
            }
        }
        None => pair_matrix(ids.len(), &|_, _| 0.0),
    }
}

/// Streaming sibling of [`similarity_matrix`]: emit only the upper-triangle
/// pairs whose score reaches `threshold`, without materializing the N×N
/// matrix. Returns `(i, j, score)` triples in (i, j) order, where `i`/`j`
/// index into `ids`. Scoring is identical to the matrix path — same
/// decode-once kernels, same variant cross-product with the 0.9999
/// early-exit — so the result equals the ≥-threshold cells of
/// `similarity_matrix` on the same inputs.
pub fn similarity_pairs_above(
    vectors: &FeatureMap,
    ids: &[i64],
    algorithm: &str,
    rotation_invariant: bool,
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let keep = move |s: f64| s >= threshold;
    match extractor_for_algo(algorithm) {
        Some(ext) => {
            let variants = compared_variants(rotation_invariant && !ext.rotation_invariant());
            match ext.matrix_kernel() {
                Some(k) => {
                    let decoded = decoded_rows(vectors, ids, k, variants);
                    pair_select(
                        ids.len(),
                        &|i, j| match (&decoded[i], &decoded[j]) {
                            (Some(ma), Some(mb)) => variant_max_decoded(ma, mb, variants, k),
                            _ => 0.0,
                        },
                        &keep,
                    )
                }
                None => {
                    let maps: Vec<Option<&HashMap<u8, Vec<u8>>>> =
                        ids.iter().map(|id| vectors.get(id)).collect();
                    pair_select(
                        ids.len(),
                        &|i, j| match (maps[i], maps[j]) {
                            (Some(ma), Some(mb)) => {
                                variant_max(ma, mb, variants, |a, b| ext.similarity(a, b))
                            }
                            _ => 0.0,
                        },
                        &keep,
                    )
                }
            }
        }
        // unknown algorithm mirrors the matrix path's 0.0-filled off-diagonal
        None => pair_select(ids.len(), &|_, _| 0.0, &keep),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FeatureMap of `ids` × `variants` filled from a byte pattern
    /// derived per (id, variant) — deterministic, no fixtures.
    fn map_with(ids: &[i64], variants: &[u8], dims: usize) -> FeatureMap {
        ids.iter()
            .map(|&id| {
                let vm = variants
                    .iter()
                    .map(|&v| {
                        let bytes: Vec<u8> = (0..dims)
                            .map(|k| {
                                (id as u8)
                                    .wrapping_mul(31)
                                    .wrapping_add(v)
                                    .wrapping_add(k as u8)
                            })
                            .collect();
                        (v, bytes)
                    })
                    .collect();
                (id, vm)
            })
            .collect()
    }

    /// `similarity_pairs_above` must equal the ≥-threshold upper-triangle
    /// cells of `similarity_matrix` — across the simple kernels (phash →
    /// Bits, histogram → Cosine) and the custom-layout kernels (blockhash →
    /// tile-row masks, sliceprofile → Profile).
    #[test]
    fn pairs_above_matches_matrix_cells() {
        let ids = [10i64, 11, 12, 13];
        let variants: Vec<u8> = (0..NUM_VARIANTS).collect();
        for (algo, dims) in [
            ("phash", 8usize),
            ("blockhash", 128),
            ("sliceprofile", 128),
            ("histogram", 216 * 4),
        ] {
            for rot_inv in [false, true] {
                let map = map_with(&ids, &variants, dims);
                let m = similarity_matrix(&map, &ids, algo, rot_inv);
                for &threshold in &[0.0f64, 0.5, 0.9, 1.0] {
                    let pairs = similarity_pairs_above(&map, &ids, algo, rot_inv, threshold);
                    let expected: Vec<(usize, usize, f64)> = (0..ids.len())
                        .flat_map(|i| (i + 1..ids.len()).map(move |j| (i, j)))
                        .filter(|&(i, j)| m[i][j] >= threshold)
                        .map(|(i, j)| (i, j, m[i][j]))
                        .collect();
                    assert_eq!(
                        pairs, expected,
                        "algo={algo} rot_inv={rot_inv} th={threshold}"
                    );
                }
            }
        }
    }

    /// Unknown algorithm mirrors the matrix path: off-diagonal cells are
    /// 0.0 — emitted as pairs only when the threshold admits them.
    #[test]
    fn pairs_above_unknown_algo() {
        let ids = [1i64, 2];
        let map = map_with(&ids, &[0], 8);
        assert_eq!(
            similarity_pairs_above(&map, &ids, "nope", true, 0.0),
            vec![(0, 1, 0.0)]
        );
        assert!(similarity_pairs_above(&map, &ids, "nope", true, 0.5).is_empty());
    }

    /// Rotation-invariant extractors emit one variant-0 row; the rest emit
    /// all 8. Total rows = 8·(N−3) + 3.
    #[test]
    fn invariant_extractors_emit_variant0_only() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 3) as u8, (y * 5) as u8, (x ^ y) as u8])
        }));
        let rows = compute_all_variants(&img);
        let n_inv = EXTRACTORS.iter().filter(|e| e.rotation_invariant()).count();
        let n_all = EXTRACTORS.len();
        assert_eq!(
            rows.len(),
            (NUM_VARIANTS as usize) * (n_all - n_inv) + n_inv
        );
        for (v, name, _, _) in &rows {
            let ext = extractor_for_feature(name).unwrap();
            if ext.rotation_invariant() {
                assert_eq!(*v, 0, "invariant {name} emitted variant {v}");
            }
        }
        // every non-invariant feature still covers all 8 variants
        for ext in EXTRACTORS.iter().filter(|e| !e.rotation_invariant()) {
            let got: Vec<u8> = rows
                .iter()
                .filter(|(_, n, _, _)| n == ext.feature_name())
                .map(|(v, _, _, _)| *v)
                .collect();
            assert_eq!(
                got,
                (0..NUM_VARIANTS).collect::<Vec<u8>>(),
                "{}",
                ext.feature_name()
            );
        }
    }

    /// Every `MatrixKernel`'s decode-once scoring must produce the identical
    /// matrix the generic per-pair `ext.similarity` byte path produces —
    /// `similarity_matrix` (which dispatches to the kernel) is compared
    /// against `generic_similarity_matrix` driven by the extractor's own
    /// `similarity`, covering every kernel incl. blockhash/sliceprofile.
    #[test]
    fn kernels_match_generic_byte_path() {
        let ids = [20i64, 21, 22, 23, 24];
        let variants: Vec<u8> = (0..NUM_VARIANTS).collect();
        // (algo, payload bytes) — dims chosen so each kernel decodes fully.
        for (algo, dims) in [
            ("phash", 8usize),      // Bits
            ("edgehash", 8),        // Bits
            ("histogram", 216 * 4), // Cosine
            ("orbscale", 32 * 4),   // CenteredCosine
            ("hu", 7 * 4),          // ExpDist
            ("ssim", 256),          // U8Cosine (gray_flat bytes)
            ("blockhash", 128),     // Blockhash rows
            ("sliceprofile", 128),  // SliceProfile struct
            ("sliceprofile", 100),  // malformed → cosine fallback arm
        ] {
            let ext = extractor_for_algo(algo).unwrap();
            assert!(
                ext.matrix_kernel().is_some(),
                "{algo} should have a matrix kernel"
            );
            for rot_inv in [false, true] {
                let map = map_with(&ids, &variants, dims);
                let got = similarity_matrix(&map, &ids, algo, rot_inv);
                let effective = rot_inv && !ext.rotation_invariant();
                let want =
                    generic_similarity_matrix(&map, &ids, effective, &|a, b| ext.similarity(a, b));
                assert_eq!(got, want, "algo={algo} rot_inv={rot_inv}");
            }
        }
    }

    /// `compute_ctx` must return byte-identical payloads to `compute` for
    /// every extractor — this is what lets `variant_features` share the
    /// phash/whash 32×32 downscale through `VariantCtx`.
    #[test]
    fn compute_ctx_matches_compute() {
        let rgb_data: Vec<u8> = (0..(48 * 40 * 3))
            .map(|i| ((i * 37 + i / 3 * 11) % 256) as u8)
            .collect();
        let rgb = RgbImage::new(48, 40, rgb_data);
        let gray = image_io::to_gray(&image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(48, 40, rgb.data.clone()).unwrap(),
        ));
        let ctx = VariantCtx::new(&gray, &rgb);
        for ext in EXTRACTORS {
            assert_eq!(
                ext.compute(&gray, &rgb),
                ext.compute_ctx(&ctx),
                "{}",
                ext.feature_name()
            );
        }
    }

    /// End-to-end byte-identity for the variant pipeline: every stored row
    /// of `compute_all_variants` must equal the pre-optimization path —
    /// per-variant `to_gray`/`to_rgb` over `orientation_variants`
    /// DynamicImages — exercised on an Rgba8 source so `to_rgb` really
    /// converts (not just permutes).
    #[test]
    fn compute_all_variants_byte_identical_to_reference() {
        let mut px = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(90, 70);
        for (x, y, p) in px.enumerate_pixels_mut() {
            *p = image::Rgba([
                (x * 3 % 256) as u8,
                (y * 5 % 256) as u8,
                ((x ^ y) % 256) as u8,
                255,
            ]);
        }
        let img = image::DynamicImage::ImageRgba8(px);
        let rows = compute_all_variants(&img);
        // reference: the old DynamicImage-variant path
        let small = image_io::resize_max_side(&img, MAX_SIDE);
        let mut expected: Vec<VariantRow> = Vec::new();
        for (vi, dv) in image_io::orientation_variants(&small).iter().enumerate() {
            let g = image_io::to_gray(dv);
            let r = image_io::to_rgb(dv);
            for (name, bytes, dims) in variant_features(&g, &r, vi == 0) {
                expected.push((vi as u8, name, bytes, dims));
            }
        }
        assert_eq!(rows.len(), expected.len());
        for (got, want) in rows.iter().zip(&expected) {
            assert_eq!(
                (got.0, &got.1, got.3),
                (want.0, &want.1, want.3),
                "row meta"
            );
            assert_eq!(got.2, want.2, "v{} {}", got.0, got.1);
        }
    }
}
