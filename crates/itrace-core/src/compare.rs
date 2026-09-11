//! Pairwise scoring engine on decoded images — the "precise" path used by
//! /compare when image files are available. Matrix/vector mode lives in
//! `features.rs` and is used by smart-compare.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::descriptors::{self, DescriptorSet};
use crate::group::UnionFind;
use crate::{hashes, image_io, metrics, GrayImage, HashSet, RgbImage};
use crate::{DESCRIPTOR_ALGOS, HASH_ALGOS};

const MAX_SIDE: u32 = 512;

/// Hashes of one dihedral gray variant. Rotations/flips are exact pixel
/// permutations, so colorhash (a function of the pixel multiset) equals the
/// base image's — pass it in instead of recomputing HSV over an RGB copy.
fn variant_hash(vg: &GrayImage, colorhash: u64) -> HashSet {
    HashSet {
        phash: hashes::phash(vg),
        dhash: hashes::dhash(vg),
        ahash: hashes::ahash(vg),
        whash: hashes::whash(vg),
        colorhash,
    }
}

/// An image decoded once and prepared for pairwise scoring.
/// Descriptor sets are computed lazily by the caller-side cache because ORB
/// extraction is the expensive step; see `prepare_descriptors`.
pub struct Prepared {
    pub gray: GrayImage,
    pub rgb: RgbImage,
    pub hashes: HashSet,
    /// Hashes for the 8 dihedral variants (index 0 = original).
    pub variant_hashes: Vec<HashSet>,
    /// 8 dihedral variants of `gray` (≤512 on long side).
    pub variant_grays: Vec<GrayImage>,
    pub histogram: Vec<f32>,
    /// algo → descriptors on the base image.
    pub descs: HashMap<String, DescriptorSet>,
    /// algo → descriptors per variant (only when rot-inv descriptors needed).
    pub variant_descs: HashMap<String, Vec<DescriptorSet>>,
}

impl Prepared {
    /// Decode + prepare. `desc_algos` controls which descriptor extractors run;
    /// `rot_inv` additionally computes descriptor + hash variants.
    pub fn load(
        path: &Path,
        desc_algos: &[String],
        rot_inv: bool,
    ) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes, desc_algos, rot_inv)
    }

    pub fn from_bytes(
        bytes: &[u8],
        desc_algos: &[String],
        rot_inv: bool,
    ) -> anyhow::Result<Self> {
        let img = image_io::decode(bytes)?;
        let small = image_io::resize_max_side(&img, MAX_SIDE);
        let gray = image_io::to_gray(&small);
        let rgb = image_io::to_rgb(&small);
        let hashes = hashes::compute_all(&gray, &rgb);
        let histogram = metrics::hsv_histogram(&rgb);

        let mut variant_hashes = Vec::new();
        let mut variant_grays = Vec::new();
        if rot_inv {
            // Gray variants of `gray` (already ≤ MAX_SIDE — no resize needed).
            // They are pixel-identical to `to_gray(orientation_variants(&small)[i])`
            // because to_luma8 commutes with exact permutations, so hashing them
            // directly reproduces the old RGB-transform pipeline bit-for-bit.
            let vgs = image_io::gray_orientation_variants(&gray);
            variant_hashes = vgs
                .par_iter()
                .map(|vg| variant_hash(vg, hashes.colorhash))
                .collect();
            variant_grays = vgs;
        }

        let mut prepared = Prepared {
            gray,
            rgb,
            hashes,
            variant_hashes,
            variant_grays,
            histogram,
            descs: HashMap::new(),
            variant_descs: HashMap::new(),
        };
        prepared.prepare_descriptors(desc_algos, rot_inv);
        Ok(prepared)
    }

    /// Run descriptor extraction. All detects — base image plus every dihedral
    /// variant — are flattened into a single rayon job set so a lone algo still
    /// gets its base and variant extracts running concurrently.
    pub fn prepare_descriptors(&mut self, desc_algos: &[String], rot_inv: bool) {
        let exts: Vec<_> = desc_algos
            .iter()
            .filter_map(|a| descriptors::extractor_for(a).map(|e| (a.clone(), e)))
            .collect();
        let nv = if rot_inv { self.variant_grays.len() } else { 0 };
        // Job list: (algo_idx, None) = base image, (algo_idx, Some(v)) = variant v.
        let mut jobs = Vec::with_capacity(exts.len() * (nv + 1));
        for ai in 0..exts.len() {
            jobs.push((ai, None));
            jobs.extend((0..nv).map(|vi| (ai, Some(vi))));
        }
        let results: Vec<(usize, Option<usize>, DescriptorSet)> = {
            let (gray, vgs) = (&self.gray, &self.variant_grays);
            jobs.par_iter()
                .map(|&(ai, vi)| {
                    let g = match vi {
                        None => gray,
                        Some(vi) => &vgs[vi],
                    };
                    (ai, vi, exts[ai].1.detect(g, 512))
                })
                .collect()
        };
        let mut slots: Vec<Vec<Option<DescriptorSet>>> =
            exts.iter().map(|_| vec![None; nv]).collect();
        for (ai, vi, d) in results {
            match vi {
                None => {
                    self.descs.insert(exts[ai].0.clone(), d);
                }
                Some(vi) => slots[ai][vi] = Some(d),
            }
        }
        if nv > 0 {
            for (ai, algo_slots) in slots.into_iter().enumerate() {
                let sets = algo_slots
                    .into_iter()
                    .map(|s| s.expect("every variant job ran"))
                    .collect();
                self.variant_descs.insert(exts[ai].0.clone(), sets);
            }
        }
    }
}

/// Direct single-algorithm score on prepared images (no rotation variants).
fn base_score(algo: &str, a: &Prepared, b: &Prepared) -> f64 {
    if HASH_ALGOS.contains(&algo) {
        let ha = hashes::hash_of(&a.hashes, algo).unwrap_or(0);
        let hb = hashes::hash_of(&b.hashes, algo).unwrap_or(0);
        return hashes::hash_similarity(ha, hb);
    }
    match algo {
        "ssim" => ssim_common(&a.gray, &b.gray),
        "histogram" => metrics::histogram_correlation(&a.histogram, &b.histogram),
        "template" => template_common(&a.gray, &b.gray),
        _ if DESCRIPTOR_ALGOS.contains(&algo) => {
            let (Some(da), Some(db)) = (a.descs.get(algo), b.descs.get(algo)) else {
                return 0.0;
            };
            descriptors::match_score(da, db, 64)
        }
        "auto" => hybrid_score(a, b),
        _ => 0.0,
    }
}

/// Borrow `g` when it already is `w`×`h`; owned exact resize otherwise.
/// (`resize_gray_exact` clones the buffer even on a no-op.)
fn fit_exact(g: &GrayImage, w: u32, h: u32) -> Cow<'_, GrayImage> {
    if g.width == w && g.height == h {
        Cow::Borrowed(g)
    } else {
        Cow::Owned(image_io::resize_gray_exact(g, w, h))
    }
}

/// Borrow `g` when already ≤ `max_side`; owned downscale otherwise.
fn fit_max(g: &GrayImage, max_side: u32) -> Cow<'_, GrayImage> {
    if g.width.max(g.height) <= max_side {
        Cow::Borrowed(g)
    } else {
        Cow::Owned(image_io::resize_gray_max(g, max_side))
    }
}

fn ssim_common(a: &GrayImage, b: &GrayImage) -> f64 {
    let h = a.height.min(b.height);
    let w = a.width.min(b.width);
    metrics::ssim(&fit_exact(a, w, h), &fit_exact(b, w, h))
}

fn template_common(a: &GrayImage, b: &GrayImage) -> f64 {
    let ra = fit_max(a, 256);
    let rb = fit_max(b, 256);
    let h = ra.height.min(rb.height);
    let w = ra.width.min(rb.width);
    metrics::ncc(&fit_exact(&ra, w, h), &fit_exact(&rb, w, h))
}

/// 0.3·phash + 0.3·ssim + 0.4·orb fusion over borrowed parts — shared by
/// `hybrid_score` and the rot-inv per-variant loop so variants can be scored
/// without cloning images/histograms into a throwaway `Prepared`.
fn hybrid_parts(
    a_gray: &GrayImage,
    a_phash: u64,
    a_orb: Option<&DescriptorSet>,
    b_gray: &GrayImage,
    b_phash: u64,
    b_orb: Option<&DescriptorSet>,
) -> f64 {
    let weights: [(f64, &str); 3] = [(0.3, "phash"), (0.3, "ssim"), (0.4, "orb")];
    let mut tw = 0.0;
    let mut ts = 0.0;
    for (w, algo) in weights {
        let s = match algo {
            "phash" => hashes::hash_similarity(a_phash, b_phash),
            "ssim" => ssim_common(a_gray, b_gray),
            "orb" => {
                let (Some(da), Some(db)) = (a_orb, b_orb) else {
                    continue;
                };
                descriptors::match_score(da, db, 64)
            }
            _ => 0.0,
        };
        ts += s * w;
        tw += w;
    }
    if tw > 0.0 { ts / tw } else { 0.0 }
}

fn hybrid_score(a: &Prepared, b: &Prepared) -> f64 {
    hybrid_parts(
        &a.gray,
        a.hashes.phash,
        a.descs.get("orb"),
        &b.gray,
        b.hashes.phash,
        b.descs.get("orb"),
    )
}

/// Rotation/flip-invariant score: max over B's 8 dihedral variants.
fn rotated_score(algo: &str, a: &Prepared, b: &Prepared) -> f64 {
    if b.variant_hashes.is_empty() && b.variant_grays.is_empty() {
        return base_score(algo, a, b);
    }
    let mut best = base_score(algo, a, b);
    if HASH_ALGOS.contains(&algo) {
        let ha = hashes::hash_of(&a.hashes, algo).unwrap_or(0);
        for vb in &b.variant_hashes {
            let hb = hashes::hash_of(vb, algo).unwrap_or(0);
            best = best.max(hashes::hash_similarity(ha, hb));
            if best >= 0.95 {
                break;
            }
        }
        return best;
    }
    match algo {
        "ssim" | "template" => {
            for vg in b.variant_grays.iter().skip(1) {
                let s = if algo == "ssim" {
                    ssim_common(&a.gray, vg)
                } else {
                    template_common(&a.gray, vg)
                };
                best = best.max(s);
                if best >= 0.95 {
                    break;
                }
            }
            best
        }
        "histogram" => best, // histogram is already rotation-invariant
        _ if DESCRIPTOR_ALGOS.contains(&algo) => {
            if let (Some(da), Some(per_variant)) =
                (a.descs.get(algo), b.variant_descs.get(algo))
            {
                for db in per_variant.iter().skip(1) {
                    best = best.max(descriptors::match_score(da, db, 64));
                    if best >= 0.95 {
                        break;
                    }
                }
            }
            best
        }
        "auto" => {
            let a_orb = a.descs.get("orb");
            let b_orb = b.variant_descs.get("orb");
            for (i, vb) in b.variant_hashes.iter().enumerate() {
                let vg = b.variant_grays.get(i).unwrap_or(&b.gray);
                let vo = b_orb.and_then(|dv| dv.get(i));
                best = best.max(hybrid_parts(
                    &a.gray,
                    a.hashes.phash,
                    a_orb,
                    vg,
                    vb.phash,
                    vo,
                ));
                if best >= 0.95 {
                    break;
                }
            }
            best
        }
        _ => best,
    }
}

/// Score a pair under an algorithm; rotation_invariant takes max over variants.
pub fn pair_score(algo: &str, a: &Prepared, b: &Prepared, rot_inv: bool) -> f64 {
    let s = if rot_inv { rotated_score(algo, a, b) } else { base_score(algo, a, b) };
    (s * 10000.0).round() / 10000.0
}

/// Which descriptor algorithms the scorer needs prepared.
pub fn desc_algos_for(algorithm: &str) -> Vec<String> {
    match algorithm {
        "auto" => vec!["orb".to_string()],
        a if DESCRIPTOR_ALGOS.contains(&a) => vec![a.to_string()],
        _ => Vec::new(),
    }
}

/// Full N×N precise pairwise matrix (rayon-parallel).
pub fn pairwise_matrix(
    prepared: &[Prepared],
    algorithm: &str,
    rot_inv: bool,
) -> Vec<Vec<f64>> {
    let n = prepared.len();
    let mut m = vec![vec![0f64; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, pair_score(algorithm, &prepared[i], &prepared[j], rot_inv)))
        .collect();
    for (i, j, s) in results {
        m[i][j] = s;
        m[j][i] = s;
    }
    m
}

/// (groups, ungrouped, matrix) — single-algorithm analysis.
pub fn analyze(
    prepared: &[Prepared],
    algorithm: &str,
    threshold: f64,
    rot_inv: bool,
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<Vec<f64>>) {
    let m = pairwise_matrix(prepared, algorithm, rot_inv);
    let (groups, ungrouped) = crate::group::cluster(&m, threshold);
    (groups, ungrouped, m)
}

/// Union-Find over a confirmed-pair set (smart-compare gate).
pub fn components_from_pairs(n: usize, pairs: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(n);
    for &(a, b) in pairs {
        uf.union(a, b);
    }
    uf.groups()
}

pub use crate::group::cluster;

/// Convenience alias used by services.
pub fn histogram_of(rgb: &RgbImage) -> Vec<f32> {
    metrics::hsv_histogram(rgb)
}
