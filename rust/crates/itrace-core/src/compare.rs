//! Pairwise scoring engine on decoded images — the "precise" path used by
//! /compare when image files are available. Matrix/vector mode lives in
//! `features.rs` and is used by smart-compare.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use crate::descriptors::{self, DescriptorSet};
use crate::group::UnionFind;
use crate::{hashes, image_io, metrics, GrayImage, HashSet, RgbImage};
use crate::{DESCRIPTOR_ALGOS, HASH_ALGOS};

const MAX_SIDE: u32 = 512;

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
            let vgs = image_io::gray_orientation_variants(&gray);
            for vg in &vgs {
                let vg_small = image_io::resize_gray_max(vg, MAX_SIDE);
                variant_grays.push(vg_small);
            }
            let variants = image_io::orientation_variants(&small);
            for v in &variants {
                let g = image_io::to_gray(v);
                let r = image_io::to_rgb(v);
                variant_hashes.push(hashes::compute_all(&g, &r));
            }
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

    pub fn prepare_descriptors(&mut self, desc_algos: &[String], rot_inv: bool) {
        for algo in desc_algos {
            let Some(ext) = descriptors::extractor_for(algo) else {
                continue;
            };
            self.descs
                .insert(algo.clone(), ext.detect(&self.gray, 512));
            if rot_inv && !self.variant_grays.is_empty() {
                let per_variant: Vec<DescriptorSet> = self
                    .variant_grays
                    .par_iter()
                    .map(|g| ext.detect(g, 512))
                    .collect();
                self.variant_descs.insert(algo.clone(), per_variant);
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

fn ssim_common(a: &GrayImage, b: &GrayImage) -> f64 {
    let h = a.height.min(b.height);
    let w = a.width.min(b.width);
    let ra = image_io::resize_gray_exact(a, w, h);
    let rb = image_io::resize_gray_exact(b, w, h);
    metrics::ssim(&ra, &rb)
}

fn template_common(a: &GrayImage, b: &GrayImage) -> f64 {
    let ra = image_io::resize_gray_max(a, 256);
    let rb = image_io::resize_gray_max(b, 256);
    let h = ra.height.min(rb.height);
    let w = ra.width.min(rb.width);
    let ra = image_io::resize_gray_exact(&ra, w, h);
    let rb = image_io::resize_gray_exact(&rb, w, h);
    metrics::ncc(&ra, &rb)
}

fn hybrid_score(a: &Prepared, b: &Prepared) -> f64 {
    let weights: [(f64, &str); 3] = [(0.3, "phash"), (0.3, "ssim"), (0.4, "orb")];
    let mut tw = 0.0;
    let mut ts = 0.0;
    for (w, algo) in weights {
        let s = match algo {
            "phash" => hashes::hash_similarity(a.hashes.phash, b.hashes.phash),
            "ssim" => ssim_common(&a.gray, &b.gray),
            "orb" => {
                let (Some(da), Some(db)) = (a.descs.get("orb"), b.descs.get("orb")) else {
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
            for (i, vb) in b.variant_hashes.iter().enumerate() {
                let mut vb_p = Prepared {
                    gray: b.variant_grays.get(i).cloned().unwrap_or_else(|| b.gray.clone()),
                    rgb: b.rgb.clone(),
                    hashes: vb.clone(),
                    variant_hashes: Vec::new(),
                    variant_grays: Vec::new(),
                    histogram: b.histogram.clone(),
                    descs: HashMap::new(),
                    variant_descs: HashMap::new(),
                };
                if let Some(dv) = b.variant_descs.get("orb") {
                    if let Some(d) = dv.get(i) {
                        vb_p.descs.insert("orb".to_string(), d.clone());
                    }
                }
                best = best.max(hybrid_score(a, &vb_p));
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
    let pairs: Vec<(usize, usize)> =
        (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| (i, j, pair_score(algorithm, &prepared[i], &prepared[j], rot_inv)))
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
