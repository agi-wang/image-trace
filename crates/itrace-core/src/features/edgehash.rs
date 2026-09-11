//! Edge-orientation hash (`edgehash`): Sobel gradient-orientation histograms.
//!
//! A Sobel pass over the normalized 128×128 luma raster feeds a 16×16 grid
//! of per-cell 4-bin undirected orientation histograms (weighted by gradient
//! magnitude, softly split between the two nearest bins). Cells are pooled
//! into 4×4 supercells, each normalized to relative bin strengths, and the
//! resulting 4×4×4 = 64 values are median-binarized into a 64-bit signature,
//! hamming-compared via the shared `Bits` helpers.
//!
//! Orientation proportions and median binarization are invariant to
//! brightness/contrast/color transforms (including inversion, since edges are
//! undirected) and stable under mild crops; rot90/flip matching is covered by
//! the 8-variant orientation mechanism.

use crate::{hashes, image_io, GrayImage, RgbImage};

use super::{pack_bits, unpack_bits, FeatureExtractor, FeatureKind};

/// Normalized raster side: 16 cells × 8 px.
const WORK: u32 = 128;
/// Cells per side.
const GRID: usize = 16;
/// Undirected orientation bins per cell (0°, 45°, 90°, 135°).
const ORIENT: usize = 4;
/// Supercells per side; SUPER² supercells × ORIENT bins = 64 encoded values.
const SUPER: usize = 4;
/// Minimum Sobel magnitude counted — flat/noise pixels carry no orientation.
const MAG_FLOOR: f64 = 16.0;

/// `edgehash` feature extractor — see module docs.
pub struct EdgeHashExtractor;

impl FeatureExtractor for EdgeHashExtractor {
    fn feature_name(&self) -> &'static str {
        "edgehash_bits"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["edgehash"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Bits
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        pack_bits(edgehash(gray))
    }
    fn dims(&self, _data: &[u8]) -> usize {
        64
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        hashes::hash_similarity(unpack_bits(a), unpack_bits(b))
    }
}

/// 64-bit edge-orientation signature of a grayscale image.
pub fn edgehash(gray: &GrayImage) -> u64 {
    let small = image_io::resize_gray_exact(gray, WORK, WORK);
    let w = WORK as usize;
    let cell_px = (WORK as usize) / GRID;

    // Sobel gradients → per-cell orientation histograms.
    let mut cells = [0f64; GRID * GRID * ORIENT];
    for y in 1..(WORK - 1) {
        for x in 1..(WORK - 1) {
            let i = (y as usize) * w + (x as usize);
            let gx = small.data[i - w + 1] as i32 + 2 * small.data[i + 1] as i32
                + small.data[i + w + 1] as i32
                - small.data[i - w - 1] as i32
                - 2 * small.data[i - 1] as i32
                - small.data[i + w - 1] as i32;
            let gy = small.data[i + w - 1] as i32 + 2 * small.data[i + w] as i32
                + small.data[i + w + 1] as i32
                - small.data[i - w - 1] as i32
                - 2 * small.data[i - w] as i32
                - small.data[i - w + 1] as i32;
            // compare squared magnitude to squared floor — sqrt is monotonic,
            // so the filter decision is identical and the sqrt runs only on
            // pixels that pass
            let mag2 = (gx * gx + gy * gy) as f64;
            if mag2 < MAG_FLOOR * MAG_FLOOR {
                continue;
            }
            let mag = mag2.sqrt();
            // undirected orientation in [0, π) at 45° bin spacing, cyclic;
            // magnitude splits linearly between the two nearest bins
            let pos = (gy as f64)
                .atan2(gx as f64)
                .rem_euclid(std::f64::consts::PI)
                / std::f64::consts::FRAC_PI_4;
            let b0 = pos as usize;
            let f = pos - pos.floor();
            let c = ((y as usize / cell_px) * GRID + (x as usize) / cell_px) * ORIENT;
            cells[c + b0] += (1.0 - f) * mag;
            cells[c + (b0 + 1) % ORIENT] += f * mag;
        }
    }

    // pool 16×16 cells → 4×4 supercells, normalize each to relative strengths
    let mut v = [0f64; SUPER * SUPER * ORIENT];
    for cy in 0..GRID {
        for cx in 0..GRID {
            let s = (cy / (GRID / SUPER)) * SUPER + cx / (GRID / SUPER);
            for k in 0..ORIENT {
                v[s * ORIENT + k] += cells[(cy * GRID + cx) * ORIENT + k];
            }
        }
    }
    for s in 0..(SUPER * SUPER) {
        let sum: f64 = v[s * ORIENT..s * ORIENT + ORIENT].iter().sum();
        if sum > 0.0 {
            for k in 0..ORIENT {
                v[s * ORIENT + k] /= sum;
            }
        }
    }

    // median-binarize the 64 relative bin strengths
    let mut sorted = v;
    sorted.sort_by(f64::total_cmp);
    let med = (sorted[31] + sorted[32]) / 2.0;
    let mut bits = 0u64;
    for (i, &val) in v.iter().enumerate() {
        if val > med {
            bits |= 1 << i;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::{edgehash, unpack_bits, EdgeHashExtractor};
    use crate::features::{extractor_for_algo, FeatureExtractor, FeatureKind};
    use crate::{hashes, image_io};

    /// Deterministic textured scene: faint lattice + random 2px segments and
    /// filled rects — dense edges at all orientations, no fixtures needed.
    fn make_scene(w: u32, h: u32, seed: u32) -> image::DynamicImage {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (((x * 7 + y * 5 + seed * 13) % 17) * 8 + 48) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let mut s = seed.max(1);
        let mut next = |m: u32| -> u32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 8) % m.max(1)
        };
        for _ in 0..48 {
            let (x0, y0) = (next(w) as i32, next(h) as i32);
            let (x1, y1) = (next(w) as i32, next(h) as i32);
            let c = if next(2) == 0 { 20u8 } else { 235u8 };
            let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
            for t in 0..=steps {
                let (x, y) = (x0 + (x1 - x0) * t / steps, y0 + (y1 - y0) * t / steps);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let px = (x + dx).clamp(0, w as i32 - 1) as u32;
                    let py = (y + dy).clamp(0, h as i32 - 1) as u32;
                    img.put_pixel(px, py, image::Rgb([c, c, c]));
                }
            }
        }
        for _ in 0..10 {
            let (rx, ry) = (next(w - 24), next(h - 24));
            let (rw, rh) = (8 + next(16), 8 + next(16));
            let c = [next(256) as u8, next(256) as u8, next(256) as u8];
            for y in ry..(ry + rh).min(h) {
                for x in rx..(rx + rw).min(w) {
                    img.put_pixel(x, y, image::Rgb(c));
                }
            }
        }
        image::DynamicImage::ImageRgb8(img)
    }

    fn signature(img: &image::DynamicImage) -> u64 {
        let g = image_io::to_gray(img);
        let r = image_io::to_rgb(img);
        unpack_bits(&EdgeHashExtractor.compute(&g, &r))
    }

    /// Same comparison path as `generic_similarity_matrix` with rot_inv: the
    /// max over all pairs of the 8 dihedral orientation variants.
    fn covered_sim(a: &image::DynamicImage, b: &image::DynamicImage) -> f64 {
        let mut best = 0f64;
        for va in image_io::orientation_variants(a) {
            for vb in image_io::orientation_variants(b) {
                best = best.max(hashes::hash_similarity(signature(&va), signature(&vb)));
            }
        }
        best
    }

    #[test]
    fn edgehash_deterministic() {
        let a = make_scene(168, 120, 7);
        assert_eq!(signature(&a), signature(&a));
    }

    #[test]
    fn edgehash_photometric_invariant() {
        let a = make_scene(168, 120, 7);
        let transforms = [
            image::DynamicImage::ImageRgb8(image::imageops::brighten(&a.to_rgb8(), 60)),
            image::DynamicImage::ImageRgb8(image::imageops::contrast(&a.to_rgb8(), 0.55)),
            {
                let mut inv = a.to_rgb8();
                image::imageops::invert(&mut inv);
                image::DynamicImage::ImageRgb8(inv)
            },
        ];
        let s = signature(&a);
        for t in &transforms {
            let sim = hashes::hash_similarity(s, signature(t));
            assert!(sim >= 0.85, "photometric transform sim {sim}");
        }
    }

    #[test]
    fn edgehash_mild_crop_robust() {
        let a = make_scene(168, 120, 7);
        let crop = a.crop_imm(6, 5, 156, 110);
        let sim = hashes::hash_similarity(signature(&a), signature(&crop));
        assert!(sim >= 0.7, "mild crop sim {sim}");
    }

    #[test]
    fn edgehash_rot90_flip_via_variant_coverage() {
        let a = make_scene(168, 120, 7);
        assert!(covered_sim(&a, &a.rotate90()) >= 0.95, "rot90 covered");
        assert!(covered_sim(&a, &a.fliph()) >= 0.95, "fliph covered");
        // same-orientation baseline: the hash really encodes orientation
        // layout, it is not a near-constant signature
        let plain = hashes::hash_similarity(signature(&a), signature(&a.rotate90()));
        assert!(plain < 0.9, "same-variant rot90 sim {plain}");
    }

    #[test]
    fn edgehash_different_image_low() {
        let a = make_scene(168, 120, 7);
        let b = make_scene(168, 120, 41);
        let sim = hashes::hash_similarity(signature(&a), signature(&b));
        assert!(sim <= 0.75, "different image sim {sim}");
    }

    #[test]
    fn edgehash_flat_image_zero() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            64,
            image::Rgb([128; 3]),
        ));
        let g = image_io::to_gray(&img);
        assert_eq!(edgehash(&g), 0);
    }

    #[test]
    fn edgehash_registered_64bit() {
        let ext = extractor_for_algo("edgehash").expect("edgehash registered");
        assert_eq!(ext.feature_name(), "edgehash_bits");
        let a = make_scene(96, 96, 3);
        let data = ext.compute(&image_io::to_gray(&a), &image_io::to_rgb(&a));
        assert_eq!(data.len(), 8);
        assert_eq!(ext.dims(&data), 64);
        assert_eq!(ext.kind(), FeatureKind::Bits);
    }
}
