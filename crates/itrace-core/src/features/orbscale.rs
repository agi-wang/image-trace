//! `orbscale`: multi-scale ORB feature.
//!
//! A 3-level Gaussian pyramid is built over the gray image and ORB runs
//! over it via [`orb::detect_orb_with_pyramid`] — the provided levels are
//! used directly as ORB's detection pyramid, so no second per-level
//! pyramid is built. The union of descriptors across all levels is
//! mean-pooled into the same 32-dim f32 signature
//! [`super::OrbPooledExtractor`] emits — but the pooled vector now covers
//! content at 1.0×, 0.5× and 0.25× scale, so a downscaled copy still
//! produces a near-identical signature.

use super::{pack_f32, pool_descriptors, unpack_f32, FeatureExtractor, FeatureKind};
use crate::descriptors::{orb, DescriptorSet};
use crate::{image_io, GrayImage, RgbImage};
use image::{ImageBuffer, Luma};

/// Pyramid depth: 1.0×, 0.5×, 0.25× — a half-scale copy's level-0
/// content reappears on the original's level 1.
const NUM_LEVELS: u32 = 3;
/// ORB feature budget per pyramid level.
const LEVEL_MAX_FEATURES: usize = 256;
/// Gaussian sigma applied before each 2× downsample.
const PYRAMID_SIGMA: f32 = 1.0;

fn gaussian_blur(gray: &GrayImage, sigma: f32) -> GrayImage {
    // Borrowed view over `gray.data` — blur only reads its source.
    let buf: ImageBuffer<Luma<u8>, &[u8]> =
        ImageBuffer::from_raw(gray.width, gray.height, gray.data.as_slice())
            .expect("gray buffer size matches");
    GrayImage::new(
        gray.width,
        gray.height,
        image::imageops::blur(&buf, sigma).into_raw(),
    )
}

/// Gaussian-pyramid levels *beyond* `gray` (which is level 0): level *i*+1
/// is a blurred 2× downsample of level *i* (classic Burt–Adelson reduce).
fn gaussian_pyramid(gray: &GrayImage) -> Vec<GrayImage> {
    let mut levels = Vec::with_capacity((NUM_LEVELS - 1) as usize);
    for _ in 1..NUM_LEVELS {
        let prev: &GrayImage = levels.last().unwrap_or(gray);
        let blurred = gaussian_blur(prev, PYRAMID_SIGMA);
        levels.push(image_io::resize_gray_exact(
            &blurred,
            (prev.width / 2).max(1),
            (prev.height / 2).max(1),
        ));
    }
    levels
}

/// ORB descriptors pooled across the pyramid: the Gaussian levels are fed
/// to [`orb::detect_orb_with_pyramid`] in one call — the provided level
/// images serve as ORB's detection pyramid, so no second 8-level pyramid
/// is built per level. Keypoints are tagged with the pyramid level and
/// reported in the base-level frame (scale 2^level).
fn detect_multiscale(gray: &GrayImage) -> DescriptorSet {
    let extra = gaussian_pyramid(gray);
    let pyr: Vec<(f64, &GrayImage)> = extra
        .iter()
        .enumerate()
        .map(|(i, l)| ((1u32 << (i + 1)) as f64, l))
        .collect();
    orb::detect_orb_with_pyramid(gray, LEVEL_MAX_FEATURES, &pyr)
}

/// Multi-scale pooled ORB signature serving the `orbscale` algorithm.
pub struct OrbScaleExtractor;

impl FeatureExtractor for OrbScaleExtractor {
    fn feature_name(&self) -> &'static str {
        "orbscale"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["orbscale"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        pack_f32(&pool_descriptors(&detect_multiscale(gray)))
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() / 4
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        // pooled payloads are 32 dims — stack-decode instead of Vec allocs
        let mut fa = [0f32; POOLED_DIMS];
        let mut fb = [0f32; POOLED_DIMS];
        if a.len() == POOLED_DIMS * 4 && b.len() == POOLED_DIMS * 4 {
            for (v, c) in fa.iter_mut().zip(a.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*c);
            }
            for (v, c) in fb.iter_mut().zip(b.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*c);
            }
            centered_cosine(&fa, &fb)
        } else {
            centered_cosine(&unpack_f32(a), &unpack_f32(b))
        }
    }
    fn matrix_kernel(&self) -> Option<super::MatrixKernel> {
        Some(super::MatrixKernel::CenteredCosine)
    }
}

/// Pooled-vector dimensionality (ORB descriptors are 32 bytes).
const POOLED_DIMS: usize = 32;

/// Cosine over mean-centered pooled vectors. Raw pooled byte-means are
/// all-positive and share ORB's bit-distribution bias, so plain cosine
/// reports ~0.95+ even for unrelated images; removing each vector's own
/// mean makes the score track distribution *shape* instead. `pub(crate)`:
/// also the decode-once matrix kernel's scorer.
pub(crate) fn centered_cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ma = a[..n].iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let mb = b[..n].iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let x = a[i] as f64 - ma;
        let y = b[i] as f64 - mb;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na * nb).sqrt()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb};

    /// Textured test image: gradient + deterministic blob splats driven by
    /// an LCG — same recipe as tests/pipeline.rs, no fixture files.
    fn make_photo(w: u32, h: u32, seed: u8) -> DynamicImage {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x ^ y) as u8).wrapping_add(seed);
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        let mut k = seed as u32 + 1;
        for _ in 0..48 {
            k = k.wrapping_mul(1103515245).wrapping_add(12345);
            let cx = (k >> 8) % w.max(1);
            k = k.wrapping_mul(1103515245).wrapping_add(12345);
            let cy = (k >> 8) % h.max(1);
            let rad = 3 + (k >> 4) % 12;
            for dy in -(rad as i32)..=rad as i32 {
                for dx in -(rad as i32)..=rad as i32 {
                    if dx * dx + dy * dy <= (rad * rad) as i32 {
                        let px = (cx as i32 + dx).clamp(0, w as i32 - 1) as u32;
                        let py = (cy as i32 + dy).clamp(0, h as i32 - 1) as u32;
                        let cur = img.get_pixel(px, py)[0];
                        img.put_pixel(px, py, Rgb([255 - cur, 200, (k >> 16) as u8]));
                    }
                }
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    /// A structurally different texture: concentric rings.
    fn make_rings(w: u32, h: u32) -> DynamicImage {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as i32 - w as i32 / 2;
                let dy = y as i32 - h as i32 / 2;
                let v = ((((dx * dx + dy * dy) as f32).sqrt() * 0.6).sin() * 127.0 + 128.0) as u8;
                img.put_pixel(x, y, Rgb([v, v.wrapping_add(40), 255 - v]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    /// A second structurally different texture: fine checkerboard.
    fn make_checker(w: u32, h: u32, cell: u32) -> DynamicImage {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x / cell + y / cell).is_multiple_of(2) { 240 } else { 15 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    fn compute_for(img: &DynamicImage) -> Vec<u8> {
        let g = image_io::to_gray(img);
        let r = image_io::to_rgb(img);
        OrbScaleExtractor.compute(&g, &r)
    }

    fn sim(a: &[u8], b: &[u8]) -> f64 {
        OrbScaleExtractor.similarity(a, b)
    }

    #[test]
    fn detect_multiscale_uses_three_levels_in_base_frame() {
        let img = make_photo(192, 192, 11);
        let g = image_io::to_gray(&img);
        let ds = detect_multiscale(&g);
        assert_eq!(ds.desc_len, 32);
        assert!(ds.len() > 60, "only {} keypoints", ds.len());
        for l in 0..NUM_LEVELS {
            assert!(
                ds.keypoints.iter().any(|k| k.level == l as u8),
                "no keypoints from level {l}"
            );
        }
        // coordinates rescaled to the base-level frame, not left in level coords
        assert!(
            ds.keypoints.iter().all(|k| k.x < g.width as f32 && k.y < g.height as f32),
            "keypoint escaped base frame"
        );
        // some level-2 point must sit past its own 48x48 level image
        assert!(
            ds.keypoints.iter().any(|k| k.level == 2 && (k.x >= 48.0 || k.y >= 48.0)),
            "level-2 keypoints not rescaled to base frame"
        );
    }

    #[test]
    fn rotation_and_flip_stay_similar() {
        let img = make_photo(192, 192, 21);
        let fa = compute_for(&img);
        let s_rot = sim(&fa, &compute_for(&img.rotate90()));
        assert!(s_rot >= 0.9, "rot90 similarity {s_rot}");
        let s_flip = sim(&fa, &compute_for(&img.fliph()));
        assert!(s_flip >= 0.8, "fliph similarity {s_flip}");
    }

    #[test]
    fn different_images_score_low() {
        let a = make_photo(192, 192, 5);
        let fa = compute_for(&a);
        let s_rings = sim(&fa, &compute_for(&make_rings(192, 192)));
        assert!(s_rings < 0.5, "rings similarity {s_rings}");
        let s_checker = sim(&fa, &compute_for(&make_checker(192, 192, 6)));
        assert!(s_checker < 0.5, "checker similarity {s_checker}");
    }

    /// The property this extractor exists for: pooling descriptors across
    /// the Gaussian pyramid keeps a half-scale copy recognizable.
    #[test]
    fn half_scale_still_matches() {
        let img = make_photo(256, 256, 9);
        let fa = compute_for(&img);
        let half = image_io::resize_max_side(&img, 128);
        let s = sim(&fa, &compute_for(&half));
        assert!(s >= 0.9, "50%-scale similarity {s}");

        // the same signature still separates an unrelated texture
        let s_other = sim(&fa, &compute_for(&make_rings(256, 256)));
        assert!(
            s > s_other + 0.3,
            "half-scale {s} not enough above rings {s_other}"
        );
    }

    /// Encoding contract shared with OrbPooledExtractor: f32-LE pooled
    /// descriptor means, 32 dims (128-byte payload).
    #[test]
    fn encoding_matches_orb_pooled_layout() {
        let img = make_photo(128, 128, 3);
        let f = compute_for(&img);
        assert_eq!(f.len(), 128);
        assert_eq!(OrbScaleExtractor.dims(&f), 32);
        // same payload decodes identically under the pooled-ORB extractor
        let pooled = crate::features::EXTRACTORS
            .iter()
            .find(|e| e.feature_name() == "orb_pooled")
            .unwrap();
        let s = pooled.similarity(&f, &f);
        assert!((s - 1.0).abs() < 1e-9, "self-similarity under orb_pooled {s}");
    }
}
