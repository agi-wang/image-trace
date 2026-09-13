//! Crop/slice recall feature (`crophash`).
//!
//! A single 64-bit global phash cannot see a crop: cutting away part of the
//! frame reshuffles every DCT coefficient, so crop/slice near-duplicates sit
//! at ~0.6 similarity — under any reasonable threshold and outside the MIH
//! recall radius. Instead we store phash keys for a fixed set of fractional
//! sub-windows, so a cropped copy keeps at least one window whose content is
//! (up to rescale) one of the parent's indexed windows:
//!
//! - `CENTER` scales — a chain (1.0 → 0.25) so a crop's own windows also land
//!   near a parent window one step down (crop70.full ≈ parent.c80, etc.);
//! - `HALF` offsets — 50% windows on a 3×3 offset grid covering the 2×2
//!   slice quadrants and off-center half-size crops.
//!
//! Payload: `N_KEYS` u64 keys packed little-endian in `WINDOWS` order.
//! `similarity` = best cross-slot key-pair hash similarity — a crop's
//! full-frame key lands on the parent's matching window slot, so the score
//! itself is crop-aware (it is a vote signal, not a precise verifier; dedup
//! verifies candidates with NCC containment).
//!
//! Per orientation variant: `N_KEYS` keys ≈ `8·N_KEYS` indexed keys per
//! image — the crop/slice recall channel of `index::crop_candidates`.

use crate::{hashes, GrayImage, RgbImage};

use super::{FeatureExtractor, FeatureKind};

/// Fractional window (x0, y0, w, h) of the variant raster.
struct Win(f64, f64, f64, f64);

/// Indexed sub-windows, in payload order.
const WINDOWS: &[Win] = &[
    // center-scale chain: crop70 ≈ c80/c65, crop-of-crop chains step down
    Win(0.0, 0.0, 1.0, 1.0),
    Win(0.10, 0.10, 0.80, 0.80),
    Win(0.175, 0.175, 0.65, 0.65),
    Win(0.25, 0.25, 0.50, 0.50),
    Win(0.3125, 0.3125, 0.375, 0.375),
    Win(0.375, 0.375, 0.25, 0.25),
    // 50% windows on a {0, 0.25, 0.5}² offset grid — the 2×2 slice
    // quadrants sit at the corners; the rest cover off-center halves.
    Win(0.0, 0.0, 0.5, 0.5),
    Win(0.25, 0.0, 0.5, 0.5),
    Win(0.5, 0.0, 0.5, 0.5),
    Win(0.0, 0.25, 0.5, 0.5),
    Win(0.25, 0.25, 0.5, 0.5),
    Win(0.5, 0.25, 0.5, 0.5),
    Win(0.0, 0.5, 0.5, 0.5),
    Win(0.25, 0.5, 0.5, 0.5),
    Win(0.5, 0.5, 0.5, 0.5),
];

/// Number of u64 keys in one `crophash_keys` payload.
pub const N_KEYS: usize = WINDOWS.len(); // 15
const PAYLOAD_BYTES: usize = N_KEYS * 8;

/// Extract the fractional window of `gray` as a new raster (nearest-pixel
/// bounds, clamped non-empty).
fn window_crop(gray: &GrayImage, w: &Win) -> GrayImage {
    let (iw, ih) = (gray.width as f64, gray.height as f64);
    let x0 = (w.0 * iw).round() as u32;
    let y0 = (w.1 * ih).round() as u32;
    let nw = ((w.2 * iw).round() as u32).max(1).min(gray.width - x0);
    let nh = ((w.3 * ih).round() as u32).max(1).min(gray.height - y0);
    let mut data = Vec::with_capacity((nw * nh) as usize);
    for row in gray.data[y0 as usize * gray.width as usize..]
        .chunks_exact(gray.width as usize)
        .take(nh as usize)
    {
        data.extend_from_slice(&row[x0 as usize..(x0 + nw) as usize]);
    }
    GrayImage::new(nw, nh, data)
}

/// phash of each indexed window, packed as `N_KEYS` little-endian u64s.
fn compute_keys(gray: &GrayImage) -> [u64; N_KEYS] {
    let mut keys = [0u64; N_KEYS];
    for (k, w) in WINDOWS.iter().enumerate() {
        keys[k] = hashes::phash(&window_crop(gray, w));
    }
    keys
}

/// Decode a stored payload to its per-window keys (`None` when malformed).
/// Used by the dedup crop-recall channel to turn stored blobs into
/// indexable u64 key sets.
pub fn payload_keys(data: &[u8]) -> Option<Vec<u64>> {
    if data.len() != PAYLOAD_BYTES {
        return None;
    }
    Some(
        data.as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect(),
    )
}

/// Best cross-slot key-pair similarity of two payloads — the `similarity`
/// core shared by the extractor and the `MultiBits` matrix kernel.
pub(crate) fn keys_similarity(a: &[u64], b: &[u64]) -> f64 {
    let mut best = 0.0f64;
    for &ka in a {
        for &kb in b {
            best = best.max(hashes::hash_similarity(ka, kb));
            if best >= 1.0 {
                return 1.0;
            }
        }
    }
    best
}

/// Multi-window crop-recall extractor.
pub struct CrophashExtractor;

impl FeatureExtractor for CrophashExtractor {
    fn feature_name(&self) -> &'static str {
        "crophash_keys"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["crophash"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Bits
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        let keys = compute_keys(gray);
        let mut out = Vec::with_capacity(PAYLOAD_BYTES);
        for k in keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() * 8
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        match (payload_keys(a), payload_keys(b)) {
            (Some(ka), Some(kb)) => keys_similarity(&ka, &kb),
            _ => 0.0,
        }
    }
    fn matrix_kernel(&self) -> Option<super::MatrixKernel> {
        Some(super::MatrixKernel::MultiBits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;
    use image::{DynamicImage, ImageBuffer, Luma};

    /// Deterministic textured image (same generator style as blockhash).
    fn textured(seed: u32, size: u32) -> DynamicImage {
        let mut st = seed | 1;
        let mut rng = move || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let cell = 8usize;
        let gw = size as usize / cell + 2;
        let noise: Vec<u8> = (0..gw * gw).map(|_| (rng() & 0xff) as u8).collect();
        let mut buf = ImageBuffer::<Luma<u8>, Vec<u8>>::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 / cell as f32;
                let fy = y as f32 / cell as f32;
                let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - ix as f32, fy - iy as f32);
                let s = |xx: usize, yy: usize| noise[yy * gw + xx] as f32;
                let v = s(ix, iy) * (1.0 - tx) * (1.0 - ty)
                    + s(ix + 1, iy) * tx * (1.0 - ty)
                    + s(ix + 1, iy + 1) * tx * ty;
                let grad = (x as f32 / size as f32) * 40.0 + (y as f32 / size as f32) * 25.0;
                buf.put_pixel(x, y, Luma([(v * 0.85 + grad).clamp(0.0, 255.0) as u8]));
            }
        }
        DynamicImage::ImageLuma8(buf)
    }

    fn sig(img: &DynamicImage) -> Vec<u8> {
        let g = image_io::to_gray(img);
        let r = image_io::to_rgb(img);
        CrophashExtractor.compute(&g, &r)
    }

    #[test]
    fn payload_layout_and_identity() {
        let ext = CrophashExtractor;
        let img = textured(1, 128);
        let g = image_io::to_gray(&img);
        let r = image_io::to_rgb(&img);
        let data = ext.compute(&g, &r);
        assert_eq!(data.len(), PAYLOAD_BYTES);
        assert_eq!(ext.feature_name(), "crophash_keys");
        assert_eq!(ext.algorithms(), &["crophash"]);
        assert_eq!(ext.similarity(&data, &data), 1.0);
        assert_eq!(payload_keys(&data).unwrap().len(), N_KEYS);
        assert!(payload_keys(&data[..8]).is_none());
    }

    #[test]
    fn crop_and_slice_score_high() {
        let ext = CrophashExtractor;
        let img = textured(5, 128);
        let a = sig(&img);
        // center crop keeping 70% per side — the dataset's crop70 shape
        let c70 = img.crop_imm(19, 19, 90, 90);
        let s = ext.similarity(&a, &sig(&c70));
        assert!(s >= 0.85, "crop70 sim {s}");
        // 2x2 grid slice (a quadrant tile)
        let tile = img.crop_imm(0, 0, 64, 64);
        let s = ext.similarity(&a, &sig(&tile));
        assert!(s >= 0.85, "quadrant-slice sim {s}");
    }

    #[test]
    fn discriminates_other_images() {
        let ext = CrophashExtractor;
        let a = sig(&textured(11, 128));
        let b = sig(&textured(999, 128));
        let s = ext.similarity(&a, &b);
        assert!(s <= 0.75, "different-image sim {s}");
    }
}
