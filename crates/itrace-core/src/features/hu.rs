//! Hu moment invariants: 7 log-transformed shape moments of the binarized
//! grayscale image (`"hu"`), cosine-compared.
//!
//! The gray raster is downscaled (aspect preserved), thresholded by Otsu and
//! the minority side is taken as ink, so the descriptor is polarity- and
//! padding-invariant. Normalized central moments η_pq = μ_pq/m00^(1+(p+q)/2)
//! give rotation/scale invariance by construction. The 7th Hu invariant only
//! discriminates mirror images (its sign flips under reflection), so its sign
//! is dropped before the log transform to keep flips invariant.

use super::{pack_f32, unpack_f32, FeatureExtractor, FeatureKind};
use crate::{image_io, GrayImage, RgbImage};

/// Largest image side used for moment accumulation.
const HU_MAX_SIDE: u32 = 128;

/// `feature_name="hu_moments"`, `algorithms=["hu"]`, kind=Cosine.
pub struct HuExtractor;

impl FeatureExtractor for HuExtractor {
    fn feature_name(&self) -> &'static str {
        "hu_moments"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["hu"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        let h = hu_log_moments(gray);
        let v: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        pack_f32(&v)
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() / 4
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        // Plain cosine on log-Hu vectors saturates (~0.99 for unrelated
        // shapes: all components land in the same positive range). Compare in
        // log-moment space instead: exp(-||Δt||₂) — 1.0 for identical shapes,
        // decaying smoothly as moment profiles diverge.
        let (ta, tb) = (unpack_f32(a), unpack_f32(b));
        let n = ta.len().min(tb.len());
        if n == 0 {
            return 0.0;
        }
        let d = (0..n)
            .map(|i| (ta[i] as f64 - tb[i] as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        (-d).exp()
    }
}

/// Otsu between-class-variance threshold over a 256-bin histogram.
fn otsu_threshold(data: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in data {
        hist[v as usize] += 1;
    }
    let total = data.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let mut w0 = 0u64;
    let mut sum0 = 0.0f64;
    let mut best_t = 0u8;
    let mut best_var = 0.0f64;
    for (t, &c) in hist.iter().enumerate() {
        w0 += c;
        sum0 += t as f64 * c as f64;
        let w1 = data.len() as u64 - w0;
        if w0 == 0 || w1 == 0 {
            continue;
        }
        let m0 = sum0 / w0 as f64;
        let m1 = (sum_all - sum0) / w1 as f64;
        let var = (w0 as f64 / total) * (w1 as f64 / total) * (m0 - m1).powi(2);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// sign-preserving log10 used for Hu moments; 0 for degenerate values.
fn signed_log(h: f64) -> f64 {
    if h == 0.0 || !h.is_finite() {
        0.0
    } else {
        -h.signum() * h.abs().log10()
    }
}

/// Seven log-transformed Hu invariants of the Otsu-binarized raster.
fn hu_log_moments(gray: &GrayImage) -> [f64; 7] {
    let g = image_io::resize_gray_max(gray, HU_MAX_SIDE);
    let (w, h) = (g.width, g.height);
    if w == 0 || h == 0 {
        return [0.0; 7];
    }
    let t = otsu_threshold(&g.data);
    // ink = minority side of the split (ties → bright side)
    let above = g.data.iter().filter(|&&v| v > t).count() as f64;
    let total = g.data.len() as f64;
    let ink_above = above <= total - above;
    let m00 = if ink_above { above } else { total - above };
    if m00 <= 0.0 {
        return [0.0; 7];
    }
    let is_ink = |v: u8| (v > t) == ink_above;

    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            if is_ink(g.get(x, y)) {
                cx += x as f64;
                cy += y as f64;
            }
        }
    }
    cx /= m00;
    cy /= m00;

    let (mut mu20, mut mu02, mut mu11) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mu30, mut mu03, mut mu21, mut mu12) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        let dy = y as f64 - cy;
        let dy2 = dy * dy;
        let dy3 = dy2 * dy;
        for x in 0..w {
            if !is_ink(g.get(x, y)) {
                continue;
            }
            let dx = x as f64 - cx;
            let dx2 = dx * dx;
            mu20 += dx2;
            mu02 += dy2;
            mu11 += dx * dy;
            mu30 += dx2 * dx;
            mu03 += dy3;
            mu21 += dx2 * dy;
            mu12 += dx * dy2;
        }
    }

    // η_pq = μ_pq / m00^(1 + (p+q)/2) — scale + intensity invariant
    let n2 = m00 * m00; // exponent 1+2/2 = 2
    let n3 = n2 * m00.sqrt(); // exponent 1+3/2 = 2.5
    let e20 = mu20 / n2;
    let e02 = mu02 / n2;
    let e11 = mu11 / n2;
    let e30 = mu30 / n3;
    let e03 = mu03 / n3;
    let e21 = mu21 / n3;
    let e12 = mu12 / n3;

    let a = e30 + e12;
    let b = e21 + e03;
    let c = e30 - 3.0 * e12;
    let d = 3.0 * e21 - e03;
    let h1 = e20 + e02;
    let h2 = (e20 - e02).powi(2) + 4.0 * e11 * e11;
    let h3 = c * c + d * d;
    let h4 = a * a + b * b;
    let h5 = c * a * (a * a - 3.0 * b * b) + d * b * (3.0 * a * a - b * b);
    let h6 = (e20 - e02) * (a * a - b * b) + 4.0 * e11 * a * b;
    let h7 = d * a * (a * a - 3.0 * b * b) - c * b * (3.0 * a * a - b * b);

    // h7's sign is the only mirror discriminator: |h7| keeps flips invariant.
    [
        signed_log(h1),
        signed_log(h2),
        signed_log(h3),
        signed_log(h4),
        signed_log(h5),
        signed_log(h6),
        if h7 == 0.0 || !h7.is_finite() {
            0.0
        } else {
            -h7.abs().log10()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;

    /// Deterministic asymmetric shape: dark ellipse + stem + notch on light bg.
    fn shape_a() -> GrayImage {
        let (w, h) = (96u32, 96u32);
        let mut d = vec![220u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64 - 42.0) / 22.0;
                let dy = (y as f64 - 36.0) / 16.0;
                if dx * dx + dy * dy <= 1.0 {
                    d[(y * w + x) as usize] = 30;
                }
            }
        }
        for y in 48..78 {
            for x in 58..66 {
                d[(y * w + x) as usize] = 30;
            }
        }
        for y in 28..34 {
            for x in 30..38 {
                d[(y * w + x) as usize] = 220;
            }
        }
        GrayImage::new(w, h, d)
    }

    /// Very different shape: deterministic pseudo-random scatter of dark blobs.
    fn shape_b() -> GrayImage {
        let (w, h) = (96u32, 96u32);
        let mut d = vec![220u8; (w * h) as usize];
        let mut s = 0x9E3779B9u32;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..26 {
            let cx = (rnd() % 86 + 5) as i32;
            let cy = (rnd() % 86 + 5) as i32;
            let r = (rnd() % 4 + 2) as i32;
            for y in (cy - r)..(cy + r) {
                for x in (cx - r)..(cx + r) {
                    if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
                        d[(y as u32 * w + x as u32) as usize] = 30;
                    }
                }
            }
        }
        GrayImage::new(w, h, d)
    }

    #[test]
    fn hu_dims_and_name() {
        let ext = HuExtractor;
        let bytes = ext.compute(&shape_a(), &shape_a_rgb());
        assert_eq!(bytes.len(), 28);
        assert_eq!(ext.dims(&bytes), 7);
        assert_eq!(ext.feature_name(), "hu_moments");
        assert_eq!(ext.algorithms(), &["hu"]);
    }

    fn shape_a_rgb() -> RgbImage {
        let g = shape_a();
        let mut rgb = Vec::with_capacity(g.data.len() * 3);
        for &v in &g.data {
            rgb.extend_from_slice(&[v, v, v]);
        }
        RgbImage::new(g.width, g.height, rgb)
    }

    #[test]
    fn hu_rotation_and_flip_invariance() {
        let ext = HuExtractor;
        let a = shape_a();
        let rgb = shape_a_rgb();
        let base = ext.compute(&a, &rgb);
        let variants = image_io::gray_orientation_variants(&a);
        for (i, v) in variants.iter().enumerate().skip(1) {
            let s = ext.similarity(&base, &ext.compute(v, &rgb));
            assert!(s >= 0.99, "variant {i} sim {s}");
        }
    }

    #[test]
    fn hu_scale_invariance() {
        let ext = HuExtractor;
        let a = shape_a();
        let rgb = shape_a_rgb();
        let base = ext.compute(&a, &rgb);
        let small = image_io::resize_gray_exact(&a, 48, 48);
        let s = ext.similarity(&base, &ext.compute(&small, &rgb));
        assert!(s >= 0.85, "half-scale sim {s}");
    }

    #[test]
    fn hu_distinct_shapes_differ() {
        let ext = HuExtractor;
        let a = shape_a();
        let b = shape_b();
        let rgb = shape_a_rgb();
        let s = ext.similarity(&ext.compute(&a, &rgb), &ext.compute(&b, &rgb));
        assert!(s <= 0.7, "distinct-shape sim {s}");
    }

    #[test]
    fn hu_blank_image_is_zero() {
        let ext = HuExtractor;
        let g = GrayImage::new(32, 32, vec![200u8; 32 * 32]);
        let rgb = RgbImage::new(32, 32, vec![200u8; 32 * 32 * 3]);
        let bytes = ext.compute(&g, &rgb);
        assert_eq!(bytes, vec![0u8; 28]);
        // identical payloads are a perfect match; a blank vs real image is not
        let s = ext.similarity(&bytes, &ext.compute(&shape_a(), &shape_a_rgb()));
        assert!(s <= 0.1, "blank-vs-shape sim {s}");
    }
}
