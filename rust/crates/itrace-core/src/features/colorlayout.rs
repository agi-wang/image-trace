//! MPEG-7-style color layout descriptor (`colorlayout`).
//!
//! Pipeline: downsample to an 8×8 block grid, convert to YCbCr, run an
//! 8×8 DCT-II on each plane, keep the low-frequency coefficients in
//! zigzag order.
//!
//! The stored vector keeps the DC coefficient signed (it carries the
//! dominant luma / chroma of the image) and stores AC terms as
//! magnitudes `|F(u,v)|`. Mirror transforms only flip coefficient signs
//! and a 90° rotation transposes the coefficient grid, so magnitudes
//! keep the descriptor stable under orientation changes — on top of the
//! crop/scale robustness from normalizing every input to an 8×8 grid.

use crate::{GrayImage, RgbImage};

use super::{cosine, pack_f32, unpack_f32, FeatureExtractor, FeatureKind};

/// Side of the downsampled grid / DCT block.
const N: usize = 8;
/// Low-frequency coefficients kept per plane in zigzag order.
const KEEP_Y: usize = 12;
const KEEP_C: usize = 6;
/// Stored vector length: Y + Cb + Cr.
const DIMS: usize = KEEP_Y + 2 * KEEP_C;

/// Zigzag visit order for an 8×8 block (linear index = v*8 + u).
fn zigzag_order() -> [usize; N * N] {
    let mut order = [0usize; N * N];
    let mut i = 0;
    let mut diag = 0;
    while diag < 2 * N - 1 {
        // Even anti-diagonals run bottom-left → top-right, odd the reverse.
        let up = diag % 2 == 0;
        let u_lo = diag.saturating_sub(N - 1);
        let u_hi = diag.min(N - 1);
        let mut k = u_lo;
        while k <= u_hi {
            let (u, v) = if up { (k, diag - k) } else { (diag - k, k) };
            order[i] = v * N + u;
            i += 1;
            k += 1;
        }
        diag += 1;
    }
    order
}

/// Separable orthonormal 2D DCT-II on an N×N f64 block.
fn dct_2d(input: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    let pi_n = std::f64::consts::PI / n as f64;
    let mut tmp = vec![0.0f64; n * n];
    for (y, row) in tmp.chunks_exact_mut(n).enumerate() {
        for (u, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for x in 0..n {
                s += input[y * n + x] * ((x as f64 + 0.5) * u as f64 * pi_n).cos();
            }
            *o = s * if u == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        }
    }
    for v in 0..n {
        for u in 0..n {
            let mut s = 0.0;
            for y in 0..n {
                s += tmp[y * n + u] * ((y as f64 + 0.5) * v as f64 * pi_n).cos();
            }
            out[v * n + u] = s * if v == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        }
    }
    out
}

/// Area-average the RGB raster down to an 8×8 grid.
fn downsample(rgb: &RgbImage) -> [[f64; N * N]; 3] {
    let mut planes = [[0.0f64; N * N]; 3];
    if rgb.width == 0 || rgb.height == 0 {
        return planes;
    }
    // Map each output cell to its source rectangle and box-average it.
    // Exact under any dihedral transform, unlike resample filters.
    for by in 0..N {
        let y0 = (by as u32 * rgb.height) / N as u32;
        let y1 = ((by as u32 + 1) * rgb.height) / N as u32;
        for bx in 0..N {
            let x0 = (bx as u32 * rgb.width) / N as u32;
            let x1 = ((bx as u32 + 1) * rgb.width) / N as u32;
            let mut acc = [0.0f64; 3];
            let mut cnt = 0u32;
            for y in y0..y1.max(y0 + 1).min(rgb.height) {
                for x in x0..x1.max(x0 + 1).min(rgb.width) {
                    let (r, g, b) = rgb.pixel(x, y);
                    acc[0] += f64::from(r);
                    acc[1] += f64::from(g);
                    acc[2] += f64::from(b);
                    cnt += 1;
                }
            }
            let c = cnt.max(1) as f64;
            for (p, plane) in planes.iter_mut().enumerate() {
                plane[by * N + bx] = acc[p] / c;
            }
        }
    }
    planes
}

/// RGB [0,255] → YCbCr (BT.601 full-range) planes.
fn to_ycbcr(planes: &[[f64; N * N]; 3]) -> [[f64; N * N]; 3] {
    let mut out = [[0.0f64; N * N]; 3];
    for i in 0..N * N {
        let (r, g, b) = (planes[0][i], planes[1][i], planes[2][i]);
        out[0][i] = 0.299 * r + 0.587 * g + 0.114 * b;
        out[1][i] = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        out[2][i] = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    }
    out
}

/// Low-frequency zigzag coefficients of one DCT'd plane.
///
/// `out[0]` is the signed DC scaled to mean amplitude (plane mean minus
/// `dc_center`); the rest are AC magnitudes — mirrors only flip DCT
/// signs and a 90° rotation transposes the coefficient grid, so
/// magnitudes keep the descriptor stable under orientation changes.
fn take_low(dct: &[f64], keep: usize, dc_center: f64, out: &mut Vec<f32>) {
    let order = zigzag_order();
    out.push((dct[0] / N as f64 - dc_center) as f32);
    for &idx in order.iter().take(keep).skip(1) {
        out.push(dct[idx].abs() as f32);
    }
}

/// MPEG-7-style color layout extractor; cosine-compared float vector.
pub struct ColorLayoutExtractor;

impl FeatureExtractor for ColorLayoutExtractor {
    fn feature_name(&self) -> &'static str {
        "colorlayout"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["colorlayout"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }
    fn compute(&self, _gray: &GrayImage, rgb: &RgbImage) -> Vec<u8> {
        let ycc = to_ycbcr(&downsample(rgb));
        let mut v = Vec::with_capacity(DIMS);
        take_low(&dct_2d(&ycc[0], N), KEEP_Y, 0.0, &mut v);
        take_low(&dct_2d(&ycc[1], N), KEEP_C, 128.0, &mut v);
        take_low(&dct_2d(&ycc[2], N), KEEP_C, 128.0, &mut v);
        pack_f32(&v)
    }
    fn dims(&self, data: &[u8]) -> usize {
        data.len() / 4
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        cosine(&unpack_f32(a), &unpack_f32(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;

    /// Deterministic textured color image: gradient + sinusoidal bands + a
    /// saturated blob off-center so the layout has real low-freq structure.
    fn gen_image(w: u32, h: u32, variant: u8) -> RgbImage {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64 / w as f64;
                let fy = y as f64 / h as f64;
                let band = (6.0 * std::f64::consts::TAU * fx + 4.0 * std::f64::consts::TAU * fy).sin();
                let blob = ((fx - 0.7).powi(2) + (fy - 0.3).powi(2)) < 0.02;
                let (r, g, b) = match variant {
                    0 => (
                        80.0 + 120.0 * fx + 40.0 * band,
                        60.0 + 100.0 * fy - 30.0 * band,
                        140.0 - 60.0 * fx + 20.0 * band,
                    ),
                    _ => {
                        // checkerboard + radial rings, magenta/green palette
                        let chk = ((x / 6) + (y / 6)) % 2 == 0;
                        let ring = (26.0 * std::f64::consts::TAU * ((fx - 0.4).powi(2) + (fy - 0.6).powi(2)).sqrt()).sin();
                        if chk {
                            (30.0 + 30.0 * ring, 150.0 + 60.0 * ring, 40.0)
                        } else {
                            (190.0, 50.0 + 40.0 * ring, 190.0 - 30.0 * ring)
                        }
                    }
                };
                let (r, g, b) = if blob && variant == 0 { (230.0, 40.0, 30.0) } else { (r, g, b) };
                data.push(r.clamp(0.0, 255.0) as u8);
                data.push(g.clamp(0.0, 255.0) as u8);
                data.push(b.clamp(0.0, 255.0) as u8);
            }
        }
        RgbImage::new(w, h, data)
    }

    fn to_dyn(rgb: &RgbImage) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(rgb.width, rgb.height, rgb.data.clone()).unwrap(),
        )
    }

    fn variant_of(rgb: &RgbImage, idx: usize) -> RgbImage {
        image_io::to_rgb(&image_io::orientation_variants(&to_dyn(rgb))[idx])
    }

    /// Similarity through the real payload path (compute → bytes → similarity).
    fn sim(a: &RgbImage, b: &RgbImage) -> f64 {
        let ext = ColorLayoutExtractor;
        let ga = GrayImage::new(a.width, a.height, vec![0u8; (a.width * a.height) as usize]);
        let gb = GrayImage::new(b.width, b.height, vec![0u8; (b.width * b.height) as usize]);
        ext.similarity(&ext.compute(&ga, a), &ext.compute(&gb, b))
    }

    #[test]
    fn encodes_fixed_size_f32_vector() {
        let ext = ColorLayoutExtractor;
        let img = gen_image(37, 23, 0); // odd size exercises the box partition
        let gray = GrayImage::new(img.width, img.height, vec![0u8; (img.width * img.height) as usize]);
        let bytes = ext.compute(&gray, &img);
        assert_eq!(bytes.len(), DIMS * 4);
        assert_eq!(ext.dims(&bytes), DIMS);
        assert_eq!(ext.feature_name(), "colorlayout");
        assert!(ext.algorithms().contains(&"colorlayout"));
        assert_eq!(ext.kind(), FeatureKind::Cosine);
        // deterministic: same input → identical payload
        assert_eq!(bytes, ext.compute(&gray, &img));
        // registered in the extractor registry
        let ext2 = crate::features::extractor_for_algo("colorlayout").unwrap();
        assert_eq!(ext2.feature_name(), "colorlayout");
    }

    /// Mirror variants (flip / rot180) keep |coefficients| identical → ~1.0.
    /// Transposing variants (rot90/rot270) permute the low-freq grid → ~0.87.
    /// Measured: 1.0000 for variants 2/4/6, 0.8704 for 1/3/5/7.
    #[test]
    fn invariant_to_orientation() {
        let a = gen_image(96, 80, 0);
        for (v, min) in [(1usize, 0.80f64), (2, 0.95), (3, 0.80), (4, 0.95), (5, 0.80), (6, 0.95), (7, 0.80)] {
            let s = sim(&a, &variant_of(&a, v));
            assert!(s >= min, "variant {v}: sim {s:.4} < {min}");
        }
    }

    /// The 8×8 grid normalization makes the descriptor insensitive to
    /// resolution and moderate cropping. Measured: scale 1.0000, crop 0.9877.
    #[test]
    fn robust_to_crop_and_scale() {
        let a = gen_image(96, 80, 0);
        let big = image_io::to_rgb(&to_dyn(&a).resize_exact(200, 180, image::imageops::FilterType::Triangle));
        let small = image_io::to_rgb(&to_dyn(&a).resize_exact(32, 32, image::imageops::FilterType::Triangle));
        let crop = image_io::to_rgb(&to_dyn(&a).crop_imm(10, 8, 76, 64));
        assert!(sim(&a, &big) >= 0.95, "scale-up sim {:.4}", sim(&a, &big));
        assert!(sim(&a, &small) >= 0.95, "scale-down sim {:.4}", sim(&a, &small));
        assert!(sim(&a, &crop) >= 0.90, "crop sim {:.4}", sim(&a, &crop));
    }

    /// A structurally and chromatically different generated image scores low.
    /// Measured: 0.3770.
    #[test]
    fn distinguishes_other_images() {
        let a = gen_image(96, 80, 0);
        let other = gen_image(96, 80, 1);
        let s = sim(&a, &other);
        assert!(s <= 0.60, "different-image sim {s:.4} >= 0.60");
        // and still strictly below every orientation variant of the original
        for v in 1..8 {
            assert!(sim(&a, &variant_of(&a, v)) > s, "variant {v} not above diff");
        }
    }

    #[test]
    fn identical_image_scores_one() {
        let a = gen_image(96, 80, 0);
        assert_eq!(sim(&a, &a), 1.0);
    }
}
