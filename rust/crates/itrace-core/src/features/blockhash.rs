//! Crop-tolerant block hash feature (`blockhash`).
//!
//! The image is normalized to a fixed `CANON`×`CANON` raster and split into a
//! `GRID`×`GRID` grid of tiles. Each tile contributes one bit: 1 when its mean
//! luminance exceeds the median of all tile means — the classic block-median
//! hash layout, here 32×32 tiles → a 1024-bit signature packed LSB-first into
//! 128 bytes.
//!
//! `similarity` is best-overlap Hamming: the two tile grids are aligned under
//! every dihedral orientation, integer tile shifts, and a small set of zoom
//! factors — a crop rescales the kept region, so a fraction `s` of one grid
//! overlays a rectangular window of the other. Only tiles that land inside the
//! implied overlap are compared and the best normalized score wins, so a
//! center-cropped variant still scores high while unrelated images stay low.

use crate::{image_io, GrayImage, RgbImage};

use super::{FeatureExtractor, FeatureKind};

/// Tile-grid side length; the signature is `GRID`² bits.
const GRID: usize = 32;
/// Canonical raster side; each tile is `CANON/GRID` px square.
const CANON: u32 = 128;
const BITS: usize = GRID * GRID; // 1024
/// Fewer overlapping tiles than this makes the normalized score too noisy.
const MIN_OVERLAP: usize = BITS / 4;
/// Zoom factors for candidate crop windows below unity: `s` = fraction of the
/// other image's tile grid this signature's tiles cover.
const ZOOMS: &[f64] = &[0.75, 0.5];
/// Integer tile-shift bound for the unscaled alignment.
const MAX_SHIFT: i32 = 8;

/// `blockhash`: 1024-bit median-thresholded tile-grid signature.
pub struct BlockhashExtractor;

impl FeatureExtractor for BlockhashExtractor {
    fn feature_name(&self) -> &'static str {
        "blockhash_bits"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["blockhash"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Bits
    }
    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        let small = image_io::resize_gray_exact(gray, CANON, CANON);
        let tp = (CANON as usize) / GRID; // tile edge in px
        let mut means = [0f64; BITS];
        for ty in 0..GRID {
            for tx in 0..GRID {
                let mut sum = 0u64;
                for y in ty * tp..(ty + 1) * tp {
                    for x in tx * tp..(tx + 1) * tp {
                        sum += small.get(x as u32, y as u32) as u64;
                    }
                }
                means[ty * GRID + tx] = sum as f64 / (tp * tp) as f64;
            }
        }
        let mut sorted = means;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = (sorted[BITS / 2 - 1] + sorted[BITS / 2]) / 2.0;
        let mut out = vec![0u8; BITS / 8];
        for (i, &m) in means.iter().enumerate() {
            if m > median {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }
    fn dims(&self, _data: &[u8]) -> usize {
        BITS
    }
    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        let ga = unpack_grid(a);
        let gb = unpack_grid(b);
        // Both directions: either image may be the cropped one.
        best_overlap(&ga, &gb).max(best_overlap(&gb, &ga))
    }
}

/// Unpack an LSB-first bit payload into a row-major GRID×GRID 0/1 tile map.
fn unpack_grid(data: &[u8]) -> [u8; BITS] {
    let mut g = [0u8; BITS];
    for (i, v) in g.iter_mut().enumerate() {
        *v = (data[i / 8] >> (i % 8)) & 1;
    }
    g
}

/// Source tile index for output tile (x, y) under dihedral orientation `k`,
/// matching `image_io::orientation_variants` order.
fn oriented_src(k: u8, x: usize, y: usize) -> usize {
    let n = GRID;
    let (sx, sy) = match k {
        0 => (x, y),                    // identity
        1 => (y, n - 1 - x),            // rot90
        2 => (n - 1 - x, n - 1 - y),    // rot180
        3 => (n - 1 - y, x),            // rot270
        4 => (n - 1 - x, y),            // fliph
        5 => (n - 1 - y, n - 1 - x),    // fliph+rot90
        6 => (x, n - 1 - y),            // fliph+rot180 (flipv)
        _ => (y, x),                    // fliph+rot270 (transpose)
    };
    sy * n + sx
}

/// Best-overlap Hamming of `b`'s tile grid mapped onto `a`'s. For each
/// dihedral orientation of `b`:
/// - unscaled: `b` slides over `a` at integer tile shifts and only the
///   overlapping tiles are compared;
/// - zoomed by `s`: `b`'s tiles cover only an `s`-fraction of `a`'s grid, so
///   `b` is majority-downsampled to a `⌊GRID·s⌋`² tile map and slid over the
///   windows of `a` — the tile region a same-fraction crop would occupy.
///
/// Returns the best hits/overlap ratio found.
fn best_overlap(a: &[u8; BITS], b: &[u8; BITS]) -> f64 {
    let mut best = 0.0f64;
    for k in 0..8u8 {
        let mut bg = [0u8; BITS];
        for y in 0..GRID {
            for x in 0..GRID {
                bg[y * GRID + x] = b[oriented_src(k, x, y)];
            }
        }
        for oy in -MAX_SHIFT..=MAX_SHIFT {
            for ox in -MAX_SHIFT..=MAX_SHIFT {
                let mut hits = 0usize;
                let mut total = 0usize;
                for by in 0..GRID {
                    let ay = by as i32 + oy;
                    if ay < 0 || ay >= GRID as i32 {
                        continue;
                    }
                    for bx in 0..GRID {
                        let ax = bx as i32 + ox;
                        if ax < 0 || ax >= GRID as i32 {
                            continue;
                        }
                        total += 1;
                        if a[ay as usize * GRID + ax as usize] == bg[by * GRID + bx] {
                            hits += 1;
                        }
                    }
                }
                if total >= MIN_OVERLAP {
                    best = best.max(hits as f64 / total as f64);
                }
            }
        }
        for &s in ZOOMS {
            let w = (GRID as f64 * s).round() as usize;
            // Majority-downsample `bg` to w×w: window tile t takes the strict
            // majority of the bg tiles mapping into it (½ on a tie).
            let mut wt = [0f64; BITS];
            for wy in 0..w {
                for wx in 0..w {
                    let (mut ones, mut n) = (0usize, 0usize);
                    for by in 0..GRID {
                        for bx in 0..GRID {
                            if ((bx as f64 + 0.5) * s).floor() as usize == wx
                                && ((by as f64 + 0.5) * s).floor() as usize == wy
                            {
                                ones += bg[by * GRID + bx] as usize;
                                n += 1;
                            }
                        }
                    }
                    wt[wy * GRID + wx] = ones as f64 / n as f64;
                }
            }
            for oy in 0..=(GRID - w) {
                for ox in 0..=(GRID - w) {
                    let mut score = 0f64;
                    for wy in 0..w {
                        for wx in 0..w {
                            let abit = a[(wy + oy) * GRID + wx + ox] as f64;
                            let bbit = wt[wy * GRID + wx];
                            score += if (bbit - 0.5).abs() < 1e-9 {
                                0.5
                            } else if (bbit > 0.5) == (abit > 0.5) {
                                1.0
                            } else {
                                0.0
                            };
                        }
                    }
                    best = best.max(score / (w * w) as f64);
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Luma};

    /// Deterministic textured image: bilinear value noise (8px cells) on a
    /// low-frequency gradient — no fixtures, no network.
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
                    + s(ix, iy + 1) * (1.0 - tx) * ty
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
        BlockhashExtractor.compute(&g, &r)
    }

    #[test]
    fn blockhash_layout_and_identity() {
        let ext = BlockhashExtractor;
        let img = textured(1, 128);
        let g = image_io::to_gray(&img);
        let r = image_io::to_rgb(&img);
        let data = ext.compute(&g, &r);
        assert_eq!(data.len(), BITS / 8);
        assert_eq!(ext.dims(&data), BITS);
        assert_eq!(ext.feature_name(), "blockhash_bits");
        assert_eq!(ext.algorithms()[0], "blockhash");
        assert_eq!(ext.kind(), FeatureKind::Bits);
        assert_eq!(ext.similarity(&data, &data), 1.0);
    }

    #[test]
    fn blockhash_orientation_invariance() {
        let ext = BlockhashExtractor;
        let img = textured(3, 128);
        let a = sig(&img);
        // Grid permutes exactly under dihedral transforms → ~1.0.
        assert!(ext.similarity(&a, &sig(&img.rotate90())) >= 0.95);
        assert!(ext.similarity(&a, &sig(&img.rotate180())) >= 0.95);
        assert!(ext.similarity(&a, &sig(&img.fliph())) >= 0.95);
    }

    #[test]
    fn blockhash_crop_tolerance() {
        let ext = BlockhashExtractor;
        let img = textured(5, 128);
        let a = sig(&img);
        // Center crop keeping 50% of each side.
        let c50 = img.crop_imm(32, 32, 64, 64);
        let s50 = ext.similarity(&a, &sig(&c50));
        assert!(s50 >= 0.75, "center-crop 50% sim {s50} < 0.75");
        // Off-center 50% crop: best-overlap shift should still find it.
        let corner = img.crop_imm(0, 0, 64, 64);
        let sc = ext.similarity(&a, &sig(&corner));
        assert!(sc >= 0.75, "corner-crop 50% sim {sc} < 0.75");
        // Center crop keeping 75% per side.
        let c75 = img.crop_imm(16, 16, 96, 96);
        assert!(ext.similarity(&a, &sig(&c75)) >= 0.8);
    }

    #[test]
    fn blockhash_discriminates_other_images() {
        let ext = BlockhashExtractor;
        let a = sig(&textured(11, 128));
        let b = sig(&textured(999, 128));
        let s = ext.similarity(&a, &b);
        assert!(s <= 0.7, "different-image sim {s} > 0.7");
        // Uniform images carry no structure: identical content, empty grid.
        let flat = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(64, 64, Luma([128u8])));
        let f = sig(&flat);
        assert!(f.iter().all(|&v| v == 0));
    }
}
