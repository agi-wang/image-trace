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
/// Tile edge in pixels (`CANON/GRID` = 4).
const TP: usize = CANON as usize / GRID;
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
        // one row-major sweep scattering pixels into per-tile u64 sums —
        // the same exact integer accumulation the per-tile loops did
        let mut sums = [0u64; BITS];
        for (y, row) in small
            .data
            .as_chunks::<{ CANON as usize }>()
            .0
            .iter()
            .enumerate()
        {
            let tile_row = (y / TP) * GRID;
            for (tx, px) in row.as_chunks::<TP>().0.iter().enumerate() {
                sums[tile_row + tx] += px.iter().map(|&v| v as u64).sum::<u64>();
            }
        }
        let mut means = [0f64; BITS];
        for (t, &s) in sums.iter().enumerate() {
            means[t] = s as f64 / (TP * TP) as f64;
        }
        // select_nth_unstable for the upper middle, then take the max of the
        // low partition — the same two order statistics the full sort averaged.
        let mut sorted = means;
        sorted.select_nth_unstable_by(BITS / 2, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lo = sorted[..BITS / 2]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let median = (lo + sorted[BITS / 2]) / 2.0;
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
        let ga = payload_rows(a);
        let gb = payload_rows(b);
        // Both directions: either image may be the cropped one.
        best_overlap(&ga, &gb).max(best_overlap(&gb, &ga))
    }
}

/// Pack an LSB-first bit payload into GRID row masks (bit x of row y =
/// tile y*GRID+x — the same mapping `unpack_grid` produced, kept in u64s so
/// overlap windows compare with XOR + popcount). Reads `BITS/8` bytes like
/// the old unpacking (extra bytes ignored, short payloads panic).
fn payload_rows(data: &[u8]) -> [u64; GRID] {
    let mut r = [0u64; GRID];
    for i in 0..BITS / 8 {
        // 8 consecutive tile bits per byte; all land in row i/4 at (i%4)*8
        r[i / 4] |= (data[i] as u64) << ((i % 4) * 8);
    }
    r
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
///
/// Tile rows are u64 bitmasks: an overlap region is a rectangle, so each of
/// its rows scores with one XOR + popcount (hits = total − mismatches), and
/// a shift is abandoned as soon as its remaining rows cannot raise `best` —
/// identical scores to the per-tile comparison this replaces.
fn best_overlap(a: &[u64; GRID], b: &[u64; GRID]) -> f64 {
    let mut best = 0.0f64;
    for k in 0..8u8 {
        // `b` under dihedral orientation k, packed: row y bit x = b[src(x,y)]
        let mut rb = [0u64; GRID];
        for (y, row) in rb.iter_mut().enumerate() {
            let mut bits = 0u64;
            for x in 0..GRID {
                let src = oriented_src(k, x, y);
                bits |= ((b[src / GRID] >> (src % GRID)) & 1) << x;
            }
            *row = bits;
        }
        for oy in -MAX_SHIFT..=MAX_SHIFT {
            // compared b-rows: by ∈ [by0, by1) with ay = by + oy
            let (by0, by1) = ((-oy).max(0) as usize, (GRID as i32 - oy).min(GRID as i32) as usize);
            let nrows = by1 - by0;
            for ox in -MAX_SHIFT..=MAX_SHIFT {
                let bx0 = (-ox).max(0) as usize;
                let ax0 = ox.max(0) as usize;
                let width = GRID - ox.unsigned_abs() as usize;
                let mask = (1u64 << width) - 1;
                let total = nrows * width;
                let mut mism = 0usize;
                for (i, &rbrow) in rb[by0..by1].iter().enumerate() {
                    let ay = (by0 + i) as i32 + oy;
                    mism += (((a[ay as usize] >> ax0) ^ (rbrow >> bx0)) & mask).count_ones() as usize;
                    // score bound: even all-matching remaining rows can't
                    // beat `best` — stop (the max update below is then ≤ best
                    // and a no-op, so no completion flag is needed)
                    if (total - mism) as f64 <= best * total as f64 {
                        break;
                    }
                }
                if total >= MIN_OVERLAP {
                    best = best.max((total - mism) as f64 / total as f64);
                }
            }
        }
        for &s in ZOOMS {
            let w = (GRID as f64 * s).round() as usize;
            // Majority-downsample `rb` to w×w: each source tile maps into
            // exactly one window cell — one pass accumulates ones/n per cell.
            let mut ones = [0u32; BITS];
            let mut cnt = [0u32; BITS];
            for (by, &rbrow) in rb.iter().enumerate() {
                let wy = ((by as f64 + 0.5) * s).floor() as usize;
                for bx in 0..GRID {
                    let wx = ((bx as f64 + 0.5) * s).floor() as usize;
                    cnt[wy * GRID + wx] += 1;
                    ones[wy * GRID + wx] += ((rbrow >> bx) & 1) as u32;
                }
            }
            // window tile = strict majority of its source tiles (½ on a tie);
            // rows packed as bit masks: set bits + half-credit tie cells
            let mut wbits = [0u64; GRID];
            let mut wtie = [0u64; GRID];
            for wy in 0..w {
                for wx in 0..w {
                    let wt = ones[wy * GRID + wx] as f64 / cnt[wy * GRID + wx] as f64;
                    if (wt - 0.5).abs() < 1e-9 {
                        wtie[wy] |= 1 << wx;
                    } else if wt > 0.5 {
                        wbits[wy] |= 1 << wx;
                    }
                }
            }
            let ntie: usize = wtie[..w].iter().map(|r| r.count_ones() as usize).sum();
            let (w2, half_tie) = ((w * w) as f64, ntie as f64 * 0.5);
            let mask = (1u64 << w) - 1;
            for oy in 0..=(GRID - w) {
                for ox in 0..=(GRID - w) {
                    let mut mism = 0usize;
                    for wy in 0..w {
                        mism += ((((a[wy + oy] >> ox) ^ wbits[wy]) & mask) & !wtie[wy])
                            .count_ones() as usize;
                        if w2 - mism as f64 - half_tie <= best * w2 {
                            break; // bound can't beat best — same no-op update
                        }
                    }
                    // score = (non-tie matches + ½·ties) / w²
                    //       = (w² − mism − ntie/2) / w² — the exact dyadic
                    //       sum the per-cell accumulation produced
                    best = best.max((w2 - mism as f64 - half_tie) / w2);
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
