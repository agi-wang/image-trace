//! Slice / crop / sub-image robust matching.
//!
//! Handles the cases the original pipeline misses: an image that was cut into
//! tiles and reassembled (sliced), embedded as a sub-image, or saved as a crop.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::metrics::template_match;
use crate::{image_io, GrayImage};

const MAX_SIDE: u32 = 384;

/// Downscale `a` and `b` by one shared factor so the larger fits
/// `max_side`. Independent `fit_max` calls would break containment
/// geometry: a >max_side crop and its >max_side parent would both scale
/// to the same size, making the "smaller" image impossible to locate
/// inside the other. A shared factor preserves relative extents.
fn fit_pair<'a>(
    a: &'a GrayImage,
    b: &'a GrayImage,
    max_side: u32,
) -> (Cow<'a, GrayImage>, Cow<'a, GrayImage>) {
    let m = a.width.max(a.height).max(b.width).max(b.height);
    if m <= max_side {
        return (Cow::Borrowed(a), Cow::Borrowed(b));
    }
    let s = max_side as f64 / m as f64;
    let rs = |g: &GrayImage| {
        let w = ((g.width as f64) * s).round().max(1.0) as u32;
        let h = ((g.height as f64) * s).round().max(1.0) as u32;
        image_io::resize_gray_exact(g, w, h)
    };
    (Cow::Owned(rs(a)), Cow::Owned(rs(b)))
}

/// Result for one slice cell.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SliceCell {
    pub row: u32,
    pub col: u32,
    /// Best NCC score over all rotations of this cell.
    pub score: f64,
    /// Top-left of the best matching region inside image A (in A's ≤MAX_SIDE coords).
    pub best_x: Option<u32>,
    pub best_y: Option<u32>,
    /// Rotation of the cell that produced the best score (0/90/180/270).
    pub rotation: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SliceMatchResult {
    pub rows: u32,
    pub cols: u32,
    pub cells: Vec<SliceCell>,
    /// Fraction of cells with score >= threshold.
    pub coverage: f64,
    pub mean_score: f64,
    /// True when coverage is high → B is a sliced/shuffled copy of (part of) A.
    pub is_slice_of_a: bool,
}

/// Cut `img` into rows×cols cells (edge cells absorb the remainder).
fn grid_cells(img: &GrayImage, rows: u32, cols: u32) -> Vec<GrayImage> {
    let (w, h) = (img.width, img.height);
    let mut cells = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let x0 = c * w / cols;
            let x1 = (c + 1) * w / cols;
            let y0 = r * h / rows;
            let y1 = (r + 1) * h / rows;
            let (cw, ch) = (x1 - x0, y1 - y0);
            if cw == 0 || ch == 0 {
                continue;
            }
            let mut data = Vec::with_capacity((cw * ch) as usize);
            for y in y0..y1 {
                data.extend_from_slice(&img.data[(y * w + x0) as usize..(y * w + x1) as usize]);
            }
            cells.push(GrayImage::new(cw, ch, data));
        }
    }
    cells
}

/// One rotation of a cell: `ri` quarter-turns clockwise (0°, 90°, 180°, 270°).
/// Pixel-identical to `image_io::gray_orientation_variants` indices 0..=3 but
/// builds only the requested rotation — the 4 flip variants are never used.
fn rotate_cell(cell: &GrayImage, ri: u32) -> GrayImage {
    let (w, h) = (cell.width, cell.height);
    let at = |x: u32, y: u32| cell.data[(y * w + x) as usize];
    let build = |nw: u32, nh: u32, f: &dyn Fn(u32, u32) -> u8| {
        let mut data = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            for x in 0..nw {
                data.push(f(x, y));
            }
        }
        GrayImage::new(nw, nh, data)
    };
    match ri {
        0 => cell.clone(),
        1 => build(h, w, &|x, y| at(y, h - 1 - x)), // rot90 cw
        2 => build(w, h, &|x, y| at(w - 1 - x, h - 1 - y)), // rot180
        _ => build(h, w, &|x, y| at(w - 1 - y, x)), // rot270 cw
    }
}

/// Slide each cell of B over A (4 rotations per cell), take the best.
/// A cell's rotation loop stops early on a perfect 1.0 match: NCC cannot
/// exceed 1, so remaining rotations cannot change the recorded best.
pub fn slice_match(a: &GrayImage, b: &GrayImage, rows: u32, cols: u32, threshold: f64) -> SliceMatchResult {
    // Shared downscale: B's cells must stay proportional to their true
    // extent inside A (see `fit_pair`).
    let (a_small, b_small) = fit_pair(a, b, MAX_SIDE);
    let cells = grid_cells(&b_small, rows, cols);

    let results: Vec<SliceCell> = cells
        .par_iter()
        .enumerate()
        .map(|(i, cell)| {
            let row = (i as u32) / cols;
            let col = (i as u32) % cols;
            let mut best = (0.0f64, 0u32, 0u32, 0u32);
            for ri in 0..4u32 {
                // Rotated dims — (h,w) for 90°/270° — are known before the
                // rotation is built, so oversized rotations never materialize.
                let (rw, rh) = if ri % 2 == 0 {
                    (cell.width, cell.height)
                } else {
                    (cell.height, cell.width)
                };
                if rw > a_small.width || rh > a_small.height {
                    continue;
                }
                // ri == 0 borrows the cell; other rotations are built on demand.
                let owned;
                let rot = if ri == 0 {
                    cell
                } else {
                    owned = rotate_cell(cell, ri);
                    &owned
                };
                let (s, x, y) = template_match(&a_small, rot);
                if s > best.0 {
                    best = (s, x, y, ri * 90);
                }
                if best.0 >= 1.0 {
                    break;
                }
            }
            SliceCell {
                row,
                col,
                score: best.0,
                best_x: if best.0 >= threshold { Some(best.1) } else { None },
                best_y: if best.0 >= threshold { Some(best.2) } else { None },
                rotation: best.3,
            }
        })
        .collect();

    let n = results.len().max(1) as f64;
    let covered = results.iter().filter(|c| c.score >= threshold).count() as f64;
    let coverage = covered / n;
    let mean_score = results.iter().map(|c| c.score).sum::<f64>() / n;
    SliceMatchResult {
        rows,
        cols,
        cells: results,
        coverage,
        mean_score,
        is_slice_of_a: coverage >= 0.75,
    }
}

/// Whole-image containment test: is either image a crop/sub-image of the other?
/// Returns (score, contained). Downscales for speed; tries both directions.
/// The two direction checks are independent template matches — they run in
/// parallel. Return semantics are unchanged: a ≥0.8 b-in-a score short-circuits,
/// otherwise the a-in-b result (or none) decides.
pub fn contains(a: &GrayImage, b: &GrayImage) -> (f64, bool) {
    let (a_s, b_s) = fit_pair(a, b, MAX_SIDE);
    // `fwd` = score of b inside a, `rev` = score of a inside b; each is `None`
    // when the inner image cannot fit inside the outer one.
    let (fwd, rev) = rayon::join(
        || {
            if b_s.width <= a_s.width && b_s.height <= a_s.height {
                Some(template_match(&a_s, &b_s).0)
            } else {
                None
            }
        },
        || {
            if a_s.width <= b_s.width && a_s.height <= b_s.height {
                Some(template_match(&b_s, &a_s).0)
            } else {
                None
            }
        },
    );
    // b inside a?
    if let Some(s) = fwd {
        if s >= 0.8 {
            return (s, true);
        }
    }
    // a inside b?
    match rev {
        Some(s) => (s, s >= 0.8),
        None => (0.0, false),
    }
}

/// `contains` under all four relative quarter-turns: the best score over
/// `b` rotated 0/90/180/270 (rotating one side covers every relative
/// rotation, in both containment directions). Catches a crop/slice whose
/// content was also rotated.
pub fn contains_rot4(a: &GrayImage, b: &GrayImage) -> (f64, bool) {
    let mut best = contains(a, b);
    if best.1 {
        return best;
    }
    let mut rb = b.clone();
    for _ in 0..3 {
        rb = rotate_cell(&rb, 1);
        let s = contains(a, &rb);
        if s.0 > best.0 {
            best = s;
        }
        if best.1 {
            return best;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Luma};

    /// Deterministic textured image larger than MAX_SIDE so the shared
    /// downscale path in `contains`/`slice_match` is exercised.
    fn big_textured(seed: u32, size: u32) -> GrayImage {
        let mut st = seed | 1;
        let mut rng = move || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let cell = 16usize;
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
                    + s(ix, iy + 1) * tx * ty;
                buf.put_pixel(x, y, Luma([v as u8]));
            }
        }
        image_io::to_gray(&DynamicImage::ImageLuma8(buf))
    }

    #[test]
    fn contains_crop_of_large_image() {
        // Both parent and 70% crop exceed MAX_SIDE: a shared downscale
        // must keep the crop findable inside the parent.
        let img = big_textured(3, 640);
        let (w, h) = (img.width, img.height);
        let (cw, ch) = (w * 7 / 10, h * 7 / 10);
        let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
        let mut data = Vec::with_capacity((cw * ch) as usize);
        for y in y0..y0 + ch {
            data.extend_from_slice(
                &img.data[(y * w + x0) as usize..(y * w + x0 + cw) as usize],
            );
        }
        let crop = GrayImage::new(cw, ch, data);
        let (s, contained) = contains(&img, &crop);
        assert!(contained && s >= 0.9, "crop containment score {s}");
        let (s, contained) = contains_rot4(&img, &crop);
        assert!(contained && s >= 0.9, "crop containment(rot) score {s}");
        // unrelated large image must NOT be contained
        let other = big_textured(77, 640);
        let (s2, c2) = contains(&img, &other);
        assert!(!c2, "foreign containment score {s2}");
    }
}
