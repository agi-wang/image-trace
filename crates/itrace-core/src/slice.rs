//! Slice / crop / sub-image robust matching.
//!
//! Handles the cases the original pipeline misses: an image that was cut into
//! tiles and reassembled (sliced), embedded as a sub-image, or saved as a crop.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::metrics::template_match;
use crate::{image_io, GrayImage};

const MAX_SIDE: u32 = 384;

/// Borrow `g` when already ≤ `max_side`; owned downscale otherwise.
/// (`resize_gray_max` clones the buffer even on a no-op.)
fn fit_max(g: &GrayImage, max_side: u32) -> Cow<'_, GrayImage> {
    if g.width.max(g.height) <= max_side {
        Cow::Borrowed(g)
    } else {
        Cow::Owned(image_io::resize_gray_max(g, max_side))
    }
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
    let a_small = fit_max(a, MAX_SIDE);
    let b_small = fit_max(b, MAX_SIDE);
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
    let a_s = fit_max(a, MAX_SIDE);
    let b_s = fit_max(b, MAX_SIDE);
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
