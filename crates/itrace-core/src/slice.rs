//! Slice / crop / sub-image robust matching.
//!
//! Handles the cases the original pipeline misses: an image that was cut into
//! tiles and reassembled (sliced), embedded as a sub-image, or saved as a crop.

use rayon::prelude::*;

use crate::metrics::template_match;
use crate::{image_io, GrayImage};

const MAX_SIDE: u32 = 384;

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
    let mut cells = Vec::new();
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

/// 4 rotations of a cell (0°, 90°, 180°, 270°).
fn rotations(cell: &GrayImage) -> Vec<GrayImage> {
    let vars = image_io::gray_orientation_variants(cell);
    vec![vars[0].clone(), vars[1].clone(), vars[2].clone(), vars[3].clone()]
}

/// Slide each cell of B over A (4 rotations per cell), take the best.
pub fn slice_match(a: &GrayImage, b: &GrayImage, rows: u32, cols: u32, threshold: f64) -> SliceMatchResult {
    let a_small = image_io::resize_gray_max(a, MAX_SIDE);
    let b_small = image_io::resize_gray_max(b, MAX_SIDE);
    let cells = grid_cells(&b_small, rows, cols);

    let results: Vec<SliceCell> = cells
        .par_iter()
        .enumerate()
        .map(|(i, cell)| {
            let row = (i as u32) / cols;
            let col = (i as u32) % cols;
            let mut best = (0.0f64, 0u32, 0u32, 0u32);
            for (ri, rot) in rotations(cell).iter().enumerate() {
                if rot.width > a_small.width || rot.height > a_small.height {
                    continue;
                }
                let (s, x, y) = template_match(&a_small, rot);
                if s > best.0 {
                    best = (s, x, y, (ri * 90) as u32);
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

/// Whole-image containment test: is `small` a crop/sub-image of `big`?
/// Returns (score, x, y). Downscales for speed; tries both directions.
pub fn contains(a: &GrayImage, b: &GrayImage) -> (f64, bool) {
    let a_s = image_io::resize_gray_max(a, MAX_SIDE);
    let b_s = image_io::resize_gray_max(b, MAX_SIDE);
    // b inside a?
    if b_s.width <= a_s.width && b_s.height <= a_s.height {
        let (s, _, _) = template_match(&a_s, &b_s);
        if s >= 0.8 {
            return (s, true);
        }
    }
    // a inside b?
    if a_s.width <= b_s.width && a_s.height <= b_s.height {
        let (s, _, _) = template_match(&b_s, &a_s);
        return (s, s >= 0.8);
    }
    (0.0, false)
}
