//! Pure-Rust ORB: FAST-9 corner detection on an image pyramid,
//! intensity-centroid orientation, and rotated BRIEF-256 descriptors.

use rayon::prelude::*;
use super::{DescriptorExtractor, DescriptorSet, Keypoint};
use crate::GrayImage;

const PYRAMID_LEVELS: usize = 8;
const SCALE_FACTOR: f64 = 1.2;
const FAST_THRESHOLD: i32 = 20;
const FAST_THRESHOLD_LOW: i32 = 9;
const PATCH_RADIUS: i32 = 15; // 31×31 BRIEF patch
const N_PAIRS: usize = 256;
const HARRIS_K: f64 = 0.04;

#[derive(Default)]
pub struct OrbExtractor;

impl DescriptorExtractor for OrbExtractor {
    fn name(&self) -> &'static str {
        "orb"
    }

    fn detect(&self, gray: &GrayImage, max_features: usize) -> DescriptorSet {
        detect_orb(gray, max_features)
    }
}

/// Deterministic BRIEF pattern: 256 pairs inside a 31×31 patch,
/// generated with a fixed-seed xorshift (self-consistent across runs).
struct BriefPattern {
    pairs: [(f32, f32, f32, f32); N_PAIRS],
}

fn brief_pattern() -> &'static BriefPattern {
    use std::sync::OnceLock;
    static PAT: OnceLock<BriefPattern> = OnceLock::new();
    PAT.get_or_init(|| {
        let mut pairs = [(0f32, 0f32, 0f32, 0f32); N_PAIRS];
        let mut st: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        // Gaussian-ish sampling: average of two uniform draws
        for p in pairs.iter_mut() {
            let mut u = |lo: f32, hi: f32| {
                let a = (next() & 0xFFFF) as f32 / 65535.0;
                let b = (next() & 0xFFFF) as f32 / 65535.0;
                lo + (hi - lo) * (a + b) / 2.0
            };
            *p = (
                u(-15.0, 15.0),
                u(-15.0, 15.0),
                u(-15.0, 15.0),
                u(-15.0, 15.0),
            );
        }
        BriefPattern { pairs }
    })
}

// ---------- FAST-9 ----------

const CIRCLE: [(i32, i32); 16] = [
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 3),
    (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
];

/// FAST-9 corner score on row-major `data` (w×h) at pixel (x,y).
/// `Some(score)` when ≥9 contiguous circle pixels are all brighter or
/// all darker than centre ± `t`; score is the max contrast, capped 255.
fn fast9_score(data: &[u8], w: usize, h: usize, x: usize, y: usize, t: i32) -> Option<i32> {
    if x < 3 || y < 3 || x + 3 >= w || y + 3 >= h {
        return None;
    }
    let at = |dx: i32, dy: i32| {
        data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32
    };
    let c = at(0, 0);
    // quick reject on cardinal points (indices 0,4,8,12)
    let cards = [at(0, -3), at(3, 0), at(0, 3), at(-3, 0)];
    let bright = cards.iter().filter(|&&v| v > c + t).count();
    let dark = cards.iter().filter(|&&v| v < c - t).count();
    if bright < 3 && dark < 3 {
        return None;
    }
    let mut vals = [0i32; 16];
    let (mut mb, mut md) = (0u32, 0u32);
    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        let v = at(dx, dy);
        vals[i] = v;
        if v > c + t {
            mb |= 1 << i;
        }
        if v < c - t {
            md |= 1 << i;
        }
    }
    // circular run of ≥9 set bits: duplicate the 16-bit mask and test
    // every window of 9 consecutive bits at once
    let run9 = |m: u32| {
        let mm = m | (m << 16);
        mm & (mm >> 1) & (mm >> 2) & (mm >> 3) & (mm >> 4)
            & (mm >> 5) & (mm >> 6) & (mm >> 7) & (mm >> 8)
            != 0
    };
    if run9(mb) {
        Some(vals.iter().map(|&v| (v - c).max(0)).max().unwrap_or(0).min(255))
    } else if run9(md) {
        Some(vals.iter().map(|&v| (c - v).max(0)).max().unwrap_or(0).min(255))
    } else {
        None
    }
}

/// Per-pixel Sobel gradient-product maps (Ix², Iy², Ix·Iy) for one
/// pyramid level. Harris responses then become 3×3 window sums instead
/// of recomputing 18 Sobel taps per candidate pixel.
struct GradMaps {
    ixx: Vec<i32>,
    iyy: Vec<i32>,
    ixy: Vec<i32>,
    w: usize,
    h: usize,
}

impl GradMaps {
    fn new(gray: &GrayImage) -> Self {
        let (w, h) = (gray.width as usize, gray.height as usize);
        let mut m = GradMaps {
            ixx: vec![0; w * h],
            iyy: vec![0; w * h],
            ixy: vec![0; w * h],
            w,
            h,
        };
        for y in 1..h.saturating_sub(1) {
            let up = &gray.data[(y - 1) * w..y * w];
            let mid = &gray.data[y * w..(y + 1) * w];
            let dn = &gray.data[(y + 1) * w..(y + 2) * w];
            for x in 1..w.saturating_sub(1) {
                let gx = up[x + 1] as i32 + 2 * mid[x + 1] as i32 + dn[x + 1] as i32
                    - up[x - 1] as i32 - 2 * mid[x - 1] as i32 - dn[x - 1] as i32;
                let gy = dn[x - 1] as i32 + 2 * dn[x] as i32 + dn[x + 1] as i32
                    - up[x - 1] as i32 - 2 * up[x] as i32 - up[x + 1] as i32;
                m.ixx[y * w + x] = gx * gx;
                m.iyy[y * w + x] = gy * gy;
                m.ixy[y * w + x] = gx * gy;
            }
        }
        m
    }

    /// Harris response at (x,y) over a 3×3 window — identical to
    /// per-candidate Sobel recomputation: Sobel outputs are integers
    /// ≤1020, so every product and 9-element sum is exact in f64.
    fn response(&self, x: u32, y: u32) -> f64 {
        let (x, y) = (x as usize, y as usize);
        if x < 1 || y < 1 || x + 1 >= self.w || y + 1 >= self.h {
            return 0.0;
        }
        let (mut sxx, mut syy, mut sxy) = (0i64, 0i64, 0i64);
        for dy in 0..3 {
            let r = (y - 1 + dy) * self.w;
            for dx in 0..3 {
                let i = r + x - 1 + dx;
                sxx += self.ixx[i] as i64;
                syy += self.iyy[i] as i64;
                sxy += self.ixy[i] as i64;
            }
        }
        let (sxx, syy, sxy) = (sxx as f64, syy as f64, sxy as f64);
        sxx * syy - sxy * sxy - HARRIS_K * (sxx + syy) * (sxx + syy)
    }
}

// ---------- pyramid ----------

fn downscale(gray: &GrayImage, factor: f64) -> GrayImage {
    let nw = ((gray.width as f64) / factor).max(1.0) as u32;
    let nh = ((gray.height as f64) / factor).max(1.0) as u32;
    crate::image_io::resize_gray_exact(gray, nw, nh)
}

/// Single-pixel 3×3 box average with in-bounds count `n` — used for
/// border pixels where the fast full-window path doesn't apply.
#[inline]
fn blur3_px(data: &[u8], w: usize, h: usize, x: usize, y: usize) -> u8 {
    let mut s = 0u32;
    let mut n = 0u32;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let px = x as i32 + dx;
            let py = y as i32 + dy;
            if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                s += data[py as usize * w + px as usize] as u32;
                n += 1;
            }
        }
    }
    (s / n) as u8
}

/// 3×3 box blur — smooths noise for more stable BRIEF tests. Interior
/// pixels take a bounds-free row-slice path (identical `s/9` result);
/// only the 1-px border uses the counted generic path.
fn blur3(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width as usize, gray.height as usize);
    let mut out = vec![0u8; w * h];
    if w == 0 || h == 0 {
        return GrayImage::new(w as u32, h as u32, out);
    }
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 == h || w <= 2 {
            for (x, o) in row.iter_mut().enumerate() {
                *o = blur3_px(&gray.data, w, h, x, y);
            }
        } else {
            row[0] = blur3_px(&gray.data, w, h, 0, y);
            row[w - 1] = blur3_px(&gray.data, w, h, w - 1, y);
            let up = &gray.data[(y - 1) * w..y * w];
            let mid = &gray.data[y * w..(y + 1) * w];
            let dn = &gray.data[(y + 1) * w..(y + 2) * w];
            for x in 1..w - 1 {
                let s = up[x - 1] as u32 + up[x] as u32 + up[x + 1] as u32
                    + mid[x - 1] as u32 + mid[x] as u32 + mid[x + 1] as u32
                    + dn[x - 1] as u32 + dn[x] as u32 + dn[x + 1] as u32;
                row[x] = (s / 9) as u8;
            }
        }
    });
    GrayImage::new(w as u32, h as u32, out)
}

/// Row half-widths of the radius-15 patch circle: `sqrt(15² − dy²)`.
fn circle_half_widths() -> &'static [i32; 31] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[i32; 31]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0i32; 31];
        for dy in -PATCH_RADIUS..=PATCH_RADIUS {
            t[(dy + PATCH_RADIUS) as usize] =
                ((PATCH_RADIUS * PATCH_RADIUS - dy * dy) as f64).sqrt() as i32;
        }
        t
    })
}

/// Intensity-centroid orientation within a circular patch (radius 15).
///
/// Callers only pass keypoints ≥ PATCH_RADIUS+2 px from every border,
/// so the whole circle is in bounds; per-row moment sums are exact
/// integers, so grouping by row (`dy·Σv`) equals per-pixel `Σ(dy·v)`.
fn centroid_angle(gray: &GrayImage, x: u32, y: u32) -> f32 {
    let w = gray.width as usize;
    let (cx, cy) = (x as usize, y as usize);
    let hw = circle_half_widths();
    let mut m10 = 0f64;
    let mut m01 = 0f64;
    for dy in -PATCH_RADIUS..=PATCH_RADIUS {
        let half = hw[(dy + PATCH_RADIUS) as usize] as usize;
        let py = (cy as i32 + dy) as usize;
        let row = &gray.data[py * w + cx - half..py * w + cx + half + 1];
        let mut row_v = 0f64;
        let mut row_dv = 0f64;
        for (i, &v) in row.iter().enumerate() {
            let vf = v as f64;
            row_v += vf;
            row_dv += (i as i32 - half as i32) as f64 * vf;
        }
        m10 += row_dv;
        m01 += dy as f64 * row_v;
    }
    m01.atan2(m10) as f32
}

/// Sample rotated BRIEF pair test at keypoint → 32-byte descriptor.
fn brief_describe(gray: &GrayImage, x: f32, y: f32, angle: f32, pat: &BriefPattern) -> [u8; 32] {
    let w = gray.width as i32;
    let h = gray.height as i32;
    let (sin, cos) = angle.sin_cos();
    let sample = |px: f32, py: f32| -> i32 {
        let rx = cos * px - sin * py;
        let ry = sin * px + cos * py;
        let xi = (x + rx).round() as i32;
        let yi = (y + ry).round() as i32;
        let xi = xi.clamp(0, w - 1);
        let yi = yi.clamp(0, h - 1);
        gray.data[(yi * w + xi) as usize] as i32
    };
    let mut out = [0u8; 32];
    for (i, &(x1, y1, x2, y2)) in pat.pairs.iter().enumerate() {
        if sample(x1, y1) < sample(x2, y2) {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// FAST candidates over the border-excluded interior, in scan order.
/// Falls back to a lower threshold when the strict pass finds <32
/// corners; the second pass dedupes through an occupancy grid so each
/// position appears at most once, first hit (strict pass) winning.
fn fast_candidates(sm: &GrayImage, border: i32) -> Vec<(u32, u32, i32)> {
    let (w, h) = (sm.width as usize, sm.height as usize);
    let b = border as usize;
    // rows are independent; collecting per-row then concatenating keeps
    // the same scan order as a sequential pass
    let scan = |t: i32| -> Vec<(u32, u32, i32)> {
        (b..h - b)
            .into_par_iter()
            .map(|y| {
                let mut hits = Vec::new();
                for x in b..w - b {
                    if let Some(s) = fast9_score(&sm.data, w, h, x, y, t) {
                        hits.push((x as u32, y as u32, s));
                    }
                }
                hits
            })
            .collect::<Vec<_>>()
            .concat()
    };
    let mut found = scan(FAST_THRESHOLD);
    if found.len() < 32 {
        let mut occupied = vec![false; w * h];
        for &(x, y, _) in &found {
            occupied[y as usize * w + x as usize] = true;
        }
        for (x, y, s) in scan(FAST_THRESHOLD_LOW) {
            let i = y as usize * w + x as usize;
            if !occupied[i] {
                occupied[i] = true;
                found.push((x, y, s));
            }
        }
    }
    found
}

/// Full ORB pipeline: one blurred pyramid computed once per level and
/// shared by FAST detection, Harris scoring, orientation and BRIEF
/// description (each level is downscaled from `gray`, never chained,
/// so all per-level work is independent and runs in parallel).
pub fn detect_orb(gray: &GrayImage, max_features: usize) -> DescriptorSet {
    let pat = brief_pattern();
    let border = PATCH_RADIUS + 2;

    // cumulative scales: level l is `gray` downscaled by scales[l]
    let mut scales = Vec::with_capacity(PYRAMID_LEVELS);
    let mut s = 1.0f64;
    for _ in 0..PYRAMID_LEVELS {
        scales.push(s);
        s *= SCALE_FACTOR;
    }

    // level dims shrink monotonically, so the original "stop when the
    // level is too small" is equivalent to counting usable levels here
    let mut n_levels = 0;
    while n_levels < PYRAMID_LEVELS {
        let (lw, lh) = if n_levels == 0 {
            (gray.width, gray.height)
        } else {
            (
                ((gray.width as f64) / scales[n_levels]).max(1.0) as u32,
                ((gray.height as f64) / scales[n_levels]).max(1.0) as u32,
            )
        };
        if lw <= 2 * border as u32 || lh <= 2 * border as u32 {
            break;
        }
        n_levels += 1;
    }

    // per level: blurred image (kept for the describe phase) + keypoints
    let levels: Vec<(GrayImage, Vec<Keypoint>)> = (0..n_levels)
        .into_par_iter()
        .map(|l| {
            let owned;
            let base: &GrayImage = if l == 0 {
                gray
            } else {
                owned = downscale(gray, scales[l]);
                &owned
            };
            let sm = blur3(base);
            let found = fast_candidates(&sm, border);
            let grads = GradMaps::new(&sm);
            let scale = scales[l] as f32;
            let kps = found
                .par_iter()
                .map(|&(x, y, fast_s)| Keypoint {
                    x: x as f32 * scale,
                    y: y as f32 * scale,
                    level: l as u8,
                    angle: centroid_angle(&sm, x, y),
                    response: grads.response(x, y) as f32 * (1.0 + fast_s as f32 / 255.0),
                })
                .collect();
            (sm, kps)
        })
        .collect();
    let mut cands: Vec<Keypoint> =
        levels.iter().flat_map(|(_, k)| k.iter().copied()).collect();

    // keep top max_features by response, roughly balanced per level
    cands.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(max_features.max(64));

    // describe each keypoint on its own level's blurred image
    cands.sort_by_key(|k| k.level);
    let bytes: Vec<[u8; 32]> = cands
        .par_iter()
        .map(|kp| {
            let l = kp.level as usize;
            let li = &levels[l].0;
            brief_describe(li, kp.x / scales[l] as f32, kp.y / scales[l] as f32, kp.angle, pat)
        })
        .collect();
    let mut data = Vec::with_capacity(bytes.len() * 32);
    for b in &bytes {
        data.extend_from_slice(b);
    }
    DescriptorSet { keypoints: cands, desc_len: 32, data }
}
