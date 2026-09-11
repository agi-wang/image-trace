//! Pure-Rust ORB: FAST-9 corner detection on an image pyramid,
//! intensity-centroid orientation, and rotated BRIEF-256 descriptors.

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

fn fast9_score(gray: &GrayImage, x: u32, y: u32, t: i32) -> Option<i32> {
    let w = gray.width as i32;
    let h = gray.height as i32;
    let (x, y) = (x as i32, y as i32);
    if x < 3 || y < 3 || x >= w - 3 || y >= h - 3 {
        return None;
    }
    let at = |px: i32, py: i32| gray.data[(py * w + px) as usize] as i32;
    let c = at(x, y);
    // quick reject on cardinal points (indices 0,4,8,12)
    let cards = [at(x + CIRCLE[0].0, y + CIRCLE[0].1), at(x + CIRCLE[4].0, y + CIRCLE[4].1),
        at(x + CIRCLE[8].0, y + CIRCLE[8].1), at(x + CIRCLE[12].0, y + CIRCLE[12].1)];
    let bright = cards.iter().filter(|&&v| v > c + t).count();
    let dark = cards.iter().filter(|&&v| v < c - t).count();
    if bright < 3 && dark < 3 {
        return None;
    }
    let mut vals = [0i32; 16];
    for (i, (dx, dy)) in CIRCLE.iter().enumerate() {
        vals[i] = at(x + dx, y + dy);
    }
    // contiguous run of ≥9 all brighter or all darker
    let mut best_run = 0i32;
    let mut run = 0i32;
    for k in 0..32 {
        let v = vals[k % 16];
        if v > c + t {
            run += 1;
        } else {
            best_run = best_run.max(run);
            run = 0;
        }
    }
    best_run = best_run.max(run);
    if best_run >= 9 {
        let min_bright = vals.iter().map(|&v| (v - c).max(0)).max().unwrap_or(0);
        Some(min_bright.min(255))
    } else {
        let mut run = 0i32;
        let mut best = 0i32;
        for k in 0..32 {
            let v = vals[k % 16];
            if v < c - t {
                run += 1;
            } else {
                best = best.max(run);
                run = 0;
            }
        }
        best = best.max(run);
        if best >= 9 {
            let min_dark = vals.iter().map(|&v| (c - v).max(0)).max().unwrap_or(0);
            Some(min_dark.min(255))
        } else {
            None
        }
    }
}

/// Harris response at (x,y) using Sobel derivatives over a 3×3 window.
fn harris_response(gray: &GrayImage, x: u32, y: u32) -> f64 {
    let w = gray.width as i32;
    let h = gray.height as i32;
    let (x, y) = (x as i32, y as i32);
    if x < 1 || y < 1 || x >= w - 1 || y >= h - 1 {
        return 0.0;
    }
    let at = |px: i32, py: i32| gray.data[(py * w + px) as usize] as f64;
    let sobel_x = |px: i32, py: i32| {
        at(px + 1, py - 1) + 2.0 * at(px + 1, py) + at(px + 1, py + 1)
            - at(px - 1, py - 1) - 2.0 * at(px - 1, py) - at(px - 1, py + 1)
    };
    let sobel_y = |px: i32, py: i32| {
        at(px - 1, py + 1) + 2.0 * at(px, py + 1) + at(px + 1, py + 1)
            - at(px - 1, py - 1) - 2.0 * at(px, py - 1) - at(px + 1, py - 1)
    };
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let gx = sobel_x(x + dx, y + dy);
            let gy = sobel_y(x + dx, y + dy);
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    sxx * syy - sxy * sxy - HARRIS_K * (sxx + syy) * (sxx + syy)
}

// ---------- pyramid ----------

fn downscale(gray: &GrayImage, factor: f64) -> GrayImage {
    let nw = ((gray.width as f64) / factor).max(1.0) as u32;
    let nh = ((gray.height as f64) / factor).max(1.0) as u32;
    crate::image_io::resize_gray_exact(gray, nw, nh)
}

/// 3×3 box blur — smooths noise for more stable BRIEF tests.
fn blur3(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width as usize, gray.height as usize);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            let mut n = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let px = x as i32 + dx;
                    let py = y as i32 + dy;
                    if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                        s += gray.data[(py as usize) * w + px as usize] as u32;
                        n += 1;
                    }
                }
            }
            out[y * w + x] = (s / n) as u8;
        }
    }
    GrayImage::new(w as u32, h as u32, out)
}

/// Intensity-centroid orientation within a circular patch (radius 15).
fn centroid_angle(gray: &GrayImage, x: u32, y: u32) -> f32 {
    let w = gray.width as i32;
    let h = gray.height as i32;
    let (cx, cy) = (x as i32, y as i32);
    let r = PATCH_RADIUS;
    let mut m10 = 0f64;
    let mut m01 = 0f64;
    for dy in -r..=r {
        // circular patch
        let half_w = ((r * r - dy * dy) as f64).sqrt() as i32;
        for dx in -half_w..=half_w {
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let v = gray.data[(py * w + px) as usize] as f64;
            m10 += dx as f64 * v;
            m01 += dy as f64 * v;
        }
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

/// Full ORB pipeline.
pub fn detect_orb(gray: &GrayImage, max_features: usize) -> DescriptorSet {
    let pat = brief_pattern();
    let mut cands: Vec<Keypoint> = Vec::new();

    // build pyramid, detect per level
    let mut level_img = gray.clone();
    let mut scale = 1.0f64;
    for level in 0..PYRAMID_LEVELS {
        if level > 0 {
            scale *= SCALE_FACTOR;
            level_img = downscale(gray, scale);
        }
        let sm = blur3(&level_img);
        let w = sm.width as i32;
        let h = sm.height as i32;
        let b = PATCH_RADIUS + 2;
        if w <= 2 * b || h <= 2 * b {
            break;
        }
        // FAST pass at both thresholds; dedupe by position
        let mut found: Vec<(u32, u32, i32)> = Vec::new();
        for y in (b as u32)..(h - b) as u32 {
            for x in (b as u32)..(w - b) as u32 {
                if let Some(s) = fast9_score(&sm, x, y, FAST_THRESHOLD) {
                    found.push((x, y, s));
                }
            }
        }
        if found.len() < 32 {
            for y in (b as u32)..(h - b) as u32 {
                for x in (b as u32)..(w - b) as u32 {
                    if let Some(s) = fast9_score(&sm, x, y, FAST_THRESHOLD_LOW) {
                        if !found.iter().any(|&(fx, fy, _)| fx == x && fy == y) {
                            found.push((x, y, s));
                        }
                    }
                }
            }
        }
        for (x, y, fast_s) in found {
            let resp = harris_response(&sm, x, y);
            let angle = centroid_angle(&sm, x, y);
            cands.push(Keypoint {
                x: x as f32 * scale as f32,
                y: y as f32 * scale as f32,
                level: level as u8,
                angle,
                response: resp as f32 * (1.0 + fast_s as f32 / 255.0),
            });
        }
    }

    // keep top max_features by response, roughly balanced per level
    cands.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(max_features.max(64));

    // describe each keypoint at its own pyramid level
    // (recompute the level's blurred image lazily per level group)
    cands.sort_by_key(|k| k.level);
    let mut desc = DescriptorSet { keypoints: Vec::new(), desc_len: 32, data: Vec::new() };
    let mut level_imgs: Vec<Option<GrayImage>> = vec![None; PYRAMID_LEVELS];
    let mut scale = 1.0f64;
    let mut scales = vec![1.0f64];
    for _ in 1..PYRAMID_LEVELS {
        scale *= SCALE_FACTOR;
        scales.push(scale);
    }
    for kp in &cands {
        let l = kp.level as usize;
        if level_imgs[l].is_none() {
            let base = if l == 0 { gray.clone() } else { downscale(gray, scales[l]) };
            level_imgs[l] = Some(blur3(&base));
        }
        let li = level_imgs[l].as_ref().unwrap();
        let lx = kp.x / scales[l] as f32;
        let ly = kp.y / scales[l] as f32;
        let bytes = brief_describe(li, lx, ly, kp.angle, pat);
        desc.keypoints.push(*kp);
        desc.data.extend_from_slice(&bytes);
    }
    // restore original order irrelevant for matching
    desc
}
