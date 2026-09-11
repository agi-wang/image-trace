//! Tier 2 pixel/structure metrics: SSIM, HSV histogram correlation, NCC template.

use std::cmp::Ordering;

use crate::{GrayImage, RgbImage};

/// Structural similarity index (uniform 7×7 window, K1=0.01 K2=0.03).
/// Both inputs must share dimensions.
///
/// Window statistics come from u64 summed-area tables (Σa, Σb, Σa², Σb², Σab)
/// instead of filtered f64 planes. Every accumulated value is an exact
/// integer — at the ≤512-px working size a table entry is ≤ 255²·512² ≪ 2⁵³ —
/// so the resulting score matches the old box-filter version to ~1e-13.
pub fn ssim(a: &GrayImage, b: &GrayImage) -> f64 {
    let w = a.width.min(b.width) as usize;
    let h = a.height.min(b.height) as usize;
    if w < 8 || h < 8 {
        return 0.0;
    }
    let (aw, bw) = (a.width as usize, b.width as usize);
    let sw = w + 1; // table row stride
    let cells = sw * (h + 1);
    let mut sa = vec![0u64; cells];
    let mut sb = vec![0u64; cells];
    let mut saa = vec![0u64; cells];
    let mut sbb = vec![0u64; cells];
    let mut sab = vec![0u64; cells];
    for y in 0..h {
        let arow = &a.data[y * aw..y * aw + w];
        let brow = &b.data[y * bw..y * bw + w];
        let (up, cur) = (y * sw, (y + 1) * sw);
        let (mut ra, mut rb, mut raa, mut rbb, mut rab) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for x in 0..w {
            let (va, vb) = (arow[x] as u64, brow[x] as u64);
            ra += va;
            rb += vb;
            raa += va * va;
            rbb += vb * vb;
            rab += va * vb;
            sa[cur + x + 1] = sa[up + x + 1] + ra;
            sb[cur + x + 1] = sb[up + x + 1] + rb;
            saa[cur + x + 1] = saa[up + x + 1] + raa;
            sbb[cur + x + 1] = sbb[up + x + 1] + rbb;
            sab[cur + x + 1] = sab[up + x + 1] + rab;
        }
    }
    // Exact-integer sum of `t` over the rect [x0,x1) × [y0,y1).
    // (A + D) − (B + C) ordering: the bracketed terms never underflow.
    let wsum = |t: &[u64], x0: usize, y0: usize, x1: usize, y1: usize| -> f64 {
        ((t[y1 * sw + x1] + t[y0 * sw + x0]) - (t[y0 * sw + x1] + t[y1 * sw + x0])) as f64
    };
    let c1 = (0.01f64 * 255.0).powi(2);
    let c2 = (0.03f64 * 255.0).powi(2);

    let mut sum = 0.0;
    // skip a half-window border like scikit-image's crop
    let r = 3usize;
    let pad = r.min(w / 2).min(h / 2);
    debug_assert_eq!(pad, r); // w,h ≥ 8 ⇒ pad == r ⇒ every scored window is full 7×7
    let wn = ((2 * r + 1) * (2 * r + 1)) as f64;
    let mut cnt = 0usize;
    for y in pad..h - pad {
        for x in pad..w - pad {
            let (x0, x1) = (x - r, x + r + 1);
            let (y0, y1) = (y - r, y + r + 1);
            let mux = wsum(&sa, x0, y0, x1, y1) / wn;
            let muy = wsum(&sb, x0, y0, x1, y1) / wn;
            let vx = (wsum(&saa, x0, y0, x1, y1) / wn - mux * mux).max(0.0);
            let vy = (wsum(&sbb, x0, y0, x1, y1) / wn - muy * muy).max(0.0);
            let cxy = wsum(&sab, x0, y0, x1, y1) / wn - mux * muy;
            let num = (2.0 * mux * muy + c1) * (2.0 * cxy + c2);
            let den = (mux * mux + muy * muy + c1) * (vx + vy + c2);
            sum += num / den;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return 0.0;
    }
    (sum / cnt as f64).clamp(0.0, 1.0)
}

/// HSV histogram: 50 hue × 60 sat bins, L2-normalized (OpenCV-compatible dims).
///
/// The output is a persisted feature, so the per-pixel f32 arithmetic is kept
/// exactly as before — only the pixel loop is sliced (`chunks_exact(3)`) to
/// drop bounds checks.
pub fn hsv_histogram(rgb: &RgbImage) -> Vec<f32> {
    const HBINS: usize = 50;
    const SBINS: usize = 60;
    const HF: f32 = HBINS as f32;
    const SF: f32 = SBINS as f32;
    let mut hist = vec![0f32; HBINS * SBINS];
    let n = (rgb.width as usize) * (rgb.height as usize);
    for px in rgb.data[..n * 3].chunks_exact(3) {
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        let (h, s, _v) = rgb_to_hsv_f32(r, g, b);
        // OpenCV hue range is [0,180)
        let hb = ((h / 180.0) * HF).floor().clamp(0.0, HF - 1.0) as usize;
        let sb = (s * SF).floor().clamp(0.0, SF - 1.0) as usize;
        hist[hb * SBINS + sb] += 1.0;
    }
    let norm: f32 = hist.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in hist.iter_mut() {
            *v /= norm;
        }
    }
    hist
}

/// Pearson correlation between two equal-length vectors → [0,1].
pub fn histogram_correlation(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    // one fused pass for both means (same accumulation order as before)
    let (sa, sb) = a
        .iter()
        .zip(b.iter())
        .fold((0.0, 0.0), |(sa, sb), (&x, &y)| (sa + x as f64, sb + y as f64));
    let ma = sa / n;
    let mb = sb / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64 - ma;
        let y = y as f64 - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    let corr = num / (da * db).sqrt(); // [-1,1]
    ((corr + 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Whole-image normalized cross-correlation (TM_CCOEFF_NORMED equivalent).
/// Both inputs must share dimensions.
///
/// Accumulates in u64: every sum is an exact integer (< 2⁵³ at the ≤512-px
/// working size), so the single f64 conversion per sum reproduces the old
/// f64 accumulators bit for bit.
pub fn ncc(a: &GrayImage, b: &GrayImage) -> f64 {
    let w = a.width.min(b.width) as usize;
    let h = a.height.min(b.height) as usize;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let aw = a.width as usize;
    let bw = b.width as usize;
    let n = (w * h) as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for y in 0..h {
        let ra = &a.data[y * aw..y * aw + w];
        let rb = &b.data[y * bw..y * bw + w];
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            let (va, vb) = (va as u64, vb as u64);
            sa += va;
            sb += vb;
            saa += va * va;
            sbb += vb * vb;
            sab += va * vb;
        }
    }
    let (sa, sb, saa, sbb, sab) = (sa as f64, sb as f64, saa as f64, sbb as f64, sab as f64);
    let num = n * sab - sa * sb;
    let den = ((n * saa - sa * sa) * (n * sbb - sb * sb)).sqrt();
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).clamp(-1.0, 1.0).max(0.0)
}

// ---------------------------------------------------------------------------
// Template matching — coarse-to-fine NCC over a half-size image pyramid
// ---------------------------------------------------------------------------

/// Maximum number of pyramid halvings below the full-resolution level.
const TM_MAX_HALVES: usize = 3;
/// Stop descending once the search box fits inside ~TM_COARSE² positions.
const TM_COARSE: usize = 64;
/// Minimum template side for a coarse level to stay informative.
const TM_MIN_TPL: usize = 4;
/// Candidate positions refined per level.
const TM_TOP_K: usize = 8;
/// ±px neighborhood evaluated around each ×2-scaled candidate.
const TM_REFINE: i64 = 2;
/// Below this coarse-level best score the pyramid is untrusted → exact scan.
const TM_FALLBACK: f64 = 0.5;

/// Per-level precomputations: u64 summed-area tables of img and img² plus
/// template statistics. The integer tables are exact (< 2⁵³ at the ≤512-px
/// working size), so `as f64` casts reproduce the old f64 integral images.
struct NccPrep {
    stride: usize, // iw + 1
    integ: Vec<u64>,
    integ2: Vec<u64>,
    tn: f64,
    tmean: f64,
    tvar: f64, // Σ(t − tmean)²
}

/// (n, mean, Σ(t−mean)²) of the template — same f64 result as the old loop.
fn tpl_stats(tpl: &GrayImage) -> (f64, f64, f64) {
    let tn = tpl.data.len() as f64;
    let (mut tsum, mut tss) = (0u64, 0u64);
    for &v in &tpl.data {
        let v = v as u64;
        tsum += v;
        tss += v * v;
    }
    let (tsum, tss) = (tsum as f64, tss as f64);
    let tmean = tsum / tn;
    (tn, tmean, tss - tsum * tmean)
}

fn ncc_prep(img: &GrayImage, tpl: &GrayImage) -> NccPrep {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let stride = iw + 1;
    let mut integ = vec![0u64; stride * (ih + 1)];
    let mut integ2 = vec![0u64; stride * (ih + 1)];
    for y in 0..ih {
        let row = &img.data[y * iw..y * iw + iw];
        let (up, cur) = (y * stride, (y + 1) * stride);
        let (mut rs, mut rs2) = (0u64, 0u64);
        for x in 0..iw {
            let v = row[x] as u64;
            rs += v;
            rs2 += v * v;
            integ[cur + x + 1] = integ[up + x + 1] + rs;
            integ2[cur + x + 1] = integ2[up + x + 1] + rs2;
        }
    }
    let (tn, tmean, tvar) = tpl_stats(tpl);
    NccPrep {
        stride,
        integ,
        integ2,
        tn,
        tmean,
        tvar,
    }
}

/// NCC score of `tpl` placed at offset (x, y). `None` mirrors the old
/// `continue`: zero-variance windows (or a uniform template) score nothing.
fn ncc_at(img: &GrayImage, tpl: &GrayImage, p: &NccPrep, x: usize, y: usize) -> Option<f64> {
    if p.tvar <= 0.0 {
        return None;
    }
    let iw = img.width as usize;
    let (tw, th) = (tpl.width as usize, tpl.height as usize);
    let s = p.stride;
    let (x2, y2) = (x + tw, y + th);
    // (A + D) − (B + C) ordering keeps u64 subtraction non-negative.
    let wsum = (p.integ[y2 * s + x2] + p.integ[y * s + x]
        - (p.integ[y * s + x2] + p.integ[y2 * s + x])) as f64;
    let wss = (p.integ2[y2 * s + x2] + p.integ2[y * s + x]
        - (p.integ2[y * s + x2] + p.integ2[y2 * s + x])) as f64;
    let wmean = wsum / p.tn;
    let wvar = wss - wsum * wmean;
    if wvar <= 0.0 {
        return None;
    }
    // Cross term Σ img·tpl row by row: u32 products widened into a u64 sum —
    // integer-exact, so identical to the old f64 accumulation.
    let mut xterm = 0u64;
    for ty in 0..th {
        let irow = &img.data[(y + ty) * iw + x..(y + ty) * iw + x + tw];
        let trow = &tpl.data[ty * tw..ty * tw + tw];
        xterm += irow
            .iter()
            .zip(trow.iter())
            .map(|(&a, &b)| (a as u32 * b as u32) as u64)
            .sum::<u64>();
    }
    let cov = xterm as f64 - p.tn * wmean * p.tmean;
    Some(cov / (wvar * p.tvar).sqrt())
}

/// Best-first order: score desc; ties in row-major scan order (y, then x) —
/// the same tie-break the old linear scan's first-max-wins rule produced.
fn ncc_cmp(a: &(f64, u32, u32), b: &(f64, u32, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(Ordering::Equal)
        .then(a.2.cmp(&b.2))
        .then(a.1.cmp(&b.1))
}

/// Exhaustive NCC scan → every scored position, best first.
fn ncc_positions(img: &GrayImage, tpl: &GrayImage, p: &NccPrep) -> Vec<(f64, u32, u32)> {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (tw, th) = (tpl.width as usize, tpl.height as usize);
    let mut out = Vec::with_capacity((iw - tw + 1) * (ih - th + 1));
    for y in 0..=(ih - th) {
        for x in 0..=(iw - tw) {
            if let Some(s) = ncc_at(img, tpl, p, x, y) {
                out.push((s, x as u32, y as u32));
            }
        }
    }
    out.sort_unstable_by(ncc_cmp);
    out
}

/// The original exhaustive scan → (best score ≥ 0, x, y). Used directly for
/// small search areas and as the exact fallback for weak pyramid matches.
fn ncc_scan(img: &GrayImage, tpl: &GrayImage) -> (f64, u32, u32) {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (tw, th) = (tpl.width as usize, tpl.height as usize);
    let p = ncc_prep(img, tpl);
    let mut best = f64::MIN;
    let (mut bx, mut by) = (0u32, 0u32);
    for y in 0..=(ih - th) {
        for x in 0..=(iw - tw) {
            if let Some(s) = ncc_at(img, tpl, &p, x, y) {
                if s > best {
                    best = s;
                    bx = x as u32;
                    by = y as u32;
                }
            }
        }
    }
    (best.max(0.0), bx, by)
}

/// 2×2 box half of `g` (odd edges dropped). Only ever feeds pyramid levels —
/// its rounding affects coarse guidance, never the returned score.
fn halve(g: &GrayImage) -> GrayImage {
    let (w, h) = (g.width as usize, g.height as usize);
    let (nw, nh) = (w / 2, h / 2);
    let mut out = vec![0u8; nw * nh];
    for y in 0..nh {
        let s0 = &g.data[2 * y * w..2 * y * w + 2 * nw];
        let s1 = &g.data[(2 * y + 1) * w..(2 * y + 1) * w + 2 * nw];
        let orow = &mut out[y * nw..y * nw + nw];
        for x in 0..nw {
            orow[x] = ((s0[2 * x] as u16
                + s0[2 * x + 1] as u16
                + s1[2 * x] as u16
                + s1[2 * x + 1] as u16
                + 2)
                / 4) as u8;
        }
    }
    GrayImage::new(nw as u32, nh as u32, out)
}

/// Template matching: slide `tpl` over `img`, return max NCC score and its (x, y).
/// Both grayscale; tpl must be <= img in each dim.
///
/// Large search areas run a coarse-to-fine pyramid: an exhaustive NCC scan on
/// the smallest half-size level whose search box is ≤ ~64×64, then each of
/// the ~8 best candidates is refined through a ±2 px neighborhood on the
/// next finer level down to full resolution. When the coarsest match is weak
/// (< 0.5) the pyramid is untrusted and the exact full-resolution scan runs
/// instead, preserving the old worst-case semantics. Small search areas go
/// straight to the exact scan.
pub fn template_match(img: &GrayImage, tpl: &GrayImage) -> (f64, u32, u32) {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (tw, th) = (tpl.width as usize, tpl.height as usize);
    if tw == 0 || th == 0 || iw < tw || ih < th {
        return (0.0, 0, 0);
    }
    let (_, _, tvar) = tpl_stats(tpl);
    if tvar <= 0.0 {
        return (0.0, 0, 0);
    }
    if iw - tw < TM_COARSE && ih - th < TM_COARSE {
        return ncc_scan(img, tpl); // small search area → exact scan
    }

    // Pyramid of half-size levels until the search box fits ~TM_COARSE²
    // (≤ TM_MAX_HALVES levels, template stays ≥ TM_MIN_TPL).
    let mut img_pyr: Vec<GrayImage> = Vec::new();
    let mut tpl_pyr: Vec<GrayImage> = Vec::new();
    loop {
        let lvl = img_pyr.len();
        let (cw, ch, dw, dh) = if lvl == 0 {
            (iw, ih, tw, th)
        } else {
            (
                img_pyr[lvl - 1].width as usize,
                img_pyr[lvl - 1].height as usize,
                tpl_pyr[lvl - 1].width as usize,
                tpl_pyr[lvl - 1].height as usize,
            )
        };
        let coarse = cw - dw < TM_COARSE && ch - dh < TM_COARSE;
        if coarse || lvl >= TM_MAX_HALVES || dw / 2 < TM_MIN_TPL || dh / 2 < TM_MIN_TPL {
            break;
        }
        img_pyr.push(halve(if lvl == 0 { img } else { &img_pyr[lvl - 1] }));
        tpl_pyr.push(halve(if lvl == 0 { tpl } else { &tpl_pyr[lvl - 1] }));
    }
    if img_pyr.is_empty() {
        return ncc_scan(img, tpl); // template too small to halve
    }

    // Exhaustive scan at the coarsest level → best ~TM_TOP_K candidates.
    let depth = img_pyr.len() - 1;
    let (ci, ct) = (&img_pyr[depth], &tpl_pyr[depth]);
    let prep = ncc_prep(ci, ct);
    let mut cand = ncc_positions(ci, ct, &prep);
    let best_coarse = cand.first().map_or(f64::MIN, |c| c.0);
    if best_coarse < TM_FALLBACK {
        return ncc_scan(img, tpl); // weak coarse match → exact scan
    }
    cand.truncate(TM_TOP_K);

    // Refine: every candidate seeds a ±TM_REFINE neighborhood around its ×2
    // position on the next finer level; keep the best TM_TOP_K again.
    for lvl in (0..img_pyr.len()).rev() {
        let (li, lt) = if lvl == 0 {
            (img, tpl)
        } else {
            (&img_pyr[lvl - 1], &tpl_pyr[lvl - 1])
        };
        let lp = ncc_prep(li, lt);
        let mx = (li.width - lt.width) as i64;
        let my = (li.height - lt.height) as i64;
        let mut pos: Vec<(u32, u32)> = Vec::with_capacity(cand.len() * 25);
        for &(_, cx, cy) in &cand {
            let (bx, by) = (2 * cx as i64, 2 * cy as i64);
            for dy in -TM_REFINE..=TM_REFINE {
                for dx in -TM_REFINE..=TM_REFINE {
                    let (nx, ny) = (bx + dx, by + dy);
                    if (0..=mx).contains(&nx) && (0..=my).contains(&ny) {
                        pos.push((nx as u32, ny as u32));
                    }
                }
            }
        }
        pos.sort_unstable();
        pos.dedup();
        let mut scored: Vec<(f64, u32, u32)> = pos
            .into_iter()
            .filter_map(|(x, y)| ncc_at(li, lt, &lp, x as usize, y as usize).map(|s| (s, x, y)))
            .collect();
        if scored.is_empty() {
            return ncc_scan(img, tpl); // every window flat → exact scan
        }
        scored.sort_unstable_by(ncc_cmp);
        scored.truncate(TM_TOP_K);
        cand = scored;
    }
    let (s, x, y) = cand[0];
    (s.max(0.0), x, y)
}

#[inline]
fn rgb_to_hsv_f32(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max > 0.0 { d / max } else { 0.0 };
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h.rem_euclid(360.0), s, v)
}

// ---------------------------------------------------------------------------
// Equivalence checks against the original (pre-optimization) implementations
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured gray image (x^y texture + gradients + LCG blobs).
    fn photo(w: u32, h: u32, seed: u8) -> GrayImage {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 255) / w;
                let g = (y * 255) / h;
                let b = (x ^ y).wrapping_add(seed as u32);
                data.push(((r + g + b) / 3) as u8);
            }
        }
        let mut k = seed as u32 + 1;
        for _ in 0..40 {
            k = k.wrapping_mul(1103515245).wrapping_add(12345);
            let cx = (k >> 8) % w.max(1);
            k = k.wrapping_mul(1103515245).wrapping_add(12345);
            let cy = (k >> 8) % h.max(1);
            let rad = 3 + (k >> 4) % 12;
            for dy in -(rad as i32)..=rad as i32 {
                for dx in -(rad as i32)..=rad as i32 {
                    if dx * dx + dy * dy <= (rad * rad) as i32 {
                        let px = (cx as i32 + dx).clamp(0, w as i32 - 1) as u32;
                        let py = (cy as i32 + dy).clamp(0, h as i32 - 1) as u32;
                        let i = (py * w + px) as usize;
                        data[i] = 255 - data[i];
                    }
                }
            }
        }
        GrayImage::new(w, h, data)
    }

    fn crop(g: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
        let mut data = Vec::with_capacity((w * h) as usize);
        for row in y..y + h {
            let i = (row * g.width + x) as usize;
            data.extend_from_slice(&g.data[i..i + w as usize]);
        }
        GrayImage::new(w, h, data)
    }

    // ---- verbatim references of the old implementations ----

    fn ref_box_mean(data: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
        let mut tmp = vec![0.0; w * h];
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            let row = &data[y * w..y * w + w];
            let mut acc = 0.0;
            for &v in row.iter().take(r.min(w - 1) + 1) {
                acc += v;
            }
            for x in 0..w {
                let left = x.saturating_sub(r);
                let right = (x + r).min(w - 1);
                tmp[y * w + x] = acc / (right - left + 1) as f64;
                let left_next = (x + 1).saturating_sub(r);
                if left_next > left {
                    acc -= row[left];
                }
                let right_next = (x + 1 + r).min(w - 1);
                if right_next > right {
                    acc += row[right_next];
                }
            }
        }
        for x in 0..w {
            let mut acc = 0.0;
            for y in 0..=r.min(h - 1) {
                acc += tmp[y * w + x];
            }
            for y in 0..h {
                let top = y.saturating_sub(r);
                let bot = (y + r).min(h - 1);
                out[y * w + x] = acc / (bot - top + 1) as f64;
                let top_next = (y + 1).saturating_sub(r);
                if top_next > top {
                    acc -= tmp[top * w + x];
                }
                let bot_next = (y + 1 + r).min(h - 1);
                if bot_next > bot {
                    acc += tmp[bot_next * w + x];
                }
            }
        }
        out
    }

    fn ref_ssim(a: &GrayImage, b: &GrayImage) -> f64 {
        let w = a.width.min(b.width) as usize;
        let h = a.height.min(b.height) as usize;
        if w < 8 || h < 8 {
            return 0.0;
        }
        let aw = a.width as usize;
        let bw = b.width as usize;
        let av: Vec<f64> = (0..h)
            .flat_map(|y| (0..w).map(move |x| a.data[y * aw + x] as f64))
            .collect();
        let bv: Vec<f64> = (0..h)
            .flat_map(|y| (0..w).map(move |x| b.data[y * bw + x] as f64))
            .collect();
        let n = w * h;
        let a2: Vec<f64> = av.iter().map(|v| v * v).collect();
        let b2: Vec<f64> = bv.iter().map(|v| v * v).collect();
        let ab: Vec<f64> = (0..n).map(|i| av[i] * bv[i]).collect();
        let r = 3;
        let ua = ref_box_mean(&av, w, h, r);
        let ub = ref_box_mean(&bv, w, h, r);
        let ua2 = ref_box_mean(&a2, w, h, r);
        let ub2 = ref_box_mean(&b2, w, h, r);
        let uab = ref_box_mean(&ab, w, h, r);
        let c1 = (0.01f64 * 255.0).powi(2);
        let c2 = (0.03f64 * 255.0).powi(2);
        let mut sum = 0.0;
        let pad = r.min(w / 2).min(h / 2);
        let mut cnt = 0usize;
        for y in pad..h - pad {
            for x in pad..w - pad {
                let i = y * w + x;
                let mux = ua[i];
                let muy = ub[i];
                let vx = (ua2[i] - mux * mux).max(0.0);
                let vy = (ub2[i] - muy * muy).max(0.0);
                let cxy = uab[i] - mux * muy;
                let num = (2.0 * mux * muy + c1) * (2.0 * cxy + c2);
                let den = (mux * mux + muy * muy + c1) * (vx + vy + c2);
                sum += num / den;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return 0.0;
        }
        (sum / cnt as f64).clamp(0.0, 1.0)
    }

    fn ref_ncc(a: &GrayImage, b: &GrayImage) -> f64 {
        let w = a.width.min(b.width) as usize;
        let h = a.height.min(b.height) as usize;
        if w == 0 || h == 0 {
            return 0.0;
        }
        let aw = a.width as usize;
        let bw = b.width as usize;
        let n = (w * h) as f64;
        let mut sa = 0.0;
        let mut sb = 0.0;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        let mut sab = 0.0;
        for y in 0..h {
            for x in 0..w {
                let va = a.data[y * aw + x] as f64;
                let vb = b.data[y * bw + x] as f64;
                sa += va;
                sb += vb;
                saa += va * va;
                sbb += vb * vb;
                sab += va * vb;
            }
        }
        let num = n * sab - sa * sb;
        let den = ((n * saa - sa * sa) * (n * sbb - sb * sb)).sqrt();
        if den <= 0.0 {
            return 0.0;
        }
        (num / den).clamp(-1.0, 1.0).max(0.0)
    }

    fn ref_template_match(img: &GrayImage, tpl: &GrayImage) -> (f64, u32, u32) {
        let (iw, ih) = (img.width as usize, img.height as usize);
        let (tw, th) = (tpl.width as usize, tpl.height as usize);
        if tw == 0 || th == 0 || iw < tw || ih < th {
            return (0.0, 0, 0);
        }
        let tn = (tw * th) as f64;
        let mut tsum = 0.0;
        let mut tss = 0.0;
        for v in &tpl.data {
            let f = *v as f64;
            tsum += f;
            tss += f * f;
        }
        let tmean = tsum / tn;
        let tvar = tss - tsum * tmean;
        if tvar <= 0.0 {
            return (0.0, 0, 0);
        }
        let mut integ = vec![0f64; (iw + 1) * (ih + 1)];
        let mut integ2 = vec![0f64; (iw + 1) * (ih + 1)];
        for y in 0..ih {
            let mut rs = 0.0;
            let mut rs2 = 0.0;
            for x in 0..iw {
                let v = img.data[y * iw + x] as f64;
                rs += v;
                rs2 += v * v;
                integ[(y + 1) * (iw + 1) + x + 1] = integ[y * (iw + 1) + x + 1] + rs;
                integ2[(y + 1) * (iw + 1) + x + 1] = integ2[y * (iw + 1) + x + 1] + rs2;
            }
        }
        let mut best = f64::MIN;
        let (mut bx, mut by) = (0u32, 0u32);
        for y in 0..=(ih - th) {
            for x in 0..=(iw - tw) {
                let x2 = x + tw;
                let y2 = y + th;
                let wsum = integ[y2 * (iw + 1) + x2] - integ[y * (iw + 1) + x2]
                    - integ[y2 * (iw + 1) + x] + integ[y * (iw + 1) + x];
                let wss = integ2[y2 * (iw + 1) + x2] - integ2[y * (iw + 1) + x2]
                    - integ2[y2 * (iw + 1) + x] + integ2[y * (iw + 1) + x];
                let wmean = wsum / tn;
                let wvar = wss - wsum * wmean;
                if wvar <= 0.0 {
                    continue;
                }
                let mut xterm = 0.0;
                for ty in 0..th {
                    let row = (y + ty) * iw + x;
                    let trow = ty * tw;
                    for tx in 0..tw {
                        xterm += img.data[row + tx] as f64 * tpl.data[trow + tx] as f64;
                    }
                }
                let cov = xterm - tn * wmean * tmean;
                let score = cov / (wvar * tvar).sqrt();
                if score > best {
                    best = score;
                    bx = x as u32;
                    by = y as u32;
                }
            }
        }
        (best.max(0.0), bx, by)
    }

    // ---- equivalence tests ----

    #[test]
    fn ssim_matches_reference() {
        let a = photo(160, 120, 3);
        let b = photo(160, 120, 9);
        assert!((ssim(&a, &b) - ref_ssim(&a, &b)).abs() <= 1e-9);
        assert!((ssim(&a, &a) - ref_ssim(&a, &a)).abs() <= 1e-9);
        assert_eq!(ssim(&a, &a), 1.0);
        // small / degenerate
        let tiny = photo(8, 8, 1);
        assert_eq!(ssim(&tiny, &tiny), ref_ssim(&tiny, &tiny));
    }

    #[test]
    fn ncc_matches_reference() {
        let a = photo(160, 120, 3);
        let b = photo(160, 120, 9);
        assert_eq!(ncc(&a, &b), ref_ncc(&a, &b)); // integer-exact sums ⇒ bit-identical
        assert_eq!(ncc(&a, &a), ref_ncc(&a, &a));
    }

    #[test]
    fn template_matches_reference() {
        let img = photo(200, 200, 7);
        // exact subimage → pyramid path (search 129×145 > 64)
        let tpl = crop(&img, 60, 45, 72, 56);
        let (s, x, y) = template_match(&img, &tpl);
        let (rs, rx, ry) = ref_template_match(&img, &tpl);
        assert!((s - rs).abs() <= 1e-9, "score {s} vs {rs}");
        assert_eq!((x, y), (rx, ry));
        assert!((s - 1.0).abs() < 1e-6);
        // subimage touching the far search edge
        let tpl2 = crop(&img, 140, 150, 60, 50);
        assert_eq!(template_match(&img, &tpl2), ref_template_match(&img, &tpl2));
        // small search → direct scan (identical)
        let small = crop(&img, 0, 0, 100, 90);
        let stpl = crop(&img, 33, 27, 50, 40);
        assert_eq!(
            template_match(&small, &stpl),
            ref_template_match(&small, &stpl)
        );
        // uniform template → (0,0,0)
        let flat = GrayImage::new(40, 40, vec![128u8; 1600]);
        assert_eq!(template_match(&img, &flat), ref_template_match(&img, &flat));
        // tpl == img, tpl > img
        assert_eq!(template_match(&img, &img), ref_template_match(&img, &img));
        let big = photo(240, 240, 2);
        assert_eq!(template_match(&img, &big), ref_template_match(&img, &big));
    }

    #[test]
    fn template_absent_template_still_scans_exact() {
        // A template that is not present: the coarse pyramid match is weak,
        // so the fallback must reproduce the reference result bit for bit.
        let img = photo(200, 200, 7);
        let tpl = photo(64, 64, 200);
        let (s, x, y) = template_match(&img, &tpl);
        let (rs, rx, ry) = ref_template_match(&img, &tpl);
        // scores can never exceed the true max; equal when fallback engaged
        assert!(s <= rs + 1e-9);
        if s > 0.0 {
            assert!((s - rs).abs() <= 1e-9, "score {s} vs {rs}");
            assert_eq!((x, y), (rx, ry));
        }
    }

    /// Manual perf smoke: `cargo test --release -p itrace-core perf_smoke -- --ignored --nocapture`
    #[test]
    #[ignore = "timing only; run manually"]
    fn perf_smoke() {
        let img = photo(480, 360, 5);
        let tpl = crop(&img, 200, 120, 96, 96); // present → pyramid path
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        for _ in 0..10 {
            acc += template_match(&img, &tpl).0;
        }
        eprintln!("template_match 480x360/96x96 (match):   {:>7.2} ms/op", t.elapsed().as_secs_f64() * 100.0);
        let absent = photo(96, 96, 99); // absent → fallback path
        let t = std::time::Instant::now();
        for _ in 0..3 {
            acc += template_match(&img, &absent).0;
        }
        eprintln!("template_match 480x360/96x96 (fallback): {:>7.2} ms/op", t.elapsed().as_secs_f64() * 1000.0 / 3.0);
        let a = photo(480, 360, 5);
        let b = photo(480, 360, 6);
        let t = std::time::Instant::now();
        for _ in 0..50 {
            acc += ssim(&a, &b);
        }
        eprintln!("ssim 480x360:                          {:>7.2} ms/op", t.elapsed().as_secs_f64() * 20.0);
        assert!(acc.is_finite());
    }
}
