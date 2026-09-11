//! Tier 2 pixel/structure metrics: SSIM, HSV histogram correlation, NCC template.

use crate::{GrayImage, RgbImage};

/// Separable uniform (box) filter: each output = mean of the (2r+1) window
/// clamped to the image bounds (truncated windows near edges).
fn box_mean(data: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; w * h];
    let mut out = vec![0.0; w * h];
    // horizontal pass
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
    // vertical pass
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

/// Structural similarity index (uniform 7×7 window, K1=0.01 K2=0.03).
/// Both inputs must share dimensions.
pub fn ssim(a: &GrayImage, b: &GrayImage) -> f64 {
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

    let r = 3; // 7×7
    let ua = box_mean(&av, w, h, r);
    let ub = box_mean(&bv, w, h, r);
    let ua2 = box_mean(&a2, w, h, r);
    let ub2 = box_mean(&b2, w, h, r);
    let uab = box_mean(&ab, w, h, r);

    let c1 = (0.01f64 * 255.0).powi(2);
    let c2 = (0.03f64 * 255.0).powi(2);

    let mut sum = 0.0;
    // skip a half-window border like scikit-image's crop
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

/// HSV histogram: 50 hue × 60 sat bins, L2-normalized (OpenCV-compatible dims).
pub fn hsv_histogram(rgb: &RgbImage) -> Vec<f32> {
    const HBINS: usize = 50;
    const SBINS: usize = 60;
    let mut hist = vec![0f32; HBINS * SBINS];
    let n = (rgb.width as usize) * (rgb.height as usize);
    for i in 0..n {
        let r = rgb.data[i * 3] as f32 / 255.0;
        let g = rgb.data[i * 3 + 1] as f32 / 255.0;
        let b = rgb.data[i * 3 + 2] as f32 / 255.0;
        let (h, s, _v) = rgb_to_hsv_f32(r, g, b);
        // OpenCV hue range is [0,180)
        let hb = ((h / 180.0) * HBINS as f32).floor().clamp(0.0, HBINS as f32 - 1.0) as usize;
        let sb = (s * SBINS as f32).floor().clamp(0.0, SBINS as f32 - 1.0) as usize;
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
    let ma: f64 = a.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mb: f64 = b.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let x = a[i] as f64 - ma;
        let y = b[i] as f64 - mb;
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
pub fn ncc(a: &GrayImage, b: &GrayImage) -> f64 {
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

/// Template matching: slide `tpl` over `img`, return max NCC score and its (x, y).
/// Both grayscale; tpl must be <= img in each dim.
pub fn template_match(img: &GrayImage, tpl: &GrayImage) -> (f64, u32, u32) {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (tw, th) = (tpl.width as usize, tpl.height as usize);
    if tw == 0 || th == 0 || iw < tw || ih < th {
        return (0.0, 0, 0);
    }
    // template stats
    let tn = (tw * th) as f64;
    let mut tsum = 0.0;
    let mut tss = 0.0;
    for v in &tpl.data {
        let f = *v as f64;
        tsum += f;
        tss += f * f;
    }
    let tmean = tsum / tn;
    let tvar = tss - tsum * tmean; // sum((t-tm)^2)
    if tvar <= 0.0 {
        return (0.0, 0, 0);
    }

    // integral images of img and img^2
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
            // window stats from integral images
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
            // cross term: sum(img*t) — direct loop (template is small)
            let mut xterm = 0.0;
            for ty in 0..th {
                let row = (y + ty) * iw + x;
                let trow = ty * tw;
                for tx in 0..tw {
                    xterm += img.data[row + tx] as f64 * tpl.data[trow + tx] as f64;
                }
            }
            // cov sum: xterm - tn*wmean*tmean
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
