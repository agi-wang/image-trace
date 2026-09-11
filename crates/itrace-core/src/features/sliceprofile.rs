//! Slice-boundary profile feature (`sliceprofile`).
//!
//! Per-row and per-column intensity/gradient projections, `BINS` bins each,
//! stored as 4*BINS raw bytes:
//!   [0..32)    row_mean — mean intensity per row band
//!   [32..64)   row_grad — mean |d/dy| per row band (x4, clamped)
//!   [64..96)   col_mean — mean intensity per column band
//!   [96..128)  col_grad — mean |d/dx| per column band (x4, clamped)
//!
//! similarity = max of
//!   * orientation-invariant Pearson cosine over the four signals under all
//!     8 dihedral transforms of B (axis swap + per-axis reversal covers
//!     rot90/180/270 and flips), and
//!   * slice score: best contiguous-window cross-correlation of the child's
//!     row (or column) profile inside the parent's, gated by the
//!     cross-axis profile agreement — detects "A is a contiguous
//!     horizontal/vertical band of B" in any orientation, either direction.

use super::{FeatureExtractor, FeatureKind};
use crate::{image_io, GrayImage, RgbImage};

const BINS: usize = 32;
const WORK_SIDE: u32 = 64;
const GRAD_SCALE: f64 = 4.0;
const EPS: f64 = 1e-6;

/// Decoded payload: four 32-bin signals.
#[derive(Clone)]
struct Profile {
    row_mean: [f64; BINS],
    row_grad: [f64; BINS],
    col_mean: [f64; BINS],
    col_grad: [f64; BINS],
}

/// Resample a dense signal to `n` bins (linear interpolation), writing
/// `out[..n]` — a stack buffer instead of a Vec per window.
fn resample(vals: &[f64], out: &mut [f64], n: usize) {
    let m = vals.len();
    if m == 0 {
        out[..n].fill(0.0);
        return;
    }
    if m == n {
        out[..n].copy_from_slice(vals);
        return;
    }
    for (i, o) in out.iter_mut().enumerate().take(n) {
        let pos = (i as f64 + 0.5) * m as f64 / n as f64 - 0.5;
        *o = if pos <= 0.0 {
            vals[0]
        } else if pos >= (m - 1) as f64 {
            vals[m - 1]
        } else {
            let lo = pos.floor() as usize;
            let t = pos - lo as f64;
            vals[lo] * (1.0 - t) + vals[lo + 1] * t
        };
    }
}

/// Signal mean — the same forward-order sum `pcc` computes, hoisted so each
/// signal is summed once per transform instead of once per call site.
fn mean(s: &[f64]) -> f64 {
    s.iter().sum::<f64>() / s.len() as f64
}

/// The four signal means of a profile, for `pcc_with_means`.
struct Means {
    row_mean: f64,
    row_grad: f64,
    col_mean: f64,
    col_grad: f64,
}

fn profile_means(p: &Profile) -> Means {
    Means {
        row_mean: mean(&p.row_mean),
        row_grad: mean(&p.row_grad),
        col_mean: mean(&p.col_mean),
        col_grad: mean(&p.col_grad),
    }
}

/// Pearson correlation between two signals with precomputed means, in [0,1].
/// Centering removes DC so the score tracks profile *shape*.
fn pcc_with_means(a: &[f64], ma: f64, b: &[f64], mb: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        dot += da * db;
        na += da * da;
        nb += db * db;
    }
    if na <= EPS || nb <= EPS {
        // Both flat ⇒ profiles agree (no structure to contradict); one flat ⇒ no match.
        return if na <= EPS && nb <= EPS { 1.0 } else { 0.0 };
    }
    (dot / (na * nb).sqrt()).clamp(0.0, 1.0)
}

/// Reversed copy of a signal.
fn rev(s: &[f64; BINS]) -> [f64; BINS] {
    let mut out = [0.0; BINS];
    for i in 0..BINS {
        out[i] = s[BINS - 1 - i];
    }
    out
}

/// Apply one dihedral transform to a profile: optional axis swap, then
/// per-axis reversal (row reversal applies to the presented row signals).
fn transform(p: &Profile, swap: bool, rev_r: bool, rev_c: bool) -> Profile {
    let (rm, rg, cm, cg) = if swap {
        (p.col_mean, p.col_grad, p.row_mean, p.row_grad)
    } else {
        (p.row_mean, p.row_grad, p.col_mean, p.col_grad)
    };
    Profile {
        row_mean: if rev_r { rev(&rm) } else { rm },
        row_grad: if rev_r { rev(&rg) } else { rg },
        col_mean: if rev_c { rev(&cm) } else { cm },
        col_grad: if rev_c { rev(&cg) } else { cg },
    }
}

/// Equal-weight correlation over the four signal pairs.
fn profile_cosine(a: &Profile, am: &Means, b: &Profile, bm: &Means) -> f64 {
    (pcc_with_means(&a.row_mean, am.row_mean, &b.row_mean, bm.row_mean)
        + pcc_with_means(&a.row_grad, am.row_grad, &b.row_grad, bm.row_grad)
        + pcc_with_means(&a.col_mean, am.col_mean, &b.col_mean, bm.col_mean)
        + pcc_with_means(&a.col_grad, am.col_grad, &b.col_grad, bm.col_grad))
        / 4.0
}

/// Best correlation of `child` against any contiguous window of `parent`
/// — the cross-correlation that detects a slice. Both sides are compared
/// at the window's native resolution (child downsampled to `len` bins), so
/// thin slices aren't penalized for keeping finer detail than the parent.
/// The child's mean/norm are hoisted per `len` — identical `pcc` values.
fn band_match(child: &[f64; BINS], parent: &[f64; BINS]) -> f64 {
    let mut best = 0.0f64;
    let mut cds = [0.0f64; BINS];
    for len in 4..=BINS {
        resample(child, &mut cds, len);
        let ca = &cds[..len];
        let ma = mean(ca);
        let na: f64 = ca.iter().map(|v| (v - ma) * (v - ma)).sum();
        for start in 0..=(BINS - len) {
            let wb = &parent[start..start + len];
            let mb = mean(wb);
            let (mut dot, mut nb) = (0.0, 0.0);
            for i in 0..len {
                let db = wb[i] - mb;
                dot += (ca[i] - ma) * db;
                nb += db * db;
            }
            let s = if na <= EPS || nb <= EPS {
                if na <= EPS && nb <= EPS { 1.0 } else { 0.0 }
            } else {
                (dot / (na * nb).sqrt()).clamp(0.0, 1.0)
            };
            best = best.max(s);
            if best >= 1.0 {
                return 1.0; // max possible — later windows can't change it
            }
        }
    }
    best
}

/// Score "child is a contiguous band of parent": band correlation on one
/// axis corroborated by direct correlation on the other axis.
fn slice_score(child: &Profile, cm: &Means, parent: &Profile, pm: &Means) -> f64 {
    let horizontal = (band_match(&child.row_mean, &parent.row_mean)
        + band_match(&child.row_grad, &parent.row_grad)
        + pcc_with_means(&child.col_mean, cm.col_mean, &parent.col_mean, pm.col_mean)
        + pcc_with_means(&child.col_grad, cm.col_grad, &parent.col_grad, pm.col_grad))
        / 4.0;
    let vertical = (band_match(&child.col_mean, &parent.col_mean)
        + band_match(&child.col_grad, &parent.col_grad)
        + pcc_with_means(&child.row_mean, cm.row_mean, &parent.row_mean, pm.row_mean)
        + pcc_with_means(&child.row_grad, cm.row_grad, &parent.row_grad, pm.row_grad))
        / 4.0;
    horizontal.max(vertical)
}

fn decode(data: &[u8]) -> Option<Profile> {
    if data.len() != 4 * BINS {
        return None;
    }
    let read = |s: &[u8]| {
        let mut out = [0.0; BINS];
        for (i, &v) in s.iter().enumerate() {
            out[i] = v as f64;
        }
        out
    };
    Some(Profile {
        row_mean: read(&data[0..BINS]),
        row_grad: read(&data[BINS..2 * BINS]),
        col_mean: read(&data[2 * BINS..3 * BINS]),
        col_grad: read(&data[3 * BINS..4 * BINS]),
    })
}

/// Slice-boundary profile extractor.
pub struct SliceProfileExtractor;

impl FeatureExtractor for SliceProfileExtractor {
    fn feature_name(&self) -> &'static str {
        "slice_profile"
    }
    fn algorithms(&self) -> &'static [&'static str] {
        &["sliceprofile"]
    }
    fn kind(&self) -> FeatureKind {
        FeatureKind::Cosine
    }

    fn compute(&self, gray: &GrayImage, _rgb: &RgbImage) -> Vec<u8> {
        let g = image_io::resize_gray_max(gray, WORK_SIDE);
        let (w, h) = (g.width as usize, g.height as usize);

        // Dense per-line signals, then resampled to BINS — handles any size.
        let mut row_m = vec![0.0f64; h];
        let mut row_g = vec![0.0f64; h];
        let mut col_m = vec![0.0f64; w];
        let mut col_g = vec![0.0f64; w];
        for y in 0..h {
            // centered difference (forward/backward at the borders) so the
            // signal reverses and transposes exactly under the dihedral group
            let (y0, y1, half) = if h < 2 {
                (0, 0, 1.0)
            } else if y == 0 {
                (0, 1, 1.0)
            } else if y + 1 == h {
                (h - 2, h - 1, 1.0)
            } else {
                (y - 1, y + 1, 2.0)
            };
            let mut s = 0.0;
            let mut gs = 0.0;
            for x in 0..w {
                let v = g.data[y * w + x] as f64;
                s += v;
                gs += (g.data[y1 * w + x] as f64 - g.data[y0 * w + x] as f64).abs() / half;
            }
            row_m[y] = s / w as f64;
            row_g[y] = gs / w as f64;
        }
        for x in 0..w {
            let (x0, x1, half) = if w < 2 {
                (0, 0, 1.0)
            } else if x == 0 {
                (0, 1, 1.0)
            } else if x + 1 == w {
                (w - 2, w - 1, 1.0)
            } else {
                (x - 1, x + 1, 2.0)
            };
            let mut s = 0.0;
            let mut gs = 0.0;
            for y in 0..h {
                let v = g.data[y * w + x] as f64;
                s += v;
                gs += (g.data[y * w + x1] as f64 - g.data[y * w + x0] as f64).abs() / half;
            }
            col_m[x] = s / h as f64;
            col_g[x] = gs / h as f64;
        }

        let mut out = Vec::with_capacity(4 * BINS);
        let mut buf = [0.0f64; BINS];
        let mut push = |vals: &[f64], scale: f64, out: &mut Vec<u8>| {
            resample(vals, &mut buf, BINS);
            for v in buf {
                out.push((v * scale).round().clamp(0.0, 255.0) as u8);
            }
        };
        push(&row_m, 1.0, &mut out);
        push(&row_g, GRAD_SCALE, &mut out);
        push(&col_m, 1.0, &mut out);
        push(&col_g, GRAD_SCALE, &mut out);
        out
    }

    fn dims(&self, data: &[u8]) -> usize {
        data.len()
    }

    fn similarity(&self, a: &[u8], b: &[u8]) -> f64 {
        let (pa, pb) = match (decode(a), decode(b)) {
            (Some(pa), Some(pb)) => (pa, pb),
            _ => {
                let fa: Vec<f32> = a.iter().map(|&v| v as f32).collect();
                let fb: Vec<f32> = b.iter().map(|&v| v as f32).collect();
                return super::cosine(&fa, &fb);
            }
        };
        let ma = profile_means(&pa);
        let mut best = 0.0f64;
        for swap in [false, true] {
            for rev_r in [false, true] {
                for rev_c in [false, true] {
                    let tb = transform(&pb, swap, rev_r, rev_c);
                    let mb = profile_means(&tb);
                    best = best.max(profile_cosine(&pa, &ma, &tb, &mb));
                    if !swap {
                        // Axis swap is already covered by slice_score's
                        // horizontal/vertical symmetry — only scan the
                        // four reversal transforms.
                        best = best.max(slice_score(&pa, &ma, &tb, &mb));
                        best = best.max(slice_score(&tb, &mb, &pa, &ma));
                    }
                    if best >= 1.0 {
                        // every term is clamped ≤ 1 — the max can't move
                        return 1.0;
                    }
                }
            }
        }
        best.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;
    use image::{DynamicImage, GrayImage as ILuma, ImageBuffer, Luma};

    /// Deterministic textured image: smooth sine structure (distinctive
    /// row/col profiles) + pseudo-noise (gradient energy).
    fn textured(f1: f64, f2: f64, f3: f64, noise: f64, w: u32, h: u32) -> DynamicImage {
        let mut buf: ILuma = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let n = ((x.wrapping_mul(31) ^ y.wrapping_mul(17) ^ x.wrapping_mul(y)) % 64) as f64;
                let v = 120.0
                    + 70.0 * (x as f64 * f1).sin() * (y as f64 * f2).cos()
                    + 45.0 * ((x as f64 + 2.0 * y as f64) * f3).sin()
                    + (n - 32.0) * noise;
                buf.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
            }
        }
        DynamicImage::ImageLuma8(buf)
    }

    fn ext() -> SliceProfileExtractor {
        SliceProfileExtractor
    }

    fn sim(e: &SliceProfileExtractor, a: &DynamicImage, b: &DynamicImage) -> f64 {
        let (ga, ra) = (image_io::to_gray(a), image_io::to_rgb(a));
        let (gb, rb) = (image_io::to_gray(b), image_io::to_rgb(b));
        e.similarity(&e.compute(&ga, &ra), &e.compute(&gb, &rb))
    }

    fn img_a() -> DynamicImage {
        textured(0.31, 0.17, 0.23, 1.0, 128, 96)
    }

    /// Structurally different image: blocky checkerboard + different
    /// frequencies, so its row/col profiles share no shape with `img_a`.
    fn img_b() -> DynamicImage {
        let (w, h) = (128u32, 96u32);
        let mut buf: ILuma = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let n = ((x.wrapping_mul(53) ^ y.wrapping_mul(29) ^ x.wrapping_mul(3).wrapping_mul(y)) % 64) as f64;
                let checker = (((x / 13) + (y / 11)) % 2) as f64 * 90.0;
                let v = 60.0 + checker + 40.0 * (x as f64 * 0.07 + y as f64 * 0.41).sin() + (n - 32.0);
                buf.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
            }
        }
        DynamicImage::ImageLuma8(buf)
    }

    /// Crop `keep` rows (or cols) starting at `off` — a contiguous slice.
    fn crop_slice(img: &DynamicImage, off: u32, keep: u32, vertical: bool) -> DynamicImage {
        let g = image_io::to_gray(img);
        let (w, h) = (g.width, g.height);
        let (nw, nh) = if vertical { (keep, h) } else { (w, keep) };
        let mut data = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let (sx, sy) = if vertical { (x + off, y) } else { (x, y + off) };
                data.push(g.get(sx, sy));
            }
        }
        let buf: ILuma = ImageBuffer::from_raw(nw, nh, data).unwrap();
        DynamicImage::ImageLuma8(buf)
    }

    #[test]
    fn payload_layout() {
        let e = ext();
        let a = img_a();
        let (g, r) = (image_io::to_gray(&a), image_io::to_rgb(&a));
        let d = e.compute(&g, &r);
        assert_eq!(d.len(), 128);
        assert_eq!(e.dims(&d), 128);
    }

    #[test]
    fn identical_is_one() {
        let e = ext();
        let s = sim(&e, &img_a(), &img_a());
        assert!(s > 0.99, "self-sim {s}");
    }

    #[test]
    fn orientation_variants_stay_high() {
        let e = ext();
        let a = img_a();
        for (name, v) in [
            ("rot90", a.rotate90()),
            ("rot180", a.rotate180()),
            ("rot270", a.rotate270()),
            ("fliph", a.fliph()),
            ("flipv", a.flipv()),
        ] {
            let s = sim(&e, &a, &v);
            assert!(s >= 0.80, "{name} sim {s}");
            eprintln!("{name}: {s:.3}");
        }
    }

    #[test]
    fn different_image_is_low() {
        let e = ext();
        let s = sim(&e, &img_a(), &img_b());
        assert!(s < 0.70, "different-image sim {s}");
        eprintln!("different: {s:.3}");
    }

    #[test]
    fn contiguous_slice_scores_high() {
        let e = ext();
        let a = img_a();
        // top quarter (horizontal band) and left quarter (vertical band)
        for (name, sl) in [
            ("top-1/4", crop_slice(&a, 0, 24, false)),
            ("mid-1/4", crop_slice(&a, 36, 24, false)),
            ("left-1/4", crop_slice(&a, 0, 32, true)),
        ] {
            let s = sim(&e, &sl, &a);
            assert!(s >= 0.75, "{name} slice sim {s}");
            eprintln!("{name}: {s:.3}");
        }
        // a slice of a different image must not match
        let foreign = crop_slice(&img_b(), 0, 24, false);
        let s = sim(&e, &foreign, &a);
        assert!(s < 0.70, "foreign-slice sim {s}");
        eprintln!("foreign slice: {s:.3}");
    }
}
