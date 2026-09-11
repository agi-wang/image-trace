//! Perf harness on synthetic images (no external assets needed).
//! Usage: cargo run --release -p itrace-core --example perf [--quick]

use std::time::Instant;

use image::{Rgb, RgbImage};

use itrace_core::compare::{self, Prepared};
use itrace_core::descriptors::{extractor_for, match_score};
use itrace_core::{features, hashes, image_io, index, metrics, slice};

fn make_photo(w: u32, h: u32, seed: u8) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    // Four genuinely different base structures — without this every image
    // shares one gradient skeleton and perceptual hashes rightly call
    // them near-duplicates (cross-family false positives).
    let mode = seed % 4;
    for y in 0..h {
        for x in 0..w {
            let px = match mode {
                0 => Rgb([
                    ((x * 255) / w) as u8,
                    ((y * 255) / h) as u8,
                    ((x ^ y) as u8).wrapping_add(seed),
                ]),
                1 => Rgb([
                    ((y * 255) / h) as u8,
                    (((w - x) * 255) / w) as u8,
                    ((x + y * 3) as u8).wrapping_mul(3).wrapping_add(seed),
                ]),
                2 => {
                    let ck = ((((x / 32) ^ (y / 32)) & 1) * 200) as u8;
                    Rgb([
                        ck.wrapping_add(seed),
                        (((x * 7) ^ y) % 256) as u8,
                        (((y * 5) + x) % 256) as u8,
                    ])
                }
                _ => {
                    // noise-dominant field (low-frequency structure is weak)
                    let n = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)
                        ^ (seed as u32).wrapping_mul(2246822519)) as u8;
                    Rgb([n, n.wrapping_add(37), (x as u8).wrapping_sub(n)])
                }
            };
            img.put_pixel(x, y, px);
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
                    let cur = img.get_pixel(px, py)[0];
                    img.put_pixel(px, py, Rgb([255 - cur, 200, (k >> 16) as u8]));
                }
            }
        }
    }
    img
}

fn png_bytes(img: &RgbImage) -> Vec<u8> {
    let mut cur = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img.clone())
        .write_to(&mut cur, image::ImageFormat::Png)
        .unwrap();
    cur.into_inner()
}

fn jpg_bytes(img: &RgbImage, quality: u8) -> Vec<u8> {
    let mut cur = std::io::Cursor::new(Vec::new());
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cur, quality);
    enc.encode_image(img).unwrap();
    cur.into_inner()
}

/// One near-duplicate family around `base`: the images a real duplicate
/// pipeline must group together. Mixes PNG and JPEG so both decode paths
/// and the positive (hit) scoring paths get exercised.
/// Returns (suffix, encoded bytes) pairs — `orig` first.
fn dup_family(base: &RgbImage) -> Vec<(&'static str, Vec<u8>)> {
    use image::imageops;
    let dyn_img = || image::DynamicImage::ImageRgb8(base.clone());
    let (w, h) = base.dimensions();
    let mut out = vec![("orig", png_bytes(base))];
    // orientation variants (exact dihedral transforms)
    out.push(("rot90", png_bytes(&dyn_img().rotate90().to_rgb8())));
    out.push(("fliph", png_bytes(&dyn_img().fliph().to_rgb8())));
    // half-size — scale invariance + sub-512 path
    out.push(("half", png_bytes(&imageops::resize(
        base, w / 2, h / 2, imageops::FilterType::Triangle,
    ))));
    // center crop 80% — containment/slice path
    let (cw, ch) = (w * 4 / 5, h * 4 / 5);
    out.push(("crop", png_bytes(&imageops::crop_imm(
        base, (w - cw) / 2, (h - ch) / 2, cw, ch,
    ).to_image())));
    // +30 brightness — photometric near-dup
    out.push(("bright", png_bytes(&imageops::brighten(base, 30))));
    // JPEG recompress — codec artifacts + the JPEG decode path
    out.push(("jpg80", jpg_bytes(base, 80)));
    out
}

fn time<F: FnMut()>(name: &str, n: usize, mut f: F) {
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{name:<42} {ms:>9.1} ms total  {:>8.2} ms/op", ms / n as f64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let n_arg = args
        .iter()
        .position(|a| a == "--n")
        .and_then(|p| args.get(p + 1))
        .and_then(|s| s.parse::<usize>().ok());
    // n_base = distinct base photos; each expands into a near-duplicate
    // family (rot90/fliph/half/crop/bright/jpg80) — the corpus therefore
    // contains real positive pairs, not just misses.
    let (n_base, w, h) = match (quick, n_arg) {
        (true, _) => (4, 480, 360),
        (false, Some(k)) => (k, 800, 600),
        (false, None) => (24, 800, 600),
    };

    // ---- corpus ----
    let bases: Vec<RgbImage> = (0..n_base).map(|s| make_photo(w, h, s as u8)).collect();
    let mut names: Vec<String> = Vec::new();
    let mut pngs: Vec<Vec<u8>> = Vec::new();
    for (i, b) in bases.iter().enumerate() {
        for (suffix, bytes) in dup_family(b) {
            names.push(format!("fam{i:03}_{suffix}"));
            pngs.push(bytes);
        }
    }
    let n_img = pngs.len();

    // `--emit-corpus <dir>`: write the corpus files and exit — used by
    // scripts/perf_e2e.sh to feed the server a deterministic workload
    // (including real duplicate families).
    if let Some(pos) = args.iter().position(|a| a == "--emit-corpus") {
        let dir = args.get(pos + 1).expect("--emit-corpus <dir>");
        std::fs::create_dir_all(dir).unwrap();
        for (name, b) in names.iter().zip(&pngs) {
            let ext = if name.ends_with("jpg80") { "jpg" } else { "png" };
            std::fs::write(format!("{dir}/{name}.{ext}"), b).unwrap();
        }
        println!("wrote {} images to {dir}", pngs.len());
        return;
    }

    let grays: Vec<_> = pngs
        .iter()
        .map(|b| {
            let img = image_io::decode(b).unwrap();
            image_io::to_gray(&image_io::resize_max_side(&img, 512))
        })
        .collect();
    let rgbs: Vec<_> = pngs
        .iter()
        .map(|b| {
            let img = image_io::decode(b).unwrap();
            image_io::to_rgb(&image_io::resize_max_side(&img, 512))
        })
        .collect();
    println!(
        "corpus: {n_base} base photos {w}x{h} -> {n_img} images \
         ({}-image dup families, prepared at <=512)\n",
        n_img / n_base
    );

    // ---- decode + prepare (upload path) ----
    time("Prepared::from_bytes rot_inv=false", 3, || {
        for b in &pngs[..4.min(n_img)] {
            let _ = Prepared::from_bytes(b, &["orb".to_string()], false).unwrap();
        }
    });
    time("Prepared::from_bytes rot_inv=true", 2, || {
        for b in &pngs[..4.min(n_img)] {
            let _ = Prepared::from_bytes(b, &["orb".to_string()], true).unwrap();
        }
    });
    time("hashes::compute_all", n_img, || {
        for (g, r) in grays.iter().zip(rgbs.iter()) {
            let _ = hashes::compute_all(g, r);
        }
    });
    time("metrics::hsv_histogram", n_img, || {
        for r in &rgbs {
            let _ = metrics::hsv_histogram(r);
        }
    });

    // ---- feature extractors (precompute path) ----
    for ext in features::EXTRACTORS {
        let name = ext.feature_name();
        time(&format!("extractor {name}"), n_img, || {
            for (g, r) in grays.iter().zip(rgbs.iter()) {
                let _ = ext.compute(g, r);
            }
        });
    }

    // ---- per-pair metrics ----
    let pairs: Vec<(usize, usize)> = (0..n_img)
        .flat_map(|i| (i + 1..n_img).map(move |j| (i, j)))
        .collect();
    let ga = &grays[0];
    time("metrics::ssim", pairs.len(), || {
        for &(i, j) in &pairs {
            let _ = metrics::ssim(&grays[i], &grays[j]);
        }
    });
    time("metrics::ncc", pairs.len(), || {
        for &(i, j) in &pairs {
            let _ = metrics::ncc(&grays[i], &grays[j]);
        }
    });
    time("metrics::histogram_correlation", pairs.len(), || {
        let ha = metrics::hsv_histogram(&rgbs[0]);
        for &(i, _) in &pairs {
            let _ = metrics::histogram_correlation(&ha, &metrics::hsv_histogram(&rgbs[i]));
        }
    });
    // positive/negative probes: grays[1] is fam000's rot90, grays[4] its
    // center-crop, grays[7] (or last) is a different family entirely.
    let gneg = &grays[(n_img - 1).max(7).min(n_img - 1)];
    let gcrop = &grays[4.min(n_img - 1)];
    let patch = image_io::resize_gray_exact(ga, 48, 48);
    time("metrics::template_match", 10, || {
        let _ = metrics::template_match(gneg, &patch);
    });

    // ---- descriptors ----
    let ex = extractor_for("orb").unwrap();
    time("orb detect", n_img, || {
        for g in &grays {
            let _ = ex.detect(g, 512);
        }
    });
    let da = ex.detect(ga, 512);
    let db = ex.detect(gneg, 512);
    time("orb match_score", 200, || {
        let _ = match_score(&da, &db, 64);
    });

    // ---- full pairwise ----
    let prepared: Vec<Prepared> = pngs
        .iter()
        .map(|b| Prepared::from_bytes(b, &["orb".to_string()], true).unwrap())
        .collect();
    for algo in ["phash", "ssim", "orb", "auto"] {
        let m = compare::pairwise_matrix(&prepared, algo, false);
        let t = Instant::now();
        let m2 = compare::pairwise_matrix(&prepared, algo, false);
        println!(
            "{:<42} {:>9.1} ms total",
            format!("pairwise_matrix {algo} (n={n_img})"),
            t.elapsed().as_secs_f64() * 1000.0
        );
        drop((m, m2));
    }

    // ---- slice match (negative pair + positive crop pair) ----
    time("slice::slice_match 2x2 (miss)", 10, || {
        let _ = slice::slice_match(ga, gneg, 2, 2, 0.6);
    });
    time("slice::slice_match 2x2 (hit crop)", 10, || {
        let _ = slice::slice_match(ga, gcrop, 2, 2, 0.6);
    });
    let (sc, contained) = slice::contains(ga, gcrop);
    println!("contains(orig, crop) = {sc:.3} contained={contained}\n");

    // ---- MIH dedup index ----
    let keys: Vec<u64> = (0..n_img).map(|i| hashes::phash(&grays[i])).collect();
    time("mih build+query (x1000 keys)", 10, || {
        let mut idx = index::MihIndex::new();
        for i in 0..1000u32 {
            idx.insert(keys[(i as usize) % keys.len()] ^ i as u64, i);
        }
        for i in 0..100u64 {
            let _ = idx.query(keys[(i as usize) % keys.len()] ^ i, 10);
        }
    });
}
