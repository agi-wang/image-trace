//! Perf harness on synthetic images (no external assets needed).
//! Usage: cargo run --release -p itrace-core --example perf [--quick]

use std::time::Instant;

use image::{Rgb, RgbImage};

use itrace_core::compare::{self, Prepared};
use itrace_core::descriptors::{extractor_for, match_score};
use itrace_core::{features, hashes, image_io, index, metrics, slice};

fn make_photo(w: u32, h: u32, seed: u8) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = ((x ^ y) as u8).wrapping_add(seed);
            img.put_pixel(x, y, Rgb([r, g, b]));
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

fn time<F: FnMut()>(name: &str, n: usize, mut f: F) {
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{name:<42} {ms:>9.1} ms total  {:>8.2} ms/op", ms / n as f64);
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let (n_img, w, h) = if quick { (12, 480, 360) } else { (24, 800, 600) };

    // ---- corpus ----
    let imgs: Vec<RgbImage> = (0..n_img).map(|s| make_photo(w, h, s as u8)).collect();
    let pngs: Vec<Vec<u8>> = imgs.iter().map(png_bytes).collect();

    // `--emit-corpus <dir>`: write the corpus PNGs and exit — used by
    // scripts/perf_e2e.sh to feed the server a deterministic workload.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--emit-corpus") {
        let dir = args.get(pos + 1).expect("--emit-corpus <dir>");
        std::fs::create_dir_all(dir).unwrap();
        for (i, b) in pngs.iter().enumerate() {
            std::fs::write(format!("{dir}/img_{i:03}.png"), b).unwrap();
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
    println!("corpus: {n_img} images {w}x{h} (prepared at <=512)\n");

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
    let gb = &grays[1];
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
    let patch = image_io::resize_gray_exact(ga, 48, 48);
    time("metrics::template_match", 10, || {
        let _ = metrics::template_match(gb, &patch);
    });

    // ---- descriptors ----
    let ex = extractor_for("orb").unwrap();
    time("orb detect", n_img, || {
        for g in &grays {
            let _ = ex.detect(g, 512);
        }
    });
    let da = ex.detect(ga, 512);
    let db = ex.detect(gb, 512);
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

    // ---- slice match ----
    time("slice::slice_match 2x2", 10, || {
        let _ = slice::slice_match(ga, gb, 2, 2, 0.6);
    });

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
