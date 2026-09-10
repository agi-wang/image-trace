//! End-to-end algorithm tests on synthetic images:
//! similarity, rotation-invariance, crop/slice detection, document sniffing.

use itrace_core::compare::{self, Prepared};
use itrace_core::descriptors::{extractor_for, match_cross_check, match_score};
use itrace_core::{documents, features, group, hashes, image_io, metrics, slice};
use image::{Rgb, RgbImage};

/// A textured test image: gradient background + distinctive shapes.
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
    // blob texture for hashing + keypoints
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

fn to_dyn(img: RgbImage) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(img)
}

fn rgba_dyn(img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> image::DynamicImage {
    image::DynamicImage::ImageRgba8(img)
}

fn png_bytes(img: &image::DynamicImage) -> Vec<u8> {
    let mut cur = std::io::Cursor::new(Vec::new());
    img.write_to(&mut cur, image::ImageFormat::Png).unwrap();
    cur.into_inner()
}

#[test]
fn identical_image_hashes_match() {
    let a = to_dyn(make_photo(128, 128, 1));
    let ga = image_io::to_gray(&a);
    let ra = image_io::to_rgb(&a);
    let h1 = hashes::compute_all(&ga, &ra);
    let h2 = hashes::compute_all(&ga, &ra);
    assert_eq!(h1.phash, h2.phash);
    assert_eq!(h1.dhash, h2.dhash);
    assert!(hashes::hash_similarity(h1.phash, h2.phash) >= 0.99);
}

#[test]
fn different_images_diverge() {
    let a = to_dyn(make_photo(128, 128, 1));
    let b = to_dyn(make_photo(128, 128, 200));
    let ha = hashes::compute_all(&image_io::to_gray(&a), &image_io::to_rgb(&a));
    let hb = hashes::compute_all(&image_io::to_gray(&b), &image_io::to_rgb(&b));
    assert!(hashes::hamming(ha.phash, hb.phash) >= 8);
}

#[test]
fn ssim_identical_is_one() {
    let a = to_dyn(make_photo(96, 96, 3));
    let g = image_io::to_gray(&a);
    let s = metrics::ssim(&g, &g);
    assert!((s - 1.0).abs() < 1e-6, "ssim identical = {s}");
}

#[test]
fn template_finds_subimage() {
    let a = to_dyn(make_photo(160, 160, 7));
    let ga = image_io::to_gray(&a);
    let patch = image::imageops::crop_imm(&a, 40, 50, 48, 48).to_image();
    let gp = image_io::to_gray(&rgba_dyn(patch));
    let (score, x, y) = metrics::template_match(&ga, &gp);
    assert!(score > 0.95, "template score {score}");
    assert!((x as i32 - 40).abs() <= 2 && (y as i32 - 50).abs() <= 2, "found at {x},{y}");
}

#[test]
fn orb_detects_keypoints() {
    let a = to_dyn(make_photo(160, 160, 11));
    let g = image_io::to_gray(&a);
    let ex = extractor_for("orb").unwrap();
    let d = ex.detect(&g, 500);
    assert!(d.keypoints.len() > 20, "only {} keypoints", d.keypoints.len());
    assert_eq!(d.desc_len, 32);
}

#[test]
fn orb_matches_across_rotation() {
    let a = to_dyn(make_photo(192, 192, 21));
    let rot = a.rotate90();
    let ga = image_io::to_gray(&a);
    let gb = image_io::to_gray(&rot);
    let ex = extractor_for("orb").unwrap();
    let da = ex.detect(&ga, 800);
    let db = ex.detect(&gb, 800);
    let matches = match_cross_check(&da, &db);
    let s = match_score(&da, &db, 64);
    assert!(s > 0.3, "rotation match score {s} ({} matches)", matches.len());
}

#[test]
fn rotated_pair_scores_high() {
    let a = to_dyn(make_photo(200, 200, 5));
    let rot = a.rotate180();
    let pa = Prepared::from_bytes(&png_bytes(&a), &["phash".into(), "orb".into()], true).unwrap();
    let pb = Prepared::from_bytes(&png_bytes(&rot), &["phash".into(), "orb".into()], true).unwrap();
    let s = compare::pair_score("phash", &pa, &pb, true);
    assert!(s > 0.8, "rotated phash {s}");
    let s = compare::pair_score("orb", &pa, &pb, true);
    assert!(s > 0.4, "rotated orb {s}");
}

#[test]
fn slice_detects_quarter() {
    let a = to_dyn(make_photo(200, 200, 9));
    let ga = image_io::to_gray(&a);
    let q = image::imageops::crop_imm(&a, 0, 0, 96, 96).to_image();
    let gq = image_io::to_gray(&rgba_dyn(q));
    let r = slice::slice_match(&ga, &gq, 2, 2, 0.5);
    assert!(r.is_slice_of_a, "coverage {:.2}", r.coverage);
    assert!(!r.cells.is_empty());
}

#[test]
fn slice_match_returns_cells() {
    let a = to_dyn(make_photo(180, 180, 13));
    let ga = image_io::to_gray(&a);
    let q = image::imageops::crop_imm(&a, 30, 30, 88, 88).to_image();
    let gq = image_io::to_gray(&rgba_dyn(q));
    let r = slice::slice_match(&ga, &gq, 2, 2, 0.5);
    assert!(r.coverage >= 0.5, "coverage {}", r.coverage);
    let best = r.cells.iter().max_by(|x, y| x.score.partial_cmp(&y.score).unwrap()).unwrap();
    assert!(
        (best.best_x.unwrap() as i32 - 30).abs() < 12 && (best.best_y.unwrap() as i32 - 30).abs() < 12,
        "best at {:?},{:?}", best.best_x, best.best_y
    );
}

#[test]
fn union_find_clusters() {
    let m = vec![
        vec![1.0, 0.9, 0.1, 0.1],
        vec![0.9, 1.0, 0.2, 0.1],
        vec![0.1, 0.2, 1.0, 0.95],
        vec![0.1, 0.1, 0.95, 1.0],
    ];
    let (groups, uniq) = group::cluster(&m, 0.85);
    assert_eq!(groups.len(), 2);
    assert!(uniq.is_empty());
    assert!(groups.iter().any(|g| g.contains(&0) && g.contains(&1)));
    assert!(groups.iter().any(|g| g.contains(&2) && g.contains(&3)));
}

#[test]
fn feature_pack_roundtrip() {
    let v = 0xDEADBEEF12345678u64;
    let b = features::pack_bits(v);
    assert_eq!(features::unpack_bits(&b), v);
}

#[test]
fn document_sniff() {
    assert_eq!(documents::sniff_extension(b"\x89PNG\r\n\x1a\nrest"), "png");
    assert_eq!(documents::sniff_extension(b"\xff\xd8\xff\xe0xxx"), "jpg");
    assert_eq!(documents::sniff_extension(b"GIF89axxxx"), "gif");
    // unknown bytes fall back to jpg (Office media without extension)
    assert_eq!(documents::sniff_extension(b"garbage!"), "jpg");
}

#[test]
fn orientation_variants_count() {
    let a = to_dyn(make_photo(64, 48, 3));
    let vars = image_io::orientation_variants(&a);
    assert_eq!(vars.len(), features::NUM_VARIANTS as usize);
    let g0 = image_io::to_gray(&a);
    let g2 = image_io::to_gray(&vars[2]);
    assert_ne!(g0.data, g2.data);
}

// ---------- index (billion-scale recall) ----------

#[test]
fn canonical_rot64_dihedral_invariant() {
    use itrace_core::index::canonical_rot64;
    let h = 0xDEADBEEF12345678u64;
    assert_eq!(canonical_rot64(h), canonical_rot64(h));
    // bit-matrix rotate must canon to the same value as the original
    let rot90 = |v: u64| -> u64 {
        let mut out = 0u64;
        for i in 0..8usize {
            for j in 0..8usize {
                if (v >> (i * 8 + j)) & 1 == 1 {
                    out |= 1u64 << (j * 8 + (7 - i));
                }
            }
        }
        out
    };
    let mut r = h;
    for _ in 0..4 {
        assert_eq!(canonical_rot64(r), canonical_rot64(h));
        r = rot90(r);
    }
}

#[test]
fn canonical_ahash_matches_rotated_image() {
    use itrace_core::index::canonical_rot64;
    let a = to_dyn(make_photo(64, 48, 3));
    let b = a.rotate90();
    let ga = image_io::to_gray(&a);
    let gb = image_io::to_gray(&b);
    // ahash is dihedral-equivariant: its 8×8 bit grid rotates with the image
    let ha = canonical_rot64(hashes::ahash(&ga));
    let hb = canonical_rot64(hashes::ahash(&gb));
    assert_eq!(ha, hb);
}

#[test]
fn mih_finds_near_duplicate_keys() {
    use itrace_core::index::{dedup_candidates, DedupKeys, MihIndex};
    let mut idx = MihIndex::new();
    let base = 0x0123456789ABCDEFu64;
    idx.insert(base, 0);
    // flip 6 low bits → hamming 6, inside radius
    let near = base ^ 0b111111;
    idx.insert(!base, 1); // far key
    let hits = idx.query(near, 10);
    assert_eq!(hits, vec![0]); // only base is near
    assert!(idx.query(near, 5).is_empty()); // outside radius

    // dedup_candidates: per-algo variant keys; pair flagged by ≥2 gate hashes
    let keys = |v: u64| vec![v; 8]; // an image's 8 variants of one hash
    let entries = vec![
        DedupKeys { image_id: 1, variant_keys: vec![keys(base), keys(0xAAAA), keys(0xBBBB)] },
        DedupKeys { image_id: 2, variant_keys: vec![keys(base), keys(0xAAAA), keys(0xFFFF)] }, // 2/3 match
        DedupKeys { image_id: 3, variant_keys: vec![keys(!base), keys(!0xAAAAu64), keys(!0xBBBBu64)] },
    ];
    let pairs = dedup_candidates(&entries, 8, 2);
    assert_eq!(pairs, vec![(0, 1)]);
}
