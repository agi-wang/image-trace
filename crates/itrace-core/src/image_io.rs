//! Image loading, grayscale conversion, resizing, orientation variants.

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};

use crate::{GrayImage, RgbImage};

/// Extensions we attempt to decode as raster images.
pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "jfif", "jpe", "png", "apng", "gif", "bmp", "dib", "tif", "tiff", "webp", "ico",
    "cur", "tga", "qoi", "pbm", "pgm", "ppm", "pnm", "ff", "farbfeld", "exr", "hdr",
];

/// Extensions treated as documents for image extraction.
pub const SUPPORTED_DOCUMENT_EXTENSIONS: &[&str] = &["pdf", "docx", "pptx"];

pub fn is_supported_image(filename: &str) -> bool {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| SUPPORTED_IMAGE_EXTENSIONS.iter().any(|s| e.eq_ignore_ascii_case(s)))
}

pub fn is_supported_document(filename: &str) -> bool {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| SUPPORTED_DOCUMENT_EXTENSIONS.iter().any(|s| e.eq_ignore_ascii_case(s)))
}

/// Decode a raster image from raw bytes.
pub fn decode(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    Ok(image::load_from_memory(bytes)?)
}

/// Decode an image file on disk.
pub fn decode_file(path: &std::path::Path) -> anyhow::Result<DynamicImage> {
    let bytes = std::fs::read(path)?;
    decode(&bytes)
}

pub fn to_gray(img: &DynamicImage) -> GrayImage {
    let luma = img.to_luma8();
    let (w, h) = luma.dimensions();
    GrayImage::new(w, h, luma.into_raw())
}

pub fn to_rgb(img: &DynamicImage) -> RgbImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    RgbImage::new(w, h, rgb.into_raw())
}

/// Downscale so max(width, height) <= max_side. Returns original if already small.
pub fn resize_max_side(img: &DynamicImage, max_side: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let m = w.max(h);
    if m <= max_side {
        return img.clone();
    }
    let s = max_side as f64 / m as f64;
    img.resize(
        ((w as f64) * s).round() as u32,
        ((h as f64) * s).round() as u32,
        image::imageops::FilterType::Triangle,
    )
}

/// Exact-size resize to (w, h) grayscale.
pub fn resize_gray_exact(gray: &GrayImage, w: u32, h: u32) -> GrayImage {
    if gray.width == w && gray.height == h {
        return gray.clone();
    }
    // View over the borrowed data — resize only needs a pixel source.
    let buf: ImageBuffer<Luma<u8>, &[u8]> =
        ImageBuffer::from_raw(gray.width, gray.height, gray.data.as_slice())
            .expect("gray buffer size matches");
    let out = image::imageops::resize(&buf, w, h, image::imageops::FilterType::Triangle);
    GrayImage::new(w, h, out.into_raw())
}

/// Downscale a GrayImage so max side <= max_side (keeps aspect).
pub fn resize_gray_max(gray: &GrayImage, max_side: u32) -> GrayImage {
    let m = gray.width.max(gray.height);
    if m <= max_side {
        return gray.clone();
    }
    let s = max_side as f64 / m as f64;
    let w = ((gray.width as f64) * s).round() as u32;
    let h = ((gray.height as f64) * s).round() as u32;
    resize_gray_exact(gray, w, h)
}

/// The 8 dihedral transforms applied to a decoded image (index 0 = identity).
///
/// 0 orig, 1 rot90, 2 rot180, 3 rot270,
/// 4 fliph, 5 fliph+rot90, 6 fliph+rot180, 7 fliph+rot270
pub fn orientation_variants(img: &DynamicImage) -> Vec<DynamicImage> {
    // Compute the flip's rotations before moving `flip` into its slot —
    // saves one full-buffer clone of the image.
    let flip = img.fliph();
    let flip90 = flip.rotate90();
    let flip180 = flip.rotate180();
    let flip270 = flip.rotate270();
    vec![
        img.clone(),
        img.rotate90(),
        img.rotate180(),
        img.rotate270(),
        flip,
        flip90,
        flip180,
        flip270,
    ]
}

/// Variants for a grayscale buffer — cheaper than going through DynamicImage.
pub fn gray_orientation_variants(gray: &GrayImage) -> Vec<GrayImage> {
    let (w, h) = (gray.width, gray.height);
    let (wu, hu) = (w as usize, h as usize);
    // Degenerate 0-sized buffer — every variant is empty (rotated variants
    // swap dims). Guarded because chunks_exact_mut(0) would panic.
    if wu == 0 || hu == 0 {
        let rot = || GrayImage::new(h, w, Vec::new());
        return vec![
            gray.clone(),
            rot(),
            gray.clone(),
            rot(),
            gray.clone(),
            rot(),
            gray.clone(),
            rot(),
        ];
    }
    // out(x,y) = gray[src(x,y)]; monomorphized per transform — no dispatch.
    // Used only for the strided (transpose-class) transforms; the row-parallel
    // ones below take memcpy/reverse fast paths.
    fn mapped(
        gray: &GrayImage,
        nw: u32,
        nh: u32,
        src: impl Fn(u32, u32) -> (u32, u32),
    ) -> GrayImage {
        let (nwu, w) = (nw as usize, gray.width as usize);
        let mut data = vec![0u8; nwu * nh as usize];
        // Iterate by destination row — kills the per-pixel dst multiply and
        // its bounds check.
        for (y, dst_row) in data.chunks_exact_mut(nwu).enumerate() {
            let y = y as u32;
            for (x, px) in dst_row.iter_mut().enumerate() {
                let (sx, sy) = src(x as u32, y);
                *px = gray.data[sy as usize * w + sx as usize];
            }
        }
        GrayImage::new(nw, nh, data)
    }
    // flip_horizontal in place on a fresh buffer: dst(x,y) = src(w-1-x,y) —
    // a per-row reverse.
    let mut flip_data = gray.data.clone();
    for row in flip_data.chunks_exact_mut(wu) {
        row.reverse();
    }
    // rot180 = full-buffer reverse of row-major data:
    // dst(y*w+x) = src(w*h-1-(y*w+x)).
    let mut rot180_data = gray.data.clone();
    rot180_data.reverse();
    // flip+rot180 = flip_vertical: dst row y = src row h-1-y — memcpy rows.
    let mut flipv_data = vec![0u8; wu * hu];
    for (y, dst_row) in flipv_data.chunks_exact_mut(wu).enumerate() {
        let sy = (hu - 1 - y) * wu;
        dst_row.copy_from_slice(&gray.data[sy..sy + wu]);
    }
    vec![
        gray.clone(),
        // rot90 cw: new(x,y) = old(y, w-1-x)
        mapped(gray, h, w, |x, y| (y, h - 1 - x)),
        GrayImage::new(w, h, rot180_data),
        mapped(gray, h, w, |x, y| (w - 1 - y, x)),
        GrayImage::new(w, h, flip_data),
        // flip + rot90: rotate90(flip(img)) → new(x,y) = flip(y, h-1-x) = old(w-1-y, h-1-x)
        mapped(gray, h, w, |x, y| (w - 1 - y, h - 1 - x)),
        // flip + rot180 = flip_vertical
        GrayImage::new(w, h, flipv_data),
        // flip + rot270: rotate270(flip) → new(x,y)=flip(w-1-y, x) = old(w-1-(w-1-y), x)=old(y,x)... transpose
        mapped(gray, h, w, |x, y| (y, x)),
    ]
}

/// Variants for an RGB buffer — the 3-byte-per-pixel sibling of
/// [`gray_orientation_variants`]: identical transforms applied to pixel
/// triples. `to_rgb` is a per-pixel conversion, so it commutes with the
/// dihedral permutations and the result is byte-identical to
/// `to_rgb(orientation_variants(&img)[i])` — for any `DynamicImage` color
/// type — while skipping the per-variant DynamicImage clones/conversions.
pub fn rgb_orientation_variants(rgb: &RgbImage) -> Vec<RgbImage> {
    let (w, h) = (rgb.width, rgb.height);
    let (wu, hu) = (w as usize, h as usize);
    // Degenerate 0-sized buffer — every variant is empty (rotated variants
    // swap dims). Mirrors the gray path's guard.
    if wu == 0 || hu == 0 {
        let rot = || RgbImage::new(h, w, Vec::new());
        return vec![
            rgb.clone(),
            rot(),
            rgb.clone(),
            rot(),
            rgb.clone(),
            rot(),
            rgb.clone(),
            rot(),
        ];
    }
    // out(x,y) = rgb[src(x,y)] — same source mapping as the gray `mapped`,
    // copying whole 3-byte pixels. Used only for the strided
    // (transpose-class) transforms.
    fn mapped(rgb: &RgbImage, nw: u32, nh: u32, src: impl Fn(u32, u32) -> (u32, u32)) -> RgbImage {
        let (nwu, w) = (nw as usize, rgb.width as usize);
        let mut data = vec![0u8; nwu * nh as usize * 3];
        // Iterate by destination row — kills the per-pixel dst multiply and
        // its bounds check; each dst pixel copies the src triple.
        for (y, dst_row) in data.chunks_exact_mut(nwu * 3).enumerate() {
            let y = y as u32;
            for (x, px) in dst_row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (sx, sy) = src(x as u32, y);
                let s = (sy as usize * w + sx as usize) * 3;
                px.copy_from_slice(&rgb.data[s..s + 3]);
            }
        }
        RgbImage::new(nw, nh, data)
    }
    // flip_horizontal: dst(x,y) = src(w-1-x,y) — per-row pixel reversal.
    let mut flip_data = rgb.data.clone();
    for row in flip_data.chunks_exact_mut(wu * 3) {
        row.as_chunks_mut::<3>().0.reverse();
    }
    // rot180 = full-buffer pixel-order reversal:
    // dst(y*w+x) = src(w*h-1-(y*w+x)) on triples.
    let mut rot180_data = rgb.data.clone();
    rot180_data.as_chunks_mut::<3>().0.reverse();
    // flip+rot180 = flip_vertical: dst row y = src row h-1-y — memcpy rows.
    let mut flipv_data = vec![0u8; wu * hu * 3];
    for (y, dst_row) in flipv_data.chunks_exact_mut(wu * 3).enumerate() {
        let sy = (hu - 1 - y) * wu * 3;
        dst_row.copy_from_slice(&rgb.data[sy..sy + wu * 3]);
    }
    vec![
        rgb.clone(),
        // rot90 cw: new(x,y) = old(y, w-1-x)
        mapped(rgb, h, w, |x, y| (y, h - 1 - x)),
        RgbImage::new(w, h, rot180_data),
        mapped(rgb, h, w, |x, y| (w - 1 - y, x)),
        RgbImage::new(w, h, flip_data),
        // flip + rot90: rotate90(flip(img)) → new(x,y) = old(w-1-y, h-1-x)
        mapped(rgb, h, w, |x, y| (w - 1 - y, h - 1 - x)),
        // flip + rot180 = flip_vertical
        RgbImage::new(w, h, flipv_data),
        // flip + rot270 → transpose: new(x,y) = old(y,x)
        mapped(rgb, h, w, |x, y| (y, x)),
    ]
}

/// Encode an RGB buffer as JPEG bytes.
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> anyhow::Result<Vec<u8>> {
    let buf: ImageBuffer<image::Rgb<u8>, &[u8]> =
        ImageBuffer::from_raw(img.width, img.height, img.data.as_slice())
            .ok_or_else(|| anyhow::anyhow!("invalid rgb buffer"))?;
    let mut cur = std::io::Cursor::new(Vec::new());
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cur, quality);
    enc.encode_image(&buf)?;
    Ok(cur.into_inner())
}

/// Save an RGB buffer as JPEG for thumbnails / visualizations.
pub fn save_jpeg(img: &RgbImage, path: &std::path::Path, quality: u8) -> anyhow::Result<()> {
    std::fs::write(path, encode_jpeg(img, quality)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured photo: gradient + pseudo-random blobs — the
    /// same generator style the pipeline tests use (no fixtures).
    fn photo_rgb(w: u32, h: u32, seed: u8) -> DynamicImage {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = ((x ^ y) as u8).wrapping_add(seed);
                img.put_pixel(x, y, image::Rgb([r, g, b]));
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
                        img.put_pixel(px, py, image::Rgb([(k >> 16) as u8, 200, (k >> 3) as u8]));
                    }
                }
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    /// `rgb_orientation_variants` must be byte-identical to the reference
    /// path `to_rgb(orientation_variants(img)[i])` it replaces — checked on
    /// Rgb8 (permutation only), Rgba8 and Luma8 (conversion + permutation)
    /// sources, at odd sizes that stress the transpose mappings.
    #[test]
    fn rgb_variants_match_dynamicimage_path() {
        let rgba = {
            let mut buf = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(53, 37);
            for (x, y, p) in buf.enumerate_pixels_mut() {
                *p = image::Rgba([
                    (x * 5 % 256) as u8,
                    (y * 7 % 256) as u8,
                    ((x * y) % 256) as u8,
                    128 + (x % 64) as u8,
                ]);
            }
            DynamicImage::ImageRgba8(buf)
        };
        let luma = {
            let mut buf = image::ImageBuffer::<Luma<u8>, Vec<u8>>::new(61, 41);
            for (x, y, p) in buf.enumerate_pixels_mut() {
                *p = Luma([((x * 3 + y * 11) % 256) as u8]);
            }
            DynamicImage::ImageLuma8(buf)
        };
        for img in [photo_rgb(97, 61, 3), rgba, luma] {
            let rgb0 = to_rgb(&img);
            let fast = rgb_orientation_variants(&rgb0);
            let reference = orientation_variants(&img);
            assert_eq!(fast.len(), reference.len());
            for i in 0..reference.len() {
                let want = to_rgb(&reference[i]);
                assert_eq!(fast[i].width, want.width, "variant {i} width");
                assert_eq!(fast[i].height, want.height, "variant {i} height");
                assert_eq!(fast[i].data, want.data, "variant {i} payload");
            }
        }
    }

    /// Degenerate/1-px buffers: every transform is still well-formed.
    #[test]
    fn rgb_variants_edge_sizes() {
        for (w, h) in [(1u32, 1u32), (2, 1), (1, 3), (4, 2)] {
            let data: Vec<u8> = (0..(w * h * 3) as u8).map(|v| v.wrapping_mul(37)).collect();
            let rgb = RgbImage::new(w, h, data);
            let vars = rgb_orientation_variants(&rgb);
            assert_eq!(vars.len(), 8);
            for (i, v) in vars.iter().enumerate() {
                let (ew, eh) = if i % 2 == 0 { (w, h) } else { (h, w) };
                assert_eq!((v.width, v.height), (ew, eh), "variant {i} dims {w}x{h}");
            }
            // identity + flips on these sizes are self-checking: variant 0 = src
            assert_eq!(vars[0].data, rgb.data);
        }
    }
}
