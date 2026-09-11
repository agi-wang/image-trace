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
