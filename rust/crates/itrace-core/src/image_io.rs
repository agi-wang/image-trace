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
        .map(|e| SUPPORTED_IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn is_supported_document(filename: &str) -> bool {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_DOCUMENT_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
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
    let buf: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(gray.width, gray.height, gray.data.clone())
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
    let flip = img.fliph();
    vec![
        img.clone(),
        img.rotate90(),
        img.rotate180(),
        img.rotate270(),
        flip.clone(),
        flip.rotate90(),
        flip.rotate180(),
        flip.rotate270(),
    ]
}

/// Variants for a grayscale buffer — cheaper than going through DynamicImage.
pub fn gray_orientation_variants(gray: &GrayImage) -> Vec<GrayImage> {
    let (w, h) = (gray.width, gray.height);
    let make = |f: &dyn Fn(u32, u32) -> u8, nw: u32, nh: u32| -> GrayImage {
        let mut data = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            for x in 0..nw {
                data.push(f(x, y));
            }
        }
        GrayImage::new(nw, nh, data)
    };
    let g = &gray.data;
    let at = move |x: u32, y: u32| g[(y * w + x) as usize];
    vec![
        gray.clone(),
        // rot90 cw: new(x,y) = old(y, w-1-x) — checked below
        make(&|x, y| at(y, h - 1 - x), h, w),
        make(&|x, y| at(w - 1 - x, h - 1 - y), w, h),
        make(&|x, y| at(w - 1 - y, x), h, w),
        make(&|x, y| at(w - 1 - x, y), w, h),
        // flip + rot90: rotate90(flip(img)) → new(x,y) = flip(y, h-1-x) = old(w-1-y, h-1-x)
        make(&|x, y| at(w - 1 - y, h - 1 - x), h, w),
        // flip + rot180 = flip_vertical
        make(&|x, y| at(x, h - 1 - y), w, h),
        // flip + rot270: rotate270(flip) → new(x,y)=flip(w-1-y, x) = old(w-1-(w-1-y), x)=old(y,x)... transpose
        make(&|x, y| at(y, x), h, w),
    ]
}

/// Save an RGB buffer as JPEG for thumbnails / visualizations.
pub fn save_jpeg(img: &RgbImage, path: &std::path::Path, quality: u8) -> anyhow::Result<()> {
    let buf: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(img.width, img.height, img.data.clone())
            .ok_or_else(|| anyhow::anyhow!("invalid rgb buffer"))?;
    buf.save_with_format(path, image::ImageFormat::Jpeg)?;
    let _ = quality;
    Ok(())
}
