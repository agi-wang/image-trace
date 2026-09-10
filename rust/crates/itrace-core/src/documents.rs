//! Document image extraction: DOCX/PPTX (OOXML zip media) and PDF
//! embedded images via lopdf.

use std::io::Read;
use std::path::Path;

/// One extracted image, raw bytes + provenance.
#[derive(Debug)]
pub struct ExtractedImage {
    pub filename: String,
    pub data: Vec<u8>,
    pub page_number: Option<u32>,
    pub image_index: u32,
    pub method: &'static str,
}

/// Sniff a raster format from magic bytes.
pub fn sniff_extension(data: &[u8]) -> &'static str {
    if data.starts_with(b"\xff\xd8\xff") {
        "jpg"
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "png"
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        "gif"
    } else if data.starts_with(b"BM") {
        "bmp"
    } else if data.starts_with(b"II*\x00") || data.starts_with(b"MM\x00*") {
        "tiff"
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        "webp"
    } else if data.starts_with(b"qoif") {
        "qoi"
    } else {
        "jpg"
    }
}

/// Dispatch by extension.
pub fn extract(path: &Path) -> anyhow::Result<Vec<ExtractedImage>> {
    let data = std::fs::read(path)?;
    extract_named(
        path.file_name().and_then(|n| n.to_str()).unwrap_or("doc"),
        &data,
    )
}

/// Extract from in-memory bytes; `name` supplies the extension.
pub fn extract_named(name: &str, data: &[u8]) -> anyhow::Result<Vec<ExtractedImage>> {
    let ext = name
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let stem = name.rsplit('.').nth(1).unwrap_or("doc");
    match ext.as_str() {
        "docx" | "pptx" => extract_office(stem, data),
        "pdf" => extract_pdf(stem, data),
        other => anyhow::bail!("不支持的文档格式: {other}"),
    }
}

// ---------- OOXML ----------

fn extract_office(stem: &str, data: &[u8]) -> anyhow::Result<Vec<ExtractedImage>> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(data))?;
    let mut out = Vec::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().to_string();
        let is_media = name.starts_with("word/media/") || name.starts_with("ppt/media/");
        if !is_media {
            continue;
        }
        let mut data = Vec::new();
        entry.read_to_end(&mut data)?;
        let orig = Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let ext = match orig.as_str() {
            "" | "emf" | "wmf" => sniff_extension(&data).to_string(),
            _ => orig,
        };
        let idx = out.len() as u32;
        out.push(ExtractedImage {
            filename: format!("{stem}_media{}.{ext}", idx + 1),
            data,
            page_number: None,
            image_index: idx,
            method: "office_media",
        });
    }
    Ok(out)
}

// ---------- PDF ----------

/// Extract embedded images from a PDF page by page.
/// Handles DCTDecode (JPEG) and FlateDecode (raw RGB/Gray → re-encoded PNG).
/// Page-render fallback (for PDFs with zero embedded images) is not included
/// in the pure-Rust build — requires a pdfium/mupdf backend feature.
fn extract_pdf(stem: &str, data: &[u8]) -> anyhow::Result<Vec<ExtractedImage>> {
    let doc = lopdf::Document::load_mem(data)?;
    let mut out = Vec::new();
    let pages = doc.get_pages();
    let mut idx = 0u32;
    for (page_no, page_id) in pages {
        let images = match doc.get_page_images(page_id) {
            Ok(v) => v,
            Err(_) => continue,
        };
        for img in images {
            let filters = img.filters.clone().unwrap_or_default();
            let filename;
            let data;
            if filters.iter().any(|f| f == "DCTDecode") {
                // already a JPEG codestream
                filename = format!("{stem}_page{page_no}_img{}.jpg", idx + 1);
                data = img.content.to_vec();
            } else if filters.iter().any(|f| f == "FlateDecode") {
                match decode_flate_image(&img) {
                    Some(png) => {
                        filename = format!("{stem}_page{page_no}_img{}.png", idx + 1);
                        data = png;
                    }
                    None => continue,
                }
            } else {
                // JPXDecode / CCITT etc. — no decoder in the pure-Rust build
                continue;
            }
            out.push(ExtractedImage {
                filename,
                data,
                page_number: Some(page_no),
                image_index: idx,
                method: "pdf_embedded",
            });
            idx += 1;
        }
    }
    Ok(out)
}

fn decode_flate_image(img: &lopdf::xobject::PdfImage) -> Option<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let bpc = img.bits_per_component.unwrap_or(8);
    if bpc != 8 || w == 0 || h == 0 {
        return None;
    }
    let inflated = inflate_zlib(img.content)?;
    let cs = img.color_space.as_deref().unwrap_or("");
    let rgb: Vec<u8> = match cs {
        "DeviceRGB" | "CalRGB" => {
            if inflated.len() < w * h * 3 {
                return None;
            }
            inflated[..w * h * 3].to_vec()
        }
        "DeviceGray" | "CalGray" | "" => {
            if inflated.len() < w * h {
                return None;
            }
            inflated[..w * h].iter().flat_map(|&g| [g, g, g]).collect()
        }
        "DeviceCMYK" => {
            if inflated.len() < w * h * 4 {
                return None;
            }
            inflated[..w * h * 4]
                .chunks_exact(4)
                .flat_map(|c| {
                    let (cy, m, y, k) = (
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                        c[3] as f32 / 255.0,
                    );
                    [
                        ((1.0 - cy) * (1.0 - k) * 255.0) as u8,
                        ((1.0 - m) * (1.0 - k) * 255.0) as u8,
                        ((1.0 - y) * (1.0 - k) * 255.0) as u8,
                    ]
                })
                .collect()
        }
        _ => return None,
    };
    let buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, rgb)?;
    let mut png = std::io::Cursor::new(Vec::new());
    buf.write_to(&mut png, image::ImageFormat::Png).ok()?;
    Some(png.into_inner())
}

fn inflate_zlib(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut dec = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).ok()?;
    Some(out)
}
