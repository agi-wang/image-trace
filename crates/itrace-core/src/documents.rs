//! Document image extraction: DOCX/PPTX (OOXML zip media) and PDF
//! embedded images via lopdf.

use std::io::{Read, Seek};
use std::path::Path;

use rayon::prelude::*;

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
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("doc");
    let ext = name
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "docx" || ext == "pptx" {
        // OOXML is a zip — stream it off disk instead of buffering the
        // whole file in memory first.
        let stem = name.rsplit('.').nth(1).unwrap_or("doc");
        return extract_office(stem, std::fs::File::open(path)?);
    }
    extract_named(name, &std::fs::read(path)?)
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
        "docx" | "pptx" => extract_office(stem, std::io::Cursor::new(data)),
        "pdf" => extract_pdf(stem, data),
        other => anyhow::bail!("不支持的文档格式: {other}"),
    }
}

// ---------- OOXML ----------

/// Hard cap per zip member — a corrupted archive could otherwise inflate
/// a media entry into an OOM-sized buffer.
const MAX_MEDIA_BYTES: u64 = 512 << 20; // 512 MiB

fn extract_office<R: Read + Seek>(stem: &str, reader: R) -> anyhow::Result<Vec<ExtractedImage>> {
    let mut zip = zip::ZipArchive::new(reader)?;
    let mut out = Vec::new();
    for i in 0..zip.len() {
        let entry = zip.by_index(i)?;
        let name = entry.name().to_string();
        let is_media = name.starts_with("word/media/") || name.starts_with("ppt/media/");
        if !is_media || entry.size() > MAX_MEDIA_BYTES {
            continue;
        }
        // Cap the read as well — the declared size can understate the
        // real inflated length on malformed archives.
        let mut data = Vec::new();
        entry.take(MAX_MEDIA_BYTES + 1).read_to_end(&mut data)?;
        if data.len() as u64 > MAX_MEDIA_BYTES {
            continue;
        }
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
    // Collect owned decode jobs first — `PdfImage` borrows the document, so
    // the inflate/PNG-encode work is fanned out over owned payloads while
    // job order (hence result order) stays page order.
    let mut jobs: Vec<PdfJob> = Vec::new();
    for (page_no, page_id) in doc.get_pages() {
        let images = match doc.get_page_images(page_id) {
            Ok(v) => v,
            Err(_) => continue,
        };
        for img in images {
            let filters = img.filters.as_deref().unwrap_or(&[]);
            if filters.iter().any(|f| f == "DCTDecode") {
                // already a JPEG codestream
                jobs.push(PdfJob::Jpeg {
                    page_no,
                    content: img.content.to_vec(),
                });
            } else if filters.iter().any(|f| f == "FlateDecode") {
                jobs.push(PdfJob::Flate(FlateImage {
                    page_no,
                    content: img.content.to_vec(),
                    width: img.width,
                    height: img.height,
                    color_space: img.color_space.clone(),
                    bits_per_component: img.bits_per_component,
                }));
            }
            // JPXDecode / CCITT etc. — no decoder in the pure-Rust build
        }
    }
    // Parallel decode; `filter_map`+`collect` preserves job order.
    let decoded: Vec<(u32, &'static str, Vec<u8>)> = jobs
        .into_par_iter()
        .filter_map(|job| match job {
            PdfJob::Jpeg { page_no, content } => Some((page_no, "jpg", content)),
            PdfJob::Flate(f) => decode_flate_image(&f).map(|png| (f.page_no, "png", png)),
        })
        .collect();
    let mut out = Vec::with_capacity(decoded.len());
    for (idx, (page_no, ext, data)) in decoded.into_iter().enumerate() {
        let idx = idx as u32;
        out.push(ExtractedImage {
            filename: format!("{stem}_page{page_no}_img{}.{ext}", idx + 1),
            data,
            page_number: Some(page_no),
            image_index: idx,
            method: "pdf_embedded",
        });
    }
    Ok(out)
}

/// One decodable PDF image payload, owned so decode work can run on rayon
/// workers without borrowing the parsed document.
enum PdfJob {
    /// DCTDecode — content is already a JPEG codestream.
    Jpeg { page_no: u32, content: Vec<u8> },
    /// FlateDecode — raw samples to inflate, convert, and re-encode as PNG.
    Flate(FlateImage),
}

/// Owned subset of `lopdf::xobject::PdfImage` fields used by the
/// FlateDecode path.
struct FlateImage {
    page_no: u32,
    content: Vec<u8>,
    width: i64,
    height: i64,
    color_space: Option<String>,
    bits_per_component: Option<i64>,
}

fn decode_flate_image(img: &FlateImage) -> Option<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let bpc = img.bits_per_component.unwrap_or(8);
    if bpc != 8 || w == 0 || h == 0 {
        return None;
    }
    // Reject absurd geometries before any w*h arithmetic can overflow.
    let npix = w.checked_mul(h)?;
    let mut inflated = inflate_zlib(&img.content)?;
    let cs = img.color_space.as_deref().unwrap_or("");
    let rgb: Vec<u8> = match cs {
        "DeviceRGB" | "CalRGB" => {
            if inflated.len() / 3 < npix {
                return None;
            }
            inflated.truncate(npix * 3);
            inflated
        }
        "DeviceGray" | "CalGray" | "" => {
            if inflated.len() < npix {
                return None;
            }
            let mut rgb = Vec::with_capacity(npix * 3);
            for &g in &inflated[..npix] {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        "DeviceCMYK" => {
            if inflated.len() / 4 < npix {
                return None;
            }
            let mut rgb = Vec::with_capacity(npix * 3);
            for c in inflated[..npix * 4].as_chunks::<4>().0 {
                let (cy, m, y, k) = (
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                );
                rgb.extend_from_slice(&[
                    ((1.0 - cy) * (1.0 - k) * 255.0) as u8,
                    ((1.0 - m) * (1.0 - k) * 255.0) as u8,
                    ((1.0 - y) * (1.0 - k) * 255.0) as u8,
                ]);
            }
            rgb
        }
        _ => return None,
    };
    let buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, rgb)?;
    let mut png = std::io::Cursor::new(Vec::with_capacity(npix / 2));
    buf.write_to(&mut png, image::ImageFormat::Png).ok()?;
    Some(png.into_inner())
}

/// Inflate a zlib stream with a hard output cap — malformed or hostile
/// streams bail instead of growing an unbounded buffer.
fn inflate_zlib(data: &[u8]) -> Option<Vec<u8>> {
    const MAX_INFLATED: u64 = 512 << 20; // 512 MiB
    let mut dec = flate2::read::ZlibDecoder::new(data).take(MAX_INFLATED + 1);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).ok()?;
    if out.len() as u64 > MAX_INFLATED {
        return None;
    }
    Some(out)
}
