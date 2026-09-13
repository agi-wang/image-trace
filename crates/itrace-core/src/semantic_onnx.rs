//! ONNX Runtime-backed [`SemanticEmbedder`] (Phase 5 R1), compiled only
//! with the `semantic-onnx` cargo feature.
//!
//! `ort` is built with `load-dynamic`: the runtime shared library is
//! `dlopen`'d at load time, so nothing is linked or downloaded at build
//! time and the default (feature-off) build never sees the dependency.
//! Dylib resolution order:
//!
//! 1. `ITRACE_ORT_DYLIB` (this crate's knob — an explicit path), then
//! 2. `ORT_DYLIB_PATH` (the `ort` crate's own env var), then
//! 3. the default soname (`libonnxruntime.so` / `.dylib` /
//!    `onnxruntime.dll`) resolved next to the executable, then on the
//!    loader search path.
//!
//! # Model contract
//!
//! The backend targets DINOv2-class dense models exported to ONNX —
//! it is **not** a generic ONNX runner:
//!
//! - **Input:** one f32 NCHW tensor `[1, 3, 224, 224]`, pixels scaled to
//!   `[0, 1]` then normalized with ImageNet mean/std. The input named
//!   `pixel_values` is preferred; otherwise the first declared input is
//!   used.
//! - **Output:** the first output must be an f32 tensor shaped `[1, D]`
//!   (pooled embedding) or `[1, T, D]` (token map → the CLS/first token
//!   row is taken). Anything else is an error, not a guess.
//!
//! Real DINOv2 weights are a follow-up and are never downloaded by CI —
//! see `docs/SCALE_10_11.md`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, ensure, Context};

use crate::semantic::SemanticEmbedder;

/// Env var naming the ONNX Runtime shared library (checked before the
/// `ort` crate's own `ORT_DYLIB_PATH`).
pub const ORT_DYLIB_ENV: &str = "ITRACE_ORT_DYLIB";

/// Square side the input image is resized to (DINOv2 patch grid).
const ONNX_INPUT_SIDE: u32 = 224;

/// ImageNet normalization — the DINOv2 preprocessing contract.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// ONNX Runtime-backed [`SemanticEmbedder`] for DINOv2-class dense
/// models — see the module docs for the input/output contract.
///
/// `load` never panics: a missing/incompatible dylib, or missing/invalid
/// weights, returns `Err` and `embedder_from_env` maps that to `None` —
/// the channel goes inert rather than falling back to the stub.
pub struct OnnxSemanticEmbedder {
    /// `Session::run` takes `&mut self`; `embed_bytes` is `&self`.
    session: Mutex<ort::session::Session>,
    /// `pixel_values` when the model declares it, else the first input.
    input_name: String,
    /// Static last output dim when the model declares it, else set by the
    /// first successful `embed_bytes` (`0` = unknown until then — the
    /// callers' zero-vector fallback handles that gracefully).
    dim: AtomicUsize,
}

impl OnnxSemanticEmbedder {
    /// Initialize ORT (dlopen the dylib) and load `model_path`.
    ///
    /// `ort` *panics* (rather than returning an error) when the dylib
    /// can't be loaded, so the whole init+load is run under
    /// `catch_unwind` — a broken runtime install degrades to `Err` and
    /// an inert channel instead of aborting the process.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        ensure!(
            model_path.is_file(),
            "ONNX weights not found: {}",
            model_path.display()
        );
        catch_unwind(AssertUnwindSafe(|| Self::load_inner(model_path)))
            .map_err(|_| anyhow!("onnxruntime init panicked (missing/incompatible dylib?)"))?
    }

    fn load_inner(model_path: &Path) -> anyhow::Result<Self> {
        match std::env::var(ORT_DYLIB_ENV) {
            Ok(p) if !p.trim().is_empty() => ort::init_from(p.trim()).commit(),
            _ => ort::init().commit(),
        }
        .map_err(|e| anyhow!("onnxruntime init: {e}"))?;
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow!("ort session builder: {e}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("load ONNX weights {}", model_path.display()))?;
        let input_name = session
            .inputs
            .iter()
            .find(|i| i.name == "pixel_values")
            .or_else(|| session.inputs.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow!("model declares no inputs"))?;
        // Static last dim of the first tensor output, if declared
        // (dynamic dims come back as -1 and are learned on first embed).
        let dim = session
            .outputs
            .first()
            .and_then(|o| match &o.output_type {
                ort::value::ValueType::Tensor { shape, .. } => {
                    shape.last().copied().filter(|d| *d > 0).map(|d| d as usize)
                }
                _ => None,
            })
            .unwrap_or(0);
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            dim: AtomicUsize::new(dim),
        })
    }
}

/// Decode → resize to `224×224` → NCHW f32 `[1, 3, 224, 224]` normalized
/// with ImageNet mean/std (the DINOv2 contract). Pure function — no
/// `ort` types — so it stays unit-testable without the runtime dylib.
fn dinov2_preprocess(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let img = crate::image_io::decode(bytes)?
        .resize_exact(
            ONNX_INPUT_SIDE,
            ONNX_INPUT_SIDE,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let side = ONNX_INPUT_SIDE as usize;
    let mut data = vec![0f32; 3 * side * side];
    for (x, y, px) in img.enumerate_pixels() {
        for (c, ch) in px.0.iter().enumerate() {
            data[c * side * side + y as usize * side + x as usize] =
                (*ch as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    Ok(data)
}

impl SemanticEmbedder for OnnxSemanticEmbedder {
    fn dim(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    fn embed_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        let data = dinov2_preprocess(bytes)?;
        let side = ONNX_INPUT_SIDE as usize;
        let input =
            ort::value::TensorRef::from_array_view(([1usize, 3, side, side], data.as_slice()))
                .map_err(|e| anyhow!("ort input tensor: {e}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("ort session poisoned"))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| anyhow!("ort run: {e}"))?;
        let (shape, buf) = outputs[0usize]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extract f32 output: {e}"))?;
        let vec = match shape.len() {
            // [1, D] pooled embedding, or [1, tokens, D] token map →
            // take the CLS (first token) row.
            2 | 3 => {
                ensure!(shape[0] == 1, "expected batch dim 1, got {shape:?}");
                let d = *shape.last().unwrap() as usize;
                ensure!(d > 0 && buf.len() >= d, "bad output shape {shape:?}");
                buf[..d].to_vec()
            }
            _ => bail!(
                "unsupported semantic output rank {} ({shape:?})",
                shape.len()
            ),
        };
        self.dim.store(vec.len(), Ordering::Relaxed);
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal protobuf writer — enough to hand-encode the tiny ONNX
    /// `ModelProto` fixture below (keeps the test self-contained: no
    /// protoc / onnx-python dependency).
    fn pb_varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn pb_field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
        pb_varint(out, (field as u64) << 3);
        pb_varint(out, v);
    }
    fn pb_field_len(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
        pb_varint(out, ((field as u64) << 3) | 2);
        pb_varint(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }

    /// ~200-byte ONNX model: `emb = Flatten(pixel_values, axis=1)` —
    /// input f32 `[1, 3, H, W]` (H/W dynamic), output f32 `[1, N]`.
    /// Enough to drive load + `embed_bytes` end-to-end on a dev box
    /// that has libonnxruntime, without downloading any weights.
    fn tiny_flatten_onnx() -> Vec<u8> {
        fn shape_proto(dims: &[i64]) -> Vec<u8> {
            let mut s = Vec::new();
            for &d in dims {
                let mut dim = Vec::new();
                if d > 0 {
                    pb_field_varint(&mut dim, 1, d as u64); // Dimension.dim_value
                }
                pb_field_len(&mut s, 1, &dim); // TensorShapeProto.dim
            }
            s
        }
        fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
            let mut tensor = Vec::new();
            pb_field_varint(&mut tensor, 1, 1); // elem_type = FLOAT
            pb_field_len(&mut tensor, 2, &shape_proto(dims));
            let mut ty = Vec::new();
            pb_field_len(&mut ty, 1, &tensor); // TypeProto.tensor_type
            let mut vi = Vec::new();
            pb_field_len(&mut vi, 1, name.as_bytes()); // ValueInfoProto.name
            pb_field_len(&mut vi, 2, &ty); // ValueInfoProto.type
            vi
        }
        // AttributeProto{ name:"axis", i:1, type:INT }
        let mut attr = Vec::new();
        pb_field_len(&mut attr, 1, b"axis");
        pb_field_varint(&mut attr, 3, 1); // i = 1
        pb_field_varint(&mut attr, 20, 2); // type = INT
                                           // NodeProto{ input:["pixel_values"], output:["emb"],
                                           //            op_type:"Flatten", attribute:[axis=1] }
        let mut node = Vec::new();
        pb_field_len(&mut node, 1, b"pixel_values");
        pb_field_len(&mut node, 2, b"emb");
        pb_field_len(&mut node, 4, b"Flatten");
        pb_field_len(&mut node, 5, &attr);
        // GraphProto{ node, name:"g", input, output }
        let mut graph = Vec::new();
        pb_field_len(&mut graph, 1, &node);
        pb_field_len(&mut graph, 2, b"g");
        pb_field_len(&mut graph, 11, &value_info("pixel_values", &[1, 3, 0, 0]));
        pb_field_len(&mut graph, 12, &value_info("emb", &[1, 0]));
        // OperatorSetIdProto{ version:13 } (domain "" = default ai.onnx)
        let mut opset = Vec::new();
        pb_field_varint(&mut opset, 2, 13);
        // ModelProto{ ir_version:8, graph, opset_import }
        let mut m = Vec::new();
        pb_field_varint(&mut m, 1, 8);
        pb_field_len(&mut m, 7, &graph);
        pb_field_len(&mut m, 8, &opset);
        m
    }

    fn test_png(seed: u8) -> Vec<u8> {
        let mut img = image::RgbImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, image::Rgb([x as u8 * 8, y as u8 * 8, seed]));
            }
        }
        let mut cur = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cur, image::ImageFormat::Png)
            .unwrap();
        cur.into_inner()
    }

    /// Would `ort` find a runtime dylib? Mirrors `dylib_path()`'s
    /// resolution; the happy-path test skips (not fails) on CI.
    fn ort_dylib_reachable() -> bool {
        for var in [ORT_DYLIB_ENV, "ORT_DYLIB_PATH"] {
            if let Ok(p) = std::env::var(var) {
                return std::path::Path::new(&p).is_file();
            }
        }
        #[cfg(target_os = "windows")]
        const SONAME: &str = "onnxruntime.dll";
        #[cfg(target_os = "macos")]
        const SONAME: &str = "libonnxruntime.dylib";
        #[cfg(all(unix, not(target_os = "macos")))]
        const SONAME: &str = "libonnxruntime.so";
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|d| d.join(SONAME)))
            .is_some_and(|p| p.is_file())
    }

    #[test]
    fn preprocess_shape_and_normalization() {
        let v = dinov2_preprocess(&test_png(7)).unwrap();
        assert_eq!(v.len(), 3 * 224 * 224);
        // ImageNet-normalized pixels stay in a sane band, not raw [0,255].
        assert!(v.iter().all(|&x| (-4.0..=4.0).contains(&x)));
        assert!(v.iter().any(|&x| x != 0.0));
    }

    /// Failure modes that never touch the dylib: missing file is
    /// rejected up front; a garbage file fails to load as ONNX. Both
    /// must be `Err` — the resolver turns them into an inert channel,
    /// never a stub fallback.
    #[test]
    fn load_failures_are_errors_not_panics() {
        let dir = std::env::temp_dir().join(format!("itrace-onnx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("nope.onnx");
        assert!(OnnxSemanticEmbedder::load(&missing).is_err());
        let garbage = dir.join("garbage.onnx");
        std::fs::write(&garbage, b"not a protobuf").unwrap();
        assert!(OnnxSemanticEmbedder::load(&garbage).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `semantic-onnx` build without a runtime dylib must still fail
    /// soft: `load` returns `Err` (panic trapped), never aborts.
    #[test]
    fn missing_dylib_is_inert_error() {
        if ort_dylib_reachable() {
            return; // dylib present — the happy-path test covers it
        }
        let dir = std::env::temp_dir().join(format!("itrace-onnx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("tiny.onnx");
        std::fs::write(&model, tiny_flatten_onnx()).unwrap();
        assert!(OnnxSemanticEmbedder::load(&model).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Dev-box happy path (skips on CI, which has no libonnxruntime):
    /// the hand-encoded fixture loads and `embed_bytes` returns the
    /// flattened `[1, 3*224*224]` vector.
    #[test]
    fn tiny_fixture_loads_and_embeds() {
        if !ort_dylib_reachable() {
            eprintln!("skipping: no libonnxruntime dylib configured");
            return;
        }
        let dir = std::env::temp_dir().join(format!("itrace-onnx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("tiny.onnx");
        std::fs::write(&model, tiny_flatten_onnx()).unwrap();
        let e = OnnxSemanticEmbedder::load(&model).unwrap();
        let v = e.embed_bytes(&test_png(3)).unwrap();
        assert_eq!(v.len(), 3 * 224 * 224);
        assert_eq!(e.dim(), v.len());
        assert_eq!(v, e.embed_bytes(&test_png(3)).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fixture_stays_tiny() {
        assert!(tiny_flatten_onnx().len() < 1024);
    }
}
