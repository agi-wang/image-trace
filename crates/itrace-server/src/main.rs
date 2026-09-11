//! Image Trace RS — axum HTTP API implementing docs/openapi.yaml.

mod draw;
mod service;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing_subscriber::EnvFilter;

use itrace_core::{self as core, DESCRIPTOR_ALGOS};
use itrace_store::{ImageRecord, Store};

// ---------- state ----------

#[derive(Clone)]
struct AppState {
    store: Arc<Store>,
    /// Small decoded-gray cache (≤64 entries, keyed by image id) shared by
    /// the match/slice endpoints — they re-decode the same blobs per call.
    gray_cache:
        Arc<std::sync::Mutex<std::collections::HashMap<i64, Arc<core::GrayImage>>>>,
}

/// Storage backend name, set once at startup for /v1/system/info.
static APP_STORAGE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

type ApiResult<T> = Result<T, ApiError>;

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    detail: String,
}

impl ApiError {
    fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Self { status, detail: detail.into() }
    }
    fn bad(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, detail)
    }
    fn not_found(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, detail)
    }
    fn internal(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, detail)
    }
    fn unprocessable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, detail)
    }
}

#[derive(Serialize)]
struct ErrBody {
    error: String,
    detail: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status_name = self.status.canonical_reason().unwrap_or("error").to_string();
        (self.status, Json(ErrBody { error: status_name, detail: self.detail })).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::internal(format!("{e:#}"))
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        Self::internal(e.to_string())
    }
}

/// Run synchronous store/IO work on tokio's blocking pool so it never
/// stalls an async worker thread.
async fn blocking<T, F>(f: F) -> ApiResult<T>
where
    F: FnOnce() -> ApiResult<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
}

// ---------- request/response DTOs ----------

#[derive(Deserialize)]
struct ProjectCreate {
    name: String,
    description: Option<String>,
}

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default)]
    skip: i64,
    #[serde(default = "default_limit")]
    limit: i64,
}
fn default_limit() -> i64 {
    100
}

#[derive(Deserialize)]
struct CompareRequest {
    algorithm: Option<String>,
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default)]
    rotation_invariant: bool,
}
fn default_threshold() -> f64 {
    0.85
}

#[derive(Deserialize)]
struct SmartCompareRequest {
    #[serde(default = "default_smart_threshold")]
    threshold: f64,
    #[serde(default = "default_min_agree")]
    min_agree: usize,
}
fn default_smart_threshold() -> f64 {
    0.92
}
fn default_min_agree() -> usize {
    3
}

#[derive(Deserialize)]
struct DedupRequest {
    /// Hamming radius for index recall (64-bit hashes).
    #[serde(default = "default_dedup_radius")]
    radius: u32,
    #[serde(default = "default_threshold")]
    threshold: f64,
    /// Gate hashes that must flag a pair for it to be verified.
    #[serde(default = "default_dedup_votes")]
    min_votes: u32,
}
fn default_dedup_radius() -> u32 {
    10
}
fn default_dedup_votes() -> u32 {
    2
}

#[derive(Deserialize)]
struct MatchRequest {
    image_a_id: i64,
    image_b_id: i64,
    #[serde(default = "default_match_algo")]
    algorithm: String,
}
fn default_match_algo() -> String {
    "orb".into()
}

#[derive(Deserialize)]
struct SliceRequest {
    image_a_id: i64,
    image_b_id: i64,
    #[serde(default = "default_grid")]
    rows: u32,
    #[serde(default = "default_grid")]
    cols: u32,
}
fn default_grid() -> u32 {
    2
}

#[derive(Deserialize)]
struct MatrixQuery {
    #[serde(default = "default_matrix_algo")]
    algorithm: String,
    #[serde(default)]
    rotation_invariant: bool,
}
fn default_matrix_algo() -> String {
    "phash".into()
}

#[derive(Deserialize)]
struct ReportQuery {
    #[serde(default = "default_matrix_algo")]
    algorithm: String,
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default)]
    rotation_invariant: bool,
}

#[derive(Deserialize)]
struct RunsQuery {
    project_id: i64,
    #[serde(default)]
    skip: i64,
    #[serde(default = "default_limit")]
    limit: i64,
}

#[derive(Deserialize)]
struct ThumbQuery {
    #[serde(default = "default_thumb")]
    size: u32,
}
fn default_thumb() -> u32 {
    400
}

// ---------- helpers ----------

fn validate_algorithm(a: &str) -> ApiResult<()> {
    if core::is_known_algorithm(a) {
        Ok(())
    } else {
        Err(ApiError::bad(format!(
            "不支持的比对算法: {a}。支持: {}",
            core::all_algorithms().join(",")
        )))
    }
}

/// Spawn background feature precomputation for one image already decoded
/// (the bytes were just written by the upload path — no second read/decode).
fn enqueue_precompute_decoded(state: &AppState, image_id: i64, img: image::DynamicImage) {
    let store = state.store.clone();
    tokio::task::spawn_blocking(move || {
        run_precompute(&store, image_id, || Ok(img))
    });
}

/// Spawn background feature precomputation reading the blob back
/// (recompute path for images whose decode happened before upload-time).
fn enqueue_precompute(state: &AppState, image_id: i64, key: String) {
    let store = state.store.clone();
    tokio::task::spawn_blocking(move || {
        run_precompute(&store, image_id, || {
            core::image_io::decode(&store.read_file(&key)?)
        })
    });
}

fn run_precompute(
    store: &Arc<itrace_store::Store>,
    image_id: i64,
    get_img: impl FnOnce() -> anyhow::Result<image::DynamicImage>,
) {
    if let Err(e) = store.set_feature_status(image_id, "computing") {
        tracing::warn!("status update failed: {e}");
        return;
    }
    let res = (|| -> anyhow::Result<()> {
        let img = get_img()?;
        let rows = core::features::compute_all_variants(&img);
        // one batched write instead of ~112 individual upserts
        let refs: Vec<(u8, &str, &[u8], usize)> = rows
            .iter()
            .map(|(v, n, b, d)| (*v, n.as_str(), b.as_slice(), *d))
            .collect();
        store.put_features(image_id, &refs)
    })();
    match res {
        Ok(()) => {
            let _ = store.set_feature_status(image_id, "ready");
        }
        Err(e) => {
            tracing::warn!("precompute failed for image {image_id}: {e:#}");
            let _ = store.set_feature_status(image_id, "pending");
        }
    }
}

fn sanitize_filename(name: &str) -> ApiResult<String> {
    let file = std::path::Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim()
        .replace('\0', "");
    if file.is_empty() || file == "." || file == ".." {
        return Err(ApiError::bad("文件名无效"));
    }
    Ok(file)
}

/// Unique blob key under a prefix ("uploads", "extracted", "thumbnails").
fn unique_key(store: &Store, prefix: &str, name: &str) -> String {
    let candidate = format!("{prefix}/{name}");
    if !store.file_exists(&candidate) {
        return candidate;
    }
    let stem = std::path::Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();
    let ext = std::path::Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| format!(".{s}"))
        .unwrap_or_default();
    for i in 1.. {
        let c = format!("{prefix}/{stem}_{i}{ext}");
        if !store.file_exists(&c) {
            return c;
        }
    }
    unreachable!()
}

// ---------- handlers ----------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

async fn system_info() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "rust-native",
        "storage": *APP_STORAGE.get().unwrap_or(&"unknown"),
        "algorithms": core::all_algorithms(),
        "features": {
            "akaze": cfg!(feature = "akaze"),
            "deep_embedding": false,
            "pdf_page_render": false,
        }
    }))
}

async fn create_project(
    State(s): State<AppState>,
    Json(body): Json<ProjectCreate>,
) -> ApiResult<impl IntoResponse> {
    if body.name.trim().is_empty() {
        return Err(ApiError::bad("项目名称不能为空"));
    }
    blocking(move || {
        let p = s.store.create_project(&body.name, body.description.as_deref())?;
        Ok((StatusCode::CREATED, Json(p)))
    })
    .await
}

async fn list_projects(
    State(s): State<AppState>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<itrace_store::Project>>> {
    blocking(move || Ok(Json(s.store.list_projects(q.skip, q.limit.min(500))?))).await
}

async fn get_project(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<itrace_store::Project>> {
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在")).map(Json)
    })
    .await
}

async fn delete_project(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    blocking(move || {
        let images = s.store.delete_project(id).map_err(|e| {
            if e.to_string().contains("不存在") {
                ApiError::not_found("项目不存在")
            } else {
                ApiError::from(e)
            }
        })?;
        for img in images {
            let _ = s.store.delete_file(&img.file_path);
            if let Ok(thumbs) = s.store.blobs().list(&format!("thumbnails/{}_", img.id)) {
                for t in thumbs {
                    let _ = s.store.delete_file(&t);
                }
            }
        }
        Ok(Json(serde_json::json!({"message": "项目已删除"})))
    })
    .await
}

async fn list_images(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<ImageRecord>>> {
    let (skip, limit) = (q.skip, q.limit.min(500));
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        Ok(Json(s.store.list_images(id, skip, limit)?))
    })
    .await
}

async fn delete_image(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    blocking(move || {
        let rec = s.store.delete_image(id).map_err(|_| ApiError::not_found("图像不存在"))?;
        let _ = s.store.delete_file(&rec.file_path);
        if let Ok(thumbs) = s.store.blobs().list(&format!("thumbnails/{id}_")) {
            for t in thumbs {
                let _ = s.store.delete_file(&t);
            }
        }
        Ok(Json(serde_json::json!({"message": "图像已删除"})))
    })
    .await
}

async fn get_thumbnail(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ThumbQuery>,
) -> ApiResult<impl IntoResponse> {
    let size = q.size.clamp(16, 2048);
    let bytes = blocking(move || {
        let rec = s.store.get_image(id).map_err(|_| ApiError::not_found("图像不存在"))?;
        let thumb_key = format!("thumbnails/{id}_{size}.jpg");
        if !s.store.file_exists(&thumb_key) {
            let img = core::image_io::decode(&s.store.read_file(&rec.file_path)?)?;
            let th = core::image_io::resize_max_side(&img, size);
            let rgb = core::image_io::to_rgb(&th);
            s.store.write_file(&thumb_key, &core::image_io::encode_jpeg(&rgb, 85)?)?;
        }
        Ok(s.store.read_file(&thumb_key)?)
    })
    .await?;
    Ok(([(axum::http::header::CONTENT_TYPE, "image/jpeg")], bytes))
}

async fn upload(
    State(s): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Json<serde_json::Value>> {
    let mut project_id: Option<i64> = None;
    let mut filename: Option<String> = None;
    let mut data: Option<Vec<u8>> = None;
    while let Some(field) = multipart.next_field().await.map_err(|e| ApiError::bad(e.to_string()))? {
        match field.name() {
            Some("project_id") => {
                project_id = field.text().await.ok().and_then(|t| t.parse().ok());
            }
            Some("file") => {
                filename = field.file_name().map(|s| s.to_string());
                data = Some(field.bytes().await.map_err(|e| ApiError::bad(e.to_string()))?.to_vec());
            }
            _ => {}
        }
    }
    let project_id = project_id.ok_or_else(|| ApiError::bad("缺少 project_id"))?;
    let filename = sanitize_filename(&filename.ok_or_else(|| ApiError::bad("文件名为空"))?)?;
    let data = data.ok_or_else(|| ApiError::bad("缺少文件内容"))?;

    blocking(move || {
        s.store.get_project(project_id).map_err(|_| ApiError::not_found("项目不存在"))?;
        service::handle_upload(&s, project_id, &filename, data)
    })
    .await
    .map(Json) // handle_upload already returns ApiResult
}

async fn feature_status(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        let items: Vec<serde_json::Value> = images
            .iter()
            .map(|i| {
                serde_json::json!({"id": i.id, "filename": i.filename, "status": i.feature_status})
            })
            .collect();
        let ready = items.iter().filter(|i| i["status"] == "ready").count();
        Ok(Json(serde_json::json!({
            "project_id": id, "total": items.len(), "ready": ready,
            "all_ready": ready == items.len() && !items.is_empty(), "images": items
        })))
    })
    .await
}

async fn recompute_features(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    let st = s.clone();
    let pending = blocking(move || {
        st.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = st.store.list_images(id, 0, i64::MAX)?;
        Ok(images
            .into_iter()
            .filter(|i| i.feature_status != "ready" && st.store.file_exists(&i.file_path))
            .collect::<Vec<_>>())
    })
    .await?;
    let triggered = pending.len();
    for img in pending {
        enqueue_precompute(&s, img.id, img.file_path);
    }
    Ok(Json(serde_json::json!({
        "triggered": triggered,
        "message": format!("已触发 {triggered} 张图片的特征计算")
    })))
}

async fn compare(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Json(body): Json<CompareRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let algo = body.algorithm.unwrap_or_else(|| "phash".into());
    validate_algorithm(&algo)?;
    if !(0.0..=1.0).contains(&body.threshold) {
        return Err(ApiError::bad("阈值必须在0-1之间"));
    }
    let (threshold, rot) = (body.threshold, body.rotation_invariant);
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        Ok(Json(service::run_compare(&s, &images, &algo, threshold, rot)?))
    })
    .await
}

async fn smart_compare(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    body: Option<Json<SmartCompareRequest>>,
) -> ApiResult<Json<serde_json::Value>> {
    let body = body.map(|b| b.0).unwrap_or(SmartCompareRequest {
        threshold: default_smart_threshold(),
        min_agree: default_min_agree(),
    });
    let (threshold, min_agree) = (body.threshold, body.min_agree);
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        Ok(Json(service::run_smart_compare(&s, &images, threshold, min_agree)?))
    })
    .await
}

async fn dedup(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    body: Option<Json<DedupRequest>>,
) -> ApiResult<Json<serde_json::Value>> {
    let body = body.map(|b| b.0).unwrap_or(DedupRequest {
        radius: default_dedup_radius(),
        threshold: default_threshold(),
        min_votes: default_dedup_votes(),
    });
    let (radius, threshold, min_votes) = (body.radius, body.threshold, body.min_votes);
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        Ok(Json(service::run_dedup_scan(&s, &images, radius, threshold, min_votes)?))
    })
    .await
}

async fn matrix(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MatrixQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_algorithm(&q.algorithm)?;
    let (algo, rot) = (q.algorithm, q.rotation_invariant);
    blocking(move || {
        s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        Ok(Json(service::pairwise_matrix(&s, &images, &algo, rot)?))
    })
    .await
}

async fn report(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ReportQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_algorithm(&q.algorithm)?;
    let (algo, threshold, rot) = (q.algorithm, q.threshold, q.rotation_invariant);
    blocking(move || {
        let project = s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
        let images = s.store.list_images(id, 0, i64::MAX)?;
        Ok(Json(service::build_report(&s, &project, &images, &algo, threshold, rot)?))
    })
    .await
}

async fn list_runs(
    State(s): State<AppState>,
    Query(q): Query<RunsQuery>,
) -> ApiResult<Json<Vec<itrace_store::AnalysisRunRecord>>> {
    blocking(move || {
        s.store.get_project(q.project_id).map_err(|_| ApiError::not_found("项目不存在"))?;
        Ok(Json(s.store.list_runs(q.project_id, q.skip, q.limit.min(500))?))
    })
    .await
}

async fn get_run(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    blocking(move || {
        let run = s.store.get_run(id).map_err(|_| ApiError::not_found("分析记录不存在"))?;
        let parsed: Option<serde_json::Value> =
            run.summary.as_deref().and_then(|s| serde_json::from_str(s).ok());
        Ok(Json(serde_json::json!({"run": run, "result": parsed})))
    })
    .await
}

async fn match_pairs(
    State(s): State<AppState>,
    Json(body): Json<MatchRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let algo = body.algorithm.to_lowercase();
    if !DESCRIPTOR_ALGOS.contains(&algo.as_str()) {
        return Err(ApiError::bad(format!("仅支持描述子算法: {}", DESCRIPTOR_ALGOS.join(","))));
    }
    let (a, b) = (body.image_a_id, body.image_b_id);
    blocking(move || {
        let ia = s.store.get_image(a).map_err(|_| ApiError::not_found("图像不存在"))?;
        let ib = s.store.get_image(b).map_err(|_| ApiError::not_found("图像不存在"))?;
        service::match_data(&s, &ia, &ib, &algo)
    })
    .await
    .map(Json)
}

async fn visualize_match(
    State(s): State<AppState>,
    Json(body): Json<MatchRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let algo = body.algorithm.to_lowercase();
    if !DESCRIPTOR_ALGOS.contains(&algo.as_str()) {
        return Err(ApiError::bad(format!("仅支持描述子算法: {}", DESCRIPTOR_ALGOS.join(","))));
    }
    let (a, b) = (body.image_a_id, body.image_b_id);
    blocking(move || {
        let ia = s.store.get_image(a).map_err(|_| ApiError::not_found("图像不存在"))?;
        let ib = s.store.get_image(b).map_err(|_| ApiError::not_found("图像不存在"))?;
        service::visualize(&s, &ia, &ib, &algo)
    })
    .await
    .map(Json)
}

async fn slice_match(
    State(s): State<AppState>,
    Json(body): Json<SliceRequest>,
) -> ApiResult<Json<core::slice::SliceMatchResult>> {
    let (a, b) = (body.image_a_id, body.image_b_id);
    let rows = body.rows.clamp(1, 8);
    let cols = body.cols.clamp(1, 8);
    blocking(move || {
        let ia = s.store.get_image(a).map_err(|_| ApiError::not_found("图像不存在"))?;
        let ib = s.store.get_image(b).map_err(|_| ApiError::not_found("图像不存在"))?;
        Ok(Json(service::slice_match(&s, &ia, &ib, rows, cols)?))
    })
    .await
}

/// Blob prefixes a client may download — anything else (e.g. the SQLite DB
/// file under the fs backend) is off-limits.
const DOWNLOAD_PREFIXES: &[&str] = &["uploads/", "extracted/", "thumbnails/", "visualizations/"];

async fn download_file(
    State(s): State<AppState>,
    Path(path): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let rel = path.strip_prefix("data/").unwrap_or(&path).to_string();
    if !DOWNLOAD_PREFIXES.iter().any(|p| rel.starts_with(p)) {
        return Err(ApiError::not_found("文件不存在"));
    }
    let mime = match rel.rsplit('.').next().unwrap_or("") {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    };
    let bytes =
        blocking(move || s.store.read_file(&rel).map_err(|_| ApiError::not_found("文件不存在")))
            .await?;
    Ok(([(axum::http::header::CONTENT_TYPE, mime)], bytes))
}

// ---------- main ----------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string());
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000);

    let store = Store::open(std::path::Path::new(&data_dir))?;
    let _ = APP_STORAGE.set(store.storage_kind());
    tracing::info!("storage backend: {}", store.storage_kind());
    let state = AppState { store: Arc::new(store), gray_cache: Default::default() };

    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/system/info", get(system_info))
        .route("/v1/projects", get(list_projects).post(create_project))
        .route("/v1/projects/{id}", get(get_project).delete(delete_project))
        .route("/v1/projects/{id}/images", get(list_images))
        .route("/v1/images/{id}", delete(delete_image))
        .route("/v1/images/{id}/thumbnail", get(get_thumbnail))
        .route("/v1/upload", post(upload))
        .route("/v1/projects/{id}/feature-status", get(feature_status))
        .route("/v1/projects/{id}/features/recompute", post(recompute_features))
        .route("/v1/projects/{id}/compare", post(compare))
        .route("/v1/projects/{id}/smart-compare", post(smart_compare))
        .route("/v1/projects/{id}/dedup", post(dedup))
        .route("/v1/projects/{id}/matrix", get(matrix))
        .route("/v1/projects/{id}/report", get(report))
        .route("/v1/analysis-runs", get(list_runs))
        .route("/v1/analysis-runs/{id}", get(get_run))
        .route("/v1/match/pairs", post(match_pairs))
        .route("/v1/match/visualize", post(visualize_match))
        .route("/v1/match/slices", post(slice_match))
        .route("/v1/files/{*path}", get(download_file))
        .nest_service("/static", ServeDir::new(state.store.data_dir()))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!("Image Trace RS listening on http://{addr}  (data: {data_dir})");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
