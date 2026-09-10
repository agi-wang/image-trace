//! Image Trace RS — axum HTTP API implementing docs/openapi.yaml.

mod draw;
mod service;

use std::net::SocketAddr;
use std::path::PathBuf;
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
}

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

/// Spawn background feature precomputation for one image.
fn enqueue_precompute(state: &AppState, image_id: i64, path: PathBuf) {
    let store = state.store.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = store.set_feature_status(image_id, "computing") {
            tracing::warn!("status update failed: {e}");
            return;
        }
        let res = (|| -> anyhow::Result<()> {
            let img = core::image_io::decode_file(&path)?;
            let rows = core::features::compute_all_variants(&img);
            for (variant, name, bytes, dims) in rows {
                store.put_feature(image_id, variant, &name, &bytes, dims)?;
            }
            Ok(())
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
    });
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

fn unique_path(dir: &std::path::Path, name: &str) -> PathBuf {
    let candidate = dir.join(name);
    if !candidate.exists() {
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
        let c = dir.join(format!("{stem}_{i}{ext}"));
        if !c.exists() {
            return c;
        }
    }
    unreachable!()
}

fn rel_to_data(store: &Store, path: &std::path::Path) -> String {
    path.strip_prefix(store.data_dir())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

// ---------- handlers ----------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

async fn system_info() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "rust-native",
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
    let p = s.store.create_project(&body.name, body.description.as_deref())?;
    Ok((StatusCode::CREATED, Json(p)))
}

async fn list_projects(
    State(s): State<AppState>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<itrace_store::Project>>> {
    Ok(Json(s.store.list_projects(q.skip, q.limit.min(500))?))
}

async fn get_project(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<itrace_store::Project>> {
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在")).map(Json)
}

async fn delete_project(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    let images = s.store.delete_project(id).map_err(|e| {
        if e.to_string().contains("不存在") {
            ApiError::not_found("项目不存在")
        } else {
            ApiError::from(e)
        }
    })?;
    for img in images {
        if let Some(p) = s.store.resolve(&img.file_path) {
            let _ = std::fs::remove_file(p);
        }
    }
    Ok(Json(serde_json::json!({"message": "项目已删除"})))
}

async fn list_images(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<ImageRecord>>> {
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    Ok(Json(s.store.list_images(id, q.skip, q.limit.min(500))?))
}

async fn delete_image(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    let rec = s.store.delete_image(id).map_err(|_| ApiError::not_found("图像不存在"))?;
    if let Some(p) = s.store.resolve(&rec.file_path) {
        let _ = std::fs::remove_file(&p);
    }
    // thumbnails
    let thumbs = s.store.data_dir().join("thumbnails");
    if let Ok(rd) = std::fs::read_dir(&thumbs) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with(&format!("{id}_")) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(Json(serde_json::json!({"message": "图像已删除"})))
}

async fn get_thumbnail(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ThumbQuery>,
) -> ApiResult<impl IntoResponse> {
    let size = q.size.clamp(16, 2048);
    let rec = s.store.get_image(id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let thumb_dir = s.store.data_dir().join("thumbnails");
    let thumb_path = thumb_dir.join(format!("{id}_{size}.jpg"));
    if !thumb_path.exists() {
        let src = s
            .store
            .resolve(&rec.file_path)
            .filter(|p| p.exists())
            .ok_or_else(|| ApiError::not_found("图像文件不存在"))?;
        let path = thumb_path.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let img = core::image_io::decode_file(&src)?;
            let th = core::image_io::resize_max_side(&img, size);
            let rgb = core::image_io::to_rgb(&th);
            core::image_io::save_jpeg(&rgb, &path, 85)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;
    }
    let bytes = std::fs::read(&thumb_path)?;
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

    s.store.get_project(project_id).map_err(|_| ApiError::not_found("项目不存在"))?;

    let state = s.clone();
    tokio::task::spawn_blocking(move || {
        service::handle_upload(&state, project_id, &filename, data)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map(Json)
}

async fn feature_status(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let items: Vec<serde_json::Value> = images
        .iter()
        .map(|i| serde_json::json!({"id": i.id, "filename": i.filename, "status": i.feature_status}))
        .collect();
    let ready = items.iter().filter(|i| i["status"] == "ready").count();
    Ok(Json(serde_json::json!({
        "project_id": id, "total": items.len(), "ready": ready,
        "all_ready": ready == items.len() && !items.is_empty(), "images": items
    })))
}

async fn recompute_features(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let mut triggered = 0;
    for img in images {
        if img.feature_status != "ready" {
            if let Some(p) = s.store.resolve(&img.file_path).filter(|p| p.exists()) {
                enqueue_precompute(&s, img.id, p);
                triggered += 1;
            }
        }
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
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let state = s.clone();
    let threshold = body.threshold;
    let rot = body.rotation_invariant;
    let result = tokio::task::spawn_blocking(move || {
        service::run_compare(&state, &images, &algo, threshold, rot)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(result))
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
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let state = s.clone();
    let result = tokio::task::spawn_blocking(move || {
        service::run_smart_compare(&state, &images, body.threshold, body.min_agree)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(result))
}

async fn matrix(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MatrixQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_algorithm(&q.algorithm)?;
    s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let state = s.clone();
    let algo = q.algorithm.clone();
    let rot = q.rotation_invariant;
    let result = tokio::task::spawn_blocking(move || {
        service::pairwise_matrix(&state, &images, &algo, rot)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(result))
}

async fn report(
    State(s): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<ReportQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_algorithm(&q.algorithm)?;
    let project = s.store.get_project(id).map_err(|_| ApiError::not_found("项目不存在"))?;
    let images = s.store.list_images(id, 0, i64::MAX)?;
    let state = s.clone();
    let algo = q.algorithm.clone();
    let result = tokio::task::spawn_blocking(move || {
        service::build_report(&state, &project, &images, &algo, q.threshold, q.rotation_invariant)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(result))
}

async fn list_runs(
    State(s): State<AppState>,
    Query(q): Query<RunsQuery>,
) -> ApiResult<Json<Vec<itrace_store::AnalysisRunRecord>>> {
    s.store.get_project(q.project_id).map_err(|_| ApiError::not_found("项目不存在"))?;
    Ok(Json(s.store.list_runs(q.project_id, q.skip, q.limit.min(500))?))
}

async fn get_run(
    State(s): State<AppState>,
    Path(id): Path<i64>,
) -> ApiResult<Json<serde_json::Value>> {
    let run = s.store.get_run(id).map_err(|_| ApiError::not_found("分析记录不存在"))?;
    let parsed: Option<serde_json::Value> = run
        .summary
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(Json(serde_json::json!({"run": run, "result": parsed})))
}

async fn match_pairs(
    State(s): State<AppState>,
    Json(body): Json<MatchRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    let algo = body.algorithm.to_lowercase();
    if !DESCRIPTOR_ALGOS.contains(&algo.as_str()) {
        return Err(ApiError::bad(format!("仅支持描述子算法: {}", DESCRIPTOR_ALGOS.join(","))));
    }
    let ia = s.store.get_image(body.image_a_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let ib = s.store.get_image(body.image_b_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let state = s.clone();
    tokio::task::spawn_blocking(move || service::match_data(&state, &ia, &ib, &algo))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
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
    let ia = s.store.get_image(body.image_a_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let ib = s.store.get_image(body.image_b_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let state = s.clone();
    tokio::task::spawn_blocking(move || service::visualize(&state, &ia, &ib, &algo))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map(Json)
}

async fn slice_match(
    State(s): State<AppState>,
    Json(body): Json<SliceRequest>,
) -> ApiResult<Json<core::slice::SliceMatchResult>> {
    let ia = s.store.get_image(body.image_a_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let ib = s.store.get_image(body.image_b_id).map_err(|_| ApiError::not_found("图像不存在"))?;
    let rows = body.rows.clamp(1, 8);
    let cols = body.cols.clamp(1, 8);
    let state = s.clone();
    let result = tokio::task::spawn_blocking(move || {
        service::slice_match(&state, &ia, &ib, rows, cols)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(result))
}

async fn download_file(
    State(s): State<AppState>,
    Path(path): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let rel = path.strip_prefix("data/").unwrap_or(&path);
    let full = s
        .store
        .resolve(rel)
        .filter(|p| p.is_file())
        .ok_or_else(|| ApiError::not_found("文件不存在"))?;
    let bytes = tokio::fs::read(&full).await?;
    let mime = match full.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    };
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
    let state = AppState { store: Arc::new(store) };

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
