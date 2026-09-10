# Image Trace RS

`agi-wang/image-trace` 的 Rust 重写版：从底层重新设计的图像比对 / 查重引擎。

- **纯 Rust**，无 OpenCV 依赖：感知哈希、SSIM、HSV 直方图、模板 NCC、自研 ORB（FAST-9 + Harris 排序 + 质心方向 + 旋转 BRIEF-256）。
- **变换鲁棒识别**：8 个二面体方向变体（4 旋转 × 镜像）特征预计算 + 网格切片模板匹配 + 多尺度关键点金字塔 —— 旋转 / 裁剪 / 切片重组后仍可识别。
- **智能查重**：10 算法 → 阈值命中 → `min_agree` 投票 + 哈希门控 → Union-Find 连通分量分组。
- **文档提取**：DOCX/PPTX（zip media）、PDF 内嵌图（DCTDecode/FlateDecode）。
- **SQLite + WAL**：特征向量以 BLOB 存储（8 变体 × 特征），`pair_cache` 相似度缓存、`analysis_runs` 审计。
- **可插拔存储**：文件负载走 `BlobStore` 抽象 —— 本地 `fs`（默认）或 MinIO/S3（`object_store`），元数据恒在 SQLite。
- **模块化特征提取**：`FeatureExtractor` 注册表，每特征自含编码器与相似度度量；新特征（如 DINOv2）只需加一个注册项。

## 架构

```
crates/
  itrace-core    图像 IO / 哈希 / 像素指标 / 描述子 / 切片匹配 / 文档提取 / 分组
  itrace-store   SQLite 持久层 + BlobStore 文件后端（fs | s3）
  itrace-server  axum HTTP API（/v1，详见 docs/openapi.yaml）
  itrace-cli     离线 CLI（add / compare / smart / report / slice）
docs/
  openapi.yaml       OpenAPI 3.1 规范
  ARCHITECTURE.md    架构与算法分层说明
```

## 构建

```bash
cargo build --release --workspace
# 可选 AKAZE（akaze crate，纯 Rust 非线性尺度空间）
cargo build --release --features akaze -p itrace-server
```

## 运行

```bash
# HTTP 服务（默认 0.0.0.0:8000，fs 存储）
DATA_DIR=data PORT=8000 ./target/release/itrace-server

# MinIO/S3 存储：先起对象存储（docker-compose 自带 minio + 建桶 itrace）
docker compose up -d minio init-bucket
ITRACE_STORAGE=s3 S3_ENDPOINT=http://localhost:9000 \
S3_BUCKET=itrace S3_ACCESS_KEY=minioadmin S3_SECRET_KEY=minioadmin \
DATA_DIR=data PORT=8000 ./target/release/itrace-server

# CLI
itrace create "项目A"
itrace add 1 photo.jpg report.docx
itrace compare 1 --algorithm phash --threshold 0.85 --rotation-invariant
itrace smart 1 --threshold 0.92 --min-agree 3
itrace report 1
itrace slice 1 4 --rows 2 --cols 2
```

## API 速览（详见 docs/openapi.yaml）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /v1/health | 健康检查 |
| GET | /v1/system/info | 算法清单与引擎能力 |
| CRUD | /v1/projects[/{id}] | 项目 |
| GET | /v1/projects/{id}/images | 图片列表 |
| POST | /v1/upload | multipart 上传（图片或 docx/pptx/pdf，自动提取+预计算） |
| GET | /v1/projects/{id}/feature-status | 特征就绪状态 |
| POST | /v1/projects/{id}/compare | 单算法比对（rotation_invariant 可选） |
| POST | /v1/projects/{id}/smart-compare | 多算法投票智能查重 |
| GET | /v1/projects/{id}/matrix | 相似度矩阵 |
| GET | /v1/projects/{id}/report | 查重报告 |
| POST | /v1/match/pairs | 两图关键点匹配明细 |
| POST | /v1/match/visualize | 生成关键点连线可视化图 |
| POST | /v1/match/slices | 网格切片/子图检测 |
| GET | /v1/analysis-runs?project_id= | 分析历史 |

## 算法分层

1. **感知哈希层**（ahash/dhash/phash/whash/colorhash）：O(1) 汉明距离，方向变体覆盖旋转/翻转。
2. **像素指标层**：SSIM（亮度/对比度鲁棒）、HSV 直方图（几何无关）、模板 NCC（子图定位）。
3. **局部特征层**：纯 Rust ORB——8 级金字塔 FAST-9 检测、Harris 排序、质心方向分配、256bit 旋转 BRIEF、BFMatcher crossCheck / Lowe ratio。
4. **变换鲁棒层**（新增）：切片网格 × 4 旋转模板匹配（`is_slice_of_a`）、全方向变体指纹库、`contains` 子图判定。

深度嵌入（DINOv2 via ONNX）预留为 feature 插槽。

## 测试

```bash
cargo test --workspace   # 13 项端到端算法测试（合成图像 + 旋转/切片验证）
```
