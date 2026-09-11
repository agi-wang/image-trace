# Image Trace RS — Rust 重写版架构

> 基于原仓库 `agi-wang/image-trace`（FastAPI + SQLModel + OpenCV + PyMuPDF）从底层重新设计。
> 目标：单二进制、无 Python/OpenCV 运行时依赖、旋转/裁剪/切片鲁棒的多算法图像查重引擎。

## 与原版的差异

| 维度 | 原版（Python） | 新版（Rust） |
|---|---|---|
| 运行时 | Python + FastAPI + Uvicorn | 单静态二进制（axum） |
| 图像解码 | Pillow + 各插件 | `image` crate（jpg/png/gif/bmp/tiff/webp/ico/qoi…） |
| 描述子 | OpenCV（ORB/BRISK/SIFT/AKAZE/KAZE） | 纯 Rust ORB + AKAZE；SIFT 预留 trait 插槽 |
| 矩阵引擎 | numpy BLAS | rayon 并行 + 位运算 XOR/popcount |
| 文档解析 | PyMuPDF + zipfile | lopdf（PDF 内嵌图）+ zip（OOXML media） |
| 存储 | SQLModel/SQLite，feature 向量 base64 TEXT | rusqlite（元数据/特征 BLOB）+ `BlobStore` 抽象：本地 fs 或 MinIO/S3 对象存储 |
| 缓存 | 进程内 LRU + SQLite 双写 | 统一走 `feature_store` + `pair_cache` 表 |
| 部署 | PyInstaller/Nuitka 打包 | `cargo build --release` 一个文件 |

## 分层

```
crates/
  itrace-core    纯计算库：解码、哈希、像素指标、ORB/AKAZE、变体、分组、切片匹配
  itrace-store   rusqlite 持久化 + BlobStore（fs | s3）：projects/images/features/pair_cache/analysis_runs
  itrace-server  axum HTTP API（docs/openapi.yaml 的实现）
  itrace-cli     clap 命令行：init/add/compare/smart/report/serve
```

## 存储后端（BlobStore）

文件负载（uploads/extracted/thumbnails/visualizations）走 `BlobStore` trait 抽象，
元数据与特征向量始终在 SQLite。`ITRACE_STORAGE` 选择后端：

- `fs`（默认）：与原版一致，`data/` 目录直存，`local_path()` 直通文件系统。
- `s3`：MinIO 或任意 S3 兼容端点（`object_store::aws`）。env：`S3_ENDPOINT`
  `S3_BUCKET` `S3_REGION` `S3_ACCESS_KEY` `S3_SECRET_KEY`（含 S3_FORCE_PATH_STYLE）。
  对象 key 与 fs 路径同构（`uploads/x.jpg`）；`local_path()` 返回 `None`，
  调用方一律走 `read_file`/`write_file`/`delete_file`。docker-compose 附带
  MinIO + 自动建桶。

## 特征提取（FeatureExtractor 注册表）

每个存储特征是一个 `FeatureExtractor` 插件（`features.rs::EXTRACTORS`）：
`compute`（gray,rgb→bytes）与 `similarity`（bytes×bytes→score）自含于
单个实现中，矩阵引擎 `generic_similarity_matrix` 统一做变体-max + rayon
并行打分。新增特征（如 DINOv2 嵌入）= 实现 trait + 加一条注册项，
无需改动比对/存储路径。算法名 → 特征经 `extractor_for_algo` 解析
（`ssim`/`template` 共享 `gray_flat`，`orb` 预筛用 `orb_pooled`）。

## 算法栈（识别能力设计）

**Tier 1 — 感知哈希**（64 bit，XOR + popcount）
- `phash`：32×32 灰度 → DCT-II → 取 8×8 低频 → 中位数阈值
- `dhash`：9×8 灰度相邻差分
- `ahash`：8×8 灰度均值阈值
- `whash`：32×32 灰度 → Haar 小波 → LL 子带中位数阈值
- `colorhash`：HSV 量化直方图签名（色相 16 bin × 饱和度 4 bin）

**Tier 2 — 像素/结构**
- `ssim`：结构相似度（均值/方差/协方差滑窗），灰度 ≤512 边长
- `histogram`：HSV H+S 二维直方图相关性（50×60 bin）
- `template`：归一化互相关 NCC

**Tier 3 — 局部特征描述子**
- `orb`：纯 Rust 实现 — FAST-9 角点 + Harris 响应筛选 + 强度质心方向 + 旋转补偿 BRIEF-256 + 交叉验证匹配
- `akaze`：`akaze` crate（可选 feature）
- `sift`：预留 `DescriptorExtractor` trait；可接 `opencv` feature 或 ONNX 模型

**Tier 4 — 变换鲁棒层（新设计，原版没有）**
- 方向变体：8 个方向（4 旋转 × 翻转态）对哈希/描述子取 max —— 识别旋转/翻转
- 切片检测 `slice_match`：B 切 R×C 网格 → 每片在 A 上滑窗 NCC + 灰度哈希 → 覆盖率判断 B 是否为 A 的切片重组或局部放大 —— 识别切片/裁剪/拼接
- 多尺度：比对时对灰度金字塔下采样重试 —— 识别缩放 + 局部截取的混合变换

**智能查重 `smart-compare`**
- 每对图片跑全部可用算法 → 超阈值记一票 → `min_agree` 票 + 至少一票来自哈希门控 → 并查集连通分组 → confidence = 该组最大得分

## 特征预计算管线

上传即入库（`feature_status=pending`）→ 后台 tokio 任务：解码一次 → 生成 8 方向变体（内存中，不落盘）→ 每变体计算全部特征 → 向量以 `BLOB` 写入 `feature_store`（image_id × variant_idx × algorithm 唯一）→ `ready`。
比对阶段直接 `SELECT` 拉取向量做批量矩阵运算，不再碰原图文件。

## 存储 schema（SQLite）

```
projects(id, name, description, created_at)
images(id, project_id→projects, filename, file_path, file_hash,
       phash, dhash, ahash, whash, colorhash,
       extracted_from, file_size, width, height, feature_status, created_at)
feature_store(id, image_id→images, variant_idx, algorithm, vector BLOB,
              dimensions, created_at, UNIQUE(image_id, variant_idx, algorithm))
pair_cache(id, hash_a, hash_b, algorithm, rotation_invariant, score, created_at)
analysis_runs(id, project_id, algorithm, threshold, total_images,
              groups_count, unique_count, summary JSON, created_at)
```

`file_hash` 改用 BLAKE3（替代 MD5）：更快、无碰撞顾虑。

## 亿级规模（10^8+）

N×N 矩阵比对只适用于中小项目；亿级走 `/v1/projects/{id}/dedup` 索引扫描：

**召回 — 多索引哈希 MIH**（`itrace-core/src/index.rs`）
- 64bit 哈希拆 8×8bit 子串建倒排表；汉明距 ≤7 的键必然共享一个完整子串（鸽巢保证召回），radius≤15 经验召回仍高。
- 每张图的全部 8 方向变体键都入索引 → 召回语义与"变体-max"比对完全等价（旋转文件的变体键集合是原图的置换）。
- 3 个门控哈希各建一索引，候选对需 `min_votes` 个**不同算法**命中（默认 2/3，位掩码去重，同算法的多变体命中不重复计票）。
- 复杂度 ≈ O(N·log bucket)；每键 ≈44B（key+owner+8表项）、每图每算法 8 键 → ≈350B/图/算法索引（1 亿图 × 3 算法 ≈ 105GB，按 key 前缀或项目分片）。

**验证** — 候选对拉取 `feature_store` 已存向量做交叉变体精确打分（(v,w) 全对取 max），阈值后并查集分组。实测 39 图：45 候选对 vs 全量 741 对，2ms。

**容量与吞吐配套**
- 元数据：SQLite 亿级行可行（索引扫描），超大可换 `PostgreSQL`/`TiKV` 槽位——`Store` trait 已隔离 SQL。
- 文件负载：S3/MinIO 分桶水平扩展；`BlobStore` 接口天然分布式。
- 更深的近似检索（float 特征 → HNSW/IVF-PQ，描述子 → FAISS/ScaNN）预留为 `FeatureExtractor` 插件 + 外部索引服务插槽。
- 写入吞吐：预计算是 CPU 密集 —— 水平扩 server 副本 + 任务队列即可线性扩展；SQLite 单写 WAL 模式下元数据写入足够，热点可切 Postgres。
