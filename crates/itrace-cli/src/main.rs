//! itrace — offline CLI for Image Trace.
//! Operates directly on the store; for the HTTP API run `itrace-server`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use rayon::prelude::*;

use itrace_core::compare::{self, Prepared};
use itrace_core::{documents, hashes, image_io, features};
use itrace_store::{NewImage, Store};

#[derive(Parser)]
#[command(name = "itrace", version, about = "Image Trace — 图像比对与查重")]
struct Cli {
    /// 数据目录（含 image-trace.db 与 uploads/extracted/）
    #[arg(long, global = true, default_value = "data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 创建项目
    Create { name: String, #[arg(long)] description: Option<String> },
    /// 项目列表
    Projects,
    /// 向项目添加文件（图片或 PDF/DOCX/PPTX 文档）
    Add { project_id: i64, files: Vec<PathBuf> },
    /// 项目内图片列表
    Images { project_id: i64 },
    /// 预计算特征（add 已自动执行；用于补算）
    Precompute { project_id: i64 },
    /// 单算法比对
    Compare {
        project_id: i64,
        #[arg(long, default_value = "phash")]
        algorithm: String,
        #[arg(long, default_value = "0.85")]
        threshold: f64,
        #[arg(long)]
        rotation_invariant: bool,
    },
    /// 智能查重（多算法投票）
    Smart {
        project_id: i64,
        #[arg(long, default_value = "0.92")]
        threshold: f64,
        #[arg(long, default_value = "3")]
        min_agree: usize,
    },
    /// 重复图片报告
    Report {
        project_id: i64,
        #[arg(long, default_value = "phash")]
        algorithm: String,
        #[arg(long, default_value = "0.85")]
        threshold: f64,
    },
    /// 两图切片/子图检测
    Slice {
        image_a: i64,
        image_b: i64,
        #[arg(long, default_value = "2")]
        rows: u32,
        #[arg(long, default_value = "2")]
        cols: u32,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let store = Store::open(&cli.data_dir).context("打开数据目录失败")?;

    match cli.cmd {
        Cmd::Create { name, description } => {
            let p = store.create_project(&name, description.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&p)?);
        }
        Cmd::Projects => {
            println!("{}", serde_json::to_string_pretty(&store.list_projects(0, 500)?)?);
        }
        Cmd::Add { project_id, files } => {
            store.get_project(project_id).context("项目不存在")?;
            for f in files {
                add_file(&store, project_id, &f)?;
            }
        }
        Cmd::Images { project_id } => {
            println!("{}", serde_json::to_string_pretty(&store.list_images(project_id, 0, i64::MAX)?)?);
        }
        Cmd::Precompute { project_id } => {
            let images = store.list_images(project_id, 0, i64::MAX)?;
            // decode + feature compute in parallel; DB writes stay sequential
            let computed: Vec<_> = images
                .par_iter()
                .filter(|i| i.feature_status != "ready")
                .map(|i| (i.id, compute_rows(&store, &i.file_path)))
                .collect();
            for (id, rows) in computed {
                if let Err(e) = write_features(&store, id, rows) {
                    eprintln!("image {id} precompute failed: {e}");
                }
            }
            println!("done");
        }
        Cmd::Compare { project_id, algorithm, threshold, rotation_invariant } => {
            let images = store.list_images(project_id, 0, i64::MAX)?;
            let desc_algos = compare::desc_algos_for(&algorithm);
            // Keep original positions — members index `prepared`, not `images`
            let (idx, prepared): (Vec<usize>, Vec<Prepared>) = images
                .par_iter()
                .enumerate()
                .filter_map(|(k, img)| {
                    let bytes = store.read_file(&img.file_path).ok()?;
                    Prepared::from_bytes(&bytes, &desc_algos, rotation_invariant)
                        .ok()
                        .map(|p| (k, p))
                })
                .unzip();
            let (groups, ungrouped, _m) =
                compare::analyze(&prepared, &algorithm, threshold, rotation_invariant);
            println!("{} images, {} groups, {} unique", images.len(), groups.len(), ungrouped.len());
            for (gi, members) in groups.iter().enumerate() {
                println!("group {}:", gi + 1);
                for &m in members {
                    println!("  - {} ({})", images[idx[m]].filename, images[idx[m]].id);
                }
            }
        }
        Cmd::Smart { project_id, threshold, min_agree } => {
            let images = store.list_images(project_id, 0, i64::MAX)?;
            let n = images.len();
            if n < 2 {
                println!("图片不足");
                return Ok(());
            }
            let ids: Vec<i64> = images.iter().map(|i| i.id).collect();
            if !store.features_ready(&ids)? {
                eprintln!("特征尚未就绪，先运行 `itrace precompute {project_id}`");
                return Ok(());
            }
            let t0 = std::time::Instant::now();
            let variants: Vec<u8> = (0..features::NUM_VARIANTS).collect();
            let mut pair_hits: std::collections::HashMap<(usize, usize), Vec<String>> =
                std::collections::HashMap::new();
            // deduped feature list — one batched load covers every algo
            let mut feat_names: Vec<&'static str> = Vec::new();
            let algo_feat: Vec<Option<usize>> = itrace_core::SMART_ALGOS
                .iter()
                .map(|a| {
                    features::algo_to_feature(a).map(|f| {
                        match feat_names.iter().position(|&x| x == f) {
                            Some(i) => i,
                            None => {
                                feat_names.push(f);
                                feat_names.len() - 1
                            }
                        }
                    })
                })
                .collect();
            let maps = store.load_feature_maps(&ids, &feat_names, &variants)?;
            for (algo, fi) in itrace_core::SMART_ALGOS.iter().zip(&algo_feat) {
                let Some(fi) = *fi else { continue };
                let map = &maps[fi];
                if map.is_empty() {
                    continue;
                }
                // Same scoring as similarity_matrix, streamed — no N×N.
                let pairs =
                    features::similarity_pairs_above(map, &ids, algo, true, threshold);
                for (i, j, _s) in pairs {
                    pair_hits.entry((i, j)).or_default().push(algo.to_string());
                }
            }
            let confirmed: Vec<(usize, usize)> = pair_hits
                .iter()
                .filter(|(_, hits)| {
                    hits.len() >= min_agree
                        && hits.iter().any(|a| itrace_core::HASH_GATE_ALGOS.contains(&a.as_str()))
                })
                .map(|(&k, _)| k)
                .collect();
            let groups = compare::components_from_pairs(n, &confirmed);
            let mut shown = 0;
            for members in groups {
                if members.len() < 2 {
                    continue;
                }
                shown += 1;
                println!("duplicate group {shown}:");
                for &m in &members {
                    println!("  - {} ({})", images[m].filename, images[m].id);
                }
            }
            println!("scan: {:.2}s, {} dup groups", t0.elapsed().as_secs_f64(), shown);
        }
        Cmd::Report { project_id, algorithm, threshold } => {
            let images = store.list_images(project_id, 0, i64::MAX)?;
            let desc_algos = compare::desc_algos_for(&algorithm);
            let prepared: Vec<Prepared> = images
                .par_iter()
                .filter_map(|img| {
                    let bytes = store.read_file(&img.file_path).ok()?;
                    Prepared::from_bytes(&bytes, &desc_algos, false).ok()
                })
                .collect();
            let m = compare::pairwise_matrix(&prepared, &algorithm, false);
            let (groups, ungrouped) = itrace_core::group::cluster(&m, threshold);
            println!("{} images, {} similar groups, {} unique", images.len(), groups.len(), ungrouped.len());
            for row in &m {
                let line: Vec<String> = row.iter().map(|v| format!("{v:.2}")).collect();
                println!("{}", line.join(" "));
            }
        }
        Cmd::Slice { image_a, image_b, rows, cols } => {
            let ia = store.get_image(image_a)?;
            let ib = store.get_image(image_b)?;
            let ga = image_io::to_gray(&image_io::decode(&store.read_file(&ia.file_path)?)?);
            let gb = image_io::to_gray(&image_io::decode(&store.read_file(&ib.file_path)?)?);
            let res = itrace_core::slice::slice_match(&ga, &gb, rows, cols, 0.7);
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
    }
    Ok(())
}

fn add_file(store: &Store, project_id: i64, path: &PathBuf) -> anyhow::Result<()> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    if image_io::is_supported_image(name) {
        let data = std::fs::read(path)?;
        let rel = unique_key(store, "uploads", name);
        store.write_file(&rel, &data)?;
        // Decode once — the insert-time hashes and precompute share it
        // (previously compute_image_features_bytes + precompute decoded
        // twice and re-read the blob).
        let img = image_io::decode(&data)?;
        let feats = hashes::compute_image_features_decoded(
            &img,
            hashes::blake3_hex(&data),
            data.len() as u64,
        );
        let rec = store.insert_image(&NewImage {
            project_id,
            filename: name.to_string(),
            file_path: rel,
            file_hash: feats.file_hash,
            phash: Some(hashes::to_hex(feats.hashes.phash)),
            dhash: Some(hashes::to_hex(feats.hashes.dhash)),
            ahash: Some(hashes::to_hex(feats.hashes.ahash)),
            whash: Some(hashes::to_hex(feats.hashes.whash)),
            colorhash: Some(hashes::to_hex(feats.hashes.colorhash)),
            extracted_from: None,
            file_size: Some(feats.file_size as i64),
            width: Some(feats.width as i64),
            height: Some(feats.height as i64),
        })?;
        precompute_decoded(store, rec.id, &img)?;
        println!("+ {} -> image id {}", name, rec.id);
        return Ok(());
    }
    if image_io::is_supported_document(name) {
        let data = std::fs::read(path)?;
        let rel_doc = unique_key(store, "uploads", name);
        store.write_file(&rel_doc, &data)?;
        let extracted = documents::extract_named(name, &data)?;
        for img in extracted {
            let key = unique_key(store, "extracted", &img.filename);
            store.write_file(&key, &img.data)?;
            let decoded = image_io::decode(&img.data)?;
            let feats = hashes::compute_image_features_decoded(
                &decoded,
                hashes::blake3_hex(&img.data),
                img.data.len() as u64,
            );
            let rec = store.insert_image(&NewImage {
                project_id,
                filename: img.filename.clone(),
                file_path: key,
                file_hash: feats.file_hash,
                phash: Some(hashes::to_hex(feats.hashes.phash)),
                dhash: Some(hashes::to_hex(feats.hashes.dhash)),
                ahash: Some(hashes::to_hex(feats.hashes.ahash)),
                whash: Some(hashes::to_hex(feats.hashes.whash)),
                colorhash: Some(hashes::to_hex(feats.hashes.colorhash)),
                extracted_from: Some(rel_doc.clone()),
                file_size: Some(feats.file_size as i64),
                width: Some(feats.width as i64),
                height: Some(feats.height as i64),
            })?;
            precompute_decoded(store, rec.id, &decoded)?;
            println!("+ {} -> image id {}", img.filename, rec.id);
        }
        return Ok(());
    }
    bail!("不支持的文件格式: {name}")
}

/// Computed feature rows for one image: (variant_idx, name, bytes, dims).
type FeatureRows = Vec<(u8, String, Vec<u8>, usize)>;

/// Decode `key` and compute all feature rows (CPU-heavy; no DB access).
fn compute_rows(store: &Store, key: &str) -> anyhow::Result<FeatureRows> {
    Ok(features::compute_all_variants(&image_io::decode(&store.read_file(key)?)?))
}

/// Status protocol: computing → batched write → ready / pending.
fn write_features(
    store: &Store,
    image_id: i64,
    rows: anyhow::Result<FeatureRows>,
) -> anyhow::Result<()> {
    store.set_feature_status(image_id, "computing")?;
    let res = rows.and_then(|rows| {
        let refs: Vec<(u8, &str, &[u8], usize)> =
            rows.iter().map(|(v, n, b, d)| (*v, n.as_str(), b.as_slice(), *d)).collect();
        store.put_features(image_id, &refs)
    });
    match res {
        Ok(()) => store.set_feature_status(image_id, "ready"),
        Err(e) => {
            let _ = store.set_feature_status(image_id, "pending");
            Err(e)
        }
    }
}

/// Feature rows from an already-decoded image (the `add` path decoded it
/// for the insert-time hashes — no second decode, no blob re-read).
fn precompute_decoded(
    store: &Store,
    image_id: i64,
    img: &image::DynamicImage,
) -> anyhow::Result<()> {
    write_features(store, image_id, Ok(features::compute_all_variants(img)))
}

fn unique_key(store: &Store, prefix: &str, name: &str) -> String {
    let c = format!("{prefix}/{name}");
    if !store.file_exists(&c) {
        return c;
    }
    let stem = std::path::Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("f").to_string();
    let ext = std::path::Path::new(name).extension().and_then(|s| s.to_str()).map(|e| format!(".{e}")).unwrap_or_default();
    for i in 1.. {
        let c = format!("{prefix}/{stem}_{i}{ext}");
        if !store.file_exists(&c) {
            return c;
        }
    }
    unreachable!()
}
