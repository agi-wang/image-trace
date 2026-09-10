//! itrace — offline CLI for Image Trace.
//! Operates directly on the store; for the HTTP API run `itrace-server`.

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
            images.par_iter().for_each(|img| {
                if img.feature_status == "ready" {
                    return;
                }
                if let Some(p) = store.resolve(&img.file_path) {
                    if let Err(e) = precompute(&store, img.id, &p) {
                        eprintln!("image {} precompute failed: {e}", img.id);
                    }
                }
            });
            println!("done");
        }
        Cmd::Compare { project_id, algorithm, threshold, rotation_invariant } => {
            let images = store.list_images(project_id, 0, i64::MAX)?;
            let desc_algos = compare::desc_algos_for(&algorithm);
            let prepared: Vec<Prepared> = images
                .par_iter()
                .filter_map(|img| {
                    let p = store.resolve(&img.file_path)?;
                    Prepared::load(&p, &desc_algos, rotation_invariant).ok()
                })
                .collect();
            let (groups, ungrouped, _m) =
                compare::analyze(&prepared, &algorithm, threshold, rotation_invariant);
            println!("{} images, {} groups, {} unique", images.len(), groups.len(), ungrouped.len());
            for (gi, members) in groups.iter().enumerate() {
                println!("group {}:", gi + 1);
                for &m in members {
                    println!("  - {} ({})", images[m].filename, images[m].id);
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
            for algo in itrace_core::SMART_ALGOS {
                let Some(feat) = features::algo_to_feature(algo) else { continue };
                let map = store.load_feature_map(&ids, feat, &variants)?;
                if map.is_empty() {
                    continue;
                }
                let m = features::similarity_matrix(&map, &ids, algo, true);
                for (i, row) in m.iter().enumerate() {
                    for (j, &s) in row.iter().enumerate().skip(i + 1) {
                        if s >= threshold {
                            pair_hits.entry((i, j)).or_default().push(algo.to_string());
                        }
                    }
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
                    let p = store.resolve(&img.file_path)?;
                    Prepared::load(&p, &desc_algos, false).ok()
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
            let pa = store.resolve(&ia.file_path).context("file missing")?;
            let pb = store.resolve(&ib.file_path).context("file missing")?;
            let ga = image_io::to_gray(&image_io::decode_file(&pa)?);
            let gb = image_io::to_gray(&image_io::decode_file(&pb)?);
            let res = itrace_core::slice::slice_match(&ga, &gb, rows, cols, 0.7);
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
    }
    Ok(())
}

fn add_file(store: &Store, project_id: i64, path: &PathBuf) -> anyhow::Result<()> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    if image_io::is_supported_image(name) {
        let dest = unique_in(&store.upload_dir(), name);
        std::fs::copy(path, &dest)?;
        let feats = hashes::compute_image_features(&dest)?;
        let rel = rel(store, &dest);
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
        precompute(store, rec.id, &dest)?;
        println!("+ {} -> image id {}", name, rec.id);
        return Ok(());
    }
    if image_io::is_supported_document(name) {
        let dest = unique_in(&store.upload_dir(), name);
        std::fs::copy(path, &dest)?;
        let rel_doc = rel(store, &dest);
        let extracted = documents::extract(&dest)?;
        for img in extracted {
            let out = unique_in(&store.extract_dir(), &img.filename);
            std::fs::write(&out, &img.data)?;
            let feats = hashes::compute_image_features(&out)?;
            let rec = store.insert_image(&NewImage {
                project_id,
                filename: img.filename.clone(),
                file_path: rel(store, &out),
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
            precompute(store, rec.id, &out)?;
            println!("+ {} -> image id {}", img.filename, rec.id);
        }
        return Ok(());
    }
    bail!("不支持的文件格式: {name}")
}

fn precompute(store: &Store, image_id: i64, path: &std::path::Path) -> anyhow::Result<()> {
    store.set_feature_status(image_id, "computing")?;
    let res = (|| -> anyhow::Result<()> {
        let img = image_io::decode_file(path)?;
        for (variant, name, bytes, dims) in features::compute_all_variants(&img) {
            store.put_feature(image_id, variant, &name, &bytes, dims)?;
        }
        Ok(())
    })();
    match res {
        Ok(()) => store.set_feature_status(image_id, "ready"),
        Err(e) => {
            let _ = store.set_feature_status(image_id, "pending");
            Err(e)
        }
    }
}

fn unique_in(dir: &std::path::Path, name: &str) -> PathBuf {
    let c = dir.join(name);
    if !c.exists() {
        return c;
    }
    let stem = std::path::Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("f").to_string();
    let ext = std::path::Path::new(name).extension().and_then(|s| s.to_str()).map(|e| format!(".{e}")).unwrap_or_default();
    for i in 1.. {
        let c = dir.join(format!("{stem}_{i}{ext}"));
        if !c.exists() {
            return c;
        }
    }
    unreachable!()
}

fn rel(store: &Store, p: &std::path::Path) -> String {
    p.strip_prefix(store.data_dir()).map(|r| r.to_string_lossy().to_string()).unwrap_or_else(|_| p.to_string_lossy().to_string())
}
