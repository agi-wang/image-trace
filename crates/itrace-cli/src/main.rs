//! itrace — offline CLI for Image Trace.
//! Operates directly on the store; for the HTTP API run `itrace-server`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use rayon::prelude::*;

use itrace_core::compare::{self, Prepared};
use itrace_core::{documents, hashes, image_io, features, index, semantic, slice};
use itrace_core::HASH_GATE_ALGOS;
use itrace_store::{ImageStore, NewImage};

#[derive(Parser)]
#[command(name = "itrace", version, about = "Image Trace — 图像比对与查重")]
struct Cli {
    /// 数据目录（含 image-trace.db 与 uploads/extracted/）
    #[arg(long, global = true, default_value = "data")]
    data_dir: PathBuf,

    /// 元数据后端：sqlite（默认）| postgres（需 ITRACE_DATABASE_URL）。
    /// 优先级：--store > $ITRACE_STORE > sqlite。
    #[arg(long, global = true, value_enum)]
    store: Option<StoreKind>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum StoreKind {
    Sqlite,
    Postgres,
}

impl StoreKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sqlite => "sqlite",
            Self::Postgres => "postgres",
        }
    }
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
    /// 索引查重（MIH 候选召回 + 门控哈希复核；对齐 server /dedup 默认）
    Dedup {
        project_id: i64,
        /// Hamming radius for MIH recall (server default 10)
        #[arg(long, default_value = "10")]
        radius: u32,
        #[arg(long, default_value = "0.85")]
        threshold: f64,
        /// Distinct gate algorithms that must flag a pair (server default 2)
        #[arg(long, default_value = "2")]
        min_votes: u32,
        /// MIH shard bits (0..=16). Default: env ITRACE_MIH_SHARD_BITS or 8.
        #[arg(long)]
        shard_bits: Option<u32>,
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
    let store = itrace_store::open_image_store_as(
        &cli.data_dir,
        cli.store.map(StoreKind::as_str),
    )
    .context("打开数据目录失败")?;

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
                add_file(&*store, project_id, &f)?;
            }
        }
        Cmd::Images { project_id } => {
            println!("{}", serde_json::to_string_pretty(&store.list_images(project_id, 0, i64::MAX)?)?);
        }
        Cmd::Precompute { project_id } => {
            let images = store.list_image_meta(project_id)?;
            // Backfill when `ready` predates newly registered extractors
            // (e.g. crophash): stored algorithm coverage < EXTRACTORS.len().
            let want = features::EXTRACTORS.len() as i64;
            let computed: Vec<_> = images
                .par_iter()
                .filter(|i| {
                    i.feature_status != "ready"
                        || store.feature_algorithm_count(i.id).unwrap_or(0) < want
                })
                .map(|i| (i.id, compute_rows(&*store, &i.file_path)))
                .collect();
            for (id, rows) in computed {
                if let Err(e) = write_features(&*store, id, rows) {
                    eprintln!("image {id} precompute failed: {e}");
                }
            }
            println!("done");
        }
        Cmd::Compare { project_id, algorithm, threshold, rotation_invariant } => {
            let images = store.list_image_meta(project_id)?;
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
            let images = store.list_image_meta(project_id)?;
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
                    itrace_core::smart_pair_confirmed(
                        hits.iter().map(|s| s.as_str()),
                        min_agree,
                    )
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
        Cmd::Dedup { project_id, radius, threshold, min_votes, shard_bits } => {
            run_cli_dedup(&*store, project_id, radius, threshold, min_votes, shard_bits)?;
        }
        Cmd::Report { project_id, algorithm, threshold } => {
            let images = store.list_image_meta(project_id)?;
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
            let pa = store.get_image_path(image_a)?;
            let pb = store.get_image_path(image_b)?;
            let ga = image_io::to_gray(&image_io::decode(&store.read_file(&pa)?)?);
            let gb = image_io::to_gray(&image_io::decode(&store.read_file(&pb)?)?);
            let res = itrace_core::slice::slice_match(&ga, &gb, rows, cols, 0.7);
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
    }
    Ok(())
}


/// CLI path mirroring server `run_dedup_scan`: load gate-hash variant keys
/// from the store, MIH recall via `dedup_confirmed`, print groups like smart.
fn run_cli_dedup(
    store: &dyn ImageStore,
    project_id: i64,
    radius: u32,
    threshold: f64,
    min_votes: u32,
    shard_bits_opt: Option<u32>,
) -> anyhow::Result<()> {
    let shard_bits = index::resolve_shard_bits(shard_bits_opt);
    let images = store.list_image_meta(project_id)?;
    let n = images.len();
    if n < 2 {
        println!("图片不足");
        return Ok(());
    }
    let ready: Vec<&itrace_store::ImageMeta> = images
        .iter()
        .filter(|i| i.feature_status == "ready")
        .collect();
    if ready.len() < 2 {
        eprintln!("特征尚未就绪，先运行 `itrace precompute {project_id}`");
        return Ok(());
    }
    let ids: Vec<i64> = ready.iter().map(|i| i.id).collect();
    let variants: Vec<u8> = (0..features::NUM_VARIANTS).collect();
    let mut feat_names: Vec<&'static str> = Vec::new();
    let gate_feats: Vec<usize> = HASH_GATE_ALGOS
        .iter()
        .filter_map(|a| {
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
    // Crop-recall channel feature rides the same batched load.
    let crop_fi = features::algo_to_feature("crophash").map(|f| {
        feat_names.push(f);
        feat_names.len() - 1
    });
    let maps = store.load_feature_maps(&ids, &feat_names, &variants)?;
    let entries: Vec<index::DedupKeys> = ready
        .iter()
        .filter_map(|img| {
            let variant_keys: Vec<Vec<u64>> = gate_feats
                .iter()
                .map(|&fi| {
                    variants
                        .iter()
                        .filter_map(|v| maps[fi].get(&img.id).and_then(|vm| vm.get(v)))
                        .map(|b| features::unpack_bits(b))
                        .collect()
                })
                .collect();
            if variant_keys.iter().all(|k| k.len() == variants.len()) {
                Some(index::DedupKeys {
                    image_id: img.id,
                    variant_keys,
                })
            } else {
                None
            }
        })
        .collect();
    if entries.len() < 2 {
        println!("可索引图片不足（需要完整 gate 特征）");
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let index_dir = index::resolve_mih_index_dir()
        .map(|base| index::project_mih_index_path(&base, project_id));
    let ready_by_id: std::collections::HashMap<i64, &itrace_store::ImageMeta> =
        ready.iter().map(|r| (r.id, *r)).collect();

    // Semantic recall channel (Phase 4 R2, opt-in via ITRACE_SEMANTIC=1):
    // embed every indexed image, ANN-probe for high-cosine neighbours and
    // union the hits into the MIH candidate set BEFORE verification —
    // confirm_pairs is unchanged, so a semantic hit still needs the
    // cross-variant hash score ≥ threshold and cannot widen merges.
    // Entries that fail to read/embed get a zero vector (cosine 0 to
    // everything → never a candidate), keeping `sem_entries` index-aligned
    // with `entries`.
    // With ITRACE_MIH_INDEX_DIR set, vectors persist in a
    // `project_{id}_sem/` bundle (embedder-fingerprint + image-set
    // invalidated — stale vectors are never reused); a matching bundle
    // skips model inference entirely.
    let (sem_pairs, sem_index_loaded): (Vec<(u32, u32)>, bool) =
        if semantic::semantic_channel_enabled() {
            match semantic::embedder_from_env() {
                Some(embedder) => {
                    let image_ids: Vec<i64> = entries.iter().map(|e| e.image_id).collect();
                    let sem_dir = index::resolve_mih_index_dir()
                        .map(|base| semantic::project_sem_index_path(&base, project_id));
                    let (sem_entries, loaded) = semantic::load_or_build_project_sem_index(
                        sem_dir.as_deref(),
                        &image_ids,
                        &*embedder,
                        |id| {
                            ready_by_id
                                .get(&id)
                                .and_then(|m| store.read_file(&m.file_path).ok())
                                .and_then(|b| embedder.embed_bytes(&b).ok())
                        },
                    )?;
                    (
                        semantic::semantic_candidates(
                            &sem_entries,
                            semantic::resolve_semantic_k(None),
                            semantic::resolve_semantic_min_cosine(None),
                        ),
                        loaded,
                    )
                }
                None => (Vec::new(), false),
            }
        } else {
            (Vec::new(), false)
        };

    let (cand_n, confirmed, index_loaded) = index::dedup_confirmed_cached_with_extra(
        &entries,
        radius,
        threshold,
        min_votes,
        shard_bits,
        index_dir.as_deref(),
        &sem_pairs,
    )?;

    // Crop/slice recall channel: windowed-phash keys → hit-counted
    // candidates verified by NCC containment on the decoded images.
    let crop_entries: Vec<index::CropKeys> = match crop_fi.filter(|&fi| !maps[fi].is_empty()) {
        Some(fi) => ready
            .par_iter()
            .filter_map(|img| {
                let vm = maps[fi].get(&img.id)?;
                let mut keys =
                    Vec::with_capacity(features::crophash::N_KEYS * variants.len());
                for &v in &variants {
                    keys.extend_from_slice(&features::crophash::payload_keys(vm.get(&v)?)?);
                }
                keys.sort_unstable();
                keys.dedup();
                Some(index::CropKeys {
                    image_id: img.id,
                    keys,
                })
            })
            .collect(),
        None => Vec::new(),
    };
    let (crop_pairs, crop_index_loaded) = if crop_entries.len() >= 2 {
        let crop_dir = index::resolve_mih_index_dir()
            .map(|base| index::project_crop_mih_index_path(&base, project_id));
        index::crop_candidates_cached(
            &crop_entries,
            radius,
            index::resolve_crop_min_hits(None),
            shard_bits,
            crop_dir.as_deref(),
        )?
    } else {
        (Vec::new(), false)
    };
    let crop_confirmed: Vec<(i64, i64, f64)> = crop_pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let ia = crop_entries[i as usize].image_id;
            let ib = crop_entries[j as usize].image_id;
            let (Some(ra), Some(rb)) = (ready_by_id.get(&ia), ready_by_id.get(&ib)) else {
                return None;
            };
            let ga = image_io::to_gray(&image_io::decode(&store.read_file(&ra.file_path).ok()?).ok()?);
            let gb = image_io::to_gray(&image_io::decode(&store.read_file(&rb.file_path).ok()?).ok()?);
            let (s, contained) = slice::contains_rot4(&ga, &gb);
            if contained && s >= threshold {
                Some((ia, ib, s))
            } else {
                None
            }
        })
        .collect();

    let pos: std::collections::HashMap<i64, usize> =
        ready.iter().enumerate().map(|(p, r)| (r.id, p)).collect();
    let mut pairs: Vec<(usize, usize)> = confirmed
        .iter()
        .filter_map(|&(i, j, _)| {
            let ia = entries[i].image_id;
            let ib = entries[j].image_id;
            let pi = *pos.get(&ia)?;
            let pj = *pos.get(&ib)?;
            Some((pi.min(pj), pi.max(pj)))
        })
        .collect();
    for &(ia, ib, _) in &crop_confirmed {
        if let (Some(&pi), Some(&pj)) = (pos.get(&ia), pos.get(&ib)) {
            pairs.push((pi.min(pj), pi.max(pj)));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    // Build score lookup for group confidence
    let mut score_cache: std::collections::HashMap<(i64, i64), f64> =
        std::collections::HashMap::new();
    for &(i, j, s) in &confirmed {
        let ia = entries[i].image_id;
        let ib = entries[j].image_id;
        score_cache.insert((ia.min(ib), ia.max(ib)), s);
    }
    for &(ia, ib, s) in &crop_confirmed {
        score_cache
            .entry((ia.min(ib), ia.max(ib)))
            .and_modify(|e| *e = e.max(s))
            .or_insert(s);
    }
    let groups = compare::components_from_pairs(ready.len(), &pairs);
    let mut shown = 0;
    let mut grouped = 0usize;
    for members in groups {
        if members.len() < 2 {
            continue;
        }
        shown += 1;
        grouped += members.len();
        let mut best = 0.0f64;
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let ia = ready[members[a]].id;
                let ib = ready[members[b]].id;
                if let Some(&s) = score_cache.get(&(ia.min(ib), ia.max(ib))) {
                    best = best.max(s);
                }
            }
        }
        println!("duplicate group {shown} (confidence {best:.4}):");
        for &m in &members {
            println!("  - {} ({})", ready[m].filename, ready[m].id);
        }
    }
    let naive = (n as u64) * (n as u64 - 1) / 2;
    // Per-channel contribution note appears only when the channel is
    // armed, keeping default output byte-identical.
    let sem_note = if semantic::semantic_channel_enabled() {
        format!(
            " +{} sem{}",
            sem_pairs.len(),
            if sem_index_loaded { " (cached)" } else { "" }
        )
    } else {
        String::new()
    };
    println!(
        "indexed {}, candidates {} (+{} crop{}), naive {}, {} dup groups / {} images, scan {:.3}s, index_loaded={}, crop_index_loaded={}",
        entries.len(),
        cand_n,
        crop_pairs.len(),
        sem_note,
        naive,
        shown,
        grouped,
        t0.elapsed().as_secs_f64(),
        index_loaded,
        crop_index_loaded
    );
    Ok(())
}

fn add_file(store: &dyn ImageStore, project_id: i64, path: &PathBuf) -> anyhow::Result<()> {
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
        // Decode + insert-time hashes in parallel (same shape as the
        // server's document upload); unique_key/write/insert stay
        // sequential so key assignment, row ids, output order and the
        // first-error bail position are all unchanged.
        let decoded: Vec<anyhow::Result<(image::DynamicImage, itrace_core::ImageFeatures)>> =
            extracted
                .par_iter()
                .map(|im| {
                    let d = image_io::decode(&im.data)?;
                    let feats = hashes::compute_image_features_decoded(
                        &d,
                        hashes::blake3_hex(&im.data),
                        im.data.len() as u64,
                    );
                    Ok((d, feats))
                })
                .collect();
        for (img, pair) in extracted.iter().zip(decoded) {
            let key = unique_key(store, "extracted", &img.filename);
            store.write_file(&key, &img.data)?;
            let (decoded, feats) = pair?;
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
fn compute_rows(store: &dyn ImageStore, key: &str) -> anyhow::Result<FeatureRows> {
    Ok(features::compute_all_variants(&image_io::decode(&store.read_file(key)?)?))
}

/// Status protocol: computing → batched write → ready / pending.
fn write_features(
    store: &dyn ImageStore,
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
    store: &dyn ImageStore,
    image_id: i64,
    img: &image::DynamicImage,
) -> anyhow::Result<()> {
    write_features(store, image_id, Ok(features::compute_all_variants(img)))
}

fn unique_key(store: &dyn ImageStore, prefix: &str, name: &str) -> String {
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
