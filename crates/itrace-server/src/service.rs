//! Business logic invoked by HTTP handlers (runs on blocking pool).

use std::collections::HashMap;

use anyhow::Context;
use rayon::prelude::*;
use serde_json::{json, Value};

use itrace_core::compare::{self, Prepared};
use itrace_core::descriptors;
use itrace_core::features;
use itrace_core::{documents, hashes, image_io, index, slice};
use itrace_core::{HASH_GATE_ALGOS, SMART_ALGOS};
use itrace_store::{ImageRecord, NewImage, NewRun, Project, Store};

use crate::{enqueue_precompute_decoded, unique_key, ApiError, ApiResult, AppState};

// ---------- upload ----------

pub fn handle_upload(
    state: &AppState,
    project_id: i64,
    filename: &str,
    data: Vec<u8>,
) -> ApiResult<Value> {
    let store = &state.store;
    let rel = unique_key(store, "uploads", filename);
    store.write_file(&rel, &data)?;
    let file_size = data.len() as i64;

    if image_io::is_supported_image(filename) {
        let img = image_io::decode(&data)
            .map_err(|e| ApiError::unprocessable(format!("图片解码失败: {e:#}")))?;
        let rec = insert_one_image(store, project_id, &rel, filename, None, &data, &img)?;
        enqueue_precompute_decoded(state, rec.id, img);
        return Ok(json!({
            "project_id": project_id, "filename": filename, "file_path": rel,
            "file_size": file_size, "file_type": "image",
            "processed_images": [{
                "id": rec.id, "filename": rec.filename,
                "file_path": rec.file_path, "type": "direct_upload"
            }],
            "error": null,
        }));
    }

    if image_io::is_supported_document(filename) {
        let extracted = documents::extract_named(filename, &data)
            .map_err(|e| ApiError::unprocessable(format!("文档解析失败: {e:#}")))?;
        let mut processed = Vec::new();
        for ex in extracted {
            let Ok(img) = image_io::decode(&ex.data) else { continue };
            let key = unique_key(store, "extracted", &ex.filename);
            store.write_file(&key, &ex.data)?;
            match insert_one_image(store, project_id, &key, &ex.filename, Some(&rel), &ex.data, &img) {
                Ok(rec) => {
                    enqueue_precompute_decoded(state, rec.id, img);
                    processed.push(json!({
                        "id": rec.id, "filename": rec.filename,
                        "file_path": rec.file_path, "type": "extracted_from_document"
                    }));
                }
                Err(e) => tracing::warn!("extracted image insert failed: {e}"),
            }
        }
        return Ok(json!({
            "project_id": project_id, "filename": filename, "file_path": rel,
            "file_size": file_size, "file_type": "document",
            "processed_images": processed, "error": null,
        }));
    }

    let _ = store.delete_file(&rel);
    Err(ApiError::bad(format!("不支持的文件格式: {filename}")))
}

fn insert_one_image(
    store: &Store,
    project_id: i64,
    key: &str,
    filename: &str,
    extracted_from: Option<&str>,
    raw: &[u8],
    img: &image::DynamicImage,
) -> anyhow::Result<ImageRecord> {
    let feats = hashes::compute_image_features_decoded(
        img,
        hashes::blake3_hex(raw),
        raw.len() as u64,
    );
    store.insert_image(&NewImage {
        project_id,
        filename: filename.to_string(),
        file_path: key.to_string(),
        file_hash: feats.file_hash,
        phash: Some(hashes::to_hex(feats.hashes.phash)),
        dhash: Some(hashes::to_hex(feats.hashes.dhash)),
        ahash: Some(hashes::to_hex(feats.hashes.ahash)),
        whash: Some(hashes::to_hex(feats.hashes.whash)),
        colorhash: Some(hashes::to_hex(feats.hashes.colorhash)),
        extracted_from: extracted_from.map(|s| s.to_string()),
        file_size: Some(feats.file_size as i64),
        width: Some(feats.width as i64),
        height: Some(feats.height as i64),
    })
}

// ---------- compare ----------

/// Decode every image once for the precise path. Returns (kept index into
/// `images`, Prepared) pairs — callers must map result indexes through
/// `kept` because undecodable images are dropped.
fn load_prepared(
    store: &Store,
    images: &[ImageRecord],
    algo: &str,
    rot_inv: bool,
) -> (Vec<usize>, Vec<Prepared>) {
    let desc_algos = compare::desc_algos_for(algo);
    let mut kept = Vec::new();
    let mut prepared = Vec::new();
    images
        .par_iter()
        .enumerate()
        .filter_map(|(i, img)| {
            let bytes = store.read_file(&img.file_path).ok()?;
            Prepared::from_bytes(&bytes, &desc_algos, rot_inv).ok().map(|p| (i, p))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(i, p)| {
            kept.push(i);
            prepared.push(p);
        });
    (kept, prepared)
}

/// Precise per-pair analysis (with pair_cache short-circuit for hash algos).
pub fn run_compare(
    state: &AppState,
    images: &[ImageRecord],
    algo: &str,
    threshold: f64,
    rot_inv: bool,
) -> anyhow::Result<Value> {
    let store = &state.store;
    if images.is_empty() {
        return Ok(json!({
            "project_id": null, "total_images": 0,
            "groups": [], "unique_images": [], "run_id": null
        }));
    }

    // Registered feature algos that have no precise-path implementation
    // (base_score only covers HASH_ALGOS/PIXEL_ALGOS/DESCRIPTOR_ALGOS/auto)
    // are served from stored feature vectors.
    let precise = itrace_core::HASH_ALGOS.contains(&algo)
        || itrace_core::PIXEL_ALGOS.contains(&algo)
        || itrace_core::DESCRIPTOR_ALGOS.contains(&algo)
        || algo == "auto";
    if !precise {
        let Some(feat) = features::algo_to_feature(algo) else {
            anyhow::bail!("unsupported algorithm: {algo}");
        };
        let ids: Vec<i64> = images.iter().map(|i| i.id).collect();
        let variants: Vec<u8> =
            if rot_inv { (0..features::NUM_VARIANTS).collect() } else { vec![0] };
        let map = store.load_feature_map(&ids, feat, &variants)?;
        anyhow::ensure!(
            !map.is_empty(),
            "该算法特征尚未预计算（先调用 feature precompute）"
        );
        let matrix = features::similarity_matrix(&map, &ids, algo, rot_inv);
        let (groups, ungrouped) = compare::cluster(&matrix, threshold);
        let mut group_json = Vec::new();
        for members in &groups {
            let mut sum = 0.0;
            let mut cnt = 0;
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    sum += matrix[members[i]][members[j]];
                    cnt += 1;
                }
            }
            let avg = if cnt > 0 { sum / cnt as f64 } else { 1.0 };
            group_json.push(json!({
                "group_id": group_json.len() + 1,
                "similarity_score": (avg * 10000.0).round() / 10000.0,
                "images": members.iter().map(|&m| image_json(&images[m])).collect::<Vec<_>>(),
            }));
        }
        let unique: Vec<Value> =
            ungrouped.iter().map(|&i| image_json(&images[i])).collect();
        let mut result = json!({
            "project_id": images[0].project_id,
            "total_images": images.len(),
            "groups": group_json,
            "unique_images": unique,
            "run_id": null
        });
        let run_id = store.insert_run(&NewRun {
            project_id: images[0].project_id,
            algorithm: algo.to_string(),
            threshold,
            total_images: images.len() as i64,
            groups_count: groups.len() as i64,
            unique_count: ungrouped.len() as i64,
            summary: Some(result.to_string()),
        })?;
        result["run_id"] = json!(run_id);
        return Ok(result);
    }

    let (kept, prepared) = load_prepared(store, images, algo, rot_inv);
    let (groups, ungrouped, matrix) = compare::analyze(&prepared, algo, threshold, rot_inv);

    // group avg similarity
    let mut group_json = Vec::new();
    for members in &groups {
        let mut sum = 0.0;
        let mut cnt = 0;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                sum += matrix[members[i]][members[j]];
                cnt += 1;
            }
        }
        let avg = if cnt > 0 { sum / cnt as f64 } else { 1.0 };
        group_json.push(json!({
            "group_id": group_json.len() + 1,
            "similarity_score": (avg * 10000.0).round() / 10000.0,
            "images": members.iter().map(|&m| image_json(&images[kept[m]])).collect::<Vec<_>>(),
        }));
    }
    let unique: Vec<Value> =
        ungrouped.iter().map(|&i| image_json(&images[kept[i]])).collect();

    let mut result = json!({
        "project_id": images[0].project_id,
        "total_images": images.len(),
        "groups": group_json,
        "unique_images": unique,
        "run_id": null
    });
    let run_id = store.insert_run(&NewRun {
        project_id: images[0].project_id,
        algorithm: algo.to_string(),
        threshold,
        total_images: images.len() as i64,
        groups_count: groups.len() as i64,
        unique_count: ungrouped.len() as i64,
        summary: Some(result.to_string()),
    })?;
    result["run_id"] = json!(run_id);
    Ok(result)
}

fn image_json(img: &ImageRecord) -> Value {
    serde_json::to_value(img).unwrap_or(Value::Null)
}

// ---------- smart compare ----------

pub fn run_smart_compare(
    state: &AppState,
    images: &[ImageRecord],
    threshold: f64,
    min_agree: usize,
) -> anyhow::Result<Value> {
    let store = &state.store;
    let n = images.len();
    if n < 2 {
        return Ok(json!({
            "total_images": n, "algorithms_used": SMART_ALGOS.len(),
            "found_duplicates": false, "duplicate_groups": [],
            "unique_count": n, "scan_seconds": 0.0,
            "summary": "项目图片不足，无法查重"
        }));
    }
    let ids: Vec<i64> = images.iter().map(|i| i.id).collect();
    if !store.features_ready(&ids)? {
        return Ok(json!({
            "total_images": n, "algorithms_used": 0,
            "found_duplicates": false, "duplicate_groups": [],
            "unique_count": n, "scan_seconds": 0.0,
            "features_pending": true,
            "summary": "部分图片特征尚未计算完成，请稍后重试"
        }));
    }

    let t0 = std::time::Instant::now();
    let variants: Vec<u8> = (0..features::NUM_VARIANTS).collect();
    let mut pair_hits: HashMap<(usize, usize), Vec<(String, f64)>> = HashMap::new();

    for algo in SMART_ALGOS {
        let Some(feat) = features::algo_to_feature(algo) else { continue };
        let map = store.load_feature_map(&ids, feat, &variants)?;
        if map.is_empty() {
            continue;
        }
        let m = features::similarity_matrix(&map, &ids, algo, true);
        for (i, row) in m.iter().enumerate() {
            for (j, &s) in row.iter().enumerate().skip(i + 1) {
                if s >= threshold {
                    pair_hits
                        .entry((i, j))
                        .or_default()
                        .push((algo.to_string(), (s * 10000.0).round() / 10000.0));
                }
            }
        }
    }

    // gate: enough votes AND at least one hash-algorithm hit
    let mut confirmed = Vec::new();
    let mut confirmed_map: HashMap<(usize, usize), Vec<String>> = HashMap::new();
    for ((i, j), hits) in &pair_hits {
        if hits.len() < min_agree {
            continue;
        }
        if !hits.iter().any(|(a, _)| HASH_GATE_ALGOS.contains(&a.as_str())) {
            continue;
        }
        confirmed.push((*i, *j));
        confirmed_map.insert((*i, *j), hits.iter().map(|(a, _)| a.clone()).collect());
    }

    let t = t0.elapsed().as_secs_f64();
    let groups_idx = compare::components_from_pairs(n, &confirmed);
    let mut dup_groups = Vec::new();
    let mut grouped = std::collections::HashSet::new();
    for members in groups_idx {
        if members.len() < 2 {
            continue;
        }
        grouped.extend(members.iter().copied());
        let mut algos = std::collections::BTreeSet::new();
        let mut best = 0.0f64;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let key = (members[i].min(members[j]), members[i].max(members[j]));
                if let Some(hits) = pair_hits.get(&key) {
                    for (a, s) in hits {
                        algos.insert(a.clone());
                        best = best.max(*s);
                    }
                }
            }
        }
        let matched: Vec<String> = algos.into_iter().collect();
        dup_groups.push(json!({
            "images": members.iter().map(|&m| json!({
                "id": images[m].id, "filename": images[m].filename,
                "file_path": images[m].file_path
            })).collect::<Vec<_>>(),
            "confidence": (best * 10000.0).round() / 10000.0,
            "matched_algorithms": matched,
            "matched_count": matched.len(),
        }));
    }
    dup_groups.sort_by(|a, b| {
        b["confidence"].as_f64().unwrap_or(0.0).total_cmp(&a["confidence"].as_f64().unwrap_or(0.0))
    });

    let found = !dup_groups.is_empty();
    let dup_n = grouped.len();
    Ok(json!({
        "total_images": n,
        "algorithms_used": SMART_ALGOS.len(),
        "found_duplicates": found,
        "duplicate_groups": dup_groups,
        "unique_count": n - dup_n,
        "scan_seconds": (t * 100.0).round() / 100.0,
        "summary": if found {
            format!("在 {n} 张图片中使用 {} 种特征比对，发现 {} 组共 {dup_n} 张疑似相似图片",
                SMART_ALGOS.len(), dup_groups.len())
        } else {
            format!("在 {n} 张图片中使用 {} 种特征比对，未发现相似图片", SMART_ALGOS.len())
        }
    }))
}

// ---------- indexed dedup (10^8-scale scan) ----------

/// Near-duplicate scan without an N×N matrix: MIH candidate recall over
/// canonical rotation-invariant hash keys, then variant-max verification.
/// Gate hashes (phash/dhash/whash) act as independent recall voters.
pub fn run_dedup_scan(
    state: &AppState,
    images: &[ImageRecord],
    radius: u32,
    threshold: f64,
    min_votes: u32,
) -> anyhow::Result<Value> {
    let store = &state.store;
    let n = images.len();
    if n < 2 {
        return Ok(json!({
            "total_images": n, "found_duplicates": false, "duplicate_groups": [],
            "unique_count": n, "candidate_pairs": 0, "naive_pairs": 0,
            "scan_seconds": 0.0, "summary": "项目图片不足，无法查重"
        }));
    }

    // candidate recall: MIH over all 8-variant keys of each gate hash —
    // equivalent coverage to variant-max comparison, sub-linear per image
    let ready: Vec<&ImageRecord> = images
        .iter()
        .filter(|i| i.feature_status == "ready")
        .collect();
    let ids: Vec<i64> = ready.iter().map(|i| i.id).collect();
    let variants: Vec<u8> = (0..features::NUM_VARIANTS).collect();
    let gate_features: Vec<&str> =
        HASH_GATE_ALGOS.iter().filter_map(|a| features::algo_to_feature(a)).collect();
    let maps: Vec<features::FeatureMap> = gate_features
        .iter()
        .map(|f| store.load_feature_map(&ids, f, &variants))
        .collect::<anyhow::Result<_>>()?;
    let exts: Vec<&dyn features::FeatureExtractor> = HASH_GATE_ALGOS
        .iter()
        .filter_map(|a| features::extractor_for_algo(a))
        .collect();

    let mut entries: Vec<index::DedupKeys> = Vec::with_capacity(ready.len());
    for img in &ready {
        let variant_keys: Vec<Vec<u64>> = maps
            .iter()
            .map(|m| {
                variants
                    .iter()
                    .filter_map(|v| m.get(&img.id).and_then(|vm| vm.get(v)))
                    .map(|b| features::unpack_bits(b))
                    .collect()
            })
            .collect();
        if variant_keys.iter().all(|k| k.len() == variants.len()) {
            entries.push(index::DedupKeys { image_id: img.id, variant_keys });
        }
    }

    let t0 = std::time::Instant::now();
    let pairs = index::dedup_candidates(&entries, radius, min_votes);
    let naive = (n as u64) * (n as u64 - 1) / 2;

    // cross-variant max: a rotated file's variant set is a permutation of
    // the original's, so (v, w) — not (v, v) — carries the true maximum
    let score_pair = |ia: i64, ib: i64| -> f64 {
        let mut best = 0.0f64;
        for (map, ext) in maps.iter().zip(&exts) {
            let (Some(ma), Some(mb)) = (map.get(&ia), map.get(&ib)) else { continue };
            for &v in &variants {
                for &w in &variants {
                    if let (Some(a), Some(b)) = (ma.get(&v), mb.get(&w)) {
                        best = best.max(ext.similarity(a, b));
                    }
                }
            }
        }
        best
    };

    // position of each image_id inside `ready`
    let pos: HashMap<i64, usize> =
        ready.iter().enumerate().map(|(p, i)| (i.id, p)).collect();
    let confirmed: Vec<(usize, usize)> = pairs
        .iter()
        .filter(|&&(i, j)| {
            score_pair(entries[i as usize].image_id, entries[j as usize].image_id) >= threshold
        })
        .filter_map(|&(i, j)| {
            Some((*pos.get(&entries[i as usize].image_id)?, *pos.get(&entries[j as usize].image_id)?))
        })
        .collect();

    let groups_idx = compare::components_from_pairs(ready.len(), &confirmed);
    let mut dup_groups = Vec::new();
    let mut grouped = std::collections::HashSet::new();
    for members in groups_idx {
        if members.len() < 2 {
            continue;
        }
        grouped.extend(members.iter().copied());
        let mut best = 0.0f64;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                best = best.max(score_pair(ready[members[i]].id, ready[members[j]].id));
            }
        }
        dup_groups.push(json!({
            "images": members.iter().map(|&m| json!({
                "id": ready[m].id, "filename": ready[m].filename,
                "file_path": ready[m].file_path
            })).collect::<Vec<_>>(),
            "confidence": (best * 10000.0).round() / 10000.0,
        }));
    }
    dup_groups.sort_by(|a, b| {
        b["confidence"].as_f64().unwrap_or(0.0).total_cmp(&a["confidence"].as_f64().unwrap_or(0.0))
    });

    let t = t0.elapsed().as_secs_f64();
    let dup_n = grouped.len();
    Ok(json!({
        "total_images": n,
        "indexed_images": entries.len(),
        "candidate_pairs": pairs.len(),
        "naive_pairs": naive,
        "found_duplicates": !dup_groups.is_empty(),
        "duplicate_groups": dup_groups,
        "unique_count": n - dup_n,
        "scan_seconds": (t * 1000.0).round() / 1000.0,
        "summary": format!(
            "索引扫描 {} 张图片：召回候选 {} 对（全量需 {} 对），确认 {} 组共 {} 张",
            entries.len(), pairs.len(), naive, dup_groups.len(), dup_n)
    }))
}

// ---------- pairwise matrix ----------

pub fn pairwise_matrix(
    state: &AppState,
    images: &[ImageRecord],
    algo: &str,
    rot_inv: bool,
) -> anyhow::Result<Value> {
    let store = &state.store;
    let ids: Vec<i64> = images.iter().map(|i| i.id).collect();
    let names: Vec<&str> = images.iter().map(|i| i.filename.as_str()).collect();

    // fast path: stored features
    if let Some(feat) = features::algo_to_feature(algo) {
        let variants: Vec<u8> =
            if rot_inv { (0..features::NUM_VARIANTS).collect() } else { vec![0] };
        let map = store.load_feature_map(&ids, feat, &variants)?;
        if !map.is_empty() {
            let m = features::similarity_matrix(&map, &ids, algo, rot_inv);
            return Ok(json!({
                "names": names, "image_ids": ids, "matrix": m,
                "algorithm": algo, "engine": "rust-matrix"
            }));
        }
    }

    // fallback: precise path on decoded images; kept[] maps matrix indexes
    // back to `images` (undecodable files are dropped)
    let (kept, prepared) = load_prepared(store, images, algo, rot_inv);
    let m = compare::pairwise_matrix(&prepared, algo, rot_inv);
    let kept_names: Vec<&str> = kept.iter().map(|&k| images[k].filename.as_str()).collect();
    let kept_ids: Vec<i64> = kept.iter().map(|&k| images[k].id).collect();
    Ok(json!({
        "names": kept_names, "image_ids": kept_ids, "matrix": m,
        "algorithm": algo, "engine": "precise"
    }))
}

// ---------- report ----------

pub fn build_report(
    state: &AppState,
    project: &Project,
    images: &[ImageRecord],
    algo: &str,
    threshold: f64,
    rot_inv: bool,
) -> anyhow::Result<Value> {
    let matrix_payload = pairwise_matrix(state, images, algo, rot_inv)?;
    let m: Vec<Vec<f64>> = serde_json::from_value(matrix_payload["matrix"].clone())?;
    let (groups, ungrouped) = itrace_core::group::cluster(&m, threshold);

    let mut group_json = Vec::new();
    for members in &groups {
        let mut pair_matches = Vec::new();
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                pair_matches.push(json!({
                    "image_a_id": images[members[i]].id,
                    "image_b_id": images[members[j]].id,
                    "score": m[members[i]][members[j]],
                }));
            }
        }
        let avg = pair_matches.iter().map(|p| p["score"].as_f64().unwrap_or(0.0)).sum::<f64>()
            / pair_matches.len().max(1) as f64;
        group_json.push(json!({
            "group_id": group_json.len() + 1,
            "similarity_score": (avg * 10000.0).round() / 10000.0,
            "images": members.iter().map(|&i| json!({
                "id": images[i].id, "filename": images[i].filename
            })).collect::<Vec<_>>(),
            "pair_matches": pair_matches,
        }));
    }

    let total = images.len();
    let dup = total - ungrouped.len();
    Ok(json!({
        "project": {"id": project.id, "name": project.name, "description": project.description},
        "generated_at": chrono_now(),
        "algorithm": algo, "threshold": threshold, "rotation_invariant": rot_inv,
        "images": images.iter().map(|i| json!({
            "id": i.id, "filename": i.filename, "width": i.width,
            "height": i.height, "file_size": i.file_size
        })).collect::<Vec<_>>(),
        "matrix": {
            "names": matrix_payload["names"], "image_ids": matrix_payload["image_ids"],
            "values": m, "engine": matrix_payload["engine"]
        },
        "groups": group_json,
        "summary": {
            "total_images": total, "similar_groups": groups.len(),
            "unique_images": ungrouped.len(),
            "duplicate_rate": if total > 0 { (dup as f64 / total as f64 * 100.0 * 10.0).round() / 10.0 } else { 0.0 }
        }
    }))
}

fn chrono_now() -> String {
    // RFC3339 without pulling chrono
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil-from-days (Howard Hinnant)
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// ---------- match data / visualize / slices ----------

fn load_gray(store: &Store, img: &ImageRecord) -> anyhow::Result<itrace_core::GrayImage> {
    let bytes = store.read_file(&img.file_path).context("图像文件不存在")?;
    let decoded = image_io::decode(&bytes)?;
    let small = image_io::resize_max_side(&decoded, 1024);
    Ok(image_io::to_gray(&small))
}

pub fn match_data(
    state: &AppState,
    ia: &ImageRecord,
    ib: &ImageRecord,
    algo: &str,
) -> ApiResult<Value> {
    let store = &state.store;
    let ext = descriptors::extractor_for(algo)
        .ok_or_else(|| ApiError::bad(format!("算法 {algo} 在此构建中不可用")))?;
    let ga = load_gray(store, ia).map_err(|e| ApiError::not_found(e.to_string()))?;
    let gb = load_gray(store, ib).map_err(|e| ApiError::not_found(e.to_string()))?;
    let da = ext.detect(&ga, 500);
    let db = ext.detect(&gb, 500);
    let matches = descriptors::match_knn_ratio(&da, &db, 0.75);
    let matches: Vec<descriptors::Match> = matches.into_iter().take(50).collect();

    // compact keypoint lists only for matched points
    let mut ka = Vec::new();
    let mut kb = Vec::new();
    let mut map_a: HashMap<usize, usize> = HashMap::new();
    let mut map_b: HashMap<usize, usize> = HashMap::new();
    let mut mjson = Vec::new();
    for m in &matches {
        let ai = *map_a.entry(m.a_idx).or_insert_with(|| {
            let kp = da.keypoints[m.a_idx];
            ka.push(json!({"x": (kp.x * 10.0).round() / 10.0, "y": (kp.y * 10.0).round() / 10.0}));
            ka.len() - 1
        });
        let bi = *map_b.entry(m.b_idx).or_insert_with(|| {
            let kp = db.keypoints[m.b_idx];
            kb.push(json!({"x": (kp.x * 10.0).round() / 10.0, "y": (kp.y * 10.0).round() / 10.0}));
            kb.len() - 1
        });
        mjson.push(json!({"a_idx": ai, "b_idx": bi, "distance": m.distance}));
    }
    let norm = (da.desc_len * 8) as f64;
    let quality = if mjson.is_empty() {
        0.0
    } else {
        let avg = mjson.iter().map(|m| m["distance"].as_f64().unwrap_or(0.0)).sum::<f64>()
            / mjson.len() as f64;
        1.0 - (avg.min(norm) / norm)
    };
    let coverage = (mjson.len() as f64 / 64.0).min(1.0);
    let score = quality * coverage;
    Ok(json!({
        "image_a": {"width": ga.width, "height": ga.height, "keypoints": ka},
        "image_b": {"width": gb.width, "height": gb.height, "keypoints": kb},
        "matches": mjson,
        "match_count": mjson.len(),
        "score": (score * 10000.0).round() / 10000.0,
    }))
}

pub fn visualize(
    state: &AppState,
    ia: &ImageRecord,
    ib: &ImageRecord,
    algo: &str,
) -> ApiResult<Value> {
    let store = &state.store;
    let ext = descriptors::extractor_for(algo)
        .ok_or_else(|| ApiError::bad(format!("算法 {algo} 在此构建中不可用")))?;
    let imga = image_io::decode(&store.read_file(&ia.file_path).map_err(|_| ApiError::not_found("图像文件不存在"))?)?;
    let imgb = image_io::decode(&store.read_file(&ib.file_path).map_err(|_| ApiError::not_found("图像文件不存在"))?)?;
    let imga = image_io::resize_max_side(&imga, 640);
    let imgb = image_io::resize_max_side(&imgb, 640);
    let ga = image_io::to_gray(&imga);
    let gb = image_io::to_gray(&imgb);
    let da = ext.detect(&ga, 500);
    let db = ext.detect(&gb, 500);
    let matches = descriptors::match_cross_check(&da, &db);
    let matches: Vec<descriptors::Match> = matches.into_iter().take(40).collect();

    let ra = image_io::to_rgb(&imga);
    let rb = image_io::to_rgb(&imgb);
    let vis = crate::draw::draw_matches(&ra, &rb, &da, &db, &matches);

    let fname = format!("match_{}_{}.jpg", algo, uuid_short());
    store.write_file(&format!("visualizations/{fname}"), &image_io::encode_jpeg(&vis, 90)?)?;
    Ok(json!({
        "file_path": format!("visualizations/{fname}"),
        "media_type": "image/jpeg"
    }))
}

fn uuid_short() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{:08x}", (nanos as u64) ^ (nanos >> 64) as u64)
}

pub fn slice_match(
    state: &AppState,
    ia: &ImageRecord,
    ib: &ImageRecord,
    rows: u32,
    cols: u32,
) -> anyhow::Result<slice::SliceMatchResult> {
    let store = &state.store;
    let ga = load_gray(store, ia)?;
    let gb = load_gray(store, ib)?;
    Ok(slice::slice_match(&ga, &gb, rows, cols, 0.7))
}


