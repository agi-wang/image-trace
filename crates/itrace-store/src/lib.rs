//! SQLite store: projects / images / feature vectors / pair cache / runs.
//! File payloads go through `blob::BlobStore` — local fs or MinIO/S3.

pub mod blob;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use itrace_core::features::FeatureMap;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub use blob::BlobStore;

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
-- Readers wait up to 10s on a writer instead of erroring; WAL makes
-- synchronous=NORMAL crash-safe; larger page cache + mmap cut syscall
-- and page-cache traffic on scan-heavy queries.
PRAGMA busy_timeout = 10000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -262144;
PRAGMA mmap_size = 268435456;

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    phash TEXT, dhash TEXT, ahash TEXT, whash TEXT, colorhash TEXT,
    extracted_from TEXT,
    file_size INTEGER, width INTEGER, height INTEGER,
    feature_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_project_status ON images(project_id, feature_status);

CREATE TABLE IF NOT EXISTS feature_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    variant_idx INTEGER NOT NULL DEFAULT 0,
    algorithm TEXT NOT NULL,
    vector BLOB NOT NULL,
    dims INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(image_id, variant_idx, algorithm)
);
CREATE INDEX IF NOT EXISTS idx_fs_image ON feature_store(image_id);
CREATE INDEX IF NOT EXISTS idx_fs_algo ON feature_store(algorithm);
-- Covers load_feature_map: WHERE algorithm = ? AND image_id IN (…)
-- AND variant_idx IN (…).
CREATE INDEX IF NOT EXISTS idx_fs_cover ON feature_store(algorithm, image_id, variant_idx);

CREATE TABLE IF NOT EXISTS pair_cache (
    hash_a TEXT NOT NULL,
    hash_b TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    rotation_invariant INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (hash_a, hash_b, algorithm, rotation_invariant)
);
-- delete_image prunes by hash_b as well; the PK only covers hash_a.
CREATE INDEX IF NOT EXISTS idx_pair_cache_b ON pair_cache(hash_b);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    threshold REAL NOT NULL,
    total_images INTEGER NOT NULL,
    groups_count INTEGER NOT NULL,
    unique_count INTEGER NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON analysis_runs(project_id);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub image_count: i64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: i64,
    pub project_id: i64,
    pub filename: String,
    pub file_path: String,
    pub file_hash: String,
    pub phash: Option<String>,
    pub dhash: Option<String>,
    pub ahash: Option<String>,
    pub whash: Option<String>,
    pub colorhash: Option<String>,
    pub extracted_from: Option<String>,
    pub file_size: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub feature_status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRunRecord {
    pub id: i64,
    pub project_id: i64,
    pub algorithm: String,
    pub threshold: f64,
    pub total_images: i64,
    pub groups_count: i64,
    pub unique_count: i64,
    pub summary: Option<String>,
    pub created_at: String,
}

/// Thread-safe store. SQLite serializes writers anyway; a Mutex'd single
/// connection in WAL mode is the right shape for a local-first tool.
pub struct Store {
    conn: Mutex<Connection>,
    data_dir: PathBuf,
    blobs: Arc<dyn BlobStore>,
}

impl Store {
    pub fn open(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let conn = Connection::open(data_dir.join("image-trace.db"))?;
        conn.execute_batch(SCHEMA)?;
        let blobs = blob::blob_store_from_env(data_dir)?;
        Ok(Self {
            conn: Mutex::new(conn),
            data_dir: data_dir.to_path_buf(),
            blobs,
        })
    }

    /// Explicit backend override (tests).
    pub fn with_blobs(mut self, blobs: Arc<dyn BlobStore>) -> Self {
        self.blobs = blobs;
        self
    }

    pub fn blobs(&self) -> &Arc<dyn BlobStore> {
        &self.blobs
    }
    pub fn storage_kind(&self) -> &'static str {
        self.blobs.kind()
    }
    pub fn write_file(&self, key: &str, data: &[u8]) -> anyhow::Result<()> {
        self.blobs.put(key, data)
    }
    pub fn read_file(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        self.blobs.get(key)
    }
    pub fn delete_file(&self, key: &str) -> anyhow::Result<()> {
        self.blobs.delete(key)
    }
    pub fn file_exists(&self, key: &str) -> bool {
        self.blobs.exists(key)
    }

    pub fn data_dir(&self) -> &std::path::Path {
        &self.data_dir
    }
    pub fn upload_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }
    pub fn extract_dir(&self) -> PathBuf {
        self.data_dir.join("extracted")
    }

    /// Resolve a stored key (e.g. "uploads/x.jpg") to a real fs path when the
    /// blob backend is local; None under object storage — use read_file instead.
    pub fn resolve(&self, rel: &str) -> Option<PathBuf> {
        let rel = rel.strip_prefix("data/").unwrap_or(rel);
        self.blobs.local_path(rel)
    }

    // ---------- projects ----------

    pub fn create_project(&self, name: &str, description: Option<&str>) -> anyhow::Result<Project> {
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached("INSERT INTO projects(name, description) VALUES(?1, ?2)")?
            .execute(params![name, description])?;
        let id = conn.last_insert_rowid();
        drop(conn);
        self.get_project(id)
    }

    pub fn get_project(&self, id: i64) -> anyhow::Result<Project> {
        let conn = self.conn.lock().unwrap();
        let found = conn
            .prepare_cached(
                "SELECT p.id, p.name, p.description, p.created_at, COUNT(i.id)
                 FROM projects p LEFT JOIN images i ON i.project_id = p.id
                 WHERE p.id = ?1
                 GROUP BY p.id",
            )?
            .query_row(params![id], |r| {
                Ok(Project {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    description: r.get(2)?,
                    created_at: r.get(3)?,
                    image_count: r.get(4)?,
                })
            })
            .optional()?;
        found.with_context(|| format!("项目不存在: {id}"))
    }

    pub fn list_projects(&self, skip: i64, limit: i64) -> anyhow::Result<Vec<Project>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare_cached(
            "SELECT p.id, p.name, p.description, p.created_at, COUNT(i.id)
             FROM projects p LEFT JOIN images i ON i.project_id = p.id
             GROUP BY p.id ORDER BY p.id LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt
            .query_map(params![limit, skip], |r| {
                Ok(Project {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    description: r.get(2)?,
                    created_at: r.get(3)?,
                    image_count: r.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Delete project row (CASCADE removes images/runs/features);
    /// caller removes files. Errors when the project does not exist.
    pub fn delete_project(&self, id: i64) -> anyhow::Result<Vec<ImageRecord>> {
        let images = self.list_images(id, 0, i64::MAX)?;
        let conn = self.conn.lock().unwrap();
        let n = conn
            .prepare_cached("DELETE FROM projects WHERE id = ?1")?
            .execute(params![id])?;
        if n == 0 {
            anyhow::bail!("项目不存在: {id}");
        }
        Ok(images)
    }

    // ---------- images ----------

    pub fn insert_image(&self, rec: &NewImage) -> anyhow::Result<ImageRecord> {
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached(
            "INSERT INTO images(project_id, filename, file_path, file_hash,
                phash, dhash, ahash, whash, colorhash, extracted_from,
                file_size, width, height, feature_status)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,'pending')",
        )?
        .execute(params![
            rec.project_id, rec.filename, rec.file_path, rec.file_hash,
            rec.phash, rec.dhash, rec.ahash, rec.whash, rec.colorhash,
            rec.extracted_from, rec.file_size, rec.width, rec.height,
        ])?;
        let id = conn.last_insert_rowid();
        drop(conn);
        self.get_image(id)
    }

    pub fn get_image(&self, id: i64) -> anyhow::Result<ImageRecord> {
        let conn = self.conn.lock().unwrap();
        Self::row_to_image(&conn, "WHERE i.id = ?1", params![id])?
            .into_iter()
            .next()
            .with_context(|| format!("图像不存在: {id}"))
    }

    pub fn list_images(&self, project_id: i64, skip: i64, limit: i64) -> anyhow::Result<Vec<ImageRecord>> {
        let conn = self.conn.lock().unwrap();
        Self::row_to_image(
            &conn,
            "WHERE i.project_id = ?1 ORDER BY i.id LIMIT ?2 OFFSET ?3",
            params![project_id, limit, skip],
        )
    }

    fn row_to_image(
        conn: &Connection,
        tail: &str,
        p: impl rusqlite::Params,
    ) -> anyhow::Result<Vec<ImageRecord>> {
        let sql = format!(
            "SELECT i.id, i.project_id, i.filename, i.file_path, i.file_hash,
                    i.phash, i.dhash, i.ahash, i.whash, i.colorhash,
                    i.extracted_from, i.file_size, i.width, i.height,
                    i.feature_status, i.created_at
             FROM images i {tail}"
        );
        // `tail` has only a handful of distinct shapes, so the cached
        // statements still hit on repeat calls.
        let mut stmt = conn.prepare_cached(&sql)?;
        let rows = stmt
            .query_map(p, |r| {
                Ok(ImageRecord {
                    id: r.get(0)?,
                    project_id: r.get(1)?,
                    filename: r.get(2)?,
                    file_path: r.get(3)?,
                    file_hash: r.get(4)?,
                    phash: r.get(5)?,
                    dhash: r.get(6)?,
                    ahash: r.get(7)?,
                    whash: r.get(8)?,
                    colorhash: r.get(9)?,
                    extracted_from: r.get(10)?,
                    file_size: r.get(11)?,
                    width: r.get(12)?,
                    height: r.get(13)?,
                    feature_status: r.get(14)?,
                    created_at: r.get(15)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn delete_image(&self, id: i64) -> anyhow::Result<ImageRecord> {
        let rec = self.get_image(id)?;
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached("DELETE FROM pair_cache WHERE hash_a = ?1 OR hash_b = ?1")?
            .execute(params![rec.file_hash])?;
        conn.prepare_cached("DELETE FROM images WHERE id = ?1")?
            .execute(params![id])?;
        Ok(rec)
    }

    // ---------- feature store ----------

    pub fn set_feature_status(&self, image_id: i64, status: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached("UPDATE images SET feature_status = ?1 WHERE id = ?2")?
            .execute(params![status, image_id])?;
        Ok(())
    }

    pub fn put_feature(
        &self,
        image_id: i64,
        variant_idx: u8,
        algorithm: &str,
        vector: &[u8],
        dims: usize,
    ) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached(
            "INSERT INTO feature_store(image_id, variant_idx, algorithm, vector, dims)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(image_id, variant_idx, algorithm) DO UPDATE SET
               vector = excluded.vector, dims = excluded.dims",
        )?
        .execute(params![image_id, variant_idx as i64, algorithm, vector, dims as i64])?;
        Ok(())
    }

    /// Batch-upsert all feature rows for one image in ONE transaction.
    ///
    /// `rows`: `(variant_idx, algorithm, vector, dims)` — one prepared
    /// upsert is reused for the whole batch, so a variant×algorithm
    /// matrix lands in a single commit instead of one commit per row.
    pub fn put_features(
        &self,
        image_id: i64,
        rows: &[(u8, &str, &[u8], usize)],
    ) -> anyhow::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO feature_store(image_id, variant_idx, algorithm, vector, dims)
                 VALUES(?1,?2,?3,?4,?5)
                 ON CONFLICT(image_id, variant_idx, algorithm) DO UPDATE SET
                   vector = excluded.vector, dims = excluded.dims",
            )?;
            for &(variant_idx, algorithm, vector, dims) in rows {
                stmt.execute(params![
                    image_id,
                    variant_idx as i64,
                    algorithm,
                    vector,
                    dims as i64,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// vectors[image_id][variant_idx] = raw bytes for `feature`.
    pub fn load_feature_map(
        &self,
        image_ids: &[i64],
        feature: &str,
        variants: &[u8],
    ) -> anyhow::Result<FeatureMap> {
        let conn = self.conn.lock().unwrap();
        Self::load_feature_map_conn(&conn, image_ids, feature, variants)
    }

    /// Like `load_feature_map` but for several algorithms in one lock
    /// acquisition. Returns one FeatureMap per requested feature name,
    /// in the same order as `features`.
    pub fn load_feature_maps(
        &self,
        image_ids: &[i64],
        features: &[&str],
        variants: &[u8],
    ) -> anyhow::Result<Vec<FeatureMap>> {
        let conn = self.conn.lock().unwrap();
        features
            .iter()
            .map(|f| Self::load_feature_map_conn(&conn, image_ids, f, variants))
            .collect()
    }

    /// Shared body for the feature-map loaders. `IN` lists are bound
    /// `?N` placeholders (values as params, not interpolated text) and
    /// id lists are chunked so total bound variables stay under
    /// SQLite's classic 999 ceiling.
    fn load_feature_map_conn(
        conn: &Connection,
        image_ids: &[i64],
        feature: &str,
        variants: &[u8],
    ) -> anyhow::Result<FeatureMap> {
        let mut map: FeatureMap = FeatureMap::new();
        if image_ids.is_empty() || variants.is_empty() {
            return Ok(map);
        }
        // ?1 = algorithm; ids start at ?2, variants follow.
        let chunk_len = (900usize.saturating_sub(variants.len() + 1)).max(1);
        for ids in image_ids.chunks(chunk_len) {
            let id_ph = placeholders(2, ids.len());
            let v_ph = placeholders(2 + ids.len(), variants.len());
            let sql = format!(
                "SELECT image_id, variant_idx, vector FROM feature_store
                 WHERE algorithm = ?1 AND image_id IN ({id_ph}) AND variant_idx IN ({v_ph})"
            );
            let mut bind: Vec<rusqlite::types::Value> =
                Vec::with_capacity(1 + ids.len() + variants.len());
            bind.push(feature.to_string().into());
            bind.extend(ids.iter().map(|&id| id.into()));
            bind.extend(variants.iter().map(|&v| v.into()));
            let mut stmt = conn.prepare_cached(&sql)?;
            let rows = stmt.query_map(params_from_iter(bind), |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)? as u8, r.get::<_, Vec<u8>>(2)?))
            })?;
            for row in rows {
                let (id, v, vec) = row?;
                map.entry(id).or_default().insert(v, vec);
            }
        }
        Ok(map)
    }

    /// Count images in `ids` whose feature_status = 'ready'.
    pub fn features_ready(&self, ids: &[i64]) -> anyhow::Result<bool> {
        let conn = self.conn.lock().unwrap();
        for chunk in ids.chunks(900) {
            let ph = placeholders(1, chunk.len());
            let n: i64 = conn
                .prepare_cached(&format!(
                    "SELECT COUNT(*) FROM images WHERE id IN ({ph}) AND feature_status = 'ready'"
                ))?
                .query_row(params_from_iter(chunk.iter()), |r| r.get(0))?;
            if n as usize != chunk.len() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // ---------- pair cache ----------

    pub fn get_pair_score(
        &self,
        hash_a: &str,
        hash_b: &str,
        algo: &str,
        rot_inv: bool,
    ) -> anyhow::Result<Option<f64>> {
        let (a, b) = ordered_pair(hash_a, hash_b);
        let conn = self.conn.lock().unwrap();
        let score = conn
            .prepare_cached(
                "SELECT score FROM pair_cache
                 WHERE hash_a=?1 AND hash_b=?2 AND algorithm=?3 AND rotation_invariant=?4",
            )?
            .query_row(params![a, b, algo, rot_inv as i64], |r| r.get(0))
            .optional()?;
        Ok(score)
    }

    pub fn put_pair_score(
        &self,
        hash_a: &str,
        hash_b: &str,
        algo: &str,
        rot_inv: bool,
        score: f64,
    ) -> anyhow::Result<()> {
        let (a, b) = ordered_pair(hash_a, hash_b);
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached(
            "INSERT OR REPLACE INTO pair_cache(hash_a,hash_b,algorithm,rotation_invariant,score)
             VALUES(?1,?2,?3,?4,?5)",
        )?
        .execute(params![a, b, algo, rot_inv as i64, score])?;
        Ok(())
    }

    // ---------- analysis runs ----------

    pub fn insert_run(&self, run: &NewRun) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.prepare_cached(
            "INSERT INTO analysis_runs(project_id, algorithm, threshold,
                total_images, groups_count, unique_count, summary)
             VALUES(?1,?2,?3,?4,?5,?6,?7)",
        )?
        .execute(params![
            run.project_id, run.algorithm, run.threshold, run.total_images,
            run.groups_count, run.unique_count, run.summary
        ])?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_runs(&self, project_id: i64, skip: i64, limit: i64) -> anyhow::Result<Vec<AnalysisRunRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare_cached(
            "SELECT id, project_id, algorithm, threshold, total_images,
                    groups_count, unique_count, summary, created_at
             FROM analysis_runs WHERE project_id=?1 ORDER BY id DESC LIMIT ?2 OFFSET ?3",
        )?;
        let rows = stmt
            .query_map(params![project_id, limit, skip], |r| {
                Ok(AnalysisRunRecord {
                    id: r.get(0)?,
                    project_id: r.get(1)?,
                    algorithm: r.get(2)?,
                    threshold: r.get(3)?,
                    total_images: r.get(4)?,
                    groups_count: r.get(5)?,
                    unique_count: r.get(6)?,
                    summary: r.get(7)?,
                    created_at: r.get(8)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_run(&self, run_id: i64) -> anyhow::Result<AnalysisRunRecord> {
        let conn = self.conn.lock().unwrap();
        let found = conn
            .prepare_cached(
                "SELECT id, project_id, algorithm, threshold, total_images,
                        groups_count, unique_count, summary, created_at
                 FROM analysis_runs WHERE id=?1",
            )?
            .query_row(params![run_id], |r| {
                Ok(AnalysisRunRecord {
                    id: r.get(0)?,
                    project_id: r.get(1)?,
                    algorithm: r.get(2)?,
                    threshold: r.get(3)?,
                    total_images: r.get(4)?,
                    groups_count: r.get(5)?,
                    unique_count: r.get(6)?,
                    summary: r.get(7)?,
                    created_at: r.get(8)?,
                })
            })
            .optional()?;
        found.with_context(|| format!("分析记录不存在: {run_id}"))
    }

    pub fn latest_run(
        &self,
        project_id: i64,
        algorithm: &str,
        threshold: f64,
    ) -> anyhow::Result<Option<AnalysisRunRecord>> {
        let conn = self.conn.lock().unwrap();
        let run = conn
            .prepare_cached(
                "SELECT id, project_id, algorithm, threshold, total_images,
                        groups_count, unique_count, summary, created_at
                 FROM analysis_runs
                 WHERE project_id=?1 AND algorithm=?2 AND threshold=?3
                 ORDER BY id DESC LIMIT 1",
            )?
            .query_row(params![project_id, algorithm, threshold], |r| {
                Ok(AnalysisRunRecord {
                    id: r.get(0)?,
                    project_id: r.get(1)?,
                    algorithm: r.get(2)?,
                    threshold: r.get(3)?,
                    total_images: r.get(4)?,
                    groups_count: r.get(5)?,
                    unique_count: r.get(6)?,
                    summary: r.get(7)?,
                    created_at: r.get(8)?,
                })
            })
            .optional()?;
        Ok(run)
    }
}

/// `?start,?start+1,…` — explicit numbered placeholders for a bound `IN` list.
fn placeholders(start: usize, n: usize) -> String {
    (start..start + n)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Fields for a new image row.
pub struct NewImage {
    pub project_id: i64,
    pub filename: String,
    pub file_path: String,
    pub file_hash: String,
    pub phash: Option<String>,
    pub dhash: Option<String>,
    pub ahash: Option<String>,
    pub whash: Option<String>,
    pub colorhash: Option<String>,
    pub extracted_from: Option<String>,
    pub file_size: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

pub struct NewRun {
    pub project_id: i64,
    pub algorithm: String,
    pub threshold: f64,
    pub total_images: i64,
    pub groups_count: i64,
    pub unique_count: i64,
    pub summary: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "itrace-store-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn img(project_id: i64, name: &str) -> NewImage {
        NewImage {
            project_id,
            filename: name.into(),
            file_path: format!("u/{name}"),
            file_hash: format!("h{name}"),
            phash: None,
            dhash: None,
            ahash: None,
            whash: None,
            colorhash: None,
            extracted_from: None,
            file_size: None,
            width: None,
            height: None,
        }
    }

    #[test]
    fn put_features_batch_and_load_maps() {
        let dir = tmp_dir("feat");
        let s = Store::open(&dir).unwrap();
        let p = s.create_project("t", None).unwrap();
        let a = s.insert_image(&img(p.id, "a.jpg")).unwrap();
        let b = s.insert_image(&img(p.id, "b.jpg")).unwrap();

        // 2 algorithms × 2 variants in one transaction, then a re-upsert
        // exercises the ON CONFLICT path.
        s.put_features(
            a.id,
            &[
                (0, "phash", &[1u8; 8][..], 64),
                (1, "phash", &[2u8; 8][..], 64),
                (0, "dhash", &[3u8; 8][..], 64),
                (1, "dhash", &[4u8; 8][..], 64),
            ],
        )
        .unwrap();
        s.put_features(a.id, &[(0, "phash", &[9u8; 8][..], 64)]).unwrap();

        let ids = vec![a.id, b.id];
        let variants = vec![0u8, 1u8];
        let maps = s
            .load_feature_maps(&ids, &["phash", "dhash"], &variants)
            .unwrap();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0][&a.id][&0], vec![9u8; 8]);
        assert_eq!(maps[0][&a.id][&1], vec![2u8; 8]);
        assert_eq!(maps[1][&a.id][&0], vec![3u8; 8]);
        assert!(!maps[0].contains_key(&b.id));
        // empty id/variant lists return an empty map, not an error
        assert!(s.load_feature_map(&[], "phash", &variants).unwrap().is_empty());
        assert!(s.load_feature_map(&ids, "phash", &[]).unwrap().is_empty());

        assert!(!s.features_ready(&ids).unwrap()); // 'pending' by default
        s.set_feature_status(a.id, "ready").unwrap();
        s.set_feature_status(b.id, "ready").unwrap();
        assert!(s.features_ready(&ids).unwrap());
        assert!(s.features_ready(&[]).unwrap());

        // image_count comes through the LEFT JOIN + GROUP BY path
        assert_eq!(s.get_project(p.id).unwrap().image_count, 2);
        assert_eq!(s.list_projects(0, 10).unwrap()[0].image_count, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
