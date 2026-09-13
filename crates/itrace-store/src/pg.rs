//! Postgres metadata backend — same logical schema as the SQLite `SCHEMA`
//! in `lib.rs`, behind the same `ImageStore` trait. File payloads still go
//! through `BlobStore` (`StoreBase`); only metadata/features move to PG.
//!
//! Driver stack: sync `postgres` crate (tokio-postgres sync facade) +
//! `r2d2` pool. `ImageStore` is a synchronous trait, so a sync client keeps
//! callers free of `block_on`/`block_in_place` hazards inside server
//! handlers; the pool gives real connection concurrency (vs the single
//! Mutex'd connection `SqliteStore` needs anyway).
//!
//! ## Type mapping (SQLite → Postgres)
//!
//! | SQLite | Postgres |
//! |--------|----------|
//! | `INTEGER PRIMARY KEY AUTOINCREMENT` | `BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY` |
//! | `INTEGER` (ids, counts, dims, variant_idx) | `BIGINT` |
//! | `TEXT` | `TEXT` |
//! | `BLOB` (feature vectors) | `BYTEA` |
//! | `REAL` (score, threshold) | `DOUBLE PRECISION` (PG `REAL` is f32) |
//! | `rotation_invariant INTEGER` 0/1 | `BOOLEAN` |
//! | `strftime('%Y-%m-%dT%H:%M:%fZ','now')` | `to_char(now() AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')` |
//!
//! SQLite `INSERT OR REPLACE` / `ON CONFLICT` upserts map to PG
//! `INSERT … ON CONFLICT … DO UPDATE`; `last_insert_rowid()` maps to
//! `RETURNING id`; `IN (?,…)` lists map to `= ANY($n)` array params.

use std::sync::Arc;

use anyhow::Context;
use itrace_core::features::FeatureMap;
use postgres::{Client, NoTls};
use r2d2_postgres::PostgresConnectionManager;

use crate::blob::{self, BlobStore};
use crate::{
    AnalysisRunRecord, ImageMeta, ImageRecord, ImageStore, NewImage, NewRun, Project, StoreBase,
};

const PG_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS projects (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')
);

CREATE TABLE IF NOT EXISTS images (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    phash TEXT, dhash TEXT, ahash TEXT, whash TEXT, colorhash TEXT,
    extracted_from TEXT,
    file_size BIGINT, width BIGINT, height BIGINT,
    feature_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_project_status ON images(project_id, feature_status);

CREATE TABLE IF NOT EXISTS feature_store (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    variant_idx BIGINT NOT NULL DEFAULT 0,
    algorithm TEXT NOT NULL,
    vector BYTEA NOT NULL,
    dims BIGINT NOT NULL,
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'),
    UNIQUE(image_id, variant_idx, algorithm)
);
CREATE INDEX IF NOT EXISTS idx_fs_image ON feature_store(image_id);
CREATE INDEX IF NOT EXISTS idx_fs_algo ON feature_store(algorithm);
CREATE INDEX IF NOT EXISTS idx_fs_cover ON feature_store(algorithm, image_id, variant_idx);

CREATE TABLE IF NOT EXISTS pair_cache (
    hash_a TEXT NOT NULL,
    hash_b TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    rotation_invariant BOOLEAN NOT NULL DEFAULT FALSE,
    score DOUBLE PRECISION NOT NULL,
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'),
    PRIMARY KEY (hash_a, hash_b, algorithm, rotation_invariant)
);
CREATE INDEX IF NOT EXISTS idx_pair_cache_b ON pair_cache(hash_b);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    total_images BIGINT NOT NULL,
    groups_count BIGINT NOT NULL,
    unique_count BIGINT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON analysis_runs(project_id);
"#;

type Pool = r2d2::Pool<PostgresConnectionManager<NoTls>>;

/// Postgres implementation of [`ImageStore`]. Construct via
/// [`open_image_store`](crate::open_image_store) (`ITRACE_STORE=postgres`)
/// or `PostgresStore::connect`.
pub struct PostgresStore {
    pool: Pool,
    base: StoreBase,
}

impl PostgresStore {
    /// `url` is a libpq-style string, e.g.
    /// `postgres://user:pass@host:5432/dbname`. `data_dir` still anchors
    /// the local `FsBlobStore` when `ITRACE_STORAGE=fs`.
    pub fn connect(url: &str, data_dir: &std::path::Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let mgr = PostgresConnectionManager::new(url.parse()?, NoTls);
        let pool = Pool::builder()
            .max_size(8)
            .build(mgr)
            .context("postgres connect")?;
        pool.get()?.batch_execute(PG_SCHEMA)?;
        let blobs = blob::blob_store_from_env(data_dir)?;
        Ok(Self {
            pool,
            base: StoreBase::new(data_dir, blobs),
        })
    }

    fn conn(&self) -> anyhow::Result<r2d2::PooledConnection<PostgresConnectionManager<NoTls>>> {
        Ok(self.pool.get()?)
    }

    fn row_to_image(
        conn: &mut Client,
        tail: &str,
        p: &[&(dyn postgres::types::ToSql + Sync)],
    ) -> anyhow::Result<Vec<ImageRecord>> {
        let sql = format!(
            "SELECT i.id, i.project_id, i.filename, i.file_path, i.file_hash,
                    i.phash, i.dhash, i.ahash, i.whash, i.colorhash,
                    i.extracted_from, i.file_size, i.width, i.height,
                    i.feature_status, i.created_at
             FROM images i {tail}"
        );
        let rows = conn
            .query(&sql, p)?
            .iter()
            .map(|r| ImageRecord {
                id: r.get(0),
                project_id: r.get(1),
                filename: r.get(2),
                file_path: r.get(3),
                file_hash: r.get(4),
                phash: r.get(5),
                dhash: r.get(6),
                ahash: r.get(7),
                whash: r.get(8),
                colorhash: r.get(9),
                extracted_from: r.get(10),
                file_size: r.get(11),
                width: r.get(12),
                height: r.get(13),
                feature_status: r.get(14),
                created_at: r.get(15),
            })
            .collect();
        Ok(rows)
    }

    /// Shared body for the feature-map loaders. `IN` lists become
    /// `= ANY($n)` array params — Postgres has no 999-variable ceiling,
    /// so no chunking is needed.
    fn load_feature_map_conn(
        conn: &mut Client,
        image_ids: &[i64],
        feature: &str,
        variants: &[u8],
    ) -> anyhow::Result<FeatureMap> {
        let mut map: FeatureMap = FeatureMap::new();
        if image_ids.is_empty() || variants.is_empty() {
            return Ok(map);
        }
        let ids: Vec<i64> = image_ids.to_vec();
        let vars: Vec<i64> = variants.iter().map(|&v| v as i64).collect();
        for row in conn.query(
            "SELECT image_id, variant_idx, vector FROM feature_store
             WHERE algorithm = $1 AND image_id = ANY($2) AND variant_idx = ANY($3)",
            &[&feature, &ids, &vars],
        )? {
            let id: i64 = row.get(0);
            let v: i64 = row.get(1);
            let vec: Vec<u8> = row.get(2);
            map.entry(id).or_default().insert(v as u8, vec);
        }
        Ok(map)
    }
}

impl ImageStore for PostgresStore {
    fn blobs(&self) -> &Arc<dyn BlobStore> {
        &self.base.blobs
    }
    fn storage_kind(&self) -> &'static str {
        self.base.blobs.kind()
    }
    fn write_file(&self, key: &str, data: &[u8]) -> anyhow::Result<()> {
        self.base.write_file(key, data)
    }
    fn read_file(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        self.base.read_file(key)
    }
    fn delete_file(&self, key: &str) -> anyhow::Result<()> {
        self.base.delete_file(key)
    }
    fn file_exists(&self, key: &str) -> bool {
        self.base.file_exists(key)
    }
    fn data_dir(&self) -> &std::path::Path {
        &self.base.data_dir
    }
    fn upload_dir(&self) -> std::path::PathBuf {
        self.base.upload_dir()
    }
    fn extract_dir(&self) -> std::path::PathBuf {
        self.base.extract_dir()
    }
    fn resolve(&self, rel: &str) -> Option<std::path::PathBuf> {
        self.base.resolve(rel)
    }

    // ---------- projects ----------

    fn create_project(&self, name: &str, description: Option<&str>) -> anyhow::Result<Project> {
        let mut conn = self.conn()?;
        let row = conn.query_one(
            "INSERT INTO projects(name, description) VALUES($1, $2) RETURNING id",
            &[&name, &description],
        )?;
        let id: i64 = row.get(0);
        drop(conn);
        self.get_project(id)
    }

    fn get_project(&self, id: i64) -> anyhow::Result<Project> {
        let mut conn = self.conn()?;
        let found = conn
            .query_opt(
                "SELECT p.id, p.name, p.description, p.created_at, COUNT(i.id)
                 FROM projects p LEFT JOIN images i ON i.project_id = p.id
                 WHERE p.id = $1
                 GROUP BY p.id",
                &[&id],
            )?
            .map(|r| Project {
                id: r.get(0),
                name: r.get(1),
                description: r.get(2),
                created_at: r.get(3),
                image_count: r.get(4),
            });
        found.with_context(|| format!("项目不存在: {id}"))
    }

    fn list_projects(&self, skip: i64, limit: i64) -> anyhow::Result<Vec<Project>> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT p.id, p.name, p.description, p.created_at, COUNT(i.id)
                 FROM projects p LEFT JOIN images i ON i.project_id = p.id
                 GROUP BY p.id ORDER BY p.id LIMIT $1 OFFSET $2",
                &[&limit, &skip],
            )?
            .iter()
            .map(|r| Project {
                id: r.get(0),
                name: r.get(1),
                description: r.get(2),
                created_at: r.get(3),
                image_count: r.get(4),
            })
            .collect();
        Ok(rows)
    }

    fn ensure_project(&self, id: i64) -> anyhow::Result<()> {
        let mut conn = self.conn()?;
        if conn
            .query_opt("SELECT 1 FROM projects WHERE id = $1", &[&id])?
            .is_none()
        {
            anyhow::bail!("项目不存在: {id}");
        }
        Ok(())
    }

    /// Delete project row (CASCADE removes images/runs/features);
    /// caller removes files. Errors when the project does not exist.
    fn delete_project(&self, id: i64) -> anyhow::Result<Vec<ImageRecord>> {
        let images = self.list_images(id, 0, i64::MAX)?;
        let mut conn = self.conn()?;
        let n = conn.execute("DELETE FROM projects WHERE id = $1", &[&id])?;
        if n == 0 {
            anyhow::bail!("项目不存在: {id}");
        }
        Ok(images)
    }

    // ---------- images ----------

    fn insert_image(&self, rec: &NewImage) -> anyhow::Result<ImageRecord> {
        let mut conn = self.conn()?;
        let row = conn.query_one(
            "INSERT INTO images(project_id, filename, file_path, file_hash,
                phash, dhash, ahash, whash, colorhash, extracted_from,
                file_size, width, height, feature_status)
             VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,'pending')
             RETURNING id, created_at, feature_status",
            &[
                &rec.project_id,
                &rec.filename,
                &rec.file_path,
                &rec.file_hash,
                &rec.phash,
                &rec.dhash,
                &rec.ahash,
                &rec.whash,
                &rec.colorhash,
                &rec.extracted_from,
                &rec.file_size,
                &rec.width,
                &rec.height,
            ],
        )?;
        Ok(ImageRecord {
            id: row.get(0),
            project_id: rec.project_id,
            filename: rec.filename.clone(),
            file_path: rec.file_path.clone(),
            file_hash: rec.file_hash.clone(),
            phash: rec.phash.clone(),
            dhash: rec.dhash.clone(),
            ahash: rec.ahash.clone(),
            whash: rec.whash.clone(),
            colorhash: rec.colorhash.clone(),
            extracted_from: rec.extracted_from.clone(),
            file_size: rec.file_size,
            width: rec.width,
            height: rec.height,
            feature_status: row.get(2),
            created_at: row.get(1),
        })
    }

    fn get_image(&self, id: i64) -> anyhow::Result<ImageRecord> {
        let mut conn = self.conn()?;
        Self::row_to_image(&mut conn, "WHERE i.id = $1", &[&id])?
            .into_iter()
            .next()
            .with_context(|| format!("图像不存在: {id}"))
    }

    fn list_images(
        &self,
        project_id: i64,
        skip: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<ImageRecord>> {
        let mut conn = self.conn()?;
        Self::row_to_image(
            &mut conn,
            "WHERE i.project_id = $1 ORDER BY i.id LIMIT $2 OFFSET $3",
            &[&project_id, &limit, &skip],
        )
    }

    fn list_image_meta(&self, project_id: i64) -> anyhow::Result<Vec<ImageMeta>> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT id, filename, file_path, feature_status,
                        file_size, width, height
                 FROM images WHERE project_id = $1 ORDER BY id",
                &[&project_id],
            )?
            .iter()
            .map(|r| ImageMeta {
                id: r.get(0),
                filename: r.get(1),
                file_path: r.get(2),
                feature_status: r.get(3),
                file_size: r.get(4),
                width: r.get(5),
                height: r.get(6),
            })
            .collect();
        Ok(rows)
    }

    fn get_image_path(&self, id: i64) -> anyhow::Result<String> {
        let mut conn = self.conn()?;
        let path = conn
            .query_opt("SELECT file_path FROM images WHERE id = $1", &[&id])?
            .map(|r| r.get(0));
        path.with_context(|| format!("图像不存在: {id}"))
    }

    fn delete_image(&self, id: i64) -> anyhow::Result<ImageRecord> {
        let rec = self.get_image(id)?;
        let mut conn = self.conn()?;
        conn.execute(
            "DELETE FROM pair_cache WHERE hash_a = $1 OR hash_b = $1",
            &[&rec.file_hash],
        )?;
        conn.execute("DELETE FROM images WHERE id = $1", &[&id])?;
        Ok(rec)
    }

    // ---------- feature store ----------

    fn set_feature_status(&self, image_id: i64, status: &str) -> anyhow::Result<()> {
        let mut conn = self.conn()?;
        conn.execute(
            "UPDATE images SET feature_status = $1 WHERE id = $2",
            &[&status, &image_id],
        )?;
        Ok(())
    }

    fn put_feature(
        &self,
        image_id: i64,
        variant_idx: u8,
        algorithm: &str,
        vector: &[u8],
        dims: usize,
    ) -> anyhow::Result<()> {
        let mut conn = self.conn()?;
        let variant_idx = variant_idx as i64;
        let dims = dims as i64;
        conn.execute(
            "INSERT INTO feature_store(image_id, variant_idx, algorithm, vector, dims)
             VALUES($1,$2,$3,$4,$5)
             ON CONFLICT(image_id, variant_idx, algorithm) DO UPDATE SET
               vector = EXCLUDED.vector, dims = EXCLUDED.dims",
            &[&image_id, &variant_idx, &algorithm, &vector, &dims],
        )?;
        Ok(())
    }

    /// Batch-upsert all feature rows for one image in ONE transaction —
    /// same contract as the SQLite path.
    fn put_features(&self, image_id: i64, rows: &[(u8, &str, &[u8], usize)]) -> anyhow::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn()?;
        let mut tx = conn.transaction()?;
        {
            let stmt = tx.prepare(
                "INSERT INTO feature_store(image_id, variant_idx, algorithm, vector, dims)
                 VALUES($1,$2,$3,$4,$5)
                 ON CONFLICT(image_id, variant_idx, algorithm) DO UPDATE SET
                   vector = EXCLUDED.vector, dims = EXCLUDED.dims",
            )?;
            for &(variant_idx, algorithm, vector, dims) in rows {
                let variant_idx = variant_idx as i64;
                let dims = dims as i64;
                tx.execute(&stmt, &[&image_id, &variant_idx, &algorithm, &vector, &dims])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn load_feature_map(
        &self,
        image_ids: &[i64],
        feature: &str,
        variants: &[u8],
    ) -> anyhow::Result<FeatureMap> {
        let mut conn = self.conn()?;
        Self::load_feature_map_conn(&mut conn, image_ids, feature, variants)
    }

    fn load_feature_maps(
        &self,
        image_ids: &[i64],
        features: &[&str],
        variants: &[u8],
    ) -> anyhow::Result<Vec<FeatureMap>> {
        let mut conn = self.conn()?;
        features
            .iter()
            .map(|f| Self::load_feature_map_conn(&mut conn, image_ids, f, variants))
            .collect()
    }

    fn feature_algorithm_count(&self, image_id: i64) -> anyhow::Result<i64> {
        let mut conn = self.conn()?;
        let row = conn.query_one(
            "SELECT COUNT(DISTINCT algorithm) FROM feature_store
             WHERE image_id = $1 AND variant_idx = 0",
            &[&image_id],
        )?;
        Ok(row.get(0))
    }

    fn features_ready(&self, ids: &[i64]) -> anyhow::Result<bool> {
        if ids.is_empty() {
            return Ok(true);
        }
        let mut conn = self.conn()?;
        let ids_v: Vec<i64> = ids.to_vec();
        let n: i64 = conn
            .query_one(
                "SELECT COUNT(*) FROM images WHERE id = ANY($1) AND feature_status = 'ready'",
                &[&ids_v],
            )?
            .get(0);
        Ok(n as usize == ids.len())
    }

    // ---------- pair cache ----------

    fn get_pair_score(
        &self,
        hash_a: &str,
        hash_b: &str,
        algo: &str,
        rot_inv: bool,
    ) -> anyhow::Result<Option<f64>> {
        let (a, b) = crate::ordered_pair(hash_a, hash_b);
        let mut conn = self.conn()?;
        let score = conn
            .query_opt(
                "SELECT score FROM pair_cache
                 WHERE hash_a=$1 AND hash_b=$2 AND algorithm=$3 AND rotation_invariant=$4",
                &[&a, &b, &algo, &rot_inv],
            )?
            .map(|r| r.get(0));
        Ok(score)
    }

    fn put_pair_score(
        &self,
        hash_a: &str,
        hash_b: &str,
        algo: &str,
        rot_inv: bool,
        score: f64,
    ) -> anyhow::Result<()> {
        let (a, b) = crate::ordered_pair(hash_a, hash_b);
        let mut conn = self.conn()?;
        conn.execute(
            "INSERT INTO pair_cache(hash_a,hash_b,algorithm,rotation_invariant,score)
             VALUES($1,$2,$3,$4,$5)
             ON CONFLICT(hash_a,hash_b,algorithm,rotation_invariant)
             DO UPDATE SET score = EXCLUDED.score",
            &[&a, &b, &algo, &rot_inv, &score],
        )?;
        Ok(())
    }

    // ---------- analysis runs ----------

    fn insert_run(&self, run: &NewRun) -> anyhow::Result<i64> {
        let mut conn = self.conn()?;
        let row = conn.query_one(
            "INSERT INTO analysis_runs(project_id, algorithm, threshold,
                total_images, groups_count, unique_count, summary)
             VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING id",
            &[
                &run.project_id,
                &run.algorithm,
                &run.threshold,
                &run.total_images,
                &run.groups_count,
                &run.unique_count,
                &run.summary,
            ],
        )?;
        Ok(row.get(0))
    }

    fn list_runs(
        &self,
        project_id: i64,
        skip: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<AnalysisRunRecord>> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT id, project_id, algorithm, threshold, total_images,
                        groups_count, unique_count, summary, created_at
                 FROM analysis_runs WHERE project_id=$1 ORDER BY id DESC LIMIT $2 OFFSET $3",
                &[&project_id, &limit, &skip],
            )?
            .iter()
            .map(|r| AnalysisRunRecord {
                id: r.get(0),
                project_id: r.get(1),
                algorithm: r.get(2),
                threshold: r.get(3),
                total_images: r.get(4),
                groups_count: r.get(5),
                unique_count: r.get(6),
                summary: r.get(7),
                created_at: r.get(8),
            })
            .collect();
        Ok(rows)
    }

    fn get_run(&self, run_id: i64) -> anyhow::Result<AnalysisRunRecord> {
        let mut conn = self.conn()?;
        let found = conn
            .query_opt(
                "SELECT id, project_id, algorithm, threshold, total_images,
                        groups_count, unique_count, summary, created_at
                 FROM analysis_runs WHERE id=$1",
                &[&run_id],
            )?
            .map(|r| AnalysisRunRecord {
                id: r.get(0),
                project_id: r.get(1),
                algorithm: r.get(2),
                threshold: r.get(3),
                total_images: r.get(4),
                groups_count: r.get(5),
                unique_count: r.get(6),
                summary: r.get(7),
                created_at: r.get(8),
            });
        found.with_context(|| format!("分析记录不存在: {run_id}"))
    }

    fn latest_run(
        &self,
        project_id: i64,
        algorithm: &str,
        threshold: f64,
    ) -> anyhow::Result<Option<AnalysisRunRecord>> {
        let mut conn = self.conn()?;
        let run = conn
            .query_opt(
                "SELECT id, project_id, algorithm, threshold, total_images,
                        groups_count, unique_count, summary, created_at
                 FROM analysis_runs
                 WHERE project_id=$1 AND algorithm=$2 AND threshold=$3
                 ORDER BY id DESC LIMIT 1",
                &[&project_id, &algorithm, &threshold],
            )?
            .map(|r| AnalysisRunRecord {
                id: r.get(0),
                project_id: r.get(1),
                algorithm: r.get(2),
                threshold: r.get(3),
                total_images: r.get(4),
                groups_count: r.get(5),
                unique_count: r.get(6),
                summary: r.get(7),
                created_at: r.get(8),
            });
        Ok(run)
    }
}
