//! Blob storage backends: local filesystem or S3-compatible object store
//! (MinIO). Keys are slash-separated paths like `uploads/photo.jpg`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use object_store::ObjectStore;

pub trait BlobStore: Send + Sync {
    fn put(&self, key: &str, data: &[u8]) -> anyhow::Result<()>;
    fn get(&self, key: &str) -> anyhow::Result<Vec<u8>>;
    fn delete(&self, key: &str) -> anyhow::Result<()>;
    fn exists(&self, key: &str) -> bool;
    /// All keys under a prefix (e.g. "thumbnails/12_").
    fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>>;
    fn kind(&self) -> &'static str;
    /// Real filesystem path when the backend is local (None for object stores).
    fn local_path(&self, key: &str) -> Option<PathBuf>;
}

fn safe_key(key: &str) -> anyhow::Result<&str> {
    let k = key.trim_start_matches('/');
    if k.is_empty() || k.split('/').any(|seg| seg == ".." || seg.is_empty()) {
        anyhow::bail!("invalid key: {key}")
    }
    Ok(k)
}

/// Local filesystem rooted at `root`.
pub struct FsBlobStore {
    pub root: PathBuf,
}

impl FsBlobStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    fn full(&self, key: &str) -> anyhow::Result<PathBuf> {
        Ok(self.root.join(safe_key(key)?))
    }
}

impl BlobStore for FsBlobStore {
    fn put(&self, key: &str, data: &[u8]) -> anyhow::Result<()> {
        let p = self.full(key)?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(p, data)?;
        Ok(())
    }
    fn get(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        Ok(std::fs::read(self.full(key)?)?)
    }
    fn delete(&self, key: &str) -> anyhow::Result<()> {
        let p = self.full(key)?;
        if p.exists() {
            std::fs::remove_file(p)?;
        }
        Ok(())
    }
    fn exists(&self, key: &str) -> bool {
        self.full(key).map(|p| p.is_file()).unwrap_or(false)
    }
    fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        let base = self.full(prefix)?;
        let dir = if base.is_dir() {
            base
        } else {
            base.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| self.root.clone())
        };
        let mut out = Vec::new();
        if dir.is_dir() {
            for e in std::fs::read_dir(dir)? {
                let e = e?;
                let path = e.path();
                let rel = path.strip_prefix(&self.root).unwrap_or(&path).to_string_lossy().to_string();
                if rel.starts_with(prefix) {
                    out.push(rel);
                }
            }
        }
        Ok(out)
    }
    fn kind(&self) -> &'static str {
        "fs"
    }
    fn local_path(&self, key: &str) -> Option<PathBuf> {
        self.full(key).ok()
    }
}

/// S3-compatible object store (MinIO) via `object_store`. A dedicated
/// current-thread runtime keeps the `Store` API synchronous.
pub struct S3BlobStore {
    inner: object_store::aws::AmazonS3,
    rt: tokio::runtime::Runtime,
}

impl S3BlobStore {
    pub fn from_env() -> anyhow::Result<Self> {
        let endpoint =
            std::env::var("S3_ENDPOINT").context("S3_ENDPOINT required (e.g. http://localhost:9000)")?;
        let bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "itrace".into());
        let region = std::env::var("S3_REGION").unwrap_or_else(|_| "us-east-1".into());
        let ak = std::env::var("S3_ACCESS_KEY").context("S3_ACCESS_KEY required")?;
        let sk = std::env::var("S3_SECRET_KEY").context("S3_SECRET_KEY required")?;
        let inner = object_store::aws::AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_endpoint(endpoint)
            .with_region(region)
            .with_access_key_id(ak)
            .with_secret_access_key(sk)
            .with_allow_http(true)
            .with_virtual_hosted_style_request(false)
            .build()?;
        // multi_thread (1 worker): spawned tasks are driven autonomously by the
        // runtime's own worker — a current_thread runtime would never poll
        // them without someone calling block_on.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()?;
        Ok(Self { inner, rt })
    }

    fn obj_path(key: &str) -> anyhow::Result<object_store::path::Path> {
        Ok(object_store::path::Path::from(safe_key(key)?))
    }

    /// Run a future on the store's private runtime from any thread —
    /// `block_on` panics inside a tokio context, so we spawn the task and
    /// wait on a channel (works under async handlers and spawn_blocking).
    fn run<T: Send + 'static>(
        &self,
        fut: impl std::future::Future<Output = T> + Send + 'static,
    ) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rt.spawn(async move {
            let _ = tx.send(fut.await);
        });
        rx.recv().expect("s3 task dropped")
    }
}

impl BlobStore for S3BlobStore {
    fn put(&self, key: &str, data: &[u8]) -> anyhow::Result<()> {
        let p = Self::obj_path(key)?;
        let inner = self.inner.clone();
        let payload = object_store::PutPayload::from(data.to_vec());
        self.run(async move { inner.put(&p, payload).await })?;
        Ok(())
    }
    fn get(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        let p = Self::obj_path(key)?;
        let inner = self.inner.clone();
        let bytes = self.run(async move {
            let r = inner.get(&p).await?;
            r.bytes().await.map(|b| b.to_vec())
        })?;
        Ok(bytes)
    }
    fn delete(&self, key: &str) -> anyhow::Result<()> {
        let p = Self::obj_path(key)?;
        let inner = self.inner.clone();
        match self.run(async move { inner.delete(&p).await }) {
            Ok(()) => Ok(()),
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
    fn exists(&self, key: &str) -> bool {
        let p = match Self::obj_path(key) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let inner = self.inner.clone();
        self.run(async move { inner.head(&p).await.is_ok() })
    }
    fn list(&self, prefix: &str) -> anyhow::Result<Vec<String>> {
        use futures::StreamExt;
        let p = object_store::path::Path::from(prefix.trim_start_matches('/'));
        let inner = self.inner.clone();
        let metas = self.run(async move { inner.list(Some(&p)).collect::<Vec<_>>().await });
        Ok(metas.into_iter().filter_map(|m| m.ok().map(|m| m.location.to_string())).collect())
    }
    fn kind(&self) -> &'static str {
        "s3"
    }
    fn local_path(&self, _key: &str) -> Option<PathBuf> {
        None
    }
}

/// Select backend from `ITRACE_STORAGE` (`fs` default, `s3` for MinIO).
/// `local_root` anchors the fs backend (the data dir).
pub fn blob_store_from_env(local_root: &std::path::Path) -> anyhow::Result<Arc<dyn BlobStore>> {
    match std::env::var("ITRACE_STORAGE").unwrap_or_else(|_| "fs".into()).as_str() {
        "s3" => Ok(Arc::new(S3BlobStore::from_env()?)),
        _ => Ok(Arc::new(FsBlobStore::new(local_root))),
    }
}
