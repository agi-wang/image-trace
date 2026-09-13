//! Semantic recall channel (Phase 4): dense-embedding ANN retrieval.
//!
//! The gate and crop channels index discrete 64-bit hash keys; they cannot
//! recall pairs that survive only in continuous embedding space (heavy
//! appearance transforms that keep scene semantics). This module adds the
//! third channel's foundation:
//!
//! - [`SemanticEmbedder`] — pluggable image → `f32` embedding backend. The
//!   intended production impl is a DINOv2 ONNX model loaded from a weights
//!   path (follow-up; CI never downloads weights). Tests and wiring use
//!   lightweight in-crate stubs.
//! - [`HnswIndex`] — in-memory HNSW (Hierarchical Navigable Small World)
//!   approximate k-NN index over cosine-normalized vectors: `insert` +
//!   `query(k)`. Deterministic: node levels come from a seeded SplitMix64
//!   stream, so identical insert sequences build identical graphs.
//! - [`semantic_candidates`] / [`semantic_candidates_with_index`] — the
//!   recall half of the channel, emitting `(entry_i, entry_j)` candidate
//!   pairs under the same contract as `index::dedup_candidates*` /
//!   `index::crop_candidates*` so they can be unioned directly.
//!
//! # Dedup wiring (opt-in, `ITRACE_SEMANTIC=1`)
//!
//! The channel is **off by default**. When [`semantic_channel_enabled`]
//! and [`embedder_from_env`] resolves a backend, both dedup entry points
//! (`itrace-cli dedup` and the server's `/dedup` scan) run it:
//!
//! 1. Embed: each image in the gate scan's `entries` is embedded via
//!    `embedder.embed_bytes(blob)` — per-scan decoding today (R2 dev
//!    path); a persisted feature/`project_{id}_sem/` bundle mirroring
//!    `ITMIHP1` is the follow-up.
//! 2. Recall: [`semantic_candidates`] builds an in-memory [`HnswIndex`]
//!    over those vectors and emits `(entry_i, entry_j)` candidate pairs
//!    at `resolve_semantic_k` / `resolve_semantic_min_cosine`.
//! 3. Union + verify: [`union_candidate_pairs`] folds them into the MIH
//!    candidate set **before** `confirm_pairs` / the variant-max hash
//!    re-score. Semantic hits are *candidates only* — a pair still needs
//!    the existing hash/crop confirmation, so the channel can add recall
//!    but cannot silently widen merges. With the flag off the path is
//!    skipped entirely and dedup output is bit-identical.
//!
//! Backend resolution: `ITRACE_SEMANTIC_MODEL` names a DINOv2 ONNX
//! weights path — with the `semantic-onnx` cargo feature an
//! `ort`-backed [`crate::semantic_onnx::OnnxSemanticEmbedder`] loads it
//! (missing/invalid weights or a missing runtime dylib → inert, never
//! silently the stub). Without the feature a configured model path stays
//! inert too. The stub is used only when `ITRACE_SEMANTIC_STUB=1` or
//! when the channel is armed with no model path.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;

/// Pluggable image → dense-embedding backend (DINOv2 or equivalent).
///
/// Implementations return raw `f32` embeddings; [`HnswIndex`] L2-normalizes
/// internally, so backends need not normalize. Object-safe: dedup will hold
/// `Box<dyn SemanticEmbedder>` / `Arc<dyn SemanticEmbedder>`. `Send + Sync`
/// because dedup embeds the batch in parallel.
pub trait SemanticEmbedder: Send + Sync {
    /// Output embedding dimension (e.g. 384 for DINOv2-vits14, 768 for base).
    fn dim(&self) -> usize;
    /// Embed raw image file bytes → embedding vector of length [`Self::dim`].
    fn embed_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>>;
    /// Embed an image file from disk. Default: read bytes, then
    /// [`Self::embed_bytes`]; backends with a streaming decoder may override.
    fn embed_path(&self, path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
        self.embed_bytes(&std::fs::read(path)?)
    }
}

/// Production embedder seam: resolves the configured embedding backend.
///
/// - `ITRACE_SEMANTIC` unset/off → `None` (channel disarmed; default).
/// - Armed + `ITRACE_SEMANTIC_MODEL` set + `ITRACE_SEMANTIC_STUB` unset →
///   try the ONNX backend (`semantic-onnx` feature): a configured
///   production weights path that fails to load — missing/invalid file,
///   missing runtime dylib, or a build without the feature — resolves
///   to `None` (inert), never silently to the stub.
/// - Armed otherwise (no model path, or `ITRACE_SEMANTIC_STUB=1`) →
///   [`StubEmbedder`]. The stub is a deterministic dev/test stand-in —
///   **not** semantic DINOv2 embeddings.
pub fn embedder_from_env() -> Option<Box<dyn SemanticEmbedder>> {
    if !semantic_channel_enabled() {
        return None;
    }
    let model = std::env::var(SEMANTIC_MODEL_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty());
    let stub_forced = std::env::var(SEMANTIC_STUB_ENV)
        .map(|v| semantic_flag_on(&v))
        .unwrap_or(false);
    resolve_embedder(model.as_deref(), stub_forced, onnx_load)
}

/// Backend precedence, separated from env reads so the load boundary is
/// mockable in tests (CI has no ONNX weights or runtime dylib).
fn resolve_embedder(
    model: Option<&str>,
    stub_forced: bool,
    load_onnx: impl FnOnce(&std::path::Path) -> Option<Box<dyn SemanticEmbedder>>,
) -> Option<Box<dyn SemanticEmbedder>> {
    match (model, stub_forced) {
        // Production path: attempt ONNX; on any load failure the loader
        // returns `None` and the channel stays inert — never the stub.
        (Some(path), false) => load_onnx(std::path::Path::new(path.trim())),
        // `ITRACE_SEMANTIC_STUB=1` or armed with no model path.
        _ => Some(Box::new(StubEmbedder::default())),
    }
}

/// `semantic-onnx` build: attempt the real ONNX backend; any failure
/// logs a line and resolves to inert (`None`) — never the stub.
#[cfg(feature = "semantic-onnx")]
fn onnx_load(path: &std::path::Path) -> Option<Box<dyn SemanticEmbedder>> {
    match crate::semantic_onnx::OnnxSemanticEmbedder::load(path) {
        Ok(e) => Some(Box::new(e)),
        Err(e) => {
            eprintln!("itrace: semantic model load failed; channel inert ({e:#})");
            None
        }
    }
}

/// Feature-off build: a configured model path stays inert (the backend
/// doesn't exist in this build) — still never the stub.
#[cfg(not(feature = "semantic-onnx"))]
fn onnx_load(_path: &std::path::Path) -> Option<Box<dyn SemanticEmbedder>> {
    None
}

/// Env flag arming the semantic recall channel.
pub const SEMANTIC_ENV: &str = "ITRACE_SEMANTIC";
/// Env var naming the production embedding weights file (DINOv2 ONNX;
/// loaded by `semantic_onnx::OnnxSemanticEmbedder` when built with the
/// `semantic-onnx` feature).
pub const SEMANTIC_MODEL_ENV: &str = "ITRACE_SEMANTIC_MODEL";
/// Env var forcing [`StubEmbedder`] even when `ITRACE_SEMANTIC_MODEL` is
/// configured.
pub const SEMANTIC_STUB_ENV: &str = "ITRACE_SEMANTIC_STUB";
/// Default k for per-image ANN probes in the semantic channel.
pub const DEFAULT_SEMANTIC_K: usize = 32;
/// Default minimum cosine similarity for a semantic candidate pair.
pub const DEFAULT_SEMANTIC_MIN_COSINE: f32 = 0.75;

/// `ITRACE_SEMANTIC` parser: any non-empty value except `0`/`false`/`off`
/// (case-insensitive) enables the channel.
fn semantic_flag_on(v: &str) -> bool {
    let v = v.trim();
    !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
}

/// Whether the semantic recall channel is armed (`ITRACE_SEMANTIC=1` …).
/// Arming without an embedder + stored embeddings is a no-op; default
/// (unset) is off.
pub fn semantic_channel_enabled() -> bool {
    std::env::var(SEMANTIC_ENV)
        .map(|v| semantic_flag_on(&v))
        .unwrap_or(false)
}

/// Resolve semantic probe width k. Precedence: `override_k` > env
/// `ITRACE_SEMANTIC_K` > [`DEFAULT_SEMANTIC_K`]; clamped to ≥1.
pub fn resolve_semantic_k(override_k: Option<usize>) -> usize {
    override_k
        .or_else(|| {
            std::env::var("ITRACE_SEMANTIC_K")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or(DEFAULT_SEMANTIC_K)
        .max(1)
}

/// Resolve the semantic candidate cosine floor. Precedence: `override_c` >
/// env `ITRACE_SEMANTIC_MIN_COS` > [`DEFAULT_SEMANTIC_MIN_COSINE`];
/// clamped to `[-1, 1]`.
pub fn resolve_semantic_min_cosine(override_c: Option<f32>) -> f32 {
    override_c
        .or_else(|| {
            std::env::var("ITRACE_SEMANTIC_MIN_COS")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
        })
        .unwrap_or(DEFAULT_SEMANTIC_MIN_COSINE)
        .clamp(-1.0, 1.0)
}

/// Default [`StubEmbedder`] grid side: 16×16 → a 256-dim embedding.
pub const DEFAULT_STUB_GRID: usize = 16;

/// Deterministic dev/test embedder — **not** a semantic model.
///
/// Decodes the image, converts to grayscale, block-averages it onto a
/// `grid × grid` layout and mean-subtracts (a coarse content-smooth
/// projection, closer to a tiny perceptual embedding than to DINOv2).
/// Near-duplicate images land at high cosine, unrelated ones near 0, so
/// the channel can be exercised end-to-end without weights. Bytes that
/// don't decode fall back to a hash-derived vector (still deterministic)
/// so a corrupt blob can never fail a scan.
pub struct StubEmbedder {
    /// Downsample grid side; `dim() = grid²`.
    grid: usize,
}

impl StubEmbedder {
    /// `grid × grid` grayscale-block embedding (`dim = grid²`).
    pub fn with_grid(grid: usize) -> Self {
        assert!((1..=64).contains(&grid), "stub grid out of range");
        Self { grid }
    }

    /// Block-average `gray` onto `grid × grid` cells, then mean-subtract so
    /// a global brightness shift doesn't dominate cosine. Degenerate
    /// (zero-area) images embed to the zero vector — cosine 0 against
    /// everything, i.e. never a candidate.
    fn embed_gray(&self, gray: &crate::GrayImage) -> Vec<f32> {
        let g = self.grid;
        let mut out = vec![0.0f32; g * g];
        let (w, h) = (gray.width as usize, gray.height as usize);
        if w == 0 || h == 0 {
            return out;
        }
        for cy in 0..g {
            let y0 = cy * h / g;
            let y1 = (((cy + 1) * h) / g).max(y0 + 1).min(h);
            for cx in 0..g {
                let x0 = cx * w / g;
                let x1 = (((cx + 1) * w) / g).max(x0 + 1).min(w);
                let mut sum = 0.0f64;
                for y in y0..y1 {
                    for &px in &gray.data[y * w + x0..y * w + x1] {
                        sum += px as f64;
                    }
                }
                out[cy * g + cx] = (sum / ((y1 - y0) * (x1 - x0)) as f64) as f32;
            }
        }
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        for v in &mut out {
            *v -= mean;
        }
        out
    }
}

impl Default for StubEmbedder {
    fn default() -> Self {
        Self::with_grid(DEFAULT_STUB_GRID)
    }
}

/// Deterministic pseudo-embedding for bytes that don't decode as an image:
/// a SplitMix64 stream seeded by blake3 — stable across runs and
/// near-orthogonal to real image vectors.
fn hash_embed(dim: usize, bytes: &[u8]) -> Vec<f32> {
    let h = blake3::hash(bytes);
    let mut s = u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap());
    (0..dim)
        .map(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // uniform in [-1, 1)
            ((z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) * 2.0 - 1.0) as f32
        })
        .collect()
}

impl SemanticEmbedder for StubEmbedder {
    fn dim(&self) -> usize {
        self.grid * self.grid
    }

    fn embed_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        match crate::image_io::decode(bytes) {
            Ok(img) => Ok(self.embed_gray(&crate::image_io::to_gray(&img))),
            Err(_) => Ok(hash_embed(self.dim(), bytes)),
        }
    }
}

/// Cosine similarity of two equal-length vectors (both normalized here, so
/// callers may pass raw embeddings). Returns 0.0 for a zero vector.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine dim mismatch");
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = na.sqrt() * nb.sqrt();
    if d <= 0.0 {
        0.0
    } else {
        (dot / d).clamp(-1.0, 1.0)
    }
}

/// L2-normalized copy of `v`; a zero vector normalizes to zeros (cosine 0
/// against everything — a valid "no neighbors" state, not a panic).
fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n <= 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / n).collect()
}

/// Cosine distance `1 - cos` on already-normalized vectors, clamped to
/// `[0, 2]` (float noise can push the raw dot slightly past ±1).
#[inline]
fn cos_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (1.0 - dot).clamp(0.0, 2.0)
}

/// Heap element: `(node, cosine distance)`. `Ord` makes `BinaryHeap<Cand>`
/// a max-heap on distance (peek = worst) and `BinaryHeap<Reverse<Cand>>`
/// a min-heap (pop = best).
#[derive(Clone, Copy, PartialEq)]
struct Cand {
    node: u32,
    dist: f32,
}

impl Eq for Cand {}

impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.node.cmp(&o.node))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct Node {
    /// External label (dense owner slot — see [`semantic_candidates`]).
    id: u32,
    /// L2-normalized embedding.
    vec: Vec<f32>,
    /// `neighbors[l]` = node ids linked on layer `l`; `neighbors.len() - 1`
    /// is the node's level.
    neighbors: Vec<Vec<u32>>,
}

/// In-memory HNSW approximate nearest-neighbor index over cosine-normalized
/// `f32` embeddings.
///
/// Subset of Malkov & Yashunin (2016): probabilistic levels, greedy descent
/// to the insertion layer, `ef_construction` beam search for connections,
/// neighbour lists pruned to `m` (layer 0: `2m`). Query = greedy descent to
/// layer 0, then an `ef_search` beam search returning the best `k` by
/// cosine. Deterministic for a fixed seed and insert order — SplitMix64
/// supplies level draws (no `rand` dep).
pub struct HnswIndex {
    dim: usize,
    /// Max neighbours per node on layers ≥ 1; layer 0 allows `2*m`.
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    /// `1/ln(m)` — level distribution scale.
    level_scale: f64,
    /// SplitMix64 state for level assignment.
    rng: u64,
    /// Entry point node + its level (the graph's top layer).
    entry: Option<u32>,
    max_level: u32,
    nodes: Vec<Node>,
}

/// Cap on generated levels — bounds the per-node layer vector; the
/// uncapped tail is astronomically rare anyway.
const MAX_LEVEL: u32 = 30;
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

impl HnswIndex {
    /// Default parameters: `m = 16`, `ef_construction = 200`,
    /// `ef_search = 64`, fixed seed (deterministic builds).
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, 16, 200, 64, DEFAULT_SEED)
    }

    pub fn with_params(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
    ) -> Self {
        assert!(dim > 0, "dim must be >= 1");
        let m = m.max(2);
        Self {
            dim,
            m,
            ef_construction: ef_construction.max(m),
            ef_search: ef_search.max(1),
            level_scale: 1.0 / (m as f64).ln(),
            rng: seed,
            entry: None,
            max_level: 0,
            nodes: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Exponentially decaying level draw: P(level ≥ l) = m^-l.
    fn gen_level(&mut self) -> u32 {
        let u = ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
        let u = u.max(1e-300); // keep ln finite
        (-u.ln() * self.level_scale).min(MAX_LEVEL as f64) as u32
    }

    /// Max neighbours allowed on `layer` (layer 0 is denser).
    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m * 2
        } else {
            self.m
        }
    }

    /// Greedy single-path descent on `layer`: repeatedly move to the closest
    /// neighbour until no improvement.
    fn greedy_closest(&self, q: &[f32], ep: u32, layer: usize) -> u32 {
        let mut cur = ep;
        let mut cur_d = cos_dist(q, &self.nodes[cur as usize].vec);
        loop {
            let mut improved = false;
            for &n in &self.nodes[cur as usize].neighbors[layer] {
                let d = cos_dist(q, &self.nodes[n as usize].vec);
                if d < cur_d {
                    cur = n;
                    cur_d = d;
                    improved = true;
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Beam search on one layer: returns the `ef` closest visited nodes to
    /// `q`, sorted by distance ascending.
    fn search_layer(&self, q: &[f32], ep: u32, ef: usize, layer: usize) -> Vec<Cand> {
        let mut visited = vec![false; self.nodes.len()];
        let mut cand: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();
        let d0 = cos_dist(q, &self.nodes[ep as usize].vec);
        visited[ep as usize] = true;
        let c0 = Cand { node: ep, dist: d0 };
        cand.push(Reverse(c0));
        results.push(c0);
        while let Some(Reverse(c)) = cand.pop() {
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            for &n in &self.nodes[c.node as usize].neighbors[layer] {
                if visited[n as usize] {
                    continue;
                }
                visited[n as usize] = true;
                let d = cos_dist(q, &self.nodes[n as usize].vec);
                let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
                if d < worst || results.len() < ef {
                    let cn = Cand { node: n, dist: d };
                    cand.push(Reverse(cn));
                    results.push(cn);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<Cand> = results.into_vec();
        out.sort_unstable();
        out
    }

    /// Insert `vec` with external label `id` (typically a dense owner slot).
    /// `vec` need not be normalized. Panics on dim mismatch.
    ///
    /// The node is pushed before linking so prune sorts can distance
    /// against `self.nodes[idx]` directly; its empty neighbour lists make
    /// it unreachable to the beam searches that follow.
    pub fn insert(&mut self, id: u32, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "embedding dim mismatch");
        let v = l2_normalized(vec);
        let level = self.gen_level();
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            id,
            vec: v,
            neighbors: vec![Vec::new(); level as usize + 1],
        });
        let Some(mut ep) = self.entry else {
            self.entry = Some(idx);
            self.max_level = level;
            return;
        };
        // Greedy descent through layers the new node doesn't reach.
        let mut l = self.max_level;
        while l > level {
            ep = self.greedy_closest(&self.nodes[idx as usize].vec, ep, l as usize);
            l -= 1;
        }
        // Beam-search + link on every shared layer, top-down.
        for l in (0..=level.min(self.max_level)).rev() {
            let lu = l as usize;
            let cands =
                self.search_layer(&self.nodes[idx as usize].vec, ep, self.ef_construction, lu);
            if let Some(first) = cands.first() {
                ep = first.node;
            }
            let max = self.m_max(lu);
            let chosen: Vec<u32> = cands.iter().take(max).map(|c| c.node).collect();
            self.nodes[idx as usize].neighbors[lu] = chosen.clone();
            // Back-link + prune each chosen neighbour to its `max`
            // closest (the new node included).
            for &n in &chosen {
                let mut nb = std::mem::take(&mut self.nodes[n as usize].neighbors[lu]);
                nb.push(idx);
                if nb.len() > max {
                    let nv = &self.nodes[n as usize].vec;
                    nb.sort_unstable_by(|&a, &b| {
                        cos_dist(nv, &self.nodes[a as usize].vec)
                            .partial_cmp(&cos_dist(nv, &self.nodes[b as usize].vec))
                            .unwrap_or(Ordering::Equal)
                    });
                    nb.truncate(max);
                }
                self.nodes[n as usize].neighbors[lu] = nb;
            }
        }
        if level > self.max_level {
            self.entry = Some(idx);
            self.max_level = level;
        }
    }

    /// Approximate k nearest neighbours: `(id, cosine)` sorted by cosine
    /// descending (ties by id), at most `k` entries, deduplicated by `id`
    /// (two nodes may share a label). Panics on dim mismatch.
    pub fn query(&self, vec: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(vec.len(), self.dim, "embedding dim mismatch");
        let Some(mut ep) = self.entry else {
            return Vec::new();
        };
        let q = l2_normalized(vec);
        for l in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&q, ep, l as usize);
        }
        let cands = self.search_layer(&q, ep, self.ef_search.max(k), 0);
        // node → (id, cosine); keep the best cosine per external id.
        let mut best: HashMap<u32, f32> = HashMap::new();
        for c in cands {
            let id = self.nodes[c.node as usize].id;
            let sim = 1.0 - c.dist;
            best.entry(id)
                .and_modify(|s| *s = s.max(sim))
                .or_insert(sim);
        }
        let mut out: Vec<(u32, f32)> = best.into_iter().collect();
        out.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k);
        out
    }
}

/// One image's semantic embedding for the recall channel — mirrors
/// `index::CropKeys` (`image_id` + payload).
#[derive(Clone)]
pub struct SemanticVecs {
    pub image_id: i64,
    pub vector: Vec<f32>,
}

/// Sorted image_ids (owner slots) and per-entry owner ids — same dense
/// `u32` owner plan as the gate/crop bundles, so a future persisted
/// `project_{id}_sem/` bundle can reuse the `image_ids.bin` contract.
fn owner_plan(entries: &[SemanticVecs]) -> (Vec<i64>, Vec<u32>) {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&i| entries[i].image_id);
    let image_ids: Vec<i64> = order.iter().map(|&i| entries[i].image_id).collect();
    let mut owner_for_entry = vec![0u32; entries.len()];
    for (owner, &ei) in order.iter().enumerate() {
        owner_for_entry[ei] = owner as u32;
    }
    (image_ids, owner_for_entry)
}

/// Candidate pairs from a pre-built [`HnswIndex`]. `idx` owners are dense
/// slots into `image_ids` (same contract as the MIH bundles); returned
/// pairs are **entry indices** into `entries`, normalized `(min, max)`.
///
/// Each entry probes for `k + 1` neighbours — its own vector sits in the
/// index and takes one slot — and keeps hits with cosine ≥ `min_cosine`.
pub fn semantic_candidates_with_index(
    entries: &[SemanticVecs],
    idx: &HnswIndex,
    image_ids: &[i64],
    k: usize,
    min_cosine: f32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() {
        return Vec::new();
    }
    // owner slot → entry index in the caller's `entries` slice
    let id_to_entry: HashMap<i64, u32> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.image_id, i as u32))
        .collect();
    let entry_of_owner: Vec<Option<u32>> = image_ids
        .iter()
        .map(|id| id_to_entry.get(id).copied())
        .collect();

    let mut out: Vec<(u32, u32)> = entries
        .par_iter()
        .enumerate()
        .flat_map(|(i, e)| {
            let i = i as u32;
            idx.query(&e.vector, k.saturating_add(1))
                .into_iter()
                .filter(|(_, sim)| *sim >= min_cosine)
                .filter_map(|(owner, _)| entry_of_owner.get(owner as usize).copied().flatten())
                .filter(|&j| j != i)
                .map(|j| (j.min(i), j.max(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// In-memory sibling of [`semantic_candidates_with_index`]: builds an
/// [`HnswIndex`] over `entries` and queries it — the semantic analogue of
/// `index::crop_candidates`. All vectors must share one dimension.
pub fn semantic_candidates(entries: &[SemanticVecs], k: usize, min_cosine: f32) -> Vec<(u32, u32)> {
    if entries.is_empty() {
        return Vec::new();
    }
    let (image_ids, owner_for_entry) = owner_plan(entries);
    let mut idx = HnswIndex::new(entries[0].vector.len());
    for (i, e) in entries.iter().enumerate() {
        idx.insert(owner_for_entry[i], &e.vector);
    }
    semantic_candidates_with_index(entries, &idx, &image_ids, k, min_cosine)
}

/// Union extra `(i, j)` candidate pairs into `base`: normalize each to
/// `(min, max)`, drop self-pairs, sort, dedup. This is the merge step the
/// dedup scan uses to fold semantic-channel pairs into the MIH/crop
/// candidate set before verification.
pub fn union_candidate_pairs(
    base: &mut Vec<(u32, u32)>,
    extra: impl IntoIterator<Item = (u32, u32)>,
) {
    base.extend(
        extra
            .into_iter()
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (a.min(b), a.max(b))),
    );
    base.sort_unstable();
    base.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic SplitMix64 — same pattern as `ownership` tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn f64(&mut self) -> f64 {
            ((self.next() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        }

        /// ~N(0,1) via 12-uniform sum.
        fn gauss(&mut self) -> f32 {
            let s: f64 = (0..12).map(|_| self.f64()).sum();
            (s - 6.0) as f32
        }

        fn gauss_vec(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.gauss()).collect()
        }
    }

    /// Clustered embedding set: `nc` far-apart gaussian centres (pairwise
    /// cosine ≈ 0 in high dim), `per` members each = centre + small noise.
    /// Returns (vectors, cluster_of).
    fn clustered(
        dim: usize,
        nc: usize,
        per: usize,
        noise: f32,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rng = Rng(seed);
        let centers: Vec<Vec<f32>> = (0..nc).map(|_| rng.gauss_vec(dim)).collect();
        let mut vecs = Vec::new();
        let mut owner = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..per {
                let v: Vec<f32> = ctr.iter().map(|&x| x + noise * rng.gauss()).collect();
                vecs.push(v);
                owner.push(c);
            }
        }
        (vecs, owner)
    }

    /// Exact top-k by cosine (brute force) — ground truth for recall tests.
    fn exact_topk(q: &[f32], vecs: &[Vec<f32>], k: usize, skip: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != skip)
            .map(|(i, v)| (i, cosine_similarity(q, v)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn hnsw_recalls_same_cluster_neighbors() {
        let dim = 64;
        let (vecs, owner) = clustered(dim, 6, 40, 0.15, 1);
        let mut idx = HnswIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u32, v);
        }
        // Every member's top-10 must be same-cluster, and a held-out probe
        // near a centre must land in that cluster too.
        for (i, v) in vecs.iter().enumerate() {
            let hits = idx.query(v, 10);
            let own = hits
                .iter()
                .filter(|&&(id, _)| owner[id as usize] == owner[i])
                .count();
            assert_eq!(own, 10, "entry {i}: top-10 not all same-cluster: {hits:?}");
            assert!(hits.iter().all(|&(_, s)| s >= 0.8));
        }
        let mut rng = Rng(7);
        for c in 0..6usize {
            let probe: Vec<f32> = {
                // rebuild centre c deterministically: centres were drawn first
                let mut r = Rng(1);
                let centers: Vec<Vec<f32>> = (0..6).map(|_| r.gauss_vec(dim)).collect();
                centers[c].iter().map(|&x| x + 0.1 * rng.gauss()).collect()
            };
            let hits = idx.query(&probe, 5);
            assert!(
                hits.iter().all(|&(id, _)| owner[id as usize] == c),
                "probe for cluster {c} hit foreign clusters: {hits:?}"
            );
        }
    }

    #[test]
    fn hnsw_far_clusters_not_in_topk() {
        let dim = 64;
        let (vecs, owner) = clustered(dim, 4, 50, 0.2, 2);
        let mut idx = HnswIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u32, v);
        }
        // Query a cluster-0 member: no foreign-cluster id may appear in top-8.
        let hits = idx.query(&vecs[0], 8);
        assert!(!hits.is_empty());
        for &(id, sim) in &hits {
            assert_eq!(
                owner[id as usize], owner[0],
                "foreign neighbour {id} sim {sim}"
            );
        }
    }

    #[test]
    fn hnsw_recall_vs_brute_force() {
        let dim = 48;
        let n = 1500;
        let k = 8;
        let mut rng = Rng(3);
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| rng.gauss_vec(dim)).collect();
        let mut idx = HnswIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u32, v);
        }
        let mut hit = 0usize;
        let mut total = 0usize;
        let mut top1_exact = 0usize;
        for (qi, q) in vecs.iter().enumerate().step_by(37) {
            let exact = exact_topk(q, &vecs, k, qi);
            let approx: Vec<u32> = idx.query(q, k + 1).into_iter().map(|(id, _)| id).collect();
            let approx: Vec<usize> = approx
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| i != qi)
                .take(k)
                .collect();
            hit += exact.iter().filter(|e| approx.contains(e)).count();
            total += exact.len();
            if exact.first() == approx.first() {
                top1_exact += 1;
            }
        }
        let recall = hit as f64 / total as f64;
        assert!(
            recall >= 0.9,
            "recall@{k} = {recall:.3} ({hit}/{total}) below 0.9"
        );
        assert!(top1_exact >= 35, "top-1 exact only {top1_exact}/41");
    }

    #[test]
    fn hnsw_edge_cases() {
        let mut idx = HnswIndex::new(4);
        assert!(idx.is_empty());
        assert!(idx.query(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
        idx.insert(7, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
        let hits = idx.query(&[1.0, 0.0, 0.0, 0.0], 5); // k > len → all
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 7);
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
        // zero vector inserts fine, cosine ~0 to everything
        idx.insert(9, &[0.0; 4]);
        let h2 = idx.query(&[0.0; 4], 8);
        assert_eq!(h2.len(), 2);
        assert!(h2.iter().all(|&(_, s)| s.is_finite()));
    }

    #[test]
    #[should_panic(expected = "dim mismatch")]
    fn hnsw_insert_dim_mismatch_panics() {
        let mut idx = HnswIndex::new(4);
        idx.insert(0, &[1.0, 2.0]);
    }

    #[test]
    fn semantic_candidates_only_same_cluster() {
        // Clusters with controlled separation: A and B centres sit at
        // cosine ≈ 0.7, C is orthogonal (cosine ≈ 0). A 0.85 floor admits
        // only within-cluster pairs; 0.5 additionally admits A↔B pairs —
        // the same k, so only the cosine gate differs.
        let dim = 32;
        let mut rng = Rng(5);
        let e = |i: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0;
            v
        };
        // centre B = normalize(0.7·e0 + 0.7·e1) → cos(A,B) ≈ 0.7
        let ctr = [
            e(0),
            l2_normalized(
                &e(0)
                    .iter()
                    .zip(&e(1))
                    .map(|(a, b)| 0.7 * a + 0.7 * b)
                    .collect::<Vec<_>>(),
            ),
            e(2),
        ];
        // 8 members/cluster, k=12 → each probe's top-k reaches past its
        // own cluster, so the cosine floor is what filters foreigners.
        let mut vecs = Vec::new();
        let mut owner = Vec::new();
        for (c, center) in ctr.iter().enumerate() {
            for _ in 0..8 {
                vecs.push(
                    center
                        .iter()
                        .map(|&x| x + 0.05 * rng.gauss())
                        .collect::<Vec<f32>>(),
                );
                owner.push(c);
            }
        }
        let entries: Vec<SemanticVecs> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| SemanticVecs {
                image_id: (i * 7 + 3) as i64, // unsorted ids exercise owner plan
                vector: v.clone(),
            })
            .collect();
        let strict = semantic_candidates(&entries, 12, 0.85);
        assert!(!strict.is_empty());
        for &(a, b) in &strict {
            assert!(a < b);
            assert_eq!(
                owner[a as usize], owner[b as usize],
                "cross-cluster pair ({a},{b})"
            );
        }
        // Same k, lower floor → superset incl. cross-cluster A↔B pairs.
        let loose = semantic_candidates(&entries, 12, 0.5);
        assert!(loose.len() > strict.len());
        assert!(loose
            .iter()
            .any(|&(a, b)| owner[a as usize] != owner[b as usize]));
        // C stays isolated even at the loose floor.
        assert!(!loose
            .iter()
            .any(|&(a, b)| (owner[a as usize] == 2) != (owner[b as usize] == 2)));
    }

    #[test]
    fn semantic_candidates_with_index_maps_owners() {
        // owner slots come from sorted image_ids, not entry order
        let entries = vec![
            SemanticVecs {
                image_id: 9,
                vector: vec![1.0, 0.0],
            },
            SemanticVecs {
                image_id: 1,
                vector: vec![0.99, 0.01],
            },
            SemanticVecs {
                image_id: 5,
                vector: vec![0.0, 1.0],
            },
        ];
        let image_ids = vec![1i64, 5, 9];
        let mut idx = HnswIndex::new(2);
        idx.insert(2, &entries[0].vector); // id 9 → slot 2
        idx.insert(0, &entries[1].vector); // id 1 → slot 0
        idx.insert(1, &entries[2].vector); // id 5 → slot 1
        let pairs = semantic_candidates_with_index(&entries, &idx, &image_ids, 4, 0.9);
        assert_eq!(pairs, vec![(0, 1)]); // entries 0&1 are the near pair
    }

    #[test]
    fn union_candidate_pairs_normalizes_and_dedups() {
        let mut base = vec![(0u32, 3u32), (1, 4)];
        union_candidate_pairs(&mut base, vec![(3, 0), (2, 5), (4, 1), (7, 7)]);
        assert_eq!(base, vec![(0, 3), (1, 4), (2, 5)]);
    }

    #[test]
    fn embedder_trait_object_and_path() {
        struct Stub;
        impl SemanticEmbedder for Stub {
            fn dim(&self) -> usize {
                4
            }
            fn embed_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
                // deterministic pseudo-embedding from bytes
                Ok(vec![
                    bytes.len() as f32,
                    bytes.iter().map(|&b| b as f32).sum::<f32>(),
                    1.0,
                    0.0,
                ])
            }
        }
        let e: Box<dyn SemanticEmbedder> = Box::new(Stub);
        let v = e.embed_bytes(b"abc").unwrap();
        assert_eq!(v.len(), e.dim());
        let dir = std::env::temp_dir().join(format!("itrace-sem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.bin");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(e.embed_path(&p).unwrap(), v);
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn semantic_env_gate() {
        for (v, want) in [
            ("1", true),
            ("true", true),
            ("ON", true),
            ("0", false),
            ("false", false),
            ("off", false),
            ("", false),
        ] {
            assert_eq!(semantic_flag_on(v), want, "flag {v:?}");
        }
    }

    /// All env-mutating `embedder_from_env` coverage lives in ONE test so
    /// parallel test threads can't race on the process env.
    #[test]
    fn embedder_from_env_registration() {
        for v in [SEMANTIC_ENV, SEMANTIC_STUB_ENV, SEMANTIC_MODEL_ENV] {
            std::env::remove_var(v);
        }
        assert!(embedder_from_env().is_none(), "channel off → no embedder");

        // armed + no model path → auto-stub (dev path)
        std::env::set_var(SEMANTIC_ENV, "1");
        let e = embedder_from_env().expect("armed + no model → stub");
        assert_eq!(e.dim(), DEFAULT_STUB_GRID * DEFAULT_STUB_GRID);

        // a configured model path without the stub override keeps the
        // production seam inert (load fails → `None`) — never the stub
        std::env::set_var(SEMANTIC_MODEL_ENV, "/tmp/nonexistent-dinov2.onnx");
        assert!(embedder_from_env().is_none());
        // …unless the stub is explicitly forced
        std::env::set_var(SEMANTIC_STUB_ENV, "1");
        assert!(embedder_from_env().is_some());

        // disarmed wins over every other knob
        std::env::set_var(SEMANTIC_ENV, "0");
        assert!(embedder_from_env().is_none());

        for v in [SEMANTIC_ENV, SEMANTIC_STUB_ENV, SEMANTIC_MODEL_ENV] {
            std::env::remove_var(v);
        }
    }

    /// `resolve_embedder` precedence with a mock loader — the ONNX load
    /// boundary is exercised without any weights or runtime dylib.
    #[test]
    fn resolve_embedder_precedence_with_mock_loader() {
        use std::cell::Cell;
        let attempted = Cell::new(false);
        let loader = |_: &std::path::Path| -> Option<Box<dyn SemanticEmbedder>> {
            attempted.set(true);
            None
        };
        // armed + model path, stub unset → ONNX attempted; a failed load
        // resolves to inert, never the stub
        assert!(resolve_embedder(Some("/m.onnx"), false, loader).is_none());
        assert!(attempted.get());
        // a successful load surfaces the backend
        assert!(resolve_embedder(Some("/m.onnx"), false, |_| Some(Box::new(
            StubEmbedder::default()
        )))
        .is_some());
        // STUB=1 wins over a configured model path — ONNX not attempted
        let attempted = Cell::new(false);
        let loader = |_: &std::path::Path| -> Option<Box<dyn SemanticEmbedder>> {
            attempted.set(true);
            None
        };
        assert!(resolve_embedder(Some("/m.onnx"), true, loader).is_some());
        assert!(!attempted.get());
        // no model path → stub, loader untouched
        assert!(resolve_embedder(None, false, loader).is_some());
        assert!(!attempted.get());
    }

    /// Synthetic PNG: gradient + per-seed channel offset — the same
    /// generator style the image_io/pipeline tests use (no fixtures).
    fn test_png(seed: u8, w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        ((x * 255) / w) as u8,
                        ((y * 255) / h) as u8,
                        ((x ^ y) as u8).wrapping_add(seed),
                    ]),
                );
            }
        }
        let mut cur = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cur, image::ImageFormat::Png)
            .unwrap();
        cur.into_inner()
    }

    /// Structurally different image: vertical stripes + coarse blocks —
    /// lands far from the gradient `test_png` in stub space.
    fn stripes_png(w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x / 4 + y / 16) % 2 == 0 { 30 } else { 220 };
                img.put_pixel(x, y, image::Rgb([v, 255 - v, (x % 251) as u8]));
            }
        }
        let mut cur = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cur, image::ImageFormat::Png)
            .unwrap();
        cur.into_inner()
    }

    /// Near-duplicate of a PNG: decode, nudge ~4% of pixels slightly,
    /// re-encode — a stand-in for recompressed/retouched duplicates.
    fn near_dup_png(png: &[u8], delta: u8) -> Vec<u8> {
        let mut img = crate::image_io::decode(png).unwrap().to_rgb8();
        for (x, y, p) in img.enumerate_pixels_mut() {
            if x % 5 == 0 && y % 5 == 0 {
                p.0[0] = p.0[0].wrapping_add(delta);
                p.0[1] = p.0[1].wrapping_sub(delta / 2);
            }
        }
        let mut cur = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cur, image::ImageFormat::Png)
            .unwrap();
        cur.into_inner()
    }

    #[test]
    fn stub_embedder_deterministic_and_dim() {
        let e = StubEmbedder::default();
        assert_eq!(e.dim(), DEFAULT_STUB_GRID * DEFAULT_STUB_GRID);
        let bytes = test_png(3, 96, 64);
        let a = e.embed_bytes(&bytes).unwrap();
        assert_eq!(a, e.embed_bytes(&bytes).unwrap());
        assert_eq!(a.len(), e.dim());
        // structured images embed well away from the zero vector
        let n = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(n > 0.5, "structured image should not embed to ~zero");
        // grid variant + undecodable-bytes fallback (deterministic, dim ok)
        let small = StubEmbedder::with_grid(4);
        assert_eq!(small.dim(), 16);
        let h1 = small.embed_bytes(b"not an image").unwrap();
        assert_eq!(h1, small.embed_bytes(b"not an image").unwrap());
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn stub_embedder_near_dup_close_far_apart() {
        let e = StubEmbedder::default();
        let orig = test_png(7, 96, 96);
        let dup = near_dup_png(&orig, 3);
        let other = stripes_png(96, 96);
        let (vo, vd, vs) = (
            e.embed_bytes(&orig).unwrap(),
            e.embed_bytes(&dup).unwrap(),
            e.embed_bytes(&other).unwrap(),
        );
        let near = cosine_similarity(&vo, &vd);
        let far = cosine_similarity(&vo, &vs);
        assert!(near > 0.9, "near-dup cosine {near}");
        assert!(
            far < DEFAULT_SEMANTIC_MIN_COSINE,
            "unrelated cosine {far} must stay under the default floor"
        );
    }

    /// The R2 dev path end-to-end in miniature: stub-embed a batch, run
    /// `semantic_candidates` — the near-dup pair must be emitted at the
    /// default floor while the unrelated image stays out.
    #[test]
    fn stub_channel_emits_candidate_pair() {
        let e = StubEmbedder::default();
        let orig = test_png(11, 96, 96);
        let blobs = [
            e.embed_bytes(&orig).unwrap(),
            e.embed_bytes(&near_dup_png(&orig, 4)).unwrap(),
            e.embed_bytes(&stripes_png(96, 96)).unwrap(),
        ];
        let entries: Vec<SemanticVecs> = blobs
            .into_iter()
            .enumerate()
            .map(|(i, vector)| SemanticVecs {
                image_id: i as i64 + 1,
                vector,
            })
            .collect();
        let pairs = semantic_candidates(&entries, 8, DEFAULT_SEMANTIC_MIN_COSINE);
        assert!(pairs.contains(&(0, 1)), "near-dup pair missing: {pairs:?}");
        assert!(!pairs.iter().any(|&(a, b)| a == 2 || b == 2));
    }

    #[test]
    fn resolvers_defaults_and_env() {
        std::env::remove_var("ITRACE_SEMANTIC_K");
        std::env::remove_var("ITRACE_SEMANTIC_MIN_COS");
        assert_eq!(resolve_semantic_k(None), DEFAULT_SEMANTIC_K);
        assert_eq!(resolve_semantic_k(Some(7)), 7);
        assert!((resolve_semantic_min_cosine(None) - DEFAULT_SEMANTIC_MIN_COSINE).abs() < 1e-6);
        std::env::set_var("ITRACE_SEMANTIC_K", "12");
        std::env::set_var("ITRACE_SEMANTIC_MIN_COS", "0.5");
        assert_eq!(resolve_semantic_k(None), 12);
        assert!((resolve_semantic_min_cosine(None) - 0.5).abs() < 1e-6);
        std::env::remove_var("ITRACE_SEMANTIC_K");
        std::env::remove_var("ITRACE_SEMANTIC_MIN_COS");
    }
}
