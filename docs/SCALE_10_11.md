# Scaling Image Trace MIH toward 10^11

This note projects Multi-Index Hash (MIH) memory and sketches how
`ShardedMihIndex` makes ~10^11 images architecturally reachable. Full
distributed storage (Postgres/TiKV, etc.) is out of scope here — the index
foundation is the deliverable.

## Per-image footprint

For each gate algorithm we index **8 orientation variants** as u64 keys:

| Component | Bytes (approx.) |
|-----------|-----------------|
| `u64` key | 8 |
| `u32` owner | 4 |
| 8× `u32` table refs (one per 8-bit substring bucket) | 32 |
| **Per indexed key** | **≈ 44 B** |
| Fixed direct-mapped bucket arrays per `MihIndex` | ≈ 48 KiB |

With **G gate algorithms** (today `HASH_GATE_ALGOS` has 4; planning often
uses 3: phash/dhash/whash) and **V = 8** variants:

\[
\text{bytes/image} \approx G \times V \times 44 = G \times 352
\]

Common planning figure: **3 algos × 8 variants × ~350 B ≈ 1.05 KB/image**
for the key payload (ignoring the small per-shard fixed tables), or
**~350 B/image/algorithm** as in `index.rs`.

Fixed overhead scales with shard count:

\[
\text{fixed} \approx (\#\text{algos}) \times 2^{\text{shard\_bits}} \times 48\,\text{KiB}
\]

## Projections (3 algos × 8 variants × ~350 B/algo)

Using **≈ 1.05 KB/image** payload (+ note fixed tables separately):

| N | Payload RAM | Fixed @ shard_bits=8 (256×3×48 KiB) | Fixed @ shard_bits=12 |
|---|-------------|--------------------------------------|------------------------|
| 10^8 | ≈ 105 GB | ≈ 36 MB | ≈ 0.6 GB |
| 10^9 | ≈ 1.05 TB | ≈ 36 MB | ≈ 0.6 GB |
| 10^11 | ≈ 105 TB | ≈ 36 MB | ≈ 0.6 GB |

Disk for `save_dir` is smaller than RAM: each shard stores only
`keys:u64[]` + `owners:u32[]` (**12 B/key**), tables rebuilt on load:

| N | On-disk keys+owners (3×8) |
|---|---------------------------|
| 10^8 | ≈ 29 GB |
| 10^9 | ≈ 288 GB |
| 10^11 | ≈ 29 TB |

## Recommended `shard_bits` for 16–64 GB RAM/node

Target: keep **one node’s resident shards** in the 16–64 GB band.

Payload per shard ≈ `N_total / 2^{shard_bits} × 1.05 KB` if keys are
uniform over high bits (good for random perceptual hashes).

| Node RAM budget | Comfortable keys/node | `shard_bits` sketch for N=10^11 |
|-----------------|----------------------|----------------------------------|
| 16 GB | ~1e7 images (≈ 10 GB payload) | 14 → 2^14 = 16384 shards ≈ 6e6 img/shard |
| 32 GB | ~2e7 images | 13–14 |
| 64 GB | ~4–5e7 images | 12–13 |

Practical defaults:

- **Single fat node / lab box:** `shard_bits = 0..8` (monolithic or 256-way) up to ~10^8.
- **Multi-node toward 10^9–10^11:** `shard_bits = 12..16`, assign contiguous
  shard-id ranges per node so each holds tens of millions of images.

Always size for **query fan-out**: a query with Hamming radius `r` probes
all shards whose `shard_bits`-bit prefix is within distance `r` of the
query prefix. When `r ≥ shard_bits` every shard is probed — prefer
moderate radius (server default 10) and/or keep `shard_bits` from growing
past the working radius without a multi-node scatter/gather layer.

## Multi-node ownership by shard id

1. Fix global `shard_bits` (e.g. 14).
2. Shard id = high `shard_bits` of each u64 key (`shard_id_for`).
3. Partition `{0 .. 2^{shard_bits}-1}` across nodes (by range or consistent hash).
4. **Insert:** compute shard id → send key/owner to owning node’s local `MihIndex`.
5. **Query:** compute the Hamming ball of prefixes within `radius`; fan out
   to the nodes that own those shard ids; merge owner sets; continue with
   exact re-scoring as today (`dedup_confirmed` / `run_dedup_scan`).
6. Persist each node’s shard subset with the same on-disk layout
   (`meta.json` + `shards/NNNN.bin`), possibly only the locally owned `NNNN`.

## On-disk layout summary

```text
<dir>/                                    # single ShardedMihIndex
  meta.json     {"magic":"ITMIH1","version":1,"shard_bits":N,"key_count":K}
  shards/
    0000.bin    u64 n | n×u64 keys | n×u32 owners   (little-endian)
    0001.bin
    ...

{ITRACE_MIH_INDEX_DIR}/project_{id}/      # project gate-index bundle
  meta.json     {"magic":"ITMIHP1","version":1,"shard_bits":N,
                 "gate_algo_count":M,"image_count":K,
                 "feature_fingerprint":"<blake3 hex>"}
  image_ids.bin K × i64 LE  (owner slot → image_id, sorted ascending)
  indexes/
    0/ … M-1/   each a ShardedMihIndex (ITMIH1) for one gate algo
```

See `itrace_core::index` module docs, `ShardedMihIndex::{save_dir,load_dir}`,
and `load_or_build_project_gate_index` / `dedup_candidates_sharded_cached`.

## Implemented

Server + CLI wiring for sharded MIH recall **and** optional persistent index:

- **Server** `POST /v1/projects/{id}/dedup` → `run_dedup_scan` uses
  `dedup_candidates_sharded_cached` (JSON shape unchanged; optional
  `index_loaded: bool` when a dir is configured).
- **CLI** `itrace-cli dedup` → `dedup_confirmed_cached` with optional
  `--shard-bits`; prints `index_loaded=…`.
- Shared helpers in `itrace_core::index` (used by both paths):
  - `resolve_shard_bits` / `resolve_mih_index_dir` / `project_mih_index_path`
  - `load_or_build_project_gate_index` — dense `u32` owners ↔ `image_ids.bin`
  - **Invalidation** on load: `shard_bits`, `gate_algo_count`, exact sorted
    image_id set vs current ready entries, **and `feature_fingerprint`** —
    a BLAKE3 hash over the indexed key material (canonical
    sorted-by-image_id order; see “Feature fingerprint” below). Any
    mismatch → rebuild + overwrite. Feature-blob changes with an
    unchanged image set are detected: recomputed or backfilled vectors
    invalidate even when every image_id stays the same.
- **`resolve_shard_bits(override)`**:
  1. request/CLI `--shard-bits` / `DedupRequest.shard_bits` if present
  2. else env `ITRACE_MIH_SHARD_BITS`
  3. else default **8**
  4. clamp to **0..=16** (`0` = single shard ≡ monolithic)

### Crop/slice recall channel

Whole-image gate hashes structurally cannot recall crops or slice tiles —
a 70% center crop or a 2×2 quadrant moves ~40% of the bits, far outside
any MIH radius. A second channel indexes the `crophash_keys` stored
feature: 15 windowed phash keys (a 1.0→0.25 center-scale chain plus a 3×3
grid of half-size windows) × 8 orientation variants per image, flattened
into one sharded MIH (`project_{id}_crop/`, magic `ITMIHC1`, same
invalidation rules as the gate bundle). A pair becomes a candidate when
`min_hits` distinct probe keys hit the other owner
(`ITRACE_CROP_MIN_HITS`, default 2), and is then **verified by NCC
containment** (`slice::contains_rot4`, shared-scale downscale + all four
quarter-turns) before joining the confirmed groups — hash hits alone
never confirm.

Measured on `datasets/built` (36 images, 6 microscopy families):
crop70 and slice tiles recover 6/6 and 12/12 in dedup — previously 0 —
with zero cross-family merges; `crophash` similarity is ~0.9–1.0 on
same-source crops/slices vs ≤0.8 foreign. `smart` accepts `crophash` as a
crop-gate vote (`smart_pair_confirmed`); `blockhash` was evaluated and
rejected as a gate (best-overlap tiling over-fires ~0.88 on unrelated
small tiles).

Scale note: the crop channel indexes ~120 keys/image (~8× the gate
channel's per-image key count at 3 gate algos), so its memory/disk
footprint scales the same way — shard ownership, persist, verify. Exact
verification cost is bounded by `min_hits` + radius, not pair count.

### Env knobs

| Variable | Effect |
|----------|--------|
| `ITRACE_MIH_SHARD_BITS` | Default shard bit-width when body/CLI omit override |
| `ITRACE_MIH_INDEX_DIR` | If set, dedup load-or-builds `{DIR}/project_{id}/` gate, `project_{id}_crop/`, and (when armed) `project_{id}_sem/` bundles. Unset → identical in-memory rebuild behaviour as before. |
| `ITRACE_CROP_MIN_HITS` | Distinct probe-key hits required for a crop-channel candidate (default 2) |
| `ITRACE_SEMANTIC*` | Semantic channel knobs — see Phase 4 table below |

### Example

```bash
export ITRACE_MIH_INDEX_DIR=/var/lib/itrace/mih
export ITRACE_MIH_SHARD_BITS=8
# first dedup for project 7 builds /var/lib/itrace/mih/project_7/
# second dedup reuses it when image_ids + shard_bits + gate count +
# feature_fingerprint all match
itrace-cli dedup 7
```

## Feature fingerprint

`meta.json.feature_fingerprint` is `blake3` hex over the exact key
material persisted in the bundle, streamed in canonical order
(entries sorted by `image_id`):

- **Gate bundle (`ITMIHP1`):** for each `DedupKeys` — `image_id` (i64 LE),
  `variant_keys.len()` (u32 LE), then per variant `keys.len()` (u32 LE)
  and every `u64` key (LE).
- **Crop bundle (`ITMIHC1`):** for each `CropKeys` — `image_id` (i64 LE),
  `keys.len()` (u32 LE), then every `u64` key (LE).

The check runs at load after the cheap magic/version/shard_bits/
image-set checks; a mismatch (including bundles written before the
field existed) falls through to build + overwrite. Because the hash
covers the vectors themselves — not just the image set — any feature
recompute, backfill, or corruption that leaves `image_id`s identical
still forces a rebuild.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | `ImageStore` trait extraction + feature-fingerprint MIH invalidation | **done** |
| 2 | `PostgresStore` backend behind the `ImageStore` trait | **done** |
| 3 | Multi-node shard ownership + scatter/gather | **done — in-process foundation + `ITMIHN1` persist** |
| 4 | Semantic DINOv2/HNSW recall channel | **done — foundation** (stub + HNSW + wiring; R3 double regression) |
| 5 | ONNX `SemanticEmbedder` backend + persisted semantic bundle | **done — R1 ONNX skeleton + R2 `project_{id}_sem/` persist/fingerprint + R3 double regression** |
| 6 | Persisted HNSW graph + sharded semantic recall | **done — R1 `hnsw.bin` + R2 harness + R3 double regression** |
| 7 | `ITMIHN1` multi-node MIH project persist | **done — R1 wire + R2 harness + R3 double regression** |
| 8 | Multi-node semantic/HNSW sharding | **in progress — R1 `project_{id}_sem_mn/` (`ITSEMN1`) + R2 harness done** |

### Phase 1 delivered

- `itrace_store::ImageStore` — object-safe trait covering
  projects/images/features/precompute/pair-cache/runs plus the blob
  facade; `SqliteStore` is the default impl and `pub type Store =
  SqliteStore` keeps the old name compiling.
- `itrace-cli` and `itrace-server` hold `&dyn ImageStore` /
  `Arc<dyn ImageStore>` — a future Postgres backend plugs in without
  touching call sites.
- `feature_fingerprint` (above) in both `ITMIHP1` and `ITMIHC1` metas;
  tested: unchanged vectors reuse (`index_loaded=true`), same-ids
  mutated vectors rebuild (`index_loaded=false`).

### Phase 2 delivered — `PostgresStore`

`itrace_store::pg::PostgresStore` implements the full `ImageStore`
surface on the sync `postgres` crate (tokio-postgres sync facade) over
an `r2d2` pool (8 connections). The trait is synchronous, so a sync
driver keeps server handlers free of `block_on`/`block_in_place`
hazards; the pool supplies the concurrency SQLite's single Mutex'd
connection never could.

Backend selection is env-only — call sites stay untouched:

```bash
# default: sqlite (unchanged behavior)
itrace-cli dedup 7
# postgres metadata, fs/s3 blobs as before
ITRACE_STORE=postgres \
ITRACE_DATABASE_URL=postgres://itrace:itrace@localhost:5432/itrace \
itrace-server
```

`docker compose up -d postgres` provides the dev instance (schema
auto-applies on connect — `CREATE TABLE IF NOT EXISTS` DDL mirrors the
SQLite `SCHEMA`). Type mapping: `INTEGER PK AUTOINCREMENT`→`BIGINT
GENERATED ALWAYS AS IDENTITY`, `BLOB`→`BYTEA`, `REAL`→`DOUBLE
PRECISION`, `rotation_invariant` int→`BOOLEAN`, ISO-8601 `created_at`
TEXT preserved via `to_char(now() AT TIME ZONE 'UTC', …)`; `IN (…)`
lists become `= ANY($n)` array params (no 999-variable chunking).

Parity: `crates/itrace-store/tests/store_contract.rs` runs one
contract suite against both backends — sqlite always, postgres when
`ITRACE_TEST_DATABASE_URL` is set (CI provides a `postgres:16` service
container on the `build-test` job).

Selection precedence: `itrace-cli --store` flag > `ITRACE_STORE` >
sqlite default; URL via `ITRACE_DATABASE_URL`/`DATABASE_URL`.
Switching backends and what does **not** auto-migrate (sqlite file →
pg needs re-ingest or a future dump tool): `docs/POSTGRES.md`.

### Phase 3 delivered (foundation) — shard ownership + scatter/gather

`itrace_core::ownership` implements the "Multi-node ownership by shard
id" design above, in-process:

- **`ShardOwnership`** — contiguous `ShardRange` per node covering
  `0 .. 2^shard_bits` exactly once (`even()` balanced split or
  `from_ranges()` custom partition, validated for gaps/overlaps;
  empty ranges allowed for staged drain/join). `owner_of(shard_id)`
  resolves via `partition_point` over the sorted ranges.
- **`MultiNodeMihIndex`** — N logical nodes; each `NodeIndex` stores
  `MihIndex`es for *only its owned shards* (no memory spent on remote
  shards). `insert` routes via `owner_of(shard_id_for(key))`;
  `query_with_nodes` applies the same Hamming-ball probe filter as
  `ShardedMihIndex::query_into` (prefix distance ≤ radius), fans out
  to only the nodes owning probe-set shards, and returns the merged
  owner set plus the contacted-node list for fan-out inspection.

Parity vs monolithic `ShardedMihIndex` is property-tested (random keys,
radii 0..shard_bits+4, 2- and 4-node layouts plus `shard_bits=0`); a
dedicated test proves `radius < shard_bits` queries skip whole nodes.

**Persisting owned shards (`ITMIHN1`).** `MultiNodeMihIndex::save_dir`
writes a cluster bundle; `save_node_dir(node_id, dir)` writes just one
node's share — the unit a real deployment persists per host:

```text
<dir>/
  meta.json      {"magic":"ITMIHN1","version":1,"shard_bits":N,
                  "key_count":K,"ranges":[{"start":S,"end":E}, ...]}
  node_0/
    meta.json    {"magic":"ITMIHN1","version":1,"shard_bits":N,
                  "node_id":0,"range":{...},
                  "shard_ids":[...non-empty owned shards...],
                  "key_count":Ki}
    shards/
      NNNN.bin   only owned, non-empty shards; same LE payload as
                 ITMIH1 (u64 n | n×u64 keys | n×u32 owners)
  node_1/ ...
```

`load_dir` rebuilds `ShardOwnership` from the stored ranges (validated
for exact coverage) and replays each node's bins — empty owned shards
stay in memory only. Round-trip parity vs the live index and a
monolithic `ShardedMihIndex` is tested.

**Simulating N nodes.** `ITRACE_MIH_NODES=N` (N > 1) routes the
gate scan (`dedup_candidates_sharded` / `dedup_confirmed`) through a
`MultiNodeMihIndex` — inserts routed by ownership, queries
scatter/gathered. Default/unset/invalid = 1 → unchanged
`ShardedMihIndex`. With `ITRACE_MIH_INDEX_DIR` set, the persistent
bundle follows the same switch: `project_{id}/` (`ITMIHP1`) for
single-node, `project_{id}_mn/` (`ITMIHN1`) for N > 1 (Phase 7 R1).
Example:

```bash
ITRACE_MIH_NODES=4 itrace-cli dedup 7   # 4 logical nodes in-process
```

**RPC seam:** a networked deployment keeps `ShardOwnership` on a
coordinator, replaces each `NodeIndex` with a transport stub
implementing `insert(shard_id, key, owner)` /
`query_shards(key, radius, &[shard_ids])`, and merges replies exactly
as `query_with_nodes` does — no semantic change to probe sets, recall,
or the downstream `dedup_confirmed` re-scoring. Real transport and
service discovery are follow-up work; wiring `ITMIHN1` cluster dirs
into the persistent project bundle flow landed in Phase 7 R1.

### Phase 4 (done — foundation) — semantic recall channel

`itrace_core::semantic` adds the third recall channel's foundation —
dense-embedding ANN retrieval for pairs whose images share scene semantics
but diverge beyond every hash radius (heavy recolor/composite transforms):

- **`SemanticEmbedder`** — object-safe pluggable backend
  (`dim` / `embed_bytes` / `embed_path`). The production impl is
  `OnnxSemanticEmbedder` (Phase 5, `semantic-onnx` feature) loading a
  DINOv2 ONNX model from `ITRACE_SEMANTIC_MODEL`; **weights downloads are
  a follow-up and never happen in CI** — tests and wiring use in-crate
  stubs.
- **`StubEmbedder`** (R2) — deterministic dev/test stand-in: decode →
  grayscale → block-average onto a 16×16 grid (256-dim), mean-subtracted;
  undecodable bytes fall back to a blake3-seeded pseudo-vector so a scan
  never fails. Content-smooth enough that near-duplicates land at high
  cosine, but **not** a semantic model — it exists to exercise the channel
  end-to-end until DINOv2 lands.
- **`HnswIndex`** — in-memory HNSW ANN index over cosine-normalized `f32`
  vectors (`insert` + `query(k)`, deterministic seeded level draws, no new
  deps). Unit-tested on synthetic clusters: same-cluster members fill top-k
  and far clusters stay out; recall@k is checked against exact brute force.
- **`semantic_candidates` / `semantic_candidates_with_index`** emit
  `(entry_i, entry_j)` pairs under the same dense-owner contract as the
  gate/crop channels; `union_candidate_pairs` merges them into the MIH
  candidate list before verification.

**Wiring (default off):** with `ITRACE_SEMANTIC=1`, `embedder_from_env()`
resolves a backend — `ITRACE_SEMANTIC_MODEL` set + `ITRACE_SEMANTIC_STUB`
unset → the ONNX backend is attempted and any load failure resolves to
`None` (a configured production path never silently stubs); otherwise
`StubEmbedder`. Both dedup entry points then embed each indexed
image (`run_cli_dedup` / `run_dedup_scan` re-decode blobs per scan — the
persisted-embedding precompute is the follow-up), run
`semantic_candidates` at `ITRACE_SEMANTIC_K` / `ITRACE_SEMANTIC_MIN_COS`,
and fold the hits into the MIH candidate set ahead of the variant-max
hash verification (`dedup_confirmed_cached_with_extra` /
`union_candidate_pairs`). Semantic hits are *candidates only* — a pair
still needs the existing hash/crop confirmation, so the channel can add
recall without silently widening merges. Flag unset → the path is skipped
and dedup output is bit-identical to the pre-R2 baseline.

**Follow-ups:** real DINOv2 weights + calibration (Phase 5; the
`semantic-onnx` backend skeleton is in — see below); persisted
`project_{id}_sem/` bundle (**shipped Phase 5 R2** — `image_ids.bin` +
`vectors.bin`, `SemanticEmbedder::fingerprint` + BLAKE3 payload
invalidation like `ITMIHP1`/`ITMIHC1`; the in-memory HNSW still
rebuilds per scan); sharding the
graph across nodes (per-shard HNSW or per-node full graph) once the
`ITMIHN1` transport seam is real.

### Phase 5 (done) — ONNX semantic backend

`itrace_core::semantic_onnx` (R1, behind the opt-in `semantic-onnx`
cargo feature) provides **`OnnxSemanticEmbedder`**: a `SemanticEmbedder`
over `ort` built with `load-dynamic`, so `libonnxruntime` is `dlopen`'d
at run time and **nothing is linked or downloaded at build time** —
default builds and CI never see the dependency.

- **Dylib resolution:** `ITRACE_ORT_DYLIB` → `ORT_DYLIB_PATH` → default
  soname (`libonnxruntime.so`) next to the executable / loader path.
- **Weights:** `ITRACE_SEMANTIC_MODEL` points at a local `.onnx` file.
  `ort` *panics* on a missing dylib, so `OnnxSemanticEmbedder::load`
  traps that under `catch_unwind` — missing/invalid weights or a
  missing/incompatible runtime resolve to `Err` → `None` → **inert
  channel, never a silent stub**.
- **Model contract:** one f32 NCHW input `[1, 3, 224, 224]`, ImageNet
  mean/std (`pixel_values` input preferred, else first declared input);
  first output must be f32 `[1, D]` (pooled) or `[1, T, D]` (token map →
  CLS row). Arbitrary ONNX graphs are rejected, not guessed.
- **Testing without downloads:** `resolve_embedder` precedence is
  covered by a mock-loader unit test; feature-gated tests cover
  `Err`-not-panic load failures, the preprocess tensor, and a
  ~200-byte hand-encoded `Flatten` ONNX fixture that runs `load` +
  `embed_bytes` end-to-end on a dev box with a runtime installed
  (skipped on CI). **Real DINOv2 weights are never fetched by tests or
  CI.**

**R2 — persisted `project_{id}_sem/` bundle.** When
`ITRACE_MIH_INDEX_DIR` is set and the channel is armed, dedup
load-or-builds `{DIR}/project_{id}_sem/` so repeat scans skip model
inference (Phase 6 R1 additionally persists the HNSW graph in
`hnsw.bin`):

```text
project_{id}_sem/
  meta.json      {"magic":"ITSEMP1","version":1,
                  "embedder":"<SemanticEmbedder::fingerprint>",
                  "dim":D,"image_count":K,
                  "feature_fingerprint":"<blake3 hex>"}
  image_ids.bin  K × i64 LE — owner slots, sorted ascending
  vectors.bin    K × D × f32 LE — row i belongs to image_ids.bin[i]
  hnsw.bin       ITSEMH1 graph (Phase 6 R1) — see below
```

`SemanticEmbedder::fingerprint()` is the invalidation key — `stub:g16`
for the stub, `onnx:{path}:{blake3-of-weights}` for the ONNX backend —
so a changed model path, changed weights bytes, or a backend swap
refuses the stale bundle and rebuilds. `try_load_project_sem_index`
additionally requires magic/version, `dim` (`0` = wildcard for
dynamic-dim backends), the exact image_id set, and a recomputed BLAKE3
`feature_fingerprint` over the stored payload; any mismatch or
corruption returns `None` → rebuild + overwrite, never silent reuse.
Wiring: `load_or_build_project_sem_index` is called by both
`run_cli_dedup` and `run_dedup_scan` when armed; CLI prints
`+N sem (cached)` on a bundle hit, the `/dedup` JSON gains
`sem_index_loaded`. Flag off → everything skipped as before.

**R3 evidence:** `scripts/smoke_semantic_persist.sh` dedups one project
three times on a shared index dir — flag-off baseline, then armed twice:
run 1 builds `project_{id}_sem/` (`+N sem`), run 2 reports
`+N sem (cached)` with `index_loaded=true`, and confirmed-group
membership is asserted identical across all three scans. See
`datasets/built/reports/PHASE5_R3_DOUBLE_REGRESSION.md`.

Remaining follow-ups: real `dinov2_vits14`/`vitb14` weights + recall
calibration on transformed-image fixtures.

### Phase 6 (done — R1+R2; R3 evidence gathered) — persisted HNSW graph

**R1 — `project_{id}_sem/hnsw.bin` (`ITSEMH1` v1).** Sibling-file choice:
the graph gets its own magic/version rather than extending `ITSEMP1`, so
the vectors bundle stays a pure `(image_ids, vectors)` payload and the
graph validates independently — a missing/stale `hnsw.bin` degrades to a
graph rebuild off the verified vectors, never a re-embed. The header
embeds the same BLAKE3 `feature_fingerprint` as `meta.json`, so a stale
graph can never pair with rebuilt or mismatched vectors.

```text
hnsw.bin
  magic       7B  "ITSEMH1"          version   u64 LE = 1
  fingerprint 64B ASCII hex — must equal meta's recomputed
              feature_fingerprint (binds graph to vectors.bin)
  dim, m, ef_construction, ef_search, rng, max_level   u64 LE each
  entry       u64 LE — entry-point node idx (u64::MAX = empty)
  node_count  u64 LE — must equal meta's image_count
  per node:   id u32 LE (owner slot), n_layers u64 LE,
              vec dim×f32 LE (L2-normalized), then per layer
              u64 count + count×u32 neighbour node idxs
```

`load_or_build_project_sem_index` now returns `ProjectSemIndex`
(`entries` + `owner_ids` + `index` + `vecs_loaded`/`graph_loaded`);
callers use `semantic_candidates_with_index` directly. A full cache hit
restores the graph without re-inserting — CLI reports
`+N sem (cached), sem_hnsw_loaded=true`, `/dedup` JSON gains
`sem_hnsw_loaded`. Load validation is bounds-checked end to end
(entry/neighbour refs in range, `n_layers ≤ MAX_LEVEL+1`, exact
end-of-file); any violation → rebuild + overwrite.

**R2 — harness polish.** `scripts/smoke_semantic_persist.sh` now asserts
the graph half, not just the vectors: armed run 1 must print
`sem_hnsw_loaded=false` (graph built), armed run 2 must print both
`+N sem (cached)` and `sem_hnsw_loaded=true` (graph restored — a vec-only
hit would show `false`), `hnsw.bin` must exist in the bundle, and the
flag-off run must show no sem fields at all. Confirmed-group membership
is diffed identical across off/on1/on2.

**R3 evidence:** flag-off `smoke_itrace_dataset.sh` ×2 reproduced the
6/6/6/12 baseline; `smoke_semantic_persist.sh` ×2 (`r6b`, `r6c`) each
showed the build → cached+graph-loaded transition
(`sem_hnsw_loaded=false` → `true`) with identical confirmed groups. See
`datasets/built/reports/PHASE6_R3_DOUBLE_REGRESSION.md`.

Remaining follow-ups: multi-node semantic/HNSW sharding (per-shard graph
or per-node full graph) once the `ITMIHN1` seam is real; graph format
compaction (u16 neighbour refs, omitting stored vecs by deriving them
from `vectors.bin`) if bundle size matters.

### Phase 7 (done — R1–R3) — ITMIHN1 project persist

**R1 — `project_{id}_mn/` (`ITMIHN1` v1).** When `ITRACE_MIH_INDEX_DIR`
is set *and* `ITRACE_MIH_NODES` > 1, the gate scan persists the
in-process `MultiNodeMihIndex` clusters under a sibling directory of
`project_{id}/` rather than inside it — the `ITMIHP1` single-node format
is untouched, so flipping `ITRACE_MIH_NODES` between 1 and N>1 can never
read a bundle of the wrong shape:

```text
project_{id}_mn/
  meta.json      {"magic":"ITMIHN1","version":1,"shard_bits":N,
                  "node_count":M,"gate_algo_count":A,"image_count":K,
                  "key_count":T,"feature_fingerprint":"<blake3 hex>"}
  image_ids.bin  K × i64 LE — owner slots, sorted ascending
  indexes/{a}/   one MultiNodeMihIndex::save_dir cluster per gate algo:
                 top-level ITMIHN1 meta (ranges) + node_N/ dirs with
                 owned non-empty shards (same NNNN.bin LE payload)
```

Load requires exact match of `shard_bits`, `node_count`,
`gate_algo_count`, the sorted image_id set, and the BLAKE3
`feature_fingerprint`; each cluster additionally re-validates its
`ITMIHN1` ranges and shard geometry through `MultiNodeMihIndex::load_dir`.
Any miss → rebuild + overwrite, never silent reuse. `index_loaded=true`
on a hit reflects cluster restore, not rebuild. Tests cover round-trip
parity vs a live multi-node index, and shard_bits/node_count/image-set/
fingerprint/corrupt-cluster-magic/missing-node-dir invalidation.

**R2 — persist smoke harness.** `scripts/smoke_mih_nodes_persist.sh`
runs one project through four scans on a shared index dir — mono build
(`ITMIHP1`, `index_loaded=false`) → multi-node build (`_mn`,
`index_loaded=false`) → multi-node hit (`index_loaded=true`) → mono hit
(`index_loaded=true`) — and fails unless the `_mn` bundle has the
ITMIHN1 top meta + `indexes/{a}/node_N/` layout, `project_{id}/` keeps
`ITMIHP1`, and all four scans emit identical sorted group membership.
`scripts/smoke_mih_nodes.sh` (non-persist parity) now compares sorted
member lines — group emission order is not part of the contract.

**R3 evidence:** single-node `smoke_itrace_dataset.sh` ×2 reproduced the
6/6/6/12 baseline with `index_loaded=true` persist hits;
`smoke_mih_nodes_persist.sh` ×2 (`r7c`, `r7d`) each showed the
mono-build → `_mn` build → `_mn` hit → mono-hit transition with
`ITMIHN1`/`ITMIHP1` coexistence and identical membership;
`smoke_mih_nodes.sh` parity green. See
`datasets/built/reports/PHASE7_R3_DOUBLE_REGRESSION.md`.

Remaining follow-ups: real RPC transport + service discovery (non-goal
for now), cross-node dedup of `project_{id}_crop/` under multi-node.
Multi-node semantic/HNSW sharding landed in Phase 8 R1 below.

### Phase 8 (in progress — R1+R2 done) — multi-node semantic/HNSW sharding

**R1 — `project_{id}_sem_mn/` (`ITSEMN1` v1).** When `ITRACE_SEMANTIC`
is armed *and* `ITRACE_MIH_NODES` > 1 *and* `ITRACE_MIH_INDEX_DIR` is
set, the semantic channel shards its HNSW index across the same
in-process node count as the gate MIH scan. Sorted image ids are split
into `node_count` contiguous non-empty ranges; each node gets a
complete single-node semantic bundle over its owned ids:

```text
project_{id}_sem_mn/
  meta.json      {"magic":"ITSEMN1","version":1,"node_count":N,
                  "embedder":"<fp>","dim":D,"image_count":K,
                  "feature_fingerprint":"<blake3 over all sorted
                  ids+vecs>","ranges":[{"node":i,"first":id,"count":c}]}
  node_{i}/      standard bundle — ITSEMP1 meta.json + image_ids.bin +
                 vectors.bin (owned ids only) + ITSEMH1 hnsw.bin
```

Load requires the top meta to match `node_count`, embedder
fingerprint, `dim`, `image_count`, and the recomputed partition ranges;
each `node_{i}/` then validates through the normal `ITSEMP1` loader
against its expected chunk, and the assembled sorted payload must match
the global BLAKE3 `feature_fingerprint`. Per-node `hnsw.bin` graphs
validate via the existing `ITSEMH1` fingerprint binding — a stale or
corrupt node graph rebuilds *that* node only. Any vector/meta miss →
rebuild the whole bundle, never silent reuse. On a full hit
`vecs_loaded`/`graphs_loaded` report the restore; embedding is skipped
entirely.

Query: every probe vector scatters to **all** node graphs
(`semantic_candidates_multi`), results map back through node-local
owner slots, pairs normalize/sort/dedupe. With `k ≥` total image count
the candidate set is exactly the single-node set; smaller `k` can
return a superset (each node answers `k` locally) — harmless because
semantic output stays candidate-only and the existing hash/crop
verification decides confirmed merges.

Single-node `project_{id}_sem/` is a different directory: flipping
`ITRACE_MIH_NODES` between 1 and N>1 never reads a bundle of the wrong
shape, mirroring the `project_{id}/` vs `project_{id}_mn/` design. CLI
and server both branch on `index::mih_node_count()` at the same point
they arm the channel.

Tests: `load_or_build_sem_index_multi_persists_and_parity` (build →
layout → full cache hit with zero embed calls → graph-corruption
rebuilds one node → candidate parity vs single-node) and
`multi_node_sem_bundle_invalidation` (node_count / embedder fp /
image-set / missing node dir / corrupt node meta / corrupt top meta →
rebuild; rebuilt bundle cache-hits).

**R2 — persist smoke harness.** `scripts/smoke_semantic_mn_persist.sh`
runs one armed-stub project through five scans on a shared index dir —
single-node build (`project_{id}_sem/`, `sem_hnsw_loaded=false`) →
multi-node build (`_sem_mn`, `sem_hnsw_loaded=false`) → multi-node hit
(`+N sem (cached)`, `sem_hnsw_loaded=true`) → single-node hit →
multi-node hit again — and fails unless `_sem_mn` has the ITSEMN1 top
meta + contiguous `ranges` + `node_{i}/` ITSEMP1/ITSEMH1 layout,
`project_{id}_sem/` keeps `ITSEMP1`, and all five scans emit identical
sorted group membership (semantic stays candidates-only).

Remaining follow-ups: real RPC/service discovery (non-goal),
`project_{id}_crop/` multi-node dedup, double-regression evidence (R3).

| Variable | Effect |
|----------|--------|
| `ITRACE_SEMANTIC` | `1`/`true`/`on` arms the semantic channel (unset/`0`/`false`/`off` = off) |
| `ITRACE_SEMANTIC_MODEL` | Local DINOv2 ONNX weights path; with `semantic-onnx` builds the backend loads it — missing/invalid path or missing runtime resolves inert, never silently to the stub |
| `ITRACE_ORT_DYLIB` | Explicit `libonnxruntime` shared-library path for `semantic-onnx` builds (falls back to `ORT_DYLIB_PATH`, then the default soname) |
| `ITRACE_SEMANTIC_STUB` | `1` forces `StubEmbedder` even when `ITRACE_SEMANTIC_MODEL` is set; stub is auto-selected whenever the channel is armed with no model path |
| `ITRACE_SEMANTIC_K` | Per-image ANN probe width (default 32) |
| `ITRACE_SEMANTIC_MIN_COS` | Cosine floor for a semantic candidate (default 0.75) |

## Microbench

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run -p itrace-core --release --example mih_shard_bench
```
