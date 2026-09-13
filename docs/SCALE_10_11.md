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
| `ITRACE_MIH_INDEX_DIR` | If set, dedup load-or-builds `{DIR}/project_{id}/` gate-index bundle. Unset → identical in-memory rebuild behaviour as before. |
| `ITRACE_CROP_MIN_HITS` | Distinct probe-key hits required for a crop-channel candidate (default 2) |

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
| 3 | Multi-node shard ownership + scatter/gather | **in progress — in-process foundation done** |
| 4 | Semantic DINOv2/HNSW recall channel | planned |

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
non-persistent gate scan (`dedup_candidates_sharded`, used by
`dedup_confirmed` when no index dir is set) through a
`MultiNodeMihIndex` — inserts routed by ownership, queries
scatter/gathered. Default/unset/invalid = 1 → unchanged
`ShardedMihIndex`. Persistent `ITMIHP1`/`ITMIHC1` bundles always stay
single-node. Example:

```bash
ITRACE_MIH_NODES=4 itrace-cli dedup 7   # 4 logical nodes in-process
```

**RPC seam:** a networked deployment keeps `ShardOwnership` on a
coordinator, replaces each `NodeIndex` with a transport stub
implementing `insert(shard_id, key, owner)` /
`query_shards(key, radius, &[shard_ids])`, and merges replies exactly
as `query_with_nodes` does — no semantic change to probe sets, recall,
or the downstream `dedup_confirmed` re-scoring. Real transport, service
discovery, and wiring the `ITMIHN1` per-node dirs into the persistent
project bundle flow are follow-up work.

## Microbench

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run -p itrace-core --release --example mih_shard_bench
```
