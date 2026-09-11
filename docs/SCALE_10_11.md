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
                 "gate_algo_count":M,"image_count":K}
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
  - **Invalidation** on load: `shard_bits`, `gate_algo_count`, and exact
    sorted image_id set vs current ready entries; mismatch → rebuild+overwrite.
    Feature-blob changes with an unchanged image set are **not** detected
    (delete the project dir to force rebuild).
- **`resolve_shard_bits(override)`**:
  1. request/CLI `--shard-bits` / `DedupRequest.shard_bits` if present
  2. else env `ITRACE_MIH_SHARD_BITS`
  3. else default **8**
  4. clamp to **0..=16** (`0` = single shard ≡ monolithic)

### Env knobs

| Variable | Effect |
|----------|--------|
| `ITRACE_MIH_SHARD_BITS` | Default shard bit-width when body/CLI omit override |
| `ITRACE_MIH_INDEX_DIR` | If set, dedup load-or-builds `{DIR}/project_{id}/` gate-index bundle. Unset → identical in-memory rebuild behaviour as before. |

### Example

```bash
export ITRACE_MIH_INDEX_DIR=/var/lib/itrace/mih
export ITRACE_MIH_SHARD_BITS=8
# first dedup for project 7 builds /var/lib/itrace/mih/project_7/
# second dedup reuses it when ready image_ids + shard_bits + gate count match
itrace-cli dedup 7
```

## Microbench

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run -p itrace-core --release --example mih_shard_bench
```
