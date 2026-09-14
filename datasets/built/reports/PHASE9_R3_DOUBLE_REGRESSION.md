# Phase 9 Round 3 — Double Regression Evidence (crop multi-node MIH persist)

Date: 2026-09-14
Commit under test: `15ed1d3` (merge of PR #25, `chore/phase9-r2-crop-mn-harness`) on `main`

Purpose: gate evidence for **Phase 9 Final** — multi-node crop/slice
MIH sharding via `project_{id}_crop_mn/` (`ITMIHCN1` v1: top-level meta
+ contiguous sorted-image-id `ranges` + one complete `ITMIHC1`
`node_{i}/` bundle per range; `crop_candidates_multi` scatter/merge
with exact-parity candidate contract), plus the persist smoke harness
that asserts the build → cache-hit transition end-to-end. In-process
multi-node only — no RPC, no service discovery, no weight downloads.

## 1. Flag-off baseline — `scripts/smoke_itrace_dataset.sh` × 2

Both runs back-to-back with `ITRACE_MIH_NODES`, `ITRACE_MIH_INDEX_DIR`,
and all `ITRACE_SEMANTIC*` knobs unset; the harness recreates its
data/report/index dirs each invocation.

```bash
env -u ITRACE_SEMANTIC -u ITRACE_SEMANTIC_STUB -u ITRACE_SEMANTIC_MODEL \
    -u ITRACE_SEMANTIC_K -u ITRACE_SEMANTIC_MIN_COS \
    -u ITRACE_MIH_NODES -u ITRACE_MIH_INDEX_DIR \
    bash scripts/smoke_itrace_dataset.sh   # ×2
```

### Run 1 — PASS (`SMOKE_OK`)

| command | groups | rot90 | hflip | crop70 | slices |
|---|---|---|---|---|---|
| compare (phash,rot-inv) | 6 | 6/6 | 6/6 | 0/6 | 0/12 |
| smart | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (in-mem) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist build) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist load) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |

- 36 images, 6 families; persist-load `index_loaded=true,
  crop_index_loaded=true`

### Run 2 — PASS (`SMOKE_OK`)

`FUNCTIONAL_TEST.md` identical modulo the `Generated:` timestamp line;
persist-load again `index_loaded=true, crop_index_loaded=true`; sorted
group membership identical between runs.

Result: 6/6/6/12 baseline reproduced twice — the Phase 9 `_crop_mn`
wiring left the default path bit-identical (nodes unset →
`project_{id}_crop/` = `ITMIHC1` only, `_crop_mn` never written).
Artifacts: `*_r9base1.*`, `*_r9base2.*` copies of the report outputs.

## 2. Multi-node crop persist — `scripts/smoke_crop_mn_persist.sh` × 2

Two independent evidence dirs (`r9c`, `r9d`). Each pass runs one
project (`ITRACE_MIH_INDEX_DIR` shared, semantic off) through five
scans:

```bash
bash scripts/smoke_crop_mn_persist.sh r9c
bash scripts/smoke_crop_mn_persist.sh r9d
```

| scan | env | signals | meaning |
|---|---|---|---|
| sn1 | nodes unset | `index_loaded=false, crop_index_loaded=false` | fresh `project_1/` + `project_1_crop/` |
| mn1 | `NODES=3` | `index_loaded=false, crop_index_loaded=false` | fresh `project_1_mn/` + `project_1_crop_mn/` |
| mn2 | `NODES=3` | `index_loaded=true, crop_index_loaded=true` | `_mn` + `_crop_mn` node bundles restored |
| sn2 | nodes unset | `index_loaded=true, crop_index_loaded=true` | `ITMIHP1`/`ITMIHC1` hit — `_crop_mn` untouched |
| mn3 | `NODES=3` | `index_loaded=true, crop_index_loaded=true` | `_crop_mn` still hits after sn scans |

Identical on both passes; each exits `CROP_MN_PERSIST_SMOKE_OK`.

`_crop_mn` layout verified by the harness (and re-checked here):

```text
project_1_crop_mn/meta.json: {"magic":"ITMIHCN1","version":1,
  "shard_bits":8,"node_count":3,"image_count":8,"key_count":893,
  "feature_fingerprint":"<blake3 hex>",
  "ranges":[{"start":1,"end":2},{"start":3,"end":5},{"start":6,"end":8}]}
project_1_crop_mn/node_{0,1,2}/: meta.json(ITMIHC1) + image_ids.bin +
                                index/ shards
project_1_crop/meta.json:      ITMIHC1 (sibling, never aliased)
```

Every scan reports `+8 crop` candidates; group membership
`sn1=mn1=mn2=sn2=mn3=2` groups, sorted member lines identical — the
multi-node crop channel is exact-parity with single-node (each image's
keys live in exactly one node index).

## 3. Bypass regressions — gate `_mn` + semantic paths

| harness | result |
|---|---|
| `scripts/smoke_mih_nodes_persist.sh` (`r9g`) | `MIH_NODES_PERSIST_SMOKE_OK` — gate `ITMIHN1` build → `index_loaded=true` hit |
| `scripts/smoke_semantic_mn_persist.sh` (`r9s`) | `SEMANTIC_MN_PERSIST_SMOKE_OK` — `_sem_mn` `ITSEMN1` build → `sem_hnsw_loaded=true` hit |
| `scripts/smoke_semantic_persist.sh` (`r9p`) | `SEMANTIC_PERSIST_SMOKE_OK` — single-node `project_{id}_sem/` path untouched |

## 4. Quality gates

| gate | result |
|---|---|
| `cargo test --workspace` | 8/8 test binaries, 0 failures |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy --workspace --all-targets --features semantic-onnx -- -D warnings` | clean |

## 5. Non-goals respected

- No RPC / service discovery / network transport — in-process only
- No DINOv2/ONNX weights or downloads — stub-free (crop channel needs
  no embedder at all)
- Semantic contracts unchanged; confirmed merges still decided by
  hash/crop verification (semantic candidates-only preserved)
- No release tag / version bump; default smoke baseline unchanged

## Verdict

**Phase 9 foundation safe to close** — no blockers. Scheduler owns the
Final GATE.
