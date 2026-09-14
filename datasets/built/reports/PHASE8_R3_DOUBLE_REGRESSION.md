# Phase 8 Round 3 — Double Regression Evidence (semantic multi-node HNSW persist)

Date: 2026-09-14
Commit under test: `e51b932` (merge of PR #22, `chore/phase8-r2-sem-mn-harness`) on `main`

Purpose: gate evidence for **Phase 8 Final** — multi-node semantic/HNSW
sharding via `project_{id}_sem_mn/` (`ITSEMN1` v1: top-level meta +
contiguous `ranges` + one complete `ITSEMP1`/`ITSEMH1` `node_{i}/`
bundle per sorted-id range; `semantic_candidates_multi` scatter/merge;
semantic remains candidates-only), plus the persist smoke harness that
asserts the build → cache-hit transition end-to-end. This exercises the
in-process multi-node path only — no RPC, no service discovery, no
weight downloads (forced `StubEmbedder`).

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

- 36 images, 6 families (sem/fluorescence/histology)
- persist-load pass: `index_loaded=true, crop_index_loaded=true`
- no sem fields (`+N sem`, `sem_hnsw_loaded`) anywhere in flag-off output

### Run 2 — PASS (`SMOKE_OK`)

`FUNCTIONAL_TEST.md` identical modulo the `Generated:` timestamp line;
persist-load again `index_loaded=true, crop_index_loaded=true`; sorted
group membership identical between runs.

Result: 6/6/6/12 baseline reproduced twice — the Phase 8 `_sem_mn`
wiring left the default path bit-identical (semantic off → no semantic
bundle written or read at all). Artifacts: `*_r8base1.*`, `*_r8base2.*`
copies of the report outputs.

## 2. Multi-node semantic persist — `scripts/smoke_semantic_mn_persist.sh` × 2

Two independent evidence dirs (`r8c`, `r8d`). Each pass runs one
armed-stub project (`ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1`,
`ITRACE_MIH_INDEX_DIR` shared) through five scans:

```bash
bash scripts/smoke_semantic_mn_persist.sh r8c
bash scripts/smoke_semantic_mn_persist.sh r8d
```

| scan | env | sem signal | `index_loaded` | meaning |
|---|---|---|---|---|
| sn1 | nodes unset | `+1 sem, sem_hnsw_loaded=false` | `false` | fresh `project_1/` + `project_1_sem/` |
| mn1 | `NODES=3` | `+1 sem, sem_hnsw_loaded=false` | `false` | fresh `project_1_mn/` + `project_1_sem_mn/` |
| mn2 | `NODES=3` | `+1 sem (cached), sem_hnsw_loaded=true` | `true` | `_sem_mn` vectors + all node graphs restored |
| sn2 | nodes unset | `+1 sem (cached), sem_hnsw_loaded=true` | `true` | `ITSEMP1` hit — `_sem_mn` untouched |
| mn3 | `NODES=3` | `+1 sem (cached), sem_hnsw_loaded=true` | `true` | `_sem_mn` still hits after sn scans |

Identical on both passes; each exits `SEMANTIC_MN_PERSIST_SMOKE_OK`.

`_sem_mn` layout verified by the harness (and re-checked here):

```text
project_1_sem_mn/meta.json: {"magic":"ITSEMN1","version":1,
  "node_count":3,"embedder":"stub:g16","dim":256,"image_count":9,
  "feature_fingerprint":"<blake3 hex>",
  "ranges":[{"start":1,"end":3},{"start":4,"end":6},{"start":7,"end":9}]}
project_1_sem_mn/node_{0,1,2}/: meta.json(ITSEMP1) + image_ids.bin +
                               vectors.bin + hnsw.bin(ITSEMH1)
project_1_sem/meta.json:     ITSEMP1 (sibling, never aliased)
```

Group membership: `sn1=mn1=mn2=sn2=mn3=2` groups, sorted member lines
identical across all five scans — the sharded semantic channel neither
loses nor confirms anything extra.

## 3. Single-node coexistence regression — `scripts/smoke_semantic_persist.sh` × 1

```bash
bash scripts/smoke_semantic_persist.sh r8e
```

PASS (`SEMANTIC_PERSIST_SMOKE_OK`): off `+17 crop` → on1 `+1 sem,
sem_hnsw_loaded=false` → on2 `+1 sem (cached), sem_hnsw_loaded=true`;
`off=on1=on2=2` groups with identical membership. The R1/R2 single-node
`project_{id}_sem/` path is untouched by the `_sem_mn` sibling.

## 4. Quality gates

| gate | result |
|---|---|
| `cargo test --workspace` | 8/8 test binaries, 0 failures |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy --workspace --all-targets --features semantic-onnx -- -D warnings` | clean |
| `cargo test -p itrace-core --features semantic-onnx` | clean (feature tests) |

## 5. Non-goals respected

- No RPC / service discovery / network transport — in-process only
- No DINOv2/ONNX weights or downloads — stub-only evidence; default
  features do not even compile the ONNX backend
- `project_{id}_crop/` multi-node untouched (later round)
- Semantic remains candidates-only — confirmed merges decided by
  hash/crop verification exactly as before
- No release tag / version bump; default smoke baseline unchanged

## Verdict

**Phase 8 foundation safe to close** — no blockers. Scheduler owns the
Final GATE.
