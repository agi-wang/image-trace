# Phase 6 Round 3 — Double Regression Evidence (HNSW persist)

Date: 2026-09-14
Commit under test: `22683c3` (merge of PR #16, `chore/phase6-r2-hnsw-harness`) on `main`

Purpose: gate evidence for **Phase 6 Final** — persisted HNSW graph
(`project_{id}_sem/hnsw.bin`, `ITSEMH1` v1, fingerprint-bound to the
verified vectors, miss → rebuild + overwrite) plus the hardened smoke
harness that asserts `sem_hnsw_loaded=true` on a cache hit. This does
not claim a production DINOv2 deployment — the stub is a deterministic
dev stand-in, real weights were never downloaded, and semantic hits
remain candidates-only behind the unchanged hash/crop verification.

## 1. Flag-off baseline — `scripts/smoke_itrace_dataset.sh` × 2

Both runs back-to-back on the same tree with `ITRACE_SEMANTIC`,
`ITRACE_SEMANTIC_STUB`, `ITRACE_SEMANTIC_MODEL`, `ITRACE_MIH_INDEX_DIR`
all unset; the harness recreates its data/report/index dirs each
invocation.

```bash
env -u ITRACE_SEMANTIC -u ITRACE_SEMANTIC_STUB -u ITRACE_SEMANTIC_MODEL \
    -u ITRACE_MIH_INDEX_DIR bash scripts/smoke_itrace_dataset.sh   # ×2
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
- slice orig↔r0c0: **PASS**
- no sem fields anywhere in the output (`+N sem`, `sem_hnsw_loaded` absent)

### Run 2 — PASS (`SMOKE_OK`)

Identical `FUNCTIONAL_TEST.md` table and checks (modulo the `Generated:`
timestamp line); persist-load again `index_loaded=true,
crop_index_loaded=true`. Group membership identical between runs
(sorted-member diff clean).

Result: 6/6/6/12 baseline reproduced twice — Phase 6 persistence left
the default path untouched. Artifacts: `phase6_off1/`, `phase6_off2/`.

## 2. Flag-on semantic + HNSW persist — `scripts/smoke_semantic_persist.sh` × 2

Two independent harness passes (`r6b`, `r6c`), each with a fresh
`--data-dir` + shared `--index-dir`: create → add 9 images
(`sem_synth_1` × orig/rot90/hflip/crop70/slice_r0c0, `fluor_synth_1` ×
orig/rot90/crop70, plus generated `sem_synth_1__nudged.png`) → `dedup`
three times on the same project:

1. flag-off baseline (no `ITRACE_MIH_INDEX_DIR`)
2. `ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR=$IDX`
   — armed, run 1 (must **build** vectors + graph)
3. same env — armed, run 2 (must **hit both**)

Forced stub, no `ITRACE_SEMANTIC_MODEL`, default feature set → the ONNX
backend is not even compiled in; **no weights or runtime are fetched
anywhere**.

### Run `r6b` — PASS (`SEMANTIC_PERSIST_SMOKE_OK`)

| scan | candidates line | bundle state |
|---|---|---|
| off | `+17 crop` | — |
| on1 | `+17 crop +1 sem, sem_hnsw_loaded=false` | built (`project_1_sem/` incl. `hnsw.bin`) |
| on2 | `+17 crop +1 sem (cached), sem_hnsw_loaded=true` | **hit** — vectors AND graph restored |

- `off=2 on1=2 on2=2` duplicate groups; member lines identical across
  all three scans (sorted diff clean — no shrink, no spurious merges)
- bundle files: `meta.json`, `image_ids.bin`, `vectors.bin`, `hnsw.bin`
- `meta.json`: `{"magic":"ITSEMP1","version":1,"embedder":"stub:g16",
  "dim":256,"image_count":9}` + BLAKE3 `feature_fingerprint`
- `hnsw.bin` begins with `ITSEMH1` magic; on2 also loads the persisted
  gate/crop bundles (`index_loaded=true, crop_index_loaded=true`)

### Run `r6c` — PASS (`SEMANTIC_PERSIST_SMOKE_OK`)

Identical table: `+17 crop` → `+1 sem, sem_hnsw_loaded=false` (build) →
`+1 sem (cached), sem_hnsw_loaded=true` (hit); `off=2 on1=2 on2=2` with
identical membership; same `ITSEMP1`/`stub:g16`/256-dim/9-image metadata
and `ITSEMH1` graph file.

The `+1 sem` pair is `(sem_synth_1__orig, sem_synth_1__nudged)` —
stub-emitted, unioned pre-verify, confirmed by the unchanged hash gate;
on run 2 it is re-derived entirely from the persisted vectors + graph
(zero embedding, zero re-insertion).

## 3. Quality gate (once, this tree)

| gate | result |
|---|---|
| `cargo test --workspace` | 8/8 test binaries, 0 failures |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy -p itrace-core --all-targets --features semantic-onnx -- -D warnings` | clean |
| `cargo test -p itrace-core --features semantic-onnx` | 103+16 passed, 0 failed (fixture test skips without a runtime; no downloads) |

## Verdict

**Phase 6 foundation safe to close** — scheduler owns the Final GATE.

- Flag-off 6/6/6/12 baseline: reproduced ×2, identical tables/membership.
- Armed persist path: `hnsw.bin` builds on first armed scan, restores on
  second (`(cached)` + `sem_hnsw_loaded=true`), confirmed groups
  identical to flag-off.
- No model weights or ONNX runtime binaries were downloaded; the stub
  exercised wiring + persistence only.

Named follow-ups (non-blocking): multi-node semantic/HNSW sharding;
graph format compaction (u16 neighbour refs, vecs derived from
`vectors.bin`) if bundle size matters; real DINOv2 weights + recall
calibration.
