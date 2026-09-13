# Phase 5 Round 3 — Double Regression Evidence (ONNX backend + sem persist)

Date: 2026-09-13
Commit under test: `28aa33b` (merge of PR #13, `feat/semantic-persist-r2`) on `main`

Purpose: gate evidence for **Phase 5 Final** — the ONNX semantic backend
(`OnnxSemanticEmbedder` behind `semantic-onnx`, `load-dynamic`, no build-time
download) and the persisted `project_{id}_sem/` bundle
(`meta.json`/`image_ids.bin`/`vectors.bin`, `ITSEMP1` v1, embedder-fingerprint
+ BLAKE3 payload invalidation). This does not claim a production DINOv2
deployment — the stub is a deterministic dev stand-in, real weights were
never downloaded, and semantic hits remain candidates-only behind the
unchanged hash/crop verification.

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

### Run 2 — PASS (`SMOKE_OK`)

Identical `FUNCTIONAL_TEST.md` table and checks; persist-load again
`index_loaded=true, crop_index_loaded=true`. Group membership identical
between runs (verified by sorted-member diff).

Result: 6/6/6/12 baseline reproduced twice — the Phase 5 ONNX skeleton
and semantic persistence left the default path untouched. Artifacts:
`phase5_off1/`, `phase5_off2/`.

## 2. Flag-on semantic + persist — `scripts/smoke_semantic_persist.sh` × 2

New harness: fresh `--data-dir` + a shared `--index-dir` per tag; create →
add 9 images (`sem_synth_1` × orig/rot90/hflip/crop70/slice_r0c0,
`fluor_synth_1` × orig/rot90/crop70, plus a generated
`sem_synth_1__nudged.png` — orig with ~4% of pixels nudged — giving the
stub a guaranteed high-cosine pair) → `dedup` **three** times on the same
project:

1. flag-off baseline (no `ITRACE_MIH_INDEX_DIR` — the pure default path)
2. `ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR=$IDX`
   — armed, run 1 (must **build** the bundle)
3. same env — armed, run 2 (must **hit** the bundle)

Forced stub, no `ITRACE_SEMANTIC_MODEL`, default feature set → the ONNX
backend is not even compiled in; **no weights or runtime are fetched
anywhere**.

### Run `r3a` — PASS (`SEMANTIC_PERSIST_SMOKE_OK`)

| scan | candidates line | sem bundle |
|---|---|---|
| off | `+17 crop` | — |
| on1 | `+17 crop +1 sem` | built (`project_1_sem/` created) |
| on2 | `+17 crop +1 sem (cached)` | **hit** (`index_loaded=true`) |

- `off=2 on1=2 on2=2` duplicate groups; member lines identical across
  all three scans (sorted diff clean — no shrink, no spurious merges)
- bundle files present: `meta.json`, `image_ids.bin`, `vectors.bin`
- `meta.json`: `{"magic":"ITSEMP1","version":1,"embedder":"stub:g16",
  "dim":256,"image_count":9}` + BLAKE3 `feature_fingerprint`
- on2 also loads the persisted gate/crop bundles
  (`index_loaded=true, crop_index_loaded=true`) — only the embedding
  inference is skipped on the sem side; the HNSW still rebuilds per scan

### Run `r3b` — PASS (`SEMANTIC_PERSIST_SMOKE_OK`)

Identical table: `+17 crop` → `+1 sem` (build) → `+1 sem (cached)`
(hit); `off=2 on1=2 on2=2` with identical membership; same
`ITSEMP1`/`stub:g16`/256-dim/9-image metadata.

The `+1 sem` pair is `(sem_synth_1__orig, sem_synth_1__nudged)` — emitted
by the stub channel, unioned pre-verify, confirmed by the unchanged hash
gate on run 1, and re-derived from the persisted vectors on run 2.

## 3. Quality gate (once, this tree)

| gate | result |
|---|---|
| `cargo test --workspace` | 8/8 test binaries, 0 failures |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy -p itrace-core --all-targets --features semantic-onnx -- -D warnings` | clean |
| `cargo test -p itrace-core --features semantic-onnx` | 100 passed, 0 failed (fixture test skips without a runtime; no downloads) |

## Verdict

**Phase 5 foundation safe to close** — scheduler owns the Final GATE.

- Flag-off 6/6/6/12 baseline: reproduced ×2, bit-identical tables.
- Armed persist path: bundle builds on first armed scan, hits on second
  (`(cached)` + `index_loaded=true`), confirmed groups identical to
  flag-off, fingerprint metadata correct.
- No model weights or ONNX runtime binaries were downloaded; the stub
  exercised wiring + persistence only.

Named follow-ups (non-blocking): real `dinov2_vits14`/`vitb14` weights +
recall calibration on transformed fixtures; persisting the HNSW graph
itself if rebuild cost matters at scale; multi-node semantic sharding.
