# Phase 4 Round 3 — Double Regression Evidence (foundation)

Date: 2026-09-13
Commit under test: `fb892a9` (merge of PR #10, `feat/semantic-hnsw-r2`) on `main`

Purpose: gate evidence for **Phase 4 foundation Final** — the semantic
recall channel foundation (`SemanticEmbedder` trait, `StubEmbedder`,
in-memory `HnswIndex`, `semantic_candidates` + `union_candidate_pairs`,
`dedup_confirmed_cached_with_extra`, and the `ITRACE_SEMANTIC`-gated
wiring in `run_cli_dedup` / `run_dedup_scan`). This does not claim a
production DINOv2 deployment — the stub is a deterministic dev stand-in
and semantic hits remain candidates-only behind the unchanged
hash/crop verification.

## 1. Flag-off baseline — `scripts/smoke_itrace_dataset.sh` × 2

Both runs back-to-back on the same tree with `ITRACE_SEMANTIC`,
`ITRACE_SEMANTIC_STUB`, `ITRACE_SEMANTIC_MODEL` all unset; the harness
recreates its data/report/index dirs each invocation.

```bash
env -u ITRACE_SEMANTIC -u ITRACE_SEMANTIC_STUB -u ITRACE_SEMANTIC_MODEL \
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
- slice orig↔r0c0: **PASS**

### Run 2 — PASS (`SMOKE_OK`)

Identical `FUNCTIONAL_TEST.md` table and checks; persist-load again
`index_loaded=true, crop_index_loaded=true`. Raw `dedup_mem.txt` group
**ordering** differed between runs (pre-existing rayon nondeterminism in
print order); group membership was identical.

Result: 6/6/6/12 baseline reproduced twice — the Phase 4 wiring left the
default path untouched.

## 2. Flag-on semantic stub — `scripts/smoke_semantic_stub.sh` × 2

New harness: fresh `--data-dir` per run; create → add 9 images
(`sem_synth_1` × orig/rot90/hflip/crop70/slice_r0c0, `fluor_synth_1` ×
orig/rot90/crop70, plus a generated `sem_synth_1__nudged.png` — orig with
~4% of pixels nudged — giving the stub a guaranteed high-cosine pair) →
`dedup` twice on the same project: flag-off baseline, then
`ITRACE_SEMANTIC=1` (auto-stub; `ITRACE_SEMANTIC_MODEL` unset).
All other `ITRACE_SEMANTIC*`/`ITRACE_MIH_INDEX_DIR` knobs are unset so
the exercised path is exactly `embedder_from_env` → `StubEmbedder` →
`semantic_candidates` → `union_candidate_pairs` → unchanged verify.

Exact commands (script equivalent):

```bash
itrace-cli --data-dir <dir> create semantic-stub-<tag>
itrace-cli --data-dir <dir> add <pid> <9 files>
itrace-cli --data-dir <dir> dedup <pid>                   # flag off
ITRACE_SEMANTIC=1 itrace-cli --data-dir <dir> dedup <pid> # auto-stub
```

### Invocation 1 (`r3a`) — PASS (`SEMANTIC_STUB_SMOKE_OK`)

```
flag off: indexed 9, candidates 7 (+17 crop), 2 dup groups / 9 images
flag on : indexed 9, candidates 7 (+17 crop +1 sem), 2 dup groups / 9 images
flag-off groups=2  flag-on groups=2  sem=1
```

`+1 sem` = the `(sem_synth_1__orig, sem_synth_1__nudged)` pair — emitted by
the stub channel, unioned pre-verify, confirmed as before. Group
membership identical between the two scans (asserted via diff).

### Invocation 2 (`r3b`) — PASS (`SEMANTIC_STUB_SMOKE_OK`)

Identical output on a fresh data dir — `sem=1`, `flag-off groups=2`,
`flag-on groups=2`.

Result: armed channel emits real candidate pairs end-to-end and confirmed
groups do not shrink — while every semantic hit still passes the
unchanged hash verification, so merges cannot widen silently.

Unit/integration coverage backing this:
`embedder_from_env_registration` (env precedence incl. auto-stub and the
never-silent-stub rule), `stub_embedder_deterministic_and_dim`,
`stub_embedder_near_dup_close_far_apart`, `stub_channel_emits_candidate_pair`,
`dedup_confirmed_cached_with_extra_unions_before_verify` (extra pair
missed by MIH confirms via the union; rejected pair stays out),
`dedup_confirmed_cached_with_extra_empty_parity` (flag-off ≡ baseline).

## 3. Quality gate

| check | result |
|---|---|
| `cargo test --workspace` | PASS — 8/8 test binaries, 0 failures |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS — clean |

## Verdict

Phase 4 foundation behaves exactly as designed: default path
bit-identical (6/6/6/12 twice), armed path emits semantic candidates and
unions them pre-verification twice with no group shrink and no crash.
**Phase4 foundation safe to close** — pending scheduler Final GATE.
Follow-ups stay open: DINOv2 ONNX backend behind `ITRACE_SEMANTIC_MODEL`,
persisted `project_{id}_sem/` embedding bundle, per-scan decode →
precomputed feature vectors.
