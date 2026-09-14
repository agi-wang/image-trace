# Phase 7 Round 3 — Double Regression Evidence (ITMIHN1 project persist)

Date: 2026-09-14
Commit under test: `d443f5f` (merge of PR #19, `chore/phase7-r2-mihn-harness`) on `main`

Purpose: gate evidence for **Phase 7 Final** — wiring the `ITMIHN1`
multi-node MIH layout into the project index-dir flow
(`project_{id}_mn/` sibling bundle: top-level `ITMIHN1` meta +
`image_ids.bin` + `indexes/{a}/` cluster dirs of `node_N/` shard bins),
plus the persist smoke harness that asserts the build → cache-hit
transition end-to-end. This exercises the in-process multi-node path
only — no RPC, no service discovery, no semantic/HNSW sharding, no
weight downloads.

## 1. Single-node baseline — `scripts/smoke_itrace_dataset.sh` × 2

Both runs back-to-back on the same tree with `ITRACE_MIH_NODES`,
`ITRACE_MIH_INDEX_DIR`, and all `ITRACE_SEMANTIC*` knobs unset; the
harness recreates its data/report/index dirs each invocation.

```bash
env -u ITRACE_SEMANTIC -u ITRACE_SEMANTIC_STUB -u ITRACE_SEMANTIC_MODEL \
    -u ITRACE_MIH_INDEX_DIR -u ITRACE_MIH_NODES \
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

Identical `FUNCTIONAL_TEST.md` table and checks (modulo the
`Generated:` timestamp line); persist-load again `index_loaded=true,
crop_index_loaded=true`. Group membership identical between runs
(sorted-member diff clean).

Result: 6/6/6/12 baseline reproduced twice — the Phase 7 `_mn` wiring
left the default single-node path bit-identical (`project_{id}/` =
`ITMIHP1`, `_mn` never written). Artifacts: `baseline_r7_a/`,
`baseline_r7_b/`.

## 2. Multi-node persist — `scripts/smoke_mih_nodes_persist.sh` × 2

Two independent harness passes (`r7c`, `r7d`), each with a fresh
`--data-dir` + shared `--index-dir`: create → add 8 images
(`sem_synth_1` × orig/rot90/hflip/crop70/slice_r0c0, `fluor_synth_1` ×
orig/rot90/crop70) → `dedup` four times on the same project + index
dir: mono build (nodes unset) → multi-node build
(`ITRACE_MIH_NODES=4`) → multi-node hit → mono hit.

Both passes exited `MIH_NODES_PERSIST_SMOKE_OK` with:

| scan | `index_loaded` | meaning |
|---|---|---|
| mono1 (nodes unset) | `false` | fresh `project_{id}/` `ITMIHP1` build |
| mn1 (`NODES=4`) | `false` | fresh `project_{id}_mn/` `ITMIHN1` build |
| mn2 (`NODES=4`) | `true` | `_mn` cluster shards restored, not rebuilt |
| mono2 (nodes unset) | `true` | `ITMIHP1` hit — `_mn` sibling untouched |

Verified on-disk layout each pass:

```text
project_1_mn/meta.json   magic=ITMIHN1 version=1 shard_bits=8
                         node_count=4 gate_algo_count=4 image_count=8
                         + BLAKE3 feature_fingerprint
project_1_mn/indexes/{a}/ meta.json (ranges) + node_0..node_3/
                          {meta.json, shards/NNNN.bin}
project_1/meta.json      magic=ITMIHP1 — coexists, never read or
                         corrupted by the multi-node scans
```

Group count + sorted membership identical across all four scans
(`2/2/2/2`; diff clean for mn1/mn2/mono2 vs the mono1 baseline).

Result: multi-node build → cache-hit proven twice; the two bundle
formats coexist on one index dir without interference. Artifacts:
`mih_nodes_persist_r7c/`, `mih_nodes_persist_r7d/` (outputs + `_mn`
bundles + `project_1/meta.json`).

## 3. Non-persist parity — `scripts/smoke_mih_nodes.sh` × 1

`bash scripts/smoke_mih_nodes.sh r7final` → `MIH_NODES_SMOKE_OK`:
`mono groups=2 multi groups=2`, sorted member sets identical
(multi-node scatter/gather may emit groups in a different order; the
membership contract is unchanged). Artifacts: `mih_nodes_r7final/`.

## 4. Quality gates

| gate | result |
|---|---|
| `cargo test --workspace` | 8/8 test binaries, 0 failures (itrace-core 100 passed) |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy -p itrace-core --all-targets --features semantic-onnx -- -D warnings` | clean |
| `cargo test -p itrace-core --features semantic-onnx` | 105 passed, 0 failed (fixture test skips — no runtime, no downloads) |

No model weights or runtimes were downloaded; all runs used default
features with the semantic channel off / stub-only.

## Verdict

**Phase 7 foundation safe to close.** `project_{id}_mn/` (`ITMIHN1`)
persist/load is proven end-to-end (unit round-trip + invalidation
tests, harness build→hit transitions, coexistence with `ITMIHP1`), the
single-node baseline is bit-identical, and membership parity holds
across mono and multi-node scans. Remaining non-goals (real RPC,
semantic multi-node sharding) are explicitly deferred.
