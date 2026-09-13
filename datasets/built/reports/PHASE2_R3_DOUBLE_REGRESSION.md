# Phase 2 Round 3 — Double Regression Evidence

Date: 2026-09-13
Commit under test: `639a8ee` (merge of PR #4, `feat/postgres-store-polish`) on `main`

Purpose: provide the double-regression evidence the scheduler needs to GATE
Phase 2 Final — sqlite path unchanged, Postgres path reproducible from clean
state, workspace green.

## 1. Sqlite smoke — `scripts/smoke_itrace_dataset.sh` × 2 (same tree)

Both runs executed back-to-back on the same checkout. The harness recreates its
report/index dirs each invocation; within each run it executes compare, smart,
dedup in-memory, dedup persist-build, and dedup persist-load.

### Run 1 — PASS (`SMOKE_OK`)

| command | groups | rot90 | hflip | crop70 | slices |
|---|---|---|---|---|---|
| compare (phash,rot-inv) | 6 | 6/6 | 6/6 | 0/6 | 0/12 |
| smart | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (in-mem) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist build) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist load) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |

- 36 images, 6 families (sem/fluorescence/histology; orig, rot90, hflip,
  crop70, slice_r0c0, slice_r1c1)
- persist load pass: `index_loaded=true, crop_index_loaded=true`
- slice orig↔r0c0: `is_slice_of_a=true`, coverage 1.0

### Run 2 — PASS (`SMOKE_OK`)

Identical table to Run 1; identical check results:

- persist load pass: `index_loaded=true, crop_index_loaded=true`
- persisted-index recall on both persist passes: 6/6 rot90, 6/6 hflip,
  6/6 crop70, 12/12 slices

Baseline match: Round 2 metrics (6/6/6/12) reproduced on both runs.

## 2. Postgres path — `--store postgres` × 2, fresh schema each run

Setup: real Postgres 16.4 (zonky embedded binaries) on `127.0.0.1:54329`;
`ITRACE_DATABASE_URL=postgres://itrace@127.0.0.1:54329/postgres`.

Reset procedure between runs: `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`
(full schema wipe — verified `information_schema` shows 0 public tables before
run 2; PostgresStore re-applies DDL on connect). Each run also used a fresh
`--data-dir` (`/tmp/pg-r3`, `/tmp/pg-r3b`), so no state whatsoever carried over.

Commands per run (identical):

```
itrace-cli --data-dir <dir> --store postgres create <name>
itrace-cli --data-dir <dir> --store postgres add <pid> \
  sem_synth_1__{orig,rot90,hflip,crop70}.png \
  fluor_synth_1__{orig,rot90}.png
itrace-cli --data-dir <dir> --store postgres dedup <pid>
```

### Run 1 (project `r3-a`, id 1) — PASS

```
duplicate group 1 (confidence 1.0000):
  - sem_synth_1__orig.png, sem_synth_1__rot90.png,
    sem_synth_1__hflip.png, sem_synth_1__crop70.png
duplicate group 2 (confidence 1.0000):
  - fluor_synth_1__orig.png, fluor_synth_1__rot90.png
indexed 6, candidates 4 (+7 crop), naive 15, 2 dup groups / 6 images
```

### Run 2 (project `r3-b`, id 1 — ids restart, proving the wipe) — PASS

```
duplicate group 1 (confidence 1.0000):
  - fluor_synth_1__orig.png, fluor_synth_1__rot90.png
duplicate group 2 (confidence 1.0000):
  - sem_synth_1__orig.png, sem_synth_1__rot90.png,
    sem_synth_1__hflip.png, sem_synth_1__crop70.png
indexed 6, candidates 4 (+7 crop), naive 15, 2 dup groups / 6 images
```

Both runs group orig+rot90 (plus hflip/crop70) per family at confidence 1.0;
no cross-family merges; results identical across fresh schemas — a true replay.

## 3. Workspace verification (main @ `639a8ee`)

- `cargo test --workspace` — **82 passed, 0 failed**
  (61 core + 16 server + 3 store + 2 contract; `contract_postgres` ran green
  against the live PG instance via `ITRACE_TEST_DATABASE_URL`)
- `cargo clippy --workspace --all-targets -- -D warnings` — **clean**

## Conclusion

No regressions observed on either backend. Sqlite default path behavior,
persistent MIH index reuse (gate + crop, `feature_fingerprint` invalidation),
and the Postgres backend's create/add/precompute/dedup path all reproduce the
Round 2 baseline on `main` at `639a8ee`.

**Phase 2 is safe to close.** Remaining follow-ups are explicitly out of
scope for the phase: pg-vs-sqlite perf benchmarks, sqlite→pg dump/migration
tool, Phase 3 multi-node shard ownership, Phase 4 DINOv2/HNSW.
