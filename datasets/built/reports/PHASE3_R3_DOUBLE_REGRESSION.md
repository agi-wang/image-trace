# Phase 3 Round 3 — Double Regression Evidence (foundation)

Date: 2026-09-13
Commit under test: `3ab52ef` (merge of PR #7, `feat/mih-shard-ownership-r2`) on `main`

Purpose: gate evidence for **Phase 3 foundation Final** — the in-process
shard-ownership + scatter/gather layer (`ShardOwnership`,
`MultiNodeMihIndex`, `ITMIHN1` per-node persistence, `ITRACE_MIH_NODES`
dev path). This does not claim a networked product cluster.

## 1. Sqlite default path — `scripts/smoke_itrace_dataset.sh` × 2

Both runs executed back-to-back on the same tree; the harness recreates
its data/report/index dirs each invocation.

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
- slice orig↔r0c0: `is_slice_of_a=true`, coverage 1.0

### Run 2 — PASS (`SMOKE_OK`)

Identical table and checks; persist-load again showed
`index_loaded=true, crop_index_loaded=true`.

Result: matches the Phase 2 R3 baseline exactly (6/6/6/12) — the
Phase 3 ownership work introduced no regression on the default path.

## 2. Multi-node simulation — `scripts/smoke_mih_nodes.sh` × 2

New harness: fresh `--data-dir` per mode; create → add 8 images
(2 families: `sem_synth_1`, `fluor_synth_1` × orig/rot90/hflip/crop70,
plus a `sem_synth_1` slice tile) → `dedup` twice — once with
`ITRACE_MIH_NODES` unset (monolithic `ShardedMihIndex`) and once with
`ITRACE_MIH_NODES=4` (`MultiNodeMihIndex`, 4 logical nodes). Asserts
equal group count and byte-identical group membership.
`ITRACE_MIH_INDEX_DIR` deliberately unset so the env-routed
non-persistent path is exercised.

Exact commands (script equivalent):

```bash
itrace-cli --data-dir <dir> create mih-nodes-<mode>
itrace-cli --data-dir <dir> add <pid> <8 files>
itrace-cli --data-dir <dir> dedup <pid>                 # mono
ITRACE_MIH_NODES=4 itrace-cli --data-dir <dir> dedup <pid>  # 4 nodes
```

### Invocation 1 (`r3a`) — PASS (`MIH_NODES_SMOKE_OK`)

```
mono  : duplicate group 1 (1.0000): sem_synth_1 orig/rot90/hflip/crop70/slice_r0c0
        duplicate group 2 (1.0000): fluor_synth_1 orig/rot90/crop70
        indexed 8, candidates 4 (+12 crop), 2 dup groups / 8 images
multi : identical groups, identical members, identical candidate counts
mono groups=2  multi groups=2
```

### Invocation 2 (`r3b`) — PASS (`MIH_NODES_SMOKE_OK`)

Identical output on a fresh data dir — `mono groups=2  multi groups=2`.

Result: 4-node scatter/gather reproduces mono recall exactly on gate
near-dups **and** the crop channel (the slice tile lands in group 1 via
the `+12 crop` candidates) — true replay, no leftover state.

Unit/integration coverage backing this:
`dedup_candidates_sharded_multi_node_env_parity` (env path),
`multi_node_parity_{2,4}_nodes`, `multi_node_parity_shard_bits_zero`,
`multi_node_parity_crop_like_many_keys_per_owner`,
`multi_node_query_only_contacts_probe_set_owners` (fan-out skip),
`multi_node_persist_roundtrip_parity` (ITMIHN1 save/load),
`multi_node_persist_rejects_wrong_magic`.

## 3. Workspace verification (main @ `3ab52ef`)

- `cargo test --workspace` — **94 passed, 0 failed**
  (73 core + 16 server + 3 store + 2 contract)
- `cargo clippy --workspace --all-targets -- -D warnings` — **clean**

## Conclusion

Phase 3 **foundation** is safe to close: shard ownership partitioning,
in-process multi-node scatter/gather with parity + fan-out guarantees,
and per-node `ITMIHN1` persistence are all proven, with zero regression
on the default single-node path.

Explicitly not yet delivered (future rounds): real RPC transport,
service discovery, per-node persistence wired into the persistent
`ITMIHP1`/`ITMIHC1` project-bundle flow, Phase 4 semantic channel.
