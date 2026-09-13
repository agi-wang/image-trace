# Functional Test — datasets/built microscopy subset

Generated: 2026-09-13 18:30 

Images: 36 across 6 families (sem/fluorescence/histology; orig, rot90, hflip, crop70, slice_r0c0, slice_r1c1).

## Group recovery (share of each family's variants found in the orig's group)

| command | groups | rot90 | hflip | crop70 | slices |
|---|---|---|---|---|---|
| compare (phash,rot-inv) | 6 | 6/6 | 6/6 | 0/6 | 0/12 |
| smart | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (in-mem) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist build) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |
| dedup (persist load) | 6 | 6/6 | 6/6 | 6/6 | 12/12 |

## Checks

- compare --rotation-invariant: **PASS**
- smart groups ≥ families: **PASS**
- dedup in-mem crop+slice recall: see table
- dedup persisted index loaded on 2nd run: **PASS** (crop index: PASS)
- slice orig↔r0c0: **PASS**

Raw outputs: `compare_phash.txt`, `smart.txt`, `dedup_mem.txt`,
`dedup_persist_build.txt`, `dedup_persist_load.txt`, `report_phash.txt`, `slice.json`.
