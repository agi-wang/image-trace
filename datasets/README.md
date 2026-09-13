# Microscopy test dataset

Reproducible dataset pipeline for exercising image-trace duplicate /
transform detection. Not a clinical corpus — see `manifests/SOURCES.md`
for provenance of the seeds.

## Layout

```
datasets/
  raw/<modality>/         public/open sample seeds (jpg/png)
  synthetic/<modality>/   procedural PNG stand-ins
  manifests/SOURCES.md    seed provenance
  built/<modality>/       generated outputs (git-ignored artifacts)
  built/manifest.jsonl    one record per generated image
  built/reports/          smoke-test output from scripts/smoke_itrace_dataset.sh
```

Modalities: `sem`, `fluorescence`, `histology`.

## Build

```bash
python3 scripts/build_dataset.py          # incremental, idempotent
python3 scripts/build_dataset.py --clean  # rebuild from scratch
```

Each source image is normalized to RGB PNG and expanded into 9 variants:

| variant        | transform                                   |
|----------------|---------------------------------------------|
| `orig`         | normalized copy                             |
| `rot90`        | 90° CCW rotation                            |
| `rot180`       | 180° rotation                               |
| `hflip`        | horizontal mirror                           |
| `crop70`       | center crop, ~70% of each dimension         |
| `slice_r{R}c{C}` | 2x2 grid tiles (4 outputs)                |

`manifest.jsonl` fields per line:
`id`, `collection` (raw|synthetic), `modality`, `source_path`,
`variant`, `width`, `height`, `sha256`.

## Smoke test

```bash
./scripts/smoke_itrace_dataset.sh
```

Builds `itrace-cli`, creates a project, adds every built image, then runs
`compare` (phash, rotation-invariant) and `smart` voting. Logs land in
`datasets/built/reports/`; the script exits non-zero on hard errors or if
transform variants of the same source fail to group together.
