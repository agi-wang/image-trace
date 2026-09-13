# Dataset seed sources

`datasets/raw/` is git-ignored (public sample binaries). Re-fetch the seeds
below, or drop any equivalent JPG/PNG into `raw/<modality>/` — the builder
walks whatever is present. `datasets/synthetic/` is committed so the
pipeline works out of the box.

## scikit-image samples

```bash
python3 - <<'PY'
import skimage.data, skimage.io
skimage.io.imsave('datasets/raw/sem/skimage_moon.png', skimage.data.moon())
skimage.io.imsave('datasets/raw/histology/skimage_ihc.png',
                  skimage.data.immunohistochemistry())
PY
```

- `raw/sem/skimage_moon.png` — `skimage.data.moon()` (contrast texture
  stand-in; not true SEM)
- `raw/histology/skimage_ihc.png` / `skimage_ihc2.png` —
  `skimage.data.immunohistochemistry()` (ihc2 is a second saved copy /
  alternate normalization of the same sample)

## ImageJ / Fiji samples

Bundled with any ImageJ or Fiji install: `File > Open Samples`, then export
as JPG/PNG into `raw/<modality>/`.

- `raw/fluorescence/FluorescentCells.jpg` / `fluor_cells.jpg` /
  `imagej_fluor.jpg` — NIH ImageJ "Fluorescent Cells" sample
- `raw/histology/imagej_blobs.png` — ImageJ "Blobs" sample

## Wikimedia Commons

- `raw/sem/sem_pollen_wiki.jpg` — pollen SEM-related micrograph from
  Wikimedia Commons (search "pollen SEM"; any high-contrast SEM micrograph
  is a fine substitute)

## Synthetic (committed under `synthetic/`)

- Procedural PNG stand-ins for SEM grain, fluorescence blobs, H&E-like
  histology — generated locally for reproducible transforms
  (rotate/crop/slice) in the builder.

These seeds are for pipeline/dev testing of image-trace, not a clinical
corpus.
