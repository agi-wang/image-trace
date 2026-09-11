#!/usr/bin/env python3
"""Build the microscopy test dataset for Image Trace.

Walks datasets/raw/<modality>/ and datasets/synthetic/<modality>/, normalizes
each source to RGB PNG under datasets/built/<modality>/, and emits transform
variants used to exercise itrace duplicate detection:

    orig      normalized copy
    rot90     90-degree counter-clockwise rotation
    rot180    180-degree rotation
    hflip     horizontal mirror
    crop70    center crop at ~70% of each dimension
    slice_r{R}c{C}  2x2 grid tiles (4 outputs)

Every output is recorded in datasets/built/manifest.jsonl with
id, collection, modality, source_path, variant, width, height, sha256.

Idempotent: outputs whose sha256 already matches are not rewritten; the
manifest is regenerated deterministically on every run.

Usage: python3 scripts/build_dataset.py [--clean]
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODALITIES = ("sem", "fluorescence", "histology")
COLLECTIONS = ("raw", "synthetic")
GRID_ROWS = GRID_COLS = 2

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def grid_tiles(img: Image.Image, rows: int, cols: int):
    w, h = img.size
    xs = [round(c * w / cols) for c in range(cols + 1)]
    ys = [round(r * h / rows) for r in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            yield f"slice_r{r}c{c}", img.crop((xs[c], ys[r], xs[c + 1], ys[r + 1]))


def variants(img: Image.Image):
    yield "orig", img
    yield "rot90", img.transpose(Image.ROTATE_90)
    yield "rot180", img.transpose(Image.ROTATE_180)
    yield "hflip", img.transpose(Image.FLIP_LEFT_RIGHT)
    yield "crop70", center_crop(img, 0.70)
    yield from grid_tiles(img, GRID_ROWS, GRID_COLS)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(clean: bool = False) -> int:
    built = DATASETS / "built"
    if clean and built.exists():
        import shutil

        shutil.rmtree(built)
    built.mkdir(parents=True, exist_ok=True)

    entries = []
    written = skipped = 0
    for collection in COLLECTIONS:
        for modality in MODALITIES:
            src_dir = DATASETS / collection / modality
            if not src_dir.is_dir():
                continue
            for src in sorted(src_dir.iterdir()):
                if src.suffix.lower() not in IMAGE_EXTS:
                    continue
                with Image.open(src) as im:
                    img = im.convert("RGB")
                    img.load()
                out_dir = built / modality
                out_dir.mkdir(parents=True, exist_ok=True)
                for variant, out in variants(img):
                    data = png_bytes(out)
                    digest = sha256(data)
                    out_path = out_dir / f"{src.stem}__{variant}.png"
                    if out_path.exists() and sha256(out_path.read_bytes()) == digest:
                        skipped += 1
                    else:
                        out_path.write_bytes(data)
                        written += 1
                    entries.append(
                        {
                            "id": f"{modality}/{out_path.stem}",
                            "collection": collection,
                            "modality": modality,
                            "source_path": str(src.relative_to(DATASETS)),
                            "variant": variant,
                            "width": out.size[0],
                            "height": out.size[1],
                            "sha256": digest,
                        }
                    )

    entries.sort(key=lambda e: e["id"])
    manifest = built / "manifest.jsonl"
    payload = "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
    if not manifest.exists() or manifest.read_text() != payload:
        manifest.write_text(payload)
    print(f"built: {written} written, {skipped} unchanged, "
          f"{len(entries)} manifest entries -> {manifest}")
    return len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true", help="rebuild from scratch")
    args = ap.parse_args()
    n = build(clean=args.clean)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
