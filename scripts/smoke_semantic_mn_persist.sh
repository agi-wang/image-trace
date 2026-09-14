#!/usr/bin/env bash
# Multi-node semantic persist smoke (Phase 8 R2): with ITRACE_SEMANTIC=1
# (forced stub — no model weights), ITRACE_MIH_NODES>1, and a writable
# ITRACE_MIH_INDEX_DIR, dedup the same project on the SAME index dir —
# the first multi-node scan builds project_{id}_sem_mn/ (ITSEMN1 top
# meta + one ITSEMP1+ITSEMH1 node_{i}/ bundle per contiguous sorted-id
# range), the second must hit BOTH halves ("+N sem (cached)" AND
# sem_hnsw_loaded=true — vectors AND node graphs restored). Single-node
# armed scans on the same index dir use project_{id}_sem/ (ITSEMP1) and
# must not corrupt or read the _sem_mn sibling. Confirmed duplicate
# groups must be identical across all scans (semantic is candidates-
# only; hash/crop verification still decides merges). Re-run with a
# distinct RUN_TAG for double-regression evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/semantic_mn_persist_$RUN_TAG"
IDX="$BASE/idx"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/data" "$IDX"
# All other channel knobs cleared; semantic inputs are flag + stub +
# index dir, sharding input is ITRACE_MIH_NODES only.
unset ITRACE_SEMANTIC ITRACE_SEMANTIC_STUB ITRACE_SEMANTIC_MODEL
unset ITRACE_SEMANTIC_K ITRACE_SEMANTIC_MIN_COS ITRACE_MIH_NODES

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# A nudged near-dup of sem_synth_1__orig (same trick as the single-node
# stub smoke): guarantees one high-cosine semantic candidate pair.
NUDGED="$BASE/sem_synth_1__nudged.png"
python3 - "$ROOT/datasets/built/sem/sem_synth_1__orig.png" "$NUDGED" <<'PY'
import sys
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGB")
px = img.load()
for y in range(0, img.height, 5):
    for x in range(0, img.width, 5):
        r, g, b = px[x, y]
        px[x, y] = ((r + 3) & 0xFF, (g - 1) & 0xFF, b)
img.save(sys.argv[2])
PY

FILES=(
  datasets/built/sem/sem_synth_1__orig.png
  datasets/built/sem/sem_synth_1__rot90.png
  datasets/built/sem/sem_synth_1__hflip.png
  datasets/built/sem/sem_synth_1__crop70.png
  datasets/built/sem/sem_synth_1__slice_r0c0.png
  datasets/built/fluorescence/fluor_synth_1__orig.png
  datasets/built/fluorescence/fluor_synth_1__rot90.png
  datasets/built/fluorescence/fluor_synth_1__crop70.png
  "$NUDGED"
)
for f in "${FILES[@]}"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done

PID=$("$BIN" --data-dir "$BASE/data" create "semantic-mn-persist-$RUN_TAG" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
"$BIN" --data-dir "$BASE/data" add "$PID" "${FILES[@]}" >/dev/null

# Five armed scans on the same project + index dir:
#   sn1 — nodes unset: builds project_{id}_sem/ (ITSEMP1+ITSEMH1)
#   mn1 — NODES=3:     builds project_{id}_sem_mn/ (ITSEMN1 + node_{i}/)
#   mn2 — NODES=3:     cache hit on _sem_mn (vecs + node graphs)
#   sn2 — nodes unset: cache hit on project_{id}_sem/, _sem_mn intact
#   mn3 — NODES=3:     _sem_mn still cache-hits after the sn scans
env ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/sn1.txt"
env ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn1.txt"
env ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn2.txt"
env ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/sn2.txt"
env ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn3.txt"

MN_DIR="$IDX/project_${PID}_sem_mn"
SN_DIR="$IDX/project_${PID}_sem"

# (a) Every armed scan reports semantic candidates (+N sem, N >= 1).
for tag in sn1 mn1 mn2 sn2 mn3; do
  grep -oE '\+[0-9]+ sem' "$BASE/$tag.txt" | grep -qE '[1-9][0-9]*' \
    || { echo "$tag: no semantic candidates (want +N sem, N>=1)" >&2; exit 1; }
done

# (b) Build vs cache-hit signals.
grep -q 'sem (cached)' "$BASE/sn1.txt" \
  && { echo "sn1 unexpectedly hit the cache" >&2; exit 1; }
grep -q 'sem_hnsw_loaded=false' "$BASE/sn1.txt" \
  || { echo "sn1 must report sem_hnsw_loaded=false (graph built)" >&2; exit 1; }
grep -q 'sem (cached)' "$BASE/mn1.txt" \
  && { echo "mn1 unexpectedly hit the cache" >&2; exit 1; }
grep -q 'sem_hnsw_loaded=false' "$BASE/mn1.txt" \
  || { echo "mn1 must report sem_hnsw_loaded=false (node graphs built)" >&2; exit 1; }
for tag in mn2 sn2 mn3; do
  grep -qE '\+[0-9]+ sem \(cached\)' "$BASE/$tag.txt" \
    || { echo "$tag: no vector cache hit (want '+N sem (cached)')" >&2; exit 1; }
  grep -q 'sem_hnsw_loaded=true' "$BASE/$tag.txt" \
    || { echo "$tag: node graphs not restored (want sem_hnsw_loaded=true)" >&2; exit 1; }
done

# (c) On-disk layout: _sem_mn is a real ITSEMN1 bundle — top meta +
# node_{i}/ dirs each holding a complete ITSEMP1 bundle + ITSEMH1
# hnsw.bin; project_{id}_sem/ stays the single-node format.
[[ -f "$MN_DIR/meta.json" ]] \
  || { echo "missing $MN_DIR/meta.json" >&2; exit 1; }
for i in 0 1 2; do
  for f in meta.json image_ids.bin vectors.bin hnsw.bin; do
    [[ -f "$MN_DIR/node_$i/$f" ]] \
      || { echo "missing $MN_DIR/node_$i/$f" >&2; exit 1; }
  done
done
[[ -f "$SN_DIR/meta.json" && -f "$SN_DIR/hnsw.bin" ]] \
  || { echo "missing single-node $SN_DIR bundle" >&2; exit 1; }
python3 - "$MN_DIR/meta.json" "$MN_DIR/node_0/meta.json" \
          "$MN_DIR/node_1/meta.json" "$MN_DIR/node_2/meta.json" \
          "$MN_DIR/node_0/hnsw.bin" "$SN_DIR/meta.json" <<'PY'
import json, sys
top = json.load(open(sys.argv[1]))
nodes = [json.load(open(p)) for p in sys.argv[2:5]]
hnsw = open(sys.argv[5], "rb").read(7)
single = json.load(open(sys.argv[6]))
assert top["magic"] == "ITSEMN1" and top["version"] == 1, top
assert top["node_count"] == 3 and top["image_count"] == 9, top
assert top["dim"] > 0 and "embedder" in top, top
assert "feature_fingerprint" in top, top
ranges = top["ranges"]
assert len(ranges) == 3, ranges
# Contiguous, non-empty, covering 9 sorted ids.
total = sum(r["count"] for r in ranges)
assert total == 9 and all(r["count"] >= 1 for r in ranges), ranges
for a, b in zip(ranges, ranges[1:]):
    assert b["start"] == a["end"] + 1, ranges
for n in nodes:
    assert n["magic"] == "ITSEMP1" and n["version"] == 1, n
assert hnsw == b"ITSEMH1", hnsw
assert single["magic"] == "ITSEMP1", single
PY

# (d) Group count + membership identical across all five scans —
# sharded semantic recall neither loses nor merges anything extra.
count_groups() { grep -c "^duplicate group" "$1"; }
g_sn1=$(count_groups "$BASE/sn1.txt")
echo -n "groups sn1=$g_sn1"
for tag in mn1 mn2 sn2 mn3; do
  g=$(count_groups "$BASE/$tag.txt")
  echo -n " $tag=$g"
  [[ "$g" -eq "$g_sn1" ]] \
    || { echo; echo "$tag group count differs from sn1" >&2; exit 1; }
done
echo
for tag in mn1 mn2 sn2 mn3; do
  diff <(grep "^  - " "$BASE/sn1.txt" | sort) \
       <(grep "^  - " "$BASE/$tag.txt" | sort) >/dev/null \
    || { echo "$tag membership differs from sn1" >&2; exit 1; }
done

echo "SEMANTIC_MN_PERSIST_SMOKE_OK"
