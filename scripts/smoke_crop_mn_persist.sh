#!/usr/bin/env bash
# Multi-node crop/slice persist smoke (Phase 9 R2): with
# ITRACE_MIH_NODES>1 and a writable ITRACE_MIH_INDEX_DIR, dedup the same
# project on the SAME index dir — the first multi-node scan builds
# project_{id}_crop_mn/ (ITMIHCN1 top meta + contiguous sorted-id ranges
# + one complete ITMIHC1 node_{i}/ bundle per range), the second must
# hit cache (crop_index_loaded=true — node shard indexes restored, not
# rebuilt). Single-node scans on the same index dir use
# project_{id}_crop/ (ITMIHC1) and must not corrupt or read the
# _crop_mn sibling. Confirmed duplicate groups must be identical across
# all scans (crop candidates are exact-parity with single-node; NCC
# containment still decides merges). Re-run with a distinct RUN_TAG for
# double-regression evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/crop_mn_persist_$RUN_TAG"
IDX="$BASE/idx"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/data" "$IDX"
# Channel knobs cleared; sharding input is ITRACE_MIH_NODES only.
unset ITRACE_MIH_NODES
unset ITRACE_SEMANTIC ITRACE_SEMANTIC_STUB ITRACE_SEMANTIC_MODEL
unset ITRACE_SEMANTIC_K ITRACE_SEMANTIC_MIN_COS

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# Two families with crop70 + slice variants so the crop channel emits
# real "+N crop" candidates, plus one unrelated orig.
FILES=(
  datasets/built/sem/sem_synth_1__orig.png
  datasets/built/sem/sem_synth_1__rot90.png
  datasets/built/sem/sem_synth_1__crop70.png
  datasets/built/sem/sem_synth_1__slice_r0c0.png
  datasets/built/fluorescence/fluor_synth_1__orig.png
  datasets/built/fluorescence/fluor_synth_1__rot90.png
  datasets/built/fluorescence/fluor_synth_1__crop70.png
  datasets/built/histology/histo_synth_1__orig.png
)
for f in "${FILES[@]}"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done

PID=$("$BIN" --data-dir "$BASE/data" create "crop-mn-persist-$RUN_TAG" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
"$BIN" --data-dir "$BASE/data" add "$PID" "${FILES[@]}" >/dev/null

# Five scans on the same project + index dir:
#   sn1 — nodes unset: builds project_{id}_crop/ (ITMIHC1)
#   mn1 — NODES=3:     builds project_{id}_crop_mn/ (ITMIHCN1 + node_{i}/)
#   mn2 — NODES=3:     cache hit on _crop_mn (crop_index_loaded=true)
#   sn2 — nodes unset: cache hit on project_{id}_crop/, _crop_mn intact
#   mn3 — NODES=3:     _crop_mn still cache-hits after the sn scans
env ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/sn1.txt"
env ITRACE_MIH_INDEX_DIR="$IDX" ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn1.txt"
env ITRACE_MIH_INDEX_DIR="$IDX" ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn2.txt"
env ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/sn2.txt"
env ITRACE_MIH_INDEX_DIR="$IDX" ITRACE_MIH_NODES=3 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn3.txt"

MN_DIR="$IDX/project_${PID}_crop_mn"
SN_DIR="$IDX/project_${PID}_crop"

# (a) Every scan emits crop candidates (+N crop, N >= 1) and reports the
# crop_index_loaded field.
for tag in sn1 mn1 mn2 sn2 mn3; do
  grep -oE '\+[0-9]+ crop' "$BASE/$tag.txt" | grep -qE '[1-9][0-9]*' \
    || { echo "$tag: no crop candidates (want +N crop, N>=1)" >&2; exit 1; }
  grep -q 'crop_index_loaded=' "$BASE/$tag.txt" \
    || { echo "$tag: missing crop_index_loaded field" >&2; exit 1; }
done

# (b) Build vs cache-hit signals on the crop channel.
grep -q 'crop_index_loaded=false' "$BASE/sn1.txt" \
  || { echo "sn1 must report crop_index_loaded=false (ITMIHC1 built)" >&2; exit 1; }
grep -q 'crop_index_loaded=false' "$BASE/mn1.txt" \
  || { echo "mn1 must report crop_index_loaded=false (ITMIHCN1 built)" >&2; exit 1; }
for tag in mn2 sn2 mn3; do
  grep -q 'crop_index_loaded=true' "$BASE/$tag.txt" \
    || { echo "$tag: crop index not restored (want crop_index_loaded=true)" >&2; exit 1; }
done

# (c) On-disk layout: _crop_mn is a real ITMIHCN1 bundle — top meta +
# contiguous ranges + node_{i}/ dirs each holding a complete ITMIHC1
# bundle; project_{id}_crop/ stays the single-node format.
[[ -f "$MN_DIR/meta.json" ]] \
  || { echo "missing $MN_DIR/meta.json" >&2; exit 1; }
for i in 0 1 2; do
  for f in meta.json image_ids.bin; do
    [[ -f "$MN_DIR/node_$i/$f" ]] \
      || { echo "missing $MN_DIR/node_$i/$f" >&2; exit 1; }
  done
  [[ -d "$MN_DIR/node_$i/index" ]] \
    || { echo "missing $MN_DIR/node_$i/index/" >&2; exit 1; }
done
[[ -f "$SN_DIR/meta.json" && -d "$SN_DIR/index" ]] \
  || { echo "missing single-node $SN_DIR bundle" >&2; exit 1; }
python3 - "$MN_DIR/meta.json" "$MN_DIR/node_0/meta.json" \
          "$MN_DIR/node_1/meta.json" "$MN_DIR/node_2/meta.json" \
          "$SN_DIR/meta.json" <<'PY'
import json, sys
top = json.load(open(sys.argv[1]))
nodes = [json.load(open(p)) for p in sys.argv[2:5]]
single = json.load(open(sys.argv[5]))
assert top["magic"] == "ITMIHCN1" and top["version"] == 1, top
assert top["node_count"] == 3 and top["image_count"] == 8, top
assert top["shard_bits"] > 0 and top["key_count"] > 0, top
assert "feature_fingerprint" in top, top
ranges = top["ranges"]
assert len(ranges) == 3, ranges
# Contiguous, non-empty, covering 8 sorted ids.
total = sum(r["count"] for r in ranges)
assert total == 8 and all(r["count"] >= 1 for r in ranges), ranges
for a, b in zip(ranges, ranges[1:]):
    assert b["start"] == a["end"] + 1, ranges
for n in nodes:
    assert n["magic"] == "ITMIHC1" and n["version"] == 1, n
assert single["magic"] == "ITMIHC1" and "ranges" not in single, single
PY

# (d) Group count + membership identical across all five scans — the
# multi-node crop channel is exact-parity with single-node.
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

echo "CROP_MN_PERSIST_SMOKE_OK"
