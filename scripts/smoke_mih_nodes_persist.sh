#!/usr/bin/env bash
# Multi-node MIH persist smoke (Phase 7 R2): with ITRACE_MIH_NODES=4 AND
# a writable ITRACE_MIH_INDEX_DIR, dedup the same project twice on the
# SAME index dir — run 1 builds the project_{id}_mn/ bundle (ITMIHN1
# top meta + indexes/{a}/ cluster dirs), run 2 must report
# index_loaded=true (cluster shards restored, not rebuilt). Single-node
# runs on the same index dir use project_{id}/ (ITMIHP1) and must not
# corrupt or read the _mn sibling; group membership must match the
# single-node baseline throughout. Re-run with a distinct RUN_TAG for
# double-regression evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/mih_nodes_persist_$RUN_TAG"
IDX="$BASE/idx"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/data" "$IDX"
unset ITRACE_MIH_NODES
unset ITRACE_SEMANTIC ITRACE_SEMANTIC_STUB ITRACE_SEMANTIC_MODEL
unset ITRACE_SEMANTIC_K ITRACE_SEMANTIC_MIN_COS

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# Two families with known gate near-dups (orig/rot90/hflip/crop70) plus
# a slice tile to exercise the crop channel.
FILES=(
  datasets/built/sem/sem_synth_1__orig.png
  datasets/built/sem/sem_synth_1__rot90.png
  datasets/built/sem/sem_synth_1__hflip.png
  datasets/built/sem/sem_synth_1__crop70.png
  datasets/built/sem/sem_synth_1__slice_r0c0.png
  datasets/built/fluorescence/fluor_synth_1__orig.png
  datasets/built/fluorescence/fluor_synth_1__rot90.png
  datasets/built/fluorescence/fluor_synth_1__crop70.png
)
for f in "${FILES[@]}"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done

PID=$("$BIN" --data-dir "$BASE/data" create "mih-nodes-persist-$RUN_TAG" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
"$BIN" --data-dir "$BASE/data" add "$PID" "${FILES[@]}" >/dev/null

# Four scans on the same project + index dir:
#   mono1 — nodes unset: builds project_{id}/ (ITMIHP1), index_loaded=false
#   mn1   — NODES=4:     builds project_{id}_mn/ (ITMIHN1), index_loaded=false
#   mn2   — NODES=4:     cache hit on _mn, index_loaded=true
#   mono2 — nodes unset: cache hit on project_{id}/, _mn left intact
ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mono1.txt"
ITRACE_MIH_INDEX_DIR="$IDX" ITRACE_MIH_NODES=4 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn1.txt"
ITRACE_MIH_INDEX_DIR="$IDX" ITRACE_MIH_NODES=4 \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mn2.txt"
ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/mono2.txt"

MN_DIR="$IDX/project_${PID}_mn"
SN_DIR="$IDX/project_${PID}"

# (a) Build vs cache-hit signals.
grep -q 'index_loaded=false' "$BASE/mono1.txt" \
  || { echo "mono1: expected index_loaded=false (fresh ITMIHP1 build)" >&2; exit 1; }
grep -q 'index_loaded=false' "$BASE/mn1.txt" \
  || { echo "mn1: expected index_loaded=false (fresh ITMIHN1 build)" >&2; exit 1; }
grep -q 'index_loaded=true' "$BASE/mn2.txt" \
  || { echo "mn2: expected index_loaded=true (_mn cache hit)" >&2; exit 1; }
grep -q 'index_loaded=true' "$BASE/mono2.txt" \
  || { echo "mono2: expected index_loaded=true (ITMIHP1 cache hit)" >&2; exit 1; }

# (b) On-disk layout: _mn is a real ITMIHN1 cluster bundle, project_{id}/
# stays ITMIHP1, and flipping back to single-node left _mn intact.
[[ -f "$MN_DIR/meta.json" && -f "$MN_DIR/image_ids.bin" ]] \
  || { echo "missing $MN_DIR bundle files" >&2; exit 1; }
[[ -d "$MN_DIR/indexes/0/node_0" && -d "$MN_DIR/indexes/0/node_1" ]] \
  || { echo "missing _mn cluster node dirs" >&2; exit 1; }
[[ -f "$SN_DIR/meta.json" ]] \
  || { echo "missing single-node $SN_DIR/meta.json" >&2; exit 1; }
python3 - "$MN_DIR/meta.json" "$MN_DIR/indexes/0/meta.json" \
          "$SN_DIR/meta.json" <<'PY'
import json, sys
top = json.load(open(sys.argv[1]))
cluster = json.load(open(sys.argv[2]))
single = json.load(open(sys.argv[3]))
assert top["magic"] == "ITMIHN1" and top["version"] == 1, top
assert top["node_count"] == 4, top
assert top["gate_algo_count"] >= 1 and top["image_count"] == 8, top
assert "feature_fingerprint" in top, top
assert cluster["magic"] == "ITMIHN1" and len(cluster["ranges"]) == 4, cluster
assert single["magic"] == "ITMIHP1", single
PY

# (c) Group count + membership identical across all four scans — the
# multi-node persist path neither loses nor merges anything extra.
count_groups() { grep -c "^duplicate group" "$1"; }
g_mono1=$(count_groups "$BASE/mono1.txt")
g_mn1=$(count_groups "$BASE/mn1.txt")
g_mn2=$(count_groups "$BASE/mn2.txt")
g_mono2=$(count_groups "$BASE/mono2.txt")
echo "groups mono1=$g_mono1 mn1=$g_mn1 mn2=$g_mn2 mono2=$g_mono2"
[[ "$g_mono1" -eq 2 ]] || { echo "expected 2 baseline groups" >&2; exit 1; }
[[ "$g_mn1" -eq "$g_mono1" && "$g_mn2" -eq "$g_mono1" && "$g_mono2" -eq "$g_mono1" ]] \
  || { echo "group count mismatch across scans" >&2; exit 1; }
for tag in mn1 mn2 mono2; do
  diff <(grep "^  - " "$BASE/mono1.txt" | sort) \
       <(grep "^  - " "$BASE/$tag.txt" | sort) >/dev/null \
    || { echo "$tag membership differs from mono baseline" >&2; exit 1; }
done

echo "MIH_NODES_PERSIST_SMOKE_OK"
