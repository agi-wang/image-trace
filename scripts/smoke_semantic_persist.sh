#!/usr/bin/env bash
# Semantic-persist smoke (Phase 5 R2+): with ITRACE_SEMANTIC=1 (forced
# stub — no model weights) and ITRACE_MIH_INDEX_DIR set, dedup the same
# project twice on the SAME index dir — run 1 builds the
# project_{id}_sem/ bundle, run 2 must hit it ("+N sem (cached)") —
# while confirmed duplicate groups match the flag-off baseline. The stub
# is NOT production DINOv2; this exercises the persist path + wiring
# only. Re-run with a distinct RUN_TAG for double-regression evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/semantic_persist_$RUN_TAG"
IDX="$BASE/idx"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/data" "$IDX"
# All other channel knobs cleared; the only semantic inputs are the flag
# + stub + index dir.
unset ITRACE_SEMANTIC ITRACE_SEMANTIC_STUB ITRACE_SEMANTIC_MODEL
unset ITRACE_SEMANTIC_K ITRACE_SEMANTIC_MIN_COS

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# A nudged near-dup of sem_synth_1__orig (same trick as the stub smoke):
# gives the stub channel a guaranteed high-cosine pair to prove emission.
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

PID=$("$BIN" --data-dir "$BASE/data" create "semantic-persist-$RUN_TAG" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
"$BIN" --data-dir "$BASE/data" add "$PID" "${FILES[@]}" >/dev/null

# Three scans on the same project: flag-off baseline (no index dir — the
# pure default path), then armed twice against the same index dir.
"$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/off.txt"
ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/on1.txt"
ITRACE_SEMANTIC=1 ITRACE_SEMANTIC_STUB=1 ITRACE_MIH_INDEX_DIR="$IDX" \
  "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/on2.txt"

# (a) Both armed runs must report semantic candidates: "+N sem", N >= 1.
sem1=$(grep -oE '\+[0-9]+ sem' "$BASE/on1.txt" | grep -oE '[0-9]+' || true)
sem2=$(grep -oE '\+[0-9]+ sem' "$BASE/on2.txt" | grep -oE '[0-9]+' || true)
[[ -n "$sem1" && "$sem1" -ge 1 ]] \
  || { echo "run1: no semantic candidates (want +N sem, N>=1)" >&2; exit 1; }
[[ -n "$sem2" && "$sem2" -ge 1 ]] \
  || { echo "run2: no semantic candidates (want +N sem, N>=1)" >&2; exit 1; }
# (b) run1 builds the bundle, run2 must hit it: "(cached)" only on run2,
# and the project_{id}_sem/ bundle must exist on disk.
grep -q 'sem (cached)' "$BASE/on1.txt" \
  && { echo "run1 unexpectedly hit the cache" >&2; exit 1; }
grep -qE '\+[0-9]+ sem \(cached\)' "$BASE/on2.txt" \
  || { echo "run2 did not hit the semantic bundle (want '+N sem (cached)')" >&2; exit 1; }
for f in meta.json image_ids.bin vectors.bin; do
  [[ -f "$IDX/project_${PID}_sem/$f" ]] \
    || { echo "missing bundle file project_${PID}_sem/$f" >&2; exit 1; }
done
# (c) Flag-off run must not show the sem note.
grep -qE '\+[0-9]+ sem' "$BASE/off.txt" \
  && { echo "flag-off run unexpectedly shows sem note" >&2; exit 1; }
# (d) Confirmed groups must not shrink under the armed channel, cached
# or not — membership identical to the flag-off baseline.
off_groups=$(grep -c "^duplicate group" "$BASE/off.txt")
on1_groups=$(grep -c "^duplicate group" "$BASE/on1.txt")
on2_groups=$(grep -c "^duplicate group" "$BASE/on2.txt")
echo "off=$off_groups on1=$on1_groups on2=$on2_groups sem1=$sem1 sem2=$sem2"
[[ "$on1_groups" -ge "$off_groups" && "$on2_groups" -ge "$off_groups" ]] \
  || { echo "groups shrank" >&2; exit 1; }
diff <(grep "^  - " "$BASE/off.txt" | sort) \
     <(grep "^  - " "$BASE/on1.txt" | sort) >/dev/null \
  || { echo "run1 membership differs from flag-off" >&2; exit 1; }
diff <(grep "^  - " "$BASE/off.txt" | sort) \
     <(grep "^  - " "$BASE/on2.txt" | sort) >/dev/null \
  || { echo "run2 membership differs from flag-off" >&2; exit 1; }

echo "SEMANTIC_PERSIST_SMOKE_OK"
