#!/usr/bin/env bash
# Semantic-channel smoke (Phase 4 R2+): dedup the same project twice —
# flag-off baseline vs ITRACE_SEMANTIC=1 (embedder_from_env auto-stub) —
# and assert the armed run emits semantic candidates while confirmed
# duplicate groups do not shrink. The stub is NOT production DINOv2; this
# exercises the wiring only. Re-run with a distinct RUN_TAG for the
# double-regression evidence; each run gets a fresh data dir.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/semantic_stub_$RUN_TAG"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/data"
# Channel knobs cleared so the only semantic input is the flag itself.
unset ITRACE_MIH_INDEX_DIR ITRACE_SEMANTIC ITRACE_SEMANTIC_STUB ITRACE_SEMANTIC_MODEL
unset ITRACE_SEMANTIC_K ITRACE_SEMANTIC_MIN_COS

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# A nudged near-dup of sem_synth_1__orig (tiny pixel noise): too small to
# move the gate hashes out of MIH radius, but it gives the stub channel a
# guaranteed high-cosine pair to prove emission.
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

PID=$("$BIN" --data-dir "$BASE/data" create "semantic-stub-$RUN_TAG" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
"$BIN" --data-dir "$BASE/data" add "$PID" "${FILES[@]}" >/dev/null

# Same project, two scans: flag off first, then armed (auto-stub).
"$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/off.txt"
ITRACE_SEMANTIC=1 "$BIN" --data-dir "$BASE/data" dedup "$PID" | tee "$BASE/on.txt"

# (a) Armed run must report semantic candidates: "+N sem", N >= 1.
sem_n=$(grep -oE '\+[0-9]+ sem' "$BASE/on.txt" | grep -oE '[0-9]+' || true)
[[ -n "$sem_n" && "$sem_n" -ge 1 ]] \
  || { echo "no semantic candidates emitted (want +N sem, N>=1)" >&2; exit 1; }
grep -qE '\+[0-9]+ sem' "$BASE/off.txt" \
  && { echo "flag-off run unexpectedly shows sem note" >&2; exit 1; }

# (b) Confirmed groups must not shrink under the armed channel.
off_groups=$(grep -c "^duplicate group" "$BASE/off.txt")
on_groups=$(grep -c "^duplicate group" "$BASE/on.txt")
echo "flag-off groups=$off_groups  flag-on groups=$on_groups  sem=$sem_n"
[[ "$on_groups" -ge "$off_groups" ]] || { echo "groups shrank" >&2; exit 1; }
# Every flag-off member line must still appear flag-on (no shrink).
diff <(grep "^  - " "$BASE/off.txt" | sort) \
     <(grep "^  - " "$BASE/on.txt" | sort) >/dev/null \
  || { echo "flag-on membership differs from flag-off" >&2; exit 1; }

echo "SEMANTIC_STUB_SMOKE_OK"
