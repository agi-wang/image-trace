#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

REPORT_DIR="$ROOT/datasets/built/reports"
DATA_DIR="$REPORT_DIR/data"
BIN="$ROOT/target/release/itrace-cli"
mkdir -p "$REPORT_DIR"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# Pick a compact but meaningful subset: one source family per modality with transforms
mapfile -t FILES < <(
  python3 - <<'PY'
from pathlib import Path
root = Path('datasets/built')
# Prefer synth_1 and one public sample per modality, orig+rot90+hflip+crop70
want_suffixes = ('__orig.png','__rot90.png','__hflip.png','__crop70.png')
picks = []
for modality, stems in [
    ('sem', ['sem_synth_1', 'sem_pollen_wiki']),
    ('fluorescence', ['fluor_synth_1', 'fluor_cells']),
    ('histology', ['histo_synth_1', 'skimage_ihc']),
]:
    for stem in stems:
        for suf in want_suffixes:
            p = root / modality / f'{stem}{suf}'
            if p.exists():
                picks.append(str(p))
print('\n'.join(picks))
PY
)

if [[ ${#FILES[@]} -lt 8 ]]; then
  echo "not enough built images; run: python3 scripts/build_dataset.py" >&2
  exit 1
fi

echo "using ${#FILES[@]} images"
CREATE_OUT=$("$BIN" --data-dir "$DATA_DIR" create "smoke-microscopy")
echo "$CREATE_OUT" | tee "$REPORT_DIR/create.json"
PID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$CREATE_OUT")
echo "project_id=$PID"

"$BIN" --data-dir "$DATA_DIR" add "$PID" "${FILES[@]}"
"$BIN" --data-dir "$DATA_DIR" images "$PID" | tee "$REPORT_DIR/images.json" >/dev/null

"$BIN" --data-dir "$DATA_DIR" compare "$PID" --algorithm phash --threshold 0.80 --rotation-invariant \
  | tee "$REPORT_DIR/compare_phash.json"
"$BIN" --data-dir "$DATA_DIR" smart "$PID" --threshold 0.88 --min-agree 2 \
  | tee "$REPORT_DIR/smart.json"
"$BIN" --data-dir "$DATA_DIR" report "$PID" --algorithm phash --threshold 0.80 \
  | tee "$REPORT_DIR/report_phash.json"

python3 - <<'PY'
from pathlib import Path
import re
rep = Path('datasets/built/reports')
print('--- smoke summary ---')
for name in ['compare_phash.json', 'smart.json', 'report_phash.json']:
    p = rep/name
    if not p.exists():
        print(name, 'MISSING')
        continue
    text = p.read_text(errors='replace')
    groups = len(re.findall(r'^(?:duplicate )?group \d+:', text, re.M))
    print(f'{name}: bytes={p.stat().st_size} group_headers={groups}')
    for ln in text.splitlines()[:8]:
        print(' ', ln)
print('reports_dir', rep.resolve())
PY

echo "SMOKE_OK"
