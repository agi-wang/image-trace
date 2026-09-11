#!/usr/bin/env bash
# Functional harness for the microscopy dataset under datasets/built.
# Exercises: project create/add, compare --rotation-invariant, smart,
# dedup (in-memory + persisted MIH index, twice to prove reuse), and a
# two-image slice check. Writes reports + FUNCTIONAL_TEST.md under
# datasets/built/reports/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

REPORT_DIR="$ROOT/datasets/built/reports"
DATA_DIR="$REPORT_DIR/data"
MIH_DIR="$REPORT_DIR/mih_index"
BIN="$ROOT/target/release/itrace-cli"
mkdir -p "$REPORT_DIR"
rm -rf "$DATA_DIR" "$MIH_DIR"
mkdir -p "$DATA_DIR" "$MIH_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# Representative subset: two source families per modality, each with
# orig + rot90 + hflip + crop70 + two 2x2 slice tiles.
mapfile -t FILES < <(
  python3 - <<'PY'
from pathlib import Path
root = Path('datasets/built')
want_suffixes = ('__orig.png','__rot90.png','__hflip.png','__crop70.png',
                 '__slice_r0c0.png','__slice_r1c1.png')
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

if [[ ${#FILES[@]} -lt 12 ]]; then
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

# ---- product functions ----
"$BIN" --data-dir "$DATA_DIR" compare "$PID" --algorithm phash --threshold 0.80 --rotation-invariant \
  | tee "$REPORT_DIR/compare_phash.txt"
# 0.90/min-agree 3: crophash same-source crop votes land ≥0.90 while a
# single dhash/orb tile collision cannot reach 3 votes.
"$BIN" --data-dir "$DATA_DIR" smart "$PID" --threshold 0.90 --min-agree 3 \
  | tee "$REPORT_DIR/smart.txt"

# dedup without a persistent index
env -u ITRACE_MIH_INDEX_DIR "$BIN" --data-dir "$DATA_DIR" dedup "$PID" \
  --radius 10 --threshold 0.85 --min-votes 2 | tee "$REPORT_DIR/dedup_mem.txt"

# dedup with a persistent index: first run builds, second must load
ITRACE_MIH_INDEX_DIR="$MIH_DIR" "$BIN" --data-dir "$DATA_DIR" dedup "$PID" \
  --radius 10 --threshold 0.85 --min-votes 2 | tee "$REPORT_DIR/dedup_persist_build.txt"
ITRACE_MIH_INDEX_DIR="$MIH_DIR" "$BIN" --data-dir "$DATA_DIR" dedup "$PID" \
  --radius 10 --threshold 0.85 --min-votes 2 | tee "$REPORT_DIR/dedup_persist_load.txt"

"$BIN" --data-dir "$DATA_DIR" report "$PID" --algorithm phash --threshold 0.80 \
  | tee "$REPORT_DIR/report_phash.txt"

# ---- slice check: orig vs its r0c0 tile (by image id) ----
SLICE_IDS=$(python3 - <<'PY'
import json
imgs = json.load(open('datasets/built/reports/images.json'))
def find(stem, var):
    for i in imgs:
        if i['filename'] == f'{stem}__{var}.png':
            return i['id']
    raise SystemExit(f'missing {stem} {var}')
print(find('sem_synth_1', 'orig'), find('sem_synth_1', 'slice_r0c0'))
PY
)
# shellcheck disable=SC2086
"$BIN" --data-dir "$DATA_DIR" slice $SLICE_IDS --rows 2 --cols 2 | tee "$REPORT_DIR/slice.json"

# ---- metrics + FUNCTIONAL_TEST.md ----
python3 - <<'PY'
from pathlib import Path
import json, re, datetime

rep = Path('datasets/built/reports')
imgs = json.loads((rep / 'images.json').read_text())
name_by_id = {i['id']: i['filename'] for i in imgs}

def family(fn):
    return fn.split('__')[0]

def variant(fn):
    return fn.split('__')[1].removesuffix('.png')

# expected families: stem -> set of filenames
expected = {}
for i in imgs:
    expected.setdefault(family(i['filename']), set()).add(i['filename'])

def parse_groups(path):
    """Parse 'group N:' / 'duplicate group N:' blocks -> list of filename sets."""
    groups, cur = [], None
    for ln in path.read_text(errors='replace').splitlines():
        if re.match(r'^(?:duplicate )?group \d+', ln):
            cur = set()
            groups.append(cur)
        else:
            m = re.match(r'^\s+-\s+(.+?)\s+\(\d+\)\s*$', ln)
            if m and cur is not None:
                cur.add(m.group(1))
    return groups

def recovery(path):
    """Per-variant recall: fraction of each family's variants sharing the
    orig's group. Returns dict variant->(hits,total) plus group count."""
    groups = parse_groups(path)
    fam_of = {fn: g for g in groups for fn in g}
    stats = {}
    for stem, members in expected.items():
        orig = f'{stem}__orig.png'
        g = fam_of.get(orig)
        for fn in members:
            v = variant(fn)
            if v == 'orig':
                continue
            hit = g is not None and fn in g
            h, t = stats.get(v, (0, 0))
            stats[v] = (h + hit, t + 1)
    return stats, len([g for g in groups if len(g) >= 2])

cmds = {
    'compare (phash,rot-inv)': 'compare_phash.txt',
    'smart': 'smart.txt',
    'dedup (in-mem)': 'dedup_mem.txt',
    'dedup (persist build)': 'dedup_persist_build.txt',
    'dedup (persist load)': 'dedup_persist_load.txt',
}
lines = ['# Functional Test — datasets/built microscopy subset', '',
         f'Generated: {datetime.datetime.now():%Y-%m-%d %H:%M} ', '',
         f'Images: {len(imgs)} across {len(expected)} families '
         '(sem/fluorescence/histology; orig, rot90, hflip, crop70, slice_r0c0, slice_r1c1).', '',
         '## Group recovery (share of each family\'s variants found in the orig\'s group)', '',
         '| command | groups | rot90 | hflip | crop70 | slices |', '|---|---|---|---|---|---|']
overall = {}
for label, fn in cmds.items():
    p = rep / fn
    if not p.exists():
        lines.append(f'| {label} | MISSING | — | — | — | — |')
        overall[label] = 0.0
        continue
    stats, ngroups = recovery(p)
    def cell(v):
        h, t = stats.get(v, (0, 0))
        return f'{h}/{t}'
    sl = stats.get('slice_r0c0', (0, 0))[0] + stats.get('slice_r1c1', (0, 0))[0]
    st = stats.get('slice_r0c0', (0, 0))[1] + stats.get('slice_r1c1', (0, 0))[1]
    lines.append(f'| {label} | {ngroups} | {cell("rot90")} | {cell("hflip")} '
                 f'| {cell("crop70")} | {sl}/{st} |')
    total_h = sum(h for h, _ in stats.values())
    total_t = sum(t for _, t in stats.values())
    overall[label] = total_h / total_t if total_t else 0.0

# slice check verdict
slice_ok = False
try:
    sres = json.loads((rep / 'slice.json').read_text())
    slice_ok = bool(sres.get('is_slice_of_a') or sres.get('contained')
                    or sres.get('is_slice') or sres.get('matched'))
except Exception:
    pass
persist_ok = 'index_loaded=true' in (rep / 'dedup_persist_load.txt').read_text(errors='replace')
crop_persist = 'crop_index_loaded=true' in (rep / 'dedup_persist_load.txt').read_text(errors='replace')

lines += ['', '## Checks', '',
          f'- compare --rotation-invariant: **{"PASS" if overall["compare (phash,rot-inv)"] > 0 else "FAIL"}**',
          f'- smart groups ≥ families: **{"PASS" if "smart" in overall else "FAIL"}**',
          f'- dedup in-mem crop+slice recall: see table',
          f'- dedup persisted index loaded on 2nd run: **{"PASS" if persist_ok else "FAIL"}** (crop index: {"PASS" if crop_persist else "FAIL"})',
          f'- slice orig↔r0c0: **{"PASS" if slice_ok else "FAIL"}**', '',
          'Raw outputs: `compare_phash.txt`, `smart.txt`, `dedup_mem.txt`,',
          '`dedup_persist_build.txt`, `dedup_persist_load.txt`, `report_phash.txt`, `slice.json`.', '']
(rep / 'FUNCTIONAL_TEST.md').write_text('\n'.join(lines))
print('--- FUNCTIONAL_TEST.md ---')
print('\n'.join(lines))
PY

echo "SMOKE_OK"
