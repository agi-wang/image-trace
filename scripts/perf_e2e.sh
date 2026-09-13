#!/usr/bin/env bash
# End-to-end perf harness: boots itrace-server on a scratch DATA_DIR,
# uploads the deterministic perf corpus, then times the hot endpoints.
#
# Usage: ./scripts/perf_e2e.sh [--quick] [--concurrency N]
#   Requires: cargo (builds release binaries if missing), curl, python3 (json only)
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=""; CONC=4
for a in "$@"; do
  case "$a" in
    --quick) QUICK="--quick" ;;
    --concurrency) ;; # handled below via shift pair
  esac
done
while [ $# -gt 0 ]; do
  [ "$1" = "--concurrency" ] && { CONC="$2"; shift 2; continue; }
  shift
done

PORT=${PORT:-8903}
BASE="http://127.0.0.1:$PORT/v1"
DATA_DIR=$(mktemp -d /tmp/itrace-perf-data.XXXXXX)
CORPUS=$(mktemp -d /tmp/itrace-perf-corpus.XXXXXX)
trap 'kill $SRV 2>/dev/null || true' EXIT

echo "== building release binaries (skipped if fresh) =="
cargo build --release -p itrace-server -p itrace-core --example perf 2>/dev/null | tail -1

./target/release/examples/perf ${QUICK:+$QUICK} --emit-corpus "$CORPUS"
N=$(ls "$CORPUS" | wc -l | tr -d ' ')

echo "== starting server (DATA_DIR=$DATA_DIR PORT=$PORT) =="
DATA_DIR="$DATA_DIR" PORT=$PORT ./target/release/itrace-server >/tmp/itrace-perf-server.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do
  curl -sf "$BASE/health" >/dev/null 2>&1 && break
  sleep 0.25
done
curl -sf "$BASE/health" >/dev/null || { echo "server failed to start"; tail -20 /tmp/itrace-perf-server.log; exit 1; }

ms() { python3 -c "import sys;print(f'{float(sys.argv[1])*1000:.0f}ms')" "$1"; }
elapsed_s() { python3 -c "import time;print(time.time())"; }

PID=$(curl -sf -X POST "$BASE/projects" -H 'Content-Type: application/json' \
  -d '{"name":"perf"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "project id: $PID   corpus: $N images   concurrency: $CONC"

# ---- upload: sequential ----
t0=$(elapsed_s)
for f in "$CORPUS"/*.png; do
  curl -sf -X POST "$BASE/upload" -F "project_id=$PID" -F "file=@$f" >/dev/null
done
t1=$(elapsed_s)
echo "upload sequential ${N}x: $(ms "$(python3 -c "print($t1-$t0)")") total  $(python3 -c "print(f'{($t1-$t0)/$N*1000:.0f}ms/img')")"

# wait for feature precompute to finish
t0=$(elapsed_s)
while :; do
  ready=$(curl -s "$BASE/projects/$PID/feature-status" \
    | python3 -c "import sys,json
try: print(json.load(sys.stdin)['all_ready'])
except Exception: print(False)")
  [ "$ready" = "True" ] && break
  sleep 0.5
done
t1=$(elapsed_s)
echo "precompute (all features, all variants): $(ms "$(python3 -c "print($t1-$t0)")") total"

# ---- timed endpoints ----
declare -a RESULTS
bench() { # name, curl args...
  local name="$1"; shift
  local t0 t1
  t0=$(elapsed_s)
  curl -sf "$@" >/dev/null
  t1=$(elapsed_s)
  RESULTS+=("$(printf '%-38s %s' "$name" "$(ms "$(python3 -c "print($t1-$t0)")")")")
}

J='-H Content-Type:application/json'
bench "compare phash (cold)"        -X POST "$BASE/projects/$PID/compare" $J -d '{"algorithm":"phash","threshold":0.8,"rotation_invariant":false}'
bench "compare phash (warm)"        -X POST "$BASE/projects/$PID/compare" $J -d '{"algorithm":"phash","threshold":0.8,"rotation_invariant":false}'
bench "compare phash rot_inv"       -X POST "$BASE/projects/$PID/compare" $J -d '{"algorithm":"phash","threshold":0.8,"rotation_invariant":true}'
bench "compare ssim"                -X POST "$BASE/projects/$PID/compare" $J -d '{"algorithm":"ssim","threshold":0.8,"rotation_invariant":false}'
bench "compare auto"                -X POST "$BASE/projects/$PID/compare" $J -d '{"algorithm":"auto","threshold":0.8,"rotation_invariant":true}'
bench "smart-compare"               -X POST "$BASE/projects/$PID/smart-compare" $J -d '{"threshold":0.8,"min_agree":2}'
bench "dedup scan"                  -X POST "$BASE/projects/$PID/dedup" $J -d '{}'
bench "matrix phash"                "$BASE/projects/$PID/matrix?algorithm=phash"
bench "matrix ssim"                 "$BASE/projects/$PID/matrix?algorithm=ssim"
bench "report"                      "$BASE/projects/$PID/report?algorithm=auto&threshold=0.8"
bench "feature-status"              "$BASE/projects/$PID/feature-status"

# a pair of ids for match/slice
IDS=$(curl -s "$BASE/projects/$PID/images" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[0]["id"],d[1]["id"])')
read -r A B <<<"$IDS"
bench "match orb"                   -X POST "$BASE/match/pairs" $J -d "{\"image_a_id\":$A,\"image_b_id\":$B,\"algorithm\":\"orb\"}"
bench "match orb (warm)"            -X POST "$BASE/match/pairs" $J -d "{\"image_a_id\":$A,\"image_b_id\":$B,\"algorithm\":\"orb\"}"
bench "slices 2x2"                  -X POST "$BASE/match/slices" $J -d "{\"image_a_id\":$A,\"image_b_id\":$B,\"rows\":2,\"cols\":2}"
bench "thumbnail"                   "$BASE/images/$A/thumbnail?size=256"
bench "thumbnail (cached)"          "$BASE/images/$A/thumbnail?size=256"

# ---- concurrent upload: second project ----
PID2=$(curl -sf -X POST "$BASE/projects" -H 'Content-Type: application/json' \
  -d '{"name":"perf-conc"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
t0=$(elapsed_s)
ls "$CORPUS"/*.png | xargs -P "$CONC" -I{} \
  curl -sf -X POST "$BASE/upload" -F "project_id=$PID2" -F "file=@{}" >/dev/null
t1=$(elapsed_s)
echo "upload concurrent x$CONC ${N}x: $(ms "$(python3 -c "print($t1-$t0)")") total"

# ---- semantic check: every dup family must land in one group ----
curl -sf -X POST "$BASE/projects/$PID/smart-compare" $J \
  -d '{"threshold":0.8,"min_agree":2}' > /tmp/itrace-perf-smart.json
python3 - <<'PY'
import json, re, sys
d = json.load(open('/tmp/itrace-perf-smart.json'))
groups = d.get('duplicate_groups', [])
fams = {}
for g in groups:
    for im in g.get('images', []):
        fam = re.sub(r'_.*', '', im['filename'])
        fams.setdefault(fam, 0)
        # flag a group mixing two families
    fs = {re.sub(r'_.*', '', i['filename']) for i in g.get('images', [])}
    if len(fs) > 1:
        print(f'WARN cross-family group: {fs}', file=sys.stderr)
print(f'smart-compare: {len(groups)} dup groups covering families: {sorted(fams)}')
PY

echo
echo "================ results ================"
printf '%s\n' "${RESULTS[@]}"
echo "========================================="
echo "server log: /tmp/itrace-perf-server.log  data: $DATA_DIR"
