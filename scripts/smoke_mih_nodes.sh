#!/usr/bin/env bash
# Multi-node MIH simulation smoke: dedup the same image subset twice —
# once on the default monolithic ShardedMihIndex and once with
# ITRACE_MIH_NODES=4 (MultiNodeMihIndex shard-ownership scatter/gather) —
# and assert identical duplicate-group output. The env only affects the
# non-persistent path, so ITRACE_MIH_INDEX_DIR is deliberately unset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
cd "$ROOT"

RUN_TAG="${1:-a}"
BASE="$ROOT/datasets/built/reports/mih_nodes_$RUN_TAG"
BIN="$ROOT/target/release/itrace-cli"
rm -rf "$BASE"
mkdir -p "$BASE/mono" "$BASE/multi"
unset ITRACE_MIH_INDEX_DIR

if [[ ! -x "$BIN" ]]; then
  echo "building itrace-cli..."
  cargo build --release -p itrace-cli
fi

# Two families with known gate near-dups (orig/rot90/hflip/crop70) plus
# slice tiles to exercise the crop channel.
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

run_dedup() {  # $1 = data dir, $2 = mode label, rest = env assignment
  local dir="$1" mode="$2"; shift 2
  local pid
  pid=$("$BIN" --data-dir "$dir" create "mih-nodes-$mode" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  "$BIN" --data-dir "$dir" add "$pid" "${FILES[@]}" >/dev/null
  env "$@" "$BIN" --data-dir "$dir" dedup "$pid" | tee "$BASE/$mode.txt"
}

run_dedup "$BASE/mono" mono
run_dedup "$BASE/multi" multi ITRACE_MIH_NODES=4

# Both runs must emit the same dup-group count and the same groups.
mono_groups=$(grep -c "^duplicate group" "$BASE/mono.txt")
multi_groups=$(grep -c "^duplicate group" "$BASE/multi.txt")
echo "mono groups=$mono_groups  multi groups=$multi_groups"
[[ "$mono_groups" -eq 2 ]] || { echo "expected 2 mono groups" >&2; exit 1; }
[[ "$multi_groups" -eq "$mono_groups" ]] || { echo "group count mismatch" >&2; exit 1; }
# Same membership (member lines are `  - name (id)`; ignore timing).
# Group emission order is NOT part of the contract (multi-node
# scatter/gather can emit components in a different order), so compare
# the sorted member set.
diff <(grep "^  - " "$BASE/mono.txt" | sort) \
     <(grep "^  - " "$BASE/multi.txt" | sort) >/dev/null \
  || { echo "group membership mismatch" >&2; exit 1; }

echo "MIH_NODES_SMOKE_OK"
