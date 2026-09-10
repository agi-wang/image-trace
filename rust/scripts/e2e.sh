#!/usr/bin/env bash
# End-to-end smoke test against a running itrace-server.
# Usage: BASE=http://localhost:8902/v1 ./scripts/e2e.sh
set -euo pipefail
BASE=${BASE:-http://localhost:8902/v1}
IMG=/tmp/itrace-e2e-img
mkdir -p "$IMG"

# deterministic test images (no external downloads)
python3 - "$IMG" <<'PY'
import sys
from PIL import Image
d = sys.argv[1]
img = Image.new('RGB', (256, 256))
px = img.load()
for y in range(256):
    for x in range(256):
        px[x, y] = ((x * 7 + y * 3) % 256, (x + y) % 256, (x * x + y) % 256)
img.save(f'{d}/a.jpg', 'JPEG')
img.transpose(Image.ROTATE_90).save(f'{d}/a_rot90.jpg', 'JPEG')
img.resize((128, 128)).save(f'{d}/a_half.jpg', 'JPEG')
Image.new('RGB', (256, 256), (200, 30, 40)).save(f'{d}/b.jpg', 'JPEG')
PY

curl -sf -X POST "$BASE/projects" -H 'Content-Type: application/json' -d '{"name":"e2e"}' >/dev/null
for f in a a_rot90 a_half b; do
  curl -sf -X POST "$BASE/upload" -F "project_id=1" -F "file=@$IMG/$f.jpg" >/dev/null
done

ready=""
for _ in $(seq 1 60); do
  ready=$(curl -s "$BASE/projects/1/feature-status" \
    | python3 -c "import sys,json
try: print(json.load(sys.stdin)['all_ready'])
except Exception: print(False)")
  [ "$ready" = "True" ] && break
  sleep 1
done
[ "$ready" = "True" ] || { echo "feature precompute timed out"; exit 1; }

# rotation-invariant compare must group a with its rotated copy, not with b
curl -sf -X POST "$BASE/projects/1/compare" -H 'Content-Type: application/json' \
  -d '{"algorithm":"phash","threshold":0.8,"rotation_invariant":true}' > /tmp/itrace-e2e-cmp.json
python3 - /tmp/itrace-e2e-cmp.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
groups = [[i["filename"] for i in g["images"]] for g in d["groups"]]
assert any({"a.jpg", "a_rot90.jpg"} <= set(g) for g in groups), f"rot group missing: {groups}"
assert not any("b.jpg" in g and "a.jpg" in g for g in groups), f"false positive: {groups}"
print("e2e compare OK:", groups)
PY

# slice/sub-image detection: a_half (resize) should not break; slice via REST pair match
curl -sf -X POST "$BASE/match/slices" -H 'Content-Type: application/json' \
  -d '{"image_a_id":1,"image_b_id":3,"rows":2,"cols":2}' | python3 -c "import sys,json;print('slice coverage:',json.load(sys.stdin)['coverage'])"

echo "e2e PASSED"
