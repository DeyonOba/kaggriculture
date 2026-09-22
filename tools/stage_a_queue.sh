#!/usr/bin/env bash
# Stage A, second half: the v1/v2 attribution, queued behind the running
# six-seed ablation so the two jobs never contend for the same cores.
#
# The six-seed run on encoder v2 answers "are the temporal arms positive".
# It cannot answer "positive BECAUSE of the market-channel repair", because
# that dataset changed two things at once against the old verdict: the
# encoder AND the switch from a rating floor to band stratification.
#
# This builds the SAME 1,075 sessions under encoder v1 -- same --seed 0, same
# --bands draw, `KAGG_ENC=1` reinstating min(abs(inv-10000),600)/600 -- and
# re-runs the identical arms and seeds. Differences are then attributable to
# the channel and to nothing else.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "[$(date +%H:%M)] waiting for the v2 ablation to finish ..."
while pgrep -f "s3_flow.py --ablate" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] v2 ablation done."

echo "[$(date +%H:%M)] building the v1 dataset over the same sessions ..."
KAGG_ENC=1 uv run python -u tools/s3_flow.py --build --bands --n 1200 --procs 4 \
    --data-out rl/datasets/s3_flow_v1band.npz

# The two builds must cover the SAME sessions or the comparison is just two
# different draws. `R` is the per-row session rating in draw order, so an
# exact match is a strong check that the band shuffle was reproducible.
uv run python - <<'PY'
import numpy as np, sys
a = np.load("rl/datasets/s3_flow_v1band.npz")
b = np.load("rl/datasets/s3_flow_v2band.npz")
same = a["R"].shape == b["R"].shape and np.allclose(a["R"], b["R"], equal_nan=True)
print(f"    v1 encoder stamp: {int(a['ENC'][0])}   v2: {int(b['ENC'][0])}")
print(f"    same sessions in the same order: {'PASS' if same else 'FAIL'}"
      f"   ({len(a['R']):,} vs {len(b['R']):,} rows)")
if not same:
    print("    -> the builds diverged; the v1/v2 comparison is NOT attributable")
    sys.exit(1)
PY
[ $? -ne 0 ] && { echo "[$(date +%H:%M)] aborting: session sets differ"; exit 1; }

echo "[$(date +%H:%M)] six-seed ablation on encoder v1 ..."
uv run python -u tools/s3_flow.py --ablate --data rl/datasets/s3_flow_v1band.npz \
    --base E_full_83 --arms E_full_83 T_temporal_95 T_crop_103 --seeds 0 1 2 3 4 5
echo "[$(date +%H:%M)] Stage A complete."
