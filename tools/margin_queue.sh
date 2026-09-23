#!/usr/bin/env bash
# H-margin, queued behind the v1 ablation.
#
# Pre-registered in conversation before the tool was written: an arm passes on
# >= +0.01 R2 in >= 4 of 5 bands against E_full_83, with win AUC reported
# alongside because the rating is win-based. A 2-epoch smoke run exists and is
# not evidence -- at 2 epochs E's own R2 is negative, so neither net has
# learned anything and a larger input is merely further along.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
echo "[$(date +%H:%M)] waiting for the v1 ablation ..."
while pgrep -f "s3_flow.py --ablate" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] v1 done; margin arms, 3 arms x 6 seeds ..."
uv run python -u tools/margin_arms.py --data rl/datasets/s3_flow_v2band.npz \
    --arms E_full_83 T_temporal_95 T_crop_103 --seeds 0 1 2 3 4 5
echo "[$(date +%H:%M)] H-margin complete."
