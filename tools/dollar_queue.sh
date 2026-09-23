#!/usr/bin/env bash
# H-dollar, queued behind H-margin (which is itself behind the v1 ablation).
#
# Pre-registered before the run: training with a dollar-weighted flow loss
# improves dollar-weighted held-out R2 by >= +0.01 in >= 4 of 5 rating bands
# against the equal-weight loss, on identical channels and seeds, WITHOUT
# losing more than 0.02 equal-weight R2 in any band. Both metrics are reported
# for both arms so the dollar arm cannot win by being scored on its own
# objective.
#
# A 2-epoch smoke run exists and is not evidence: the weighted loss spans a
# 50x conditioning range, so it converges more slowly, and at 2 epochs neither
# arm has converged at all.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
echo "[$(date +%H:%M)] waiting for H-margin ..."
while pgrep -f "margin_arms.py" >/dev/null || pgrep -f "margin_queue.sh" >/dev/null \
   || pgrep -f "s3_flow.py --ablate" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] running H-dollar, 2 loss arms x 6 seeds ..."
uv run python -u tools/flow_weight_arm.py --arm E_full_83 --seeds 0 1 2 3 4 5
echo "[$(date +%H:%M)] H-dollar complete."
