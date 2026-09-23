#!/usr/bin/env bash
# H-perday, queued behind H-margin and AHEAD of the dollar-weight arm.
#
# Block A is the decisive one: all three arms, the CURRENT loss, scored per
# cell in dollars instead of on the horizon sum. It re-tests the arms the stop
# rule killed, and it changes only the statistic -- so if T_crop_103 passes
# here while failing there, the difference is the metric and nothing else.
#
# Block B then sweeps the LOSS: dollar weighting x horizon decay, on E and the
# best temporal arm. gamma = 1 is in the sweep as the null, because no
# measurement supports any particular decay.
#
# tools/dollar_queue.sh is retired: it tested E_full_83 alone, equal vs dollar,
# which is a strict subset of Block B.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
echo "[$(date +%H:%M)] waiting for the v1 ablation and H-margin ..."
while pgrep -f "s3_flow.py --ablate" >/dev/null \
   || pgrep -f "margin_arms.py" >/dev/null \
   || pgrep -f "margin_queue.sh" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] BLOCK A: 3 arms, current loss, per-day dollar score ..."
uv run python -u tools/perday_arms.py --block A --seeds 0 1 2 3 4 5
echo "[$(date +%H:%M)] BLOCK B: loss sweep, dollar x gamma ..."
uv run python -u tools/perday_arms.py --block B --seeds 0 1 2 3 4 5
echo "[$(date +%H:%M)] H-perday complete."
