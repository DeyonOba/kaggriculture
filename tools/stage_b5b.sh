#!/usr/bin/env bash
# Stage B5b: the labour-gated fourth quadrant, with the land-pressure leak closed.
#
# The first B5 ran with apply_land_pressure ungated (the capped control still
# held 3.8 quadrants on the traced worlds). Now all three purchase paths sit
# behind KAGG_MAX_QUADRANTS, verified on the two traced worlds: max quadrants
# 3, idle mid-game tiles 15.9 -> 4.1 and 20.7 -> 1.9, plants unchanged.
#
# Two tests, cheapest first:
#
#   1. INFERENCE-TIME CAP. B2's three nets, unchanged, played with the cap on.
#      Isolates the rule itself from retraining noise; each arm paired against
#      its own B2 games via --reuse. ~1 h.
#   2. RETRAINED UNDER THE CAP. tools/stage_b5.sh with STAGE_OUT=rl/runs/stage_b5b:
#      control / frozen / trainable from B2's BC checkpoints, ~3 h.
#
# Pre-registered in docs/stage_b5_preregistration.md (S2 and S5).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1 KAGG_MAX_QUADRANTS=3
B2=rl/runs/stage_b2
OUT=rl/runs/stage_b5b
mkdir -p "$OUT"

if [ -s "$OUT/inference_cap_eval.json" ]; then
  echo "[$(date +%H:%M)] inference-time cap eval already done; skipping"
else
  echo "[$(date +%H:%M)] inference-time cap: B2's nets with KAGG_MAX_QUADRANTS=3, paired against their own games ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm cap_control="$B2/control/eval.json" \
      --arm cap_frozen="$B2/frozen/eval.json" \
      --arm cap_trainable="$B2/trainable/eval.json" \
      --reuse b2_control="$B2/session_eval.json:control" \
      --reuse b2_frozen="$B2/session_eval.json:frozen" \
      --reuse b2_trainable="$B2/session_eval.json:trainable" \
      --baseline b2_control \
      --per-bin 175 --procs 5 --out "$OUT/inference_cap_eval.json" 2>&1 | tee "$OUT/inference_cap_eval.log"
fi

echo "[$(date +%H:%M)] retrained under the cap ..."
STAGE_OUT="$OUT" ./tools/stage_b5.sh
echo "[$(date +%H:%M)] Stage B5b complete."
