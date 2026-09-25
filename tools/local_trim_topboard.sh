#!/usr/bin/env bash
# Minimal LOCAL hypothesis test: let the running first pass finish its two
# seed-1234 arms (board, control), then stop it before seeds 1235/1236 and run
# the paired eval on those two against champ-h8 and the capped control
# (reused). The remaining seeds and the enlarged corpus run on Kaggle.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_MAX_QUADRANTS=3 KAGG_THREADS=1
OUT=rl/runs/top_board
done_arm () { [ -f "$1/telemetry.jsonl" ] && [ "$(wc -l < "$1/telemetry.jsonl" | tr -d ' ')" -ge 20 ] && [ -f "$1/eval.json" ]; }
echo "[$(date +%H:%M)] waiting for top_board_s1234 and top_control_s1234 ..."
while ! (done_arm "$OUT/top_board_s1234" && done_arm "$OUT/top_control_s1234"); do sleep 60; done
echo "[$(date +%H:%M)] both seed-1234 arms complete; stopping the first pass (seeds 1235/1236 move to Kaggle)"
P=$(cat "$OUT/top_board.pid"); kill -- -"$P" 2>/dev/null; sleep 5
for d in "$OUT"/top_board_s1235 "$OUT"/top_board_s1236; do [ -d "$d" ] && mv "$d" "$d.partial.$(date +%m%d_%H%M%S)"; done
if [ ! -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] paired eval: two arms played; champ-h8, capped control, B2 trainable reused ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm top_board_s1234="$OUT/top_board_s1234/eval.json" \
      --arm top_control_s1234="$OUT/top_control_s1234/eval.json" \
      --reuse champ_h8=rl/runs/deploy_check/session_eval.json:champ_h8 \
      --reuse dev_knobs_h8=rl/runs/deploy_check/dev_tree_eval.json:dev_knobs_h8 \
      --reuse cap_control_s1234=rl/runs/stage_b5c/session_eval.json:control_s1234 \
      --reuse b2_trainable=rl/runs/stage_b2/session_eval.json:trainable \
      --baseline champ_h8 --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] local minimal test complete."
