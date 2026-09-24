#!/usr/bin/env bash
# Stage B4: the phase potential as the dense signal on the deployment mix.
#
# B3 showed the bottleneck: on the eval-band opponent mix the win-first
# terminal reward arrives 1-5 times in 64 episodes, and the control trained on
# it lost 2.3 points of held-out win rate to B2's control (p 0.011) on
# identical games. rl/phase_potential.py is the instrument for exactly that
# regime -- a dense, opponent-relative potential (asset lead early, money lead
# late, |Phi| <= 15) that by the PBRS theorem cannot move the optimum and can
# only speed convergence to the same win-first objective.
#
# ONE arm, ONE change against B3's control: --potential lead. Same BC start
# (B2's bc_control), same seed, same reward, same mix, same schedule. No
# board. The eval pairs it against B3's control by REUSING its 785 games
# (checked game by game) and shows B2's control for reference; only the new
# arm is played, ~19 min.
#
# Pre-registered in docs/stage_b4_preregistration.md.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1
B2=rl/runs/stage_b2
B3=rl/runs/stage_b3
OUT=rl/runs/stage_b4
ITERS=20
mkdir -p "$OUT"
COMMON="--seed 1234 --iters $ITERS --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 --reward win --eval-per-bin 8 --session-mix eval --session-frac 0.9 --potential lead"

ARM=lead
BC="$B2/bc_control/bc.pt"
[ -f "$BC" ] || { echo "missing $BC"; exit 1; }
[ -s "$B3/session_eval.json" ] || { echo "missing $B3/session_eval.json (the baseline's games)"; exit 1; }
if [ -f "$OUT/$ARM/telemetry.jsonl" ]; then
  n_done=$(wc -l < "$OUT/$ARM/telemetry.jsonl" | tr -d ' ')
  if [ "$n_done" -ge "$ITERS" ] && [ -f "$OUT/$ARM/eval.json" ]; then
    echo "[$(date +%H:%M)] $ARM complete ($n_done iterations); skipping"
  else
    stamp=$(date +%m%d_%H%M%S)
    echo "[$(date +%H:%M)] $ARM is PARTIAL ($n_done/$ITERS): moving to $ARM.partial.$stamp and rerunning"
    mv "$OUT/$ARM" "$OUT/$ARM.partial.$stamp"
    [ -f "$OUT/$ARM.log" ] && mv "$OUT/$ARM.log" "$OUT/$ARM.partial.$stamp.log"
  fi
fi
if [ ! -f "$OUT/$ARM/eval.json" ]; then
  echo "[$(date +%H:%M)] PPO arm: $ARM (from $BC, --potential lead)"
  KAGG_BOARD=0 uv run python -u rl/train.py --resume "$BC" --bc-games 0 \
      $COMMON --run "$OUT/$ARM" 2>&1 | tee "$OUT/$ARM.log" || exit 1
fi

if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] session eval already done; skipping"
else
  echo "[$(date +%H:%M)] paired eval: lead played; B3 control (baseline) and B2 control reused ..."
  KAGG_BOARD=0 uv run python -u rl/session_eval.py \
      --arm lead="$OUT/$ARM/eval.json" \
      --reuse b3_control="$B3/session_eval.json:control" \
      --reuse b2_control="$B2/session_eval.json:control" \
      --baseline b3_control \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] Stage B4 complete."
