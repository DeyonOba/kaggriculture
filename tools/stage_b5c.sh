#!/usr/bin/env bash
# Stage B5c: replicate the capped control and the capped FROZEN-trunk arm with
# two more independent training seeds, before believing +6 points at that size.
#
# B5b, one seed each: capped control 63/785 (+2.0 pts over B2 control),
# capped frozen trunk 68/785 (+6.0 over B2 frozen, margin +$24k, opponent's
# bank -$10k). Every arm in this track is one training run; the eval's se is
# the noise for a FIXED net. This measures training-run variance.
#
# --vary-worlds re-seeds the world seeds and the opponent draw from --seed, so
# each replicate is an independent run, not the same worlds shuffled. Same B2
# BC checkpoints, same configuration, KAGG_MAX_QUADRANTS=3.
#
# Eval: the four new nets played on the same 785 games; B2 control, B2
# frozen, and the seed-1234 capped nets reused game-by-game.
# Pre-registered in docs/stage_b5_preregistration.md S8.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1 KAGG_MAX_QUADRANTS=3
TRUNK=rl/datasets/trunk_margin_kaggle.pt
B2=rl/runs/stage_b2
B5B=rl/runs/stage_b5b
OUT=rl/runs/stage_b5c
ITERS=20
mkdir -p "$OUT"

for SEED in 1235 1236; do
  for ARM in control frozen; do
    case $ARM in
      control) B=0; EXTRA="";;
      frozen)  B=1; EXTRA="--board --trunk-init $TRUNK --freeze-trunk";;
    esac
    RUN="$OUT/${ARM}_s$SEED"
    BC="$B2/bc_$ARM/bc.pt"
    [ -f "$BC" ] || { echo "missing $BC"; exit 1; }
    if [ -f "$RUN/telemetry.jsonl" ]; then
      n_done=$(wc -l < "$RUN/telemetry.jsonl" | tr -d ' ')
      if [ "$n_done" -ge "$ITERS" ] && [ -f "$RUN/eval.json" ]; then
        echo "[$(date +%H:%M)] ${ARM}_s$SEED complete; skipping"; continue
      fi
      stamp=$(date +%m%d_%H%M%S)
      echo "[$(date +%H:%M)] ${ARM}_s$SEED PARTIAL ($n_done/$ITERS): moving aside and rerunning"
      mv "$RUN" "$RUN.partial.$stamp"; [ -f "$RUN.log" ] && mv "$RUN.log" "$RUN.partial.$stamp.log"
    fi
    echo "[$(date +%H:%M)] PPO arm: ${ARM}_s$SEED (cap 3, --vary-worlds)"
    KAGG_BOARD=$B uv run python -u rl/train.py --resume "$BC" --bc-games 0 \
        --seed $SEED --vary-worlds --iters $ITERS --episodes 64 --procs 5 --patience 99 \
        --plateau-cool 1.0 --reward win --eval-per-bin 8 $EXTRA --run "$RUN" 2>&1 | tee "$RUN.log" || exit 1
  done
done

if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] eval already done; skipping"
else
  echo "[$(date +%H:%M)] paired eval: four replicate nets played; B2 and seed-1234 nets reused ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm control_s1235="$OUT/control_s1235/eval.json" \
      --arm frozen_s1235="$OUT/frozen_s1235/eval.json" \
      --arm control_s1236="$OUT/control_s1236/eval.json" \
      --arm frozen_s1236="$OUT/frozen_s1236/eval.json" \
      --reuse b2_control="$B2/session_eval.json:control" \
      --reuse b2_frozen="$B2/session_eval.json:frozen" \
      --reuse control_s1234="$B5B/session_eval.json:land_control" \
      --reuse frozen_s1234="$B5B/session_eval.json:land_frozen" \
      --baseline b2_control \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] Stage B5c complete."
