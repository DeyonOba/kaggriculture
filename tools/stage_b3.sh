#!/usr/bin/env bash
# Stage B3: trainable trunk vs control, trained on the DEPLOYMENT distribution.
#
# Stage B2 answered the freeze question (frozen trunk -3.3 pts, z -4.05, worse
# in 6/6 bands; trainable = parity with control) and exposed why no arm
# improved: the frontier curriculum never left the 0-600 bin, so ~35% of
# episodes were against 0-600 tapes and ~35-42% against the heuristic, while
# the paired eval is 89% opponents rated 1000+. Both board arms beat THAT mix
# by $13-19k and lost or tied held out.
#
# This run changes exactly one thing against B2: --session-mix eval draws
# session opponents by the paired eval's band weights and --session-frac 0.9
# shrinks the heuristic/league share to 10%. Same seeds, same reward, same LR
# schedule, and the SAME BC checkpoints as B2 (rl/runs/stage_b2/bc_*/bc.pt),
# so the two arms start from exactly the policies B2's did.
#
# The eval replays B2's control as a third arm on the identical 785 games:
# B3 control vs B2 control is the paired test of the curriculum fix itself,
# and the replay must reproduce B2's 47/785 exactly (temperature 0).
#
# Pre-registered in docs/stage_b3_preregistration.md.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1
TRUNK=rl/datasets/trunk_margin_kaggle.pt
B2=rl/runs/stage_b2
OUT=rl/runs/stage_b3
ITERS=20
mkdir -p "$OUT"
COMMON="--seed 1234 --iters $ITERS --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 --reward win --eval-per-bin 8 --session-mix eval --session-frac 0.9"

for ARM in control trainable; do
  case $ARM in
    control)   B=0; EXTRA="";;
    trainable) B=1; EXTRA="--board --trunk-init $TRUNK";;
  esac
  BC="$B2/bc_$ARM/bc.pt"
  if [ ! -f "$BC" ]; then
    echo "missing $BC -- B3 must start from B2's BC checkpoints"; exit 1
  fi
  if [ -f "$OUT/$ARM/telemetry.jsonl" ]; then
    n_done=$(wc -l < "$OUT/$ARM/telemetry.jsonl" | tr -d ' ')
    if [ "$n_done" -ge "$ITERS" ] && [ -f "$OUT/$ARM/eval.json" ]; then
      echo "[$(date +%H:%M)] $ARM complete ($n_done iterations); skipping"
      continue
    fi
    stamp=$(date +%m%d_%H%M%S)
    echo "[$(date +%H:%M)] $ARM is PARTIAL ($n_done/$ITERS iterations): moving to $ARM.partial.$stamp and rerunning"
    mv "$OUT/$ARM" "$OUT/$ARM.partial.$stamp"
    [ -f "$OUT/$ARM.log" ] && mv "$OUT/$ARM.log" "$OUT/$ARM.partial.$stamp.log"
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM (from $BC)"
  KAGG_BOARD=$B uv run python -u rl/train.py --resume "$BC" --bc-games 0 \
      $COMMON $EXTRA --run "$OUT/$ARM" 2>&1 | tee "$OUT/$ARM.log" || exit 1
done

if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] session eval already done ($OUT/session_eval.json); skipping"
else
  echo "[$(date +%H:%M)] powered paired eval: B3 control (baseline), B3 trainable, B2 control replayed ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm control="$OUT/control/eval.json" \
      --arm trainable="$OUT/trainable/eval.json" \
      --arm b2_control="$B2/control/eval.json" \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] Stage B3 complete."
