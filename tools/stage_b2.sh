#!/usr/bin/env bash
# Stage B, second run: frozen vs trainable trunk, on a pipeline that checks itself.
#
# The first Stage B is void twice over. The trunk was never serialised, so the
# worker played a truncated policy while the trainer scored the full one; the
# gap between the two log-probabilities was +3.7..4.0 nats against a KL_STOP
# of 0.03, so the early stop fired on the first minibatch of every iteration
# and NO board arm performed a single policy update. Independently, --resume
# put the BC-trained trunk back over the --trunk-init one. Both are now
# refusals, not silences: train.py asserts the on-policy identity every
# iteration, logs policy_updates, loads the trunk AFTER the resume and prints
# its distance from the file, and excludes the trunk from BC.
#
# Pre-registered in docs/stage_b2_preregistration.md. Three arms, identical in
# everything but the trunk treatment:
#
#   control     no board                     (the 168-feature policy)
#   frozen      board, trunk frozen           (h1's board columns still train)
#   trainable   board, trunk trained by PPO   (same init as frozen)
#
# --reward win: the win is the terminal reward, margin a tanh-bounded
# tiebreaker, money absent. The verdict is rl/session_eval.py on the FINAL
# nets: 175 held-out games a bin where the bin has them (the 0-600 and
# 600-1000 bins hold only 64 and 21 rated held-out seats), 785 an arm,
# paired, McNemar.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1
TRUNK=rl/datasets/trunk_margin_kaggle.pt
OUT=rl/runs/stage_b2
mkdir -p "$OUT"
COMMON="--seed 1234 --iters 20 --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 --reward win --eval-per-bin 8"

for ARM in control frozen trainable; do
  case $ARM in
    control)   B=0; EXTRA="";;
    frozen)    B=1; EXTRA="--board --trunk-init $TRUNK --freeze-trunk";;
    trainable) B=1; EXTRA="--board --trunk-init $TRUNK";;
  esac
  if [ ! -f "$OUT/bc_$ARM/bc.pt" ]; then
    echo "[$(date +%H:%M)] BC for $ARM ..."
    KAGG_BOARD=$B uv run python -u rl/train.py --bc-games 60 --iters 0 --seed 1234 \
        $EXTRA --run "$OUT/bc_$ARM" 2>&1 | tee "$OUT/bc_$ARM.log" || exit 1
  fi
  if [ -s "$OUT/$ARM/telemetry.jsonl" ]; then
    echo "[$(date +%H:%M)] $ARM already has telemetry; skipping"
    continue
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM"
  KAGG_BOARD=$B uv run python -u rl/train.py --resume "$OUT/bc_$ARM/bc.pt" --bc-games 0 \
      $COMMON $EXTRA --run "$OUT/$ARM" 2>&1 | tee "$OUT/$ARM.log" || exit 1
done

echo "[$(date +%H:%M)] powered paired eval on the final nets ..."
KAGG_BOARD=1 uv run python -u rl/session_eval.py \
    --arm control="$OUT/control/eval.json" \
    --arm frozen="$OUT/frozen/eval.json" \
    --arm trainable="$OUT/trainable/eval.json" \
    --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
echo "[$(date +%H:%M)] Stage B2 complete."
