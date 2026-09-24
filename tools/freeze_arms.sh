#!/usr/bin/env bash
# Arm 3 and 4: does FREEZING the trunk recover what the board arm lost?
#
# Stage B: board arm -$10,746/iteration against its control (t -7.79), and the
# gap WIDENED, -5,336 over iterations 3-9 to -14,533 over 10-19. Two mechanisms
# fit that shape and this pair separates them from each other:
#
#   too many trainable parameters   116,553 vs the control's 65,145 on 1,280
#                                   episodes = 0.33 samples a parameter
#   PPO damaging a good encoder     the trunk saw 132k supervised samples; PPO
#                                   hands it 38k frames and a far noisier
#                                   gradient, and a widening gap is what
#                                   progressive degradation looks like
#
# Both arms load the SAME Kaggle trunk (held-out margin R2 +0.4183, full
# corpus), so the only difference between them is the freeze. Frozen drops
# trainable parameters to 74,361 -- 1.14x the control instead of 1.79x.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103
TRUNK=rl/datasets/trunk_margin_kaggle.pt

for ARM in frozen trainable; do
  [ "$ARM" = frozen ] && FRZ="--freeze-trunk" || FRZ=""
  if [ ! -f "rl/runs/stage_b/bc_$ARM/bc.pt" ]; then
    echo "[$(date +%H:%M)] BC for $ARM ..."
    KAGG_BOARD=1 uv run python -u rl/train.py --bc-games 60 --iters 0 --seed 1234 \
        --board --trunk-init $TRUNK $FRZ --run "rl/runs/stage_b/bc_$ARM" || exit 1
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM (kaggle trunk)"
  KAGG_BOARD=1 uv run python -u rl/train.py --resume "rl/runs/stage_b/bc_$ARM/bc.pt" \
      --seed 1234 --iters 20 --episodes 64 --procs 5 --patience 99 --bc-games 0 \
      --plateau-cool 1.0 --board --trunk-init $TRUNK $FRZ \
      --run "rl/runs/stage_b/$ARM"
done
echo "[$(date +%H:%M)] freeze arms complete."
