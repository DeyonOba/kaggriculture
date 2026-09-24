#!/usr/bin/env bash
# Arm 5: trainable trunk at a SMALL learning rate -- the middle the first two
# arms bracket.
#
#   trainable (scale 1.0)  money -10,746 vs control (t -7.79), win +0.052
#   frozen    (scale 0.0)  money  -5,922 vs control (t -3.99), win +0.004
#
# Freezing recovered 45% of the money and gave up all of the win edge. If the
# encoder is being overwritten by a gradient far noisier than the 132k
# supervised samples that built it, a small non-zero rate should keep the
# money near frozen and the wins near trainable. If it just interpolates, the
# board is simply not usable by these heads and the track is closed.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103
TRUNK=rl/datasets/trunk_margin_kaggle.pt

echo "[$(date +%H:%M)] waiting for the trainable arm ..."
while pgrep -f "freeze_arms.sh" >/dev/null || pgrep -f "rl/train.py" >/dev/null; do sleep 60; done

for S in 0.1 0.03; do
  ARM="slow$(echo $S | tr -d '.')"
  if [ ! -f "rl/runs/stage_b/bc_$ARM/bc.pt" ]; then
    echo "[$(date +%H:%M)] BC for $ARM ..."
    KAGG_BOARD=1 uv run python -u rl/train.py --bc-games 60 --iters 0 --seed 1234 \
        --board --trunk-init $TRUNK --run "rl/runs/stage_b/bc_$ARM" || exit 1
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM (trunk lr x $S)"
  KAGG_BOARD=1 uv run python -u rl/train.py --resume "rl/runs/stage_b/bc_$ARM/bc.pt" \
      --seed 1234 --iters 20 --episodes 64 --procs 5 --patience 99 --bc-games 0 \
      --plateau-cool 1.0 --board --trunk-init $TRUNK --trunk-lr-scale $S \
      --run "rl/runs/stage_b/$ARM"
done
echo "[$(date +%H:%M)] slow-trunk arms complete."
