#!/usr/bin/env bash
# Pretrain the trunk on margin, then run Stage B paired on KAGG_BOARD.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103

echo "[$(date +%H:%M)] waiting for the BC extraction ..."
while pgrep -f session_bc.py >/dev/null; do sleep 60; done

echo "[$(date +%H:%M)] pretraining the trunk on FINAL MARGIN ..."
uv run python -u rl/pretrain_trunk.py --n-draw 4000 --epochs 30 \
    --out rl/datasets/trunk_margin.pt || exit 1

echo "[$(date +%H:%M)] shared BC checkpoint ..."
KAGG_BOARD=1 uv run python -u rl/train.py --bc-games 60 --iters 0 --seed 1234 \
    --board --trunk-init rl/datasets/trunk_margin.pt \
    --run rl/runs/stage_b/bc || exit 1

for ARM in board noboard; do
  if [ "$ARM" = board ]; then EXTRA="--board --trunk-init rl/datasets/trunk_margin.pt"; B=1; else EXTRA=""; B=0; fi
  echo "[$(date +%H:%M)] PPO arm: $ARM"
  KAGG_BOARD=$B uv run python -u rl/train.py --resume rl/runs/stage_b/bc/bc.pt \
      --seed 1234 --iters 20 --episodes 64 --procs 5 --patience 99 --bc-games 0 \
      --plateau-cool 1.0 $EXTRA --run "rl/runs/stage_b/$ARM"
done
echo "[$(date +%H:%M)] Stage B complete."
