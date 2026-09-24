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

# ONE BC CHECKPOINT PER ARM, not one shared. The arms differ in ARCHITECTURE
# -- the board arm's first layer is 216 wide and carries a board.* submodule,
# the control's is 168 -- so a shared checkpoint cannot load into both, and
# the first run died exactly there. What is held identical is everything that
# can be: the same --seed, the same 60 BC games, the same world seeds per
# iteration, the same LR schedule. The initial WEIGHTS necessarily differ,
# which is a real limitation of this comparison and is stated with the result.
for ARM in board noboard; do
  if [ "$ARM" = board ]; then
    EXTRA="--board --trunk-init rl/datasets/trunk_margin.pt"; B=1
  else
    EXTRA=""; B=0
  fi
  if [ ! -f "rl/runs/stage_b/bc_$ARM/bc.pt" ]; then
    echo "[$(date +%H:%M)] BC checkpoint for $ARM ..."
    KAGG_BOARD=$B uv run python -u rl/train.py --bc-games 60 --iters 0 \
        --seed 1234 $EXTRA --run "rl/runs/stage_b/bc_$ARM" || exit 1
  fi
  if [ -s "rl/runs/stage_b/$ARM/telemetry.jsonl" ]; then
    echo "[$(date +%H:%M)] $ARM already has telemetry; skipping"
    continue
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM"
  KAGG_BOARD=$B uv run python -u rl/train.py --resume "rl/runs/stage_b/bc_$ARM/bc.pt" \
      --seed 1234 --iters 20 --episodes 64 --procs 5 --patience 99 --bc-games 0 \
      --plateau-cool 1.0 $EXTRA --run "rl/runs/stage_b/$ARM"
done
echo "[$(date +%H:%M)] Stage B complete."
