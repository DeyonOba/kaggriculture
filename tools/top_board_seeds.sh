#!/bin/bash
# Replicate the top-player-cloned RL arms (research note §K) with two more
# training seeds, board and no-board, then pair them on the standing 785
# held-out games against seed 1234, the three capped controls, champ-h8 and
# the dev tree.  Waits for the rule tests (rl/runs/rules_tests/cull_eval.json)
# so the two do not share the 5 procs.  Resumable: a finished arm is skipped.
#
#   bash tools/top_board_seeds.sh            (log: rl/runs/top_board/seeds_queue.log)
#
# The session index must not be rebuilt until seeds_eval.json exists: the
# --reuse arms are checked game by game against the current held-out draw.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_MAX_QUADRANTS=3 KAGG_THREADS=1
OUT=rl/runs/top_board
TRUNK=rl/datasets/trunk_margin_top.pt
[ -s "$TRUNK" ] || { echo "missing $TRUNK"; exit 1; }

echo "[$(date +%H:%M)] waiting for the rule tests (rl/runs/rules_tests/cull_eval.json) ..."
while [ ! -s rl/runs/rules_tests/cull_eval.json ]; do sleep 120; done

run_ppo () {  # name  KAGG_BOARD  seed  extra...
  local NAME=$1 B=$2 SEED=$3; shift 3
  local RUN="$OUT/$NAME"
  if [ -f "$RUN/telemetry.jsonl" ] && [ "$(wc -l < "$RUN/telemetry.jsonl" | tr -d ' ')" -ge 20 ] && [ -f "$RUN/eval.json" ]; then
    echo "[$(date +%H:%M)] $NAME complete; skipping"; return 0
  fi
  if [ -d "$RUN" ]; then mv "$RUN" "$RUN.partial.$(date +%m%d_%H%M%S)"; fi
  echo "[$(date +%H:%M)] PPO $NAME (seed $SEED, cap 3, --vary-worlds, --reward win) ..."
  KAGG_BOARD=$B uv run python -u rl/train.py --resume "$OUT/bc_$( [ "$B" = 1 ] && echo board || echo control )/bc.pt" \
      --bc-games 0 --seed $SEED --vary-worlds --iters 20 --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 \
      --reward win --eval-per-bin 8 "$@" --run "$RUN" 2>&1 | tee "$RUN.log" || exit 1
}

run_ppo top_control_s1235 0 1235
run_ppo top_board_s1235   1 1235 --board --trunk-init "$TRUNK"
run_ppo top_control_s1236 0 1236
run_ppo top_board_s1236   1 1236 --board --trunk-init "$TRUNK"

if [ ! -s "$OUT/seeds_eval.json" ]; then
  echo "[$(date +%H:%M)] paired eval: four new arms played; seed 1234, capped controls, champ-h8, dev tree reused ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm top_board_s1235="$OUT/top_board_s1235/eval.json" \
      --arm top_board_s1236="$OUT/top_board_s1236/eval.json" \
      --arm top_control_s1235="$OUT/top_control_s1235/eval.json" \
      --arm top_control_s1236="$OUT/top_control_s1236/eval.json" \
      --reuse top_board_s1234="$OUT/session_eval.json:top_board_s1234" \
      --reuse top_control_s1234="$OUT/session_eval.json:top_control_s1234" \
      --reuse cap_control_s1234=rl/runs/stage_b5c/session_eval.json:control_s1234 \
      --reuse cap_control_s1235=rl/runs/stage_b5c/session_eval.json:control_s1235 \
      --reuse cap_control_s1236=rl/runs/stage_b5c/session_eval.json:control_s1236 \
      --reuse champ_h8=rl/runs/deploy_check/session_eval.json:champ_h8 \
      --reuse dev_knobs_h8=rl/runs/deploy_check/dev_tree_eval.json:dev_knobs_h8 \
      --baseline champ_h8 --per-bin 175 --procs 5 --out "$OUT/seeds_eval.json" 2>&1 | tee "$OUT/seeds_eval.log"
fi
echo "[$(date +%H:%M)] seed replication complete."
