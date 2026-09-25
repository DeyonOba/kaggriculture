#!/usr/bin/env bash
# Train the trainable-board planner on the ENTIRE top-player corpus, then PPO,
# then the deployment comparison.
#
# "Top players" = every session in the index with a seat rated >= 2500 (the
# PPO held-out fifth is excluded from every fitting step). New against every
# earlier board arm:
#
#   1. the trunk is pretrained on margin over all top games (not 4,000 drawn);
#   2. the planner READOUT is cloned from the top players' own implied
#      standing targets, with the board -- every earlier BC cloned our
#      board-blind heuristic, so the readout never learned to use the board;
#   3. PPO then trains the trunk (the user's trainable-board model), under the
#      replicated land cap, on three independent seeds (--vary-worlds);
#   4. a no-board planner cloned from the same top games, one seed, isolates
#      what the top-player teacher gives without the board;
#   5. the eval is paired against champ-h8 and the capped control on the
#      same held-out games -- the deployment comparison, not an RL-vs-RL one.
#
# TOP_TAG names a pass (default "top"): datasets, trunk and run dir carry it,
# so a second pass over an enlarged corpus (after the session index is
# rebuilt) lands beside the first. A rebuilt index changes the held-out
# draw, so a pass other than "top" PLAYS champ-h8 and the dev champion as
# arms instead of reusing their earlier games (the reuse check would refuse).
# Stages are resumable: each skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_MAX_QUADRANTS=3
MIN_RATING=${MIN_RATING:-2500}
TAG=${TOP_TAG:-top}
OUT=rl/runs/${TAG}_board
DATA=rl/datasets/session_bc_${TAG}.jsonl
TRUNK=rl/datasets/trunk_margin_${TAG}.pt
DC=rl/runs/deploy_check
B5C=rl/runs/stage_b5c
mkdir -p "$OUT"

if [ ! -s "$DATA" ]; then
  echo "[$(date +%H:%M)] 1/5 dataset ($TAG): boards + top players' implied choices, all sessions with a seat >= $MIN_RATING ..."
  KAGG_THREADS=1 uv run python -u rl/session_bc.py --min-rating $MIN_RATING --n 0 --procs 5 --out "$DATA" 2>&1 | tee "$OUT/dataset.log" || exit 1
else echo "[$(date +%H:%M)] 1/5 dataset exists; skipping"; fi

if [ ! -s "$TRUNK" ]; then
  echo "[$(date +%H:%M)] 2/5 trunk ($TAG): margin pretrain over the top corpus, out of core ..."
  KAGG_THREADS=2 uv run python -u tools/kaggle_trunk.py --min-rating $MIN_RATING --chunk 1000 --passes 2 --procs 5 \
      --budget-s 7200 --out "$TRUNK" 2>&1 | tee "$OUT/trunk.log" || exit 1
else echo "[$(date +%H:%M)] 2/5 trunk exists; skipping"; fi

for ARM in board control; do
  case $ARM in
    board)   B=1; EXTRA="--board --trunk-init $TRUNK";;
    control) B=0; EXTRA="";;
  esac
  if [ ! -f "$OUT/bc_$ARM/bc.pt" ]; then
    echo "[$(date +%H:%M)] 3/5 BC readout ($ARM) from top players' choices ..."
    KAGG_BOARD=$B KAGG_THREADS=3 uv run python -u rl/train.py --bc-session "$DATA" --bc-session-epochs 40 \
        --bc-games 0 --iters 0 --seed 1234 $EXTRA --run "$OUT/bc_$ARM" 2>&1 | tee "$OUT/bc_$ARM.log" || exit 1
  else echo "[$(date +%H:%M)] 3/5 BC ($ARM) exists; skipping"; fi
done

run_ppo () {  # name  KAGG_BOARD  seed  extra...
  local NAME=$1 B=$2 SEED=$3; shift 3
  local RUN="$OUT/$NAME"
  if [ -f "$RUN/telemetry.jsonl" ] && [ "$(wc -l < "$RUN/telemetry.jsonl" | tr -d ' ')" -ge 20 ] && [ -f "$RUN/eval.json" ]; then
    echo "[$(date +%H:%M)] 4/5 $NAME complete; skipping"; return 0
  fi
  if [ -d "$RUN" ]; then mv "$RUN" "$RUN.partial.$(date +%m%d_%H%M%S)"; fi
  echo "[$(date +%H:%M)] 4/5 PPO $NAME (seed $SEED, cap 3) ..."
  KAGG_BOARD=$B KAGG_THREADS=1 uv run python -u rl/train.py --resume "$OUT/bc_$( [ "$B" = 1 ] && echo board || echo control )/bc.pt" \
      --bc-games 0 --seed $SEED --vary-worlds --iters 20 --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 \
      --reward win --eval-per-bin 8 "$@" --run "$RUN" 2>&1 | tee "$RUN.log" || exit 1
}
run_ppo ${TAG}_board_s1234 1 1234 --board --trunk-init "$TRUNK"
run_ppo ${TAG}_control_s1234 0 1234
run_ppo ${TAG}_board_s1235 1 1235 --board --trunk-init "$TRUNK"
run_ppo ${TAG}_board_s1236 1 1236 --board --trunk-init "$TRUNK"

if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] 5/5 eval exists; skipping"
else
  if [ "$TAG" = top ]; then
    REF="--reuse champ_h8=$DC/session_eval.json:champ_h8 --reuse cap_control_s1234=$B5C/session_eval.json:control_s1234 --reuse b2_trainable=rl/runs/stage_b2/session_eval.json:trainable"
  else
    REF="--arm champ_h8=bundle:arena/agents/champ-h8/main.py --arm dev_champion=dev:champion"
  fi
  echo "[$(date +%H:%M)] 5/5 paired eval vs champ-h8 and the dev champion ($TAG) ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm ${TAG}_board_s1234="$OUT/${TAG}_board_s1234/eval.json" \
      --arm ${TAG}_board_s1235="$OUT/${TAG}_board_s1235/eval.json" \
      --arm ${TAG}_board_s1236="$OUT/${TAG}_board_s1236/eval.json" \
      --arm ${TAG}_control_s1234="$OUT/${TAG}_control_s1234/eval.json" \
      $REF --baseline champ_h8 \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] top-player board pipeline ($TAG) complete."
