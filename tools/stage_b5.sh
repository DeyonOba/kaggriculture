#!/usr/bin/env bash
# Stage B5: the labour-gated fourth quadrant, on all three B2 agents.
#
# Stage B2's traces against 1000-1500 opponents: with the same eleven hands we
# open a FOURTH quadrant and leave 16-21 tiles idle all mid-game while they
# work three quadrants with 4-5 idle; hands per worked quadrant 2.2-3.5 vs
# their 4-5. The fourth quadrant gated -$6k twice before, and the neural land
# head buys it anyway (and apply_to_plan's crunch fallback would buy it even
# if the head refused). KAGG_MAX_QUADRANTS=3 caps ownership at the opponent's
# configuration: the head's buy choice is masked and the crunch is suppressed.
#
# ONE change against Stage B2, on each of B2's three agents: control, frozen
# trunk, trainable trunk. Same BC checkpoints, seed, reward (win-first),
# frontier mix, pv potential, schedule -- B2's configuration, the best on
# held-out wins so far. Each masked arm is paired against its own B2
# counterpart on the same 785 games via --reuse; nothing is replayed.
#
# If the cap raises wins and money together, consolidation -- knowing when to
# stop investing in land -- is a lever the planner has to learn or be given.
# Pre-registered in docs/stage_b5_preregistration.md.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_BOARD_ARM=T_crop_103 KAGG_THREADS=1 KAGG_MAX_QUADRANTS=3
TRUNK=rl/datasets/trunk_margin_kaggle.pt
B2=rl/runs/stage_b2
OUT=rl/runs/stage_b5
ITERS=20
mkdir -p "$OUT"
COMMON="--seed 1234 --iters $ITERS --episodes 64 --procs 5 --patience 99 --plateau-cool 1.0 --reward win --eval-per-bin 8"

for ARM in control frozen trainable; do
  case $ARM in
    control)   B=0; EXTRA="";;
    frozen)    B=1; EXTRA="--board --trunk-init $TRUNK --freeze-trunk";;
    trainable) B=1; EXTRA="--board --trunk-init $TRUNK";;
  esac
  BC="$B2/bc_$ARM/bc.pt"
  [ -f "$BC" ] || { echo "missing $BC -- B5 starts from B2's BC checkpoints"; exit 1; }
  if [ -f "$OUT/$ARM/telemetry.jsonl" ]; then
    n_done=$(wc -l < "$OUT/$ARM/telemetry.jsonl" | tr -d ' ')
    if [ "$n_done" -ge "$ITERS" ] && [ -f "$OUT/$ARM/eval.json" ]; then
      echo "[$(date +%H:%M)] $ARM complete ($n_done iterations); skipping"; continue
    fi
    stamp=$(date +%m%d_%H%M%S)
    echo "[$(date +%H:%M)] $ARM is PARTIAL ($n_done/$ITERS): moving to $ARM.partial.$stamp and rerunning"
    mv "$OUT/$ARM" "$OUT/$ARM.partial.$stamp"
    [ -f "$OUT/$ARM.log" ] && mv "$OUT/$ARM.log" "$OUT/$ARM.partial.$stamp.log"
  fi
  echo "[$(date +%H:%M)] PPO arm: $ARM + KAGG_MAX_QUADRANTS=3 (from $BC)"
  KAGG_BOARD=$B uv run python -u rl/train.py --resume "$BC" --bc-games 0 \
      $COMMON $EXTRA --run "$OUT/$ARM" 2>&1 | tee "$OUT/$ARM.log" || exit 1
done

if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] session eval already done; skipping"
else
  echo "[$(date +%H:%M)] paired eval: three capped arms played; B2's three reused ..."
  KAGG_BOARD=1 uv run python -u rl/session_eval.py \
      --arm land_control="$OUT/control/eval.json" \
      --arm land_frozen="$OUT/frozen/eval.json" \
      --arm land_trainable="$OUT/trainable/eval.json" \
      --reuse b2_control="$B2/session_eval.json:control" \
      --reuse b2_frozen="$B2/session_eval.json:frozen" \
      --reuse b2_trainable="$B2/session_eval.json:trainable" \
      --baseline b2_control \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] Stage B5 complete."
