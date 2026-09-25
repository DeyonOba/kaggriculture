#!/usr/bin/env bash
# Deployment check: the capped control (three seeds) against the shipped
# champion bundles, on the same 785 held-out games.
#
# Everything in the board track compared RL nets to each other. The ladder is
# the metric, and the shipped agent is a frozen main.py: champ-h8 (the
# strongest local champion, the base every gate uses) and champ-h1 (the one
# recorded as submitted). Both play OUR side via the new bundle arm; the three
# capped controls (Stage B5c, KAGG_MAX_QUADRANTS=3, seeds 1234/1235/1236) and
# B2's uncapped control are reused game-by-game. Baseline champ-h8, so the
# table reads "each net vs the champion" directly.
#
# Pre-registered in docs/stage_b5_preregistration.md S10.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_THREADS=1
B2=rl/runs/stage_b2
B5C=rl/runs/stage_b5c
OUT=rl/runs/deploy_check
mkdir -p "$OUT"
if [ -s "$OUT/session_eval.json" ]; then
  echo "[$(date +%H:%M)] deployment check already done; skipping"
else
  echo "[$(date +%H:%M)] deployment check: champ-h8 and champ-h1 played as our side; capped controls reused ..."
  uv run python -u rl/session_eval.py \
      --arm champ_h8=bundle:arena/agents/champ-h8/main.py \
      --arm champ_h1=bundle:arena/agents/champ-h1/main.py \
      --reuse cap_control_s1234="$B5C/session_eval.json:control_s1234" \
      --reuse cap_control_s1235="$B5C/session_eval.json:control_s1235" \
      --reuse cap_control_s1236="$B5C/session_eval.json:control_s1236" \
      --reuse b2_control="$B2/session_eval.json:control" \
      --baseline champ_h8 \
      --per-bin 175 --procs 5 --out "$OUT/session_eval.json" 2>&1 | tee "$OUT/session_eval.log"
fi
echo "[$(date +%H:%M)] deployment check complete."
