#!/usr/bin/env bash
# Does the DEV TREE still reproduce the frozen champion?
#
# champ-h8 is the champion plan PLUS a knob-bandit layer (BANDIT_RULES,
# kind 'knobs'); champ-h1 is the plain champion plan. On the same 785 games
# they scored 0.183 and 0.059. Both are frozen bundles from weeks ago; the
# dev tree has moved since. This plays today's kagent/ as our side, both
# ways -- the plain champion plan, and the knob bandit with champ-h8's own
# rules lifted from its bundle -- and pairs each against its frozen twin.
#
# Pre-registered in docs/stage_b5_preregistration.md S12.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. KAGG_THREADS=1
DC=rl/runs/deploy_check
B5C=rl/runs/stage_b5c
OUT=rl/runs/deploy_check
RULES=$DC/champ-h8_bandit_rules.json
[ -s "$RULES" ] || { echo "missing $RULES"; exit 1; }
if [ -s "$OUT/dev_tree_eval.json" ]; then
  echo "[$(date +%H:%M)] dev-tree check already done; skipping"
else
  echo "[$(date +%H:%M)] dev tree as our side: champion plan, and knob bandit with champ-h8's rules ..."
  uv run python -u rl/session_eval.py \
      --arm dev_knobs_h8=dev:knobs:$RULES \
      --arm dev_champion=dev:champion \
      --reuse champ_h8="$DC/session_eval.json:champ_h8" \
      --reuse champ_h1="$DC/session_eval.json:champ_h1" \
      --reuse cap_control_s1234="$B5C/session_eval.json:control_s1234" \
      --baseline champ_h8 \
      --per-bin 175 --procs 5 --out "$OUT/dev_tree_eval.json" 2>&1 | tee "$OUT/dev_tree_eval.log"
fi
echo "[$(date +%H:%M)] dev-tree check complete."
