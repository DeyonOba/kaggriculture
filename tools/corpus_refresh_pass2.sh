#!/usr/bin/env bash
# Second pass over the ENLARGED corpus: wait for the first pipeline's eval to
# draw its games (the session index must not change before then), rebuild the
# index so the 6,352 sessions the September kernels returned are visible, then
# rerun the top-player pipeline as pass "top2" (own datasets, trunk, runs; the
# eval plays champ-h8 and the dev champion on the new draw).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
echo "[$(date +%H:%M)] waiting for rl/runs/top_board/session_eval.json ..."
while [ ! -s rl/runs/top_board/session_eval.json ]; do sleep 120; done
echo "[$(date +%H:%M)] first pass evaluated; rebuilding the session index over $(ls rl/datasets/sessions | wc -l | tr -d ' ') files ..."
cp rl/datasets/session_index.json "rl/datasets/session_index.before_$(date +%m%d).json"
rm rl/datasets/session_index.json
uv run python tools/opp_coupling.py --n 1 2>&1 | tail -3
uv run python - <<'EOF'
import json
idx = json.load(open('rl/datasets/session_index.json'))
sc = [max(x for x in (v.get('scores') or []) if x is not None) for v in idx.values() if any(x is not None for x in (v.get('scores') or []))]
print(f"index rebuilt: {len(idx):,} sessions, {sum(1 for s in sc if s >= 2500):,} with a seat >= 2500")
EOF
echo "[$(date +%H:%M)] pass 2 ..."
TOP_TAG=top2 ./tools/top_board_pipeline.sh
echo "[$(date +%H:%M)] pass 2 complete."
