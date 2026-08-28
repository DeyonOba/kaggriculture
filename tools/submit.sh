#!/usr/bin/env bash
# Pre-flight + submit. Usage: tools/submit.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${1:?usage: tools/submit.sh \"submission message\"}"

echo "== build =="
uv run python tools/build_submission.py

echo "== tests =="
uv run pytest tests/ -q

echo "== pre-flight arena (dev must not lose to the best frozen version) =="
uv run python arena/run_games.py --a dev --b starter --n 5 --procs 10 | tail -4

echo "== submit =="
uv run kaggle competitions submit kaggriculture -f main.py -m "$MSG"

echo "== status =="
uv run kaggle competitions submissions kaggriculture | head -5
