#!/usr/bin/env bash
# Version a kernel run's checkpoints (trunk, BC and PPO weights, telemetry,
# eval jsons -- a few MB, never datasets) as the private Kaggle Dataset
# gideonoba/kaggriculture-checkpoints. The next kernel run mounts it and
# tools/kaggle_job/job.py copies it into its out/ so every step resumes:
# extraction, trunk and BC are skipped when their outputs exist, PPO continues
# from last.pt for the remaining iterations, and the eval runs on what exists.
#
#   tools/kaggle_job/push_checkpoints.sh rl/runs/kaggle_topboard/out/out "message"
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="${1:?directory holding trunk_margin_top.pt}"
MSG="${2:-checkpoints $(date +%Y-%m-%d)}"
ID=gideonoba/kaggriculture-checkpoints
[ -f "$SRC/trunk_margin_top.pt" ] || { echo "no trunk_margin_top.pt in $SRC"; exit 1; }
STAGE=$(mktemp -d); mkdir -p "$STAGE/ds"
# one tarball of the small files (Kaggle serves it as-is or auto-extracted; job.py handles both)
( cd "$SRC" && COPYFILE_DISABLE=1 find . -type f \( -name '*.pt' -o -name '*.json' -o -name '*.jsonl' \) -size -64M -print0 \
    | COPYFILE_DISABLE=1 tar --null -czf "$STAGE/ds/kaggriculture-checkpoints.tar.gz" -T - )
ls -la "$STAGE/ds/kaggriculture-checkpoints.tar.gz" | awk '{print "tarball", $5, "bytes"}'
cat > "$STAGE/ds/dataset-metadata.json" <<META
{"title": "kaggriculture-checkpoints", "id": "$ID", "licenses": [{"name": "CC0-1.0"}]}
META
if uv run kaggle datasets list --mine 2>/dev/null | grep -q "$ID"; then
  uv run kaggle datasets version -p "$STAGE/ds" -m "$MSG" --dir-mode skip
else
  uv run kaggle datasets create -p "$STAGE/ds" --dir-mode skip
fi
echo "pushed $ID; allow ~90s for the version to process before pushing a kernel that mounts it"
