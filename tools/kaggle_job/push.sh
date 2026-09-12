#!/usr/bin/env bash
# Push the current repo as a private Kaggle Dataset version and (re)run the job kernel.
# Usage: tools/kaggle_job/push.sh "message"
set -euo pipefail
cd "$(dirname "$0")/../.."
MSG="${1:-update}"
STAGE=$(mktemp -d)
mkdir -p "$STAGE/ds"
# Code tarball: tracked files only (no .git, no datasets, no runs).
git ls-files -z | tar --null -czf "$STAGE/ds/kaggriculture-code.tar.gz" -T -
cat > "$STAGE/ds/dataset-metadata.json" <<META
{"title": "kaggriculture-code", "id": "gideonoba/kaggriculture-code", "licenses": [{"name": "CC0-1.0"}]}
META
if uv run kaggle datasets status gideonoba/kaggriculture-code >/dev/null 2>&1; then
  uv run kaggle datasets version -p "$STAGE/ds" -m "$MSG" --dir-mode skip
else
  uv run kaggle datasets create -p "$STAGE/ds" --dir-mode skip
fi
echo "== kernel push =="
# Kaggle uploads only the script file: inject job_config.json into job.py.
PUSHDIR="$STAGE/kernel"; mkdir -p "$PUSHDIR"
python3 - "$PUSHDIR" <<'PY'
import json, sys, os
d = sys.argv[1]
cfg = json.dumps(json.load(open("tools/kaggle_job/job_config.json")))
src = open("tools/kaggle_job/job.py").read().replace("CONFIG_JSON = None", "CONFIG_JSON = " + repr(cfg))
open(os.path.join(d, "job.py"), "w").write(src)
open(os.path.join(d, "kernel-metadata.json"), "w").write(open("tools/kaggle_job/kernel-metadata.json").read())
PY
uv run kaggle kernels push -p "$PUSHDIR"
echo "poll: uv run kaggle kernels status gideonoba/kaggriculture-job"
echo "fetch: uv run kaggle kernels output gideonoba/kaggriculture-job -p <dir>"
