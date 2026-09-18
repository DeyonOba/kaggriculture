#!/usr/bin/env bash
# Push the current repo as a private Kaggle Dataset version and (re)run the job kernel.
# Usage: tools/kaggle_job/push.sh "message"
set -euo pipefail
cd "$(dirname "$0")/../.."
MSG="${1:-update}"
CODE_ONLY="${2:-}"   # pass "code-only" to version the code dataset without (re)starting the job kernel
KERNEL_ID="${KERNEL_ID:-gideonoba/kaggriculture-job}"   # override to run on another kernel slug
CONFIG="${CONFIG:-tools/kaggle_job/job_config.json}"      # override the job config file
META="${META:-tools/kaggle_job/kernel-metadata.json}"      # override the kernel metadata
                                                          # (dataset_sources, enable_gpu)
SKIP_CODE="${SKIP_CODE:-}"                                # "1": push the kernel only (code dataset already current)
STAGE=$(mktemp -d)
mkdir -p "$STAGE/ds"
# Code tarball: tracked files only (no .git, no datasets, no runs).
COPYFILE_DISABLE=1 git ls-files -z | COPYFILE_DISABLE=1 tar --null -czf "$STAGE/ds/kaggriculture-code.tar.gz" -T -
cat > "$STAGE/ds/dataset-metadata.json" <<META
{"title": "kaggriculture-code", "id": "gideonoba/kaggriculture-code", "licenses": [{"name": "CC0-1.0"}]}
META
if [ "$SKIP_CODE" != "1" ]; then
  # (the dataset status endpoint 403s with a legacy API key; use the listing)
  if uv run kaggle datasets list --mine 2>/dev/null | grep -q "gideonoba/kaggriculture-code"; then
    uv run kaggle datasets version -p "$STAGE/ds" -m "$MSG" --dir-mode skip
  else
    uv run kaggle datasets create -p "$STAGE/ds" --dir-mode skip
  fi
  echo "== waiting 90s for the new dataset version to process =="
  sleep 90
fi
if [ "$CODE_ONLY" = "code-only" ]; then echo "code-only: kernel not pushed"; exit 0; fi
echo "== kernel push =="
# Kaggle uploads only the script file: inject job_config.json into job.py.
PUSHDIR="$STAGE/kernel"; mkdir -p "$PUSHDIR"
CONFIG="$CONFIG" KERNEL_ID="$KERNEL_ID" META="$META" python3 - "$PUSHDIR" <<'PY'
import json, sys, os
d = sys.argv[1]
cfg = json.dumps(json.load(open(os.environ.get("CONFIG", "tools/kaggle_job/job_config.json"))))
src = open("tools/kaggle_job/job.py").read().replace("CONFIG_JSON = None", "CONFIG_JSON = " + repr(cfg))
open(os.path.join(d, "job.py"), "w").write(src)
meta = json.load(open(os.environ.get("META", "tools/kaggle_job/kernel-metadata.json")))
kid = os.environ.get("KERNEL_ID", meta["id"]); meta["id"] = kid; meta["title"] = kid.split("/")[-1]
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
PY
uv run kaggle kernels push -p "$PUSHDIR"
echo "poll: uv run kaggle kernels status gideonoba/kaggriculture-job"
echo "fetch: uv run kaggle kernels output gideonoba/kaggriculture-job -p <dir>"
