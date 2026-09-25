#!/usr/bin/env bash
# Run the top-player board pipeline on Kaggle's free CPU tier, end to end:
#   1. wait for the corpus push (tools/data.py push) to finish, so the kernel
#      mounts the enlarged datasets dataset (27,844 sessions);
#   2. version the code dataset and push the kernel (tools/kaggle_job/push.sh
#      with the topboard config + metadata);
#   3. poll the kernel until it completes (or errors), then fetch its output
#      into rl/runs/kaggle_topboard/out and print the eval log.
#
# Nothing is downloaded but the results (~tens of MB). The daily replay dumps
# never cross the wire: the corpus is already sessions.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
OUT=rl/runs/kaggle_topboard
KID=gideonoba/kaggriculture-topboard
mkdir -p "$OUT"

if [ -f "$OUT/corpus_push.pid" ]; then
  P=$(cat "$OUT/corpus_push.pid")
  echo "[$(date +%H:%M)] waiting for the corpus push (pid $P) ..."
  while kill -0 "$P" 2>/dev/null; do sleep 30; done
  echo "[$(date +%H:%M)] corpus push finished:"; tail -3 "$OUT/corpus_push.log"
  if ! grep -qiE "successfully|complete|created|Upload" "$OUT/corpus_push.log"; then
    echo "corpus push did not report success; not pushing the kernel"; exit 1
  fi
  echo "[$(date +%H:%M)] waiting 120s for the dataset version to process ..."; sleep 120
fi

echo "[$(date +%H:%M)] pushing code dataset + kernel $KID ..."
KERNEL_ID=$KID CONFIG=tools/kaggle_job/job_topboard.json META=tools/kaggle_job/kernel-metadata-topboard.json \
  tools/kaggle_job/push.sh "top-player board pipeline $(date +%Y-%m-%d)" 2>&1 | tail -6

echo "[$(date +%H:%M)] polling kernel status ..."
for i in $(seq 1 160); do
  sleep 300
  S=$(uv run kaggle kernels status "$KID" 2>&1 | tail -1)
  echo "[$(date +%H:%M)] $S"
  case "$S" in
    *complete*|*COMPLETE*) break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "kernel ended with: $S"; break;;
  esac
done
echo "[$(date +%H:%M)] fetching output ..."
rm -rf "$OUT/out"; uv run kaggle kernels output "$KID" -p "$OUT/out" 2>&1 | tail -3
ls -la "$OUT/out" 2>/dev/null | head -20
[ -f "$OUT/out/out/session_eval.log" ] && cat "$OUT/out/out/session_eval.log"
echo "[$(date +%H:%M)] kaggle top-board job done."
