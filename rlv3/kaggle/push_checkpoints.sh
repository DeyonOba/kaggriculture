#!/usr/bin/env bash
# Version an rlv3 run's checkpoints as the private Kaggle Dataset
# gideonoba/kaggriculture-rlv3-checkpoints, so the next kernel run mounts them
# and every stage resumes: clone/maia2/bc skip when fresh, ppo continues from
# last.pt for the remaining iterations, the gate runs on what exists.
#
#   rlv3/kaggle/push_checkpoints.sh rlv3/runs/a1 "message" [--only ppo bc]
#
# SAME SHAPE as tools/kaggle_job/push_checkpoints.sh (tar the *.pt/*.json/
# *.jsonl under 64M, `kaggle datasets version --dir-mode skip`) plus THREE
# refusals the existing script lacks, because a resumable runner turns a stale
# file into a permanent skip rather than a one-off wrong answer:
#
#   1. any artifact without a .stamp.json
#   2. any stage marked POISONED in manifest.json
#   3. any stamp whose fingerprint differs from the manifest's current one
#
# `--only <stage>...` pushes a subset. That is exactly what job_topboard.json's
# run-3 note had to do BY HAND when goose_target went from [0,8,16,24,32] to
# [0,4,6,10,16]: every stored `choices` index changed meaning, and every stale
# checkpoint would have made its step skip itself back into the old action
# space. Here the stamp catches it and this flag is the remedy.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
SRC="${1:?run directory holding manifest.json}"
MSG="${2:-rlv3 checkpoints $(date +%Y-%m-%d)}"
DRY="${DRY:-}"          # DRY=1: run the three refusals and build the tarball, upload nothing
shift 2 || true
ONLY=""
if [ "${1:-}" = "--only" ]; then shift; ONLY="$*"; fi
ID=gideonoba/kaggriculture-rlv3-checkpoints
[ -f "$SRC/manifest.json" ] || { echo "no manifest.json in $SRC -- nothing to verify against"; exit 1; }

STAGE=$(mktemp -d); mkdir -p "$STAGE/ds"
LIST="$STAGE/keep.txt"

# The three refusals and the file list are one python pass over the manifest.
SRC="$SRC" ONLY="$ONLY" LIST="$LIST" REPO="$REPO" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["REPO"])
from rlv3 import fingerprint as FP

src = os.environ["SRC"]
only = [s for s in os.environ.get("ONLY", "").split() if s]
man = json.load(open(os.path.join(src, "manifest.json")))
fp = man.get("fingerprint") or {}

poisoned = man.get("poisoned") or {}
blocked = [s for s in poisoned if not only or s in only]
if blocked:
    for s in blocked:
        print(f"REFUSING: stage {s} is POISONED: {poisoned[s].get('reason','')[:200]}")
    sys.exit(1)

# Which stages' artifacts are we shipping?
want = set(only) if only else None
wanted_paths = set()
for sl in man.get("slices") or []:
    for st in sl.get("stages") or []:
        if want is not None and st.get("name") not in want:
            continue
        for p in st.get("artifacts") or []:
            wanted_paths.add(os.path.abspath(p))
if want is None:
    wanted_paths |= set(man.get("artifacts") or {})

keep, bad = [], []
for path in sorted(wanted_paths):
    if not os.path.exists(path):
        continue
    stamp = FP.read_stamp(path)
    if stamp is None:
        bad.append(f"{path}: UNSTAMPED")
        continue
    for k, v in (stamp.get("fp") or {}).items():
        if k in fp and fp[k] != v:
            bad.append(f"{path}: stamp {k} != the manifest's current fingerprint")
            break
    else:
        keep.append(path)
        sp = FP.stamp_path(path)
        if os.path.exists(sp):
            keep.append(sp)

# The PPO run's resumable state travels too -- last.pt, telemetry, slices,
# and the run-dir stamp -- but only when its run-dir stamp is current.
for run in sorted({os.path.dirname(p) for p in keep}):
    st = FP.read_stamp(run)
    if st is None:
        continue
    stale = any(k in fp and fp[k] != v for k, v in (st.get("fp") or {}).items())
    if stale:
        bad.append(f"{run}: run-dir stamp is stale")
        continue
    keep.append(FP.stamp_path(run))
    for name in ("last.pt", "best.pt", "telemetry.jsonl", "slices.jsonl", "eval.json"):
        q = os.path.join(run, name)
        if os.path.exists(q):
            keep.append(q)
keep.append(os.path.join(src, "manifest.json"))

if bad:
    for b in bad:
        print("REFUSING:", b)
    print("push_checkpoints refuses to ship an unstamped or fingerprint-mismatched "
          "artifact: it would make its stage skip itself under a crossed action space.")
    sys.exit(1)
if not keep:
    print("nothing stamped to push"); sys.exit(1)

with open(os.environ["LIST"], "w") as fh:
    for p in sorted(set(keep)):
        if os.path.isfile(p) and os.path.getsize(p) < 64 * 1024 * 1024:
            fh.write(os.path.relpath(p, src) + "\n")
print(f"{len(set(keep))} stamped artifacts, fingerprint code={str(fp.get('code'))[:8]} "
      f"heads v{fp.get('heads_version')}")
PY

( cd "$SRC" && COPYFILE_DISABLE=1 tar -czf "$STAGE/ds/kaggriculture-rlv3-checkpoints.tar.gz" -T "$LIST" )
ls -la "$STAGE/ds/kaggriculture-rlv3-checkpoints.tar.gz" | awk '{print "tarball", $5, "bytes"}'
if [ -n "$DRY" ]; then
  echo "DRY=1: refusals passed and the tarball is built; nothing uploaded."
  exit 0
fi

cat > "$STAGE/ds/dataset-metadata.json" <<META
{"title": "kaggriculture-rlv3-checkpoints", "id": "$ID", "licenses": [{"name": "CC0-1.0"}]}
META
if uv run kaggle datasets list --mine 2>/dev/null | grep -q "$ID"; then
  uv run kaggle datasets version -p "$STAGE/ds" -m "$MSG" --dir-mode skip
else
  uv run kaggle datasets create -p "$STAGE/ds" --dir-mode skip
fi
echo "pushed $ID; allow ~90s for the version to process before pushing a kernel that mounts it"
