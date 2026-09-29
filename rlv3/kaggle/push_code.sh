#!/usr/bin/env bash
# Push the repo AS THE KERNEL WILL SEE IT -- `git ls-files` PLUS the untracked
# rlv3/ package -- as the kaggriculture-code Dataset, then push the rlv3
# kernel through tools/kaggle_job/push.sh with SKIP_CODE=1.
#
#   rlv3/kaggle/push_code.sh "message" [code-only]
#
# WHY THIS SCRIPT EXISTS AT ALL. rlv3/ is untracked by instruction (the user
# asked for it uncommitted), and tools/kaggle_job/push.sh builds its tarball
# from `git ls-files`. The stock path would upload a code dataset with NO rlv3
# package in it and a kernel that dies on `import rlv3` right after the mount
# -- the same class of failure push.sh's own comment records costing a 1.9h
# kernel run on 2026-09-26, when kagent/policy.py shipped an import of an
# untracked kagent/boardstate.py. So: add rlv3's files explicitly, then repeat
# push.sh's extract-and-import verification with rlv3 in the import line, and
# exit non-zero BEFORE anything is uploaded.
#
# Nothing here is git added, committed or stashed.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
MSG="${1:-rlv3}"
CODE_ONLY="${2:-}"          # "code-only": version the dataset, do not (re)run the kernel
DRY="${DRY:-}"              # DRY=1: build and VERIFY the tarball, upload nothing
KERNEL_ID="${KERNEL_ID:-gideonoba/kaggriculture-rlv3}"
CONFIG="${CONFIG:-rlv3/kaggle/job_rlv3.json}"
META="${META:-rlv3/kaggle/kernel-metadata-rlv3.json}"

STAGE=$(mktemp -d); mkdir -p "$STAGE/ds"
TARBALL="$STAGE/ds/kaggriculture-code.tar.gz"
LIST="$STAGE/files.txt"

# ---- the file list: tracked files, plus rlv3's sources, plus essential datasets, minus the junk.
git ls-files > "$LIST"
find rlv3 \( -name '*.py' -o -name '*.json' -o -name '*.sh' -o -name '*.md' \) \
     -not -path 'rlv3/runs/*' -not -path '*/__pycache__/*' -not -name '*.pt' \
     | sort >> "$LIST"
# Essential corpus index (27k+ sessions)
echo "rl/datasets/session_index.json" >> "$LIST"
sort -u "$LIST" -o "$LIST"

# ---- refusals, before a single byte is uploaded.
if grep -qE '^rlv3/runs/|/__pycache__/|\.pt$' "$LIST"; then
  echo "REFUSING: run output, __pycache__ or a .pt slipped into the code tarball:"
  grep -nE '^rlv3/runs/|/__pycache__/|\.pt$' "$LIST" | head
  exit 1
fi
if ! grep -q '^rlv3/train.py$' "$LIST"; then
  echo "REFUSING: rlv3/train.py is not in the tarball list -- the kernel would die on 'import rlv3'"; exit 1
fi
MISSING=0
while IFS= read -r f; do
  [ -e "$f" ] || { echo "listed but absent: $f"; MISSING=1; }
done < "$LIST"
[ "$MISSING" = 0 ] || { echo "REFUSING: the list names files that are not on disk"; exit 1; }

COPYFILE_DISABLE=1 tar -czf "$TARBALL" -T "$LIST"
echo "tarball: $(wc -l < "$LIST") files, $(wc -c < "$TARBALL") bytes"

# ---- push.sh's self-consistency check, with rlv3 added to the import line.
VERIFY=$(mktemp -d); tar -xzf "$TARBALL" -C "$VERIFY"
VPY=$(command -v uv >/dev/null && echo "uv run python3" || echo python3)
if ! ( cd "$VERIFY" && PYTHONPATH="$VERIFY" $VPY -c "
import compileall
assert compileall.compile_dir('kagent', quiet=2), 'kagent does not compile'
assert compileall.compile_dir('rl', quiet=2), 'rl does not compile'
assert compileall.compile_dir('rlv3', quiet=2), 'rlv3 does not compile'
import kagent.policy, kagent.scheduler, kagent.planner
import rlv3.train, rlv3.stages, rlv3.fingerprint, rlv3.manifest, rlv3.promote
import json
cfg = json.load(open('rlv3/jobs/rlv3_train.json'))
names = [s['name'] for s in cfg['stages']]
assert names, 'the job graph is empty'
rlv3.stages.load_stages('rlv3/jobs/rlv3_train.json')
print('tarball imports OK; stages', names)" ); then
  echo 'TARBALL IS NOT SELF-CONSISTENT -- the kernel would die after the mount. Not pushing.'
  rm -rf "$VERIFY"; exit 1
fi
rm -rf "$VERIFY"

if [ -n "$DRY" ]; then
  echo "DRY=1: tarball built and verified at $TARBALL; nothing uploaded."
  exit 0
fi

cat > "$STAGE/ds/dataset-metadata.json" <<META
{"title": "kaggriculture-code", "id": "gideonoba/kaggriculture-code", "licenses": [{"name": "CC0-1.0"}]}
META

if uv run kaggle datasets list --mine 2>/dev/null | grep -q "gideonoba/kaggriculture-code"; then
  uv run kaggle datasets version -p "$STAGE/ds" -m "$MSG" --dir-mode skip
else
  uv run kaggle datasets create -p "$STAGE/ds" --dir-mode skip
fi
echo "== waiting 90s for the new dataset version to process =="
sleep 90
if [ "$CODE_ONLY" = "code-only" ]; then echo "code-only: kernel not pushed"; exit 0; fi

# ---- the kernel itself goes through the EXISTING script, unmodified.
SKIP_CODE=1 KERNEL_ID="$KERNEL_ID" CONFIG="$CONFIG" META="$META" \
  tools/kaggle_job/push.sh "$MSG"
echo "poll:  uv run kaggle kernels status $KERNEL_ID"
echo "fetch: uv run kaggle kernels output $KERNEL_ID -p rl/runs/kaggle_rlv3"
