#!/usr/bin/env bash
# Pre-flight + submit.
#
#   tools/submit.sh "message"                  # build kagent/ -> main.py (planner lineage)
#   tools/submit.sh "message" tape-sel         # submit a FROZEN bundle from arena/agents/
#
# Why the second form exists: build_submission.py bundles kagent/, which is the
# champ-h8 PLANNER lineage (rated ~810). Our best agent by a wide margin is the
# frozen tape selector `tape-sel` (rated 1750.3 / 1688.9 on two runs; ladder win
# 0.841 vs champ-h8's 0.319). Submitting that one with this script previously
# meant hand-copying a file, and the default path silently ships the weaker
# lineage.
set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${1:?usage: tools/submit.sh \"message\" [agent-name]}"
AGENT="${2:-}"

# TESTS RUN FIRST, AND THIS ORDER IS LEAD-LINED.
# tests/test_bundle.py and tests/test_rl_bundle.py both invoke
# tools/build_submission.py, which OVERWRITES main.py with the kagent planner
# bundle. With the tests after the copy, this script validated tape-sel and
# then submitted the regenerated kagent bundle -- twice, scoring 799.8 and
# 690.1 against tape-sel's true 1688.9. The pre-flight passed on a file that no
# longer existed by submit time, which is worse than no pre-flight at all.
echo "== tests (these regenerate main.py, so they must run BEFORE the copy) =="
uv run pytest tests/ -q

if [ -n "$AGENT" ]; then
  SRC="arena/agents/$AGENT/main.py"
  [ -f "$SRC" ] || { echo "no frozen bundle at $SRC"; exit 1; }
  echo "== frozen bundle: $AGENT =="
  cp "$SRC" main.py
else
  echo "== build (kagent/ planner lineage) =="
  uv run python tools/build_submission.py
  SRC="main.py"
fi

echo "== size (limit 100 MiB) =="
python3 -c "import os;n=os.path.getsize('main.py');print(f'  {n:,} bytes = {n/1048576:.3f} MiB');exit(1 if n>100*1024*1024 else 0)"

echo "== self-containment: no repo imports, no file/path access =="
python3 - <<'PY'
import re, sys
src = open("main.py").read()
bad = [p for p in ("kagent", "arena.", "tools.", "__file__", "sys.path",
                   "os.getcwd", "open(") if p in src]
print("  suspicious tokens:", bad or "none")
sys.exit(1 if bad else 0)
PY

echo "== isolation smoke: run from a FOREIGN cwd, repo off sys.path =="
# uv run, not python3: sys.executable must be the venv interpreter that has
# kaggle-environments, since the subprocess below inherits it.
uv run python - <<'PY'
import os, shutil, subprocess, sys, tempfile
d = tempfile.mkdtemp()
shutil.copy("main.py", os.path.join(d, "main.py"))
repo = os.path.abspath(".")
code = f'''
import os, sys
os.chdir({d!r})
# Strip the REPO ROOT so kagent/tools/arena are not importable -- but keep
# site-packages, which lives under the repo at .venv/ and holds
# kaggle-environments. Filtering on the string "kaggriculture" removes both.
_repo = {repo!r}
sys.path = [p for p in sys.path
            if not (os.path.abspath(p) == _repo
                    or (os.path.abspath(p).startswith(_repo + os.sep)
                        and ".venv" not in p))]
from kaggle_environments import make
p = {os.path.join(d, "main.py")!r}
env = make("kaggriculture", configuration={{"seed": 71000}}, debug=True)
env.run([p, p])
r = [env.steps[-1][i]["reward"] for i in (0, 1)]
s = [env.steps[-1][i]["status"] for i in (0, 1)]
print("  statuses", s, "rewards", r)
# A bundle that raises on import is swallowed by kaggle-environments and
# no-ops every turn, which scores exactly $3,000 and looks valid.
assert all(x == "DONE" for x in s), "agent did not finish"
assert 3000 not in r, "DEAD BUNDLE: scored exactly $3,000"
print("  isolation OK")
'''
sys.exit(subprocess.run([sys.executable, "-c", code], cwd=d).returncode)
PY

echo "== FINGERPRINT: main.py must still be the file we validated =="
python3 - "$SRC" <<'PY'
import hashlib, sys, os
src = sys.argv[1]
a = hashlib.sha256(open("main.py", "rb").read()).hexdigest()
b = hashlib.sha256(open(src, "rb").read()).hexdigest()
print(f"  main.py  {os.path.getsize('main.py'):>8,} bytes  sha256 {a[:16]}")
print(f"  {src:<9}{os.path.getsize(src):>8,} bytes  sha256 {b[:16]}")
if a != b:
    print("  MISMATCH: something rewrote main.py after the copy. REFUSING.")
    sys.exit(1)
print("  match")
PY

echo "== quota (5/day, only the most recent 2 are ACTIVE) =="
uv run kaggle competitions submissions kaggriculture 2>/dev/null | sed -n '3,4p' \
  | awk '{print "  currently active: " $1 "  " $3}'
TODAY=$(date +%Y-%m-%d)
N=$(uv run kaggle competitions submissions kaggriculture 2>/dev/null | grep -c "$TODAY" || true)
echo "  submitted today: $N of 5"

echo "== submit =="
uv run kaggle competitions submit kaggriculture -f main.py -m "$MSG"

echo "== status =="
uv run kaggle competitions submissions kaggriculture | head -5
