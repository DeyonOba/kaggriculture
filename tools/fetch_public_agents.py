"""Fetch public competition notebooks and install their agents as arena opponents.

The public Kaggle notebooks are a far harsher benchmark than self-play: most of
the highly-voted ones are replay tapes of 2500+ ladder players (sometimes with a
small online router on top), so beating them is the actual target. Their code is
other people's work and is NOT committed here (arena/agents/pub-* is gitignored);
this script pulls it on demand.

    uv run python tools/fetch_public_agents.py            # default set
    uv run python tools/fetch_public_agents.py --list     # top kernels by votes
    uv run python tools/fetch_public_agents.py --ref owner/kernel-slug --name pub-foo

Then:  PYTHONPATH=. uv run python arena/run_games.py --a champ-h7 --b pub-ahmed-v38 --n 12
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

# Kernels that embed a runnable agent via `%%writefile`. Notebooks that only
# build a C++ .so (yhay81) or only analyse data are not listed: they cannot be
# replayed locally from the notebook alone.
DEFAULT = {
    "pub-ahmed-v38": "ahmedberatozer/kaggriculture-v38-smarter-feed-stronger-margins",
    "pub-boatlee-v16": "boatlee/v16-rc5-high-score-8c-4s-premium-market-lead",
    "pub-tschinkel-psr": "thomastschinkel/kaggriculture-93-8-win-rate-public-state-router",
    "pub-salem-2900": "salemali7/kaggriculture-2900",
}


def _kaggle(*args, **kw):
    return subprocess.run(["uv", "run", "kaggle", *args], capture_output=True, text=True, **kw)


def list_kernels(n):
    out = _kaggle("kernels", "list", "--competition", "kaggriculture",
                  "--sort-by", "voteCount", "--page-size", str(n), "-v")
    print(out.stdout or out.stderr)


def fetch(name, ref):
    tmp = tempfile.mkdtemp(prefix="kagg-kernel-")
    try:
        r = _kaggle("kernels", "pull", ref, "-p", tmp)
        nbs = [f for f in os.listdir(tmp) if f.endswith(".ipynb")]
        if not nbs:
            print(f"  {name:<20} FAILED: no notebook pulled ({(r.stderr or r.stdout).strip()[:120]})")
            return False
        nb = json.load(open(os.path.join(tmp, nbs[0])))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source") or [])
            m = re.match(r"\s*%%writefile\s+(\S+)\n", src)
            if not m:
                continue
            body = src[m.end():]
            try:
                compile(body, m.group(1), "exec")
            except SyntaxError as e:
                print(f"  {name:<20} FAILED: {m.group(1)} does not compile ({e})")
                return False
            d = os.path.join(AGENTS, name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "main.py"), "w") as f:
                f.write(body)
            with open(os.path.join(d, "SOURCE.txt"), "w") as f:
                f.write(f"https://www.kaggle.com/code/{ref}\n"
                        f"fetched by tools/fetch_public_agents.py; not committed\n")
            print(f"  {name:<20} ok  ({len(body):,} bytes from {m.group(1)})")
            return True
        print(f"  {name:<20} FAILED: no %%writefile cell (agent is not inline in the notebook)")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list top kernels by votes and exit")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--ref", default=None, help="owner/kernel-slug to fetch")
    ap.add_argument("--name", default=None, help="arena agent name for --ref")
    args = ap.parse_args()

    if args.list:
        list_kernels(args.top)
        return
    if args.ref:
        if not args.name:
            sys.exit("--ref requires --name")
        sys.exit(0 if fetch(args.name, args.ref) else 1)

    print(f"installing {len(DEFAULT)} public agents into arena/agents/")
    ok = sum(fetch(n, r) for n, r in sorted(DEFAULT.items()))
    print(f"{ok}/{len(DEFAULT)} installed")


if __name__ == "__main__":
    main()
