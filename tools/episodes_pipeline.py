"""Data management for the official daily episode datasets
(kaggle/kaggriculture-episodes-index -> kaggle/kaggriculture-episodes-<date>,
~21 GB per day). Nothing raw is ever stored: each chunk of days is MOUNTED
into a Kaggle kernel (mounts cost no quota), reduced to sessions by
tools/extract_episodes.py, and only the compact output (~15 MB/day) comes
back. A ledger records processed days so runs resume.

    PYTHONPATH=. uv run python tools/episodes_pipeline.py plan  [--since 2026-08-15]
    PYTHONPATH=. uv run python tools/episodes_pipeline.py run   [--since ...] [--days-per-run 3] [--max-runs N]
    PYTHONPATH=. uv run python tools/episodes_pipeline.py merge <output-dir>     # fold sessions_*.jsonl.gz into rl/datasets/sessions

After `run`, publish the enlarged corpus with `tools/data.py push`.
Train/test by rating bin is unchanged: sessions carry per-seat `scores`;
rl/session_bench.py ladder --heldout and rl/bandit.py select by bin, with
episode % 5 == 0 held out everywhere.
"""

import argparse
import csv
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DS = "kaggle/kaggriculture-episodes-index"
KERNEL = "gideonoba/kaggriculture-episodes-job"  # its own kernel: runs beside the knob job
LEDGER = os.path.join(REPO, "rl", "datasets", "episodes_ledger.json")
SESS_DIR = os.path.join(REPO, "rl", "datasets", "sessions")
JOB_DIR = os.path.join(REPO, "tools", "kaggle_job")


def sh(cmd, check=True, capture=False):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture, text=True)


def manifest():
    d = tempfile.mkdtemp(prefix="epidx-")
    sh(f"uv run kaggle datasets download {INDEX_DS} -p {d} --unzip -q")
    with open(os.path.join(d, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))
    return rows


def ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"done": {}, "runs": []}


def save_ledger(l):
    with open(LEDGER, "w") as f:
        json.dump(l, f, indent=2)


def plan(args):
    rows = manifest()
    l = ledger()
    todo = [r for r in rows if r["date"] >= args.since and r["date"] not in l["done"]]
    gb = sum(int(r["total_bytes"]) for r in todo) / 1e9
    eps = sum(int(r["episode_count"]) for r in todo)
    print(f"{len(todo)} days to process ({eps} episodes, {gb:.0f} GB mounted, ~{eps * 0.02:.0f} MB kept):")
    for r in todo:
        print(f"  {r['date']}  {r['episode_count']:>4} eps  {int(r['total_bytes']) / 1e9:5.1f} GB  top {float(r['top_avg_score']):.0f}  median {float(r['median_avg_score']):.0f}")
    return todo


def _push_chunk(days, procs):
    """Point the job kernel at these daily datasets and push a fresh version."""
    with open(os.path.join(JOB_DIR, "kernel-metadata.json")) as f:
        meta = json.load(f)
    base_sources = [s for s in meta["dataset_sources"] if not s.startswith("kaggle/kaggriculture-episodes-")]
    meta["dataset_sources"] = base_sources + [f"kaggle/{r['daily_dataset_slug']}" for r in days]
    meta["id"] = KERNEL
    meta["title"] = KERNEL.split("/")[-1]
    cfg = {
        "engine": "1.32.7", "procs": procs, "budget_s": 11 * 3600,
        "steps": [
            "{python} tools/extract_episodes.py --inputs-glob '/kaggle/input/**/kaggriculture-episodes-20*' "
            "--ratings rl/datasets/ratings/agents.csv --leaderboard rl/datasets/ratings/leaderboard.csv "
            "--out {out}/episodes --procs {procs}",
        ],
        "collect": [],
    }
    stage = tempfile.mkdtemp(prefix="epjob-")
    src = open(os.path.join(JOB_DIR, "job.py")).read().replace("CONFIG_JSON = None", "CONFIG_JSON = " + repr(json.dumps(cfg)))
    with open(os.path.join(stage, "job.py"), "w") as f:
        f.write(src)
    with open(os.path.join(stage, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    sh(f"uv run kaggle kernels push -p {stage}")


def _wait(poll=60, limit_s=12 * 3600):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        st = sh(f"uv run kaggle kernels status {KERNEL}", check=False, capture=True).stdout.strip().lower()
        if "complete" in st:
            return "complete"
        if "error" in st or "cancel" in st:
            return "error"
        time.sleep(poll)
    return "timeout"


def merge(out_dir):
    """Fold sessions_*.jsonl.gz into rl/datasets/sessions (one file per episode)."""
    os.makedirs(SESS_DIR, exist_ok=True)
    added = skipped = 0
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "sessions_*.jsonl.gz"), recursive=True)):
        with gzip.open(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                dst = os.path.join(SESS_DIR, f"session-{s['episode']}.json")
                if os.path.exists(dst):
                    skipped += 1
                    continue
                with open(dst, "w") as fo:
                    json.dump(s, fo)
                added += 1
    print(f"merged: +{added} sessions ({skipped} already present); total {len(glob.glob(os.path.join(SESS_DIR, 'session-*.json')))}")
    return added


def run(args):
    todo = plan(args)
    l = ledger()
    runs = 0
    for i in range(0, len(todo), args.days_per_run):
        if args.max_runs and runs >= args.max_runs:
            break
        chunk = todo[i: i + args.days_per_run]
        dates = [r["date"] for r in chunk]
        print(f"\n=== run {runs + 1}: {dates} ({sum(int(r['total_bytes']) for r in chunk) / 1e9:.0f} GB mounted) ===", flush=True)
        _push_chunk(chunk, args.procs)
        time.sleep(120)
        status = _wait()
        out = tempfile.mkdtemp(prefix="epout-")
        if status == "complete":
            sh(f"uv run kaggle kernels output {KERNEL} -p {out}", check=False)
            added = merge(out)
            for d in dates:
                l["done"][d] = {"sessions_added": added, "when": time.strftime("%Y-%m-%d %H:%M")}
        l["runs"].append({"dates": dates, "status": status, "output": out})
        save_ledger(l)
        print(f"run {runs + 1}: {status}", flush=True)
        runs += 1
    print("pipeline finished; publish with: PYTHONPATH=. uv run python tools/data.py push -m 'daily episodes'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "merge"])
    ap.add_argument("path", nargs="?", default=None, help="output dir for merge")
    ap.add_argument("--since", default="2026-08-15", help="engine 1.32.7 cutover")
    ap.add_argument("--days-per-run", type=int, default=3)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "plan":
        plan(args)
    elif args.cmd == "run":
        run(args)
    else:
        merge(args.path)


if __name__ == "__main__":
    main()
