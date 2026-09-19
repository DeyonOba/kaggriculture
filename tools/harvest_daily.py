"""Harvest daily episode dumps into our action-only session format.

PREFER tools/kaggle_train.py. This downloads 21.5 GB per day OVER YOUR
CONNECTION to extract ~175 MB of actions -- 86 GB for four days. The Kaggle
kernel mounts the same datasets read-only and converts them in place, so the
only thing that crosses the wire is the ~1 GB result. Use this locally only
when a kernel is not an option.

The daily datasets are FULL replays with every state: ~21.5 GB for ~650
episodes, i.e. ~33 MB each. Our pipeline only ever needs the ACTIONS -- every
tool replays them locally against the seed -- and a session file is ~270 KB.
That is a 120x reduction, so this downloads one day, converts, and DELETES the
raw dump before taking the next. Disk stays bounded at one day (~22 GB) no
matter how many days are requested, and `--gb-per-day` caps what is pulled:
files are fetched ONE AT A TIME and deleted straight after conversion, so peak
disk is a single 33 MB episode and the wire cost is whatever budget you set.

    PYTHONPATH=. uv run python tools/harvest_daily.py --days 2026-09-17 2026-09-16
    PYTHONPATH=. uv run python tools/harvest_daily.py --last 3
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = os.path.join(REPO, "rl/datasets/sessions")
MANIFEST = os.path.join(REPO, "rl/datasets/manifest.csv")


def fetch_manifest():
    d = os.path.dirname(MANIFEST)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["uv", "run", "kaggle", "datasets", "download",
                    "kaggle/kaggriculture-episodes-index", "-f", "manifest.csv",
                    "-p", d, "--unzip", "--force"], check=True, cwd=REPO)
    return list(csv.DictReader(open(MANIFEST)))


def convert(path):
    """One daily-dump replay -> our action-only session dict, or None.

    Daily dumps differ from the older session format: `id` is a UUID and
    `configuration.seed` is null. The real identifiers live under `info`
    (`EpisodeId`, `seed`), and `rewards` carries the finals. Reading the old
    fields silently returns None for every file, which is what made the first
    local harvest convert 0 of 6.
    """
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    info = d.get("info") or {}
    steps = d.get("steps") or []
    if not steps or len(steps[0]) < 2:
        return None
    ep = info.get("EpisodeId") or d.get("id")
    seed = info.get("seed")
    if seed is None:
        seed = (d.get("configuration") or {}).get("seed")
    if seed is None or ep is None:
        return None
    acts = [[], []]
    for st in steps:
        for i in (0, 1):
            acts[i].append(st[i].get("action"))
    return {
        "episode": ep,
        "seed": seed,
        "teams": info.get("TeamNames"),
        "finals": d.get("rewards") or [s.get("reward") for s in steps[-1][:2]],
        "shops": ((steps[-1][0].get("observation") or {}).get("town") or {})
                 .get("unlocked_shops"),
        "actions": acts,
        "module_version": d.get("module_version") or d.get("version"),
        "scores": info.get("LiveScores") or info.get("TeamScores"),
        "score_source": ["daily-dump", "daily-dump"],
    }

def list_files(slug):
    """-> [(name, bytes)] for one daily dataset, following pagination."""
    out, token = [], None
    for _ in range(40):
        cmd = ["uv", "run", "kaggle", "datasets", "files", slug, "--csv"]
        if token:
            cmd += ["--page-token", token]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        token = None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("Next Page Token = "):
                token = line.split("= ", 1)[1].strip()
                continue
            parts = line.split(",")
            if len(parts) >= 2 and parts[0].endswith(".json"):
                try:
                    out.append((parts[0], int(parts[1])))
                except ValueError:
                    pass
        if not token:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--gb-per-day", type=float, default=3.0,
                    help="byte budget per day; each episode is ~33 MB, so 3 GB "
                         "is ~90 episodes. Files are fetched ONE AT A TIME and "
                         "deleted after conversion, so peak disk is one file.")
    ap.add_argument("--work", default="/tmp/kaggri_daily")
    args = ap.parse_args()

    rows = fetch_manifest()
    days = args.days or [r["date"] for r in rows[-args.last:]] if (
        args.days or args.last) else []
    if not days:
        raise SystemExit("give --days or --last")
    by = {r["date"]: r for r in rows}
    os.makedirs(SESS, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(SESS, "*.json"))}
    budget = int(args.gb_per_day * 1e9)
    total_new = 0

    for day in days:
        r = by.get(day)
        if not r:
            print(f"  {day}: not in manifest", flush=True)
            continue
        slug = "kaggle/" + r["daily_dataset_slug"]
        work = os.path.join(args.work, day)
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        files = list_files(slug)
        print(f"  {day}: {len(files)} files listed, budget "
              f"{args.gb_per_day:.1f} GB, median avg score "
              f"{float(r['median_avg_score']):.0f}", flush=True)
        spent, n = 0, 0
        for name, size in files:
            if spent + size > budget:
                break
            try:
                subprocess.run(["uv", "run", "kaggle", "datasets", "download",
                                slug, "-f", name, "-p", work, "--unzip",
                                "--force", "-q"], check=True, cwd=REPO)
            except subprocess.CalledProcessError:
                continue
            spent += size
            for p in glob.glob(os.path.join(work, "*.json")):
                s = convert(p)
                os.remove(p)                    # delete immediately
                if not s or s.get("seed") is None or not s.get("episode"):
                    continue
                nm = f"session-{s['episode']}.json"
                if nm in have:
                    continue
                json.dump(s, open(os.path.join(SESS, nm), "w"))
                have.add(nm)
                n += 1
        shutil.rmtree(work, ignore_errors=True)
        total_new += n
        print(f"    +{n:,} new sessions from {spent/1e9:.1f} GB "
              f"(corpus {len(have):,})", flush=True)

    print(f"\n  {total_new:,} new sessions; corpus now {len(have):,} files",
          flush=True)
    print("  rebuild the index: rm rl/datasets/session_index.json && "
          "PYTHONPATH=. uv run python tools/opp_coupling.py --n 1", flush=True)


if __name__ == "__main__":
    main()
