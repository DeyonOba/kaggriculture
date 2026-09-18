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
matter how many days are requested.

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
    """One full replay -> our session dict, or None if it is unusable."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    steps = d.get("steps") or []
    if not steps:
        return None
    n = len(steps[0])
    acts = [[], []]
    for st in steps:
        for i in range(min(n, 2)):
            acts[i].append(st[i].get("action"))
    info = d.get("info") or {}
    return {
        "episode": d.get("id"),
        "seed": (d.get("configuration") or {}).get("seed"),
        "teams": info.get("TeamNames"),
        "finals": [s.get("reward") for s in steps[-1][:2]],
        "shops": (steps[-1][0].get("observation") or {}).get("town", {})
                 .get("unlocked_shops"),
        "actions": acts,
        "module_version": d.get("version"),
        "scores": info.get("LiveScores") or info.get("TeamScores"),
        "score_source": ["daily-dump", "daily-dump"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--last", type=int, default=0)
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
        gb = int(r["total_bytes"]) / 1e9
        print(f"  {day}: {r['episode_count']} episodes, {gb:.1f} GB, "
              f"median avg score {float(r['median_avg_score']):.0f} ...",
              flush=True)
        try:
            subprocess.run(["uv", "run", "kaggle", "datasets", "download", slug,
                            "-p", work, "--unzip"], check=True, cwd=REPO)
        except subprocess.CalledProcessError as e:
            print(f"    download failed: {e}", flush=True)
            shutil.rmtree(work, ignore_errors=True)
            continue
        n = 0
        for p in glob.glob(os.path.join(work, "**", "*.json"), recursive=True):
            s = convert(p)
            if not s or s.get("seed") is None or not s.get("episode"):
                continue
            name = f"session-{s['episode']}.json"
            if name in have:
                continue
            json.dump(s, open(os.path.join(SESS, name), "w"))
            have.add(name)
            n += 1
        total_new += n
        # free the 21 GB before the next day
        shutil.rmtree(work, ignore_errors=True)
        print(f"    +{n:,} new sessions (raw dump deleted)", flush=True)

    print(f"\n  {total_new:,} new sessions; corpus now "
          f"{len(have):,} files", flush=True)
    print("  rebuild the index:  rm rl/datasets/session_index.json && "
          "PYTHONPATH=. uv run python tools/opp_coupling.py --n 1", flush=True)


if __name__ == "__main__":
    main()
