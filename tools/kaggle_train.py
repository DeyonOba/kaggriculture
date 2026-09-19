"""In-kernel pipeline: daily replays -> action-only sessions -> tensors -> BoardNet.

Runs on Kaggle so the 21.5 GB/day dumps never cross the user's connection.
The daily datasets mount read-only under /kaggle/input; we parse them in place,
keep only the ACTIONS (~270 KB a game against ~33 MB of full replay, 120x
smaller) and emit the trained model plus the compressed sessions. The user
downloads ~1 GB instead of ~86 GB.

Checkpoints after every day and again after the dataset build, because a
Kaggle job that dies at the 11 h wall with nothing written returns nothing --
that has happened on this project before.

    steps in tools/kaggle_job/job_train.json; push with
      KERNEL_ID=gideonoba/kaggriculture-train CONFIG=tools/kaggle_job/job_train.json \
        tools/kaggle_job/push.sh "train boardnet on daily dumps"
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

IN = "/kaggle/input"
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
SESS = os.path.join(REPO, "rl/datasets/sessions")


def convert(path):
    """One full replay -> our session dict, or None."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    steps = d.get("steps") or []
    if not steps or len(steps[0]) < 2:
        return None
    acts = [[], []]
    for st in steps:
        for i in (0, 1):
            acts[i].append(st[i].get("action"))
    info = d.get("info") or {}
    seed = (d.get("configuration") or {}).get("seed")
    if seed is None:
        return None
    return {
        "episode": d.get("id"),
        "seed": seed,
        "teams": info.get("TeamNames"),
        "finals": [s.get("reward") for s in steps[-1][:2]],
        "shops": ((steps[-1][0].get("observation") or {}).get("town") or {})
                 .get("unlocked_shops"),
        "actions": acts,
        "module_version": d.get("version"),
        "scores": info.get("LiveScores") or info.get("TeamScores"),
        "score_source": ["daily-dump", "daily-dump"],
    }


def harvest(budget_s):
    """Convert every mounted daily dataset into rl/datasets/sessions/."""
    os.makedirs(SESS, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(SESS, "*.json"))}
    print(f"  corpus starts at {len(have):,} sessions", flush=True)
    dirs = sorted(d for d in glob.glob(os.path.join(IN, "*"))
                  if "episodes-2" in os.path.basename(d))
    if not dirs:
        print("  no daily episode datasets mounted; skipping harvest", flush=True)
        return
    t0 = time.time()
    for d in dirs:
        n = 0
        files = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
        print(f"  {os.path.basename(d)}: {len(files):,} files", flush=True)
        for p in files:
            if time.time() - t0 > budget_s:
                print("  harvest budget spent, stopping early", flush=True)
                break
            s = convert(p)
            if not s or not s.get("episode"):
                continue
            name = f"session-{s['episode']}.json"
            if name in have:
                continue
            with open(os.path.join(SESS, name), "w") as f:
                json.dump(s, f)
            have.add(name)
            n += 1
        print(f"    +{n:,} new  (corpus {len(have):,})", flush=True)
        # checkpoint the growing corpus after EVERY day
        pack(os.path.join(WORK, "sessions_new.jsonl.gz"), have)


def pack(out, names):
    """Compact the corpus into one gzipped jsonl so the user downloads once."""
    tmp = out + ".part"
    with gzip.open(tmp, "wt") as z:
        for nm in sorted(names):
            p = os.path.join(SESS, nm)
            try:
                with open(p) as f:
                    z.write(f.read().replace("\n", "") + "\n")
            except Exception:
                continue
    os.replace(tmp, out)
    print(f"    checkpoint -> {out} "
          f"({os.path.getsize(out)/1e6:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--min-rating", type=float, default=0.0,
                    help="0 = the natural distribution, which is what the gate "
                         "faces; see kaggriculture-train-on-deployment-distribution")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--harvest-budget-s", type=int, default=9000)
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="harvest only: convert the mounted day to\n                          action-only sessions and stop. One day per\n                          kernel, because Kaggle silently dropped 64 GB\n                          of inputs when three were attached.")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    if not args.skip_harvest:
        print("== harvest ==", flush=True)
        harvest(args.harvest_budget_s)

    if args.skip_train:
        for f in sorted(glob.glob(os.path.join(WORK, "*"))):
            print(f"  output: {f}  {os.path.getsize(f)/1e6:.1f} MB", flush=True)
        return

    print("\n== index ==", flush=True)
    from tools.opp_coupling import build_index
    idx = build_index()
    scored = [v for v in idx.values()
              if v["scores"] and all(x is not None for x in v["scores"])]
    print(f"  {len(scored):,} of {len(idx):,} sessions carry both scores",
          flush=True)

    print("\n== build tensors ==", flush=True)
    import subprocess
    subprocess.run([sys.executable, "-u", "tools/s3_flow.py", "--build",
                    "--n", str(args.games), "--min-rating",
                    str(args.min_rating), "--procs", str(args.procs)],
                   cwd=REPO, check=True)

    print("\n== train ==", flush=True)
    subprocess.run([sys.executable, "-u", "tools/s3_flow.py", "--train",
                    "--arm", "E_full_83", "--epochs", str(args.epochs),
                    "--out", os.path.join(WORK, "boardnet_daily.npz")],
                   cwd=REPO, check=True)

    print("\n== out-of-distribution check ==", flush=True)
    subprocess.run([sys.executable, "-u", "tools/net_ood.py", "--model",
                    os.path.join(WORK, "boardnet_daily.npz"),
                    "--arm", "E_full_83", "--n", "12",
                    "--procs", str(args.procs)], cwd=REPO, check=False)

    for f in sorted(glob.glob(os.path.join(WORK, "*"))):
        print(f"  output: {f}  {os.path.getsize(f)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
