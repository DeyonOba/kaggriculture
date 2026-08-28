"""Poll Kaggle for submission status, then list episodes.

Usage:
    uv run python tools/watch_submission.py            # latest submission status
    uv run python tools/watch_submission.py --episodes <SUBMISSION_ID>
    uv run python tools/watch_submission.py --replay <EPISODE_ID>
"""

import argparse
import subprocess
import time


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None, help="submission id to list episodes for")
    ap.add_argument("--replay", default=None, help="episode id to download replay + logs for")
    ap.add_argument("--watch", action="store_true", help="poll until submission leaves pending")
    args = ap.parse_args()

    if args.replay:
        print(run(f"uv run kaggle competitions replay {args.replay} -p arena/results/replays"))
        print(run(f"uv run kaggle competitions logs {args.replay} 0 -p arena/results/logs"))
        print(run(f"uv run kaggle competitions logs {args.replay} 1 -p arena/results/logs"))
        return
    if args.episodes:
        print(run(f"uv run kaggle competitions episodes {args.episodes}"))
        return

    while True:
        out = run("uv run kaggle competitions submissions kaggriculture")
        print(out.splitlines()[0] if out else "(no output)")
        for line in out.splitlines()[:6]:
            print(line)
        if not args.watch or "pending" not in out.lower():
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
