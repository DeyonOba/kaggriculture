"""Reduce harvested replays into SESSION files for session-replay training:
{seed, both sides' full action scripts, finals, teams}. With the true seed the
local engine reproduces the world from turn 0; a script replays one recorded
side while our policy takes the other seat.

Usage:
    uv run python tools/build_sessions.py --replays replays/top replays/v9field replays/v7field \
        --min-final 15000 --out rl/datasets/sessions
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", nargs="+", default=["replays/top"])
    ap.add_argument("--min-final", type=float, default=15000.0)
    ap.add_argument("--out", default="rl/datasets/sessions")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kept = skipped = 0
    for d in args.replays:
        for path in sorted(glob.glob(os.path.join(d, "episode-*-replay.json"))):
            eid = path.split("-")[1]
            out_path = os.path.join(args.out, f"session-{eid}.json")
            if os.path.exists(out_path):
                continue
            try:
                r = json.load(open(path))
            except Exception:
                continue
            info = r.get("info") or {}
            seed = info.get("seed")
            steps = r["steps"]
            finals = [s.get("reward") or 0 for s in steps[-1]]
            if seed is None or min(finals) < args.min_final:
                skipped += 1
                continue
            # Alignment: steps[t]["action"] is the action that PRODUCED state
            # t, i.e. the decision taken while observing state t-1. The action
            # to replay at local step t is therefore steps[t+1]["action"].
            actions = [[], []]
            for t in range(len(steps) - 1):
                for side in (0, 1):
                    actions[side].append(steps[t + 1][side].get("action") or {})
            session = {
                "episode": int(eid),
                "seed": int(seed),
                "teams": info.get("TeamNames"),
                "finals": finals,
                "shops": steps[-1][0]["observation"]["town"]["unlocked_shops"],
                "actions": actions,
            }
            with open(out_path, "w") as f:
                json.dump(session, f)
            kept += 1
    print(f"sessions written: {kept} (skipped {skipped} low-quality) -> {args.out}")


if __name__ == "__main__":
    main()
