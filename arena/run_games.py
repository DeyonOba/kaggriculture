"""Batch game runner.

Usage:
    uv run python arena/run_games.py --a dev --b starter --n 50 --procs 8 \
        --out arena/results/dev_vs_starter.jsonl

Each seed is played in both orientations (a first, then b first). Records one
JSON line per game with final money, statuses, and per-hour money series.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.registry import resolve  # noqa: E402


def play_one(job):
    seed, name_a, name_b, spec_a, spec_b, swap = job
    from kaggle_environments import make

    if swap:
        specs, names = [spec_b, spec_a], [name_b, name_a]
    else:
        specs, names = [spec_a, spec_b], [name_a, name_b]

    t0 = time.time()
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run(specs)
    duration = time.time() - t0

    final = env.steps[-1]
    # Player 0's observation carries the shared state (farms) every step.
    money_series = [
        [step[0].observation["farms"][p]["money"] for p in (0, 1)]
        for step in env.steps
        if step[0].observation["hour"] == 0 or step is env.steps[-1]
    ]
    record = {
        "seed": seed,
        "swap": swap,
        "players": names,
        "rewards": [s.reward for s in final],
        "statuses": [s.status for s in final],
        "duration_s": round(duration, 2),
        "money_hourly": money_series,
    }
    # Re-key by agent name for easy aggregation.
    record["final"] = {names[i]: final[i].reward for i in range(2)}
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=20, help="number of seeds")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-swap", action="store_true", help="only play a as player 0")
    args = ap.parse_args()

    spec_a, spec_b = resolve(args.a), resolve(args.b)
    out_path = args.out or f"arena/results/{args.a}_vs_{args.b}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    jobs = []
    for i in range(args.n):
        seed = args.base_seed + i
        jobs.append((seed, args.a, args.b, spec_a, spec_b, False))
        if not args.no_swap:
            jobs.append((seed, args.a, args.b, spec_a, spec_b, True))

    t0 = time.time()
    results = []
    if args.procs <= 1:
        for job in jobs:
            results.append(play_one(job))
            print(f"  seed={results[-1]['seed']} swap={results[-1]['swap']} final={results[-1]['final']}")
    else:
        with Pool(args.procs) as pool:
            for rec in pool.imap_unordered(play_one, jobs):
                results.append(rec)
                print(f"  seed={rec['seed']} swap={rec['swap']} final={rec['final']}")

    with open(out_path, "w") as f:
        for rec in sorted(results, key=lambda r: (r["seed"], r["swap"])):
            f.write(json.dumps(rec) + "\n")
    print(f"\n{len(results)} games in {time.time() - t0:.1f}s -> {out_path}")

    from arena.stats import summarize

    summarize(out_path)


if __name__ == "__main__":
    main()
