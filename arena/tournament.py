"""Round-robin tournament across registry agents.

Usage:
    uv run python arena/tournament.py v1 v2 v3 starter --n 5
"""

import argparse
import itertools
import os
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.registry import resolve  # noqa: E402
from arena.run_games import play_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--n", type=int, default=5, help="seeds per pairing")
    ap.add_argument("--base-seed", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    jobs = []
    for a, b in itertools.combinations(args.agents, 2):
        sa, sb = resolve(a), resolve(b)
        for i in range(args.n):
            seed = args.base_seed + i
            jobs.append((seed, a, b, sa, sb, False))
            jobs.append((seed, a, b, sa, sb, True))

    with Pool(args.procs) as pool:
        results = list(pool.imap_unordered(play_one, jobs))

    money = {a: [] for a in args.agents}
    wins = {a: 0 for a in args.agents}
    games = {a: 0 for a in args.agents}
    for rec in results:
        n0, n1 = rec["players"]
        r0, r1 = rec["rewards"]
        if r0 is None or r1 is None:
            print(f"!! bad game {rec['seed']} {rec['players']} {rec['statuses']}")
            continue
        money[n0].append(r0)
        money[n1].append(r1)
        games[n0] += 1
        games[n1] += 1
        if r0 > r1:
            wins[n0] += 1
        elif r1 > r0:
            wins[n1] += 1

    print(f"\n=== tournament: {args.agents} n={args.n} seeds/pairing ===")
    ranking = sorted(args.agents, key=lambda a: -statistics.mean(money[a]) if money[a] else 0)
    for a in ranking:
        vals = money[a]
        print(
            f"{a:>10}: mean={statistics.mean(vals):9.1f}  median={statistics.median(vals):9.1f}  "
            f"wins={wins[a]}/{games[a]}"
        )


if __name__ == "__main__":
    main()
