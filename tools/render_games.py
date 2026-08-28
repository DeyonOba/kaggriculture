"""Run games and save the engine's HTML replay viewer for each, so agent
strategy can be observed move by move.

Usage:
    uv run python tools/render_games.py --a rl-v3 --b v3 --n 2
    uv run python tools/render_games.py --a rl-v3 --b rl-v1 --seeds 123,456
    uv run python tools/render_games.py --a rl-v3 --b starter --n 3 --out replays

Seeds are random unless --seeds is given; every file name carries the seed so
a game can be reproduced or re-rendered later.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.registry import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=1, help="number of random-seed games")
    ap.add_argument("--seeds", default=None, help="comma-separated explicit seeds")
    ap.add_argument("--swap", action="store_true", help="also render the swapped orientation")
    ap.add_argument("--out", default="replays")
    args = ap.parse_args()

    from kaggle_environments import make

    os.makedirs(args.out, exist_ok=True)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        rng = random.SystemRandom()
        seeds = [rng.randrange(1, 10_000_000) for _ in range(args.n)]

    spec_a, spec_b = resolve(args.a), resolve(args.b)
    orientations = [(args.a, args.b, spec_a, spec_b)]
    if args.swap:
        orientations.append((args.b, args.a, spec_b, spec_a))

    for seed in seeds:
        for name0, name1, s0, s1 in orientations:
            t0 = time.time()
            env = make("kaggriculture", configuration={"seed": seed}, debug=False)
            env.run([s0, s1])
            final = env.steps[-1]
            r0, r1 = final[0].reward, final[1].reward
            html = env.render(mode="html", width=1100, height=820)
            path = os.path.join(args.out, f"{name0}_vs_{name1}_seed{seed}.html")
            with open(path, "w") as f:
                f.write(html)
            shops = env.steps[-1][0].observation["town"]["unlocked_shops"]
            print(
                f"{path}\n   {name0}: ${r0:,.0f}  {name1}: ${r1:,.0f}  "
                f"({time.time()-t0:.0f}s)  shops: {', '.join(shops) or 'none'}"
            )


if __name__ == "__main__":
    main()
