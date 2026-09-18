"""Round-robin: every agent against every other, paired on identical worlds.

The ladder gate scores each candidate against ONE base on recorded-seat
opponents. That answers "is this better than champ-h8" and never answers "which
of these beats which", and it never puts two of OUR agents in the same market.

That second point is the interesting one. Both seats sell into one shared
`market["inventory"]`, so two agents with the same strategy crowd each other in
a way neither does against a recorded tape. A head that looks good against a
fixed opponent can be self-defeating in a mirror.

Each pair plays every seed BOTH WAYS -- (a as seat 0, b as seat 1) and the
reverse -- so seat effects cancel exactly.

    PYTHONPATH=. uv run python tools/round_robin.py --agents tape-sel r-mrg100 \
        r-net-e r-glut champ-h8 a-lp100 --n 40
"""

import argparse
import itertools
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(spec):
    a, b, seed = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(n):
        g = {}
        exec(open(os.path.join(REPO, "arena", "agents", n, "main.py")).read(), g)
        return g["agent"]

    fa, fb = load(a), load(b)
    out = []
    for order in (0, 1):
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run([fa, fb] if order == 0 else [fb, fa])
        r = [env.steps[-1][i].reward or 0.0 for i in (0, 1)]
        ra, rb = (r[0], r[1]) if order == 0 else (r[1], r[0])
        out.append((ra, rb))
    return (a, b, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    seeds = [300_000 + rng.randrange(10**6) for _ in range(args.n)]
    pairs = list(itertools.combinations(args.agents, 2))
    specs = [(a, b, s) for a, b in pairs for s in seeds]
    print(f"{len(args.agents)} agents, {len(pairs)} pairs, {args.n} seeds, "
          f"both seat orders = {2*len(specs):,} games\n", flush=True)

    with mp.Pool(args.procs) as pool:
        rows = pool.map(job, specs)

    win = {a: {b: [0, 0] for b in args.agents} for a in args.agents}   # [w, n]
    marg = {a: {b: [] for b in args.agents} for a in args.agents}
    for a, b, out in rows:
        for ra, rb in out:
            win[a][b][1] += 1
            win[b][a][1] += 1
            if ra > rb:
                win[a][b][0] += 1
            elif rb > ra:
                win[b][a][0] += 1
            marg[a][b].append(ra - rb)
            marg[b][a].append(rb - ra)

    print("WIN RATE, row agent vs column agent (a win is simply more money):\n")
    w = max(len(x) for x in args.agents) + 1
    print(" " * w + "".join(f"{b[:10]:>11}" for b in args.agents) + f"{'overall':>10}")
    order = []
    for a in args.agents:
        cells, tw, tn = [], 0, 0
        for b in args.agents:
            if a == b:
                cells.append(f"{'-':>11}")
                continue
            ww, nn = win[a][b]
            tw += ww
            tn += nn
            cells.append(f"{ww/max(1,nn):>11.2f}")
        order.append((tw / max(1, tn), a))
        print(f"{a:<{w}}" + "".join(cells) + f"{tw/max(1,tn):>10.2f}")

    print("\nMEAN MARGIN ($ row minus column) -- the quantity a win is made of:\n")
    print(" " * w + "".join(f"{b[:10]:>11}" for b in args.agents))
    for a in args.agents:
        cells = []
        for b in args.agents:
            if a == b:
                cells.append(f"{'-':>11}")
                continue
            v = marg[a][b]
            cells.append(f"{sum(v)/max(1,len(v)):>+11,.0f}")
        print(f"{a:<{w}}" + "".join(cells))

    print("\nRANKING by overall win rate:")
    for i, (r, a) in enumerate(sorted(order, reverse=True), 1):
        print(f"  {i}. {a:<12}{r:.3f}")


if __name__ == "__main__":
    main()
