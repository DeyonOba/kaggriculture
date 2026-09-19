"""Does a pickup-batching change actually MOVE the ratio it targets?

Routing attempt #7 in the archive is "bigger shed pickup batches -- null", and
attempt #11 is recorded as "mechanism does not fire" -- a null that measured
nothing. Before re-testing batching we check the mechanism, not the money.

The target, measured over 400 recorded games (800 seats):

    pickup per FEED    corpus 0.52-0.63       champ-h8  1.07
    moves per WORK     corpus 1.20 -> 0.91    champ-h8  1.43
    work per seat      corpus 2925 -> 3374    champ-h8  2762
    moves per seat     corpus 3498 -> 3075    champ-h8  3941

A candidate that does not move `pickup per feed` toward ~0.6 has not fired,
and its gate result says nothing about batching.

    PYTHONPATH=. uv run python tools/haul_probe.py --agents champ-h8 r-net2k
"""
import argparse
import collections
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
CORPUS = {"pickup_per_feed": (0.52, 0.63), "moves_per_work": (0.91, 1.20),
          "work": (2925, 3374), "moves": (3075, 3498)}


def probe(name, opp_name, seeds):
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(n):
        g = {}
        exec(open(os.path.join(REPO, "arena", "agents", n, "main.py")).read(), g)
        return g["agent"]

    a, opp = load(name), load(opp_name)
    c = collections.Counter()
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.reset(2)
        while not env.done:
            o = env.state[0].observation
            act = a(o)
            for x in [act.get("farmer")] + list(act.get("hands") or []):
                if not x:
                    continue
                op = x[0]
                if op in MOVES:
                    c["_MOVE"] += 1
                elif op == "PASS":
                    c["_PASS"] += 1
                else:
                    c[op] += 1
                    c["_WORK"] += 1
            env.step([act, opp(env.state[1].observation)])
    n = len(seeds)
    return {"work": c["_WORK"] / n, "moves": c["_MOVE"] / n,
            "pickup": c["PICKUP"] / n, "feed": c["FEED"] / n,
            "pickup_per_feed": c["PICKUP"] / max(1, c["FEED"]),
            "moves_per_work": c["_MOVE"] / max(1, c["_WORK"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seeds = [300_000 + rng.randrange(10**6) for _ in range(args.n)]
    print(f"{args.n} games vs {args.opp}\n")
    print(f"    {'agent':<14}{'work':>7}{'moves':>7}{'PICKUP':>8}{'FEED':>6}"
          f"{'pickup/feed':>13}{'moves/work':>12}")
    for nm in args.agents:
        r = probe(nm, args.opp, seeds)
        flag = "  <- fired" if r["pickup_per_feed"] < 0.85 else ""
        print(f"    {nm:<14}{r['work']:>7.0f}{r['moves']:>7.0f}{r['pickup']:>8.0f}"
              f"{r['feed']:>6.0f}{r['pickup_per_feed']:>13.2f}"
              f"{r['moves_per_work']:>12.2f}{flag}")
    lo, hi = CORPUS["pickup_per_feed"]
    print(f"\n    corpus target pickup/feed {lo}-{hi}; "
          f"a candidate above ~0.85 has NOT fired and its gate is uninformative")


if __name__ == "__main__":
    main()
