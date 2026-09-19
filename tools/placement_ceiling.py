"""How much of our walking is bad ROUTING, and how much is bad PLACEMENT?

The executor gradient is real -- moves per work action runs 1.20 -> 0.91 with
rating and champ-h8 sits at 1.43 -- but the first attempt to close it
(tools/make_haul_variant.py) gated -$12,026. Before building a placement
model, bound what placement could possibly buy.

Each unit-day's movement decomposes exactly:

    actual moves
      = SUM over consecutive worked tiles of Manhattan(prev -> next)   [A]
      + everything else                                                [B]

    [A] is the INHERENT SPREAD of the work the planner asked for. Only moving
        the WORK can shrink it -- that is the placement lever.
    [B] is walking beyond the straight-line need: wrong unit sent, detours,
        re-visits, wandering. That is the routing lever, and it is what
        eleven prior experiments attacked.

A perfectly clustered plan would put every next task adjacent to the last, so
the floor on [A] is one step per work action. The gap between [A] and that
floor is the placement ceiling IN TURNS.

Converting turns to money is the weak link, and the honest conversion is
already measured: h-front9 added two hands (~1,440 extra actions a season) and
scored -$653; h-front11 added four and scored -$1,370. Extra capacity in this
agent is worth ZERO OR LESS, because work is underfoot 94.9% of the time.

    PYTHONPATH=. uv run python tools/placement_ceiling.py --agents champ-h8 r-mrg100
"""

import argparse
import collections
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def run(name, opp_name, seeds):
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(n):
        g = {}
        exec(open(os.path.join(REPO, "arena", "agents", n, "main.py")).read(), g)
        return g["agent"]

    a, opp = load(name), load(opp_name)
    tot = collections.Counter()
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.reset(2)
        # per unit index: last tile it did WORK on, this day
        last = {}
        day = 0
        while not env.done:
            o = env.state[0].observation
            d = o["step"] // 24
            if d != day:
                last.clear()
                day = d
            farm = o["farms"][0]
            pos = [tuple(farm.get("farmer") or (0, 0))]
            pos += [tuple(h) for h in (farm.get("hands") or [])]
            act = a(o)
            ops = [act.get("farmer")] + list(act.get("hands") or [])
            for i, op in enumerate(ops):
                if not op or i >= len(pos):
                    continue
                if op[0] in MOVES:
                    tot["moves"] += 1
                elif op[0] == "PASS":
                    tot["pass"] += 1
                else:
                    tot["work"] += 1
                    p = pos[i]
                    if i in last:
                        tot["span"] += abs(p[0] - last[i][0]) + abs(p[1] - last[i][1])
                        tot["hops"] += 1
                    last[i] = p
            env.step([act, opp(env.state[1].observation)])
    n = len(seeds)
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seeds = [300_000 + rng.randrange(10**6) for _ in range(args.n)]
    print(f"{args.n} games vs {args.opp}, per game\n")
    print(f"    {'agent':<12}{'work':>7}{'moves':>7}{'[A] spread':>12}"
          f"{'[B] routing':>13}{'B share':>9}{'A floor':>9}{'placement':>11}")
    for nm in args.agents:
        r = run(nm, args.opp, seeds)
        A = r.get("span", 0.0)
        B = max(0.0, r["moves"] - A)
        floor = r.get("hops", 0.0)          # one step per consecutive work pair
        place = max(0.0, A - floor)
        print(f"    {nm:<12}{r['work']:>7.0f}{r['moves']:>7.0f}{A:>12.0f}"
              f"{B:>13.0f}{100*B/max(1,r['moves']):>8.0f}%{floor:>9.0f}"
              f"{place:>11.0f}")
    print("\n  [A] spread   = sum of Manhattan hops between CONSECUTIVE worked")
    print("                 tiles -- the inherent spread of the plan")
    print("  [B] routing  = moves beyond that straight-line need")
    print("  placement    = [A] minus a perfectly clustered floor of 1 step")
    print("                 per hop; the most a placement model could remove")
    print("\n  CONVERSION, already measured: h-front9 added ~1,440 actions a")
    print("  season for -$653 and h-front11 ~2,880 for -$1,370. Freed turns in")
    print("  this agent convert at ZERO OR LESS, because work is underfoot")
    print("  94.9% of the time -- so turns are the wrong currency for the bound.")


if __name__ == "__main__":
    main()
