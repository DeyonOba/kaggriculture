"""What does the 10-orders-per-turn cap actually cost us?

`maxMarketOrdersPerTurn = 10` and the engine does `q[:max_orders]` -- anything
past the tenth is dropped SILENTLY, with no error and no retry. HIRE is one
order PER HAND, so hiring competes directly with selling for slots. Measured
earlier: champ-h8 saturates the cap on 21 turns a game with HIRE taking 165 of
210 contested slots.

This is the one executor decision NOT bounded by
`kaggriculture-executor-bounded`, because slots are not turns: the movement
ceiling says nothing about them, and a dropped SELL is money, not an action.

Two numbers, both oracle-style so they bound what any policy could win:

  DROPPED    every order the engine truncated, and what it was worth at the
             prices prevailing that turn -- a SELL priced through
             econ.sell_revenue, a HIRE at its fib cost, a BUY at its cost.
  DEFERRED   how many of those dropped orders the agent successfully reissued
             on a later turn. A dropped order that comes back next turn costs
             only the delay, not the value.

The gap between them is the real loss.

    PYTHONPATH=. uv run python tools/slot_cost.py --agents champ-h8 r-mrg100
"""

import argparse
import collections
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def run(name, opp_name, seeds):
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kagent import econ
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        CROPS, ANIMALS, LAND_PRICES, _hire_cost)

    def load(n):
        g = {}
        exec(open(os.path.join(REPO, "arena", "agents", n, "main.py")).read(), g)
        return g["agent"]

    a, opp = load(name), load(opp_name)
    agg = collections.Counter()
    dropped_by_op = collections.Counter()
    value_by_op = collections.Counter()
    reissued = collections.Counter()

    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.reset(2)
        pending = collections.Counter()      # (op, item) still owed from a drop
        while not env.done:
            o = env.state[0].observation
            act = a(o)
            q = list(act.get("market") or [])
            agg["turns"] += 1
            agg["orders"] += len(q)
            kept, cut = q[:10], q[10:]
            # anything previously dropped that reappears now counts as deferred
            for od in kept:
                k = (od[0], od[1] if len(od) > 1 else "")
                if pending[k] > 0:
                    pending[k] -= 1
                    reissued[od[0]] += 1
            if cut:
                agg["saturated_turns"] += 1
                agg["dropped"] += len(cut)
                inv = o["market"]["inventory"]
                for od in cut:
                    op = od[0]
                    item = od[1] if len(od) > 1 else ""
                    n = od[2] if len(od) > 2 else 1
                    dropped_by_op[op] += 1
                    pending[(op, item)] += 1
                    try:
                        if op == "SELL":
                            v = econ.sell_revenue(item, inv[item], int(n))
                        elif op == "HIRE":
                            v = -_hire_cost(o["farms"][0].get("hires_today", 0))
                        elif op == "BUY_SEED":
                            v = -CROPS[item]["seed"]
                        elif op == "BUY_ANIMAL":
                            v = -ANIMALS[item]["cost"]
                        elif op == "BUY_PRODUCT":
                            v = -econ.price(item, inv[item]) * int(n)
                        elif op == "BUY_LAND":
                            v = -LAND_PRICES[0]
                        else:
                            v = 0
                    except Exception:
                        v = 0
                    value_by_op[op] += v
            env.step([act, opp(env.state[1].observation)])
    n = len(seeds)
    return agg, dropped_by_op, value_by_op, reissued, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seeds = [300_000 + rng.randrange(10**6) for _ in range(args.n)]

    for nm in args.agents:
        agg, drop, val, reiss, n = run(nm, args.opp, seeds)
        print(f"\n=== {nm} ({args.n} games vs {args.opp}) ===")
        print(f"  turns with any market order   {agg['orders'] and agg['turns']/n:>8.0f}")
        print(f"  orders issued per game        {agg['orders']/n:>8.0f}")
        print(f"  turns SATURATED (>10 orders)  {agg['saturated_turns']/n:>8.1f}")
        print(f"  orders DROPPED per game       {agg['dropped']/n:>8.1f}")
        if not agg["dropped"]:
            print("  nothing truncated -- the cap never binds for this agent")
            continue
        print(f"\n  {'dropped op':<16}{'count/game':>12}{'reissued later':>16}"
              f"{'value if kept':>16}")
        tot = 0.0
        for op, c in drop.most_common():
            tot += val[op]
            print(f"  {op:<16}{c/n:>12.1f}{reiss.get(op,0)/n:>16.1f}"
                  f"{val[op]/n:>+16,.0f}")
        print(f"  {'TOTAL':<16}{agg['dropped']/n:>12.1f}"
              f"{sum(reiss.values())/n:>16.1f}{tot/n:>+16,.0f}")
        lost = agg["dropped"] - sum(reiss.values())
        print(f"\n  never reissued: {lost/n:.1f} orders/game "
              f"({100*lost/max(1,agg['dropped']):.0f}% of drops)")


if __name__ == "__main__":
    main()
