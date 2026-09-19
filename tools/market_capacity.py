"""Absorption CAPACITY per product over time -- the missing sell-timing input.

Every sell rule in this project prices a SINGLE unit at the CURRENT inventory.
None of them knows how much the market can still absorb, how fast that
capacity regenerates, or how much of it the opponent is about to consume. You
cannot model sell timing without that, which is why every floor experiment has
been a constant scaled by `premium_scale`.

Capacity is exact, not estimated:

    cap_p(t, f) = units sellable at t before the price falls below f x base
                = econ.units_sellable_above(p, inv_p(t), f*base, BIG)

    regen_p     = drain_p per day (6 x shop instances x mult, + town centre)
    consumed_p  = our sales + their sales, both observable

so over a horizon H the total absorbable volume is

    cap_p(t, f) + H * regen_p - (their projected supply)

and OUR share of it is what a sell decision should actually be rationing.

Reports, per product: capacity now, capacity regenerated per day, how much of
the season's total absorption each seat actually consumes, and the day
capacity first goes negative (past which every extra unit is sold below f).

    PYTHONPATH=. uv run python tools/market_capacity.py --n 40
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]



def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def job(spec):
    path, frac = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kagent import econ
    from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
    import tools.price_forecast as pf
    from tools.market_clock import daily_drain

    s = json.load(open(path))
    pf.FLOW.clear()
    ref = pf._patch_flow()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    dawn = {}
    while not env.done:
        o = env.state[0].observation
        ref["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
        env.step([a0(o), a1(env.state[1].observation)])

    out = {"episode": s["episode"], "cap": {}, "regen": {}, "exhaust": {},
           "used": {0: {}, 1: {}}}
    for p in PRODUCTS:
        base = MARKET_PARAMS[p]["base"]
        floor = max(1, int(frac * base))
        caps, exhaust = [], None
        for d in sorted(dawn):
            inv = dawn[d]["market"]["inventory"][p]
            c = econ.units_sellable_above(p, inv, floor, cap=5000)
            caps.append(c)
            if exhaust is None and c <= 0:
                exhaust = d
        out["cap"][p] = caps
        out["regen"][p] = daily_drain(dawn[max(dawn)]["town"]["unlocked_shops"], p)
        out["exhaust"][p] = exhaust if exhaust is not None else 30
        for seat in (0, 1):
            out["used"][seat][p] = sum(
                v.get(p, 0) for v in pf.FLOW.get(seat, {}).values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frac", type=float, default=0.6,
                    help="capacity is measured down to this fraction of base")
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"{len(paths)} sessions; capacity = units sellable above "
          f"{args.frac:.0%} of base\n", flush=True)
    with mp.Pool(args.procs) as pool:
        rows = pool.map(job, [(p, args.frac) for p in paths])

    n = len(rows)
    print(f"    {'product':<12}{'cap d0':>8}{'cap d10':>9}{'cap d20':>9}"
          f"{'regen/day':>11}{'exhaust day':>13}{'used us':>9}{'used them':>11}")
    for p in PRODUCTS:
        c0 = sum(r["cap"][p][0] for r in rows) / n
        c10 = sum(r["cap"][p][min(10, len(r["cap"][p]) - 1)] for r in rows) / n
        c20 = sum(r["cap"][p][min(20, len(r["cap"][p]) - 1)] for r in rows) / n
        rg = sum(r["regen"][p] for r in rows) / n
        ex = sum(r["exhaust"][p] for r in rows) / n
        u0 = sum(r["used"][0][p] for r in rows) / n
        u1 = sum(r["used"][1][p] for r in rows) / n
        print(f"    {p:<12}{c0:>8.0f}{c10:>9.0f}{c20:>9.0f}{rg:>11.1f}"
              f"{ex:>13.1f}{u0:>9.0f}{u1:>11.0f}")

    print("\n  SEASON BUDGET: capacity at day 0 plus 30 days of regeneration,")
    print("  against what both seats together actually sold:\n")
    print(f"    {'product':<12}{'season cap':>12}{'both sold':>11}"
          f"{'utilisation':>13}{'verdict':>22}")
    for p in PRODUCTS:
        c0 = sum(r["cap"][p][0] for r in rows) / n
        rg = sum(r["regen"][p] for r in rows) / n
        both = sum(r["used"][0][p] + r["used"][1][p] for r in rows) / n
        season = c0 + 30 * rg
        u = both / max(1.0, season)
        verdict = ("SCARCE - ration it" if u > 0.9 else
                   "contested" if u > 0.55 else "slack - sell freely")
        print(f"    {p:<12}{season:>12.0f}{both:>11.0f}{100*u:>12.0f}%"
              f"{verdict:>22}")


if __name__ == "__main__":
    main()
