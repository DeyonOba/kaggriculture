"""Commitment and recovery: what does a wrong planting cost, and can it be undone?

The opening bets a tile on a product for a fixed number of days. If the bet
is wrong -- the shop draw misses, the pool gluts, the opponent floods it --
the question is how fast that tile can earn again. `DIG` is the only undo:
it removes a plant, a weed or an empty structure (never a placed animal) and
frees the tile immediately, forfeiting the seed and any accrued yield.

Three things measured here:

  1. COMMITMENT   tile-days locked per crop before the tile frees naturally,
                  and the sunk cost of digging out early
  2. RECOVERY     days from a DIG to the replacement's first yield
  3. BEHAVIOUR    what top players actually DIG, and WHEN -- is DIG a risk
                  instrument, or just weed clearance?

(3) is the one that decides whether risk-managed replanting is a real policy
in this game or an idea nobody uses.

    PYTHONPATH=. uv run python tools/recovery.py --n 120
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
I0 = 10000


def span_of(c):
    cd = K.CROPS[c]
    if cd["ongoing"]:
        return cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
    return max(cd["max_yield_day"], cd["first_yield_day"])


def job(path):
    """Every DIG in one game: what was removed, when, and market state."""
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _farmer_position)
    import tools.price_forecast as pf

    digs = []
    counts = collections.Counter()
    orig = getattr(K, "_rc_unit", None) or K._apply_unit_action
    K._rc_unit = orig
    state = {"day": 0, "inv": {}}

    def unit(farm, private, idx, action, board_size, day, turns_per_day,
             shed_capacity=100):
        op = action[0] if isinstance(action, list) and action else None
        pre = None
        if op == "DIG":
            pos = _farmer_position(farm, idx)
            if pos is not None:
                t = farm["tiles"][pos[1]][pos[0]]
                if isinstance(t, dict):
                    pre = (t.get("kind"), t.get("crop"),
                           t.get("yield_units", 0),
                           day - t.get("planted_day", day))
        r = orig(farm, private, idx, action, board_size, day, turns_per_day,
                 shed_capacity)
        if pre is not None:
            kind, crop, y, age = pre
            counts[kind or "?"] += 1
            if kind == "PLANT":
                digs.append((day, crop, y, age,
                             state["inv"].get(crop, I0)))
        return r

    K._apply_unit_action = unit
    try:
        s = json.load(open(path))
        env = make("kaggriculture", configuration={"seed": s["seed"]},
                   debug=False)
        env.reset(2)
        a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
        while not env.done:
            o = env.state[0].observation
            state["day"] = o["step"] // 24
            state["inv"] = dict(o["market"]["inventory"])
            env.step([a0(o), a1(env.state[1].observation)])
    finally:
        K._apply_unit_action = orig
    return counts, digs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()

    print("\n1. COMMITMENT — tile-days a planting locks up\n")
    print(f"    {'crop':<12}{'seed':>6}{'1st yield':>11}{'tile freed':>12}"
          f"{'locked':>8}{'dig-out sunk':>14}")
    for c in CROPS:
        cd = K.CROPS[c]
        span = span_of(c)
        freed = "never (ongoing)" if cd["ongoing"] else f"day {span}"
        print(f"    {c:<12}{cd['seed']:>6}{cd['first_yield_day']:>11}"
              f"{freed:>12}{span:>8}{cd['seed']:>10} + yield")
    print("\n    An ongoing crop NEVER frees its tile on its own -- tomato and"
          "\n    strawberry hold the ground until they expire or are DUG out.")

    print("\n2. RECOVERY — days from a DIG to the replacement's first yield\n")
    print(f"    {'replace with':<14}" + "".join(f"{c[:5]:>8}" for c in CROPS))
    for c in CROPS:
        row = "".join(f"{K.CROPS[c]['first_yield_day']:>8}" for _ in CROPS)
        print(f"    {c:<14}{row}")
    print("\n    Recovery is the REPLACEMENT's clock, not the mistake's:"
          "\n    digging a melon on day 8 and sowing wheat earns again on"
          "\n    day 10, two days later, having sunk $80.")

    from tools.opp_coupling import load_index
    import random
    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(9).shuffle(paths)
    with mp.Pool(args.procs, maxtasksperchild=40) as pool:
        out = pool.map(job, paths[: args.n])
    n = len(out)
    counts = collections.Counter()
    digs = []
    for c, d in out:
        counts.update(c)
        digs += d

    print(f"\n3. WHAT TOP PLAYERS ACTUALLY DIG   ({n} games, both seats)\n")
    tot = sum(counts.values())
    for k, v in counts.most_common():
        print(f"    {k:<14}{v/n:>8.1f} per game{100*v/max(tot,1):>8.1f}%")

    if not digs:
        print("\n    no crop DIGs recorded")
        return
    print(f"\n   of the crop DIGs ({len(digs)/n:.1f} per game):\n")
    print(f"    {'crop':<12}{'n':>7}{'mean day':>10}{'mean age':>10}"
          f"{'yield lost':>12}{'mkt inv at dig':>16}")
    by = collections.defaultdict(list)
    for d, crop, y, age, inv in digs:
        by[crop].append((d, y, age, inv))
    for c in CROPS:
        v = by.get(c) or []
        if not v:
            print(f"    {c:<12}{0:>7}")
            continue
        a = np.array(v, float)
        print(f"    {c:<12}{len(v):>7}{a[:,0].mean():>10.1f}"
              f"{a[:,2].mean():>10.1f}{a[:,1].mean():>12.2f}"
              f"{a[:,3].mean():>16,.0f}")
    print(f"\n    market I0 is {I0:,}. An inventory ABOVE it means the pool"
          f"\n    was already glutted when they dug -- which is what a risk"
          f"\n    instrument would look like.")
    ripe = sum(1 for _, _, y, _, _ in digs if y > 0)
    print(f"\n    dug while still HOLDING yield: {ripe}/{len(digs)}"
          f" = {100*ripe/max(len(digs),1):.0f}%"
          f"   (throwing away harvestable units)")


if __name__ == "__main__":
    main()
