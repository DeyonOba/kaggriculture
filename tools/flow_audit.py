"""Flow model: is each product sold at the rate the town can absorb?

The market is not a stock you draw down, it is a FLOW. Shops consume at
step%4==0 (a single-product shop takes 2 of its product, a multi-product shop 1
of each it lists) and the town centre takes 1 of everything at step%24==0. That
consumption is the only thing that lowers market inventory, i.e. the only thing
that lifts the price back up. So for each product there is a daily drain rate,
set by the shop draw, and it is SHARED between the two farms:

    sustainable units/day for us = drain(shops) - opponent's supply rate

Sell under it and the price sits near base; sell over it and the ladder
collapses and every later unit is wasted. Two products have essentially no
drain at all -- FERTILIZER (no shop lists it, not in TOWN_CENTER_PRODUCTS) and
MELON (town centre only, 1/day) -- so they are mined, not farmed: their price
never recovers and the only question is when to take the stock.

Reports, per product: drain, our supply, their supply, realized price, and the
revenue we forfeited by exceeding the drain.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.getcwd())
import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

SOLD = []
_SEAT, _STEP = {}, [0]
_oc, _om = K._commit_unit, K._process_market


def _commit(op, item, price, farm, private, market, cap=100):
    ok = _oc(op, item, price, farm, private, market, cap)
    if ok and op == "SELL":
        SOLD.append((_STEP[0], _SEAT.get(id(farm), -1), item, float(price)))
    return ok


def _mk(state, env):
    o = state[0].observation
    _SEAT.clear()
    for i, f in enumerate(o.farms):
        _SEAT[id(f)] = i
    _STEP[0] = K.get(o, "step", 0)
    return _om(state, env)


K._commit_unit, K._process_market = _commit, _mk


def drain_per_day(shops):
    d = defaultdict(int)
    for s in shops:
        items = K.SHOPS.get(s, ())
        m = 2 if len(items) == 1 else 1
        for it in items:
            d[it] += m * 6
    for it in K.TOWN_CENTER_PRODUCTS:
        d[it] += 1
    return d


def run(a, b, seeds):
    agg = defaultdict(lambda: defaultdict(float))
    for sd in seeds:
        SOLD.clear()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd}, debug=False)
        env.run([a, b])
        shops = (env.steps[-1][0]["observation"].get("town") or {}).get("unlocked_shops") or []
        dr = drain_per_day(shops)
        for step, seat, item, price in SOLD:
            if seat < 0:
                continue
            agg[item][f"u{seat}"] += 1
            agg[item][f"r{seat}"] += price
        for it in K.PRODUCTS:
            agg[it]["drain"] += dr.get(it, 0)
    n = len(seeds)
    print(f"\n{n} games, {len(shops)} shops at end\n")
    print(f"{'product':<11} {'drain/day':>9} {'our u/day':>10} {'their u/day':>11} "
          f"{'our $/u':>8} {'base':>5} {'realized':>9} {'over-drain':>10}")
    for it in K.PRODUCTS:
        d = agg[it]
        dr_ = d["drain"] / n
        u0 = d["u0"] / n / 30.0
        u1 = d["u1"] / n / 30.0
        p0 = d["r0"] / max(1e-9, d["u0"])
        base = K.MARKET_PARAMS[it]["base"]
        over = (u0 + u1) - dr_
        print(f"{it:<11} {dr_:>9.1f} {u0:>10.1f} {u1:>11.1f} {p0:>8,.0f} {base:>5} "
              f"{p0/base:>9.2f} {over:>+10.1f}")
    print("\nover-drain > 0 means the two farms together sell faster than the town absorbs,")
    print("so the ladder is being walked down and every later unit fetches less.")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "arena/agents/champ-h8/main.py"
    b = sys.argv[2] if len(sys.argv) > 2 else "arena/agents/pub-ahmed-v38/main.py"
    run(a, b, [31000, 31001, 31002])
