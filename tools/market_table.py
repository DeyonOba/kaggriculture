"""Per-day price vs demand, per item: is a price high because demand is high, or
because supply has not landed yet?

Those two states call for opposite policies -- sell INTO strength, or sell BEFORE
the flood -- and nothing in this project has ever distinguished them. We record
realised prices after the fact and we have `econ.town_drain_per_day`, but the two
have never been put side by side per day.

For each day and product:

    price       live market price
    inv         market inventory (the ladder depth we sell into)
    drain       units the town removes per day from the OPEN shops
    ours        units we are holding, shed + every unit inventory
    theirs      units THEY are holding (both farms are public)
    pressure    (ours + theirs) / drain -- how many days of demand the two
                farms are already sitting on

`pressure` is the quantity a sell decision actually needs. Below 1 the town can
absorb everything on the table; above 1 someone is going to sell into a falling
price, and the only question is who goes first.

Reading the shed alone is badly wrong (produce sits in a unit's hands and is
DROPped mid-turn), so holdings() from tools/opp_belief.py is used for both seats.

    PYTHONPATH=. uv run python tools/market_table.py --item MELON --n 5
    PYTHONPATH=. uv run python tools/market_table.py --summary --n 5
"""

import argparse
import collections
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from kaggle_environments import make              # noqa: E402
from tools.opp_belief import holdings             # noqa: E402
from kagent import econ                           # noqa: E402
from kagent.constants import MARKET_PARAMS, CROPS  # noqa: E402

PRODUCTS = ["MELON", "STRAWBERRY", "MILK", "WOOL", "CARROT", "TOMATO", "EGG",
            "WHEAT", "FERTILIZER"]


def collect(agent, opp, seeds):
    rec = collections.defaultdict(lambda: collections.defaultdict(list))
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        seen = set()
        for stp in env.steps[:-1]:
            o = stp[0].observation
            d = o["day"]
            if d in seen or o["hour"] != 0:      # start of day, before the day's trades
                continue
            seen.add(d)
            mk = o.get("market") or {}
            px = mk.get("prices") or {}
            inv = mk.get("inventory") or {}
            shops = list((o.get("town") or {}).get("unlocked_shops", []))
            drain = econ.town_drain_per_day(shops)
            try:
                hu = holdings(stp[0].observation)
                ht = holdings(stp[1].observation)
            except Exception:
                continue
            for p in PRODUCTS:
                r = rec[(d, p)]
                r["price"].append(float(px.get(p, MARKET_PARAMS[p]["base"])))
                r["inv"].append(float(inv.get(p, 0)))
                r["drain"].append(float(drain.get(p, 0.0)))
                r["ours"].append(float(hu.get(p, 0)))
                r["theirs"].append(float(ht.get(p, 0)))
    return rec


def m(rec, d, p, k):
    v = rec[(d, p)][k]
    return st.mean(v) if v else 0.0


def plant_table(rec, days, last_day=29):
    """What is ONE tile planted on day d actually worth?

    Engine semantics, from kaggriculture.py:770-801 and :431-467 --

    ongoing (WHEAT/CARROT/TOMATO): one production every `interval` days
        starting at `first_yield_day`, each adding 1 unit (2 if watered AND
        fertilized). The tile DIES after exactly `max_yield` productions.
    one-shot (MELON/STRAWBERRY): units come only from WATER actions inside
        [(max_yield_day+1)//2, max_yield_day]; harvest destroys the tile.

    Each unit is priced at the market inventory prevailing on the day it lands,
    not at `base` and not at the day-29 inventory -- that is the whole point.
    Units that would land after `last_day` are worth nothing.
    """
    inv_on = {}
    for d in days:
        for p in PRODUCTS:
            inv_on[(d, p)] = m(rec, d, p, "inv")
    horizon = max(days)

    def inv_at(d, p):
        return inv_on.get((min(d, horizon), p), MARKET_PARAMS[p]["I0"])

    print("value of ONE tile planted on day d, priced at the market inventory "
          "prevailing when each unit lands\n")
    crops = list(CROPS)
    print(f"  {'day':>4} " + " ".join(f"{c[:9]:>10}" for c in crops))
    for d in range(0, 25):
        cells = []
        for c in crops:
            v = CROPS[c]
            fy, iv2 = v["first_yield_day"], max(1, v.get("interval", 1))
            my, mxd = v["max_yield"], v["max_yield_day"]
            rev, seed = 0.0, v.get("seed", 0)
            if v.get("ongoing"):
                for k in range(my):
                    dd = d + fy + k * iv2
                    if dd > last_day:
                        break
                    rev += econ.price(v.get("product", c), inv_at(dd, v.get("product", c)))
            else:
                hd = d + mxd
                waters = mxd - (mxd + 1) // 2 + 1
                units = min(my, waters)
                if hd <= last_day:
                    rev = econ.sell_revenue(v.get("product", c),
                                            inv_at(hd, v.get("product", c)), units)
            cells.append(f"{(rev - seed):>10,.0f}" if rev else f"{'—':>10}")
        print(f"  {d:>4} " + " ".join(cells))
    print("\n  Net of seed cost. '—' = cannot finish before day "
          f"{last_day}. Ongoing crops die after max_yield productions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--item", default="MELON")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=71000)
    ap.add_argument("--summary", action="store_true",
                    help="one row per product instead of a per-day table")
    ap.add_argument("--plant", action="store_true",
                    help="what one tile planted on day d is actually worth")
    args = ap.parse_args()

    seeds = [args.base_seed + k for k in range(args.n)]
    rec = collect(args.agent, args.opp, seeds)
    days = sorted({d for d, _ in rec})
    if not days:
        sys.exit("no data")

    if args.summary:
        print(f"{args.agent} vs {args.opp}, {len(seeds)} seeds — season summary\n")
        print(f"  {'item':<12} {'base':>6} {'mean px':>8} {'px/base':>8} "
              f"{'drain/day':>10} {'peak held':>10} {'peak press':>11} {'peak day':>9}")
        for p in PRODUCTS:
            px = [m(rec, d, p, "price") for d in days]
            base = MARKET_PARAMS[p]["base"]
            held = [m(rec, d, p, "ours") + m(rec, d, p, "theirs") for d in days]
            pres = [h / dr if (dr := m(rec, d, p, "drain")) > 0 else 0
                    for d, h in zip(days, held)]
            pk = max(range(len(days)), key=lambda i: pres[i]) if pres else 0
            print(f"  {p:<12} {base:>6} {st.mean(px):>8.1f} "
                  f"{st.mean(px)/base:>8.2f} {st.mean([m(rec,d,p,'drain') for d in days]):>10.2f} "
                  f"{max(held):>10.0f} {max(pres):>11.2f} {days[pk]:>9}")
        print("\n  pressure = (our holdings + theirs) / town drain per day.")
        print("  Below 1 the town absorbs everything on the table; above 1 someone")
        print("  sells into a falling price and the only question is who goes first.")
        return

    if args.plant:
        plant_table(rec, days)
        return

    p = args.item
    print(f"{p} — {args.agent} vs {args.opp}, {len(seeds)} seeds "
          f"(start of day, before the day's trades)\n")
    print(f"  {'day':>4} {'price':>7} {'px/base':>8} {'mkt inv':>9} {'drain':>7} "
          f"{'ours':>7} {'theirs':>7} {'pressure':>9}")
    base = MARKET_PARAMS[p]["base"]
    for d in days:
        dr = m(rec, d, p, "drain")
        held = m(rec, d, p, "ours") + m(rec, d, p, "theirs")
        px = m(rec, d, p, "price")
        print(f"  {d:>4} {px:>7.0f} {px/base:>8.2f} {m(rec,d,p,'inv'):>9,.0f} "
              f"{dr:>7.2f} {m(rec,d,p,'ours'):>7.1f} {m(rec,d,p,'theirs'):>7.1f} "
              f"{(held/dr if dr > 0 else 0):>9.2f}")


if __name__ == "__main__":
    main()
