"""How much of each product can the town actually absorb, and how much do the
two farms dump on top of that?

`tools/market_table.py` showed melon has no shop: its only buyer is the town
centre at a flat 1.0 unit/day, while every other product's demand grows with
each shop unlock. This sizes that effect for every product.

Per product, over the season:

    absorbed    units the town removes (sum of the per-day drain actually seen)
    supplied    units both farms put on the market, by conservation:
                    supplied = (inv_end - inv_start) + absorbed
    glut        supplied - absorbed, i.e. units that just sit there
    px_end      closing price / base -- what the last unit was worth
    mrev        marginal revenue of ONE more unit at the closing inventory

`mrev` is the number that decides whether growing more of something is worth
anything. A crop's headline economics use `base`, which is the price of the
FIRST unit; if the market is glutted, the unit our planner is deciding to grow
sells at `mrev`, not at `base`.

    PYTHONPATH=. uv run python tools/market_headroom.py --n 5
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
from kagent import econ                           # noqa: E402
from kagent.constants import MARKET_PARAMS, CROPS  # noqa: E402

PRODUCTS = ["MELON", "STRAWBERRY", "MILK", "WOOL", "CARROT", "TOMATO", "EGG",
            "WHEAT", "FERTILIZER"]
CROP_OF = {v.get("product", k): k for k, v in CROPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=71000)
    args = ap.parse_args()

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in range(args.n):
        env = make("kaggriculture", configuration={"seed": args.base_seed + k},
                   debug=False)
        env.run([os.path.join(REPO, "arena", "agents", args.agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", args.opp, "main.py")])
        first, last, drained = {}, {}, collections.defaultdict(float)
        seen = set()
        for stp in env.steps[:-1]:
            o = stp[0].observation
            d = o["day"]
            inv = (o.get("market") or {}).get("inventory") or {}
            for p in PRODUCTS:
                if p not in first:
                    first[p] = float(inv.get(p, MARKET_PARAMS[p]["I0"]))
                last[p] = float(inv.get(p, MARKET_PARAMS[p]["I0"]))
            if d in seen or o["hour"] != 0:
                continue
            seen.add(d)
            dr = econ.town_drain_per_day(
                list((o.get("town") or {}).get("unlocked_shops", [])))
            for p in PRODUCTS:
                drained[p] += dr.get(p, 0.0)
        px_last = (env.steps[-2][0].observation.get("market") or {}).get("prices") or {}
        for p in PRODUCTS:
            ab = drained[p]
            sup = (last[p] - first[p]) + ab
            acc[p]["absorbed"].append(ab)
            acc[p]["supplied"].append(sup)
            acc[p]["px_end"].append(float(px_last.get(p, MARKET_PARAMS[p]["base"])))
            acc[p]["mrev"].append(econ.price(p, last[p]))

    m = lambda p, f: st.mean(acc[p][f]) if acc[p][f] else 0.0
    print(f"{args.agent} vs {args.opp}, {args.n} seeds — season absorption\n")
    print(f"  {'item':<12} {'base':>5} {'absorbed':>9} {'supplied':>9} {'glut':>8} "
          f"{'sup/abs':>8} {'px_end':>7} {'mrev':>6} {'mrev/base':>10}")
    for p in PRODUCTS:
        ab, sup = m(p, "absorbed"), m(p, "supplied")
        base = MARKET_PARAMS[p]["base"]
        print(f"  {p:<12} {base:>5} {ab:>9.0f} {sup:>9.0f} {sup-ab:>+8.0f} "
              f"{(sup/ab if ab else float('inf')):>8.2f} {m(p,'px_end'):>7.0f} "
              f"{m(p,'mrev'):>6.0f} {m(p,'mrev')/base:>10.2f}")

    print(f"\n  crop value at BASE price vs at the MARGIN we actually face:\n")
    print(f"  {'crop':<12} {'seed':>5} {'yield':>6} {'days':>5} "
          f"{'base $/tile-day':>16} {'marginal $/tile-day':>20}")
    for prod, crop in CROP_OF.items():
        v = CROPS[crop]
        y, s, md = v.get("max_yield", 0), v.get("seed", 0), v.get("max_yield_day", 1)
        base = MARKET_PARAMS[prod]["base"]
        mr = m(prod, "mrev")
        print(f"  {crop:<12} {s:>5} {y:>6} {md:>5} "
              f"{(y*base-s)/md:>16.1f} {(y*mr-s)/md:>20.1f}")


if __name__ == "__main__":
    main()
