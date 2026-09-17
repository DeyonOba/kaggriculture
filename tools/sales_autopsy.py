"""True units sold and realised price per item, for BOTH seats.

`order[2]` is a REQUESTED quantity, not a fill -- the engine decrements
`remaining` per unit and stops when the seller runs out of stock, so an agent
that submits "SELL CARROT 9999" to mean "sell everything" reports 9999. Reading
order counts made ahmed look like it sold 17,045 carrots.

Fills have to come from conservation instead, which is what tools/opp_belief.py
already solved and validated:

    sold(turn t) = holdings(t-1) + harvest_gain(t) - consumption(t) - holdings(t)

`holdings` is shed PLUS every unit inventory (produce sits in a unit's hands and
is DROPped mid-turn, so a shed snapshot alone is badly wrong), and consumption
covers FEED/FERTILIZE, which burn stock without selling it.

Both seats' `private` state is visible in a local replay, so this runs for us
and for them. It self-validates: we size our own orders to what we actually
hold, so our computed fills should track our own requests closely -- if they do,
the opponent's numbers can be trusted too.

    PYTHONPATH=. uv run python tools/sales_autopsy.py --n 4
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

from kaggle_environments import make  # noqa: E402
from tools.opp_belief import holdings, harvest_gain, consumption  # noqa: E402
from kagent.constants import MARKET_PARAMS  # noqa: E402

PRODUCTS = ["MELON", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG",
            "WHEAT", "FERTILIZER"]


def run(agent, opp, seeds):
    sold = [collections.defaultdict(float), collections.defaultdict(float)]
    rev = [collections.defaultdict(float), collections.defaultdict(float)]
    req = [collections.defaultdict(float), collections.defaultdict(float)]
    left = [collections.defaultdict(float), collections.defaultdict(float)]
    finals = [[], []]

    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        steps = env.steps
        for t in range(1, len(steps)):
            prev, cur = steps[t - 1], steps[t]
            prices = (prev[0].observation.get("market") or {}).get("prices") or {}
            for p in (0, 1):
                act = cur[p].get("action") or {}
                h0 = holdings(prev[p].observation)
                h1 = holdings(cur[p].observation)
                gain = harvest_gain(prev[p].observation, act, seat=p)
                burn = consumption(prev[p].observation, act, seat=p)
                for it in PRODUCTS:
                    n = h0.get(it, 0) + gain.get(it, 0) - burn.get(it, 0) - h1.get(it, 0)
                    if n > 0.5:
                        sold[p][it] += n
                        rev[p][it] += n * prices.get(it, MARKET_PARAMS[it]["base"])
                for o in (act.get("market") or []):
                    if isinstance(o, (list, tuple)) and len(o) > 2 and o[0] == "SELL":
                        req[p][o[1]] += float(o[2])
        for p in (0, 1):
            finals[p].append(steps[-1][p]["reward"])
            for it, n in holdings(steps[-1][p].observation).items():
                if it in PRODUCTS:
                    left[p][it] += n
    return sold, rev, req, left, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=71000)
    args = ap.parse_args()

    seeds = [args.base_seed + k for k in range(args.n)]
    sold, rev, req, left, finals = run(args.agent, args.opp, seeds)
    k = len(seeds)

    print(f"{args.agent} vs {args.opp}, {k} seeds, same worlds")
    print(f"final: ${st.mean(finals[0]):,.0f} vs ${st.mean(finals[1]):,.0f}   "
          f"gap ${st.mean(finals[1]) - st.mean(finals[0]):+,.0f}\n")

    print(f"{'item':>11} | {'units us':>8} {'them':>7} | {'$/unit us':>9} {'them':>7}"
          f" | {'revenue us':>11} {'them':>11} | {'unsold us':>9} {'them':>6}")
    print("-" * 104)
    tot = [0.0, 0.0]
    for it in PRODUCTS:
        u = [sold[p][it] / k for p in (0, 1)]
        r = [rev[p][it] / k for p in (0, 1)]
        pu = [r[p] / u[p] if u[p] else 0 for p in (0, 1)]
        lf = [left[p][it] / k for p in (0, 1)]
        tot[0] += r[0]
        tot[1] += r[1]
        print(f"{it:>11} | {u[0]:>8,.0f} {u[1]:>7,.0f} | {pu[0]:>9,.1f} {pu[1]:>7,.1f}"
              f" | {r[0]:>11,.0f} {r[1]:>11,.0f} | {lf[0]:>9,.0f} {lf[1]:>6,.0f}")
    print("-" * 104)
    print(f"{'TOTAL':>11} | {'':>8} {'':>7} | {'':>9} {'':>7} "
          f"| {tot[0]:>11,.0f} {tot[1]:>11,.0f}")

    print(f"\nself-check -- our computed fills vs our own requests "
          f"(we size orders to holdings, so these should track):")
    print(f"{'item':>11} {'fills':>8} {'requests':>9}")
    for it in PRODUCTS:
        f, q = sold[0][it] / k, req[0][it] / k
        if f or q:
            print(f"{it:>11} {f:>8,.0f} {q:>9,.0f}")


if __name__ == "__main__":
    main()
