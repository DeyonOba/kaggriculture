"""Risk-adjusted product scores: expected rent AND its variance.

The rent rule maximises EXPECTED revenue per tile-day. It has no notion of
variance, so it cannot express "bet on stable items" or "they are on stable,
so the risky ones are uncontested".

Variance here is not estimated, it is COMPUTED. The only exogenous economic
variable in the game is the shop draw -- eight uniform draws from eight shops
every three days, capped at MAX_SHOP_INSTANCES -- so the distribution of a
product's demand is an exact multinomial, and price is a deterministic
function of inventory. Everything else (yield clocks, drain per shop, the
price curve) is an engine constant.

For each product we report:

    E[rent]     expected $/tile-day at a representative supply
    sd[rent]    its standard deviation over the shop draw
    Sharpe      E/sd -- how much return per unit of risk
    P(floor)    probability the price ends at or near the $1 floor
    crash slope above_target / T, the engine's own volatility parameter

A "stable bet" is high Sharpe and low P(floor): its payoff barely depends on
which shops open. A "risky bet" has high E and high sd.

The differentiation argument that makes this game-theoretic: both farms sell
into ONE `market["inventory"]`, so if the opponent's visible board is loaded
with stable products, those get crowded and the risky ones become
uncontested -- which RAISES the risky products' risk-adjusted value. The
opponent's board is public, so this is decidable each turn.

    PYTHONPATH=. uv run python tools/volatility.py
"""

import argparse
import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

N_DRAWS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supply", type=int, default=120,
                    help="units of this product WE put on the market")
    ap.add_argument("--days", type=int, default=20)
    args = ap.parse_args()

    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, CROPS, MARKET_PARAMS, PRODUCTS, SHOPS,
        TOWN_CENTER_PRODUCTS, market_price)
    import numpy as np

    names = sorted(SHOPS)
    # exact distribution over multisets of 8 draws is 8^8; sample the
    # per-product drain distribution instead, which is what matters
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(names), size=(20000, N_DRAWS))

    def drain_for(p, row):
        per = 0
        for j in row:
            s = names[j]
            if p in SHOPS[s]:
                per += 2 if len(SHOPS[s]) == 1 else 1
        return 6 * per + (1 if p in TOWN_CENTER_PRODUCTS else 0)

    # yield per tile-day, from the engine's own clocks
    def tile_days_and_units(p):
        if p in CROPS:
            cd = CROPS[p]
            if cd["ongoing"]:
                span = args.days
                n = min(cd["max_yield"],
                        max(0, (span - cd["first_yield_day"]) // cd["interval"] + 1))
                return span, n
            return cd["max_yield_day"] + 1, cd["max_yield"]
        for a in ANIMALS.values():
            if a["product"] == p:
                span = args.days
                n = max(0, (span - a["first_yield_day"]) // a["interval"] + 1)
                return span, n
        return args.days, 0

    print(f"our supply {args.supply} units over {args.days} days; "
          f"{len(draws):,} shop-draw samples\n")
    print(f"    {'product':<12}{'E[$/unit]':>11}{'sd':>8}{'Sharpe':>8}"
          f"{'P(floor)':>10}{'crash slope':>13}{'E[$/tile-day]':>15}")
    rows = []
    for p in PRODUCTS:
        mp = MARKET_PARAMS[p]
        td, units = tile_days_and_units(p)
        vals = []
        for row in draws[:4000]:
            d = drain_for(p, row)
            # net inventory after our supply meets `days` of drain
            inv = mp["I0"] + args.supply - d * args.days
            vals.append(market_price(p, int(inv)))
        v = np.asarray(vals, dtype=float)
        e, sd = v.mean(), v.std()
        pf = float((v <= max(2, 0.1 * mp["base"])).mean())
        slope = mp["above_target"] / mp["T"] * 1000
        rent = e * units / max(1, td)
        # No shop sells MELON or FERTILIZER, so their demand is deterministic
        # and sd is exactly 0 -- report that rather than a divide-by-epsilon.
        det = sd < 1e-9
        sharpe = float("inf") if det else e / sd
        rows.append((rent, p, e, sd, sharpe, pf, slope))
        sh = "  det" if det else f"{sharpe:>5.1f}"
        print(f"    {p:<12}{e:>11.0f}{sd:>8.0f}{sh:>8}"
              f"{100*pf:>9.0f}%{slope:>13.2f}{rent:>15.0f}")

    def line(t):
        rent, p, e, sd, sh, pf, slope = t
        tag = "deterministic" if sd < 1e-9 else f"Sharpe {sh:>5.1f}"
        return (f"    {p:<12} {tag:<16} P(floor) {100*pf:>3.0f}%   "
                f"E[$/tile-day] {rent:>4.0f}")
    contested = [r for r in rows if r[3] >= 1e-9]
    print("\n  NO SHOP-DRAW RISK AT ALL (no shop sells them, demand is fixed):")
    for t in [r for r in rows if r[3] < 1e-9]:
        print(line(t))
    print("\n  STABLE BETS among contested products -- payoff barely depends"
          "\n  on which shops open:")
    for t in sorted(contested, key=lambda r: -r[4])[:3]:
        print(line(t))
    print("\n  RISKY BETS -- worth holding only when the opponent is NOT also"
          "\n  in them, since both farms share one market inventory:")
    for t in sorted(contested, key=lambda r: r[4])[:3]:
        print(line(t))


if __name__ == "__main__":
    main()
