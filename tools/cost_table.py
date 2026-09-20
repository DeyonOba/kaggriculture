"""The engine's economics in one table: costs, clocks, and crash curves.

Four arms were built and lost this week -- melon burst, geese substitution,
the 4th quadrant, reactive hiring -- and every one of them had good
correlational support from the 21,054-game corpus. Three could have been
refused on arithmetic before any compute:

  geese      swapped a cow's $3,080 of milk for a goose's $1,170 of eggs,
             because a MARGINAL milk price was applied to an AVERAGE decision
  4th quad   added 25 tiles at hands/work already 0.47
  hiring     raised the peak to 16 hands, where the wage bill is $2,583/day
             against $232 at 11

What they had in common is that the corpus shows correlations while the
ENGINE holds convex, adversarial constants -- fibonacci wages, prices that
fall quadratically, capacity that collapses. This prints those constants
next to the realised outcomes, so a proposal can be priced before it is
built.

    PYTHONPATH=. uv run python tools/cost_table.py
"""
import glob
import itertools
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402
from kagent import econ  # noqa: E402

SEASON = 30
HARVEST = ("/private/tmp/claude-501/-Users-gideonoba-WorkSpace-OpenSource-"
           "kaggriculture/d454410e-05c5-4e3d-b208-515aa3822dbf/scratchpad")
PRODUCTS = K.PRODUCTS


def corpus():
    """Realised prices and glut rates, or None if the harvest is not local."""
    fs = sorted(glob.glob(f"{HARVEST}/kaggle_m*/out/metrics_*.npz"))
    if not fs:
        return None
    parts = [np.load(f) for f in fs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    sold = D["sell"].sum((1, 2)).sum(0)
    rev = D["rev"].sum((1, 2)).sum(0)
    glut = ((D["day_mkt"] > K.MARKET_I0) & (D["seen"][:, :, None] > 0)
            ).any(1).mean(0)
    return {"real": rev / np.maximum(sold, 1), "glut": glut,
            "n": len(D["game"])}


def units_to(frac, product):
    """Units sellable from I0 before the price falls below frac x base."""
    base = K.MARKET_PARAMS[product]["base"]
    return econ.units_sellable_above(product, K.MARKET_I0,
                                     max(1, int(frac * base)), cap=4000)


def p_no_shop(product):
    """P(no unlocked shop ever buys this product) over 8 uniform draws."""
    names = sorted(K.SHOPS)
    miss = sum(1 for s in names if product not in K.SHOPS[s])
    return (miss / len(names)) ** K.MAX_SHOP_INSTANCES


def drain_per_instance(product):
    """Units/day one shop instance removes, averaged over which shop it is."""
    names = sorted(K.SHOPS)
    tot = sum((2 if len(K.SHOPS[s]) == 1 else 1)
              for s in names if product in K.SHOPS[s])
    return 6.0 * tot / len(names)


def main():
    C = corpus()
    tag = f"realised over {C['n']:,} games" if C else "corpus not local"
    print(f"\n{'='*94}\nTHE ENGINE'S ECONOMICS   ({tag})\n{'='*94}")

    # ---- 1. the crash curve -------------------------------------------
    print("\n1. PRICE CRASH — how fast a pool dies when you oversupply it\n")
    print(f"    {'product':<12}{'base':>6}{'T':>6}{'above':>8}{'steep':>7}"
          f"{'units to 80%':>14}{'to 50%':>9}{'to 20%':>9}"
          f"{'gluts':>8}{'realised':>10}")
    for p in PRODUCTS:
        m = K.MARKET_PARAMS[p]
        g = f"{100*C['glut'][PRODUCTS.index(p)]:.0f}%" if C else "--"
        r = f"{C['real'][PRODUCTS.index(p)]:.0f}" if C else "--"
        print(f"    {p:<12}{m['base']:>6}{m['T']:>6}{m['above_func']:>8}"
              f"{m['above_target']:>7.1f}{units_to(0.8, p):>14,}"
              f"{units_to(0.5, p):>9,}{units_to(0.2, p):>9,}{g:>8}{r:>10}")
    print("\n    `above_func` is the shape of the fall past I0 and it is the"
          "\n    single most decision-relevant constant in the game:"
          "\n      log    barely moves -- WHEAT, EGG are UNCRASHABLE"
          "\n      linear steady       -- STRAWBERRY, MILK, FERTILIZER"
          "\n      sq     quadratic    -- MELON, WOOL die fastest")

    # ---- 2. demand ------------------------------------------------------
    print("\n\n2. DEMAND — who buys it, and the risk that nobody does\n")
    print(f"    {'product':<12}{'shops that buy it':<44}"
          f"{'drain/day/draw':>16}{'P(no shop)':>12}")
    for p in PRODUCTS:
        buyers = [s for s in sorted(K.SHOPS) if p in K.SHOPS[s]]
        lbl = ", ".join(b.lower() for b in buyers) or "NO SHOP"
        town = " +town" if p in K.TOWN_CENTER_PRODUCTS else ""
        print(f"    {p:<12}{(lbl + town)[:43]:<44}"
              f"{drain_per_instance(p):>16.2f}{100*p_no_shop(p):>11.1f}%")
    print(f"\n    {K.MAX_SHOP_INSTANCES} draws, one every 3 days from day 3, "
          f"uniform over {len(K.SHOPS)} shops.")
    print("    MELON is in NO shop: its only demand is the town centre, "
          "~1 unit/day, fixed.")

    # ---- 3. crops -------------------------------------------------------
    print("\n\n3. CROPS — one tile, planted once\n")
    print(f"    {'crop':<12}{'seed':>6}{'1st':>5}{'span':>6}{'max':>5}"
          f"{'ongoing':>9}{'u/tile-day':>12}{'$/tile-day':>12}"
          f"{'seed payback':>14}")
    for c in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        cd = K.CROPS[c]
        span = (cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
                if cd["ongoing"] else max(cd["max_yield_day"],
                                          cd["first_yield_day"]))
        upd = cd["max_yield"] / max(span, 1)
        price = C["real"][PRODUCTS.index(c)] if C else K.MARKET_PARAMS[c]["base"]
        gross = upd * price
        pay = cd["seed"] / max(gross, 1e-9)
        print(f"    {c:<12}{cd['seed']:>6}{cd['first_yield_day']:>5}{span:>6}"
              f"{cd['max_yield']:>5}{str(cd['ongoing']):>9}{upd:>12.2f}"
              f"{gross - cd['seed']/max(span,1):>12.1f}{pay:>12.2f}d")

    # ---- 4. livestock ---------------------------------------------------
    print("\n\n4. LIVESTOCK — bought once, NEVER removed, eats 1 WHEAT/day\n")
    print(f"    {'animal':<10}{'cost':>6}{'struct':>9}{'1st':>5}{'every':>7}"
          f"{'product':>10}{'u/tile-day':>12}{'$/tile-day':>12}"
          f"{'feed/season':>13}{'payback':>10}")
    for a in ("GOOSE", "COW", "SHEEP"):
        ad = K.ANIMALS[a]
        life = SEASON - ad["first_yield_day"]
        upd = (life / max(ad["interval"], 1)) / SEASON
        price = (C["real"][PRODUCTS.index(ad["product"])] if C
                 else K.MARKET_PARAMS[ad["product"]]["base"])
        wheat = C["real"][PRODUCTS.index("WHEAT")] if C else 25
        feed = life * wheat
        gross = upd * price
        print(f"    {a:<10}{ad['cost']:>6}{ad['structure']:>9}"
              f"{ad['first_yield_day']:>5}{ad['interval']:>7}"
              f"{ad['product']:>10}{upd:>12.2f}"
              f"{gross - ad['cost']/SEASON:>12.1f}{feed:>13,.0f}"
              f"{ad['cost']/max(gross,1e-9):>8.1f}d")
    print("\n    CARE doubles output on a fed day, so a cared animal is worth"
          "\n    ~2x the row above. Feed is priced at the realised WHEAT price"
          "\n    and is a real cost even when the wheat is home-grown.")

    # ---- 5. labour ------------------------------------------------------
    print("\n\n5. LABOUR — fibonacci, and this is where proposals die\n")
    print(f"    {'hands':>6}{'$/day':>10}{'marginal':>11}{'x15 days':>12}"
          f"{'  tiles one hand can service':<30}")
    prev = 0
    for n in range(1, 19):
        c = econ.hire_total_cost(n)
        note = ""
        if n == 11:
            note = "  <- CHAMP_HIRES ceiling, the knee"
        print(f"    {n:>6}{c:>10,}{c-prev:>11,}{c*15:>12,}{note}")
        prev = c
    print("\n    Convexity means SMOOTH beats MATCHED even against lumpy"
          "\n    demand: 11+11 costs $464, 8+14 costs $1,040 for the same"
          "\n    22 hand-days. Do not propose peak hiring.")

    # ---- 6. land --------------------------------------------------------
    print("\n\n6. LAND — and the labour it silently requires\n")
    print(f"    {'quadrant':<10}{'price':>8}{'tiles':>7}{'$/tile':>9}"
          f"{'cumulative':>12}")
    cum = 0
    for i, q in enumerate(K.LAND_ORDER):
        pr = K.LAND_PRICES[i]
        cum += pr
        print(f"    {q:<10}{pr:>8,}{25:>7}{pr/25:>9.0f}{cum:>12,}")
    print(f"\n    Opening bank is $3,000 = NE + SW exactly, so CASH binds the"
          f"\n    first two and the day gate does not. SE at $4,000 gated"
          f"\n    -$6,266 / -$6,017 because 25 more tiles arrive with zero"
          f"\n    more hands.")

    # ---- 7. break-evens -------------------------------------------------
    print("\n\n7. BREAK-EVENS — what a marginal thing has to earn\n")
    wheat = C["real"][PRODUCTS.index("WHEAT")] if C else 25
    for n in (8, 11, 14, 16):
        c = econ.hire_total_cost(n) - econ.hire_total_cost(n - 1)
        print(f"    hand #{n:<3} costs ${c:>6,}/day = "
              f"{c/max(wheat,1):>6.1f} wheat/day it must enable")
    print()
    for a in ("GOOSE", "COW", "SHEEP"):
        ad = K.ANIMALS[a]
        pr = (C["real"][PRODUCTS.index(ad["product"])] if C
              else K.MARKET_PARAMS[ad["product"]]["base"])
        print(f"    a {a:<6} must clear {ad['cost']/max(pr,1):>5.1f} units of "
              f"{ad['product']:<5} just to repay its purchase")


if __name__ == "__main__":
    main()
