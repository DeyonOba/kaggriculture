"""Per-day balance sheet: what each day did with the capital it was given.

Final money is one number at day 29. It cannot distinguish a day that spent
$3,000 on a pasture paying from day 12, from a day that spent $3,000 on seeds
that never ripen, from a day that banked it — all three look identical until the
season ends. That is why a losing-looking idea cannot currently be afforded: by
the time the bank is known, the day that caused it is unattributable.

This assembles the position instead:

    cash        farm["money"]
    inventory   shed + EVERY unit inventory, valued at live market prices
    deployed    standing crops (seed cost) + animals (purchase cost)
                + structures + land paid so far
    book        cash + inventory + deployed
    burn        change in book across the day  -> did this day create value
    locked      deployed / book                -> exposure not yet realised

Two traps this avoids. Reading the shed alone is badly wrong — produce sits in a
unit's hands and is DROPped mid-turn (a shed snapshot once showed 4 carrots
against 278 actually sold), so `holdings()` from tools/opp_belief.py is used.
And a standing crop is valued at SEED cost, not at what it might fetch: valuing
unripe assets at their hoped-for revenue is how a balance sheet lies.

Both farms are public, so this runs for the opponent too — `--compare` shows
where the two positions diverge, which is the phase-level question.

    PYTHONPATH=. uv run python tools/balance_sheet.py --agent champ-h8 --n 3
    PYTHONPATH=. uv run python tools/balance_sheet.py --agent champ-h8 --compare
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

from kaggle_environments import make                     # noqa: E402
from tools.opp_belief import holdings                    # noqa: E402
from kagent.constants import CROPS, ANIMALS, MARKET_PARAMS, LAND_PRICES  # noqa: E402

PRODUCTS = ["MELON", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG",
            "WHEAT", "FERTILIZER"]
STRUCTURE_COST = 0          # built by labour, not bought — counted as 0 capital
PHASES = [("A opening", 0, 7), ("B contest", 8, 17),
          ("C consolidate", 18, 25), ("D endgame", 26, 29)]


def position(obs, seat):
    """-> (cash, inventory value, deployed capital)."""
    farm = obs["farms"][seat]
    cash = float(farm.get("money", 0))

    px = (obs.get("market") or {}).get("prices") or {}
    inv = 0.0
    try:
        for item, n in holdings(obs).items():
            if item in PRODUCTS:
                inv += n * float(px.get(item, MARKET_PARAMS[item]["base"]))
    except Exception:
        pass

    deployed = 0.0
    for row in farm.get("tiles") or []:
        for t in row:
            if not isinstance(t, dict):
                continue
            # A standing crop is worth what it COST, not what it might fetch.
            crop = t.get("crop")
            if crop and crop in CROPS:
                deployed += float(CROPS[crop].get("seed", 0))
            a = t.get("animal")
            if a and a in ANIMALS:
                deployed += float(ANIMALS[a].get("cost", 0))
    owned = len(farm.get("unlocked_quadrants") or ["NW"]) - 1
    deployed += sum(LAND_PRICES[:max(0, owned)])
    return cash, inv, deployed


def run(agent, opp, seeds):
    days = collections.defaultdict(lambda: collections.defaultdict(list))
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        seen = set()
        for stp in env.steps[:-1]:
            o = stp[0].observation
            d = o["day"]
            if d in seen or o["hour"] != 23:   # end of day, before the wipe
                continue
            seen.add(d)
            for seat, who in ((0, "us"), (1, "them")):
                obs = stp[seat].observation if seat else o
                try:
                    c, i, dep = position(obs, seat)
                except Exception:
                    continue
                days[d][who + "_cash"].append(c)
                days[d][who + "_inv"].append(i)
                days[d][who + "_dep"].append(dep)
                days[d][who + "_book"].append(c + i + dep)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=71000)
    ap.add_argument("--compare", action="store_true", help="show the opponent too")
    args = ap.parse_args()

    seeds = [args.base_seed + k for k in range(args.n)]
    days = run(args.agent, args.opp, seeds)
    if not days:
        sys.exit("no data")

    m = lambda d, k: st.mean(days[d][k]) if days[d].get(k) else 0.0

    print(f"{args.agent} vs {args.opp}, {len(seeds)} seeds "
          f"(end of day, before the nightly wipe)\n")
    hdr = (f"  {'day':>4} {'cash':>9} {'inventory':>10} {'deployed':>9} "
           f"{'BOOK':>10} {'burn':>9} {'locked':>7}")
    if args.compare:
        hdr += f"   {'their BOOK':>11} {'gap':>10}"
    print(hdr)
    prev = None
    for d in sorted(days):
        book = m(d, "us_book")
        burn = book - prev if prev is not None else 0.0
        dep = m(d, "us_dep")
        row = (f"  {d:>4} {m(d,'us_cash'):>9,.0f} {m(d,'us_inv'):>10,.0f} "
               f"{dep:>9,.0f} {book:>10,.0f} {burn:>+9,.0f} "
               f"{dep/max(1,book)*100:>6.0f}%")
        if args.compare:
            tb = m(d, "them_book")
            row += f"   {tb:>11,.0f} {book-tb:>+10,.0f}"
        print(row)
        prev = book

    print(f"\n  {'phase':<15} {'book gained':>12} {'per day':>9} {'mean locked':>12}")
    for name, lo, hi in PHASES:
        ds = [d for d in days if lo <= d <= hi]
        if len(ds) < 2:
            continue
        gain = m(max(ds), "us_book") - m(min(ds), "us_book")
        locked = st.mean(m(d, "us_dep") / max(1, m(d, "us_book")) for d in ds)
        print(f"  {name:<15} {gain:>+12,.0f} {gain/len(ds):>+9,.0f} "
              f"{locked*100:>11.0f}%")
    print("\n  book = cash + inventory at live prices + capital deployed in "
          "standing crops/animals/land.\n  A standing crop is valued at SEED "
          "cost, never at hoped-for revenue.")


if __name__ == "__main__":
    main()
