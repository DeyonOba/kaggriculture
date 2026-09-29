"""The INVOICE: what a player committed capital to, day by day.

    PYTHONPATH=. uv run python rlv3/trajectory.py --leaders --days 0-11

WHY DOLLARS AND NOT TILES. The crop/livestock choice is usually written as a
share of the board, and a share of the board has been refuted four ways here
-- the leaders' own daily mix faithfully delivered gates -4.9 +/- 4.4, moving
to their proportions gates -1.5/-2.8/-18.2, the tile queue -15.3/-24.3, and
the LP's provable optimum ties. A share prices none of what the two sides
actually compete for: a pasture is one tile AND $400 AND the feed wheat it
eats AND the chore turns it takes, while a wheat tile is $10 and a watering.
Dollars are the conserved quantity, so the invoice is in dollars.

    invoice[d] = {"seed", "livestock", "land", "hands"}   -> shares sum to 1

FEED IS LIVESTOCK. `BUY_PRODUCT WHEAT` is wheat bought BACK to feed animals --
290 units in one traced game -- and it exists only because animals exist, so
it is charged to livestock. `BUY_PRODUCT FERTILIZER` is a crop input and is
charged to seed. Getting this backwards makes livestock look cheap and is
most of why a tile share misprices the decision.

WHAT THIS IS FOR. `rlv3/PHASE1.md`: the opening's success criterion is
agreement with the TOP BOARDS conditioned on the draw, not agreement with
champ-h11, whose ceiling we would simply inherit. This module is the
measurement half of that -- no games are played, it reads recorded replays.

The noise floor is known and is not zero: the three leaders agree with each
other at 0.085 mean pairwise gap and with THEMSELVES across their own games at
0.086. So 0.085 is the target and there is no credit for going below it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kagent import econ                                              # noqa: E402
from kagent.constants import ANIMALS, CROPS, LAND_PRICES             # noqa: E402

TURNS = 24
BUCKETS = ("seed", "livestock", "land", "hands")
LEADERS = ("mmpq", "boey", "dsm")


# ------------------------------------------------------------------ one game
def _cost_of(order, state):
    """-> (bucket, dollars) for one market order, or None.

    `state` carries the within-day and within-game counters the engine prices
    against: HIRE is marginally priced in the number hired TODAY
    (1,1,2,3,5,...) and BUY_LAND in the number of quadrants already bought
    (LAND_PRICES = 1000, 2000, 4000). Pricing either at a flat rate silently
    flattens exactly the timing this module exists to see.
    """
    op = order[0]
    item = order[1] if len(order) > 1 else None
    qty = int(order[2]) if len(order) > 2 else 1
    if op == "BUY_SEED":
        return "seed", float(CROPS.get(item, {}).get("seed", 0)) * qty
    if op == "BUY_ANIMAL":
        return "livestock", float(ANIMALS.get(item, {}).get("cost", 0)) * qty
    if op == "BUY_PRODUCT":
        # WHEAT here is FEED -- it exists only because animals do.
        return ("livestock" if item == "WHEAT" else "seed"), None
    if op == "BUY_LAND":
        i = state["land_bought"]
        state["land_bought"] += 1
        return "land", float(LAND_PRICES[i]) if i < len(LAND_PRICES) else 0.0
    if op == "HIRE":
        total = 0.0
        for _ in range(qty):
            total += float(econ.hire_marginal_cost(state["hired_today"]))
            state["hired_today"] += 1
        return "hands", total
    return None


def invoice_for(path, seat):
    """-> [{day, seed, livestock, land, hands, total, shares, shops, ...}].

    One row a day. `shops` is the draw state at DAWN of that day, which is what
    a decision on that day could actually have conditioned on.
    """
    with open(path) as fh:
        steps = json.load(fh)["steps"]
    n_days = len(steps) // TURNS
    state = {"land_bought": 0, "hired_today": 0}
    rows = []
    for day in range(n_days):
        k0 = day * TURNS
        spend = {b: 0.0 for b in BUCKETS}
        state["hired_today"] = 0
        try:
            obs = steps[k0][seat]["observation"]
            shops = tuple(sorted((obs.get("town") or {}).get("unlocked_shops") or []))
            prices = dict(((obs.get("market") or {}).get("prices") or {}))
            me = (obs.get("farms") or [{}, {}])[seat]
            money = float(me.get("money", 0))
            opp = float((obs.get("farms") or [{}, {}])[1 - seat].get("money", 0))
        except Exception:
            continue
        for h in range(TURNS):
            k = k0 + h
            if k >= len(steps):
                break
            act = steps[k][seat].get("action") or {}
            for order in (act.get("market") or []):
                got = _cost_of(order, state)
                if got is None:
                    continue
                bucket, dollars = got
                if dollars is None:
                    # BUY_PRODUCT is quoted off the live curve; the observation
                    # carries the price, so it is read rather than guessed.
                    item, qty = order[1], int(order[2]) if len(order) > 2 else 1
                    dollars = float(prices.get(item, 0)) * qty
                spend[bucket] += dollars
        total = sum(spend.values())
        rows.append({
            "day": day, **{b: round(spend[b], 1) for b in BUCKETS},
            "total": round(total, 1),
            "shares": {b: (spend[b] / total if total > 0 else 0.0) for b in BUCKETS},
            "shops": shops, "n_shops": len(shops),
            "money": money, "margin": money - opp,
        })
    return rows


def winner_seat(path):
    """The seat that actually won. Leader replays record both sides."""
    with open(path) as fh:
        d = json.load(fh)
    r = d.get("rewards") or []
    if len(r) == 2 and r[0] is not None and r[1] is not None:
        return 0 if r[0] >= r[1] else 1
    return 0


# ------------------------------------------------------------------- corpus
def leader_paths(names=LEADERS, per=None):
    out = []
    for n in names:
        fs = sorted(glob.glob(os.path.join(REPO, "replays", n, "*.json")))
        out += [(n, f) for f in (fs[:per] if per else fs)]
    return out


def collect(paths, seat="winner"):
    """-> [(tag, rows)] for each replay."""
    out = []
    for tag, p in paths:
        s = winner_seat(p) if seat == "winner" else int(seat)
        try:
            rows = invoice_for(p, s)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {type(e).__name__} {e}")
            continue
        if rows:
            out.append((f"{tag}:{os.path.basename(p)[:18]}", rows))
    return out


def band(games, day, bucket=None, draw_key=None):
    """Quantiles of the leaders' share on `day`.

    `draw_key` buckets by the DRAW STATE, which is the whole point: a leader
    who has seen a wool shop is not evidence about a game where none opened.
    Pass a key function over the row (e.g. `lambda r: r["n_shops"]`).
    """
    vals = {b: [] for b in BUCKETS}
    n = 0
    for _tag, rows in games:
        r = next((x for x in rows if x["day"] == day), None)
        if r is None or (draw_key and draw_key(r) != draw_key.want):
            continue
        n += 1
        for b in BUCKETS:
            vals[b].append(r["shares"][b])
    out = {}
    for b in BUCKETS:
        v = sorted(vals[b])
        if not v:
            out[b] = None
            continue
        out[b] = {"n": len(v), "lo": v[len(v) // 4], "med": v[len(v) // 2],
                  "hi": v[(3 * len(v)) // 4],
                  "mean": sum(v) / len(v)}
    out["_n"] = n
    return out[bucket] if bucket else out


def accumulate(rows, window=3):
    """Roll the per-day invoice into a `window`-day commitment.

    WHY THE DAILY SHARE IS THE WRONG UNIT, measured on the leaders themselves:
    their day-0 shares agree at TVD 0.084 -- the known floor -- and then days
    1-11 disagree at 0.23-0.43. That is not four different strategies. Capital
    arrives lumpy (a $400 cow on day 4 against a $10 wheat seed on day 5), so
    two players running the SAME plan a day out of step score as total
    disagreement, and a daily share measures the jitter rather than the plan.

    A 3-day window is the smallest one that spans the gap between two
    consecutive shop draws (days 3, 6, 9 -- `econ._UNLOCK_DAYS`), so it holds
    the whole response to one draw without reaching into the next. It is the
    same window the planner commits over, which is not a coincidence: the
    measurement and the decision should be the same object.
    """
    out = []
    for i, r in enumerate(rows):
        lo = max(0, i - window + 1)
        chunk = rows[lo:i + 1]
        spend = {b: sum(x[b] for x in chunk) for b in BUCKETS}
        total = sum(spend.values())
        out.append({**r, **{b: round(spend[b], 1) for b in BUCKETS},
                    "total": round(total, 1), "window": len(chunk),
                    "shares": {b: (spend[b] / total if total > 0 else 0.0)
                               for b in BUCKETS}})
    return out


def pairwise_gap(games, days):
    """Mean |share difference| between every pair of games, per day.

    This is the NOISE FLOOR, and it is measured rather than assumed: two
    leaders playing different worlds disagree by this much for reasons no
    model can predict, so a model matching them to within it has matched them.
    """
    out = {}
    for d in days:
        rows = [next((x for x in r if x["day"] == d), None) for _t, r in games]
        rows = [r for r in rows if r]
        gaps = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                gaps.append(sum(abs(rows[i]["shares"][b] - rows[j]["shares"][b])
                                for b in BUCKETS) / 2.0)   # total variation
        out[d] = statistics.mean(gaps) if gaps else float("nan")
    return out


def agreement(ours, games, days):
    """-> (mean TVD to the leader median, fraction of days inside the band)."""
    tvd, inside = [], 0
    for d in days:
        r = next((x for x in ours if x["day"] == d), None)
        bd = band(games, d)
        if r is None or bd["_n"] == 0:
            continue
        tvd.append(sum(abs(r["shares"][b] - bd[b]["med"]) for b in BUCKETS) / 2.0)
        if all(bd[b]["lo"] <= r["shares"][b] <= bd[b]["hi"] for b in BUCKETS):
            inside += 1
    return (statistics.mean(tvd) if tvd else float("nan"),
            inside / max(1, len(tvd)))


# --------------------------------------------------------------------- CLI
def _days(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaders", action="store_true")
    ap.add_argument("--replays", default=None, help="a glob instead of the leaders")
    ap.add_argument("--per", type=int, default=None, help="replays per leader")
    ap.add_argument("--days", default="0-11")
    ap.add_argument("--seat", default="winner")
    ap.add_argument("--window", type=int, default=1,
                    help="accumulate the invoice over this many days (3 = the "
                         "planner's own commitment window)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.replays:
        paths = [("x", p) for p in sorted(glob.glob(args.replays))]
    else:
        paths = leader_paths(per=args.per)
    games = collect(paths, seat=args.seat)
    if args.window > 1:
        games = [(t, accumulate(r, args.window)) for t, r in games]
    days = _days(args.days)
    print(f"{len(games)} replays, days {days[0]}-{days[-1]}, seat={args.seat}, "
          f"window={args.window}\n")

    print(f"{'day':>4}{'n':>4}{'$spent':>9}  " +
          "".join(f"{b:>22}" for b in BUCKETS))
    print(f"{'':>8}{'':>9}  " + "".join(f"{'lo   med    hi':>22}" for _ in BUCKETS))
    for d in days:
        bd = band(games, d)
        tot = [next((x["total"] for x in r if x["day"] == d), 0) for _t, r in games]
        cells = ""
        for b in BUCKETS:
            v = bd[b]
            cells += (f"{v['lo']:>8.2f}{v['med']:>7.2f}{v['hi']:>7.2f}" if v
                      else f"{'-':>22}")
        print(f"{d:>4}{bd['_n']:>4}{statistics.mean(tot) if tot else 0:>9,.0f}  {cells}")

    print("\nDRAW STATE (shops open at dawn), share of replays")
    for d in days:
        c = {}
        for _t, rows in games:
            r = next((x for x in rows if x["day"] == d), None)
            if r:
                c[r["n_shops"]] = c.get(r["n_shops"], 0) + 1
        tot = sum(c.values()) or 1
        print(f"  day {d:>2}  " + "  ".join(f"{k} shop(s) {v/tot:.0%}"
                                            for k, v in sorted(c.items())))

    gaps = pairwise_gap(games, days)
    print("\nNOISE FLOOR -- mean pairwise TVD between leader games, per day")
    print("  " + "  ".join(f"d{d}:{gaps[d]:.3f}" for d in days))
    ok = [g for g in gaps.values() if g == g]
    print(f"  mean {statistics.mean(ok):.3f}   <- a model matching to within "
          f"this has matched the leaders")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"days": days, "n_games": len(games),
                       "band": {str(d): band(games, d) for d in days},
                       "noise_floor": {str(d): gaps[d] for d in days},
                       "games": {t: r for t, r in games}}, fh)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
