"""Knob-level imitation mining: what do strong players SET, per world type?

Reads rl/datasets/sessions (seed, both action scripts, finals, shops,
ratings) and derives, for every recorded seat, the season-long "knob
equivalents" of planner.KNOB_DEFAULTS straight from the action stream — no
replay needed — plus the day-6 context (the two shops open when knobs are
decided), the seat's rating and its margin. It then reports:

  1. medians of each knob equivalent by rating tier, against our defaults;
  2. medians by day-6 demand context (consumers of each product);
  3. within-tier, within-context association of each knob with margin
     (seats above vs below the context median) — candidate rules for the
     counterfactual bandit to verify (`rl/bandit.py collect --knobs --arms`).

    PYTHONPATH=. uv run python tools/mine_knobs.py extract --procs 8
    PYTHONPATH=. uv run python tools/mine_knobs.py report --min-rating 2500
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kagent.constants import SHOPS  # noqa: E402

SESS_DIR = "rl/datasets/sessions"
OUT = "rl/datasets/mined/seats.jsonl"
PRODUCTS = ("STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "WHEAT")

# Our champion plan's equivalents, for the comparison columns.
OUR_DEFAULTS = {
    "melon_d1": 12, "straw_d9": 18, "straw_d13": 33, "cows_d18": 8, "sheep_d20": 8, "geese_d18": 0,
    "wheat_rate_10_24": None, "carrot_total": None, "tomato_total": 0,
    "hires_d0_5": 4.5, "hires_d10_25": 11, "land1_day": 6, "land2_day": 10,
    "carrot_first_day": 25,
}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def seat_features(session, side):
    acts = session["actions"][side]
    plant = defaultdict(lambda: defaultdict(int))   # day -> crop -> n
    buys = defaultdict(lambda: defaultdict(int))    # day -> animal -> n
    hires = defaultdict(int)
    land_days = []
    sells = defaultdict(lambda: defaultdict(int))   # day -> item -> units requested
    for t, a in enumerate(acts):
        day = t // 24
        for m in a.get("market") or []:
            if not m:
                continue
            if m[0] == "HIRE":
                hires[day] += 1
            elif m[0] == "BUY_LAND":
                land_days.append(day)
            elif m[0] == "BUY_ANIMAL" and len(m) > 2:
                buys[day][m[1]] += _int(m[2])
            elif m[0] == "SELL" and len(m) > 2:
                sells[day][m[1]] += min(_int(m[2]), 200)
        for u in [a.get("farmer") or []] + list(a.get("hands") or []):
            if u and u[0] == "PLANT" and len(u) > 1:
                plant[day][u[1]] += 1

    def cum(table, key, upto):
        return sum(table[d].get(key, 0) for d in range(0, upto + 1))

    def rate(table, key, d0, d1):
        return sum(table[d].get(key, 0) for d in range(d0, d1 + 1)) / max(1, d1 - d0 + 1)

    def avg_hires(d0, d1):
        return sum(hires.get(d, 0) for d in range(d0, d1 + 1)) / max(1, d1 - d0 + 1)

    carrot_days = [d for d in range(30) if plant[d].get("CARROT", 0) >= 5]
    f = {
        "melon_d1": cum(plant, "MELON", 1),
        "straw_d9": cum(plant, "STRAWBERRY", 9),
        "straw_d13": cum(plant, "STRAWBERRY", 13),
        "cows_d18": cum(buys, "COW", 18),
        "sheep_d20": cum(buys, "SHEEP", 20),
        "geese_d18": cum(buys, "GOOSE", 18),
        "wheat_rate_10_24": round(rate(plant, "WHEAT", 10, 24), 2),
        "wheat_total": cum(plant, "WHEAT", 29),
        "carrot_total": cum(plant, "CARROT", 29),
        "carrot_first_day": carrot_days[0] if carrot_days else None,
        "tomato_total": cum(plant, "TOMATO", 29),
        "hires_d0_5": round(avg_hires(0, 5), 2),
        "hires_d10_25": round(avg_hires(10, 25), 2),
        "land1_day": land_days[0] if land_days else None,
        "land2_day": land_days[1] if len(land_days) > 1 else None,
        "n_land": len(land_days),
        "wool_sell_days": sum(1 for d in sells if sells[d].get("WOOL")),
        "milk_sell_days": sum(1 for d in sells if sells[d].get("MILK")),
        "straw_sell_days": sum(1 for d in sells if sells[d].get("STRAWBERRY")),
    }
    return f


def consumers(shops, item):
    return sum(1 for s in shops if item in SHOPS.get(s, ()))


def extract_one(path):
    try:
        s = json.load(open(path))
    except Exception:
        return []
    shops = s.get("shops") or []
    day6 = shops[:2]  # shops unlock at the end of days 2 and 5 -> visible on day 6
    scores = s.get("scores") or [None, None]
    rows = []
    for side in (0, 1):
        f = seat_features(s, side)
        f.update({
            "episode": s["episode"], "side": side, "rating": scores[side], "opp_rating": scores[1 - side],
            "final": s["finals"][side], "margin": s["finals"][side] - s["finals"][1 - side],
            "win": s["finals"][side] > s["finals"][1 - side],
            "ctx6": {p: consumers(day6, p) for p in PRODUCTS},
            "ctx_all": {p: consumers(shops, p) for p in PRODUCTS},
            "n_shops": len(shops),
        })
        rows.append(f)
    return rows


def extract(args):
    paths = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with Pool(args.procs) as pool, open(OUT, "w") as f:
        for rows in pool.imap_unordered(extract_one, paths, chunksize=32):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n += 1
    print(f"wrote {n} seats from {len(paths)} sessions -> {OUT}")


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _mean_se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), 0.0, n
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


KNOBS = ["melon_d1", "straw_d9", "straw_d13", "cows_d18", "sheep_d20", "geese_d18", "wheat_rate_10_24",
         "wheat_total", "carrot_total", "carrot_first_day", "tomato_total", "hires_d0_5", "hires_d10_25",
         "land1_day", "land2_day", "wool_sell_days", "milk_sell_days", "straw_sell_days"]
KNOB_PRODUCT = {"straw_d9": "STRAWBERRY", "straw_d13": "STRAWBERRY", "cows_d18": "MILK", "sheep_d20": "WOOL",
                "geese_d18": "EGG", "carrot_total": "CARROT", "tomato_total": "TOMATO", "wheat_rate_10_24": "WHEAT"}


def report(args):
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    rated = [r for r in rows if r["rating"] is not None]
    print(f"{len(rows)} seats, {len(rated)} rated")
    tiers = [("2500+", lambda r: r["rating"] >= 2500), ("2000-2500", lambda r: 2000 <= r["rating"] < 2500),
             ("1000-2000", lambda r: 1000 <= r["rating"] < 2000), ("<1000", lambda r: r["rating"] < 1000)]
    print(f"\n== medians by rating tier (ours = champ-h5 equivalents) ==\n{'knob':<18} {'ours':>6} " + " ".join(f"{t:>10}" for t, _ in tiers))
    for k in KNOBS:
        cells = []
        for _, sel in tiers:
            cells.append(_median([r[k] for r in rated if sel(r)]))
        ours = OUR_DEFAULTS.get(k)
        print(f"{k:<18} {str(ours) if ours is not None else '-':>6} " + " ".join(f"{('-' if c is None else round(c, 1)):>10}" for c in cells))

    top = [r for r in rated if r["rating"] >= args.min_rating]
    print(f"\n== {len(top)} seats rated >= {args.min_rating}: medians by DAY-6 demand context (consumers of the knob's product: 0 / 1 / 2) ==")
    for k, prod in KNOB_PRODUCT.items():
        cells = []
        for c in (0, 1, 2):
            sub = [r[k] for r in top if r["ctx6"][prod] == c]
            cells.append(f"{_median(sub)} (n={len(sub)})" if sub else "-")
        print(f"  {k:<18} {prod:<10} " + " | ".join(f"{c:>14}" for c in cells))

    print(f"\n== within-context margin association (seats >= {args.min_rating}): mean margin of seats ABOVE vs BELOW the context median setting ==")
    cands = []
    for k, prod in KNOB_PRODUCT.items():
        for c in (0, 1, 2):
            sub = [r for r in top if r["ctx6"][prod] == c and r[k] is not None]
            if len(sub) < 60:
                continue
            med = _median([r[k] for r in sub])
            hi = [r["margin"] for r in sub if r[k] > med]
            lo = [r["margin"] for r in sub if r[k] < med]
            if len(hi) < 20 or len(lo) < 20:
                continue
            mh, seh, nh = _mean_se(hi)
            ml, sel, nl = _mean_se(lo)
            diff = mh - ml
            se = math.sqrt(seh ** 2 + sel ** 2)
            cands.append((diff / max(se, 1e-9), k, prod, c, med, diff, se, nh, nl))
    cands.sort(key=lambda x: -abs(x[0]))
    print(f"{'knob':<18} {'product':<10} {'consumers@d6':>12} {'ctx median':>10} {'above-below margin':>19} {'se':>7} {'n(hi/lo)':>10} {'z':>5}")
    for z, k, prod, c, med, diff, se, nh, nl in cands[:24]:
        print(f"{k:<18} {prod:<10} {c:>12} {med:>10} {diff:>+19,.0f} {se:>7,.0f} {nh:>4}/{nl:<5} {z:>5.1f}")
    # opponent-conditioned: do strong players plant fewer strawberries when the OPPONENT is strong?
    print("\n== opponent-strength conditioning (seats >= min rating): medians when opponent >= 2500 vs < 2000 ==")
    for k in ("straw_d13", "cows_d18", "sheep_d20", "geese_d18", "wheat_total", "carrot_total"):
        strong = _median([r[k] for r in top if (r["opp_rating"] or 0) >= 2500])
        weak = _median([r[k] for r in top if r["opp_rating"] is not None and r["opp_rating"] < 2000])
        print(f"  {k:<14} vs strong opp: {strong}   vs weak opp: {weak}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "report"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--min-rating", type=float, default=2500.0)
    args = ap.parse_args()
    {"extract": extract, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
