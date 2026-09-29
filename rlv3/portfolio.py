"""Risk-adjusted rent per tile-day -- the thing the risk knob selects WITH.

    PYTHONPATH=. uv run python rlv3/portfolio.py --day 0
    PYTHONPATH=. uv run python rlv3/portfolio.py --day 6 --alpha 0.2 --upper

THE IDEA. "We are losing, take variance" and "we are ahead, protect the lead"
do not need a control surface of their own. They are already a CROP CHOICE,
because the engine splits products into two shapes and `MARKET_PARAMS` says
which is which:

    above_target   price decay per unit of glut
    WHEAT 0.20 (log), EGG 0.20 (log)      ANNUITY -- price survives volume
    STRAWBERRY 1.60, MILK 1.60 (linear)   trap
    WOOL 3.20, MELON 3.60 (sq)            TRAP -- crashes on volume, spikes on demand

An annuity pays about the same whatever the town draws. A trap pays a great
deal if its demand lands and very little if it does not. So the risk posture
selects along that axis, and `(alpha, upper)` from `kagent.risk` is exactly
the dial: the LOWER tail of the drain distribution is the pessimistic read for
a holder, the UPPER tail is the read a player who is behind should take
(`risk.alpha_for_lead`).

Livestock carries the same axis in TIME rather than in price. An animal bought
before its shop opens is a bet on a draw that has not happened; bought after,
it is a certainty. `kagent/options.py` prices that lead-time premium at $559
for a sheep -- and the leaders pay it in all 40 recorded games at
P(never a yarn shop) = 0.34.

WHAT IS COMPUTED AND WHAT IS FITTED. Everything here is computed. Seed cost,
occupancy and watered units are engine arithmetic; the price at a future
inventory is the shipped forward model (`planner._rent_inventory`, gated
+$3,185 +/- 768); the drain distribution is the exact lattice over the eight
known shop types. The only fitted object in the whole chain is the opponent's
behaviour, which is where `rlv3.objective.ProjState` belongs and this module
does not touch.

RENT IS PER TILE-DAY, NOT PER TILE. A melon holds its tile for 11 days and a
wheat for 4. Ranking by value per tile ignores that the tile could have turned
over twice, which is how an earlier rent version picked tomato every day.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kagent import risk                                              # noqa: E402
from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, SHOPS    # noqa: E402

# Products whose price survives volume. Read from the engine table rather than
# listed by hand, so a rules change cannot leave this stale.
ANNUITY_MAX_DECAY = 0.5

# An animal eats one wheat a day (planner.wheat_reserve sizes the buffer at
# feed_days x animals). Its true cost is therefore its price PLUS the feed for
# the rest of the season, which is the term a tile share drops entirely.
FEED_WHEAT_PER_DAY = 1.0

# Each extra tile of the same asset floods its own market, so its next tile is
# worth less than its last. Without a decay the fill is degenerate and buys one
# asset until the ground runs out -- exactly what the Kelly allocator did when
# it put $250 of $3,000 into 25 wheat and nothing else.
DIMINISH = 0.80


def shape(item):
    """-> ('annuity'|'trap', above_target). The risk axis, from the engine."""
    a = float(MARKET_PARAMS.get(item, {}).get("above_target", 1.0))
    return ("annuity" if a <= ANNUITY_MAX_DECAY else "trap"), a


def demand_odds(item, unlocked, day):
    """-> (p_open_now, p_ever). How likely is anyone to want this?

    MELON is the special case that has already cost this project a wrong
    answer: it is in NO shop and has a fixed town demand of 1 unit/day, so
    "no shop carries it" is not "never sells" (kagent/options.py carries the
    same fix).
    """
    carriers = [s for s, items in SHOPS.items() if item in items]
    if not carriers:
        return 1.0, 1.0                    # MELON: fixed town demand
    open_now = any(s in unlocked for s in carriers)
    if open_now:
        return 1.0, 1.0
    budget = risk.remaining_unlocks(day, len(unlocked)) if hasattr(
        risk, "remaining_unlocks") else 0
    from kagent.econ import remaining_unlocks
    budget = remaining_unlocks(day, len(unlocked))
    p_miss_one = 1.0 - len(carriers) / len(SHOPS)
    return 0.0, 1.0 - p_miss_one ** max(0, budget)


# ------------------------------------------------------------------- crops
def crop_rent(view, crop, alpha=1.0, upper=False):
    """-> dict. Risk-adjusted value of one tile of `crop`, per tile-day.

    `alpha`/`upper` are the CVaR read, and they enter through the drain
    distribution: less drain than expected leaves more inventory and a lower
    price, so the tail of the drain IS the tail of the value and no resampling
    is needed. alpha=1.0 is the plain expectation.
    """
    from tools import conversion_value as CV

    if crop not in CROPS:
        return None
    cost, val, units = CV.plant_value(view, crop, alpha, upper)
    if units <= 0:
        return None
    # The same tile at the risk-NEUTRAL read, so the spread is visible: a big
    # gap between the two is what "this is a bet" means, quantitatively.
    _c, val_mean, _u = CV.plant_value(view, crop, 1.0, False)
    occ = CV.occupancy(crop)
    kind, decay = shape(crop)
    p_now, p_ever = demand_odds(crop, tuple(view.unlocked_shops), view.day)
    return {
        "item": crop, "kind": kind, "decay": decay,
        "cost": cost, "units": units, "occupancy": occ,
        "value": val, "value_mean": val_mean,
        "net": val - cost, "net_mean": val_mean - cost,
        "rent_per_tile_day": (val - cost) / occ,
        "rent_mean": (val_mean - cost) / occ,
        "spread": (val_mean - val) / max(1.0, abs(val_mean)),
        "p_demand_now": p_now, "p_demand_ever": p_ever,
        "roi": (val - cost) / cost if cost else 0.0,
    }


# --------------------------------------------------------------- livestock
def animal_rent(view, kind, alpha=1.0, upper=False):
    """-> dict. Same read for an animal, plus the LEAD-TIME premium.

    An animal is a stream, and its tile is held for the rest of the season, so
    its occupancy is `29 - day` rather than a crop's fixed term. The premium is
    the production an early holder gets that a buyer waiting for confirmation
    cannot recover -- the whole reason to buy before the shop opens.
    """
    from tools import conversion_value as CV

    if kind not in ANIMALS:
        return None
    cost, val, ok = CV.buy_value(view, kind, alpha, upper)
    _c, val_mean, _o = CV.buy_value(view, kind, 1.0, False)
    a = ANIMALS[kind]
    occ = max(1, 29 - view.day)
    p_now, p_ever = demand_odds(a["product"], tuple(view.unlocked_shops), view.day)
    k, decay = shape(a["product"])
    return {
        "item": kind, "product": a["product"], "kind": k, "decay": decay,
        "cost": cost, "occupancy": occ, "exercisable": ok,
        "value": val, "value_mean": val_mean,
        "net": val - cost, "net_mean": val_mean - cost,
        "rent_per_tile_day": (val - cost) / occ,
        "rent_mean": (val_mean - cost) / occ,
        "spread": (val_mean - val) / max(1.0, abs(val_mean)),
        "p_demand_now": p_now, "p_demand_ever": p_ever,
        "first_yield": a["first_yield_day"], "interval": a["interval"],
        "roi": (val - cost) / cost if cost else 0.0,
    }


# ------------------------------------------------------------------ ranking
def rank(view, alpha=None, upper=None):
    """Every crop and animal, ordered by risk-adjusted rent per tile-day.

    With no alpha given the posture comes from the position itself --
    `risk.alpha_for_lead(margin, day)` flips to the upper tail when behind and
    tightens as the season runs out.
    """
    if alpha is None or upper is None:
        margin = float(view.me.money) - float(view.opp.money)
        a, u = risk.alpha_for_lead(margin, view.day)
        alpha = a if alpha is None else alpha
        upper = u if upper is None else upper
    rows = []
    for c in CROPS:
        r = crop_rent(view, c, alpha, upper)
        if r:
            rows.append(r)
    for k in ANIMALS:
        r = animal_rent(view, k, alpha, upper)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -r["rent_per_tile_day"])
    return {"alpha": alpha, "upper": upper, "day": view.day, "rows": rows}


def target_stock(view, alpha=None, upper=None, tiles=None):
    """-> the STANDING portfolio the risk posture implies: how many tiles of
    each asset a farm of `tiles` workable tiles should hold.

    THIS IS A STOCK, AND THE DISTINCTION IS THE WHOLE POINT. Two earlier
    versions asked flow questions and got confident wrong answers. Summing
    net-per-dollar said seed 0.73-0.91 where the leaders spend 0.13-0.20 --
    you cannot add rates, and a $10 wheat seed returning 12.6x dominated a sum
    a $500 sheep at 1.9x belonged in. Water-filling today's cash bought nothing
    at all, because a leader's dawn has ZERO bare tiles and $156 in hand: they
    are fully invested, so "what would you buy now" has no answer.

    Measured on 40 leader replays, the two readings of the same farm:

        day      2     4     6     8    10    14
        tiles    .20   .28   .37   .28   .26   .27    animals' share of TILES
        capital  .75   .72   .75   .67   .69   .74    animals' share of CAPITAL

    Livestock is DENSE CAPITAL ON FEW TILES -- about a quarter of the ground
    and about seventy percent of the money. A tile share hides exactly that,
    which is one more reason every tile-share arm here has gated flat.

    So the allocation is over TILES (the binding constraint) and the ratio that
    comes out is reported in both units, so it can be compared against either
    half of `rlv3.trajectory` without a unit error.
    """
    from kagent.constants import ANIMALS as _AN

    r = rank(view, alpha, upper)
    rows = [x for x in r["rows"] if x["net"] > 0]
    if tiles is None:
        tiles = max(1, 25 * max(1, len(view.me.unlocked_quadrants)))
    order = sorted(rows, key=lambda x: -x["rent_per_tile_day"])
    if not order:
        return {**r, "tiles": tiles, "stock": {}, "split": {"seed": .5, "livestock": .5}}

    # Each asset's tile is charged its own carrying cost: an animal also needs
    # the feed wheat it eats for the rest of the season, and that wheat needs
    # GROUND, so a pasture is really 1 + feed_tiles of commitment.
    days_left = max(1, 29 - view.day)
    wheat_per_tile_day = max(1e-6, next(
        (x["units"] / CV_OCC("WHEAT") for x in rows if x["item"] == "WHEAT"), 1.0))
    stock = {}
    left = tiles
    guard = 0
    while left >= 1 and guard < 500:
        guard += 1
        best, best_cost = None, None
        for x in order:
            is_animal = x["item"] in _AN
            feed_tiles = (FEED_WHEAT_PER_DAY / wheat_per_tile_day) if is_animal else 0.0
            need = 1.0 + feed_tiles
            if need <= left:
                best, best_cost = x, need
                break
        if best is None:
            break
        stock[best["item"]] = stock.get(best["item"], 0) + 1
        left -= best_cost
        # Diminishing returns: each unit of an asset floods its own market, so
        # its next tile is worth less. Without this the fill is degenerate --
        # it buys one asset until the ground runs out, which is what the Kelly
        # allocator did when it put $250 of $3,000 into 25 wheat.
        best["rent_per_tile_day"] *= DIMINISH
        order.sort(key=lambda x: -x["rent_per_tile_day"])

    n_an = sum(v for k, v in stock.items() if k in _AN)
    n_cr = sum(v for k, v in stock.items() if k not in _AN)
    cap_an = sum(v * ANIMALS[k]["cost"] for k, v in stock.items() if k in _AN)
    cap_cr = sum(v * CROPS[k]["seed"] for k, v in stock.items() if k not in _AN)
    tot_t, tot_c = n_an + n_cr, cap_an + cap_cr
    return {**r, "tiles": tiles, "stock": stock,
            "tile_share": {"seed": n_cr / tot_t if tot_t else .5,
                           "livestock": n_an / tot_t if tot_t else .5},
            "capital_share": {"seed": cap_cr / tot_c if tot_c else .5,
                              "livestock": cap_an / tot_c if tot_c else .5},
            "split": {"seed": cap_cr / tot_c if tot_c else .5,
                      "livestock": cap_an / tot_c if tot_c else .5}}


def CV_OCC(crop):
    from tools import conversion_value as CV
    return CV.occupancy(crop)


split = target_stock          # the old name, now pointing at the stock read


# --------------------------------------------------------------------- CLI
def dawn_view(day, replay=None, seat=0):
    """A REAL dawn observation, read from a recorded game.

    Hand-building an observation means hand-maintaining a copy of the engine's
    schema, and a stale copy prices a state the engine cannot produce. The
    corpus already holds 40 leader games; hour 0 of day `day` in one of them is
    a real state, so the pricing chain is exercised against the same shape it
    will see in play.
    """
    import glob
    import json

    from kagent.gamestate import View

    if replay is None:
        cands = sorted(glob.glob(os.path.join(REPO, "replays", "mmpq", "*.json")))
        if not cands:
            raise SystemExit("no replay to read a dawn state from")
        replay = cands[0]
    with open(replay) as fh:
        steps = json.load(fh)["steps"]
    k = day * 24
    if k >= len(steps):
        raise SystemExit(f"day {day} is past the end of {os.path.basename(replay)}")
    return View(steps[k][seat]["observation"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--replay", default=None, help="read the dawn state from here")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="CVaR level; omit to derive from the position")
    # NOT store_true: that leaves None when omitted, which `split()` reads as
    # "derive from the position" -- so `--alpha 0.2` alone silently took
    # whichever tail the replayed game's margin implied, and the two postures
    # printed identical tables. Three explicit states instead.
    ap.add_argument("--tail", choices=("upper", "lower", "auto"), default="auto",
                    help="upper = behind/seek variance, lower = ahead/protect")
    args = ap.parse_args()

    view = dawn_view(args.day, args.replay, args.seat)
    shops = tuple(view.unlocked_shops)
    upper = None if args.tail == "auto" else (args.tail == "upper")
    out = split(view, args.alpha, upper)
    print(f"day {out['day']}  alpha {out['alpha']:.2f}  "
          f"{'UPPER tail (behind -> seek variance)' if out['upper'] else 'lower tail (ahead -> protect)'}"
          f"   shops {list(shops) or 'none'}\n")
    print(f"{'asset':<12}{'kind':>8}{'decay':>7}{'cost':>7}{'occ':>5}"
          f"{'rent/td':>10}{'neutral':>10}{'spread':>8}{'p_ever':>8}")
    for r in out["rows"]:
        print(f"{r['item']:<12}{r['kind']:>8}{r['decay']:>7.2f}{r['cost']:>7.0f}"
              f"{r['occupancy']:>5}{r['rent_per_tile_day']:>10.1f}"
              f"{r['rent_mean']:>10.1f}{r['spread']:>+8.2f}"
              f"{r['p_demand_ever']:>8.2f}")
    s = out["split"]
    print(f"\nimplied split   seed {s['seed']:.2f}   livestock {s['livestock']:.2f}")


if __name__ == "__main__":
    main()
