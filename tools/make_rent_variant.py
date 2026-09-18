"""Choose the crop for a freed tile by its rent at HARVEST-day price.

Everything the planner does today prices a crop at a constant or at today's
price. Both are wrong for the same reason: a strawberry planted on day 10
sells from day 20, and `tools/market_clock.py` shows strawberry going
$204 -> $82 across exactly that window.

`tools/price_forecast.py` makes the harvest-day price computable to 45%
better than persistence, and the ablation says why it works: the market only
becomes predictable when BOTH farms' boards are in it. So the rent of a tile
is

    rent(crop) = units(crop) * forecast_price(crop, day + max_yield_day)
                 / tile_days(crop)

with `units` read off the engine's own yield rules and the forecast including
the opponent's visible supply.

Why this is not the refuted LP direction: the LP plan is fixed ex ante and
prices output against the whole absorption curve. This is re-solved every day
against the REALISED shop draw and the opponent's REALISED board -- and the
recorded corpus says that is what separates strong players, who move volume
2-4x with the draw while ours is flat (strawberry 236 units across a sixfold
demand range).

alpha blends the rent ranking against the existing quota ladder, and alpha=0
is bit-identical by construction.

    PYTHONPATH=. uv run python tools/make_rent_variant.py --name r-rent50 --alpha 0.5
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

# The freed-tile branch: today it is a fixed ladder (endgame carrots, else
# the wheat engine). This is where a freed tile's crop is decided.
ANCHOR = '''        if day >= knobs["carrot_day"] and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(quota.get("CARROT", 0), min(knobs["carrot_cap"], free))
        elif day > 1 and day <= WHEAT_LAST_PLANT_DAY:
            if day < 10:
                wheat_target = 8
            else:
                wheat_target = min(knobs["wheat_engine"], crops.get("WHEAT", 0) + free)
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))'''

REPL = '''        if day >= knobs["carrot_day"] and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(quota.get("CARROT", 0), min(knobs["carrot_cap"], free))
        elif day > 1 and day <= WHEAT_LAST_PLANT_DAY:
            if day < 10:
                wheat_target = 8
            else:
                wheat_target = min(knobs["wheat_engine"], crops.get("WHEAT", 0) + free)
            _best = _rent_pick(view, day, free)
            if _RENTA > 0 and _best and _best[0] != "WHEAT":
                _n = int(round(_RENTA * free))
                if _n > 0:
                    quota[_best[0]] = quota.get(_best[0], 0) + _n
                    free = max(0, free - _n)
                    wheat_target = min(wheat_target, crops.get("WHEAT", 0) + free)
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))'''

FORECAST = """
# --- forward market model ---------------------------------------------------
# The bundle's econ module already forecasts DEMAND (town_drain_per_day,
# forecast_drain) and has never forecast SUPPLY. The ablation in
# tools/price_forecast.py says that is exactly the wrong half to have: drain
# alone is WORSE than assuming the price never moves (k=5 MAE 25.5 vs 23.1),
# and the model only beats persistence once both farms' boards are in it
# (13.3, a 42% cut). Their shed is worth nothing even as an oracle.
_RENTA = __ALPHA__
_RENT_HORIZON = 14
_PLANTABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def _rent_board_supply(tiles, day, horizon):
    out = [{} for _ in range(horizon + 1)]

    def add(k, item, n):
        if 0 <= k <= horizon and n:
            out[k][item] = out[k].get(item, 0) + n

    n_animals = 0
    for row in tiles:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                held = t.get("yield_units", 0)
                if not cd["ongoing"]:
                    ws = (cd["max_yield_day"] + 1) // 2
                    rem = max(0, cd["max_yield_day"] - max(age, ws - 1))
                    add(cd["max_yield_day"] - age, t["crop"],
                        min(cd["max_yield"], held + rem))
                else:
                    add(0, t["crop"], held)
                    n_done = ((age - cd["first_yield_day"]) // cd["interval"] + 1
                              if age >= cd["first_yield_day"] else 0)
                    nxt = cd["first_yield_day"] + n_done * cd["interval"] - age
                    while n_done < cd["max_yield"] and nxt <= horizon:
                        add(nxt, t["crop"], 1)
                        n_done += 1
                        nxt += cd["interval"]
            elif t.get("animal"):
                n_animals += 1
                a = ANIMALS[t["animal"]]
                add(0, a["product"], t.get("yield_units", 0))
                # CARE doubles output on a fed day; ignoring it under-predicted
                # milk and wool by ~2x (price bias +$125 / +$88 at k=10)
                rate = 2 if (t.get("pending_care_bonus", 0)
                             or t.get("cared_today")) else 1
                age = day - t["placed_day"]
                nxt = (a["interval"] - max(0, age - a["first_yield_day"]) % a["interval"]
                       if age >= a["first_yield_day"] else a["first_yield_day"] - age)
                while nxt <= horizon:
                    add(nxt, a["product"], rate)
                    nxt += a["interval"]
    for k in range(1, horizon + 1):
        add(k, "FERTILIZER", n_animals)   # one per animal per night
        add(k, "WHEAT", -n_animals)       # FEED eats one WHEAT per animal/day
    return out


def _rent_inventory(view, horizon):
    \"\"\"Projected market INVENTORY path, not price: the rent of a block has
    to be priced against the inventory its own units will create.\"\"\"
    cur = dict(view.market_inv)
    drain = econ.town_drain_per_day(view.unlocked_shops)
    day = view.day
    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)
    path = {p: [cur[p]] for p in cur}
    for k in range(1, horizon + 1):
        for p in cur:
            cur[p] += (mine[k].get(p, 0) + opp[k].get(p, 0)
                       - drain.get(p, 0.0))
            path[p].append(cur[p])
    return path


def _rent_pick(view, day, free):
    \"\"\"(crop, rent) with the highest $/tile-day at ITS OWN harvest price.\"\"\"
    if _RENTA <= 0 or free <= 0:
        return None
    try:
        inv = _rent_inventory(view, _RENT_HORIZON)
    except Exception:
        return None
    best = None
    for crop in _PLANTABLE:
        cd = CROPS[crop]
        if cd["ongoing"]:
            span = max(1, 29 - day)
            if span <= cd["first_yield_day"]:
                continue
            k = min(_RENT_HORIZON, cd["first_yield_day"])
            units = min(cd["max_yield"],
                        (span - cd["first_yield_day"]) // cd["interval"] + 1)
            occupies = span
        else:
            if day + cd["max_yield_day"] > 29:
                continue
            k = min(_RENT_HORIZON, cd["max_yield_day"])
            units = cd["max_yield"]
            occupies = cd["max_yield_day"] + 1
        if units <= 0:
            continue
        # Price the WHOLE BLOCK against the inventory its own units create.
        # Pricing at the marginal price instead is what made the first version
        # pick tomato every day: tomato has the highest $/tile-day at our
        # current (zero) volume and collapses past T=200 at scale.
        total = int(round(units * free))
        if total <= 0:
            continue
        rev = econ.sell_revenue(crop, int(round(inv[crop][k])), total)
        rent = rev / float(free * occupies)
        if best is None or rent > best[1]:
            best = (crop, rent)
    return best

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="fraction of each day's free tiles given to the "
                         "highest-rent crop; 0 = base, bit-identical")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    by_mod = dict(mods)
    for need, mod in (("SHOPS =", "constants"), ("CROPS =", "constants"),
                      ("ANIMALS =", "constants"),
                      ("self.market_inv", "gamestate")):
        if need not in by_mod.get(mod, ""):
            raise SystemExit(f"{mod} does not define {need.strip(' =')}")

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("freed-tile anchor not found in planner")
            text = (FORECAST.replace("__ALPHA__", repr(args.alpha))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no planner module in bundle")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + forecast rent on {args.alpha:.0%} of "
          f"freed tiles")


if __name__ == "__main__":
    main()
