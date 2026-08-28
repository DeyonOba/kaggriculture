"""Shared observation encoding and action-head spec for the neural planner.

Used identically at training time (rl/) and inference time (nnpolicy.py) so
train/serve skew is impossible. Pure stdlib.

The RL agent acts once per day: a masked multi-discrete decision that fills
the strategic fields of DayPlan; the scripted scheduler/market executors
handle the per-turn micro. Masks make rule-violating or infeasible choices
unsamplable (Huang & Ontanon: mask before softmax keeps the gradient
unbiased).
"""

import math

from kagent.constants import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    PRODUCTS,
    SHOPS,
)
from kagent import econ
from kagent.planner import apply_land_pressure

# ---- Action heads -----------------------------------------------------------
# name -> list of choice values. The network emits one categorical per head.
# Crop/animal heads are ABSOLUTE standing targets (tiles on the farm, animals
# in the herd), not per-day deltas: targets are stable across days, so the BC
# distribution isn't drowned in zeros and argmax play is coherent. The daily
# delta (plant quota / buys) is derived in apply_to_plan.
HEADS = [
    ("melon_tiles", [0, 6, 12, 18, 24]),
    ("wheat_tiles", [0, 6, 12, 18, 24]),
    ("carrot_tiles", [0, 4, 8, 12, 16]),
    ("tomato_tiles", [0, 4, 8, 12]),
    ("straw_tiles", [0, 4, 8, 12]),
    ("goose_target", [0, 8, 16, 24, 32]),
    ("cow_target", [0, 2, 4, 8]),
    ("sheep_target", [0, 3, 6, 10]),
    ("hires", [4, 8, 12, 14, 16]),
    ("land", [0, 1]),
    ("melon_floor", [1, 40, 80, 120, 180]),
    # Multipliers on the heuristic's adaptive sell floors: hold out for
    # scarcity spikes (>1) or liquidate into current demand (<1).
    ("premium_floor_scale", [0.6, 1.0, 1.4]),   # STRAWBERRY / MILK / WOOL
    ("staple_floor_scale", [0.7, 1.0, 1.3]),    # CARROT / TOMATO
]
HEAD_SIZES = [len(vals) for _, vals in HEADS]
N_ACTION_LOGITS = sum(HEAD_SIZES)

QUOTA_CROP = {
    "melon_tiles": "MELON",
    "wheat_tiles": "WHEAT",
    "carrot_tiles": "CARROT",
    "tomato_tiles": "TOMATO",
    "straw_tiles": "STRAWBERRY",
}
BUY_KIND = {"goose_target": "GOOSE", "cow_target": "COW", "sheep_target": "SHEEP"}

PLANTS_PER_DAY_CAP = 12   # labor limit on new plantings per day
BUYS_PER_DAY_CAP = 10

# Last day each crop is still worth planting (harvest must land by day 29).
LAST_PLANT = {"MELON": 19, "WHEAT": 25, "CARROT": 26, "TOMATO": 18, "STRAWBERRY": 13}

SHOP_NAMES = sorted(SHOPS)


def supply_projection(farm, day, horizon=7):
    """Rough units of each product this farm brings to market over the next
    `horizon` days from standing crops and animals (public info for both)."""
    horizon = max(0, min(horizon, 29 - day))
    supply = {p: 0.0 for p in PRODUCTS}
    for _, _, t in farm.plants():
        crop = CROPS.get(t.get("crop"))
        if not crop:
            continue
        age = day - t.get("planted_day", day)
        yld = t.get("yield_units", 0)
        if crop["ongoing"]:
            per_day = 1.0 / (crop["interval"] + 1)
            supply[t["crop"]] += min(crop["max_yield"] * 2, yld + per_day * horizon)
        else:
            window_start = (crop["max_yield_day"] + 1) // 2
            bonus_left = max(
                0, min(crop["max_yield_day"], age + horizon) - max(age, window_start)
            )
            supply[t["crop"]] += min(crop["max_yield"], yld + bonus_left + 1)
    for _, _, t in farm.animals():
        a = ANIMALS.get(t.get("animal"))
        if not a:
            continue
        supply[a["product"]] += t.get("yield_units", 0) + horizon / a["interval"]
        supply["FERTILIZER"] += horizon
    return supply


def _crop_stats(farm, day):
    stats = {}
    for crop in CROPS:
        stats[crop] = [0, 0.0, 0]  # count, mean age, harvestable yield
    for _, _, t in farm.plants():
        crop = t.get("crop")
        if crop not in stats:
            continue
        s = stats[crop]
        s[0] += 1
        s[1] += day - t.get("planted_day", day)
        s[2] += t.get("yield_units", 0)
    return stats


def encode(view, memory):
    """View (+Memory) -> flat feature list."""
    f = []
    day = view.day
    me, opp = view.me, view.opp

    # Time
    f.append(day / 30.0)
    f.append(view.hour / 24.0)
    f.append(view.steps_left / 718.0)

    # My economy
    f.append(math.log1p(max(0.0, me.money)) / 10.0)
    f.append(len(me.units) / 16.0)
    f.append(me.hires_today / 16.0)
    f.append(len(me.unlocked_quadrants) / 4.0)

    my_crops = _crop_stats(me, day)
    for crop in CROPS:
        c, age_sum, yld = my_crops[crop]
        f.append(c / 12.0)
        f.append((age_sum / c if c else 0.0) / 12.0)
        f.append(yld / 30.0)

    herd = {k: 0 for k in ANIMALS}
    empty_structs = {"COOP": 0, "PASTURE": 0}
    pending_yield = 0
    for _, _, t in me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1
        pending_yield += t.get("yield_units", 0)
    for _, _, t in me.empty_structures():
        empty_structs[t["kind"]] = empty_structs.get(t["kind"], 0) + 1
    for k in ANIMALS:
        f.append(herd[k] / 16.0)
    f.append(empty_structs["COOP"] / 10.0)
    f.append(empty_structs["PASTURE"] / 10.0)
    f.append(pending_yield / 60.0)
    f.append(sum(1 for _ in me.empty_tiles()) / 60.0)
    f.append(sum(1 for _ in me.weeds()) / 20.0)

    # Shed / seeds
    for item in PRODUCTS:
        f.append(view.shed.get(item, 0) / 30.0)
    for crop in CROPS:
        f.append(view.seeds.get(crop, 0) / 10.0)
    f.append(view.shed_total / 100.0)

    # Market: normalized inventory displacement and price ratio per product
    for item in PRODUCTS:
        p = MARKET_PARAMS[item]
        inv = view.market_inv.get(item, p["I0"])
        f.append(max(-3.0, min(3.0, (inv - p["I0"]) / p["T"])) / 3.0)
        f.append(view.market_prices.get(item, p["base"]) / (2.0 * p["base"]))

    # Town demand
    for shop in SHOP_NAMES:
        f.append(view.unlocked_shops.count(shop) / 3.0)

    # Opponent (public)
    f.append(math.log1p(max(0.0, opp.money)) / 10.0)
    f.append(len(opp.units) / 16.0)
    f.append(len(opp.unlocked_quadrants) / 4.0)
    opp_crops = _crop_stats(opp, day)
    for crop in CROPS:
        c, _, yld = opp_crops[crop]
        f.append(c / 12.0)
        f.append(yld / 30.0)
    f.append(sum(1 for _ in opp.animals()) / 16.0)

    # Demand vs supply: my and opponent's projected 7-day output per product,
    # town drain per day, the resulting gap, and price curvature (what the
    # 12th marginal unit would fetch). This is the "harvest over the field
    # relative to demand" signal.
    my_supply = supply_projection(me, day)
    opp_supply = supply_projection(opp, day)
    drain = econ.town_drain_per_day(view.unlocked_shops)
    horizon = max(0.0, min(7.0, 29 - day))
    for item in PRODUCTS:
        p = MARKET_PARAMS[item]
        f.append(my_supply[item] / 40.0)
        f.append(opp_supply[item] / 40.0)
        f.append(drain.get(item, 0.0) / 12.0)
        gap = drain.get(item, 0.0) * horizon - my_supply[item] - opp_supply[item]
        f.append(max(-2.0, min(2.0, gap / 40.0)))
        inv = view.market_inv.get(item, p["I0"])
        f.append(econ.price(item, inv + 12) / (2.0 * p["base"]))

    # Land expansion economics.
    next_idx = sum(1 for q in LAND_ORDER if q in me.unlocked_quadrants)
    if next_idx < len(LAND_ORDER):
        price = LAND_PRICES[next_idx]
        f.append(price / 4000.0)
        f.append(max(-1.0, min(1.0, (me.money - price) / 2000.0)))
    else:
        f.append(0.0)
        f.append(0.0)

    # Episode memory
    f.append(getattr(memory, "melons_sold", 0) / 150.0)

    return f


N_FEATURES = None  # resolved lazily on first encode (kept in sync by tests)


def masks(view):
    """Per-head validity masks (list of lists of 0/1), same order as HEADS.

    Targets are self-clamping in apply_to_plan (a target below the current
    count is simply a no-op), so most heads stay fully valid; only choices
    that are strictly impossible are masked.
    """
    me = view.me
    money = me.money
    out = []
    for name, vals in HEADS:
        if name == "land":
            owned = me.unlocked_quadrants
            next_idx = sum(1 for q in LAND_ORDER if q in owned)
            can = next_idx < len(LAND_ORDER) and money >= LAND_PRICES[next_idx] + 400
            m = [1 if (v == 0 or can) else 0 for v in vals]
        else:
            m = [1] * len(vals)
        if not any(m):
            m[0] = 1
        out.append(m)
    return out


def apply_to_plan(plan, choices, view):
    """Overwrite the strategic fields of a heuristic DayPlan with the network's
    per-head choices (indices into HEADS values). Safety clamps stay scripted:
    feed capacity, structure pipelines, wheat reserve, and liquidation are
    computed downstream exactly as in the heuristic planner."""
    values = {name: vals[choices[i]] for i, (name, vals) in enumerate(HEADS)}

    crop_counts = {}
    for _, _, t in view.me.plants():
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1

    quota = {}
    budget_day = PLANTS_PER_DAY_CAP
    for head, crop in QUOTA_CROP.items():
        target = values[head]
        deficit = max(0, target - crop_counts.get(crop, 0))
        n = min(deficit, budget_day)
        if n > 0 and view.day <= LAST_PLANT[crop] and not plan.liquidation:
            quota[crop] = n
            budget_day -= n
    plan.plant_quota = quota

    herd = {}
    for _, _, t in view.me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1

    plan.animal_buys = {}
    feed_capacity = view.shed.get("WHEAT", 0) + 3 * crop_counts.get("WHEAT", 0) + 10
    herd_total = sum(herd.values())
    feedable = max(0, feed_capacity - herd_total)
    budget = max(0.0, view.me.money - 300)
    buys_day = BUYS_PER_DAY_CAP
    for head, kind in BUY_KIND.items():
        target = values[head]
        deficit = max(0, target - herd.get(kind, 0) - view.shed.get(kind, 0))
        n = max(0, min(deficit, feedable, buys_day, int(budget // ANIMALS[kind]["cost"])))
        if n > 0 and not plan.liquidation and view.day <= 24:
            plan.animal_buys[kind] = n
            feedable -= n
            buys_day -= n
            budget -= n * ANIMALS[kind]["cost"]

    # Structure pipeline per kind (same logic as heuristic planner).
    empty_by_kind = {"COOP": 0, "PASTURE": 0}
    for _, _, t in view.me.empty_structures():
        empty_by_kind[t["kind"]] = empty_by_kind.get(t["kind"], 0) + 1
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = plan.animal_buys.get(kind, 0) + view.shed.get(kind, 0)
        need = pipeline - empty_by_kind.get(structure, 0)
        if need > 0:
            plan.structure_builds[structure] += need
            empty_by_kind[structure] = 0
        else:
            empty_by_kind[structure] = -need

    plan.hires_target = values["hires"]

    plan.land_buys = []
    owned = view.me.unlocked_quadrants
    next_idx = sum(1 for q in LAND_ORDER if q in owned)
    if next_idx < len(LAND_ORDER):
        # Learned head, backed by the heuristic's tuned land-crunch fallback
        # so the policy can never simply refuse to expand while tiles run out.
        empty = sum(1 for _ in view.me.empty_tiles())
        crunch = (
            view.day <= 22
            and empty < 12
            and view.me.money >= LAND_PRICES[next_idx] + 600
        )
        if values["land"] == 1 or crunch:
            plan.land_buys.append(LAND_ORDER[next_idx])

    plan.sell_floors = dict(plan.sell_floors)
    plan.sell_floors["MELON"] = (
        1 if view.step >= 714 else values["melon_floor"]
    )
    # Learned scaling of the adaptive heuristic floors (never below $1, and
    # forced to $1 in the final liquidation push).
    if view.step < 714:
        for item in ("STRAWBERRY", "MILK", "WOOL"):
            plan.sell_floors[item] = max(1, int(plan.sell_floors[item] * values["premium_floor_scale"]))
        for item in ("CARROT", "TOMATO"):
            plan.sell_floors[item] = max(1, int(plan.sell_floors[item] * values["staple_floor_scale"]))

    # Scripted safety net: expand land when planned work exceeds open tiles.
    apply_land_pressure(plan, view)
    return plan
