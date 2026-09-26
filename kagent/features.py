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
import os

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
# Scale-critical lines (wheat, strawberry, cow, sheep) are DEMAND-ANCHORED:
# anchors(view) computes supply=demand targets from the drain forecast and
# the head chooses an OFFSET around that anchor (field autopsy: we produced
# 1/3-1/5 of same-priced opponents' volume — "crops planted per demand low").
# Other lines stay absolute targets.
ABS_HEADS = [
    ("melon_tiles", [0, 6, 12, 18, 24]),
    ("wheat_tiles", [-16, -8, 0, 8, 16]),       # offset around wheat anchor
    ("carrot_tiles", [0, 6, 12, 18, 26]),
    ("tomato_tiles", [0, 4, 8, 12]),
    ("straw_tiles", [-16, -8, 0, 8, 16]),       # offset around strawberry anchor
    # Granularity fixed 2026-09-26. This head used to be [0, 8, 16, 24, 32]:
    # the three leaders run 4.6-5.9 geese, so the smallest thing the planner
    # could ask for was a 33-60% overshoot -- and 8 is the herd size that
    # lost 8-12 points in the (mis-specified) geese test. A head whose only
    # options are "none" or "more than anyone runs" learns "none", and the
    # kernel-trained net duly chose 0 on all 30 days of all 3 traced games.
    # Index 0 is still 0, so a net that always chose "none" is unchanged.
    ("goose_target", [0, 4, 6, 10, 16]),
    ("cow_target", [-8, -4, 0, 4, 8]),          # offset around cow anchor
    ("sheep_target", [-6, -3, 0, 3, 6]),        # offset around sheep anchor
    ("hires", [4, 8, 12, 14, 16]),
    ("land", [0, 1]),
    ("melon_floor", [1, 40, 80, 120, 180]),
    # Multipliers on the heuristic's adaptive sell floors: hold out for
    # scarcity spikes (>1) or liquidate into current demand (<1).
    ("premium_floor_scale", [0.6, 1.0, 1.4]),   # STRAWBERRY / MILK / WOOL
    ("staple_floor_scale", [0.7, 1.0, 1.3]),    # CARROT / TOMATO
]

ANCHORED = {"wheat_tiles": "WHEAT", "straw_tiles": "STRAWBERRY",
            "cow_target": "MILK", "sheep_target": "WOOL"}
ANCHOR_CAPS = {"wheat_tiles": 44, "straw_tiles": 40, "cow_target": 18, "sheep_target": 14}

# Care-boosted output per animal per day (fed+cared daily): production
# 1 + banked bonus per cycle => ~interval-independent ~1.4x/interval.
_OUT_PER_DAY = {"MILK": 1.5, "WOOL": 1.33}
_STRAW_PER_TILE_DAY = 0.5   # fertilized+watered ongoing strawberry


def anchors(view):
    """Supply=demand production anchors: season demand forecast MINUS the
    opponent's projected supply (their farm is public). Sharing a demand pool
    the opponent already saturates just crashes the price for both — field
    autopsy: 110 milk sold at $13-45 into a jointly-flooded market while the
    winner pivoted lines."""
    day = view.day
    days_left = max(1, 29 - day)
    fc = econ.forecast_drain(view.unlocked_shops, day)
    opp = supply_projection(view.opp, day, horizon=29 - day)
    herd_total = sum(1 for _ in view.me.animals())

    def net_per_day(item):
        return max(0.0, fc.get(item, 0.0) - opp.get(item, 0.0)) / days_left

    def crashed(item):
        base = MARKET_PARAMS[item]["base"]
        return view.market_prices.get(item, base) < 0.35 * base

    out = {}
    prod_days = max(0, 28 - day - 8)
    out["cow_target"] = 0 if (prod_days < 4 or crashed("MILK")) else min(
        ANCHOR_CAPS["cow_target"], round(net_per_day("MILK") / _OUT_PER_DAY["MILK"])
    )
    prod_days = max(0, 28 - day - 6)
    out["sheep_target"] = 0 if (prod_days < 6 or crashed("WOOL")) else min(
        ANCHOR_CAPS["sheep_target"], round(net_per_day("WOOL") / _OUT_PER_DAY["WOOL"])
    )
    out["straw_tiles"] = 0 if (day > 13 or crashed("STRAWBERRY")) else min(
        ANCHOR_CAPS["straw_tiles"], round(net_per_day("STRAWBERRY") / _STRAW_PER_TILE_DAY)
    )
    out["wheat_tiles"] = 0 if day > 25 else min(
        ANCHOR_CAPS["wheat_tiles"], herd_total + 20
    )
    return out
# RESIDUAL mode (champ-h3 lineage): every head is an OFFSET applied to the
# champion plan (planner.make_champion_plan); all-zero choices reproduce the
# shipped heuristic exactly, so a trained policy can only learn deviations
# that pay (residual policy learning). melon_floor and the floor scales stay
# absolute. `land` = 1 holds today's land purchase.
RESIDUAL_HEADS = [
    ("melon_tiles", [-6, -3, 0, 3]),
    ("wheat_tiles", [-16, -8, 0, 8]),
    ("carrot_tiles", [-8, 0, 8, 16]),
    ("tomato_tiles", [0, 4, 8]),
    ("straw_tiles", [-12, -6, 0, 6]),
    ("goose_target", [0, 2, 4]),
    ("cow_target", [-4, -2, 0, 2]),
    ("sheep_target", [-4, -2, 0, 2]),
    ("hires", [-4, -2, 0, 2]),
    ("land", [0, 1]),
    ("melon_floor", [1, 40, 80, 120, 180]),
    ("premium_floor_scale", [0.6, 1.0, 1.4]),
    ("staple_floor_scale", [0.7, 1.0, 1.3]),
]

MODE = "absolute"
HEADS = ABS_HEADS
HEAD_SIZES = [len(vals) for _, vals in HEADS]
N_ACTION_LOGITS = sum(HEAD_SIZES)


def set_mode(mode):
    """Select the head set. Consumers read HEADS / HEAD_SIZES /
    N_ACTION_LOGITS through the module at call time, so this must run before
    a network is built or a bundle's provider is constructed."""
    global MODE, HEADS, HEAD_SIZES, N_ACTION_LOGITS
    MODE = mode
    HEADS = RESIDUAL_HEADS if mode == "residual" else ABS_HEADS
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

PLANTS_PER_DAY_CAP = 20   # labor limit on new plantings per day
BUYS_PER_DAY_CAP = 12
HERD_CAP = 60             # hard sanity ceiling on total animals

# Last day an animal purchase can still pay for itself (lead time + at least
# two productions before day 29). Field autopsy: the policy once bought 4
# cows on day 22 — first milk would have landed on day 30, after the season.
ANIMAL_LAST_BUY = {"GOOSE": 24, "COW": 18, "SHEEP": 20}

# Days from investment to first sellable unit, per product (lead times).
FIRST_YIELD = {
    "WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10,
    "EGG": 4, "MILK": 8, "WOOL": 6, "FERTILIZER": 1,
}

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

    # Demand FORECAST: expected drain per product from now to season end,
    # counting shops that will open later (uniform draws every 3 days), the
    # season-scale gap vs both farms' projected output, and the lead-time
    # window left for a new investment in that product. This is the
    # "a store opens in i days — what will it want, and can I still supply
    # it profitably" signal.
    fc = econ.forecast_drain(view.unlocked_shops, day)
    my_season = supply_projection(me, day, horizon=29 - day)
    opp_season = supply_projection(opp, day, horizon=29 - day)
    for item in PRODUCTS:
        f.append(min(3.0, fc.get(item, 0.0) / 150.0))
        season_gap = fc.get(item, 0.0) - my_season[item] - opp_season[item]
        f.append(max(-2.0, min(2.0, season_gap / 150.0)))
        window = max(0, 29 - day - FIRST_YIELD[item])
        f.append(window / 29.0)
    f.append(econ.remaining_unlocks(day, len(view.unlocked_shops)) / 8.0)
    next_unlock = next((u for u in (3, 6, 9, 12, 15, 18, 21, 24) if u > day), 30)
    f.append(min(1.0, (next_unlock - day) / 6.0))

    # Land expansion economics.
    next_idx = sum(1 for q in LAND_ORDER if q in me.unlocked_quadrants)
    if next_idx < len(LAND_ORDER):
        price = LAND_PRICES[next_idx]
        f.append(price / 4000.0)
        f.append(max(-1.0, min(1.0, (me.money - price) / 2000.0)))
    else:
        f.append(0.0)
        f.append(0.0)

    # Labor productivity: yesterday's idle fraction and hands paid for —
    # the "unfruitful hires" signal, measured where it is economically real.
    f.append(getattr(memory, "yesterday_idle", 0.0))
    f.append(getattr(memory, "yesterday_hands", 0) / 16.0)

    # The policy's own demand anchors (its offset heads act around these).
    a = anchors(view)
    f.append(a["wheat_tiles"] / 44.0)
    f.append(a["straw_tiles"] / 40.0)
    f.append(a["cow_target"] / 18.0)
    f.append(a["sheep_target"] / 14.0)

    # Episode memory
    f.append(getattr(memory, "melons_sold", 0) / 150.0)

    return f


N_FEATURES = None  # resolved lazily on first encode (kept in sync by tests)

# ---- Labour-gated land -----------------------------------------------------
# Stage B2 traces, ours vs a 1000-1500 opponent, days 10-25: the same eleven
# hands, but we open a FOURTH quadrant and leave 16-21 tiles idle all mid-game
# while they work three quadrants with 4-5 idle. The fourth quadrant gated
# -$6,266 / -$6,017 replicated (memory: labour gates land) and the neural
# land head buys it anyway. KAGG_MAX_QUADRANTS=N caps ownership at N: the
# head's "buy" choice is masked, and the land-crunch fallback in apply_to_plan
# -- which buys regardless of the head once idle tiles fall below 12, i.e.
# exactly the state above -- is suppressed. 0 (default) is no cap, so bundles
# and every existing run are unchanged. Read at import so Pool workers and the
# eval inherit it.
MAX_QUADRANTS = int(os.environ.get("KAGG_MAX_QUADRANTS", "0"))


def land_allowed(view):
    """May this farm open another quadrant under the cap?"""
    return MAX_QUADRANTS <= 0 or len(view.me.unlocked_quadrants) < MAX_QUADRANTS


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
        if name == "land" and MODE != "residual":
            owned = me.unlocked_quadrants
            next_idx = sum(1 for q in LAND_ORDER if q in owned)
            can = (next_idx < len(LAND_ORDER) and money >= LAND_PRICES[next_idx] + 400
                   and land_allowed(view))
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

    # Resolve offset heads against the demand anchors.
    anchor_vals = anchors(view)
    for head in ANCHORED:
        values[head] = max(0, min(ANCHOR_CAPS[head], anchor_vals[head] + values[head]))

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
    # Feed pipeline: shed stock + growing wheat + what market top-ups sustain.
    feed_capacity = view.shed.get("WHEAT", 0) + 3 * crop_counts.get("WHEAT", 0) + 20
    herd_total = sum(herd.values())
    feedable = max(0, feed_capacity - herd_total)
    headroom = max(0, HERD_CAP - herd_total)
    # Cash floor: field replays showed day-0 all-ins starving the agent for a
    # week (no hires, no feed wheat — cows escaped by day 8 in rated games).
    budget = max(0.0, view.me.money - 600)
    buys_day = BUYS_PER_DAY_CAP
    for head, kind in BUY_KIND.items():
        target = values[head]
        deficit = max(0, target - herd.get(kind, 0) - view.shed.get(kind, 0))
        n = max(0, min(deficit, feedable, buys_day, headroom, int(budget // ANIMALS[kind]["cost"])))
        if n > 0 and not plan.liquidation and view.day <= ANIMAL_LAST_BUY[kind]:
            plan.animal_buys[kind] = n
            feedable -= n
            buys_day -= n
            headroom -= n
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
    if next_idx < len(LAND_ORDER) and land_allowed(view):
        # Learned head, backed by the heuristic's tuned land-crunch fallback
        # so the policy can never simply refuse to expand while tiles run out.
        # Both sit behind the KAGG_MAX_QUADRANTS cap.
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
    # Third purchase path (after the head and the crunch): the demand-driven
    # safety net. It bought the fourth quadrant on day 14 in the first capped
    # traces while the head read 0 -- Stage B5 ran with this leak live.
    if land_allowed(view):
        apply_land_pressure(plan, view)
    return plan


def apply_residual(plan, choices, view):
    """Residual mode: `plan` is the champion plan; every head is an offset on
    its field. Zero offsets leave the plan untouched. Scripted gates (last
    planting day, animal deadlines, herd cap, liquidation) still apply."""
    values = {name: vals[choices[i]] for i, (name, vals) in enumerate(HEADS)}
    day = view.day

    for head, crop in QUOTA_CROP.items():
        off = values[head]
        if not off:
            continue
        q = max(0, plan.plant_quota.get(crop, 0) + off)
        if plan.liquidation or day > LAST_PLANT[crop]:
            q = 0
        if q:
            plan.plant_quota[crop] = q
        else:
            plan.plant_quota.pop(crop, None)

    herd = {}
    for _, _, t in view.me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1
    herd_total = sum(herd.values())
    for head, kind in BUY_KIND.items():
        off = values[head]
        if not off:
            continue
        n = max(0, plan.animal_buys.get(kind, 0) + off)
        n = min(n, max(0, HERD_CAP - herd_total))
        if plan.liquidation or day > ANIMAL_LAST_BUY[kind]:
            n = 0
        if n:
            plan.animal_buys[kind] = n
        else:
            plan.animal_buys.pop(kind, None)
        herd_total += n
    # Structure pipeline per kind, recomputed for the adjusted buys.
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

    plan.hires_target = max(2, plan.hires_target + values["hires"])
    if values["land"] == 1 or not land_allowed(view):
        plan.land_buys = []

    plan.sell_floors = dict(plan.sell_floors)
    if view.step < 714:
        plan.sell_floors["MELON"] = values["melon_floor"]
        for item in ("STRAWBERRY", "MILK", "WOOL"):
            plan.sell_floors[item] = max(1, int(plan.sell_floors[item] * values["premium_floor_scale"]))
        for item in ("CARROT", "TOMATO"):
            plan.sell_floors[item] = max(1, int(plan.sell_floors[item] * values["staple_floor_scale"]))
    return plan


def residual_choices(plan, view, head_masks):
    """BC targets in residual mode: the champion plan itself, i.e. zero
    offsets (melon floor nearest the plan's, scales 1.0), projected onto the
    valid mask."""
    choices = []
    for (name, vals), mask in zip(HEADS, head_masks):
        if name == "melon_floor":
            target = plan.sell_floors.get("MELON", 80)
        elif name.endswith("_floor_scale"):
            target = 1.0
        else:
            target = 0
        valid = [i for i in range(len(vals)) if mask[i]]
        choices.append(min(valid, key=lambda i: abs(vals[i] - target)))
    return choices


def context(view):
    """Compact, human-readable state summary for the counterfactual bandit
    (rule extraction). The 168-dim encode() stays the model input."""
    from kagent.constants import SHOPS, MARKET_PARAMS
    consumers = {}
    for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG"):
        consumers[item] = sum(1 for shop in view.unlocked_shops if item in SHOPS.get(shop, ()))
    price_ratio = {
        item: round(view.market_prices.get(item, MARKET_PARAMS[item]["base"]) / MARKET_PARAMS[item]["base"], 3)
        for item in MARKET_PARAMS
    }
    crops, herd, opp_crops, opp_herd = {}, {}, {}, {}
    for _, _, t in view.me.plants():
        crops[t["crop"]] = crops.get(t["crop"], 0) + 1
    for _, _, t in view.me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1
    for _, _, t in view.opp.plants():
        opp_crops[t["crop"]] = opp_crops.get(t["crop"], 0) + 1
    for _, _, t in view.opp.animals():
        opp_herd[t["animal"]] = opp_herd.get(t["animal"], 0) + 1
    return {
        "day": view.day, "money": round(view.me.money), "land": len(view.me.unlocked_quadrants),
        "hands": len(view.me.units) - 1, "shops": list(view.unlocked_shops),
        "consumers": consumers, "price_ratio": price_ratio,
        "crops": crops, "herd": herd, "opp_crops": opp_crops, "opp_herd": opp_herd,
        "opp_money": round(view.opp.money), "shed_total": view.shed_total,
        "opp_land": len(view.opp.unlocked_quadrants),
    }
