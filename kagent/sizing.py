"""Plan sizing: the dawn plan is APPROVED against a budget and a crew, in a
fixed claim order, and only the approved part is asked (rlv3/EXECUTOR_V2.md,
section 6). Measured on the leaders' 40 winner seats vs h11 (size_ledger):
neither side sizes to dawn cash -- both are broke at every dawn through day
10 and spend 20-28x it, funded by the day's sales -- so the budget is dawn
cash plus today's expected sales, and the claims are, in order:

  1. feed for the whole herd at the market price (the leaders' first claim:
     30% of their spend, 4-6x ours),
  2. hands to the work: (standing tiles + today's plantings + 3 per animal)
     / CAP ops per unit-day, capped by the wage curve,
  3. seeds for every bare tile -- a tile whose planned crop is unaffordable
     today is planted with the best affordable crop instead of waiting (we
     hold 7-8 tiles bare from day 7 to 15 for $100 strawberry seed on $29
     dawns; the leaders hold 0.2-2.4),
  4. animals with what is left (structures are free), 5. land last.

Four flags (scheduler.EXEC): size_budget, size_feed, size_hires, size_ground;
`sized` turns on all four. Each is gated alone and as the package.
"""
from __future__ import annotations

import math

from kagent import econ
from kagent.constants import ANIMALS, CROPS, TURNS_PER_DAY
from kagent import scheduler

CAP = 8.0             # productive ops one unit does in a day (measured 8.9 incl. idle; leaders carry 8.5-10.5 work units per unit)
MIN_HANDS = 2
MAX_HANDS = 11        # the wage knee: 11 hands cost $232 a day
HIRE_SHARE = 0.5      # the hire bill may take at most this share of the budget
SALES_DISCOUNT = 0.8  # the opponent sells into the same market: expected sales are haircut
FILL_ORDER = ("WHEAT", "CARROT")            # the affordable crops, cheapest first
LAST_PLANT = {"WHEAT": 25, "CARROT": 26}    # the market's planting windows


def on(flag):
    return bool(scheduler.EXEC.get("sized") or scheduler.EXEC.get(flag))


def expected_sales(view, plan):
    """What the shed sells today at the exact market identity, plus the
    one-time harvests that come off today, haircut by SALES_DISCOUNT."""
    total = 0.0
    inv = view.market_inv
    for item, n in view.shed.items():
        if n <= 0 or item in ANIMALS:
            continue
        if item == "WHEAT":
            n = max(0, n - plan.wheat_reserve)
        total += econ.sell_revenue(item, inv.get(item, 0), n)
    day = view.day
    for _, _, t in view.me.plants():
        cd = CROPS.get(t["crop"])
        if cd is None or cd["ongoing"]:
            continue
        age = day - t.get("planted_day", day)
        if age >= cd["max_yield_day"] or 0 <= t.get("max_lifespan_step", -1) <= (day + 1) * TURNS_PER_DAY:
            units = t.get("yield_units", 0)
            if units > 0:
                total += econ.sell_revenue(t["crop"], inv.get(t["crop"], 0), units)
    return SALES_DISCOUNT * total


def record_spend(view, memory, orders):
    """Spend at list prices, by day, so approve_plan can read yesterday's
    realised income (money delta + spend) as the forecast floor."""
    from kagent.constants import LAND_PRICES, LAND_ORDER
    hist = memory.size_spend = getattr(memory, "size_spend", {})
    tot = hist.get(view.day, 0.0)
    for o in orders:
        if not o:
            continue
        if o[0] == "BUY_SEED" and len(o) > 2 and o[1] in CROPS:
            tot += CROPS[o[1]]["seed"] * int(o[2])
        elif o[0] == "BUY_ANIMAL" and len(o) > 2 and o[1] in ANIMALS:
            tot += ANIMALS[o[1]]["cost"] * int(o[2])
        elif o[0] == "HIRE":
            tot += econ.hire_marginal_cost(view.me.hires_today)
        elif o[0] == "BUY_PRODUCT" and len(o) > 2:
            tot += econ.buy_cost(o[1], view.market_inv.get(o[1], 0), int(o[2]))
        elif o[0] == "BUY_LAND":
            owned = sum(1 for q in LAND_ORDER if q in view.me.unlocked_quadrants)
            tot += LAND_PRICES[min(owned, len(LAND_PRICES) - 1)]
    hist[view.day] = tot


def income_yesterday(view, memory):
    dawn = memory.size_dawn = getattr(memory, "size_dawn", {})
    dawn[view.day] = float(view.me.money)
    spend = getattr(memory, "size_spend", {})
    best = None
    for y in range(view.day - 3, view.day):
        if y in dawn and y + 1 in dawn:
            inc = max(0.0, dawn[y + 1] - dawn[y] + spend.get(y, 0.0))
            best = inc if best is None else max(best, inc)
    return best


def fill_crop_today(view):
    """The cheapest crop whose planting window is open today, or None."""
    for crop in FILL_ORDER:
        if view.day <= LAST_PLANT[crop] and view.day + CROPS[crop]["first_yield_day"] <= 29:
            return crop
    return None


def refill(view, plan, memory):
    """Every hour under size_ground: ground that came free since dawn (a
    harvest the dawn could not see) is asked for again with the fill crop,
    so a tile never waits for the next dawn. Seeds are bought next turn."""
    if plan.liquidation or view.day >= 29 or view.hour > 19:
        return 0
    crop = fill_crop_today(view)
    if crop is None:
        return 0
    bare = sum(1 for _ in view.me.empty_tiles())
    deficit = bare - sum(plan.plant_quota.values())
    if deficit <= 0:
        return 0
    seed = CROPS[crop]["seed"]
    # the fill crop's whole quota stays within seeds on hand plus the cash
    cover = view.seeds.get(crop, 0) + int(max(0.0, float(view.me.money)) // seed) - plan.plant_quota.get(crop, 0)
    n = min(deficit, cover)
    if n <= 0:
        return 0
    plan.plant_quota[crop] = plan.plant_quota.get(crop, 0) + n
    return n


def ripe_today(view):
    day = view.day
    n = 0
    for _, _, t in view.me.plants():
        cd = CROPS.get(t["crop"])
        if cd is None or cd["ongoing"]:
            continue
        age = day - t.get("planted_day", day)
        if age >= cd["max_yield_day"] or 0 <= t.get("max_lifespan_step", -1) <= (day + 1) * TURNS_PER_DAY:
            n += 1
    return n


def approve_plan(view, plan, memory):
    """Cut the plan's asks to the budget and the crew; mark it approved."""
    me = view.me
    day = view.day
    log = {"day": day, "money": float(me.money)}
    if plan.liquidation or day >= 29:
        return log
    herd = sum(1 for _ in me.animals())
    standing = sum(1 for _ in me.plants())
    bare = sum(1 for _ in me.empty_tiles()) + ripe_today(view) - sum(plan.structure_builds.values())
    bare = max(0, bare)

    budget = float(me.money)
    if on("size_budget"):
        model = expected_sales(view, plan)
        realised = income_yesterday(view, memory)
        # the model sees the dawn shed and today's one-time harvests; the
        # milk, wool, eggs and fertiliser made during the day it cannot, so
        # yesterday's realised income is the floor of the forecast
        forecast = model if realised is None else max(model, SALES_DISCOUNT * realised)
        budget += forecast
        log["forecast"] = forecast
    log["budget"] = budget

    # 1. feed: first claim, any price
    if on("size_feed"):
        plan.feed_any = True
        wheat = view.shed.get("WHEAT", 0) + sum(view.unit_inv(i).get("WHEAT", 0) for i, _, _ in me.units)
        need = max(0, plan.feed_days * herd - wheat)
        if need > 0:
            feed_cost = econ.buy_cost("WHEAT", view.market_inv.get("WHEAT", 0), need)
            budget -= feed_cost
            log["feed"] = feed_cost

    # 2. hands to the work
    if on("size_hires"):
        plantings = min(sum(plan.plant_quota.values()), bare)
        work = standing + plantings + 3 * (herd + sum(plan.animal_buys.values()))
        units = int(math.ceil(work / CAP))
        hands = max(MIN_HANDS, min(MAX_HANDS, units - 1))
        while hands > MIN_HANDS and econ.hire_total_cost(hands) > HIRE_SHARE * max(0.0, budget):
            hands -= 1
        plan.hires_target = hands
        log["work"] = work
    budget -= econ.hire_total_cost(plan.hires_target)
    log["hires"] = plan.hires_target

    # 3. the ground: the cheapest open crop covers every bare tile first (a
    #    tile is never left waiting for an expensive seed), and seeds already
    #    on hand are planted at no cost
    fill_crop = fill_crop_today(view) if on("size_ground") else None
    approved = {}
    room = bare
    if on("size_ground") or on("size_budget"):
        for crop, q in plan.plant_quota.items():
            n = min(q, view.seeds.get(crop, 0), room)
            if n > 0:
                approved[crop] = n
                room -= n
        if fill_crop and room > 0:
            seed = CROPS[fill_crop]["seed"]
            have = max(0, view.seeds.get(fill_crop, 0) - approved.get(fill_crop, 0))
            n = min(room, have + int(max(0.0, budget) // seed))
            if n > 0:
                budget -= max(0, n - have) * seed
                approved[fill_crop] = approved.get(fill_crop, 0) + n
                room -= n
                log["fill"] = n

    # 4. animals with what is left (structures are free) -- the leaders' split
    #    on days 3-6 is $1,058 animals to $653 seeds; ours was $400 to $291
    if on("size_budget"):
        for kind in list(plan.animal_buys):
            want = plan.animal_buys[kind]
            cost = ANIMALS[kind]["cost"]
            n = min(want, int(max(0.0, budget) // cost))
            plan.animal_buys[kind] = n
            budget -= n * cost
        empties = {"COOP": 0, "PASTURE": 0}
        for _, _, t in me.empty_structures():
            empties[t["kind"]] = empties.get(t["kind"], 0) + 1
        for struct in ("COOP", "PASTURE"):
            coming = sum(n for k, n in plan.animal_buys.items() if ANIMALS[k]["structure"] == struct)
            coming += sum(view.shed.get(k, 0) for k in ANIMALS if ANIMALS[k]["structure"] == struct)
            plan.structure_builds[struct] = min(plan.structure_builds.get(struct, 0), max(0, coming - empties[struct]))

    # 5. the plan's own crops beyond the seeds on hand, bought with what is
    #    left, in the plan's order; the fill already covers the ground so the
    #    remainder of the quota waits for tomorrow's cash
    if on("size_ground") or on("size_budget"):
        for crop, q in plan.plant_quota.items():
            seed = CROPS[crop]["seed"]
            buy = q - approved.get(crop, 0) - max(0, view.seeds.get(crop, 0) - approved.get(crop, 0))
            n_buy = min(max(0, buy), int(max(0.0, budget) // seed))
            if n_buy > 0:
                approved[crop] = approved.get(crop, 0) + n_buy
                budget -= n_buy * seed
        log["quota_asked"] = dict(plan.plant_quota)
        log["bare"] = bare
        plan.plant_quota = approved
        plan.approved = True
    log["quota"] = dict(plan.plant_quota)

    # 6. land last
    if on("size_budget") and plan.land_buys and budget < 1000:
        plan.land_buys = []
    log["animals"] = dict(plan.animal_buys)
    log["left"] = budget
    memory.sizing_log = getattr(memory, "sizing_log", {})
    memory.sizing_log[day] = log
    return log
