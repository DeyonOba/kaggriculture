"""Day-level strategic planner: quotas, hiring, land, sell floors."""

from kagent.constants import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    TURNS_PER_DAY,
)
from kagent import econ

# ---- Tunables (arena-calibrated) -------------------------------------------

# Total melon units (ours + opponent's projected) the market absorbs at a good
# price. Depth to $1 is 158; past ~120 the marginal price collapses.
MELON_SUPPLY_TARGET = 140
MELON_LAST_PLANT_DAY = 19       # d+10 harvest must land by day 29
WHEAT_LAST_PLANT_DAY = 25
CARROT_LAST_PLANT_DAY = 26
TOMATO_LAST_PLANT_DAY = 18      # all 4 productions (ages 8-11) inside season

GEESE_TARGET = 32
GEESE_BUY_START_DAY = 2         # melon seeds soak day 0-1 cash; geese ride melon money
GEESE_LAST_BUY_DAY = 24         # fertilizer pays from day 1, eggs from day 5
GEESE_PER_DAY = 10

WHEAT_TILE_CAP = 24             # rolling wheat engine (sale + feed)
TOMATO_TILE_CAP = 12
CARROT_TILE_CAP = 14

# Hands to hire per day (index = day), before the workload/money gates.
HIRE_SCHEDULE = [5, 6, 8, 10, 10, 12, 12, 12, 14, 14,
                 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                 14, 14, 14, 12, 12, 12, 10, 10, 10, 8]

LIQUIDATION_DAY = 27
FINAL_STEP = 718                # last actable step at default config

# Sell floors: marginal-revenue cutoffs. Cash compounds (geese, hires, land),
# so holding stock for a slightly better price loses to reinvesting sooner.
MELON_FLOOR = 80
FERTILIZER_FLOOR = 50
CARROT_FLOOR = 30
TOMATO_FLOOR = 45
FLOOR_DECAY_DAY = 24            # floors slide toward $1 from here to step 718


class DayPlan:
    __slots__ = (
        "hires_target", "land_buys", "plant_quota", "animal_buys",
        "structure_builds", "sell_floors", "liquidation", "wheat_reserve",
    )


def _count_crops(farm):
    counts = {}
    for _, _, t in farm.plants():
        counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    return counts


def _projected_melons(farm):
    """Units of melon this farm will plausibly bring to market from current tiles."""
    total = 0
    for _, _, t in farm.plants():
        if t["crop"] == "MELON":
            total += max(t.get("yield_units", 1), 4)  # assume decent care
    return total


def make_day_plan(view, memory, speculative=False):
    """Day plan. `speculative=True` selects the leaderboard-meta teacher used
    only for BC data (episode 101761797: finals $105k/$96k): early ruminants
    bought on EXPECTED shop demand before any shop exists, strawberries at
    scale, investment over profit through ~day 10. The deployed fallback
    stays reactive (default)."""
    plan = DayPlan()
    day = view.day
    me = view.me
    plan.liquidation = day >= LIQUIDATION_DAY

    crops = _count_crops(me)
    my_animals = sum(1 for _ in me.animals())
    structures = sum(1 for _ in me.empty_structures()) + my_animals

    # --- Hiring ---------------------------------------------------------
    schedule = HIRE_SCHEDULE[day] if day < len(HIRE_SCHEDULE) else 8
    # Rough workload: tiles needing daily touches + animal chores.
    workload = sum(crops.values()) + 2 * my_animals + 10
    cap_by_work = max(3, workload // 3)
    plan.hires_target = min(schedule, cap_by_work)
    budget = max(150.0, me.money * 0.25)  # hands are cheap; never starve labor
    while plan.hires_target > 2 and econ.hire_total_cost(plan.hires_target) > budget:
        plan.hires_target -= 1

    # --- Land -------------------------------------------------------------
    # (May be extended below once quotas/builds are known: land_pressure.)
    plan.land_buys = []
    owned = me.unlocked_quadrants
    next_idx = sum(1 for q in LAND_ORDER if q in owned)
    if next_idx < len(LAND_ORDER) and day <= 22:
        price = LAND_PRICES[next_idx]
        # Buy only when the current land is actually filling up.
        empty = sum(1 for _ in me.empty_tiles())
        if me.money >= price + 600 and empty < 12:
            plan.land_buys.append(LAND_ORDER[next_idx])

    # --- Planting quotas (tiles to add today per crop) --------------------
    quota = {}
    if not plan.liquidation:
        # Melon: adaptive to the shared market depth and opponent supply.
        if day <= MELON_LAST_PLANT_DAY:
            opp_supply = _projected_melons(view.opp)
            my_supply = (
                _projected_melons(me)
                + view.shed.get("MELON", 0)
                + memory.melons_sold
            )
            room = MELON_SUPPLY_TARGET - opp_supply - my_supply
            if room > 0:
                quota["MELON"] = min(10, max(0, room // CROPS["MELON"]["max_yield"]))

        # Wheat: rolling engine sized to labor, feed needs, and land.
        # The speculative teacher runs a bigger engine (leaders held 15-40
        # wheat tiles as combined feed + $40+ sale inventory).
        wheat_cap = 32 if speculative else WHEAT_TILE_CAP
        wheat_target = min(wheat_cap, (10 if speculative else 8) + my_animals)
        if day <= WHEAT_LAST_PLANT_DAY:
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))

        # Carrot: scale with town demand (PET_CAFE 2x, FARMERS_MARKET 1x).
        pet = view.unlocked_shops.count("PET_CAFE")
        fm = view.unlocked_shops.count("FARMERS_MARKET")
        carrot_target = min(CARROT_TILE_CAP, 6 + 3 * pet + 2 * fm)
        if day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(0, carrot_target - crops.get("CARROT", 0))

        # Tomato: scale with pizza shops / farmers markets.
        pizza = view.unlocked_shops.count("PIZZA_SHOP")
        tomato_target = min(TOMATO_TILE_CAP, 6 + 3 * pizza + 2 * fm)
        if day <= TOMATO_LAST_PLANT_DAY:
            quota["TOMATO"] = max(0, tomato_target - crops.get("TOMATO", 0))

        # Strawberry: a trap at base demand, but 2+ consuming shops push its
        # drain far past T=100 and the price holds $200+ all season.
        straw_shops = sum(
            view.unlocked_shops.count(s)
            for s in ("BRUNCH_SPOT", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "FARMERS_MARKET")
        )
        if speculative and day <= 13:
            # Strawberry appears in 4 of 8 shop draws: plant on expectation,
            # then scale hard with each revealed consumer.
            straw_target = min(32, 10 + 6 * straw_shops)
            quota["STRAWBERRY"] = max(0, straw_target - crops.get("STRAWBERRY", 0))
        elif straw_shops >= 2 and day <= 13:  # 4 productions need planting by ~13
            straw_target = min(10, 3 * straw_shops)
            quota["STRAWBERRY"] = max(0, straw_target - crops.get("STRAWBERRY", 0))
    plan.plant_quota = quota

    # --- Animals (adaptive to town demand) ----------------------------------
    plan.animal_buys = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    herd = {}
    for _, _, t in me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1
    empty_by_kind = {"COOP": 0, "PASTURE": 0}
    for _, _, t in me.empty_structures():
        empty_by_kind[t["kind"]] = empty_by_kind.get(t["kind"], 0) + 1

    # Milk/wool are traps at base demand, but town shops can push their drain
    # far past T: then they hold near/above base price all season.
    milk_shops = sum(
        view.unlocked_shops.count(s) for s in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
    )
    yarn_shops = view.unlocked_shops.count("YARN_STORE")
    if speculative:
        # Buy ruminants on expected demand from day 0 (milk is in 3 of 8 shop
        # draws, wool doubled in 1) and ramp cows with every revealed milk
        # shop, the way the leaderboard leaders do; skip the goose engine.
        targets = {
            "GOOSE": 0,
            "COW": min(15, 3 + day // 2 + 4 * milk_shops) if day <= 20 else 0,
            "SHEEP": min(6, 2 + 3 * yarn_shops) if day <= 20 else 0,
        }
    else:
        targets = {
            "GOOSE": GEESE_TARGET if day <= GEESE_LAST_BUY_DAY else 0,
            "COW": min(8, 4 * milk_shops) if day <= 18 else 0,
            "SHEEP": min(10, 4 * yarn_shops) if day <= 20 else 0,
        }

    if not plan.liquidation and day >= (0 if speculative else GEESE_BUY_START_DAY):
        # Feed security: never grow the herd past what the wheat pipeline
        # (stock + growing tiles + market top-ups) can keep alive.
        feed_capacity = view.shed.get("WHEAT", 0) + 3 * crops.get("WHEAT", 0) + 10
        budget = max(0.0, me.money - 400)
        feedable = max(0, feed_capacity - my_animals)
        per_day_left = GEESE_PER_DAY
        for kind in ("COW", "SHEEP", "GOOSE"):  # scarce-demand animals first
            cost = ANIMALS[kind]["cost"]
            want = targets[kind] - herd.get(kind, 0) - view.shed.get(kind, 0)
            n = max(0, min(want, int(budget // cost), feedable, per_day_left))
            if n > 0:
                plan.animal_buys[kind] = n
                budget -= n * cost
                feedable -= n
                per_day_left -= n

    # Structures (free to build): cover the delivery pipeline per kind.
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = plan.animal_buys.get(kind, 0) + view.shed.get(kind, 0)
        need = pipeline - empty_by_kind.get(structure, 0)
        if need > 0:
            plan.structure_builds[structure] += need
            empty_by_kind[structure] = 0
        else:
            empty_by_kind[structure] = -need

    # --- Feed reserve -------------------------------------------------------
    plan.wheat_reserve = 0 if plan.liquidation and day >= 29 else my_animals + 2

    # --- Sell floors ---------------------------------------------------------
    plan.sell_floors = _sell_floors(view)
    return plan


def apply_land_pressure(plan, view):
    """Demand-driven land expansion safety net for the NEURAL plan path only:
    the land head overwrites the heuristic's land logic, so without this a
    learned policy can strand itself on 25 tiles while a shop wave makes
    pastures profitable. The scripted heuristic keeps its own tuned rule —
    adding this trigger there regressed it 10-17 vs v3 (cash starved the
    melon rush)."""
    if plan.land_buys or plan.liquidation or not (2 <= view.day <= 24):
        return
    owned = view.me.unlocked_quadrants
    next_idx = sum(1 for q in LAND_ORDER if q in owned)
    if next_idx >= len(LAND_ORDER):
        return
    # Only count work that could actually land today (labor caps).
    pending = min(sum(plan.plant_quota.values()), 12) + sum(plan.structure_builds.values())
    empty = sum(1 for _ in view.me.empty_tiles())
    price = LAND_PRICES[next_idx]
    if empty < pending + 2 and view.me.money >= price + 1500:
        plan.land_buys.append(LAND_ORDER[next_idx])


def _sell_floors(view):
    """Minimum acceptable marginal sell price per item. Flat during the
    season (cash compounds), sliding to $1 over the final days so nothing is
    left unsold."""
    step = view.step
    decay_start = FLOOR_DECAY_DAY * TURNS_PER_DAY

    def decayed(base_floor):
        if step < decay_start:
            return base_floor
        if step >= FINAL_STEP - 4:
            return 1
        frac = (FINAL_STEP - step) / (FINAL_STEP - decay_start)
        return max(1, int(base_floor * frac))

    # Fertilizer has ZERO town demand: its price falls monotonically as both
    # players dump animal fertilizer, and never recovers. Earliest sale is
    # always the best sale — and a fertilizer hoard blocks the shed, whose
    # midnight overflow silently destroys produce.
    # (floor stays 1; see return dict below)

    # Opponent melons about to ripen -> dump ours before the crash.
    melon_floor = MELON_FLOOR
    opp_ripening = sum(
        1
        for _, _, t in view.opp.plants()
        if t["crop"] == "MELON" and view.day - t["planted_day"] >= 8
    )
    if opp_ripening >= 3:
        melon_floor = 40

    # Milk/wool floors rise with town demand: heavy shop drain keeps their
    # scarce-side prices propped up, so only sell into strength.
    milk_shops = sum(
        view.unlocked_shops.count(s) for s in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
    )
    yarn_shops = view.unlocked_shops.count("YARN_STORE")

    straw_shops = sum(
        view.unlocked_shops.count(s)
        for s in ("BRUNCH_SPOT", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "FARMERS_MARKET")
    )
    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": decayed(melon_floor),
        "FERTILIZER": 1,
        "CARROT": decayed(CARROT_FLOOR),
        "TOMATO": decayed(TOMATO_FLOOR),
        "STRAWBERRY": decayed(180 if straw_shops >= 3 else 80),
        "MILK": decayed(130 if milk_shops >= 1 else 100),
        "WOOL": decayed(160 if yarn_shops >= 1 else 120),
    }
