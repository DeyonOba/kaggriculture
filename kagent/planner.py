"""Day-level strategic planner: quotas, hiring, land, sell floors."""

from kagent.constants import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    SHOPS,
    TURNS_PER_DAY,
)
from kagent import econ

# --- forward market model ---------------------------------------------------
# The bundle's econ module already forecasts DEMAND (town_drain_per_day,
# forecast_drain) and has never forecast SUPPLY. The ablation in
# tools/price_forecast.py says that is exactly the wrong half to have: drain
# alone is WORSE than assuming the price never moves (k=5 MAE 25.5 vs 23.1),
# and the model only beats persistence once both farms' boards are in it
# (13.3, a 42% cut). Their shed is worth nothing even as an oracle.
_RENTA = 1.0
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
    """Projected market INVENTORY path, not price: the rent of a block has
    to be priced against the inventory its own units will create."""
    cur = dict(view.market_inv)
    drain = econ.town_drain_per_day(view.unlocked_shops)
    day = view.day
    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)
    # README Town Buildings: a shop unlocks every _FUT_EVERY days, drawn
    # uniformly WITH REPLACEMENT from the eight types and capped at
    # _FUT_CAP instances. Holding `drain` at today's shops understates
    # demand over a 14-day horizon, and understates it unevenly -- melon is
    # in no shop and can never gain any, wheat is in five of eight.
    _open_now = len(view.unlocked_shops)
    path = {p: [cur[p]] for p in cur}
    for k in range(1, horizon + 1):
        _d = day + k
        _new = max(0, _d // _FUT_EVERY - day // _FUT_EVERY)
        _new = min(_FUT_CAP, _open_now + _new) - _open_now
        for p in cur:
            cur[p] += (mine[k].get(p, 0) + opp[k].get(p, 0)
                       - drain.get(p, 0.0)
                       - _FUT_LAM * _new * _FUT.get(p, 0.0))
            path[p].append(cur[p])
    return path


def _rent_pick(view, day, free):
    """(crop, rent) with the highest $/tile-day at ITS OWN harvest price."""
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

"""Day-level strategic planner: quotas, hiring, land, sell floors."""


_FUT = {'WHEAT': 3.75, 'CARROT': 2.25, 'TOMATO': 1.5, 'STRAWBERRY': 3.0, 'MELON': 0.0, 'EGG': 1.5, 'MILK': 2.25, 'WOOL': 1.5, 'FERTILIZER': 0.0}
_FUT_LAM = 1.0
_FUT_EVERY = 3
_FUT_CAP = 8

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
        "fert_reserve", "land_margin", "feed_days", "cull_kinds", "sell_pace",
        "fert_gate", "fert_budget", "cash_floor", "seed_lot", "approved", "feed_any",
        "hire_all_cash",
    )

    def __init__(self):
        self.hires_target = 0
        self.land_buys = []
        self.plant_quota = {}
        self.animal_buys = {}
        self.structure_builds = {"COOP": 0, "PASTURE": 0}
        self.sell_floors = {}
        # Sell pacing for the steep-curve products (market._sell_orders):
        # -1 = off (shipped sizing); >= 0 = the largest within-order price
        # drop allowed, in percent, while the shops drain the market.
        self.sell_pace = -1
        # Scale on the scheduler's fertiliser price gates (1.0 = shipped).
        self.fert_gate = 1.0
        # FERTILIZE actions the plan wants today; None = no cap, the shipped
        # price-gated behaviour. Set by the rlv3 readout (v5 fert_budget head)
        # and consumed by policy.act as ops are emitted, like plant_quota.
        self.fert_budget = None
        self.liquidation = False
        # Animal kinds the scheduler must stop feeding today (they escape
        # after two unfed days). Empty unless a cull knob is on.
        self.cull_kinds = set()
        self.wheat_reserve = 0
        # Fertilizer units kept out of the market for today's FERTILIZE tasks
        # (set each turn by the policy from the task list).
        self.fert_reserve = 0
        self.land_margin = 100   # cash above the land price the market controller requires
        self.feed_days = 2       # days of feed the buffer keeps in stock
        # opening_v2: cash the market controller must leave in the bank
        # (tomorrow's hire bill + a margin) and the per-turn seed lot (0 = the
        # shipped 60%-of-cash purchase). Both 0 = the shipped behaviour.
        self.cash_floor = 0.0
        self.seed_lot = 0
        # executor2: the plan's purchases are approved -- seeds for the whole
        # quota are bought above the cash floor, not rationed to 60% of cash
        self.approved = False
        # sizing: feed is bought at any wheat price (the leaders' first claim)
        self.feed_any = False
        # lead_opening: hires may take all of the dawn cash (3 hands for $4)
        self.hire_all_cash = False


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
    """Day plan. `speculative=True` selects the champion teacher
    (make_champion_plan); the default is the reactive planner."""
    if speculative:
        return make_champion_plan(view, memory)
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
        wheat_target = min(WHEAT_TILE_CAP, 8 + my_animals)
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
        if straw_shops >= 2 and day <= 13:  # 4 productions need planting by ~13
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
    targets = {
        "GOOSE": GEESE_TARGET if day <= GEESE_LAST_BUY_DAY else 0,
        "COW": min(8, 4 * milk_shops) if day <= 18 else 0,
        "SHEEP": min(10, 4 * yarn_shops) if day <= 20 else 0,
    }

    if not plan.liquidation and day >= GEESE_BUY_START_DAY:
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
    # Matches the buy-side target so the buffer is never sold and re-bought.
    plan.wheat_reserve = (
        0 if plan.liquidation and day >= 29
        else max(my_animals + 2, plan.feed_days * my_animals)
    )

    # --- Sell floors ---------------------------------------------------------
    plan.sell_floors = _sell_floors(view)
    return plan


# ---- Champion teacher --------------------------------------------------------
# Decoded from 1500+/2500+ rated sides (tools/decode_sessions.py, $117k
# finals): day-0 ruminants, land on wool money at day 5-6, 33 strawberries by
# day 12, an 8-cow/8-sheep herd by day 14 bought BEFORE demand is revealed, a
# 20-30 tile rolling wheat engine, and late carrots on freed tiles.
CHAMP_OPENING = {"MELON": 12, "WHEAT": 7}
CHAMP_STRAW_TARGET = 33
CHAMP_STRAW_EARLY = 18          # before the day-10 melon money
CHAMP_WHEAT_ENGINE = 32
CHAMP_CARROT_DAY = 22           # from here freed tiles go to 2-day carrots
CHAMP_LAND_BUYS = 2             # champions never bought the $4000 quadrant
CHAMP_LAND_DAY = 5
# Gated fourth quadrant (knob land4). Measured 2026-09-26 on the replays: all
# three leaders own the third quadrant by day 8-9 (we: day 10) with half of
# it still bare; only M & M & P & Q buys the fourth, in 8 of 12 games, on
# days 15-18 with a third of its tiles idle, 3 hands and ~$25k in the bank;
# Boey and DSM never do. So the gate is cash and the day, not fullness:
# LAND4_MAX_IDLE keeps it off a farm that cannot plant what it has, the
# window ends where a wheat cycle stops paying, and the wheat engine is
# lifted by the new ground so the purchase is planted rather than idle.
LAND4_DAY_MIN = 14
LAND4_DAY_MAX = 20
LAND4_MIN_HIRES = 9
LAND4_MAX_IDLE = 12
LAND4_WHEAT_BONUS = 25
CHAMP_HIRES = [5, 4, 4, 5, 4, 5, 7, 9, 9, 10,
               11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
               11, 11, 11, 11, 11, 11, 10, 10, 10, 8]


# Season-long KNOBS of the champion plan (the adaptive layer's parameters).
# Defaults reproduce champ-h3; `memory.knobs` (per agent) overrides them —
# set by the counterfactual-bandit collector or by the fitted BanditPolicy
# once the shops are revealed.
KNOB_DEFAULTS = {
    "straw_target": CHAMP_STRAW_TARGET,   # standing strawberry tiles from day 10
    "straw_early": CHAMP_STRAW_EARLY,     # before the day-10 melon money
    "cows": 8, "sheep": 8, "geese": 0,    # herd caps (geese from knobs["geese_day"])
    # The goose is structurally the best animal we own and the champion plan
    # has never bought one: cheapest ($300 vs 400/500), first yield on day 4
    # (vs 8/6), produces EVERY day (interval 1 vs 2/3), and EGG is one of the
    # two markets whose price survives volume (log/0.20, like WHEAT; MILK is
    # linear/1.6 and WOOL sq/3.2). All three leaders run 5-6 geese and
    # harvest eggs 60-77 times a game; we harvest 0. The earlier refutations
    # tested "+4 or +6 geese from day 6" on the OLD executor.
    "geese_day": 6,                       # first day the geese target applies
    "wheat_engine": CHAMP_WHEAT_ENGINE,
    "carrot_standing": 0,                 # standing carrot patch, days 6-26
    "tomato_standing": 0,                 # standing tomato tiles, days 6-18
    "hires_delta": 0,
    # champ-h4 (knob bandit vs champ-h3, 3,600 paired samples): premium sell
    # floors x0.7 +$1,949 +/- 157, endgame carrots from day 25 +$1,545 +/- 116.
    "premium_scale": 0.35,                # x on STRAWBERRY/MILK/WOOL sell floors (champ-h5: round-2 bandit)
    "carrot_day": 25,                     # endgame: freed tiles -> carrots
    # Widened set (round 2, twin = champ-h4); defaults reproduce champ-h4.
    "melon_opening": CHAMP_OPENING["MELON"],  # day-0 melon tiles
    # lead_opening (2026-09-30, traced on M&M&P&Q episode 114976921 vs h11):
    # the leaders' day-0 basket -- 2 cows + 3 sheep, 6 melon + 10 wheat, 4
    # hands; 4 more melons on day 1 with 3 hands hired for $4 on a $4 dawn;
    # the 10 wheat pay $544-1,457 a day on days 2-5 and buy a cow a day, and
    # the third sheep's wool on day 6 buys five cows at once (herd 14 vs our
    # 8 on day 6). All 40 winner seats hold exactly 5 animals at the end of
    # day 0. Per unit-turn the two crews are equally productive; ours planted
    # 2 of its 7 wheat seeds on day 0 because it was one hand short.
    "lead_opening": 0,
    # OPENING V2 (2026-09-30, tools/plan_ledger.py): the champion delivers 49% of
    # its own asks on days 0-8, days 1 and 4 are $1 dawns with no hand hired,
    # and the leaders (5 units on day 0, 12.3 wheat + 6.3 melon planted, wheat
    # income from day 2, 3-4 animals placed a day on $26-125 dawns) win the
    # opening race. With the knob: a 4-hand crew on days 0-2, plant the crops
    # that pay first (wheat before melon), keep tomorrow's hire bill in the
    # bank, seeds bought in lots as the crew reaches the ground, and animal
    # WANTS that the market buys as the day's takings arrive (structures still
    # follow what dawn cash affords, so no empty pastures).
    "opening_v2": 0,
    # its four parts, individually switchable for the leave-one-out gate:
    # opening_v2=1 turns all four on; each v2_* knob turns one on by itself
    "v2_crew": 0,        # 4-hand crew on days 0-2
    "v2_order": 0,       # crops that pay first are planted (and seeded) first
    "v2_floor": 0,       # tomorrow's hire bill kept in the bank; seeds in lots
    "v2_structs": 0,     # structures follow what dawn cash affords, not the want
    "land2_margin": 100,                  # cash above price required for quadrant #2
    "wheat_floor": 1,                     # wheat sell floor (1 = sell all)
    "liquidation_day": LIQUIDATION_DAY,
    "carrot_cap": 20,                     # endgame carrots per day cap
    "feed_days": 2,                       # days of feed kept in stock
    # Culls (2026-09-25, from M & M & P & Q's twelve replays: 10.7 animals let
    # go a game vs our 0.8). Cows go when milk collapses -- median 0.14 of base
    # at the loss, days 16-20; sheep go in the endgame, days 20-29, half after
    # day 25, at a wool price near base; geese are kept. An unfed animal
    # escapes after two days, and what the cull releases is the feed wheat
    # and the feed/care actions. All OFF by default; the bandit rules set them.
    "cull_milk_ratio": 0.0,               # stop feeding COWS when milk < ratio x base (from day 12)
    "cull_wool_ratio": 0.0,               # stop feeding SHEEP when wool < ratio x base (from day 12)
    "endgame_cull_day": 0,                # from this day stop feeding cows and sheep
    # Fill bare ground with the labour that stood idle yesterday (2026-09-26):
    # extra wheat tiles = fill_idle x yesterday's PASS unit-turns, capped by
    # the bare tiles. 0 = off (the crop targets and the wheat engine are caps
    # and leave 10-12% of unlocked tiles bare at dawn; the top agent leaves 3%).
    "fill_idle": 0.0,
    # Sell pacing (2026-09-26, measured on traces vs M & M & P & Q's replays):
    # our strawberry orders cut the price 38% within the step (theirs 6%),
    # wool 47% (theirs 7%), because the sizing sells down to the floor along
    # the steep part of the curve, and the shops drain it back within the day
    # (price 24 steps later 1.05x). With sell_pace >= 0 a steep-curve product
    # with open shop demand is sold per step only as far as keeps the
    # within-order price drop under sell_pace percent; the rest waits for
    # the drain. -1 = off.
    "sell_pace": -1,
    # Fertiliser: scale on the scheduler's price gates (wheat < $40 x gate,
    # carrot < $50, tomato < $80, strawberry < $150). The top agent applies
    # 220 a game to our 112; a fertilised wheat yields 6 instead of 4.
    "fert_gate": 1.0,
    # Gated fourth quadrant (2026-09-26): 1 = buy the SE quadrant on days
    # 14-20 when the crew is at size, the owned ground is mostly planted and
    # land4_margin of working capital sits above the $4,000 price. 0 = the
    # champion's constant of three quadrants. See planner.LAND4_*.
    "land4": 0,
    "land4_margin": 15000,
}


def _knobs(memory):
    k = dict(KNOB_DEFAULTS)
    k.update(getattr(memory, "knobs", None) or {})
    return k


def _champ_herd_targets(day, knobs=None):
    knobs = knobs or KNOB_DEFAULTS
    cow = 2 if day < 3 else min(knobs["cows"], 2 + (day - 2))
    # Recorded pace: 4 sheep by day 8, 6 by day 10, 8 by day 12-14.
    sheep = 2 if day < 7 else min(knobs["sheep"], 2 + (day - 6))
    if knobs.get("lead_opening"):
        sheep = 3 if day < 7 else min(knobs["sheep"], 3 + (day - 6))   # the leaders' third sheep on day 0
    geese = knobs["geese"] if knobs.get("geese_day", 6) <= day <= 18 else 0
    return {
        "GOOSE": geese,
        "COW": cow if day <= 18 else 0,
        "SHEEP": sheep if day <= 20 else 0,
    }


def make_champion_plan(view, memory):
    plan = DayPlan()
    day = view.day
    me = view.me
    knobs = _knobs(memory)
    plan.liquidation = day >= knobs["liquidation_day"]
    plan.feed_days = knobs["feed_days"]

    crops = _count_crops(me)
    my_animals = sum(1 for _ in me.animals())
    herd = {}
    for _, _, t in me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1
    empty_by_kind = {"COOP": 0, "PASTURE": 0}
    for _, _, t in me.empty_structures():
        empty_by_kind[t["kind"]] = empty_by_kind.get(t["kind"], 0) + 1

    # --- Culls: stop feeding animals whose keep no longer pays ---------------
    # See KNOB_DEFAULTS. Decided fresh every dawn from today's price, so a
    # milk crash after the bandit's last decision day still triggers it.
    plan.cull_kinds = set()
    if day >= 12:
        px = view.market_prices
        r = knobs.get("cull_milk_ratio", 0.0)
        if r > 0 and px.get("MILK", 999) < r * MARKET_PARAMS["MILK"]["base"]:
            plan.cull_kinds.add("COW")
        r = knobs.get("cull_wool_ratio", 0.0)
        if r > 0 and px.get("WOOL", 999) < r * MARKET_PARAMS["WOOL"]["base"]:
            plan.cull_kinds.add("SHEEP")
    d = knobs.get("endgame_cull_day", 0)
    if d > 0 and day >= d:
        plan.cull_kinds |= {"COW", "SHEEP"}
    feeding_animals = sum(n for k, n in herd.items() if k not in plan.cull_kinds)
    empty_tiles = sum(1 for _ in me.empty_tiles())

    # --- Hiring: lean until the second quadrant, then ~11 ---------------------
    # Knobs are decided at day 6 (shops revealed); before that the plan is
    # the fixed champion opening (extra day-0 hires also push the market
    # order list past its 10-order cap and drop the seed orders).
    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)
    workload = sum(crops.values()) + 2 * my_animals + 10
    plan.hires_target = min(schedule, max(3, workload // 3))
    budget = max(150.0, me.money * 0.25)
    while plan.hires_target > 2 and econ.hire_total_cost(plan.hires_target) > budget:
        plan.hires_target -= 1
    if knobs.get("lead_opening") and day <= 2:
        # the leaders' crew: 4, 3, 6 on days 0-2, hired with whatever cash the
        # dawn holds (3 hands for $4 on a $4 dawn), not the workload cap
        plan.hires_target = (4, 3, 6)[day]
        plan.hire_all_cash = True
    v2 = bool(knobs.get("opening_v2"))
    v2_crew = v2 or bool(knobs.get("v2_crew"))
    v2_order = v2 or bool(knobs.get("v2_order"))
    v2_floor = v2 or bool(knobs.get("v2_floor"))
    v2_structs = v2 or bool(knobs.get("v2_structs"))
    if v2_crew and day <= 2:
        plan.hires_target = max(plan.hires_target, 4)         # the leaders' day-0 crew
    if v2_floor:
        nxt = CHAMP_HIRES[day + 1] if day + 1 < len(CHAMP_HIRES) else 8
        plan.cash_floor = float(econ.hire_total_cost(min(nxt, 6)) + 20)
        plan.seed_lot = 2 * (1 + plan.hires_target)

    # --- Land: a standing wish from day 5; the market controller buys the
    # turn the cash is there (the day-6 wool lands mid-day, after planning).
    plan.land_buys = []
    owned = me.unlocked_quadrants
    next_idx = sum(1 for q in LAND_ORDER if q in owned)
    if next_idx < CHAMP_LAND_BUYS and CHAMP_LAND_DAY <= day <= 22:
        plan.land_buys.append(LAND_ORDER[next_idx])
        if next_idx == 1:
            plan.land_margin = knobs["land2_margin"]
    elif (knobs.get("land4", 0) and next_idx == 2 and LAND4_DAY_MIN <= day <= LAND4_DAY_MAX
          and plan.hires_target >= LAND4_MIN_HIRES
          and empty_tiles + sum(1 for _ in me.weeds()) <= LAND4_MAX_IDLE):
        # Gated fourth quadrant (knob land4, off by default): the champion
        # plan stops at three quadrants as a constant, and the fourth was
        # −$6k twice as a constant too, bought onto a labour-short farm.
        # Here it needs the crew at size, the owned ground mostly planted,
        # a day inside the window where a wheat cycle still pays, and
        # land4_margin of working capital above the price (M & M & P & Q buys
        # with ~$25k in the bank). See the LAND4_* constants.
        plan.land_buys.append(LAND_ORDER[next_idx])
        plan.land_margin = knobs.get("land4_margin", 15000)

    # --- Animals: speculative herd, gated by cash only (feed is bought) --------
    plan.animal_buys = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    # An animal in a carrier's pocket is bought, alive and homeless, and both
    # lines below used to look straight past it -- they read `herd + shed`.
    # It is counted on the other side of the chain (`scheduler.build_tasks`
    # puts shed + pockets in `available` and hands it a PLACE task), so the
    # two halves disagree: the buy target treats the head as still missing and
    # orders a replacement, and the structure pipeline never queues it a
    # coop. README Terrain: DIG cannot recover a structure once an animal is
    # on it, so the tile commitment has to be right the first time.
    _pocket = {}
    for _i, _px, _py in me.units:
        for _it, _n in view.unit_inv(_i).items():
            _pocket[_it] = _pocket.get(_it, 0) + _n
    affordable_now = {}
    if not plan.liquidation:
        # Full deficits, not gated on hour-0 cash: the market controller buys
        # animals before seeds the turn money lands (wool/milk arrive mid-day),
        # so the herd wins the intra-day cash race the way champions' does.
        targets = _champ_herd_targets(day, knobs)
        cash_now = max(0.0, me.money - plan.cash_floor)
        for kind in ("SHEEP", "COW", "GOOSE"):  # wool lands first (day 6) and pays for land
            if kind in plan.cull_kinds:
                continue                        # never buy what we are letting go
            want = (targets[kind] - herd.get(kind, 0) - view.shed.get(kind, 0)
                    - _pocket.get(kind, 0))
            n = max(0, min(want, 4))
            if n > 0:
                plan.animal_buys[kind] = n
                # opening_v2: how many of them dawn cash covers now -- the
                # structures below follow THAT, not the want (tools/plan_ledger
                # and the seed-7 replay: pastures built for cows the bank could
                # not buy for five days, opening ground spent on nothing)
                now = min(n, int(cash_now // ANIMALS[kind]["cost"]))
                affordable_now[kind] = now
                cash_now -= now * ANIMALS[kind]["cost"]
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        buys_for_structures = (affordable_now.get(kind, 0) if v2_structs else plan.animal_buys.get(kind, 0))
        pipeline = (buys_for_structures + view.shed.get(kind, 0)
                    + _pocket.get(kind, 0))
        need = pipeline - empty_by_kind.get(structure, 0)
        if need > 0:
            plan.structure_builds[structure] += need
            empty_by_kind[structure] = 0
        else:
            empty_by_kind[structure] = -need

    # --- Planting: opening, then strawberries, then a wheat engine ------------
    quota = {}
    if not plan.liquidation:
        # Tiles plantable today: empty now plus those harvested clean today.
        ripe_today = 0
        for _, _, t in me.plants():
            cd = CROPS.get(t["crop"])
            if cd is None:
                continue
            age = day - t.get("planted_day", day)
            if not cd["ongoing"] and age >= cd["max_yield_day"]:
                ripe_today += 1
            elif 0 <= t.get("max_lifespan_step", -1) <= (day + 1) * TURNS_PER_DAY:
                ripe_today += 1
        free = max(0, empty_tiles + ripe_today - sum(plan.structure_builds.values()))
        free0 = free
        if day <= 1:
            for crop, n in CHAMP_OPENING.items():
                if crop == "MELON":
                    n = knobs["melon_opening"]
                if knobs.get("lead_opening"):
                    n = {"MELON": (6, 10)[day], "WHEAT": 10}[crop]         # the leaders' 6 + 4 melons, 10 wheat
                quota[crop] = max(0, n - crops.get(crop, 0))
        if CHAMP_LAND_DAY <= day <= 13:
            # Strawberries take freed tiles first (4 productions by 29), staged
            # so the herd is funded before the day-10 melon money: 18 by day
            # 9, then the full 33 — the recorded champion pace.
            straw_target = knobs["straw_target"] if day >= 10 else min(knobs["straw_early"], knobs["straw_target"])
            quota["STRAWBERRY"] = max(0, straw_target - crops.get("STRAWBERRY", 0))
        if knobs["carrot_standing"] and 6 <= day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(0, knobs["carrot_standing"] - crops.get("CARROT", 0))
        if knobs["tomato_standing"] and 6 <= day <= TOMATO_LAST_PLANT_DAY:
            quota["TOMATO"] = max(0, knobs["tomato_standing"] - crops.get("TOMATO", 0))
        if day >= knobs["carrot_day"] and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(quota.get("CARROT", 0), min(knobs["carrot_cap"], free))
        elif day > 1 and day <= WHEAT_LAST_PLANT_DAY:
            if day < 10:
                wheat_target = 8
            else:
                engine = knobs["wheat_engine"] + (LAND4_WHEAT_BONUS if "SE" in owned else 0)
                wheat_target = min(engine, crops.get("WHEAT", 0) + free)
            _best = _rent_pick(view, day, free)
            if _RENTA > 0 and _best and _best[0] != "WHEAT":
                _n = int(round(_RENTA * free))
                if _n > 0:
                    quota[_best[0]] = quota.get(_best[0], 0) + _n
                    free = max(0, free - _n)
                    wheat_target = min(wheat_target, crops.get("WHEAT", 0) + free)
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))
        # --- Fill bare ground with idle labour (knob fill_idle, off by default).
        # Measured 2026-09-26 (tools/tile_metrics.py): 10-12% of our unlocked
        # tiles are bare at dawn against the top agent's 3%, and the executor
        # rules do not move it, because the crop targets and the wheat engine
        # are caps. Top the quota up with wheat, but only by what the labour
        # that stood idle yesterday can carry (a standing wheat tile costs
        # about one action a day): bare ground worked by busy hands would
        # displace higher-value chores, which is how every volume knob lost.
        fill = knobs.get("fill_idle", 0.0)
        if fill > 0 and 2 <= day <= WHEAT_LAST_PLANT_DAY:
            idle_turns = (getattr(memory, "yesterday_idle", 0.0)
                          * (getattr(memory, "yesterday_hands", 0) + 1) * TURNS_PER_DAY)
            room = max(0, free0 - sum(quota.values()))
            extra = min(room, int(idle_turns * fill))
            if extra > 0:
                quota["WHEAT"] = quota.get("WHEAT", 0) + extra
    if v2_order and quota:
        # dict order is the scheduler's planting order and the market's seed
        # order: the crops that pay first go in first (wheat 2 days, melon 10)
        quota = dict(sorted(quota.items(), key=lambda kv: CROPS.get(kv[0], {}).get("first_yield_day", 99)))
    plan.plant_quota = {k: v for k, v in quota.items() if v > 0}

    # Never sell feed we are about to buy back. The sell path used to reserve
    # `animals + 2` while the buy path topped the buffer up to
    # `feed_days * animals`, so every day we sold the gap and re-bought it: a
    # $19k round trip over a season (field ledger vs a 2500+ opponent: 488
    # wheat sold at $44, 459 bought back at $41 average and rising, because
    # BUY_PRODUCT quotes at inventory-1 and both farms drain the same pool).
    plan.wheat_reserve = (
        0 if plan.liquidation and day >= 29
        else max(feeding_animals + 2, plan.feed_days * feeding_animals)
    )
    plan.sell_floors = _sell_floors(view)
    plan.sell_pace = knobs.get("sell_pace", -1)
    plan.fert_gate = knobs.get("fert_gate", 1.0)
    if knobs["premium_scale"] != 1.0 and 6 <= day and view.step < 714:
        for item in ("STRAWBERRY", "MILK", "WOOL"):
            plan.sell_floors[item] = max(1, int(plan.sell_floors[item] * knobs["premium_scale"]))
    if knobs["wheat_floor"] > 1 and 6 <= day < FLOOR_DECAY_DAY:
        plan.sell_floors["WHEAT"] = knobs["wheat_floor"]
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
