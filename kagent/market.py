"""Market controller: builds the (priority-ordered, <=10) market order list."""

from kagent.constants import (
    ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, MARKET_PARAMS, MAX_MARKET_ORDERS,
    TOWN_CENTER_SELL_INTERVAL, TURNS_PER_DAY,
)
from kagent import econ
from kagent.planner import FINAL_STEP
# Cross-module reference, same shape as `econ` and `planner` above: the bundler
# strips these import lines and binds the module aliases into one shared
# namespace AFTER every body has run, so `scheduler` must only ever be touched
# inside a function, never at module level. An inline `from kagent import
# scheduler` inside a function does not survive that (there is no kagent
# package in a bundle) -- it raised, SafeAgent swallowed it, and the agent
# scored exactly its starting money.
from kagent import scheduler


# Products whose price falls on a steep curve above I0 (linear / sq above
# functions): the ones sell pacing applies to.
PACED = ("STRAWBERRY", "MILK", "WOOL", "MELON")


def build_orders(view, plan, memory):
    orders = []
    money = view.me.money
    final_push = view.step >= FINAL_STEP - 8

    def add(order):
        if len(orders) < MAX_MARKET_ORDERS:
            orders.append(order)
            return True
        return False  # cap reached; the engine would silently drop it

    # --- Liquidation endgame: selling takes absolute priority ---------------
    if final_push:
        for sell in _sell_orders(view, plan, memory):
            if add(sell) and sell[1] == "MELON":
                memory.melons_sold += sell[2]
        return orders

    approved = bool(getattr(plan, "approved", False))
    early_sells = []
    if approved and view.hour == 0:
        # the engine resolves orders in list order with the cash updated in
        # between, so the dawn sells pay for the hires and the seeds below
        for sell in _sell_orders(view, plan, memory):
            if len(early_sells) >= MAX_MARKET_ORDERS - 4:
                break
            early_sells.append(sell)
            if sell[1] == "MELON":
                memory.melons_sold += sell[2]
            money += econ.sell_revenue(sell[1], view.market_inv.get(sell[1], 0), sell[2])
        orders.extend(early_sells)

    # --- HIRE (first hours of the day only) ---------------------------------
    if view.hour <= 1:
        need = plan.hires_target - view.me.hires_today
        cost_guard = 0
        hired = 0
        while need > 0 and hired < 8:
            cost_guard += econ.hire_marginal_cost(view.me.hires_today + hired)
            if cost_guard > (money if (approved or getattr(plan, "hire_all_cash", False)) else money * 0.3) or not add(["HIRE"]):
                break
            hired += 1
            need -= 1

    # --- Feed buffer: keep a full day of wheat for every animal -------------
    animals = sum(1 for _ in view.me.animals())
    if animals and view.day < 29:
        wheat_stock = view.shed.get("WHEAT", 0) + sum(
            view.unit_inv(i).get("WHEAT", 0) for i, _, _ in view.me.units
        )
        # Stock today's feed plus tomorrow morning's: an unfed animal skips a
        # $100-200 production and escapes after two dry days, so feed is
        # bought at any sane price, down to the last dollar (champions buy
        # 370-480 units a game at $25-50).
        target = animals if view.day >= 28 else plan.feed_days * animals
        deficit = target - wheat_stock
        price = view.market_prices.get("WHEAT", 25)
        affordable = int(money // max(1.0, price))
        if deficit > 0 and affordable > 0 and (price <= 120 or getattr(plan, "feed_any", False)):
            add(["BUY_PRODUCT", "WHEAT", min(deficit, 15, affordable)])
            money -= min(deficit, 15, affordable) * price

    if approved:
        money = _seed_orders(view, plan, memory, money, add)

    # --- Land (bought the turn it becomes affordable) -------------------------
    remaining_land = []
    owned = sum(1 for q in LAND_ORDER if q in view.me.unlocked_quadrants)
    for i, quad in enumerate(plan.land_buys):
        idx = min(owned + i, len(LAND_PRICES) - 1)
        if money >= LAND_PRICES[idx] + plan.land_margin and add(["BUY_LAND"]):
            money -= LAND_PRICES[idx]
        else:
            remaining_land.append(quad)
    plan.land_buys = remaining_land

    # --- Animals (before seeds: a dropped seed order self-heals next turn) ----
    # opening_v2: purchases leave plan.cash_floor in the bank (tomorrow's hire
    # bill); 0 for the shipped plan. Feed above is exempt: an unfed animal
    # escapes after two dry days.
    floor = float(getattr(plan, "cash_floor", 0.0) or 0.0)
    for kind in list(plan.animal_buys):
        want = plan.animal_buys[kind]
        cost = ANIMALS[kind]["cost"]
        n = min(want, int(max(0.0, money - floor) // cost))
        if n > 0 and add(["BUY_ANIMAL", kind, n]):
            plan.animal_buys[kind] = want - n
            money -= n * cost

    if not approved:
        money = _seed_orders(view, plan, memory, money, add)

    # --- Sells fill the remaining slots ---------------------------------------
    if not early_sells:
        for sell in _sell_orders(view, plan, memory):
            if not add(sell):
                break
            if sell[1] == "MELON":
                memory.melons_sold += sell[2]

    return orders


def _seed_orders(view, plan, memory, money, add):
    """Seeds: cover the remaining quota (bought this turn, plantable next).
    Returns the cash left. Under an approved plan the whole deficit is bought
    above the cash floor; otherwise 60% of the cash above it (shipped)."""
    floor = float(getattr(plan, "cash_floor", 0.0) or 0.0)
    last_plant_day = {"MELON": 19, "WHEAT": 25, "CARROT": 26, "TOMATO": 18, "STRAWBERRY": 13}
    # rlv3/pipeline.py extends the wheat/carrot window through day 27;
    # anything else leaves this table exactly as shipped.
    last_plant_day.update(getattr(memory, "last_plant", None) or {})
    for crop, remaining in plan.plant_quota.items():
        if view.day > last_plant_day.get(crop, 29) or view.hour > 20:
            continue  # planting window closing; don't strand seeds
        deficit = remaining - view.seeds.get(crop, 0)
        if deficit > 0 and money - floor >= CROPS[crop]["seed"]:
            cap = 1.0 if getattr(plan, "approved", False) else 0.6
            affordable = int(max(0.0, money - floor) * cap // CROPS[crop]["seed"])
            n = min(deficit, max(1, affordable))
            lot = int(getattr(plan, "seed_lot", 0) or 0)
            if lot > 0:
                n = min(n, lot)            # opening_v2: seeds as the crew reaches the ground
            if n > 0 and add(["BUY_SEED", crop, n]):
                money -= n * CROPS[crop]["seed"]
    return money


def _sell_orders(view, plan, memory):
    """One SELL order per item, sized to the marginal-revenue floor."""
    floors = plan.sell_floors
    # Shed pressure valve: midnight overflow is silently destroyed, so when
    # the shed runs hot, dump the biggest stacks at any price.
    pressure = view.shed_total > 80
    if scheduler.EXEC.get("shedguard") and view.hour >= scheduler.SHEDGUARD_HOUR:
        # what will actually be dropped tonight, not just what is banked now
        carried = sum(n for i, _, _ in view.me.units
                      for item, n in view.unit_inv(i).items()
                      if item not in ANIMALS)
        pressure = pressure or (view.shed_total + carried) > 100
    final_push = view.step >= FINAL_STEP - 8
    # Sell pacing (plan.sell_pace >= 0; KNOB_DEFAULTS["sell_pace"]): the four
    # steep-curve products are quoted unit by unit down the curve, and our
    # orders cut the price 38-47% within one step where the top agent's lose
    # 6-9% (measured 2026-09-26); the shops drain the market back within the
    # day. So an order is capped at the units that keep the within-order drop
    # under `pace` percent of the first unit's price; the rest is sold in
    # later steps as the drain lifts the price. Only with open shop demand: a
    # product the town does not buy (wool without a yarn store, melon,
    # fertilizer) never drains, so its earliest sale is its best. The
    # shed-pressure valve, the final push and the crashed-market release all
    # bypass the cap.
    pace = getattr(plan, "sell_pace", -1)
    drain = econ.town_drain_per_day(view.unlocked_shops) if pace >= 0 else {}
    center_only = TURNS_PER_DAY / TOWN_CENTER_SELL_INTERVAL
    candidates = []
    for item, stock in view.shed.items():
        if stock <= 0 or item in ANIMALS:
            continue
        # Endgame: units DROP their loads before market orders resolve this
        # same turn, so size sells on shed + inbound carried stock — anything
        # dropped on the last step otherwise strands (field autopsy: 29-43
        # unsold wool at game end).
        # The end-of-day auto-drop discards anything that does not fit, so what
        # the crew is CARRYING is at risk every evening, not only on the last
        # day. Under `shedguard` the evening sizing counts it; without the flag
        # this is exactly the shipped behaviour.
        _evening = (scheduler.EXEC.get("shedguard")
                    and view.hour >= scheduler.SHEDGUARD_HOUR)
        if final_push or _evening:
            stock += sum(view.unit_inv(i).get(item, 0) for i, _, _ in view.me.units)
        if item == "WHEAT":
            stock = max(0, stock - plan.wheat_reserve)
        if item == "FERTILIZER":
            # Keep what today's FERTILIZE tasks need (strawberry doubling).
            stock = max(0, stock - plan.fert_reserve)
        if stock <= 0:
            continue
        floor = 1 if (pressure and stock >= 10) else floors.get(item, 1)
        # Crashed-market release: once a premium market has collapsed well
        # below base, the town drain cannot lift it back inside our horizon —
        # holding has no option value, so take what the market gives.
        base = MARKET_PARAMS[item]["base"]
        if view.market_prices.get(item, base) <= 0.3 * base and not scheduler.EXEC.get("hold_crash"):
            floor = 1
        inv = view.market_inv.get(item)
        if inv is None:
            continue
        n = econ.units_sellable_above(item, inv, floor, cap=stock)
        if n <= 0:
            continue
        if (pace >= 0 and item in PACED and floor > 1 and not pressure and not final_push
                and drain.get(item, 0.0) > center_only + 1e-9):
            first = econ.price(item, inv)
            n = min(n, econ.units_sellable_above(item, inv, first * (1 - pace / 100.0), cap=stock))
            if n <= 0:
                continue
        revenue = econ.sell_revenue(item, inv, n)
        candidates.append((revenue, item, n))

    candidates.sort(reverse=True)
    return [["SELL", item, n] for _, item, n in candidates]
