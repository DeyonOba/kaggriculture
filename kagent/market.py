"""Market controller: builds the (priority-ordered, <=10) market order list."""

from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, MAX_MARKET_ORDERS
from kagent import econ
from kagent.planner import FINAL_STEP


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

    # --- HIRE (first hours of the day only) ---------------------------------
    if view.hour <= 1:
        need = plan.hires_target - view.me.hires_today
        cost_guard = 0
        hired = 0
        while need > 0 and hired < 8:
            cost_guard += econ.hire_marginal_cost(view.me.hires_today + hired)
            if cost_guard > money * 0.3 or not add(["HIRE"]):
                break
            hired += 1
            need -= 1

    # --- Feed buffer: keep a full day of wheat for every animal -------------
    animals = sum(1 for _ in view.me.animals())
    if animals and view.day < 29:
        wheat_stock = view.shed.get("WHEAT", 0) + sum(
            view.unit_inv(i).get("WHEAT", 0) for i, _, _ in view.me.units
        )
        deficit = animals - wheat_stock
        hungry_now = sum(
            1
            for _, _, t in view.me.animals()
            if not t.get("fed_today") and t.get("consecutive_unfed", 0) >= 1
        )
        # Don't chase a self-inflicted wheat price spiral for buffer stock;
        # survival feeds are bought at any price.
        pricey = view.market_prices.get("WHEAT", 25) > 40
        if deficit > 0 and money > 100 and (hungry_now > wheat_stock or not pricey):
            add(["BUY_PRODUCT", "WHEAT", min(deficit, 15)])

    # --- Land ----------------------------------------------------------------
    remaining_land = []
    for quad in plan.land_buys:
        if not add(["BUY_LAND"]):
            remaining_land.append(quad)
    plan.land_buys = remaining_land

    # --- Animals (before seeds: a dropped seed order self-heals next turn) ----
    for kind in list(plan.animal_buys):
        want = plan.animal_buys[kind]
        cost = ANIMALS[kind]["cost"]
        n = min(want, int(money // cost))
        if n > 0 and add(["BUY_ANIMAL", kind, n]):
            plan.animal_buys[kind] = want - n
            money -= n * cost

    # --- Seeds: cover the remaining quota (bought this turn, plantable next) --
    last_plant_day = {"MELON": 19, "WHEAT": 25, "CARROT": 26, "TOMATO": 18, "STRAWBERRY": 13}
    for crop, remaining in plan.plant_quota.items():
        if view.day > last_plant_day.get(crop, 29) or view.hour > 20:
            continue  # planting window closing; don't strand seeds
        deficit = remaining - view.seeds.get(crop, 0)
        if deficit > 0 and money >= CROPS[crop]["seed"]:
            affordable = int(money * 0.6 // CROPS[crop]["seed"])
            n = min(deficit, max(1, affordable))
            if n > 0:
                add(["BUY_SEED", crop, n])

    # --- Sells fill the remaining slots ---------------------------------------
    for sell in _sell_orders(view, plan, memory):
        if not add(sell):
            break
        if sell[1] == "MELON":
            memory.melons_sold += sell[2]

    return orders


def _sell_orders(view, plan, memory):
    """One SELL order per item, sized to the marginal-revenue floor."""
    floors = plan.sell_floors
    # Shed pressure valve: midnight overflow is silently destroyed, so when
    # the shed runs hot, dump the biggest stacks at any price.
    pressure = view.shed_total > 80
    final_push = view.step >= FINAL_STEP - 8
    candidates = []
    for item, stock in view.shed.items():
        if stock <= 0 or item in ANIMALS:
            continue
        # Endgame: units DROP their loads before market orders resolve this
        # same turn, so size sells on shed + inbound carried stock — anything
        # dropped on the last step otherwise strands (field autopsy: 29-43
        # unsold wool at game end).
        if final_push:
            stock += sum(view.unit_inv(i).get(item, 0) for i, _, _ in view.me.units)
        if item == "WHEAT":
            stock = max(0, stock - plan.wheat_reserve)
        if stock <= 0:
            continue
        floor = 1 if (pressure and stock >= 10) else floors.get(item, 1)
        # Crashed-market release: once a premium market has collapsed well
        # below base, the town drain cannot lift it back inside our horizon —
        # holding has no option value, so take what the market gives.
        base = MARKET_PARAMS[item]["base"]
        if view.market_prices.get(item, base) <= 0.3 * base:
            floor = 1
        inv = view.market_inv.get(item)
        if inv is None:
            continue
        n = econ.units_sellable_above(item, inv, floor, cap=stock)
        if n <= 0:
            continue
        revenue = econ.sell_revenue(item, inv, n)
        candidates.append((revenue, item, n))

    candidates.sort(reverse=True)
    return [["SELL", item, n] for _, item, n in candidates]
