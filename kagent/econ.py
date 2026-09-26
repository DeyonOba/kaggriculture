"""Pure economic model: exact mirror of the engine's price function plus
marginal-revenue helpers. No state, stdlib only.
"""

import math

from kagent.constants import (
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    PRICE_FLOOR,
    HINGE_GAIN,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    TOWN_SHOP_SELL_INTERVAL,
    TOWN_CENTER_SELL_INTERVAL,
    TURNS_PER_DAY,
    FARM_HAND_COST_MULT,
    fib,
)


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price(item, inventory):
    """Engine market_price: price at a given market inventory level."""
    p = MARKET_PARAMS[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        val = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        val = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(val)))


def sell_marginal_prices(item, inventory, n):
    """Prices received for units 1..n of a solo sell starting at `inventory`.

    Engine quotes each unit at pre-sell inventory, then increments inventory
    only if the unit sold above the $1 floor.
    """
    out = []
    inv = inventory
    for _ in range(n):
        p = price(item, inv)
        out.append(p)
        if p > PRICE_FLOOR:
            inv += 1
    return out


def sell_revenue(item, inventory, n):
    return sum(sell_marginal_prices(item, inventory, n))


def units_sellable_above(item, inventory, min_price, cap=10_000):
    """How many units can be sold, one at a time, before the marginal price
    drops below `min_price`."""
    count = 0
    inv = inventory
    while count < cap:
        p = price(item, inv)
        if p < min_price:
            break
        count += 1
        if p > PRICE_FLOOR:
            inv += 1
    return count


def buy_cost(item, inventory, n):
    """Cost of BUY_PRODUCT for n units. Engine quotes each unit at the
    post-buy inventory (inv - 1), then decrements."""
    total = 0
    inv = inventory
    for _ in range(n):
        total += price(item, inv - 1)
        inv -= 1
    return total


def town_drain_per_day(unlocked_shops):
    """Expected market-inventory drain per item per day from town buildings.

    Shops tick every TOWN_SHOP_SELL_INTERVAL turns (single-product shops
    consume 2x); the town center ticks every TOWN_CENTER_SELL_INTERVAL turns.
    """
    shop_ticks = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL
    center_ticks = TURNS_PER_DAY / TOWN_CENTER_SELL_INTERVAL
    drain = {}
    for item in TOWN_CENTER_PRODUCTS:
        drain[item] = center_ticks
    for shop in unlocked_shops:
        items = SHOPS.get(shop, ())
        mult = 2 if len(items) == 1 else 1
        for item in items:
            drain[item] = drain.get(item, 0.0) + mult * shop_ticks
    return drain


# --- Demand forecasting -------------------------------------------------------
# Shop unlocks are one uniform draw (with replacement) from the 8 shop types
# every TOWN_SHOP_UNLOCK_INTERVAL days, capped at MAX_SHOP_INSTANCES. So the
# demand a not-yet-opened store will bring is predictable in expectation.

_UNLOCK_DAYS = (3, 6, 9, 12, 15, 18, 21, 24)


def shop_weight_per_day():
    """Expected market drain per day per product contributed by ONE random
    future shop unlock (single-product shops consume 2x)."""
    ticks = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL
    out = {}
    for items in SHOPS.values():
        mult = 2 if len(items) == 1 else 1
        for item in items:
            out[item] = out.get(item, 0.0) + mult
    n_shops = len(SHOPS)
    return {item: w / n_shops * ticks for item, w in out.items()}


def remaining_unlocks(day, current_count):
    """How many more shop instances will unlock after `day`."""
    scheduled = sum(1 for u in _UNLOCK_DAYS if u > day)
    return max(0, min(MAX_SHOP_INSTANCES - current_count, scheduled))


def _unlock_outcomes(item):
    """Per future unlock, the (probability, per-day drain) outcomes for one
    product. The town draws ONE shop uniformly from the eight types, with
    replacement, so a single unlock either carries this product or does not."""
    ticks = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL
    n = len(SHOPS)
    rates = {}
    for items in SHOPS.values():
        r = (2 if len(items) == 1 else 1) * ticks if item in items else 0.0
        rates[r] = rates.get(r, 0) + 1
    return [(c / n, r) for r, c in rates.items()]


def forecast_drain(unlocked_shops, day, horizon_end=29, quantile=None):
    """Total market drain per product from `day` to season end: the certain
    drain from shops already open, plus what future unlocks will add.

    `quantile=None` takes the EXPECTED future contribution (each unlock
    weighted by the chance it carries the product). A quantile in (0, 1)
    takes that percentile of the exact distribution instead: the unlock
    schedule is known, each draw is uniform over the eight shop types, and
    at most eight remain, so the sum is a short convolution rather than an
    approximation. Below 0.5 it is the conservative read -- plan for demand
    that probably arrives, not demand that might. This is the risk term: it
    shrinks the anchor early, when few shops are open and the draw could go
    either way, and converges on the mean late once the unlocks have happened."""
    current = town_drain_per_day(unlocked_shops)
    days_left = max(0, horizon_end - day)
    total = {item: current.get(item, 0.0) * days_left for item in PRODUCTS}
    budget = remaining_unlocks(day, len(unlocked_shops))
    future_days = [u for u in _UNLOCK_DAYS if u > day][:budget]
    if quantile is None:
        weights = shop_weight_per_day()
        for u in future_days:
            active = max(0, horizon_end - u)
            for item, w in weights.items():
                total[item] = total.get(item, 0.0) + w * active
        return total
    q = min(max(float(quantile), 0.0), 1.0)
    for item in PRODUCTS:
        outcomes = _unlock_outcomes(item)
        dist = {0.0: 1.0}
        for u in future_days:
            active = max(0, horizon_end - u)
            nxt = {}
            for acc, pa in dist.items():
                for pb, rate in outcomes:
                    v = acc + rate * active
                    nxt[v] = nxt.get(v, 0.0) + pa * pb
            dist = nxt
        cum = 0.0
        pick = 0.0
        for v in sorted(dist):
            cum += dist[v]
            pick = v
            if cum >= q:
                break
        total[item] = total.get(item, 0.0) + pick
    return total


def hire_marginal_cost(hires_today):
    """Cost of the next HIRE given hires already made today."""
    return FARM_HAND_COST_MULT * fib(hires_today)


def hire_total_cost(k):
    """Total cost of hiring k hands in one day: fib(k+1) - 1."""
    return sum(fib(i) for i in range(k))
