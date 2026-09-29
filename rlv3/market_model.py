"""Exact market arithmetic shared by every rlv3 component. No fitted anything.

Everything here is an identity over public engine state. The price function is
exact (kagent.econ mirrors the engine and tests/test_econ_parity.py keeps it
honest); the town's consumption schedule is exact given the open shops; the
supply arriving from either board is exact given the tiles, which are public
for both farms.

TWO ENGINE FACTS this module encodes, both verified against
kaggle_environments/envs/kaggriculture/kaggriculture.py:

1. DRAIN ORDER.  `interpreter` runs `_process_market(state, env)` and only then
   `_town_consume(env, state, step)`, both with the CURRENT step.  So the
   inventory a quote sees in step h reflects drains through step h-1.
   `_town_consume` fires the shop tick at `step % townShopSellInterval == 0`
   (4) and the town-centre tick at `step % townCenterSellInterval == 0` (24).
   hour == step % 24, so shop ticks land at hours 0, 4, 8, 12, 16, 20 and the
   compound tick (6 shops' worth + the town centre) lands at the END of the
   step whose hour is 0.  **Hour 1 is therefore the lowest-inventory hour of
   the day**, and a sell quoted there is quoted into the day's best price.

2. SLOT ORDER.  `_process_market` loops `for i in range(max_len)` over order
   slots and runs slot i for BOTH players before slot i+1.  Inside one slot
   the per-unit lockstep quotes both players at the same pre-commit inventory,
   so there is no first-seller premium WITHIN a slot -- but there is one
   ACROSS slots: our SELL at slot 0 is quoted at a strictly lower inventory
   than the opponent's SELL at their slot 3.
"""

from __future__ import annotations

from kagent import econ, features, planner
from kagent.constants import (
    MARKET_PARAMS,
    PRICE_FLOOR,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
    TURNS_PER_DAY,
)

# The town's consumption period, in STEPS. Everything in rlv3/selling.py that
# talks about "one tick of accrued drain" means this.
DRAIN_TICK = TOWN_SHOP_SELL_INTERVAL          # 4
CENTER_TICK = TOWN_CENTER_SELL_INTERVAL       # 24

# Shop ticks per day (6) and town-centre ticks per day (1).
SHOP_TICKS_PER_DAY = TURNS_PER_DAY / TOWN_SHOP_SELL_INTERVAL
CENTER_TICKS_PER_DAY = TURNS_PER_DAY / TOWN_CENTER_SELL_INTERVAL

# The lowest-inventory hour of the day, from engine fact 1.
BEST_SELL_HOUR = 1


# --------------------------------------------------------------- drain clock
def next_drain_step(step):
    """The next step at which the town consumes.

    `step` itself when step is already on a tick boundary -- the drain for
    that step fires at the END of it, after this step's market resolves.
    """
    return int(step) + (DRAIN_TICK - int(step) % DRAIN_TICK) % DRAIN_TICK


def ticks_between(step_a, step_b):
    """(shop ticks, town-centre ticks) VISIBLE between the observations at
    step_a and step_b, i.e. fired in the half-open interval [step_a, step_b).
    Zero when step_b <= step_a.

    CORRECTED 2026-09-26 against the engine, and it was one step out. The
    interpreter calls `_town_consume(env, state, step)` with the PRE-increment
    counter and the framework records the result at index step+1
    (kaggriculture.py:894-950, `_town_consume` drains when
    `step % interval == 0`). So the tick that fires during step s first appears
    in the observation at s+1, and the ticks between two OBSERVATIONS at a and
    b are the ones at a, a+1, ... b-1 -- not a+1 ... b, which is what this
    returned before. `(b-1)//T - (a-1)//T` counts the multiples of T in
    [a, b).

    The two conventions differ by exactly one tick whenever `a` or `b` sits on
    a boundary, and the cost of the old one was measured by the objective
    component on the same clock: 264 WHEAT units a game, showing up as a clean
    +107 at hour 0 against -126 at hour 1 in the inventory residual. Fixing it
    there took the opponent-flow solve to r +0.997.
    `rlv3.objective.ProjState._drain_between` is the other implementation of
    this clock and now agrees exactly; `rlv3/tests/test_agent.py` pins that.
    """
    a, b = int(step_a), int(step_b)
    if b <= a:
        return 0, 0
    return ((b - 1) // DRAIN_TICK - (a - 1) // DRAIN_TICK,
            (b - 1) // CENTER_TICK - (a - 1) // CENTER_TICK)


def per_tick_rates(item, unlocked_shops):
    """(units removed per shop tick, units removed per town-centre tick).

    Derived from econ.town_drain_per_day so the shop table is read in exactly
    one place; the division is exact because town_drain_per_day is built as
    center_ticks + sum(mult * shop_ticks).
    """
    per_day = econ.town_drain_per_day(unlocked_shops)
    center = 1.0 if item in TOWN_CENTER_PRODUCTS else 0.0
    shop = (per_day.get(item, 0.0) - center * CENTER_TICKS_PER_DAY) / SHOP_TICKS_PER_DAY
    return max(0.0, shop), center


def drain_between(item, step_a, step_b, unlocked_shops):
    """Exact units the town removes from `item`'s market inventory over the
    half-open step interval [step_a, step_b) -- the ticks VISIBLE between the
    OBSERVATIONS at step_a and step_b, exactly as `ticks_between` counts them.

    This docstring said "(step_a, step_b]" until 2026-09-27 and it was the
    cause of `sell-eval-drain-off-by-one`: `sell_eval.attach_opponent_fills`
    read it and asked for the drain that closes `inv[t] -> inv[t+1]` as
    `(t-1, t)`. The tick that closes that delta is the one fired DURING step
    t, so the window is `(t, t+1)`.
    """
    shop_ticks, center_ticks = ticks_between(step_a, step_b)
    if not shop_ticks and not center_ticks:
        return 0.0
    shop_rate, center_rate = per_tick_rates(item, unlocked_shops)
    return shop_ticks * shop_rate + center_ticks * center_rate


def drain_per_day(item, unlocked_shops):
    """econ.town_drain_per_day for one item."""
    return econ.town_drain_per_day(unlocked_shops).get(item, 0.0)


def is_drained(item, unlocked_shops):
    """True when the town actually buys this product faster than the bare
    town-centre trickle.

    MELON is in no shop and FERTILIZER is in no shop and not even a
    town-centre product, so their inventory is a one-way ratchet: nothing
    lifts the price back and the earliest sale is the best sale. Pacing them
    is strictly worse, which is why selling.plan_slices routes them around the
    schedule entirely.
    """
    return drain_per_day(item, unlocked_shops) > CENTER_TICKS_PER_DAY + 1e-9


# --------------------------------------------------------------- price curve
def price(item, inv):
    return econ.price(item, int(round(inv)))


def dPdI(item, inv):
    """The price drop caused by one more unit of inventory. Non-negative."""
    i = int(round(inv))
    return float(econ.price(item, i) - econ.price(item, i + 1))


def season_capacity(item, inv, floor_price, cap=10_000):
    """How deep the curve is above a price floor -- econ.units_sellable_above."""
    return econ.units_sellable_above(item, int(round(inv)), floor_price, cap=cap)


def crashed(item, market_price, ratio=0.35):
    """Price already below `ratio` of base: the scorecard's crashed-sale test."""
    return market_price <= ratio * MARKET_PARAMS[item]["base"]


# ------------------------------------------------------------- supply paths
def arrival_path(farm_tiles, day, horizon):
    """planner._rent_board_supply verbatim: per-day units each board delivers.

    Index k is `day + k`. Works for our tiles and for the opponent's, which
    are public.
    """
    return planner._rent_board_supply(farm_tiles, day, horizon)


def inventory_path(view, horizon_days):
    """planner._rent_inventory verbatim: I0 + both boards' projected arrivals
    - drain, per day, per product. Reused, not rewritten."""
    return planner._rent_inventory(view, horizon_days)


def own_remaining(view, item, horizon=None):
    """Units of `item` we still have to bring to market: what is in the shed
    and in hand, plus what our board delivers between now and season end."""
    if horizon is None:
        horizon = max(0, 29 - view.day)
    total = float(view.shed.get(item, 0))
    for idx, _, _ in view.me.units:
        total += view.unit_inv(idx).get(item, 0)
    if horizon > 0:
        path = arrival_path(view.me.tiles, view.day, horizon)
        for k in range(1, horizon + 1):
            total += max(0, path[k].get(item, 0))
    return total


def opponent_supply(view, item, horizon=7):
    """features.supply_projection over the opponent's public board.

    supply_projection ALREADY carries standing ripe units (`yield_units`) for
    both crops and animals, so nothing is added on top of it here: doing so
    would double-count the very units it is built from.
    """
    horizon = max(0, min(int(horizon), 29 - view.day))
    return float(features.supply_projection(view.opp, view.day, horizon).get(item, 0.0))


__all__ = [
    "DRAIN_TICK", "CENTER_TICK", "SHOP_TICKS_PER_DAY", "CENTER_TICKS_PER_DAY",
    "BEST_SELL_HOUR", "PRICE_FLOOR", "PRODUCTS", "SHOPS",
    "next_drain_step", "ticks_between", "per_tick_rates", "drain_between",
    "drain_per_day", "is_drained", "price", "dPdI", "season_capacity",
    "crashed", "arrival_path", "inventory_path", "own_remaining",
    "opponent_supply",
]
