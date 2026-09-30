"""Stage 2: sell timing on the EXACT market (rlv3/DESIGN_V4.md).

The market is an identity, not a forecast: price is a fixed function of the
inventory (econ.price), the town drains it on a known schedule (each shop
instance removes its products every TOWN_SHOP_SELL_INTERVAL steps, the town
centre one unit a day), our own sells add to it unit by unit, and the
opponent's sells are the one unknown -- read from the public inventory each
turn and, before that, from the corpus prior (kagent/opp_profile.py: the
hour-of-day share of their sells and their units a day by day band).

Measured 26 September: our realised price is 0.89 of a product's game mean,
the leaders' 1.04; 68% of our units sell at hours 0-3 into both farms' dawn
dumps, 39% of theirs at hours 20-23 after the shops have drained the market
all day. Batch pacing (sell_pace) and walking produce in early (evesell) were
both closed; this is the timing itself: the shed is held and sold on the
forecast price path.

Per product with stock s: over a horizon of H turns simulate the inventory
path I_t = I_{t-1} + ours_t + theirs_t - drain_t and find the reservation
price lambda at which selling every unit quoted at or above lambda along the
path disposes of exactly s (bisection; the price curve is monotone in the
inventory so "units above lambda" is an inventory gap). Sell now the units
quoted at or above lambda at the current inventory. Re-solved every turn, so
the horizon is a plan, not a commitment. The shed-pressure valve and the
final push bypass it, as they bypass the floors.
"""
from __future__ import annotations

from kagent import econ, scheduler
from kagent import opp_profile
from kagent.constants import (
    MARKET_PARAMS, SHOPS, TOWN_CENTER_PRODUCTS, TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL, TURNS_PER_DAY,
)

PRODUCTS_DP = ("STRAWBERRY", "MILK", "WOOL")   # the steep curves the note measured; melon is a harvest race
PRODUCTS_DP2 = ("STRAWBERRY", "MILK")          # sell_dp2: wool left to the shipped floors


def active_products():
    if scheduler.EXEC.get("sell_dp"):
        return PRODUCTS_DP
    if scheduler.EXEC.get("sell_dp2"):
        return PRODUCTS_DP2
    return ()
HORIZON = 48            # turns: two drain cycles; prices only fall over the season where supply beats the drain
SELL_FINAL_STEP = 718
EMA = 0.25              # weight of one observed day on the opponent's daily volume
MIN_HOLD_PRICE = 2      # below this a unit is not worth a turn of anyone's

_INV_CACHE = {}


def drain_at(step, unlocked_shops):
    """Units the town removes from each product's inventory AFTER the market
    at `step` (engine _town_consume)."""
    out = {}
    if step % TOWN_SHOP_SELL_INTERVAL == 0:
        for shop in unlocked_shops:
            items = SHOPS.get(shop, ())
            mult = 2 if len(items) == 1 else 1
            for item in items:
                out[item] = out.get(item, 0) + mult
    if step % TOWN_CENTER_SELL_INTERVAL == 0:
        for item in TOWN_CENTER_PRODUCTS:
            out[item] = out.get(item, 0) + 1
    return out


def inv_at_price(item, lam):
    """The largest inventory at which `item` still quotes at or above `lam`
    (econ.price is monotone non-increasing in the inventory)."""
    key = (item, int(lam))
    if key in _INV_CACHE:
        return _INV_CACHE[key]
    p = MARKET_PARAMS[item]
    lo, hi = int(p["I0"] - 4 * p["T"]), int(p["I0"] + 6 * p["T"])
    if econ.price(item, lo) < lam:
        _INV_CACHE[key] = lo - 1
        return lo - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if econ.price(item, mid) >= lam:
            lo = mid
        else:
            hi = mid
    _INV_CACHE[key] = lo
    return lo


def _band(day):
    for k in opp_profile.UNITS_PER_DAY["STRAWBERRY"]:
        lo, hi = (int(x) for x in k.split("-"))
        if lo <= day <= hi:
            return k
    return "21-26"


def observe_market(view, memory):
    """Their fills last turn = inventory delta net of our requests and the
    drain; folded into a per-product daily volume (EMA over days) and an
    hour profile (counts, blended with the prior by weight)."""
    st = memory.st_state = getattr(memory, "st_state", None) or {
        "prev_step": None, "prev_inv": {}, "ours": {}, "day_units": {}, "day": -1,
        "daily": {}, "hour_units": {p: [0.0] * TURNS_PER_DAY for p in PRODUCTS_DP}, "hour_n": {p: 0.0 for p in PRODUCTS_DP},
    }
    step = view.step
    if st["prev_step"] is not None and step == st["prev_step"] + 1:
        drain = drain_at(st["prev_step"], view.unlocked_shops)
        for item in PRODUCTS_DP:
            prev = st["prev_inv"].get(item)
            now = view.market_inv.get(item)
            if prev is None or now is None:
                continue
            theirs = max(0, now - prev + drain.get(item, 0) - st["ours"].get(item, 0))
            st["day_units"][item] = st["day_units"].get(item, 0) + theirs
            h = st["prev_step"] % TURNS_PER_DAY
            st["hour_units"][item][h] += theirs
            st["hour_n"][item] += theirs
    if view.day != st["day"]:
        if st["day"] >= 0:
            for item in PRODUCTS_DP:
                u = st["day_units"].get(item, 0)
                prev = st["daily"].get(item)
                st["daily"][item] = u if prev is None else (1 - EMA) * prev + EMA * u
        st["day"] = view.day
        st["day_units"] = {}
    st["prev_step"] = step
    st["prev_inv"] = {item: view.market_inv.get(item) for item in PRODUCTS_DP}
    st["ours"] = {}


def note_requests(memory, orders):
    st = getattr(memory, "st_state", None)
    if st is None:
        return
    for o in orders:
        if o and o[0] == "SELL" and len(o) > 2 and o[1] in PRODUCTS_DP:
            st["ours"][o[1]] = st["ours"].get(o[1], 0) + int(o[2])


def their_supply(item, view, memory, step):
    """Forecast units the opponent sells at `step` (an hour of a day)."""
    st = getattr(memory, "st_state", None) or {}
    day = step // TURNS_PER_DAY
    prior_daily = opp_profile.UNITS_PER_DAY[item][_band(day)]
    daily = st.get("daily", {}).get(item)
    daily = prior_daily if daily is None else 0.5 * (daily + prior_daily)
    hour = step % TURNS_PER_DAY
    n_obs = st.get("hour_n", {}).get(item, 0.0)
    prior_share = opp_profile.HOUR_SHARE[item][hour]
    if n_obs > 0:
        obs_share = st["hour_units"][item][hour] / n_obs
        w = min(0.7, n_obs / (n_obs + 40.0))
        share = (1 - w) * prior_share + w * obs_share
    else:
        share = prior_share
    return daily * share


def sell_now(item, stock, view, memory):
    """Units of `item` to sell this turn out of `stock` on the exact path."""
    if stock <= 0:
        return 0
    step = view.step
    inv0 = view.market_inv.get(item)
    if inv0 is None:
        return stock
    H = min(HORIZON, SELL_FINAL_STEP - 8 - step)
    if H <= 1:
        return stock
    theirs = [their_supply(item, view, memory, step + k) for k in range(H)]
    drain = [drain_at(step + k, view.unlocked_shops).get(item, 0) for k in range(H)]

    def simulate(lam):
        cap = inv_at_price(item, lam)
        I, rem, first = inv0, stock, None
        for k in range(H):
            n = min(rem, max(0, cap - I))
            if first is None:
                first = n
            I += n
            rem -= n
            I = max(0, I + theirs[k] - drain[k])
            if rem <= 0:
                break
        return stock - rem, first

    lo, hi = float(MIN_HOLD_PRICE), float(econ.price(item, inv_at_price(item, MIN_HOLD_PRICE)) )
    hi = max(hi, float(econ.price(item, int(MARKET_PARAMS[item]["I0"] - 4 * MARKET_PARAMS[item]["T"]))))
    sold_lo, first_lo = simulate(lo)
    if sold_lo < stock:
        return first_lo               # even at the floor the horizon cannot clear it: sell what it takes now
    for _ in range(18):
        mid = (lo + hi) / 2
        sold, _first = simulate(mid)
        if sold >= stock:
            lo = mid
        else:
            hi = mid
    _sold, first = simulate(lo)
    return max(0, min(stock, first))
