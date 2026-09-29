"""The load-bearing arithmetic of the allocation: $/tile-day, priced on the
engine's exact market identity. No fitted model anywhere in this file.

WHY A WATER-FILL AND NOT AN ARGMAX. `planner._rent_pick` already prices a
block of tiles at the inventory its own units will meet on its own harvest day
-- the forward market model, the project's first fitted-free win (+$3,185 +/-
768, replicated). But it takes the ARGMAX over crops and hands the whole free
block to one winner. An argmax over a concave objective is a CORNER solution
by construction, and a corner solution is exactly the 41-strawberry board the
scorecard measures against the leaders' 13-24.

`block_rent` is concave in n because `econ.sell_revenue` is: the average unit
price of a solo sell at I0 falls 116.0 -> 82.5 (strawberry), 33.6 -> 30.2
(carrot), 24.2 -> 22.7 (wheat), 156.0 -> 119.1 (milk) and 199.6 -> 170.2
(wool) from 5 units to 40. Concavity is what makes an equal-marginal SHARE
split well-posed, and `water_fill` is that split.

WHY THE OPPONENT TERM. The score is relative: a win is our money above theirs.
Their supply is worth $52,652 a game to us through the one shared inventory
(z 17.52), 97% of it strawberry + milk + wool -- the three lines we are
heaviest in -- and every refuted arm this month handed them $2-7k. So a
dollar their supply loses because our units moved the price is worth a dollar
we earn, and `relative_rent` adds it in, weighted by stance.

WHY THE WHOLE STREAM, NEVER THE MARGINAL PRICE. The geese swap lost -$5,234
(z -7.62) and -$7,934 (z -10.11), losing on WINS as well as money, because
"the last 25 milk units earn $1.40" was applied to swapping a whole cow whose
FIRST 20 units sell at $140. Every function here integrates the entire
remaining stream through `econ.sell_revenue`; none of them ever multiplies a
count by a single marginal price.
"""

from kagent.constants import ANIMALS, CROPS, MARKET_I0, MARKET_PARAMS
from kagent import econ, features, planner

SEASON_END = 29                  # last day a harvest can land
PLANTABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_KINDS = ("GOOSE", "COW", "SHEEP")

# Stance weight on the opponent's loss, per dollar. AHEAD protects a lead and
# takes the cheap profit; BEHIND crowds the pool they are heaviest in, because
# denying is the only lever left on a margin we cannot out-earn.
W_OPP_DEFAULT = {"AHEAD": 0.25, "LEVEL": 0.5, "BEHIND": 1.0}


# --- caches -----------------------------------------------------------------
# `econ.sell_revenue` walks the price curve one unit at a time, and the
# water-fill asks for it a few hundred times a dawn. The ladder for one
# (item, starting inventory) is a prefix sum of exactly that walk, so it is
# computed once and indexed after. Nothing is approximated: ladder[n] IS
# econ.sell_revenue(item, inv, n) for every n.
_LADDER = {}
_LADDER_MAX = 512


def revenue_ladder(item, inv, max_units):
    key = (item, int(inv))
    acc = _LADDER.get(key)
    if acc is None or len(acc) <= int(max_units):
        need = max(int(max_units) + 1, 64)
        if len(_LADDER) > _LADDER_MAX:
            _LADDER.clear()
        acc = [0.0]
        total = 0.0
        for p in econ.sell_marginal_prices(item, int(inv), need):
            total += p
            acc.append(total)
        _LADDER[key] = acc
    return acc


def sell_revenue(item, inv, n):
    """econ.sell_revenue with the ladder cache in front of it."""
    n = int(n)
    if n <= 0:
        return 0.0
    return revenue_ladder(item, inv, n)[n]


def opp_supply(view):
    """`features.supply_projection` for the opponent, memoised on the View
    (one board scan a dawn instead of one per candidate evaluation)."""
    key = getattr(view, "_rlv3_supply", None)
    if key is not None and key[0] == view.step:
        return key[1]
    day = view.day
    try:
        sup = features.supply_projection(view.opp, day, horizon=max(0, SEASON_END - day))
    except Exception:
        sup = {}
    try:
        view._rlv3_supply = (view.step, sup)
    except Exception:
        pass
    return sup


# --- inventory --------------------------------------------------------------

def inventory_path(view, horizon=14):
    """Projected market INVENTORY (not price) per product for days 0..horizon.

    Delegates to `planner._rent_inventory`, which is our board's supply plus
    the OPPONENT's board supply (`planner._rent_board_supply(view.opp.tiles,
    ...)`) minus the town drain minus the future-unlock drain. Returning the
    PATH is the point: every crop is then priced at the inventory its own
    units will meet on its own harvest day, not at today's price.
    """
    cached = getattr(view, "_rlv3_path", None)
    if cached is not None and cached[0] == (view.step, int(horizon)):
        return cached[1]
    try:
        path = planner._rent_inventory(view, int(horizon))
    except Exception:
        path = {}
    if not path:
        cur = dict(getattr(view, "market_inv", {}) or {})
        path = {p: [float(cur.get(p, MARKET_I0))] * (int(horizon) + 1)
                for p in MARKET_PARAMS}
    try:
        view._rlv3_path = ((view.step, int(horizon)), path)
    except Exception:
        pass
    return path


def inv_at(path, item, k):
    """Inventory of `item` k days out, clamped to a level `econ.price` accepts."""
    row = path.get(item)
    if not row:
        return MARKET_I0
    k = max(0, min(int(k), len(row) - 1))
    return max(0, int(round(row[k])))


# --- per-tile and per-animal yield shapes (engine constants, no fitting) -----

def one_time_shape(cd):
    """(units, ripe) for a one-time crop watered daily and never fertilised.

    `ripe` is the AGE at which watering stops adding units -- the age the tile
    is actually harvested at -- and it is NOT `max_yield_day`. Engine
    `_apply_unit_action` WATER: yield_units starts at 1 and gains one per
    watered day over ages [(max_yield_day+1)//2, max_yield_day], capped at
    `max_yield`. Wheat and carrot run out of WINDOW first, so the cap lands on
    `max_yield_day`; MELON runs out of CAP first -- window ages 6-12, but 1 + 5
    waterings reaches 6 at age 10, so ages 11-12 add nothing (README "Object
    Types": "Melon's bonus window is ages 6-12, but base 1 plus one unit per
    watered day reaches the cap of 6 at age 10, so ages 11-12 add nothing").
    Verified against the engine: MELON planted day 19 harvests 6 on day 29.

    Using `max_yield_day` as the ripe age cost two things. It gated MELON off
    on days 18-19, which the engine and `features.LAST_PLANT` both allow -- two
    days of the highest-value one-time crop refused. And it charged every melon
    tile 13 tile-days instead of 11, understating melon $/tile-day by 18%
    (147.45 -> 124.77 at day 6) against this module's own `per_tile_day`
    (0.545) and README's "Yield / tile / day" column.

    `first_yield_day` floors it: the engine's HARVEST returns early while
    `day - planted_day < first_yield_day`, so a crop whose bonus window closed
    before that still cannot be picked. No shipped crop needs the floor -- it
    is here so the branch is not one more thing nothing executes.
    """
    ws = (cd["max_yield_day"] + 1) // 2
    units = min(cd["max_yield"], 1 + cd["max_yield_day"] - (ws - 1))
    ripe = max(cd["first_yield_day"], (ws - 1) + (units - 1))
    return units, ripe


def per_tile_day(crop):
    """Units one tile of `crop` yields per day of occupancy.

    `features._PER_TILE_DAY` where it has an entry (strawberry's 0.5 is the
    fertilised+watered rate, twice the bare formula), the engine formula
    otherwise -- which is the only source for WHEAT, and which is the whole
    carrot argument: carrot turns 1.0 units/tile-day where strawberry turns
    0.5, and we stand 41 strawberry to 1 carrot at day 20.
    """
    table = getattr(features, "_PER_TILE_DAY", {})
    if crop in table:
        return float(table[crop])
    cd = CROPS[crop]
    if cd["ongoing"]:
        span = cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
        return cd["max_yield"] / float(max(1, span))
    units, ripe = one_time_shape(cd)
    return units / float(max(1, ripe + 1))


def crop_block(crop, day, horizon=14):
    """(k, units_per_tile, occupies) for a tile planted on `day`, or None if
    its harvest can no longer land by the season end.

    Mirrors `planner._rent_pick`'s block shape on the ONGOING branch. The
    one-time branch deliberately does not: `_rent_pick` prices those tiles at
    `max_yield` over `max_yield_day + 1` days, which is the fertilised cap over
    the bonus window, and `block_revenue` never buys a fertilizer. Here `units`
    is the total the engine delivers on WATER alone and `k`/`occupies` come off
    the RIPE age (`one_time_shape`), the age the engine actually caps and the
    tile is actually picked -- so `k` really is the crop's own harvest lag into
    the inventory path. See the comment in the branch.
    Action value decays 13x over a season ($1,317 on day 3 to $103 on day 27),
    so the rent is per tile-DAY of occupancy -- which is what makes short-cycle
    crops win the late board with no LAST_PLANT table to tell them to.
    """
    cd = CROPS.get(crop)
    if cd is None:
        return None
    if cd["ongoing"]:
        span = max(1, SEASON_END - day)
        if span <= cd["first_yield_day"]:
            return None
        k = min(horizon, cd["first_yield_day"])
        units = min(cd["max_yield"], (span - cd["first_yield_day"]) // cd["interval"] + 1)
        occupies = span
    else:
        # WATERED-ONLY TOTAL, NOT `cd["max_yield"]`, and the RIPE age, NOT
        # `max_yield_day` -- both out of `one_time_shape`, whose docstring
        # carries the engine derivation.
        #
        # `max_yield` is the FERTILISED cap -- README "Object Types" note:
        # "Wheat and Carrot only reach their listed Max Yield of 6 and 4 with
        # fertilizer; watering alone peaks at 4 and 3" -- and `block_revenue`
        # never charges the fertilizer. Pricing the fill at the fertilised cap
        # overstated wheat by 48% and carrot by 31% against strawberry and
        # planted the wrong mix. This is the identical total
        # `planner._rent_board_supply` already feeds into the inventory path
        # priced two lines down.
        #
        # `max_yield_day` is the end of the bonus WINDOW, which for MELON is
        # two days past the age the cap is reached; gating and occupancy both
        # belong on the ripe age, and only then does units/occupies equal
        # `per_tile_day` and README's "Yield / tile / day" for all three
        # (wheat 0.800, carrot 0.750, melon 6/11 = 0.545).
        units, ripe = one_time_shape(cd)
        if day + ripe > SEASON_END:
            return None
        k = min(horizon, ripe)
        occupies = ripe + 1
    if units <= 0 or occupies <= 0:
        return None
    return k, float(units), float(occupies)


def animal_block(kind, day, horizon=14):
    """(k, units_per_animal, days_held) for an animal bought on `day`, or None.

    Units come from `features._OUT_PER_DAY` (the fed+cared rate the rest of
    the agent prices animals at), integrated over the productive days left.
    """
    a = ANIMALS.get(kind)
    if a is None:
        return None
    span = max(0, SEASON_END - day)
    prod_days = span - a["first_yield_day"]
    if prod_days <= 0:
        return None
    rate = features._OUT_PER_DAY.get(a["product"], 1.0 / max(1, a["interval"]))
    units = float(rate) * prod_days
    if units <= 0:
        return None
    return min(horizon, a["first_yield_day"]), units, float(span)


# --- the four prices --------------------------------------------------------

def block_rent(view, crop, n, path, *, day=None, occupies=None):
    """$/tile-day for a block of n tiles of `crop`.

    Priced with `econ.sell_revenue(crop, path[crop][k], units * n)` at the
    crop's own harvest lag k, so OUR OWN block's self-impact is inside the
    number. That self-impact is the replicated +$3,185 +/- 768 win, and it is
    what makes this function CONCAVE in n.
    """
    rev, occ, _u, _i = block_revenue(view, crop, n, path, day=day, occupies=occupies)
    if n <= 0 or occ <= 0:
        return 0.0
    return rev / float(n * occ)


def block_revenue(view, crop, n, path, *, day=None, occupies=None):
    """(revenue, occupies, total_units, inventory_at_k) -- block_rent's parts."""
    day = view.day if day is None else int(day)
    spec = crop_block(crop, day, horizon=_horizon(path))
    if spec is None or n <= 0:
        return 0.0, 0.0, 0, MARKET_I0
    k, units, occ = spec
    if occupies is not None:
        occ = float(occupies)
    total = int(round(units * n))
    if total <= 0:
        return 0.0, occ, 0, MARKET_I0
    inv = inv_at(path, crop, k)
    return float(sell_revenue(crop, inv, total)), occ, total, inv


def opponent_delta(view, item, inv, n):
    """Revenue the OPPONENT loses because our n units moved the shared pool.

    Their projected remaining units of `item` (`features.supply_projection`,
    their farm is fully public) times the price drop our block causes. The
    market is one shared inventory and the score is relative, so this is a
    gain to us of exactly the same size as a dollar we earn.
    """
    n = int(round(n))
    if n <= 0 or item not in MARKET_PARAMS:
        return 0.0
    sup = opp_supply(view)
    units = max(0.0, float(sup.get(item, 0.0)))
    if units <= 0.0:
        return 0.0
    inv = max(0, int(round(inv)))
    drop = econ.price(item, inv) - econ.price(item, inv + n)
    return units * max(0.0, float(drop))


def w_opp(stance, params=None):
    table = dict(W_OPP_DEFAULT)
    if params:
        table.update((params.get("rent", {}) or {}).get("w_opp", {}) or {})
    return float(table.get(stance, table["LEVEL"]))


def block_value(view, crop, n, path, *, stance="LEVEL", day=None, params=None):
    """TOTAL dollars of a block of n crop tiles: our revenue plus the weighted
    revenue the opponent loses to it. The quantity `marginal_rent` differences."""
    rev, occ, total, inv = block_revenue(view, crop, n, path, day=day)
    if total <= 0:
        return 0.0, occ
    w = w_opp(stance, params)
    if w > 0.0:
        rev += w * opponent_delta(view, crop, inv, total)
    return rev, occ


def relative_rent(view, crop, n, path, *, stance="LEVEL", day=None, params=None):
    """block_rent + w_opp(stance) * opponent_delta, per tile-day.

    The win-rate objective expressed in dollars per tile: this is the whole
    difference from maximising our own bank, which is what every learned
    component here has done while the opponent banked faster.
    """
    val, occ = block_value(view, crop, n, path, stance=stance, day=day, params=params)
    if n <= 0 or occ <= 0:
        return 0.0
    return val / float(n * occ)


def marginal_rent(view, crop, n_already, path, *, step=1, stance="LEVEL",
                  day=None, params=None):
    """relative_rent(n_already + step) - relative_rent(n_already), per tile-day:
    the quantity the water-fill ranks.

    Priced over the WHOLE remaining stream of the marginal tiles (both terms
    go through `econ.sell_revenue`), NEVER at the current marginal price --
    see the module docstring for the two refutations that cost.
    """
    step = max(1, int(step))
    a, occ = block_value(view, crop, int(n_already), path, stance=stance,
                         day=day, params=params)
    b, occ_b = block_value(view, crop, int(n_already) + step, path, stance=stance,
                           day=day, params=params)
    occ = occ_b or occ
    if occ <= 0:
        return 0.0
    return (b - a) / float(step * occ)


def animal_rent(view, kind, n_already, path, chore_cost, *, step=1, stance="LEVEL",
                day=None, params=None, labour_price=0.0):
    """Marginal $/animal-day for the (n_already + step)-th animal of `kind`.

    Gross is the whole remaining product stream through `econ.sell_revenue`
    plus the weighted opponent loss, exactly as for crops. Then three costs
    the engine actually charges:
      * `chore_cost` measured unit-turns per animal per day (FEED + CARE +
        PICKUP + HARVEST + walking), priced at `labour_price` -- the tile-day
        rent the SAME labour would have earned on crops (the equal-marginal
        shadow price out of the crop water-fill);
      * one WHEAT of feed per animal per day, bought back at the path price;
      * minus one FERTILIZER per animal per night, which is revenue.

    This is the term that makes geese arithmetic-negative at OUR executor cost
    (pickups per feed 1.08 against the leaders' 0.46-0.55) and lets them turn
    positive only if the Maia2 executor actually lowers `chore_cost`. Nothing
    about geese, cows or sheep is hardcoded in or out anywhere in rlv3.
    """
    day = view.day if day is None else int(day)
    spec = animal_block(kind, day, horizon=_horizon(path))
    if spec is None:
        return 0.0
    k, units_each, days = spec
    item = ANIMALS[kind]["product"]
    step = max(1, int(step))
    w = w_opp(stance, params)
    rp = (params or {}).get("rent", {}) or {}
    hp = (params or {}).get("herd", {}) or {}

    inv = inv_at(path, item, k)
    k1 = min(1, _horizon(path))
    fert_inv = inv_at(path, "FERTILIZER", k1)
    wheat_inv = inv_at(path, "WHEAT", k1)
    feed_per_day = float(hp.get("feed_per_day", 1.0)) if rp.get("include_feed", True) else 0.0
    fert_per_day = (float(hp.get("fertilizer_per_day", 1.0))
                    if rp.get("include_fertilizer", True) else 0.0)

    def value(n):
        """TOTAL dollars a herd of n of this kind is worth over the rest of
        the season. EVERY stream is integrated, never a count times a price:
        the product, the one FERTILIZER an animal drops a night (base $100 --
        big enough that pricing it marginally would have dominated the whole
        decision), and the one WHEAT it eats, through the engine's own
        buy_cost, which quotes at inventory - 1."""
        if n <= 0:
            return 0.0
        total = int(round(units_each * n))
        if total <= 0:
            return 0.0
        v = float(sell_revenue(item, inv, total))
        if w > 0.0:
            v += w * opponent_delta(view, item, inv, total)
        if fert_per_day > 0.0:
            v += float(sell_revenue("FERTILIZER", fert_inv,
                                    int(round(fert_per_day * days * n))))
        if feed_per_day > 0.0:
            v -= float(econ.buy_cost("WHEAT", wheat_inv,
                                     int(round(feed_per_day * days * n))))
        return v

    gross = (value(int(n_already) + step) - value(int(n_already))) / float(step * days)
    return gross - float(chore_cost) * float(labour_price)


# --- the split --------------------------------------------------------------

def water_fill(capacity, candidates, score_fn, *, floors=None, caps=None, step=1):
    """Greedy equal-marginal allocation -> (allocation, trace).

    Hands out `capacity` units one `step` at a time to whichever candidate has
    the highest marginal score, stopping a candidate at its cap or the moment
    its marginal turns negative. Because every score here is concave in the
    amount already allocated, greedy IS the equal-marginal optimum, and the
    marginal of the LAST accepted step is the shadow price of one unit of
    capacity -- which is what prices labour for `animal_rent`.

    `trace` is [(name, running_total, marginal)] and is the mechanism check
    the ladder script prints: it says which crop the direction is wrong on if
    the dose ladder comes out negative.
    """
    floors = dict(floors or {})
    caps = dict(caps or {})
    step = max(1, int(step))
    alloc = {c: 0 for c in candidates}
    trace = []
    left = int(max(0, capacity))
    for c in candidates:
        f = int(floors.get(c, 0))
        cap = caps.get(c)
        if cap is not None:
            f = min(f, int(cap))
        f = max(0, min(f, left))
        if f > 0:
            alloc[c] = f
            left -= f
            trace.append((c, f, float("inf")))
    guard = 0
    limit = 4 * (int(max(0, capacity)) // step + 2)
    while left >= step and guard < limit:
        guard += 1
        best = None
        for c in candidates:
            cap = caps.get(c)
            if cap is not None and alloc[c] + step > int(cap):
                continue
            m = score_fn(c, alloc[c], step)
            if m <= 0.0:
                continue
            if best is None or m > best[1]:
                best = (c, m)
        if best is None:
            break
        c, m = best
        alloc[c] += step
        left -= step
        trace.append((c, alloc[c], m))
    return alloc, trace


def _horizon(path):
    for row in path.values():
        return max(0, len(row) - 1)
    return 14
