"""Sell scheduling from the deterministic market: WHEN to sell, HOW MUCH, and
in WHICH market-order slot.

Drop-in replacement for kagent.market.build_orders' sell half. The shipped
builder is still called exactly once, for everything that is not a SELL.

WHY THIS IS NOT `sell_pace` AGAIN
---------------------------------
`sell_pace` (kagent.market._sell_orders, plan.sell_pace >= 0) capped an order
at the units that keep the within-order price drop under N%, and
`_sell_orders` then re-offered the remainder ON THE VERY NEXT STEP. The town
drains at `step % 4 == 0`, so on three steps out of four the market has not
refilled at all and the total walk down the curve is arithmetically unchanged.
That is exactly why it read flat (+/-0.5% realised price, no eval arm).

`min_slice_gap = 4` is the whole mechanical difference: a slice is emitted
only when at least one drain tick has fired since this item's last slice, and
it is sized by the drain that ACCRUED in that gap. The schedule is anchored to
the engine's sawtooth rather than to our own turn counter. `slice_gap_steps`
(rlv3/sell_eval.py) is the anti-self-deception check: median gap below 4 means
the schedule collapsed back into sell_pace and the arm is the refuted one.

WHAT THE RECORD SAYS WILL FAIL
------------------------------
The repo record is hostile to holding: paced selling above a floor "let the
opponent sell first and lost $30k"; premium_scale is monotone toward selling
SOONER; evesell (evening share past every leader) gated -1.4 pts. Both
scorecard metrics this module is briefed to move have already been moved to
leader levels WITHOUT moving money or wins.

So everything that can hold stock back is switchable and defaults to the
least-holding setting the spec allows:
  * `cfg.participation` >= 1.0 scales the slice up, never down.
  * `cfg.enabled = False` makes build_orders byte-identical to the shipped one
    except for the slot re-ordering.
  * `cfg.slots_only = True` ships ONLY the slot re-ordering (free: it changes
    no walking, no allocation, and no sizing) with the shipped sizer intact.
  * `cfg.paced_only = True` restricts the schedule to the four steep-curve
    products, leaving WHEAT/CARROT/TOMATO/EGG on shipped sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kagent import econ, market
from kagent.constants import (
    ANIMALS,
    MARKET_PARAMS,
    MAX_MARKET_ORDERS,
    PRODUCTS,
    TURNS_PER_DAY,
)
from kagent.planner import FINAL_STEP, LIQUIDATION_DAY

from rlv3 import market_model as mm

# Products whose price falls on a steep curve above I0 (linear / sq above
# functions). Same tuple the shipped sizer paces.
PACED = market.PACED

# Never paced: their inventory is a one-way ratchet (see mm.is_drained).
RATCHET = ("MELON", "FERTILIZER")

# The SHIPPED kagent.market.build_orders, captured by install(). See the
# comment in build_orders(): install() rebinds the module attribute, so the
# attribute is no longer a safe way to reach the shipped builder.
_SHIPPED = None

SHED_PRESSURE = 80          # shipped valve: midnight overflow is destroyed
FINAL_PUSH_STEPS = 8        # shipped: step >= FINAL_STEP - 8 is the liquidation
LEAD_SCALE = 20_000.0       # money lead that saturates risk_posture


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class SellConfig:
    participation: float = 1.0    # slice size as a multiple of accrued drain
    impact_cap_pct: int = 10      # max within-order walk, % of the first unit
    min_slice_gap: int = 4        # STEPS between slices of one item == one tick
    sell_slots_lead: int = 3      # leading market-order slots reserved for SELL
    deny_ratio: float = 1.3       # opp exposure / ours above which we dump
    behind_hold: float = 1.15     # floor multiplier when losing and holding is
                                  # the only variance left
    liquidate_day: int = 27
    quantile_early: float = 0.35  # forecast_drain quantile, days 0-14
    quantile_late: float = 0.50   # days 15+

    # --- switches (see module docstring; the record is hostile to holding) --
    enabled: bool = True          # False -> shipped sizer, slot order only
    slots_only: bool = False      # True  -> shipped sizer, slot order only
    paced_only: bool = False      # True  -> schedule only market.PACED
    exposure_horizon: int = 7     # features.supply_projection horizon
    lam_alpha: float = 0.5        # EWMA on the dawn residual correction
    lam_lo: float = 0.5
    lam_hi: float = 2.0


# RE-MEASURED 2026-09-27, rlv3/sell_eval.py on 2 champ-h10 traces (bands 0 and
# 1), opponent fills solved exactly from the inventory identity, and now with
# the engine's same-slot lockstep (sell_eval.lockstep_revenue). The earlier
# 2026-09-26 figures credited us a first-seller premium INSIDE a slot that the
# engine does not pay, and the opponent is already at slot 0 on 46 of the 71
# contested product-steps, so the slot column was ~40% too high:
#
#                                           2026-09-26      2026-09-27
#   full schedule (this module's defaults)  MARGIN   +$891  MARGIN   +$656
#   paced_only=True                         MARGIN +$1,040  MARGIN   +$816
#   slots_only=True (order slot move only)  MARGIN +$1,281  MARGIN   +$772
#                                           our +$640       our  +$401
#                                           opp -$640       opp  -$372
#
# The ordering is unchanged, so RECOMMENDED is unchanged: the sizing half moves
# ONE shared curve and the opponent sells into it too, so it is margin-NEGATIVE
# on top of the slot move (+$656 vs +$772). Only the slot re-order is a
# transfer rather than a shared lift, and it withdraws no supply at all. Per-
# world sd is $34,749, so neither clears a 995-game gate on its own -- and at
# +$772 the slot move is half a percent of that sd, so it is worth carrying
# ONLY bundled with a component that moves allocation, never on its own.
RECOMMENDED = None    # set below, once SellConfig exists


@dataclass
class SellState:
    """Lives at `memory.rlv3_sell`. Reset when view.step goes backwards -- the
    arena reuses processes, so a fresh episode can land on a warm object."""
    last_slice_step: dict = field(default_factory=dict)
    sold_today: dict = field(default_factory=dict)
    day_target: dict = field(default_factory=dict)
    carry: dict = field(default_factory=dict)
    lam: dict = field(default_factory=dict)
    predicted_dawn_inv: dict = field(default_factory=dict)
    day: int = -1
    last_step: int = -1           # bookkeeping for the backwards-step reset
    dumping: dict = field(default_factory=dict)   # item -> we are the smaller
                                                  # supplier, so dump not pace

    def reset(self):
        self.last_slice_step = {}
        self.sold_today = {}
        self.day_target = {}
        self.carry = {}
        self.lam = {}
        self.predicted_dawn_inv = {}
        self.dumping = {}
        self.day = -1
        self.last_step = -1


def state_for(memory):
    st = getattr(memory, "rlv3_sell", None)
    if st is None:
        st = SellState()
        memory.rlv3_sell = st
    return st


def _sync(view, memory):
    """state_for + the episode-boundary reset."""
    st = state_for(memory)
    if view.step < st.last_step:
        st.reset()
    st.last_step = view.step
    return st


# ----------------------------------------------------------- dawn self-check
def dawn_update(view, memory, cfg=None):
    """Hour-0 only. The self-correcting loop.

    Inventory is an identity: inv(t) = I0 + sold - bought - drained. So at
    dawn the residual (actual - predicted) is EXACTLY the part of yesterday we
    got wrong, and the only unknown in it is the opponent's realised supply
    plus the actual shop draw. Nothing here is fitted; a clip bounds the damage
    when the residual is really a shop draw.
    """
    cfg = cfg or SellConfig()
    st = _sync(view, memory)
    if view.hour != 0 or st.day == view.day:
        return st
    first_day = st.day < 0
    st.day = view.day

    # 1. residual -> lam (multiplicative correction on the opponent's board)
    if not first_day:
        for item in PRODUCTS:
            pred = st.predicted_dawn_inv.get(item)
            if pred is None:
                continue
            actual = view.market_inv.get(item)
            if actual is None:
                continue
            err = float(actual) - float(pred)
            base = st.lam.get(item, 1.0)
            # The residual is supply we did not see. Scale the opponent term by
            # how big the miss is against the supply we DID predict from their
            # board; a zero denominator leaves lam alone.
            denom = max(1.0, abs(mm.opponent_supply(view, item, cfg.exposure_horizon)))
            ratio = 1.0 + err / denom
            ratio = min(max(ratio, cfg.lam_lo), cfg.lam_hi)
            lam = (1.0 - cfg.lam_alpha) * base + cfg.lam_alpha * ratio
            st.lam[item] = min(max(lam, cfg.lam_lo), cfg.lam_hi)

    # 2. re-read the shop set; season demand at the conservative quantile
    q = cfg.quantile_early if view.day < 15 else cfg.quantile_late
    season_demand = econ.forecast_drain(view.unlocked_shops, view.day, quantile=q)

    # 4. carry: yesterday's unmet target, rolled forward (zeroed in liquidation
    #    so an unmet pace target cannot block the endgame ramp)
    liquidating = view.day >= min(cfg.liquidate_day, LIQUIDATION_DAY)
    new_carry = {}
    if not first_day and not liquidating:
        for item, tgt in st.day_target.items():
            new_carry[item] = max(0.0, tgt - st.sold_today.get(item, 0))
    st.carry = new_carry
    st.sold_today = {}

    # 3. day_target: everything still to come, spread over the days left, and
    #    never above what the town will actually absorb.
    days_left = max(1, 29 - view.day)
    horizon = max(0, 29 - view.day)
    my_arrivals = mm.arrival_path(view.me.tiles, view.day, horizon) if horizon else []
    st.day_target = {}
    st.dumping = {}
    for item in PRODUCTS:
        if item in ANIMALS:
            continue
        remaining = float(view.shed.get(item, 0))
        for idx, _, _ in view.me.units:
            remaining += view.unit_inv(idx).get(item, 0)
        for k in range(1, len(my_arrivals)):
            remaining += max(0, my_arrivals[k].get(item, 0))
        drain_rate = mm.drain_per_day(item, view.unlocked_shops)
        # forecast_drain's season total, spread flat, is the demand-side cap:
        # selling faster than the town absorbs only walks the curve down.
        season_rate = season_demand.get(item, 0.0) / days_left
        st.day_target[item] = min(remaining / days_left, max(drain_rate, season_rate))

        # 5. who is the larger supplier? their exposure over ours flips the
        #    whole product from "pace it" to "dump it". This is the
        #    generalisation of the shipped melon rule (opp_ripening >= 3 ->
        #    floor 40) to every product, read off their actual board.
        theirs = opponent_exposure(view, item, cfg=cfg, st=st)
        st.dumping[item] = theirs > cfg.deny_ratio * max(1.0, remaining)

    # 6. tomorrow's prediction, so step 1 has something to measure against
    try:
        path = mm.inventory_path(view, 1)
        st.predicted_dawn_inv = {p: v[1] for p, v in path.items() if len(v) > 1}
    except Exception:
        st.predicted_dawn_inv = {}
    return st


# ------------------------------------------------------------------- caps
def participation_cap(view, item, st, cfg):
    """Units justified by the drain that ACCRUED since this item's last slice.

    Zero on the three steps of four where no tick has fired -- THIS is the
    mechanical difference from the refuted sell_pace, which re-offered the
    remainder on the very next step.
    """
    last = st.last_slice_step.get(item)
    if last is None:
        # First slice of the episode for this item: give it the drain that has
        # accrued since dawn, so a morning harvest is sellable the same day.
        last = view.day * TURNS_PER_DAY - 1
    accrued = mm.drain_between(item, last, view.step, view.unlocked_shops)
    return int(cfg.participation * accrued + st.carry.get(item, 0.0))


def impact_cap(view, item, cfg):
    """The shipped sell_pace arithmetic, kept as the SECOND cap rather than the
    only one: the units that keep the within-order walk under impact_cap_pct
    of the first unit's price."""
    inv = view.market_inv.get(item)
    if inv is None:
        return 0
    first = econ.price(item, inv)
    return econ.units_sellable_above(
        item, inv, first * (1.0 - cfg.impact_cap_pct / 100.0))


# ------------------------------------------------------------ relative terms
def opponent_exposure(view, item, horizon=None, cfg=None, st=None, memory=None):
    """Units of `item` the opponent still has to bring to market.

    features.supply_projection over their PUBLIC board, corrected by the dawn
    residual lam. Their shed is worth 1.5% of forecast error even as an oracle
    and is not readable anyway, so it is not in here.
    """
    cfg = cfg or SellConfig()
    if st is None and memory is not None:
        st = state_for(memory)
    lam = 1.0 if st is None else st.lam.get(item, 1.0)
    h = cfg.exposure_horizon if horizon is None else horizon
    return max(0.0, mm.opponent_supply(view, item, h) * lam)


def relative_marginal(view, item, n, st=None, cfg=None, memory=None):
    """THE win-rate term, in dollars of MARGIN rather than of bank.

    Our revenue for n units now, minus the price differential our n units
    impose on every unit the opponent still has to sell. The second term is
    `sum_k dPdI(inv + k)` over their projected remaining units, which
    telescopes exactly to `price(inv) - price(inv + M)`.

    Note the sign convention: the subtracted term is a PRICE differential, not
    their total loss, so this is the spec's quantity as written. Positive means
    selling n units now is worth more to us than the curve moves against us
    across their remaining block. The decision that actually flips a product
    from "pace it" to "dump it" is `dominance()` below, which is what
    `deny_ratio` reads.
    """
    inv = view.market_inv.get(item)
    if inv is None or n <= 0:
        return 0.0
    ours = float(econ.sell_revenue(item, inv, n))
    m = int(round(opponent_exposure(view, item, cfg=cfg, st=st, memory=memory)))
    if m <= 0:
        return ours
    spread = float(econ.price(item, inv) - econ.price(item, inv + m))
    return ours - spread


def dominance(view, item, st=None, cfg=None, memory=None):
    """opponent exposure / our own remaining supply.

    Above cfg.deny_ratio the opponent is about to flood this product: the
    price is going to fall whatever we do, so the first seller keeps the good
    half of the curve. That is the shipped melon rule generalised -- and it is
    the direction the whole repo record points ("the first-seller rule holds
    even in scarcity", premium_scale monotone toward selling sooner).
    """
    cfg = cfg or SellConfig()
    theirs = opponent_exposure(view, item, cfg=cfg, st=st, memory=memory)
    ours = max(1.0, mm.own_remaining(view, item))
    return theirs / ours


def risk_posture(view):
    """(our money + assets) - (theirs), scaled to [-1, +1].

    > 0 winning: take certain cash, shrink slices, never hold past liquidation.
    < 0 losing : hold the scarce-side products one extra tick -- the only
                 variance the sell module can add. Objective 3 says a smaller
                 CERTAIN loss is worth less than a shot at the win.

    Their shed is not observable, so their side is money + board value only;
    both sides use the same board valuation so the comparison is not biased by
    the missing term beyond the shed itself.
    """
    try:
        from rl import rollout
        mine = float(rollout.portfolio_value(view))
    except Exception:
        mine = float(view.me.money)
        for item, n in view.shed.items():
            inv = view.market_inv.get(item)
            if n > 0 and inv is not None and item not in ANIMALS:
                mine += econ.sell_revenue(item, inv, min(int(n), 40))
    theirs = float(view.opp.money) + _board_value(view)
    lead = mine - theirs
    return max(-1.0, min(1.0, lead / LEAD_SCALE))


def _board_value(view):
    """Public-information value of the opponent's standing board: what their
    tiles will deliver, priced at today's market."""
    horizon = max(0, 29 - view.day)
    if horizon <= 0:
        return 0.0
    path = mm.arrival_path(view.opp.tiles, view.day, horizon)
    total = 0.0
    for k in range(len(path)):
        for item, n in path[k].items():
            if n <= 0:
                continue
            total += n * view.market_prices.get(item, MARKET_PARAMS.get(item, {}).get("base", 20))
    return total


# ----------------------------------------------------------------- the plan
class Slice(tuple):
    """(item, units, reason); reason in
    {drain, impact, deny, liquidate, pressure, ratchet}."""
    __slots__ = ()

    def __new__(cls, item, units, reason):
        return super().__new__(cls, (item, int(units), reason))

    @property
    def item(self):
        return self[0]

    @property
    def units(self):
        return self[1]

    @property
    def reason(self):
        return self[2]


def _shipped_slices(view, plan, memory, reason):
    """The shipped sizer, verbatim, tagged with a bypass reason."""
    return [Slice(o[1], o[2], reason) for o in market._sell_orders(view, plan, memory)]


def _cut(slices, max_slices):
    """The room budget. `None` means "no cap" (a direct plan_slices call)."""
    return slices if max_slices is None else slices[:max(0, int(max_slices))]


def plan_slices(view, plan, memory, cfg=None, max_slices=None):
    """The sell schedule for this step.

    `max_slices` is the number of market-order slots the shipped builder
    actually left (build_orders computes it). It is applied HERE, before the
    commit loop at the bottom, and not by the caller afterwards: the commit
    loop writes `last_slice_step` / `sold_today` / `carry`, so a slice the
    caller deleted would otherwise be booked as sold. That silently (a) blocked
    the item for another whole drain tick (participation_cap reads
    `last_slice_step`), (b) credited `sold_today` with units never offered, so
    the dawn carry roll-forward saw the day's target met, and (c) wiped the
    carry the shortfall should have rolled forward. Measured on one real game
    on the `sell` preset: 38 slices, 362 units across six products.
    """
    cfg = cfg or SellConfig()
    st = _sync(view, memory)
    if view.hour == 0:
        dawn_update(view, memory, cfg)
        st = state_for(memory)

    final_push = view.step >= FINAL_STEP - FINAL_PUSH_STEPS
    pressure = view.shed_total > SHED_PRESSURE

    # --- whole-step bypasses: shipped behaviour, preserved -------------------
    # These commit no state, so the cut is pure truncation -- but it still has
    # to happen here, or build_orders would assemble more sells than slots.
    if not cfg.enabled or cfg.slots_only:
        return _cut(_shipped_slices(
            view, plan, memory, "liquidate" if final_push else "drain"), max_slices)
    if final_push:
        return _cut(_shipped_slices(view, plan, memory, "liquidate"), max_slices)
    if pressure:
        return _cut(_shipped_slices(view, plan, memory, "pressure"), max_slices)

    shipped = {o[1]: o[2] for o in market._sell_orders(view, plan, memory)}
    if not shipped:
        return []

    posture = risk_posture(view)
    liquidating = view.day >= min(cfg.liquidate_day, LIQUIDATION_DAY)
    out = []
    for item, shipped_units in shipped.items():
        if shipped_units <= 0:
            continue
        # MELON / FERTILIZER: the town never lifts their price back, so their
        # earliest sale is their best sale. Measured: we offer 2.81x the
        # season's entire melon demand -- a depth race, not a timing problem.
        if not mm.is_drained(item, view.unlocked_shops):
            out.append(Slice(item, shipped_units, "ratchet"))
            continue
        if cfg.paced_only and item not in PACED:
            out.append(Slice(item, shipped_units, "ratchet"))
            continue
        if liquidating:
            out.append(Slice(item, shipped_units, "liquidate"))
            continue
        # Crashed market: the drain cannot lift it back inside our horizon, so
        # holding has no option value (shipped release rule).
        if mm.crashed(item, view.market_prices.get(item, 0)):
            out.append(Slice(item, shipped_units, "impact"))
            continue
        # They are about to flood this product: first seller keeps the good
        # half of the curve.
        if st.dumping.get(item) or dominance(view, item, st=st, cfg=cfg) >= cfg.deny_ratio:
            out.append(Slice(item, shipped_units, "deny"))
            continue

        pcap = participation_cap(view, item, st, cfg)
        icap = impact_cap(view, item, cfg)
        if posture > 0:
            # Winning (objective 2): take certain cash in small, low-impact
            # slices instead of gambling on a better price later. The impact
            # budget tightens with the size of the lead.
            icap = econ.units_sellable_above(
                item, view.market_inv[item],
                econ.price(item, view.market_inv[item])
                * (1.0 - cfg.impact_cap_pct * (1.0 - 0.5 * posture) / 100.0))
        n = min(shipped_units, pcap, icap)
        if posture < 0 and item in PACED:
            # Losing (objective 3): the scarce-side products get one extra tick
            # of hold -- the only variance the sell module can add. A smaller
            # CERTAIN loss is worth less than a shot at the win.
            floor = plan.sell_floors.get(item, 1) * cfg.behind_hold
            inv = view.market_inv.get(item)
            if inv is not None:
                n = min(n, econ.units_sellable_above(item, inv, floor, cap=max(0, n)))
        if n <= 0:
            continue
        out.append(Slice(item, n, "impact" if n < pcap else "drain"))

    # Biggest revenue first, matching the shipped ordering, THEN the room
    # budget -- so the slices that lose their slot are the cheapest ones, and
    # nothing past the cut is booked below.
    out.sort(key=lambda s: -econ.sell_revenue(s.item, view.market_inv.get(s.item, 0), s.units))
    out = _cut(out, max_slices)
    for s in out:
        st.last_slice_step[s.item] = view.step
        st.sold_today[s.item] = st.sold_today.get(s.item, 0) + s.units
        st.carry[s.item] = max(0.0, st.carry.get(s.item, 0.0) - s.units)
    return out


# --------------------------------------------------------------- the orders
def build_orders(view, plan, memory, cfg=None):
    """Drop-in for kagent.market.build_orders.

    The shipped builder is called EXACTLY ONCE, so its mutations of
    plan.land_buys / plan.animal_buys / plan.plant_quota happen exactly as
    shipped. Its non-SELL orders are kept in their order; its SELL orders are
    discarded and replaced with our slices, placed in the first
    cfg.sell_slots_lead slots that are free after the HIRE block.

    NO non-SELL ORDER IS EVER DELETED. `base` already comes back capped at
    MAX_MARKET_ORDERS by the shipped `add()`, so every non-SELL order in it
    owns its slot; the sells get only `MAX_MARKET_ORDERS - len(hires) -
    len(rest)` slots and the lead block is a PERMUTATION of those, never an
    injection. Prepending the sells and truncating afterwards (the bug this
    guards) evicted a shipped buy on 1,580 of 8,640 real leader steps
    (18.3%) -- 1,050 of them the animal-feed BUY_PRODUCT WHEAT, one step
    closer to the two-consecutive-unfed-days escape, and 717 the BUY_SEED
    the allocator had just asked for. README "Market Action" says orders past
    the cap are silently dropped, so nothing reported it.

    HIRE is never displaced: hiring is nearly free and refusing it is refuted
    (m-hire5 / m-hire3 gated -$7,234 / -$39,678), so the HIRE block keeps the
    slots the shipped builder gave it.

    WHEAT and FERTILIZER are kept OUT of the leading slots on any step that
    also carries a feed buy: BUY_PRODUCT is quoted at inv-1, so selling wheat
    ahead of it would raise the price of our own feed.
    """
    cfg = cfg or SellConfig()
    melons_before = getattr(memory, "melons_sold", 0)
    # `_SHIPPED` and not `market.build_orders`, because `install()` REBINDS
    # that attribute to a wrapper around this function: reading it here made
    # the installed path recurse until the stack blew (confirmed 2026-09-26 --
    # `market.build_orders(None, None, None)` under install() raised
    # RecursionError; every unit test passed because each one either called
    # this function directly or stubbed the attribute out). `_SHIPPED` is
    # None until install() captures the real builder, so the direct-call path
    # is unchanged.
    base = (_SHIPPED or market.build_orders)(view, plan, memory)
    # The shipped builder counted ITS melon sells; we are replacing them.
    if hasattr(memory, "melons_sold"):
        memory.melons_sold = melons_before

    hires = [o for o in base if o and o[0] == "HIRE"]
    rest = [o for o in base if o and o[0] not in ("HIRE", "SELL")]
    feed_items = {o[1] for o in rest if o[0] == "BUY_PRODUCT" and len(o) > 1}

    # SIZE THE SELLS TO THE SLOTS THAT ARE ACTUALLY FREE. The shipped builder
    # allocated its 10 slots by priority and DEBITED the plan as it went
    # (`plan.land_buys = remaining_land`, `plan.animal_buys[kind] = want - n`),
    # and kagent.policy.HeuristicPolicy.act builds the DayPlan once at hour 0
    # and reuses it all day -- so a BUY_LAND / BUY_ANIMAL / BUY_SEED pushed
    # past MAX_MARKET_ORDERS here is silently dropped by the engine (README,
    # "Market Action") and never re-requested that day. Truncating the
    # ASSEMBLED list cut exactly those orders; truncating the slices keeps the
    # slot re-order (sells still quoted ahead of the buys) and evicts nothing.
    # The budget is handed to plan_slices rather than applied to its result,
    # because plan_slices BOOKS every slice it returns (last_slice_step /
    # sold_today / carry). Cutting afterwards left the schedule believing it
    # had sold units it never offered -- see the plan_slices docstring.
    room = MAX_MARKET_ORDERS - len(hires) - len(rest)
    slices = plan_slices(view, plan, memory, cfg, max_slices=max(0, room))
    lead_pool = [s for s in slices if s.item not in feed_items]
    demoted = [s for s in slices if s.item in feed_items]
    lead = lead_pool[:max(0, cfg.sell_slots_lead)]
    tail = lead_pool[max(0, cfg.sell_slots_lead):] + demoted

    orders = ([["SELL", s.item, s.units] for s in lead]
              + rest
              + [["SELL", s.item, s.units] for s in tail])
    orders = hires + orders
    orders = orders[:MAX_MARKET_ORDERS]

    if hasattr(memory, "melons_sold"):
        memory.melons_sold = melons_before + sum(
            int(o[2]) for o in orders if o[0] == "SELL" and o[1] == "MELON")
    return orders


# ------------------------------------------------------------- policy wiring
def as_plan_provider(inner, cfg=None):
    """Wrap a plan provider so an arm can run this module via rollout's
    dev-policy path. The wrapper is transparent to plan making; it only marks
    the memory so a patched kagent.market.build_orders routes here."""
    cfg = cfg or SellConfig()

    class _Provider:
        def __init__(self):
            self.inner = inner
            self.cfg = cfg

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def make_plan(self, view, memory):
            memory.rlv3_sell_cfg = self.cfg
            state_for(memory)
            return self.inner.make_plan(view, memory)

    return _Provider()


def install(cfg=None):
    """Monkey-patch kagent.market.build_orders for an eval arm. Returns a
    restore callable. Nothing in kagent/ is edited on disk.

    SCOPE, and it is a trap worth reading before launching anything
    (memory: "Land is bought through THREE paths" -- a behavioural switch that
    did not switch burned three hours of compute once already):

      * DEV PATH (rl.rollout's dev policy, `dev:champion`, tools/agent_scorecard
        with a non-bundle spec): kagent.policy does `from kagent import market`,
        so it holds the MODULE object and this patch IS reached. Verified by
        test_install_is_visible_on_the_dev_path.
      * A FLAT BUNDLE (arena/agents/*/main.py, i.e. `bundle:` specs and
        anything actually submitted) carries its OWN copy of build_orders in a
        namespace built from the bundled source. This patch does NOT reach it.
        Shipping rlv3 selling to a bundle needs rl/export.py to bundle this
        module too; that is out of scope here and is NOT done.
    """
    global _SHIPPED
    if _SHIPPED is not None:            # idempotent: never wrap our own wrapper
        original = _SHIPPED
    else:
        original = market.build_orders
    _SHIPPED = original

    def patched(view, plan, memory):
        # ARM ISOLATION. `cfg=None` means "route only memories an rlv3 plan
        # provider has MARKED"; an unmarked Memory -- a baseline arm sharing
        # this worker process, since nothing restores this patch between jobs
        # in rl.rollout.run_episode -- gets the shipped builder untouched.
        # With an explicit cfg the patch applies to everything, which is what
        # the unit tests and a single-arm trace want.
        use = getattr(memory, "rlv3_sell_cfg", cfg)
        if use is None:
            return original(view, plan, memory)
        return build_orders(view, plan, memory, use)

    # kagent.policy does `from kagent import market`, so it holds the MODULE
    # object: rebinding the attribute here is visible from there with no
    # further patching and nothing on disk is edited.
    market.build_orders = patched

    def restore():
        global _SHIPPED
        market.build_orders = original
        _SHIPPED = None

    return restore


# The only configuration the pre-gate supports (see the table above).
RECOMMENDED = SellConfig(slots_only=True)


__all__ = [
    "SellConfig", "SellState", "Slice", "PACED", "RATCHET", "RECOMMENDED",
    "state_for", "dawn_update", "participation_cap", "impact_cap",
    "opponent_exposure", "relative_marginal", "dominance", "risk_posture",
    "plan_slices", "build_orders", "as_plan_provider", "install",
]
