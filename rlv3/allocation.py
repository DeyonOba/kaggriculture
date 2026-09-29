"""Adaptive SHARE allocation of crops, livestock and land, with hiring read
from the schedule and never learned.

THE LIMITING DIMENSION IS ALLOCATION. The scorecard (995 band-stratified
paired games, `tools/agent_scorecard.py`) says champ-h10 runs 41.2 strawberry
tiles at day 20 where the three leaders run 22.5 / 23.5 / 12.75, and 1.0
carrot where they run 6.67 / 12.25 / 17.50. Carrot turns 1.0 units per
tile-day; strawberry turns 0.5. That single row is why we harvest 388 one-time
units against their 888-1,077 off a same-size farm, plant 150 times against
their ~270 and leave 12.4% of ground idle at dawn against their 4.7%. Ten of
eleven single-lever rules tested the same day LOST, because each moved one
piece inside that allocation and paid the cost without the benefit of the mix.

SO WHY WOULD THIS NOT BE THE LP AGAIN. It might be; the standing prior says it
is. `tools/make_lpalloc_variant.py` moved the whole tile allocation as one
object on 270 held-out paired ladder worlds and lost monotonically in dose:
a-lp33 -$9,937 (z -7.69, win 0.230), a-lp67 -$17,950 (z -14.05, win 0.196),
a-lp100 -$31,728 (z -22.47, win 0.159) against a base of 0.319. It was NOT an
execution failure -- both plans sat at ~73 of 75 tiles at day 13. "The plan IS
executable, the direction is wrong." And the constant version of THIS idea is
running right now: `rl/runs/draw995/run_alloc.sh`, arm mix_mild (strawberry
26, carrot 8), first result 0.339 against ship2_hv's 0.354. A constant mix is
already LOSING 1.5 points.

Three things are different here, and if none of them is the reason, the
component dies at lam = 0 cost:

  1. The direction is not a fixed target vector. The LP shipped one per-day
     schedule; mix_mild ships one constant. Here every target is re-derived at
     dawn from the board, the shop draw, the opponent's public farm and the
     projected inventory path, by an equal-marginal water-fill over the
     engine's exact price identity. Nothing is a schedule except hiring.
  2. The objective is RELATIVE. `rent.relative_rent` prices the revenue the
     opponent loses when our units move the shared pool, because a win is our
     money above theirs. The LP maximised our own margin, which is exactly
     what every losing arm here has done while handing them $2-7k.
  3. The dose is one parameter blended at the target SITE, so lam = 0 is
     bit-identical to champ-h10 and the ladder is readable. That is copied
     straight from the LP test, and it is the only reason its refutation could
     be believed.

WHAT THIS COMPONENT MAY NOT TOUCH. Hiring stays a per-day SCHEDULE
(`planner.CHAMP_HIRES`) with the existing workload and budget caps. There is
no hires head, no fitted hires model and no closed loop from idle labour to
hires: hiring as a constant gated -$4k to -$19k and closed-loop hiring -$11k,
while the per-day schedule is the one form that held. Hiring enters here only
as a READ -- it is the labour budget that bounds the herd and gates land --
and `hands_peak` (ours 11.0, the leaders' 11.75-12.25) is an INVARIANT the
test suite asserts. If hands_peak moves, something has coupled hiring to the
allocator and the run is void.
"""

import json
import os

from kagent.constants import (
    ANIMALS,
    LAND_ORDER,
    MARKET_PARAMS,
    SHOPS,
    TURNS_PER_DAY,
)
from kagent import econ, features, planner
from rlv3 import rent
from rlv3.land import LandDecider, observe_dawn

LAMBDA_DEFAULT = 0.0

# `__file__` is absent inside a bundle (the sources are exec'd from a string),
# so the path is resolved only in the dev tree; a bundle is handed its params
# inline as a dict and never reads the file.
PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "",
    "params", "allocation_v0.json")
_PARAMS_CACHE = {}

# Spelled out rather than aliased to rent.PLANTABLE / rent.ANIMAL_KINDS: in a
# flat bundle the module aliases are bound only after every source has
# executed, so a qualified reference at module level is a NameError at import
# time -- the same reason kagent/policy.py looks up set_mode through
# globals(). test_module_order_matches_rent keeps the two in step.
CROP_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
HERD_ORDER = ("GOOSE", "COW", "SHEEP")


def load_params(params=None):
    """A dict passes through; a path (or None -> allocation_v0.json) is read
    once and cached. The JSON is the checkpoint: tomorrow's training loop
    writes theta back to a file like this one and the run resumes from it."""
    if isinstance(params, dict):
        return params
    path = params or PARAMS_PATH
    if path not in _PARAMS_CACHE:
        with open(path) as fh:
            _PARAMS_CACHE[path] = json.load(fh)
    return _PARAMS_CACHE[path]


class Allocation:
    __slots__ = ("day", "stance", "crop_capacity", "herd_capacity", "labour",
                 "crop_shares", "herd_shares", "crop_targets", "herd_targets",
                 "plant_quota", "animal_buys", "structure_builds",
                 "hires_target", "land_buys", "land_margin", "trace")

    def __init__(self):
        self.day = -1
        self.stance = "LEVEL"
        self.crop_capacity = 0
        self.herd_capacity = 0
        self.labour = 0.0
        self.crop_shares = {}
        self.herd_shares = {}
        self.crop_targets = {}
        self.herd_targets = {}
        self.plant_quota = {}
        self.animal_buys = {}
        self.structure_builds = {"COOP": 0, "PASTURE": 0}
        self.hires_target = 0
        self.land_buys = []
        self.land_margin = 100
        self.trace = {}


# --- board reads (fresh every dawn; nothing is carried forward) --------------

def standing_crops(view):
    out = {}
    for _, _, t in view.me.plants():
        out[t["crop"]] = out.get(t["crop"], 0) + 1
    return out


def standing_herd(view):
    out = {}
    for _, _, t in view.me.animals():
        out[t["animal"]] = out.get(t["animal"], 0) + 1
    return out


def pocket_animals(view):
    """An animal in a hand's pocket is BOUGHT, ALIVE AND HOMELESS. Both the
    buy target and the structure pipeline used to look straight past it (the
    planner's `_pocket` fix); the scheduler counts it, so the two halves
    disagreed and we ordered a replacement. CARRY(n) memory."""
    out = {}
    for i, _x, _y in view.me.units:
        for item, n in view.unit_inv(i).items():
            if item in ANIMALS:
                out[item] = out.get(item, 0) + n
    return out


def empty_structures_by_kind(view):
    out = {"COOP": 0, "PASTURE": 0}
    for _, _, t in view.me.empty_structures():
        out[t["kind"]] = out.get(t["kind"], 0) + 1
    return out


# --- capacity ---------------------------------------------------------------

def capacity(view, memory, hires_target, *, base=None, params=None):
    """What is biddable today. `hires_target` is an INPUT -- the labour budget
    the schedule already fixed -- never an output of this function."""
    p = load_params(params)
    hp = p.get("herd", {}) or {}
    flow = features.tile_flow(view, horizon=1)
    reserved = sum((base.structure_builds if base else {}).values() or [0])
    crop_capacity = max(0, flow["free_now"] + flow["freeing"][1] - reserved)

    counts = standing_crops(view)
    herd = standing_herd(view)
    pocket = pocket_animals(view)
    herd_total = sum(herd.values()) + sum(view.shed.get(k, 0) for k in HERD_ORDER) \
        + sum(pocket.get(k, 0) for k in HERD_ORDER)

    # Feed pipeline, the same expression the champion plan and
    # features.apply_to_plan use: shed stock + growing wheat + market top-ups.
    feed_capacity = (view.shed.get("WHEAT", 0) + 3 * counts.get("WHEAT", 0)
                     + int(hp.get("feed_buffer", 20)))

    # Chore budget: the crew's turns, the share of them animals may have, over
    # the MEASURED chores per animal-day. `hires_target` is the schedule's.
    chore = mean_chore_cost(memory, params=p)
    turns = (int(hires_target) + 1) * TURNS_PER_DAY * float(hp.get("chore_share", 0.45))
    chore_budget = int(turns // max(0.25, chore))

    empties = empty_structures_by_kind(view)
    structures = sum(empties.values())
    buildable = max(0, crop_capacity - reserved)

    herd_capacity = max(0, min(feed_capacity, chore_budget,
                               herd_total + structures + buildable))
    return {
        "free_now": flow["free_now"],
        "freeing_today": flow["freeing"][1],
        "crop_capacity": crop_capacity,
        "standing": sum(counts.values()),
        "unlocked": 25 * max(1, len(view.me.unlocked_quadrants)),
        "structures_reserved": reserved,
        "feed_capacity": feed_capacity,
        "chore_budget": chore_budget,
        "herd_capacity": herd_capacity,
        "hands": max(0, len(view.me.units) - 1),
        "pocket": pocket,
    }


def mean_chore_cost(memory, params=None):
    """Unit-turns an animal costs per day, MEASURED where we have measured it.

    Defaults are OUR executor's rate (pickups per feed 1.08 against the
    leaders' 0.46-0.55). `memory.rlv3["chore"]` overrides per kind once the
    Maia2 executor component reports a realised cost -- which is the only way
    geese ever turn arithmetic-positive here."""
    p = load_params(params)
    hp = p.get("herd", {}) or {}
    table = dict(hp.get("chore_cost", {}) or {})
    table.update(((getattr(memory, "rlv3", None) or {}).get("chore") or {}))
    vals = [float(v) for v in table.values()] or [3.0]
    return (sum(vals) / len(vals)) * float(hp.get("chore_scale", 1.0))


def chore_cost(kind, memory, params=None):
    p = load_params(params)
    hp = p.get("herd", {}) or {}
    table = dict(hp.get("chore_cost", {}) or {})
    table.update(((getattr(memory, "rlv3", None) or {}).get("chore") or {}))
    return float(table.get(kind, 3.0)) * float(hp.get("chore_scale", 1.0))


# --- stance (the lexicographic objective) -----------------------------------

def projected_final(view, farm, path, horizon):
    """Bank plus the board's projected revenue at the inventory path."""
    try:
        sup = planner._rent_board_supply(farm.tiles, view.day, horizon)
    except Exception:
        return float(farm.money)
    total = 0.0
    for k in range(len(sup)):
        for item, n in sup[k].items():
            if n <= 0 or item not in MARKET_PARAMS:
                continue
            total += rent.sell_revenue(item, rent.inv_at(path, item, k), int(round(n)))
    return float(farm.money) + total


def stance(view, memory, *, params=None, path=None):
    """-> ("AHEAD" | "LEVEL" | "BEHIND", lead, conf).

    Drives the lexicographic objective as a SELECTION RULE over near-ties --
    it changes the opponent weight and the demand risk quantile, never a
    weighted sum of money and wins.

      AHEAD  : w_opp low, quantile 0.35. Protect the lead, take the cheap
               profit, prefer the uncrashable annuities (WHEAT log/0.20, EGG
               log/0.20) and short cycles.
      BEHIND : w_opp high, quantile 0.50. Crowd the pool THEY are heaviest in,
               because denying is the only lever on a margin we cannot
               out-earn. Accept variance.
      Loss-minimising falls out of LEVEL: with a small negative projected
      margin the split that keeps the gap closing beats the highest-mean one,
      because every candidate is already scored on the relative number.
    """
    p = load_params(params)
    sp = p.get("stance", {}) or {}
    horizon = int(sp.get("horizon", 14))
    path = path if path is not None else rent.inventory_path(view, horizon=horizon)
    mine = projected_final(view, view.me, path, horizon)
    theirs = projected_final(view, view.opp, path, horizon)
    lead = mine - theirs
    band = float(sp.get("level_band", 6000.0))
    label = "LEVEL" if abs(lead) <= band else ("AHEAD" if lead > 0 else "BEHIND")
    conf = min(1.0, abs(lead) / float(sp.get("conf_scale", 20000.0)))
    return label, lead, conf


def risk_quantile(label, params=None):
    """0.35 (the v3 ANCHOR_QUANTILE) when LEVEL or AHEAD -- plan for demand
    that probably arrives; 0.50 when BEHIND -- a behind agent needs the
    upside. An exact convolution over the remaining uniform unlock draws, so
    it converges on the mean as the unlocks happen and the plan tightens on
    its own as the season resolves."""
    p = load_params(params)
    q = (p.get("stance", {}) or {}).get("quantile", {}) or {}
    return float(q.get(label, features.ANCHOR_QUANTILE))


# --- the demand caps (our own quantile, features.anchors' arithmetic) -------

def demand_caps(view, label, *, params=None):
    """Per-crop and per-animal caps from demand MINUS the opponent's projected
    supply. Their board is public and changes daily, so the cap does too, and
    97% of the $52,652 their supply costs us is strawberry + milk + wool --
    the three lines we are heaviest in. Never plant into a pool they saturate.
    """
    p = load_params(params)
    day = view.day
    days_left = max(1, rent.SEASON_END - day)
    q = risk_quantile(label, p)
    fc = econ.forecast_drain(view.unlocked_shops, day, quantile=q)
    opp = rent.opp_supply(view)
    crop_caps = dict((p.get("crop", {}) or {}).get("caps", {}) or {})
    herd_caps = dict((p.get("herd", {}) or {}).get("caps", {}) or {})
    crash = float((p.get("rent", {}) or {}).get("crash_ratio", 0.35))

    def net_per_day(item):
        return max(0.0, fc.get(item, 0.0) - opp.get(item, 0.0)) / days_left

    def crashed(item):
        base = MARKET_PARAMS[item]["base"]
        return view.market_prices.get(item, base) < crash * base

    crops, herd, dropped = {}, {}, []
    for crop in CROP_ORDER:
        hard = int(crop_caps.get(crop, 99))
        if rent.crop_block(crop, day) is None or day > features.LAST_PLANT.get(crop, 99):
            crops[crop] = 0
            dropped.append(crop)
            continue
        if crashed(crop):
            crops[crop] = 0
            dropped.append(crop)
            continue
        crops[crop] = max(0, min(hard, int(round(net_per_day(crop) / rent.per_tile_day(crop)))))
    for kind in HERD_ORDER:
        a = ANIMALS[kind]
        hard = int(herd_caps.get(kind, 99))
        if (rent.animal_block(kind, day) is None
                or day > features.ANIMAL_LAST_BUY.get(kind, 99)):
            herd[kind] = 0
            dropped.append(kind)
            continue
        if crashed(a["product"]):
            herd[kind] = 0
            dropped.append(kind)
            continue
        rate = features._OUT_PER_DAY.get(a["product"], 1.0)
        herd[kind] = max(0, min(hard, int(round(net_per_day(a["product"]) / rate))))
    return crops, herd, dropped


# --- the whole dawn decision ------------------------------------------------

def allocate(view, memory, *, lam=None, params=None, knobs=None, base=None):
    """Pure. No engine calls, no randomness, no I/O beyond the params cache."""
    p = load_params(params)
    lam = float(p.get("lam", LAMBDA_DEFAULT) if lam is None else lam)
    if knobs:
        merged = dict(getattr(memory, "knobs", None) or {})
        merged.update(knobs)
        memory.knobs = merged
    base = base if base is not None else planner.make_day_plan(view, memory, speculative=True)
    observe_dawn(memory, view)

    day = view.day
    alloc = Allocation()
    alloc.day = day

    horizon = int((p.get("rent", {}) or {}).get("horizon", 14))
    path = rent.inventory_path(view, horizon=horizon)
    label, lead, conf = stance(view, memory, params=p, path=path)
    alloc.stance = label

    counts = standing_crops(view)
    herd = standing_herd(view)
    pocket = pocket_animals(view)
    shed = {k: view.shed.get(k, 0) for k in HERD_ORDER}

    # HIRING IS A READ. The schedule already decided it, workload- and
    # budget-capped, in make_champion_plan. Nothing below may change it.
    alloc.hires_target = base.hires_target

    cap = capacity(view, memory, alloc.hires_target, base=base, params=p)
    alloc.crop_capacity = cap["crop_capacity"]
    alloc.herd_capacity = cap["herd_capacity"]

    crop_caps, herd_caps, dropped = demand_caps(view, label, params=p)

    # --- crops: water-fill the standing board on marginal relative rent ----
    crop_cands = [c for c in CROP_ORDER if crop_caps.get(c, 0) > 0]
    # Only the CANDIDATES' standing tiles, never every crop's. The fill below
    # distributes this budget among `crop_cands` alone, so ground held by a
    # non-candidate -- MELON always, STRAWBERRY past LAST_PLANT 13, TOMATO past
    # 18, anything crashed -- would be handed to the candidates as free ground
    # it can never occupy ("Each plant or animal occupies one square"). That
    # phantom is 5-32 tiles on 14 of the 18 recorded leader boards; the share
    # it buys is unachievable, `_plant_flow` truncates it against the real
    # `crop_capacity`, and the truncation lands on whichever crop the fill
    # ranked last -- at dsm-d20-s0 it inflated WHEAT 34 -> 44 and spent 16 of
    # 19 plantable tiles on wheat where the honest budget plants 13 carrot.
    board_budget = sum(counts.get(c, 0) for c in crop_cands) + cap["crop_capacity"]
    caps_c = {c: min(crop_caps[c], board_budget) for c in crop_cands}

    def crop_score(c, n_already, step):
        return rent.marginal_rent(view, c, n_already, path, step=step,
                                  stance=label, day=day, params=p)

    step = int((p.get("rent", {}) or {}).get("step", 1))
    pass2_from = None
    share_crop, trace_c = rent.water_fill(board_budget, crop_cands, crop_score,
                                          caps=caps_c, step=step)
    # SECOND PASS: bare ground earns exactly zero. The demand anchor is a
    # "do not flood the pool they already saturate" signal, not a reason to
    # leave a tile empty, so once it binds the fill continues to the HARD
    # ceiling while the marginal is still positive. We leave 12.4% of ground
    # idle at dawn against the leaders' 4.7%, and this is the capacity-driven
    # way to close it -- no `fill_idle` knob, no labour-proxy multiplier.
    # Switchable: params["fill_bare"] = false restores the anchor as a hard cap.
    if p.get("fill_bare", True) and sum(share_crop.values()) < board_budget:
        # ... but only for a product some shop can EVER buy. MELON is in no
        # SHOPS entry at all (an engine fact, not a tuned exception), so its
        # only demand is the town centre's 1 unit a day and no future unlock
        # can lift it: the anchor IS its ceiling. We already offer 2.8x the
        # melon demand while satisfying 0.344 of the pool overall against the
        # recorded opponent's 0.700, and `planner._rent_pick` asks for 39
        # melon tiles (234 units) at states like this against the repo's own
        # MELON_SUPPLY_TARGET of 140. Switchable: fill_bare_shop_only = false
        # lets bare ground be filled with anything up to the hard cap.
        shop_only = p.get("fill_bare_shop_only", True)
        ceiling = (p.get("crop", {}) or {}).get("caps", {})
        hard = {}
        for c in crop_cands:
            top = int(ceiling.get(c, 99))
            if shop_only and not any(c in items for items in SHOPS.values()):
                top = min(top, int(caps_c.get(c, top)))
            hard[c] = min(top, board_budget)
        share_crop, trace2 = rent.water_fill(board_budget, crop_cands, crop_score,
                                             floors=share_crop, caps=hard, step=step)
        pass2_from = len(trace_c)
        trace_c = trace_c + [t for t in trace2 if t[2] != float("inf")]
    alloc.labour = _shadow_price(trace_c, p)

    # --- herd: the same split over a labour-and-feed-bounded capacity ------
    herd_cands = [k for k in HERD_ORDER if herd_caps.get(k, 0) > 0
                  and k not in (base.cull_kinds or set())]
    caps_h = {k: min(herd_caps[k], cap["herd_capacity"]) for k in herd_cands}

    def herd_score(k, n_already, step):
        return rent.animal_rent(view, k, n_already, path, chore_cost(k, memory, p),
                                step=step, stance=label, day=day, params=p,
                                labour_price=alloc.labour)

    share_herd, trace_h = rent.water_fill(cap["herd_capacity"], herd_cands, herd_score,
                                          caps=caps_h)

    alloc.crop_shares = _normalise(share_crop)
    alloc.herd_shares = _normalise(share_herd)

    # --- the dose, blended AT THE TARGET SITE ------------------------------
    # base_expr is read straight back OUT of the champion plan (standing count
    # plus the quota it asked for), so lam = 0 reproduces it bit for bit with
    # no re-derivation of the champion arithmetic to drift against.
    base_crop = {c: counts.get(c, 0) + base.plant_quota.get(c, 0) for c in CROP_ORDER}
    base_herd = {k: herd.get(k, 0) + shed.get(k, 0) + pocket.get(k, 0)
                 + base.animal_buys.get(k, 0) for k in HERD_ORDER}
    held_herd = {k: herd.get(k, 0) + shed.get(k, 0) + pocket.get(k, 0) for k in HERD_ORDER}

    deviate = (lam != 0.0
               and day >= int(p.get("first_deviation_day", 2))
               and not base.liquidation)

    if deviate:
        alloc.crop_targets = {c: max(0, int(round(base_crop[c] + lam * (share_crop.get(c, 0) - base_crop[c]))))
                              for c in CROP_ORDER}
        alloc.herd_targets = {k: max(0, int(round(base_herd[k] + lam * (share_herd.get(k, 0) - base_herd[k]))))
                              for k in HERD_ORDER}
    else:
        alloc.crop_targets = dict(base_crop)
        alloc.herd_targets = dict(base_herd)

    # --- today's FLOW out of today's capacity only -------------------------
    if not deviate:
        alloc.plant_quota = dict(base.plant_quota)
        alloc.animal_buys = dict(base.animal_buys)
        alloc.structure_builds = dict(base.structure_builds)
        alloc.land_buys = list(base.land_buys)
        alloc.land_margin = base.land_margin
        land_why = {"gate": "lam0_base"}
    else:
        alloc.plant_quota = _plant_flow(alloc.crop_targets, counts, base,
                                        cap["crop_capacity"], trace_c)
        alloc.animal_buys = _buy_flow(alloc.herd_targets, held_herd, base, day)
        alloc.structure_builds = structure_builds(view, alloc.animal_buys, pocket)
        alloc.trace = {"lead": lead, "liquidation": base.liquidation}
        buy, margin, land_why = LandDecider(p.get("land", {})).decide(view, memory, alloc)
        alloc.land_buys = []
        alloc.land_margin = base.land_margin
        if buy:
            idx = sum(1 for q in LAND_ORDER if q in view.me.unlocked_quadrants)
            alloc.land_buys = [LAND_ORDER[idx]]
            # Keep the champion's own margin when it wanted the same quadrant,
            # so the dose does not silently move a proven number.
            if base.land_buys == alloc.land_buys:
                alloc.land_margin = base.land_margin
            elif margin is not None:
                alloc.land_margin = int(margin)

    alloc.trace = {
        "hour": view.hour,
        "lam": lam,
        "stance": label,
        "lead": round(lead, 1),
        "conf": round(conf, 3),
        "quantile": risk_quantile(label, p),
        "liquidation": base.liquidation,
        "deviate": deviate,
        "capacity": cap,
        "labour_price": round(alloc.labour, 3),
        "crop_caps": crop_caps,
        "herd_caps": herd_caps,
        "dropped": dropped,
        "base_crop": base_crop,
        "base_herd": base_herd,
        "share_crop": dict(share_crop),
        "share_herd": dict(share_herd),
        "fill_crop": [(c, n, (None if m == float("inf") else round(m, 2)))
                      for c, n, m in trace_c],
        # Index in fill_crop where the bare-ground pass takes over. Each half
        # is monotone in the marginal; the join is where the demand anchor
        # stopped binding and the hard ceiling took over.
        "fill_pass2_from": pass2_from,
        "fill_herd": [(k, n, (None if m == float("inf") else round(m, 2)))
                      for k, n, m in trace_h],
        "land": land_why,
    }
    return alloc


def _normalise(alloc_map):
    total = float(sum(v for v in alloc_map.values() if v > 0))
    if total <= 0:
        return {k: 0.0 for k in alloc_map}
    return {k: (v / total if v > 0 else 0.0) for k, v in alloc_map.items()}


def _shadow_price(trace, params):
    """The marginal of the LAST accepted step: the equal-marginal price of one
    tile-day, which is what one unit-turn of labour earns on crops."""
    tpd = float((params.get("rent", {}) or {}).get("turns_per_tile_day", 1.0)) or 1.0
    vals = [t[2] for t in trace if t[2] != float("inf")]
    return (vals[-1] / tpd) if vals else 0.0


def _plant_flow(targets, counts, base, crop_capacity, trace):
    """Deficits, ranked by the water-fill's own order, clipped to what can
    actually go in the ground today.

    The clip floor is the champion's OWN ask, so the dose can only ever be
    clipped on the deviation -- lam = 0 is never touched by a capacity guard
    whose tile count differs from make_champion_plan's `free`.
    """
    order = []
    for c, _n, _m in trace:
        if c not in order:
            order.append(c)
    for c in CROP_ORDER:
        if c not in order:
            order.append(c)
    budget = max(int(crop_capacity), sum(base.plant_quota.values()))
    quota = {}
    for c in order:
        if budget <= 0:
            break
        want = max(0, int(targets.get(c, 0)) - int(counts.get(c, 0)))
        n = min(want, budget)
        if n > 0:
            quota[c] = n
            budget -= n
    return quota


def _buy_flow(targets, held, base, day):
    """Per-kind deficits under the champion's own per-day cap of 4 (never
    below what the base already asked for, so lam = 0 is untouched)."""
    buys = {}
    for kind in HERD_ORDER:
        if day > features.ANIMAL_LAST_BUY.get(kind, 99):
            continue
        want = int(targets.get(kind, 0)) - int(held.get(kind, 0))
        n = max(0, min(want, max(4, base.animal_buys.get(kind, 0))))
        if n > 0:
            buys[kind] = n
    return buys


def structure_builds(view, animal_buys, pocket):
    """The champion plan's structure pipeline, expression for expression:
    empty structures are consumed in GOOSE/COOP, COW/PASTURE, SHEEP/PASTURE
    order and only the shortfall is built. Pocket animals are counted."""
    empty_by_kind = empty_structures_by_kind(view)
    builds = {"COOP": 0, "PASTURE": 0}
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = (animal_buys.get(kind, 0) + view.shed.get(kind, 0)
                    + pocket.get(kind, 0))
        need = pipeline - empty_by_kind.get(structure, 0)
        if need > 0:
            builds[structure] += need
            empty_by_kind[structure] = 0
        else:
            empty_by_kind[structure] = -need
    return builds


# --- the plan overwrite -----------------------------------------------------

def apply_to_plan(plan, alloc, view):
    """Overwrite ONLY the six allocation fields.

    Everything else -- sell_floors, sell_pace, fert_gate, wheat_reserve,
    feed_days, cull_kinds, liquidation -- is left exactly as
    `planner.make_champion_plan` produced it. sell_floors and sell_pace belong
    to the rlv3 sell-timer component; the rest are proven and out of scope.
    Mirrors `features.apply_to_plan`'s contract, so every downstream safety
    clamp in the scheduler and the market controller is unchanged.

    `planner.apply_land_pressure` -- the third land purchase path, which bought
    the fourth quadrant on day 14 in the first capped traces while the head
    read 0 -- is NOT called here and is not reachable on the champion path.
    """
    plan.plant_quota = dict(alloc.plant_quota)
    plan.animal_buys = dict(alloc.animal_buys)
    plan.structure_builds = dict(alloc.structure_builds)
    plan.hires_target = alloc.hires_target
    plan.land_buys = list(alloc.land_buys)
    plan.land_margin = alloc.land_margin
    return plan


class AllocationPlanProvider:
    """Duck-types `kagent.policy.ChampionPlanProvider`, so
    `HeuristicPolicy(AllocationPlanProvider(...))` is the whole wiring and
    neither `rl.rollout` nor `kagent.policy` needs an edit.

    `knobs` accepts either a plain knob-override dict or a BANDIT_RULES dict
    (`{"kind": "knobs", ...}`) -- champ-h10's four rules, so the lam = 0 base
    is h10 and not h8. Runs at hour 0 only, because that is the only hour a
    day plan is ever built (`kagent.policy.HeuristicPolicy.act`), which is
    also why every pocket-state planner bug is otherwise inert.
    """

    def __init__(self, knobs=None, params=None, lam=None):
        self.params = load_params(params)
        self.lam = float(self.params.get("lam", LAMBDA_DEFAULT) if lam is None else lam)
        self._bandit = None
        self.knobs = None
        if isinstance(knobs, dict) and knobs.get("kind") in ("knobs", "bandit"):
            from kagent.banditpolicy import BanditPolicy, KnobBanditPolicy
            cls = KnobBanditPolicy if knobs.get("kind") == "knobs" else BanditPolicy
            self._bandit = cls(knobs)
        elif knobs:
            self.knobs = dict(knobs)
        self.last_alloc = None

    def make_plan(self, view, memory):
        if self._bandit is not None:
            plan = self._bandit.make_plan(view, memory)
        else:
            if self.knobs:
                merged = dict(getattr(memory, "knobs", None) or {})
                merged.update(self.knobs)
                memory.knobs = merged
            plan = planner.make_day_plan(view, memory, speculative=True)
        alloc = allocate(view, memory, lam=self.lam, params=self.params, base=plan)
        self.last_alloc = alloc
        return apply_to_plan(plan, alloc, view)
