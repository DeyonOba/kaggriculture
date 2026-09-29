"""Unit tests for rlv3's adaptive allocation.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_allocation.py -q

NO GAMES ARE PLAYED HERE. Every test builds a synthetic board from one
`env.reset(2)` observation and calls the pure planning functions, because the
thing that has to be right before a single game is worth spending is the
IDENTITY: lam = 0 must reproduce champ-h10 exactly. The LP allocation test
(`tools/make_lpalloc_variant.py`) verified that for all three of its families
before any other dose was trusted, and that is the only reason its refutation
-- a-lp100 at -$31,728 a game, win rate halved -- could be believed.

The tests are grouped by what they defend:
  * identity      lam = 0 is bit-identical, on a 200-state sweep
  * invariants    hiring is a READ; hands_peak cannot move
  * arithmetic    concavity, the whole-stream rule, the opponent term
  * one-way land  a bad theta buys LESS land, never more
  * mechanism     at lam > 0 the board is a SPLIT, not a corner solution
"""
import copy
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from kagent import econ, features, market, planner  # noqa: E402
from kagent.constants import (  # noqa: E402
    ANIMALS,
    LAND_ORDER,
    LAND_PRICES,
    MAX_MARKET_ORDERS,
    SHOPS,
)
from kagent.gamestate import Memory, View  # noqa: E402
from rlv3 import allocation, land, rent  # noqa: E402
from rlv3.tests.conftest import mid_season_cases  # noqa: E402

# champ-h10's four knob rules, resolved (arena/agents/champ-h10/main.py
# BANDIT_RULES). The lam = 0 base is h10, not h8.
H10_KNOBS = {"premium_scale": 0.35, "carrot_day": 25, "cows": 6, "sheep": 10}
H10_RULES = {
    "kind": "knobs", "min_gain": 1500, "max_knobs": 5, "models": [],
    "conditions": [
        {"knob": "premium_scale", "value": 0.35},
        {"knob": "carrot_day", "value": 25},
        {"knob": "cows", "value": 6, "max_consumers": {"MILK": 0}},
        {"knob": "sheep", "value": 10, "day": 12, "min_price": {"WOOL": 1.0}},
    ],
}

QUADRANTS = {
    "NW": [(x, y) for y in range(5) for x in range(5)],
    "NE": [(x, y) for y in range(5) for x in range(5, 10)],
    "SW": [(x, y) for y in range(5, 10) for x in range(5)],
    "SE": [(x, y) for y in range(5, 10) for x in range(5, 10)],
}

_BASE_OBS = None


def base_obs():
    global _BASE_OBS
    if _BASE_OBS is None:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"seed": 7}, debug=False)
        env.reset(2)
        _BASE_OBS = env.state[0].observation
    return copy.deepcopy(_BASE_OBS)


def plant(crop, planted_day, yield_units=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": yield_units, "max_lifespan_step": -1,
            "fertilized_until_day": -1}


def beast(kind, placed_day, yield_units=0):
    return {"kind": ANIMALS[kind]["structure"], "animal": kind,
            "placed_day": placed_day, "fed_today": True, "cared_today": True,
            "fertilizer_available": False, "yield_units": yield_units,
            "consecutive_unfed": 0, "pending_care_bonus": 0}


def empty_structure(kind):
    return {"kind": kind, "animal": None, "fed_today": False,
            "cared_today": False, "fertilizer_available": False,
            "yield_units": 0, "consecutive_unfed": 0}


def _fill(farm, quads, crops, animals, empties=(), age=3):
    """Lay `crops` (crop -> n) and `animals` (kind -> n) onto the unlocked
    quadrants in order; everything left over stays bare.

    Locked quadrants keep the engine's own marker, the STRING 'LOCKED' --
    not None. `FarmView.empty_tiles` tests `t is None`, so a board built with
    None in the locked quadrants reports 75 free tiles on a one-quadrant farm
    and every capacity number downstream is fiction."""
    slots = [xy for q in quads for xy in QUADRANTS[q]]
    i = 0
    for kind, n in animals.items():
        for _ in range(n):
            if i >= len(slots):
                return
            x, y = slots[i]
            farm["tiles"][y][x] = beast(kind, max(0, farm.get("_day", 0) - age))
            i += 1
    for kind in empties:
        if i >= len(slots):
            return
        x, y = slots[i]
        farm["tiles"][y][x] = empty_structure(kind)
        i += 1
    for crop, n in crops.items():
        for _ in range(n):
            if i >= len(slots):
                return
            x, y = slots[i]
            farm["tiles"][y][x] = plant(crop, max(0, farm.get("_day", 0) - age))
            i += 1
    while i < len(slots):
        x, y = slots[i]
        farm["tiles"][y][x] = None
        i += 1


def make_view(day=10, money=8000, quads=("NW", "NE"), crops=None, animals=None,
              empties=(), opp_quads=("NW", "NE"), opp_crops=None, opp_animals=None,
              opp_money=8000, shops=("BAKERY", "YARN_STORE"), shed=None,
              hands=8, market_inv=None, pocket=None, plant_age=3):
    obs = base_obs()
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    me, opp = obs["farms"][0], obs["farms"][1]
    me["money"], opp["money"] = money, opp_money
    me["unlocked_quadrants"] = list(quads)
    opp["unlocked_quadrants"] = list(opp_quads)
    me["_day"] = opp["_day"] = day
    me["tiles"] = [["LOCKED"] * 10 for _ in range(10)]
    opp["tiles"] = [["LOCKED"] * 10 for _ in range(10)]
    _fill(me, quads, crops or {}, animals or {}, empties, age=plant_age)
    _fill(opp, opp_quads, opp_crops or {}, opp_animals or {}, age=plant_age)
    me["hands"] = [[4, 4]] * hands
    me["hires_today"] = 0
    obs["town"] = {"unlocked_shops": list(shops)}
    inv = dict(obs.get("market", {}).get("inventory", {}))
    if market_inv:
        inv.update(market_inv)
    prices = {p: econ.price(p, int(v)) for p, v in inv.items()}
    obs["market"] = {"inventory": inv, "prices": prices}
    private = obs.setdefault("private", {})
    private["shed"] = dict(shed or {})
    invs = [{} for _ in range(hands + 1)]
    for i, (item, n) in enumerate((pocket or {}).items()):
        invs[min(i + 1, hands)][item] = n
    private["inventories"] = invs
    return View(obs)


def memory_for(view, knobs=None, idle=0.2, yhands=11):
    mem = Memory()
    mem.observe(view)
    mem.knobs = dict(knobs or H10_KNOBS)
    mem.yesterday_idle = idle
    mem.yesterday_hands = yhands
    return mem


# --- the state sweep every identity test runs over --------------------------

def sweep(n=120, seed=0):
    rng = random.Random(seed)
    days = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
            24, 25, 26, 27, 28]
    shop_sets = [(), ("BAKERY",), ("BAKERY", "YARN_STORE"),
                 ("PIZZA_SHOP", "ICE_CREAM_SHOP", "PET_CAFE"),
                 tuple(sorted(SHOPS))[:5]]
    out = []
    for _ in range(n):
        day = rng.choice(days)
        nq = rng.choice([1, 2, 2, 3, 3])
        quads = ["NW"] + LAND_ORDER[:nq - 1]
        room = 25 * nq
        crops = {}
        for c in ("WHEAT", "STRAWBERRY", "MELON", "CARROT", "TOMATO"):
            if rng.random() < 0.7:
                crops[c] = rng.randint(0, max(1, room // 4))
        animals = {k: rng.randint(0, 6) for k in ("COW", "SHEEP", "GOOSE")
                   if rng.random() < 0.7}
        total = sum(crops.values()) + sum(animals.values())
        while total > room - 2:                      # keep some bare ground
            k = rng.choice([c for c in crops if crops[c] > 0] or list(animals))
            if k in crops:
                crops[k] -= 1
            else:
                animals[k] -= 1
            total -= 1
        out.append(dict(
            day=day, money=rng.choice([200, 800, 3000, 9000, 25000, 60000]),
            quads=tuple(quads), crops=crops, animals=animals,
            empties=tuple(rng.choice([(), ("COOP",), ("PASTURE", "PASTURE")])),
            opp_quads=tuple(["NW"] + LAND_ORDER[:rng.choice([0, 1, 2])]),
            opp_crops={"STRAWBERRY": rng.randint(0, 20), "WHEAT": rng.randint(0, 20)},
            opp_animals={"COW": rng.randint(0, 8), "SHEEP": rng.randint(0, 8)},
            opp_money=rng.choice([500, 5000, 30000, 90000]),
            shops=rng.choice(shop_sets),
            shed=rng.choice([{}, {"WHEAT": 30}, {"COW": 1, "WHEAT": 10}]),
            hands=rng.choice([3, 6, 9, 11, 14]),
            pocket=rng.choice([None, {"SHEEP": 1}, {"GOOSE": 2}]),
        ))
    return out


PLAN_FIELDS = ("plant_quota", "animal_buys", "structure_builds",
               "hires_target", "land_buys", "land_margin")
UNTOUCHED = ("sell_floors", "sell_pace", "fert_gate", "wheat_reserve",
             "feed_days", "cull_kinds", "liquidation")


def _pair(state, lam, params=None):
    """(reference champion plan, allocation-overwritten plan, alloc)."""
    view = make_view(**state)
    mem = memory_for(view)
    ref = planner.make_champion_plan(view, mem)
    mem2 = memory_for(view)
    got = planner.make_champion_plan(view, mem2)
    alloc = allocation.allocate(view, mem2, lam=lam, params=params, base=got)
    allocation.apply_to_plan(got, alloc, view)
    return ref, got, alloc, view


# ============================================================================
# IDENTITY -- the only thing that makes the dose ladder readable
# ============================================================================

def test_lam0_is_bit_identical_to_the_champion_plan():
    """lam = 0 leaves every one of the six allocation fields exactly as
    `planner.make_champion_plan` produced it, on a 120-state sweep."""
    bad = []
    for state in sweep():
        ref, got, _a, _v = _pair(state, 0.0)
        for f in PLAN_FIELDS:
            if getattr(ref, f) != getattr(got, f):
                bad.append((state["day"], f, getattr(ref, f), getattr(got, f)))
    assert not bad, f"{len(bad)} deviations at lam=0, first: {bad[0]}"


def test_lam0_leaves_the_other_plan_fields_alone():
    """sell_floors, sell_pace, fert_gate, wheat_reserve, feed_days, cull_kinds
    and liquidation belong to other components or are proven; apply_to_plan
    may not touch them at ANY dose."""
    for lam in (0.0, 0.33, 1.0):
        for state in sweep(n=30, seed=3):
            ref, got, _a, _v = _pair(state, lam)
            for f in UNTOUCHED:
                assert getattr(ref, f) == getattr(got, f), (lam, state["day"], f)


def test_allocate_is_pure_and_repeatable():
    """Same state, same answer -- no randomness, no carried-forward target."""
    for state in sweep(n=20, seed=11):
        for lam in (0.0, 0.67):
            _r1, g1, a1, _v = _pair(state, lam)
            _r2, g2, a2, _v = _pair(state, lam)
            assert [getattr(g1, f) for f in PLAN_FIELDS] == [getattr(g2, f) for f in PLAN_FIELDS]
            assert a1.crop_targets == a2.crop_targets
            assert a1.herd_targets == a2.herd_targets


# ============================================================================
# INVARIANTS -- hiring is a READ, and hands_peak may not move
# ============================================================================

def test_hiring_is_never_written_by_the_allocator():
    """The relayed instruction is explicit: hiring is NOT learnt. It stays
    `planner.CHAMP_HIRES[day]` under the workload and budget caps, at every
    dose. hands_peak (ours 11.0, leaders 11.75-12.25) is an INVARIANT: if it
    moves, hiring has been coupled to the allocator and the run is void."""
    for lam in (0.0, 0.33, 0.67, 1.0):
        for state in sweep(n=40, seed=5):
            ref, got, alloc, _v = _pair(state, lam)
            assert got.hires_target == ref.hires_target, (lam, state["day"])
            assert alloc.hires_target == ref.hires_target


def test_hires_target_still_tracks_the_schedule():
    """Sanity on the read itself: the target is the schedule, capped by
    workload and by 25% of cash -- never anything the allocator computed."""
    view = make_view(day=12, money=40000, quads=("NW", "NE", "SW"),
                     crops={"WHEAT": 20, "STRAWBERRY": 20}, animals={"COW": 6})
    mem = memory_for(view)
    plan = planner.make_champion_plan(view, mem)
    alloc = allocation.allocate(view, mem, lam=1.0, base=plan)
    assert alloc.hires_target == min(planner.CHAMP_HIRES[12],
                                     max(3, (40 + 12 + 10) // 3))


def test_capacity_reads_hiring_and_never_writes_it():
    cap_lean = allocation.capacity(
        make_view(day=12, crops={"WHEAT": 20}), Memory(), 4)
    cap_full = allocation.capacity(
        make_view(day=12, crops={"WHEAT": 20}), Memory(), 11)
    assert cap_full["chore_budget"] > cap_lean["chore_budget"]
    assert cap_full["herd_capacity"] >= cap_lean["herd_capacity"]


# ============================================================================
# ARITHMETIC -- the whole-stream rule, concavity, the opponent term
# ============================================================================

def test_ladder_is_the_engine_price_identity():
    for item in ("STRAWBERRY", "CARROT", "WHEAT", "MILK", "WOOL", "EGG"):
        for n in (1, 5, 17, 40, 160):
            assert rent.sell_revenue(item, 10000, n) == econ.sell_revenue(item, 10000, n)


def test_average_unit_price_falls_with_block_size():
    """The verified engine numbers the concavity argument rests on."""
    want = {"STRAWBERRY": (116.0, 82.5), "CARROT": (33.6, 30.2),
            "WHEAT": (24.2, 22.7), "MILK": (156.0, 119.1), "WOOL": (199.6, 170.2)}
    for item, (a5, a40) in want.items():
        assert round(rent.sell_revenue(item, 10000, 5) / 5, 1) == a5
        assert round(rent.sell_revenue(item, 10000, 40) / 40, 1) == a40


def test_marginal_rent_is_concave_in_n():
    """Decreasing marginals are what make the equal-marginal split well-posed;
    `planner._rent_pick`'s argmax over the same concave objective is a corner
    solution by construction, and the corner is the 41-strawberry board."""
    view = make_view(day=6, crops={"WHEAT": 5}, quads=("NW", "NE", "SW"))
    path = rent.inventory_path(view, horizon=14)
    for crop in rent.PLANTABLE:
        if rent.crop_block(crop, 6) is None:
            continue
        ms = [rent.marginal_rent(view, crop, n, path, stance="LEVEL") for n in range(0, 30, 2)]
        assert all(b <= a + 1e-6 for a, b in zip(ms, ms[1:])), (crop, ms)
        assert ms[0] > ms[-1], crop


def test_marginal_is_not_average_for_animals():
    """The geese swap lost -$5,234 and -$7,934 because 'the last 25 milk units
    earn $1.40' was applied to a whole cow whose FIRST 20 units sell at $140.

    Removing one cow must be priced at the WHOLE remaining stream of that
    animal, which at a 6-cow herd is far above what the naive
    `units x current marginal price` would give.
    """
    view = make_view(day=8, crops={"WHEAT": 20}, animals={"COW": 2},
                     market_inv={"MILK": 10000})
    path = rent.inventory_path(view, horizon=14)
    k, units_each, days = rent.animal_block("COW", 8)
    inv = rent.inv_at(path, "MILK", k)

    def value(n):
        return rent.sell_revenue("MILK", inv, int(round(units_each * n)))

    first = value(1)
    second = value(2) - value(1)
    # The naive number the refuted swap used: a whole animal's units priced at
    # ONE marginal price, the one that holds after the herd has already sold.
    naive = units_each * econ.price("MILK", inv + int(round(units_each * 2)))
    assert first > second > 0                      # concave, as it must be
    assert second > naive * 1.1, (second, naive)   # the whole stream is worth more

    # STRUCTURAL: animal_rent's gross IS that stream difference per animal-day,
    # never a count times a single price.
    gross = rent.animal_rent(view, "COW", 1, path, 0.0, labour_price=0.0,
                             params={"rent": {"include_feed": False,
                                              "include_fertilizer": False}})
    assert gross == pytest.approx(second / days, rel=1e-9)


def test_opponent_delta_is_nonnegative_and_monotone():
    view = make_view(day=8, opp_crops={"STRAWBERRY": 20}, opp_animals={"COW": 6})
    path = rent.inventory_path(view, horizon=14)
    inv = rent.inv_at(path, "STRAWBERRY", 10)
    vals = [rent.opponent_delta(view, "STRAWBERRY", inv, n) for n in (0, 1, 5, 20, 80)]
    assert vals[0] == 0.0
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] > 0.0
    # An opponent with no board of that product loses nothing.
    bare = make_view(day=8, opp_crops={}, opp_animals={})
    assert rent.opponent_delta(bare, "STRAWBERRY", inv, 80) == 0.0


def test_relative_rent_exceeds_own_rent_when_the_opponent_holds_supply():
    """The whole difference from maximising our own bank: a dollar denied to
    them counts like a dollar earned."""
    view = make_view(day=6, opp_crops={"STRAWBERRY": 24})
    path = rent.inventory_path(view, horizon=14)
    own = rent.block_rent(view, "STRAWBERRY", 12, path)
    rel_level = rent.relative_rent(view, "STRAWBERRY", 12, path, stance="LEVEL")
    rel_behind = rent.relative_rent(view, "STRAWBERRY", 12, path, stance="BEHIND")
    assert rel_behind > rel_level > own


def test_water_fill_respects_capacity_caps_and_floors():
    scores = {"a": 10.0, "b": 6.0, "c": 1.0}
    got, trace = rent.water_fill(
        10, list(scores), lambda c, n, s: scores[c] - n,      # concave, linear
        caps={"a": 3}, floors={"b": 2})
    assert sum(got.values()) <= 10
    assert got["a"] <= 3
    assert got["b"] >= 2
    assert len(trace) >= 1
    # Nothing is allocated once its marginal goes negative.
    zero, _t = rent.water_fill(10, ["a"], lambda c, n, s: -1.0)
    assert zero["a"] == 0


def test_water_fill_is_equal_marginal():
    """Greedy on a concave objective IS the equal-marginal optimum: no
    candidate's next step beats the last accepted one."""
    view = make_view(day=6, quads=("NW", "NE", "SW"))
    path = rent.inventory_path(view, horizon=14)
    cands = [c for c in rent.PLANTABLE if rent.crop_block(c, 6)]
    got, trace = rent.water_fill(
        40, cands,
        lambda c, n, s: rent.marginal_rent(view, c, n, path, step=s, stance="LEVEL"))
    last = [m for _c, _n, m in trace if m != float("inf")][-1]
    for c in cands:
        nxt = rent.marginal_rent(view, c, got[c], path, stance="LEVEL")
        assert nxt <= last + 1e-6, (c, nxt, last)


def test_geese_are_not_hardcoded_either_way():
    """Nothing about geese, cows or sheep is written in or out. At OUR
    measured chore cost the goose has to earn its turns; if the Maia2 executor
    lowers that cost, its rent rises -- monotonically and by arithmetic."""
    view = make_view(day=6, crops={"WHEAT": 20})
    mem = memory_for(view)
    path = rent.inventory_path(view, horizon=14)
    p = allocation.load_params()
    expensive = rent.animal_rent(view, "GOOSE", 0, path,
                                 allocation.chore_cost("GOOSE", mem, p),
                                 labour_price=40.0, params=p)
    cheap_p = copy.deepcopy(p)
    cheap_p["herd"]["chore_scale"] = 0.4
    cheaper = rent.animal_rent(view, "GOOSE", 0, path,
                               allocation.chore_cost("GOOSE", mem, cheap_p),
                               labour_price=40.0, params=cheap_p)
    assert cheaper > expensive
    free = rent.animal_rent(view, "GOOSE", 0, path, 0.0, labour_price=40.0, params=p)
    assert free > cheaper


# ============================================================================
# LAND -- learned, self-correcting, and one-way safe
# ============================================================================

def test_theta0_matches_the_champion_land_rule():
    """theta_0 must reproduce champ-h10's land decision exactly, or the dose
    ladder's land column is measuring the decider and not the dose."""
    theta = allocation.load_params()["land"]
    dec = land.LandDecider(theta)
    disagree = []
    for day in range(0, 30):
        for nq in (1, 2, 3):
            for money in (100, 1500, 9000, 40000):
                view = make_view(day=day, money=money,
                                 quads=tuple(["NW"] + LAND_ORDER[:nq - 1]),
                                 crops={"WHEAT": 6})
                mem = memory_for(view)
                plan = planner.make_champion_plan(view, mem)
                alloc = allocation.Allocation()
                alloc.trace = {"liquidation": plan.liquidation, "lead": 0.0}
                buy, _m, why = dec.decide(view, mem, alloc)
                want = bool(plan.land_buys)
                if buy != want:
                    disagree.append((day, nq, money, buy, want, why["gate"]))
    assert not disagree, f"{len(disagree)} disagreements, first {disagree[0]}"


def test_a_bad_theta_can_only_buy_less_land():
    """Land cannot be sold back, so the failure mode must be one-way. The
    learned score runs LAST and can only veto what the hard gates allowed."""
    good = land.LandDecider(allocation.load_params()["land"])
    theta = dict(allocation.load_params()["land"])
    for rogue_w, rogue_b in (([9e9] * 9, 9e9), ([-9e9] * 9, 9e9), ([0.0] * 9, -1.0)):
        bad = land.LandDecider({**theta, "w": rogue_w, "b": rogue_b})
        for day in range(0, 30):
            for nq in (1, 2, 3):
                view = make_view(day=day, money=90000,
                                 quads=tuple(["NW"] + LAND_ORDER[:nq - 1]))
                mem = memory_for(view)
                alloc = allocation.Allocation()
                alloc.trace = {"liquidation": day >= 27, "lead": 0.0}
                g, _m, _w = good.decide(view, mem, alloc)
                b, _m, _w = bad.decide(view, mem, alloc)
                assert b <= g, (rogue_b, day, nq)


def test_hard_gates_run_before_the_score():
    """A score of +infinity may not open a fourth quadrant, a day outside the
    window, or a purchase the env cap forbids."""
    theta = {**allocation.load_params()["land"], "b": 1e9, "w": [1e9] * 9}
    dec = land.LandDecider(theta)
    for day, quads, gate in (
            (5, ("NW", "NE", "SW"), "max_quadrants"),
            (3, ("NW",), "day_window"),
            (28, ("NW",), "day_window"),
    ):
        view = make_view(day=day, money=90000, quads=quads)
        mem = memory_for(view)
        alloc = allocation.Allocation()
        alloc.trace = {"liquidation": False, "lead": 0.0}
        buy, _m, why = dec.decide(view, mem, alloc)
        assert not buy and why["gate"] == gate, (day, quads, why)


def test_labour_and_utilisation_floors_veto_land():
    """The fourth quadrant lost twice at hands-per-work 0.47-0.58. The gate is
    on that MEASURED ratio, not on idle tiles."""
    theta = {**allocation.load_params()["land"],
             "max_quadrants": 4, "day_max": 24,
             "hands_per_work_floor": 0.8, "util_floor": 0.9}
    dec = land.LandDecider(theta)
    # Labour-short: 3 hands against a big pending-harvest load.
    short = make_view(day=16, money=90000, quads=("NW", "NE", "SW"),
                      crops={"WHEAT": 60}, hands=3, plant_age=6)
    mem = memory_for(short, yhands=3)
    land.observe_dawn(mem, short)
    alloc = allocation.Allocation()
    alloc.trace = {"liquidation": False, "lead": 0.0}
    buy, _m, why = dec.decide(short, mem, alloc)
    assert not buy and why["gate"] in ("hands_per_work_floor", "util_floor"), why
    assert land.measured_hands_per_work(short, mem, alloc) < 0.8


def test_land_measurements_are_of_ground_already_owned():
    """The self-correction: an over-bought farm reads its OWN idleness the
    next dawn. `observe_dawn` rolls a 3-day window of measured occupancy and
    carries no target at all."""
    mem = Memory()
    full = make_view(day=10, quads=("NW",), crops={"WHEAT": 25})
    land.observe_dawn(mem, full)
    assert land.measured_utilisation(full, mem) == pytest.approx(1.0)
    for d in (11, 12, 13):
        bare = make_view(day=d, quads=("NW", "NE"), crops={"WHEAT": 5})
        land.observe_dawn(mem, bare)
    assert land.measured_utilisation(bare, mem) < 0.3
    assert set(mem.rlv3) == {"day", "occupied", "chore"}          # no targets
    assert len(mem.rlv3["occupied"]) == 3                         # 3-day window


def test_theta_checkpoints_round_trip(tmp_path):
    """The JSON IS the checkpoint, so tomorrow's training resumes from the
    last file and a compute limit costs nothing."""
    dec = land.LandDecider(allocation.load_params()["land"])
    dec.theta["w"] = [0.1 * i for i in range(len(land.FEATURE_NAMES))]
    dec.theta["b"] = -0.25
    p = tmp_path / "sub" / "allocation_v1.json"
    dec.save(str(p))
    back = land.LandDecider.load(str(p))
    assert back.theta["w"] == dec.theta["w"] and back.theta["b"] == dec.theta["b"]
    assert land.LandDecider.load(allocation.PARAMS_PATH).theta["max_quadrants"] == 3


def test_apply_land_pressure_is_suppressed():
    """The third purchase path. Stage B5 ran three hours with it live and the
    capped control still held 3.8 quadrants. rlv3 overwrites land_buys
    wholesale and never calls it, so a crunch state cannot re-open it."""
    view = make_view(day=14, money=90000, quads=("NW", "NE", "SW"),
                     crops={"WHEAT": 70}, hands=11)
    plan = planner.make_champion_plan(view, memory_for(view))
    leaked = planner.DayPlan()
    leaked.plant_quota = {"WHEAT": 20}
    leaked.liquidation = False
    planner.apply_land_pressure(leaked, view)
    assert leaked.land_buys == ["SE"], "the leak is real on the neural path"
    for lam in (0.0, 1.0):
        _ref, got, _a, _v = _pair(dict(day=14, money=90000,
                                       quads=("NW", "NE", "SW"),
                                       crops={"WHEAT": 70}, hands=11), lam)
        assert got.land_buys == [], (lam, got.land_buys)
    assert plan.land_buys == []


# ============================================================================
# MECHANISM -- at lam > 0 the board is a SPLIT, not a corner
# ============================================================================

def test_lam_moves_the_mix_off_the_corner():
    """`planner._rent_pick` hands the whole free block to ONE winner. The
    water-fill splits it. This is the mechanism the day20_strawberry /
    day20_carrot rows are asking for."""
    state = dict(day=11, money=30000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 20, "WHEAT": 10}, animals={"COW": 6, "SHEEP": 6},
                 opp_crops={"STRAWBERRY": 22, "WHEAT": 10},
                 opp_animals={"COW": 6, "SHEEP": 6},
                 shops=("BAKERY", "YARN_STORE", "PET_CAFE"))
    _ref, _got, alloc, _v = _pair(state, 1.0)
    positive = [c for c, n in alloc.crop_targets.items() if n > 0]
    assert len(positive) >= 2, alloc.crop_targets
    assert sum(alloc.crop_shares.values()) == pytest.approx(1.0, abs=1e-6)
    # And the fill really spread: no single crop took every allocated tile.
    filled = {c: n for c, n in alloc.trace["share_crop"].items() if n > 0}
    assert len(filled) >= 2, alloc.trace["fill_crop"]


def test_strawberry_is_cut_by_replacement_not_subtraction():
    """The tile freed has to be re-let the same dawn or the capacity
    accounting is wrong. At lam = 1 the plant quota must not simply shrink:
    the total asked for must still fill the ground that is available."""
    state = dict(day=11, money=30000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 18, "WHEAT": 8}, animals={"COW": 6},
                 opp_crops={"STRAWBERRY": 24}, shops=("BAKERY", "PET_CAFE"))
    ref, got, alloc, view = _pair(state, 1.0)
    base_straw = ref.plant_quota.get("STRAWBERRY", 0)
    new_straw = got.plant_quota.get("STRAWBERRY", 0)
    assert new_straw < base_straw, (base_straw, new_straw)
    # Replacement, not subtraction: the ground the cut freed is re-let the
    # same dawn, so the total ask still covers the plantable capacity.
    assert sum(got.plant_quota.values()) >= 0.9 * alloc.crop_capacity, (
        alloc.crop_capacity, got.plant_quota)
    assert len(got.plant_quota) > len(ref.plant_quota), (ref.plant_quota, got.plant_quota)


def test_bare_ground_is_filled_without_the_fill_idle_knob():
    """idle_share: ours 0.124 against the leaders' 0.047. A day-20 board whose
    crop targets are all met leaves its free tiles bare under the champion
    plan; the capacity-driven fill re-lets them."""
    state = dict(day=20, money=60000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 41, "WHEAT": 20, "CARROT": 1},
                 animals={"COW": 6, "SHEEP": 10},
                 opp_crops={"STRAWBERRY": 20, "WHEAT": 20},
                 shops=("BAKERY", "YARN_STORE", "PET_CAFE", "ICE_CREAM_SHOP"))
    ref, got, alloc, _v = _pair(state, 1.0)
    assert sum(ref.plant_quota.values()) == 0        # the champion leaves it bare
    assert alloc.crop_capacity > 0
    assert sum(got.plant_quota.values()) >= 0.9 * alloc.crop_capacity, (
        alloc.crop_capacity, got.plant_quota)
    # Switchable: with the second pass off, the demand anchor binds again.
    off = copy.deepcopy(allocation.load_params())
    off["fill_bare"] = False
    _r, got_off, _a, _v = _pair(state, 1.0, params=off)
    assert sum(got_off.plant_quota.values()) <= sum(got.plant_quota.values())


def test_bare_ground_is_never_filled_with_a_product_no_shop_buys():
    """MELON appears in no SHOPS entry at all, so no future unlock can lift
    its demand past the town centre's 1 unit a day -- its anchor IS its
    ceiling. We already offer 2.811x the melon demand, and `_rent_pick` asks
    for 39 melon tiles (234 units) against the repo's own 140-unit target."""
    assert not any("MELON" in items for items in SHOPS.values())
    state = dict(day=11, money=30000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 18, "WHEAT": 8}, animals={"COW": 6},
                 opp_crops={"STRAWBERRY": 24}, shops=("BAKERY", "PET_CAFE"))
    _ref, got, alloc, _v = _pair(state, 1.0)
    assert got.plant_quota.get("MELON", 0) <= alloc.trace["crop_caps"]["MELON"]
    loose = copy.deepcopy(allocation.load_params())
    loose["fill_bare_shop_only"] = False
    _r, got2, _a, _v = _pair(state, 1.0, params=loose)
    assert got2.plant_quota.get("MELON", 0) > got.plant_quota.get("MELON", 0)


def test_the_opening_and_liquidation_are_never_deviated():
    """Days 0-1 keep the champion opening (the day-0 order list already sits
    near MAX_MARKET_ORDERS), and a liquidating plan is left alone."""
    for day in (0, 1):
        ref, got, alloc, _v = _pair(dict(day=day, money=2000, quads=("NW",)), 1.0)
        assert got.plant_quota == ref.plant_quota, day
        assert alloc.trace["deviate"] is False
    for day in (27, 28):
        ref, got, alloc, _v = _pair(dict(day=day, money=40000, quads=("NW", "NE"),
                                         crops={"WHEAT": 20}), 1.0)
        assert got.plant_quota == ref.plant_quota == {}
        assert got.animal_buys == ref.animal_buys == {}


def test_quota_never_exceeds_the_ground_it_can_go_in():
    """A quota larger than capacity is silently dropped by the scheduler, so
    the allocator must never ask for more than it can plant (the v3 lesson in
    features.apply_to_plan). The clip floor is the champion's OWN ask, so
    lam = 0 is never touched by it."""
    for lam in (0.33, 0.67, 1.0):
        for state in sweep(n=30, seed=7):
            ref, got, alloc, _v = _pair(state, lam)
            budget = max(alloc.crop_capacity, sum(ref.plant_quota.values()))
            assert sum(got.plant_quota.values()) <= budget, (lam, state["day"])


def _honest_board(view, alloc):
    """(candidates, the ground they can actually hold today).

    README "Map Features / Farm Space": "Each plant or animal occupies one
    square on the farm". So the board a candidate crop's target may occupy is
    the tiles standing under the CANDIDATES plus today's bare/freeing ground --
    never the tiles standing under a crop that cannot be replanted.
    """
    cands = [c for c in allocation.CROP_ORDER
             if alloc.trace["crop_caps"].get(c, 0) > 0]
    counts = allocation.standing_crops(view)
    return cands, sum(counts.get(c, 0) for c in cands) + alloc.crop_capacity


def test_target_board_never_counts_ground_a_candidate_cannot_occupy():
    """`board_budget` summed the tiles standing under EVERY crop, but the
    water-fill two lines later distributes that budget only among `crop_cands`.
    Ground held by a non-candidate -- MELON always (no SHOPS entry), STRAWBERRY
    past features.LAST_PLANT 13, TOMATO past 18, anything crashed -- was handed
    to the candidates as if it were free, and "Each plant or animal occupies
    one square on the farm" says it is not.

    WHY THE TESTS ALREADY HERE MISSED IT. Every mechanism test asserts on the
    FLOW (`plant_quota`), and `_plant_flow` clips the flow against the real
    `crop_capacity` -- precisely the silent downstream clip that lets an
    impossible ask look like a working one. `test_quota_never_exceeds_the_
    ground_it_can_go_in` measures that already-clipped number, so an inflated
    TARGET board passes it unchanged. Nothing asserted on `share_crop`, the
    UNCLIPPED ask that the water-fill optimises and that the dose blends at
    the target site. And `sweep()` fills a board with whatever crops it drew
    without regard to the day, so a big non-candidate stand was rare and never
    checked for.
    """
    # A day-20 board of the shape the scorecard measures: strawberry and melon
    # standing, neither replantable, wheat and carrot the only candidates.
    state = dict(day=20, money=60000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 30, "MELON": 6, "WHEAT": 10, "CARROT": 1},
                 animals={"COW": 6},
                 opp_crops={"STRAWBERRY": 20, "WHEAT": 20},
                 shops=("BAKERY", "YARN_STORE", "PET_CAFE", "ICE_CREAM_SHOP"))
    _ref, _got, alloc, view = _pair(state, 1.0)
    counts = allocation.standing_crops(view)
    cands, honest = _honest_board(view, alloc)
    # The board really is the one the defect needs: non-candidates standing on
    # ground the fill is about to spend, and bare ground left to spend it on.
    assert "STRAWBERRY" not in cands and "MELON" not in cands, cands
    assert counts.get("STRAWBERRY", 0) + counts.get("MELON", 0) >= 20, counts
    assert alloc.crop_capacity > 0
    share = alloc.trace["share_crop"]
    assert sum(share.values()) <= honest, (share, honest, alloc.crop_capacity)
    # ... and no single candidate is asked to stand on more tiles than exist.
    for c in cands:
        assert share.get(c, 0) <= honest, (c, share, honest)

    # The same invariant across the random sweep and the recorded leader
    # mid-season boards, where the phantom was measured at 5-32 tiles.
    for state in sweep(n=40, seed=11):
        _r, _g, alloc, view = _pair(state, 1.0)
        _c, honest = _honest_board(view, alloc)
        assert sum(alloc.trace["share_crop"].values()) <= honest, state["day"]
    for label, view, plan in mid_season_cases():
        mem = memory_for(view)
        alloc = allocation.allocate(view, mem, lam=1.0, base=plan)
        _c, honest = _honest_board(view, alloc)
        assert sum(alloc.trace["share_crop"].values()) <= honest, (
            label, alloc.trace["share_crop"], honest, alloc.crop_capacity)


def test_stance_reads_the_relative_position():
    ahead = make_view(day=12, money=90000, opp_money=1000,
                      crops={"STRAWBERRY": 20}, opp_crops={})
    behind = make_view(day=12, money=1000, opp_money=90000,
                       crops={}, opp_crops={"STRAWBERRY": 20})
    level = make_view(day=12, money=20000, opp_money=20000,
                      crops={"WHEAT": 10}, opp_crops={"WHEAT": 10})
    assert allocation.stance(ahead, Memory())[0] == "AHEAD"
    assert allocation.stance(behind, Memory())[0] == "BEHIND"
    assert allocation.stance(level, Memory())[0] == "LEVEL"
    # The risk quantile is the only thing the stance moves in the forecast.
    assert allocation.risk_quantile("AHEAD") == features.ANCHOR_QUANTILE
    assert allocation.risk_quantile("BEHIND") == 0.50


def test_crashed_and_out_of_season_products_leave_the_candidate_set():
    """The same crash test features.anchors uses, and a crop re-enters the
    moment the drain lifts its price."""
    crashed = make_view(day=8, market_inv={"STRAWBERRY": 10000 + 5000})
    caps, _h, dropped = allocation.demand_caps(crashed, "LEVEL")
    assert caps["STRAWBERRY"] == 0 and "STRAWBERRY" in dropped
    late = make_view(day=22)
    caps, herd, dropped = allocation.demand_caps(late, "LEVEL")
    assert caps["STRAWBERRY"] == 0 and caps["MELON"] == 0
    assert herd["COW"] == 0                       # ANIMAL_LAST_BUY COW = 18
    assert caps["WHEAT"] >= 0 and caps["CARROT"] >= 0


def test_market_orders_stay_under_the_ten_order_cap():
    """MAX_MARKET_ORDERS = 10 and a five-crop split adds BUY_SEED orders; the
    planner already records extra day-0 hires pushing the list past the cap
    and dropping the seed orders. If they are dropped, the allocation is not
    what is being measured."""
    for lam in (0.0, 1.0):
        for state in sweep(n=25, seed=13):
            _ref, got, _a, view = _pair(state, lam)
            orders = market.build_orders(view, got, memory_for(view))
            assert len(orders) <= MAX_MARKET_ORDERS, (lam, state["day"], orders)


def test_provider_duck_types_the_champion_provider():
    """`HeuristicPolicy(AllocationPlanProvider(...))` is the whole wiring, so
    rl/rollout and kagent/policy need no edit."""
    from kagent.policy import ChampionPlanProvider
    prov = allocation.AllocationPlanProvider(knobs=H10_RULES, lam=0.0)
    assert hasattr(prov, "make_plan")
    view = make_view(day=12, crops={"WHEAT": 10, "STRAWBERRY": 20}, animals={"COW": 4})
    mem_a, mem_b = Memory(), Memory()
    mem_a.observe(view)
    mem_b.observe(view)
    got = prov.make_plan(view, mem_a)
    ref = ChampionPlanProvider().make_plan(view, mem_b)
    # The provider runs the SAME knob bandit, so at lam=0 it is champ-h10, and
    # h10's day-12 sheep rule is the only difference from the bare champion.
    assert got.hires_target == ref.hires_target
    assert prov.last_alloc.trace["hour"] == 0
    assert prov.last_alloc.trace["lam"] == 0.0


def test_provider_at_lam0_reproduces_the_h10_bandit_exactly():
    from kagent.banditpolicy import KnobBanditPolicy
    for state in sweep(n=25, seed=17):
        view = make_view(**state)
        m1, m2 = Memory(), Memory()
        m1.observe(view)
        m2.observe(view)
        ref = KnobBanditPolicy(H10_RULES).make_plan(view, m1)
        got = allocation.AllocationPlanProvider(knobs=H10_RULES, lam=0.0).make_plan(view, m2)
        for f in PLAN_FIELDS + UNTOUCHED:
            assert getattr(ref, f) == getattr(got, f), (state["day"], f)


def test_the_herd_is_switchable_but_never_hardcoded():
    """The arithmetic orders the herd SHEEP > COW > GOOSE at our measured
    chore cost, and the goose goes negative once labour is dear. It still
    buys geese where labour is cheap -- and the project record says geese
    +4/+6 cost -8 and -12.6 points on the dev tree. That is a deliberate
    re-run of a refuted lever under a different mechanism, so it is SWITCHED:
    herd caps of 0 remove a kind entirely, with no code change."""
    view = make_view(day=6, crops={"WHEAT": 20}, animals={"COW": 4, "SHEEP": 2})
    mem = memory_for(view)
    p = allocation.load_params()
    path = rent.inventory_path(view, horizon=14)
    rents = {k: rent.animal_rent(view, k, 0, path, allocation.chore_cost(k, mem, p),
                                 labour_price=28.5, params=p) for k in rent.ANIMAL_KINDS}
    assert rents["SHEEP"] > rents["COW"] > rents["GOOSE"]
    dear = {k: rent.animal_rent(view, k, 0, path, allocation.chore_cost(k, mem, p),
                                labour_price=90.0, params=p) for k in rent.ANIMAL_KINDS}
    assert dear["GOOSE"] < 0 < rents["GOOSE"]

    state = dict(day=11, money=30000, quads=("NW", "NE", "SW"),
                 crops={"STRAWBERRY": 18, "WHEAT": 8}, animals={"COW": 6},
                 shops=("BAKERY", "PET_CAFE"))
    _r, on, _a, _v = _pair(state, 1.0)
    no_geese = copy.deepcopy(allocation.load_params())
    no_geese["herd"]["caps"]["GOOSE"] = 0
    _r, off, _a, _v = _pair(state, 1.0, params=no_geese)
    assert on.animal_buys.get("GOOSE", 0) > 0
    assert off.animal_buys.get("GOOSE", 0) == 0


def test_animal_rent_integrates_the_fertiliser_and_feed_streams():
    """One FERTILIZER a night at base $100 is bigger than the milk: pricing it
    at a single marginal price would have dominated the whole herd decision.
    Both side streams go through the engine identity (sell_revenue / buy_cost)
    over the WHOLE remaining block, like the product itself."""
    view = make_view(day=6, crops={"WHEAT": 20}, market_inv={"FERTILIZER": 10000})
    path = rent.inventory_path(view, horizon=14)
    p = allocation.load_params()
    with_sides = rent.animal_rent(view, "COW", 0, path, 0.0, labour_price=0.0, params=p)
    bare = rent.animal_rent(view, "COW", 0, path, 0.0, labour_price=0.0,
                            params={"rent": {"include_feed": False,
                                             "include_fertilizer": False}})
    assert with_sides != bare
    # Concave in the herd size: the 12th animal's fertiliser is worth less
    # than the first's, which a count x price could never express.
    first = rent.animal_rent(view, "COW", 0, path, 0.0, labour_price=0.0, params=p)
    twelfth = rent.animal_rent(view, "COW", 11, path, 0.0, labour_price=0.0, params=p)
    assert first > twelfth


def test_water_fill_trace_names_the_crop_the_direction_is_wrong_on():
    """If the dose ladder comes out monotone negative, this trace is what says
    which crop the water-fill was wrong about -- the mechanism check the
    ladder script prints."""
    _r, _g, alloc, _v = _pair(dict(day=11, money=30000, quads=("NW", "NE", "SW"),
                                   crops={"STRAWBERRY": 18, "WHEAT": 8}), 1.0)
    fill = alloc.trace["fill_crop"]
    assert fill and all(len(row) == 3 for row in fill)
    cut = alloc.trace["fill_pass2_from"] or len(fill)
    for half in (fill[:cut], fill[cut:]):
        ms = [m for _c, _n, m in half if m is not None]
        assert ms == sorted(ms, reverse=True), ms[:10]
    assert alloc.trace["fill_pass2_from"] is not None
    for key in ("stance", "lead", "quantile", "crop_caps", "herd_caps",
                "base_crop", "base_herd", "capacity", "land", "labour_price"):
        assert key in alloc.trace, key


def test_module_order_matches_rent():
    """allocation.py spells the candidate orders out because a bundle binds
    module aliases only after every source has run. Keep them in step."""
    assert allocation.CROP_ORDER == rent.PLANTABLE
    assert allocation.HERD_ORDER == rent.ANIMAL_KINDS
