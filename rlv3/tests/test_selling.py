"""Unit tests for rlv3/market_model.py, rlv3/selling.py and rlv3/sell_eval.py.

No games are played: every View here is built from a synthetic observation
dict, exactly the shape kagent.gamestate.View parses in a real episode.

The tests that matter most are the MECHANISM ones, because the repo record
says both scorecard metrics this component is briefed to move have already
been moved to leader levels without moving money or wins:

  * test_participation_cap_is_zero_off_tick and
    test_slice_gap_is_at_least_one_drain_tick -- the single mechanical claim
    that distinguishes this from the refuted `sell_pace`.
  * test_hire_is_never_displaced -- refusing a hire gated -$7,234 / -$39,678.
  * test_counterfactual_is_an_identity_under_the_recorded_sizer -- the
    pre-gate has to read zero when nothing changed, or its $1.5k threshold
    means nothing.
"""

from __future__ import annotations

import json
import os
import statistics as st_
from collections import Counter

import pytest

from kagent import econ, market, planner
from kagent.constants import (
    MARKET_PARAMS, MAX_MARKET_ORDERS, TOWN_CENTER_PRODUCTS,
    TOWN_SHOP_SELL_INTERVAL, TOWN_CENTER_SELL_INTERVAL, SHOPS, TURNS_PER_DAY,
)
from kagent.gamestate import Memory, View

from rlv3 import market_model as mm
from rlv3 import sell_eval
from rlv3 import selling
from rlv3.selling import SellConfig

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
I0 = 10_000
PRODUCTS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]


# ------------------------------------------------------------------ fixtures
def blank_tiles(n=10):
    return [[None for _ in range(n)] for _ in range(n)]


def farm(money=10_000, tiles=None, hands=0, hires_today=0):
    return {
        "money": money,
        "tiles": tiles if tiles is not None else blank_tiles(),
        "farmer": [4, 4],
        "hands": [[4, 4]] * hands,
        "unlocked_quadrants": ["NW"],
        "hires_today": hires_today,
    }


def make_view(step=24, shed=None, inv=None, shops=("BRUNCH_SPOT",),
              money=10_000, opp_money=10_000, my_tiles=None, opp_tiles=None,
              seeds=None, hands=0, hires_today=0):
    shed = dict(shed or {})
    inv = {p: I0 for p in PRODUCTS_L} | dict(inv or {})
    prices = {p: econ.price(p, inv[p]) for p in PRODUCTS_L}
    obs = {
        "player": 0,
        "step": step,
        "day": step // TURNS_PER_DAY,
        "hour": step % TURNS_PER_DAY,
        "farms": [farm(money, my_tiles, hands, hires_today), farm(opp_money, opp_tiles)],
        "private": {"shed": shed, "seeds": dict(seeds or {}),
                    "inventories": [{} for _ in range(hands + 1)]},
        "market": {"inventory": inv, "prices": prices},
        "town": {"unlocked_shops": list(shops)},
    }
    return View(obs)


def make_plan(floors=None, **kw):
    p = planner.DayPlan()
    p.sell_floors = dict(floors) if floors is not None else {}
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def coop_tiles(n, animal="GOOSE"):
    t = blank_tiles()
    placed = 0
    for y in range(10):
        for x in range(10):
            if placed >= n:
                return t
            t[y][x] = {"kind": "COOP", "animal": animal, "yield_units": 0,
                       "placed_day": 0}
            placed += 1
    return t


# ============================================================ market_model
def test_drain_tick_matches_the_engine_constants():
    assert mm.DRAIN_TICK == TOWN_SHOP_SELL_INTERVAL == 4
    assert mm.CENTER_TICK == TOWN_CENTER_SELL_INTERVAL == 24
    assert mm.SHOP_TICKS_PER_DAY == 6
    assert mm.CENTER_TICKS_PER_DAY == 1


@pytest.mark.parametrize("step,want", [(0, 0), (1, 4), (2, 4), (3, 4), (4, 4), (5, 8), (23, 24)])
def test_next_drain_step(step, want):
    assert mm.next_drain_step(step) == want


def _brute_drain(item, a, b, shops):
    """The engine's own rule, stepped one step at a time.

    `range(a, b)`, not `range(a + 1, b + 1)`: `_town_consume(env, state, step)`
    drains on `step % interval == 0` and the framework records the result at
    index step + 1, so the ticks VISIBLE between the observations at a and b
    are the ones that fired at steps a .. b-1. This reference and
    `mm.ticks_between` were both one step out and agreed with each other.
    """
    total = 0.0
    for k in range(a, b):
        if k % TOWN_SHOP_SELL_INTERVAL == 0:
            for shop in shops:
                items = SHOPS[shop]
                if item in items:
                    total += 2 if len(items) == 1 else 1
        if k % TOWN_CENTER_SELL_INTERVAL == 0 and item in TOWN_CENTER_PRODUCTS:
            total += 1
    return total


@pytest.mark.parametrize("shops", [
    (), ("BAKERY",), ("YARN_STORE",), ("YARN_STORE", "YARN_STORE"),
    ("BRUNCH_SPOT", "ICE_CREAM_SHOP", "FARMERS_MARKET", "PET_CAFE"),
])
def test_drain_between_is_exact(shops):
    for item in PRODUCTS_L:
        for a, b in [(0, 0), (0, 1), (0, 4), (3, 4), (0, 24), (23, 25), (47, 96), (100, 100)]:
            assert mm.drain_between(item, a, b, shops) == pytest.approx(
                _brute_drain(item, a, b, shops)), (item, a, b, shops)


def test_drain_between_is_zero_when_the_clock_goes_backwards():
    assert mm.drain_between("WHEAT", 100, 50, ("BAKERY",)) == 0.0


def test_hour_one_is_the_lowest_inventory_hour():
    """The compound tick (every shop plus the town centre) fires DURING the
    step whose hour is 0 and is first visible in the observation at hour 1, so
    hour 1 is the day's lowest-inventory hour and the best hour to sell into.

    Indexed by the hour of the LATER observation, which is the hour an agent
    is standing in when it reads the inventory.
    """
    shops = ("BRUNCH_SPOT", "ICE_CREAM_SHOP")
    per_step = [mm.drain_between("STRAWBERRY", 24 + h - 1, 24 + h, shops)
                for h in range(24)]
    assert per_step[1] == max(per_step)
    assert per_step[1] > per_step[5] > 0        # compound tick beats a plain one
    assert per_step[2] == per_step[3] == per_step[4] == 0


def test_is_drained_routes_melon_and_fertilizer_around_the_schedule():
    all_shops = tuple(SHOPS)
    assert not mm.is_drained("MELON", all_shops)         # in no shop, ever
    assert not mm.is_drained("FERTILIZER", all_shops)    # not even town centre
    assert mm.is_drained("WHEAT", ("BAKERY",))
    assert mm.is_drained("WOOL", ("YARN_STORE",))
    assert not mm.is_drained("WOOL", ("BAKERY",))        # trickle only


def test_dpdi_is_the_one_unit_price_drop_and_telescopes():
    inv = I0
    for item in ("STRAWBERRY", "WOOL", "MILK"):
        drop = sum(mm.dPdI(item, inv + k) for k in range(30))
        assert drop == pytest.approx(econ.price(item, inv) - econ.price(item, inv + 30))
        assert all(mm.dPdI(item, inv + k) >= 0 for k in range(30))


def test_season_capacity_is_units_sellable_above():
    assert mm.season_capacity("STRAWBERRY", I0, 80) == econ.units_sellable_above(
        "STRAWBERRY", I0, 80)


def test_own_remaining_counts_shed_hands_and_the_board():
    v = make_view(step=24, shed={"WOOL": 5}, hands=1)
    v.unit_invs[1]["WOOL"] = 3
    assert mm.own_remaining(v, "WOOL") == pytest.approx(8.0)


# ================================================================== caps
def test_participation_cap_is_zero_off_tick():
    """THE claim that separates this from the refuted sell_pace: on three steps
    of four the town has not consumed anything, so there is nothing to sell
    into and the cap is zero.

    Indexed on the CORRECTED clock (see `_brute_drain`). A slice at step
    `last` is committed by `_process_market` BEFORE `_town_consume` runs for
    that same step, so when `last` is itself a tick step the town's take at
    `last` is fresh demand that has already landed by `last + 1`. The
    zero-cap window therefore follows a slice made OFF a tick, which is where
    the schedule actually puts most of them.
    """
    cfg = SellConfig()
    mem = Memory()
    for last in (25, 29, 45):                    # off-tick slices
        for gap in (1, 2, 3):                    # still before the next tick
            v = make_view(step=last + gap)
            st = selling._sync(v, mem)
            st.last_slice_step["STRAWBERRY"] = last
            assert selling.participation_cap(v, "STRAWBERRY", st, cfg) == 0, (last, gap)
        for gap in (4, 8, 12):
            v = make_view(step=last + gap)
            st = selling._sync(v, mem)
            st.last_slice_step["STRAWBERRY"] = last
            assert selling.participation_cap(v, "STRAWBERRY", st, cfg) >= 1, (last, gap)
    # ON a tick step the take at `last` is already visible one step later.
    for last in (24, 28, 44):
        v = make_view(step=last + 1)
        st = selling._sync(v, mem)
        st.last_slice_step["STRAWBERRY"] = last
        assert selling.participation_cap(v, "STRAWBERRY", st, cfg) >= 1, last


def test_participation_cap_grows_with_the_gap_and_the_carry():
    cfg = SellConfig()
    mem = Memory()
    v = make_view(step=48)
    st = selling._sync(v, mem)
    st.last_slice_step["STRAWBERRY"] = 24
    wide = selling.participation_cap(v, "STRAWBERRY", st, cfg)
    st.last_slice_step["STRAWBERRY"] = 44
    narrow = selling.participation_cap(v, "STRAWBERRY", st, cfg)
    assert wide > narrow
    st.carry["STRAWBERRY"] = 5.0
    assert selling.participation_cap(v, "STRAWBERRY", st, cfg) == narrow + 5


def test_impact_cap_keeps_the_within_order_walk_inside_the_budget():
    cfg = SellConfig(impact_cap_pct=10)
    v = make_view(step=24)
    for item in ("STRAWBERRY", "WOOL", "MILK", "MELON"):
        n = selling.impact_cap(v, item, cfg)
        assert n >= 1
        prices = econ.sell_marginal_prices(item, v.market_inv[item], n)
        assert prices[-1] >= prices[0] * 0.90 - 1e-9, item
        over = econ.sell_marginal_prices(item, v.market_inv[item], n + 1)
        assert over[-1] < prices[0] * 0.90


# ============================================================== the schedule
def _run_day(day=8, shed=None, shops=("BRUNCH_SPOT",), cfg=None, hours=24,
             **view_kw):
    """Step a whole day through plan_slices, returning the slices per step."""
    cfg = cfg or SellConfig()
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 80, "WOOL": 120, "MILK": 100, "MELON": 80})
    got = []
    for h in range(hours):
        v = make_view(step=day * TURNS_PER_DAY + h, shed=dict(shed or {}),
                      shops=shops, **view_kw)
        got.append((v.step, selling.plan_slices(v, plan, mem, cfg)))
    return got, mem


def test_slice_gap_is_at_least_one_drain_tick():
    """MECHANISM / anti-self-deception: median steps between two slices of one
    item must be >= 4. Below 4 the schedule has collapsed back into sell_pace."""
    got, _ = _run_day(shed={"STRAWBERRY": 60})
    steps = [s for s, sl in got if any(x.item == "STRAWBERRY" for x in sl)]
    assert len(steps) >= 4, steps
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert st_.median(gaps) >= SellConfig().min_slice_gap, gaps
    assert min(gaps) >= SellConfig().min_slice_gap, gaps


def test_the_shipped_sizer_does_not_gap(monkeypatch):
    """Control: champ-h10 re-offers on the very next step, which is exactly why
    sell_pace read flat. If this ever starts passing, the control is broken."""
    cfg = SellConfig(enabled=False)
    got, _ = _run_day(shed={"STRAWBERRY": 60}, cfg=cfg)
    steps = [s for s, sl in got if any(x.item == "STRAWBERRY" for x in sl)]
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert gaps and st_.median(gaps) == 1


def test_melon_and_fertilizer_bypass_the_schedule_entirely():
    mem = Memory()
    plan = make_plan({"MELON": 80})
    v = make_view(step=8 * 24 + 7, shed={"MELON": 40, "FERTILIZER": 12},
                  shops=tuple(SHOPS))
    slices = selling.plan_slices(v, plan, mem, SellConfig())
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    got = {s.item: s.units for s in slices}
    assert got.get("MELON") == shipped.get("MELON")
    assert got.get("FERTILIZER") == shipped.get("FERTILIZER")
    assert {s.reason for s in slices if s.item in ("MELON", "FERTILIZER")} == {"ratchet"}


def test_shed_pressure_and_the_final_push_use_the_shipped_sizer():
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 80})
    for step, shed in ((8 * 24 + 7, {"STRAWBERRY": 95}),      # pressure valve
                       (planner.FINAL_STEP - 3, {"STRAWBERRY": 30})):
        v = make_view(step=step, shed=shed)
        slices = selling.plan_slices(v, plan, Memory(), SellConfig())
        shipped = [(o[1], o[2]) for o in market._sell_orders(v, plan, mem)]
        assert [(s.item, s.units) for s in slices] == shipped, step


def test_a_flooding_opponent_flips_the_product_to_dump():
    """The generalisation of the shipped melon rule (opp_ripening >= 3 -> floor
    40) to every product, read off their actual board."""
    opp = blank_tiles()
    for i in range(12):
        opp[i // 10][i % 10] = {"kind": "PLANT", "crop": "STRAWBERRY",
                                "planted_day": 0, "yield_units": 3}
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 80})
    v = make_view(step=12 * 24 + 6, shed={"STRAWBERRY": 40}, opp_tiles=opp)
    assert selling.dominance(v, "STRAWBERRY") >= SellConfig().deny_ratio
    slices = {s.item: s for s in selling.plan_slices(v, plan, mem, SellConfig())}
    assert slices["STRAWBERRY"].reason == "deny"
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    assert slices["STRAWBERRY"].units == shipped["STRAWBERRY"]


def test_a_quiet_opponent_leaves_the_product_paced():
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 80})
    v = make_view(step=12 * 24 + 8, shed={"STRAWBERRY": 40})
    assert selling.dominance(v, "STRAWBERRY") < SellConfig().deny_ratio
    slices = {s.item: s for s in selling.plan_slices(v, plan, mem, SellConfig())}
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    assert slices["STRAWBERRY"].units < shipped["STRAWBERRY"]
    assert slices["STRAWBERRY"].reason in ("drain", "impact")


def test_liquidation_day_takes_everything_the_shipped_sizer_would():
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 80})
    v = make_view(step=27 * 24 + 6, shed={"STRAWBERRY": 40})
    slices = {s.item: s for s in selling.plan_slices(v, plan, mem, SellConfig())}
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    assert slices["STRAWBERRY"].units == shipped["STRAWBERRY"]
    assert slices["STRAWBERRY"].reason == "liquidate"


def test_a_crashed_market_is_released_not_paced():
    mem = Memory()
    plan = make_plan({"STRAWBERRY": 1})
    crashed_inv = I0 + 3000
    v = make_view(step=12 * 24 + 6, shed={"STRAWBERRY": 20},
                  inv={"STRAWBERRY": crashed_inv})
    assert v.market_prices["STRAWBERRY"] <= 0.35 * MARKET_PARAMS["STRAWBERRY"]["base"]
    slices = {s.item: s for s in selling.plan_slices(v, plan, mem, SellConfig())}
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    assert slices["STRAWBERRY"].units == shipped["STRAWBERRY"]


def test_paced_only_switch_leaves_the_flat_curve_products_alone():
    mem = Memory()
    plan = make_plan({"WHEAT": 1, "STRAWBERRY": 80})
    v = make_view(step=12 * 24 + 6, shed={"WHEAT": 40, "STRAWBERRY": 40},
                  shops=("BAKERY", "BRUNCH_SPOT"))
    slices = {s.item: s for s in selling.plan_slices(v, plan, mem,
                                                     SellConfig(paced_only=True))}
    shipped = {o[1]: o[2] for o in market._sell_orders(v, plan, mem)}
    assert slices["WHEAT"].units == shipped["WHEAT"]
    assert slices["STRAWBERRY"].units < shipped["STRAWBERRY"]


# ================================================================ posture
def test_risk_posture_is_bounded_and_signed():
    ahead = make_view(step=12 * 24, money=80_000, opp_money=5_000)
    behind = make_view(step=12 * 24, money=5_000, opp_money=80_000)
    assert selling.risk_posture(ahead) == pytest.approx(1.0)
    assert selling.risk_posture(behind) == pytest.approx(-1.0)
    level = selling.risk_posture(make_view(step=12 * 24, money=10_000, opp_money=10_000))
    assert -1.0 <= level <= 1.0


def test_losing_holds_the_scarce_side_one_extra_tick():
    """Objective 3: a smaller CERTAIN loss is worth less than a shot at the win,
    so behind_hold is the only variance the sell module can add."""
    mem_a, mem_b = Memory(), Memory()
    plan = make_plan({"WOOL": 120})
    common = dict(step=12 * 24 + 8, shed={"WOOL": 40}, shops=("YARN_STORE",))
    behind = make_view(money=1_000, opp_money=200_000, **common)
    ahead = make_view(money=200_000, opp_money=1_000, **common)
    assert selling.risk_posture(behind) < 0 < selling.risk_posture(ahead)
    n_behind = {s.item: s.units for s in selling.plan_slices(
        behind, plan, mem_a, SellConfig())}.get("WOOL", 0)
    n_ahead = {s.item: s.units for s in selling.plan_slices(
        ahead, plan, mem_b, SellConfig())}.get("WOOL", 0)
    assert n_behind <= n_ahead


def test_relative_marginal_uses_the_telescoped_spread():
    v = make_view(step=12 * 24, shed={"STRAWBERRY": 40})
    inv = v.market_inv["STRAWBERRY"]
    assert selling.relative_marginal(v, "STRAWBERRY", 5) == pytest.approx(
        econ.sell_revenue("STRAWBERRY", inv, 5))      # empty opponent board
    opp = blank_tiles()
    for i in range(6):
        opp[0][i] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0,
                     "yield_units": 4}
    v2 = make_view(step=12 * 24, shed={"STRAWBERRY": 40}, opp_tiles=opp)
    m = int(round(selling.opponent_exposure(v2, "STRAWBERRY")))
    assert m > 0
    want = (econ.sell_revenue("STRAWBERRY", inv, 5)
            - (econ.price("STRAWBERRY", inv) - econ.price("STRAWBERRY", inv + m)))
    assert selling.relative_marginal(v2, "STRAWBERRY", 5) == pytest.approx(want)


def test_relative_marginal_is_zero_for_an_empty_order():
    v = make_view(step=12 * 24)
    assert selling.relative_marginal(v, "STRAWBERRY", 0) == 0.0


# ============================================================ dawn update
def test_dawn_update_only_fires_at_hour_zero_once_a_day():
    mem = Memory()
    cfg = SellConfig()
    st = selling.dawn_update(make_view(step=5 * 24 + 3), mem, cfg)
    assert st.day == -1                      # not dawn
    st = selling.dawn_update(make_view(step=5 * 24, shed={"WOOL": 10}), mem, cfg)
    assert st.day == 5 and st.day_target
    st.day_target = {"MARKER": 1.0}
    selling.dawn_update(make_view(step=5 * 24, shed={"WOOL": 10}), mem, cfg)
    assert st.day_target == {"MARKER": 1.0}  # same day, not recomputed


def test_lam_tracks_the_dawn_residual_and_stays_clipped():
    """The residual (actual - predicted) is EXACTLY the part of yesterday we
    got wrong. Nothing is fitted; the clip bounds the damage when the residual
    is really a shop draw."""
    cfg = SellConfig()
    opp = blank_tiles()
    for i in range(8):
        opp[0][i] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0,
                     "yield_units": 2}
    mem = Memory()
    selling.dawn_update(make_view(step=5 * 24, opp_tiles=opp), mem, cfg)
    st = selling.state_for(mem)
    assert st.lam.get("STRAWBERRY", 1.0) == 1.0

    st.predicted_dawn_inv["STRAWBERRY"] = I0
    selling.dawn_update(make_view(step=6 * 24, inv={"STRAWBERRY": I0 + 400},
                                  opp_tiles=opp), mem, cfg)
    assert st.lam["STRAWBERRY"] > 1.0        # they supplied more than we saw

    mem2 = Memory()
    selling.dawn_update(make_view(step=5 * 24, opp_tiles=opp), mem2, cfg)
    st2 = selling.state_for(mem2)
    st2.predicted_dawn_inv["STRAWBERRY"] = I0
    selling.dawn_update(make_view(step=6 * 24, inv={"STRAWBERRY": I0 - 400},
                                  opp_tiles=opp), mem2, cfg)
    assert st2.lam["STRAWBERRY"] < 1.0
    for s in (st, st2):
        assert cfg.lam_lo <= s.lam["STRAWBERRY"] <= cfg.lam_hi


def test_lam_never_escapes_its_clip_under_an_absurd_residual():
    cfg = SellConfig()
    mem = Memory()
    selling.dawn_update(make_view(step=5 * 24), mem, cfg)
    st = selling.state_for(mem)
    for d in range(6, 20):
        st.predicted_dawn_inv = {p: I0 for p in PRODUCTS_L}
        selling.dawn_update(make_view(step=d * 24, inv={p: I0 + 9_000 for p in PRODUCTS_L}),
                            mem, cfg)
        for v in st.lam.values():
            assert cfg.lam_lo <= v <= cfg.lam_hi


def test_carry_rolls_yesterdays_shortfall_forward_and_liquidation_clears_it():
    cfg = SellConfig()
    mem = Memory()
    selling.dawn_update(make_view(step=5 * 24, shed={"WOOL": 30}), mem, cfg)
    st = selling.state_for(mem)
    st.day_target = {"WOOL": 10.0}
    st.sold_today = {"WOOL": 3}
    selling.dawn_update(make_view(step=6 * 24, shed={"WOOL": 30}), mem, cfg)
    assert st.carry["WOOL"] == pytest.approx(7.0)
    assert st.sold_today == {}

    st.day_target = {"WOOL": 10.0}
    st.sold_today = {"WOOL": 1}
    selling.dawn_update(make_view(step=planner.LIQUIDATION_DAY * 24,
                                  shed={"WOOL": 30}), mem, cfg)
    assert st.carry == {}, "an unmet pace target must not block the endgame ramp"


def test_day_target_is_capped_by_what_the_town_absorbs():
    cfg = SellConfig()
    mem = Memory()
    selling.dawn_update(make_view(step=10 * 24, shed={"WOOL": 90},
                                  shops=("YARN_STORE",)), mem, cfg)
    st = selling.state_for(mem)
    assert st.day_target["WOOL"] <= mm.drain_per_day("WOOL", ("YARN_STORE",)) + 1e-9


def test_state_resets_when_the_step_goes_backwards():
    """The arena reuses processes, so a fresh episode can land on a warm
    object; carrying one game's slice clock into the next would silently
    suppress the whole first day of sells."""
    mem = Memory()
    v1 = make_view(step=400)
    st = selling._sync(v1, mem)
    st.last_slice_step["WOOL"] = 400
    st.lam["WOOL"] = 1.9
    st.day = 16
    st2 = selling._sync(make_view(step=3), mem)
    assert st2 is st
    assert st2.last_slice_step == {} and st2.lam == {} and st2.day == -1


# ============================================================= build_orders
def test_build_orders_calls_the_shipped_builder_exactly_once(monkeypatch):
    calls = []
    original = market.build_orders

    def counting(view, plan, memory):
        calls.append(1)
        return original(view, plan, memory)

    monkeypatch.setattr(market, "build_orders", counting)
    v = make_view(step=8 * 24 + 4, shed={"STRAWBERRY": 40, "WOOL": 20})
    selling.build_orders(v, make_plan({"STRAWBERRY": 80, "WOOL": 120}), Memory())
    assert calls == [1]


def test_hire_is_never_displaced():
    """Hiring is nearly free and refusing it is refuted (m-hire5 / m-hire3
    gated -$7,234 / -$39,678), so the leading sells must never cost a HIRE."""
    v = make_view(step=8 * 24, shed={p: 30 for p in
                                     ("STRAWBERRY", "WOOL", "MILK", "MELON", "EGG")},
                  money=500_000, shops=("BRUNCH_SPOT", "YARN_STORE", "PIZZA_SHOP"))
    plan = make_plan({"STRAWBERRY": 80, "WOOL": 120, "MILK": 100, "MELON": 80},
                     hires_target=8)
    base = market.build_orders(v, make_plan(
        {"STRAWBERRY": 80, "WOOL": 120, "MILK": 100, "MELON": 80}, hires_target=8),
        Memory())
    got = selling.build_orders(v, plan, Memory())
    assert sum(1 for o in base if o[0] == "HIRE") == 8
    assert sum(1 for o in got if o[0] == "HIRE") == 8
    assert [o[0] for o in got[:8]] == ["HIRE"] * 8
    assert len(got) <= MAX_MARKET_ORDERS


def test_sell_moves_into_the_leading_slots():
    """champ-h10 lets sells 'fill the remaining slots', so ours are quoted last
    all season. Engine fact 2: slot i resolves for both players before slot
    i+1, so moving them forward is a strictly better quote and a liquidity fix
    -- and it changes no walking and no allocation."""
    common = dict(step=8 * 24 + 4, shed={"STRAWBERRY": 40, "WOOL": 20, "MILK": 20},
                  money=200_000, shops=("BRUNCH_SPOT", "YARN_STORE", "PIZZA_SHOP"),
                  seeds={})
    floors = {"STRAWBERRY": 80, "WOOL": 120, "MILK": 100}
    v = make_view(**common)
    base = market.build_orders(v, make_plan(floors, plant_quota={"WHEAT": 6}), Memory())
    got = selling.build_orders(v, make_plan(floors, plant_quota={"WHEAT": 6}), Memory())
    assert base[0][0] != "SELL", "the shipped builder quotes its sells last"
    assert got[0][0] == "SELL"
    assert sum(1 for o in got[:SellConfig().sell_slots_lead] if o[0] == "SELL") >= 1
    assert any(o[0] == "BUY_SEED" for o in base) and any(o[0] == "BUY_SEED" for o in got)


def test_feed_items_are_kept_out_of_the_leading_slots():
    """BUY_PRODUCT is quoted at inv-1, so selling wheat ahead of the feed buy
    would raise the price of our own feed."""
    tiles = coop_tiles(12)          # 12 mouths -> a 24-unit feed target
    v = make_view(step=8 * 24 + 4, shed={"WHEAT": 20, "STRAWBERRY": 30},
                  money=200_000, my_tiles=tiles,
                  shops=("BAKERY", "BRUNCH_SPOT"))
    floors = {"WHEAT": 1, "STRAWBERRY": 80}
    got = selling.build_orders(v, make_plan(floors), Memory())
    kinds = [(o[0], o[1] if len(o) > 1 else None) for o in got]
    assert ("BUY_PRODUCT", "WHEAT") in kinds
    # The leading SELL block is the run of sells before the first other order.
    lead = []
    for o in got:
        if o[0] != "SELL":
            break
        lead.append(o)
    assert lead and not any(o[1] == "WHEAT" for o in lead)
    assert ("SELL", "WHEAT") in kinds, "the wheat sell must survive, only move back"
    assert kinds.index(("SELL", "WHEAT")) > kinds.index(("BUY_PRODUCT", "WHEAT"))


def test_build_orders_never_exceeds_the_engine_cap():
    v = make_view(step=8 * 24, money=1_000_000,
                  shed={p: 30 for p in PRODUCTS_L}, shops=tuple(SHOPS))
    plan = make_plan({p: 1 for p in PRODUCTS_L}, hires_target=8,
                     plant_quota={"WHEAT": 9, "CARROT": 9})
    got = selling.build_orders(v, plan, Memory())
    assert len(got) <= MAX_MARKET_ORDERS
    assert all(isinstance(o, list) and o for o in got)


# ---------------------------------------------------------- slot eviction
BUSY_SHED = {"STRAWBERRY": 30, "MILK": 30, "WOOL": 30, "EGG": 30}
BUSY_FLOORS = {"STRAWBERRY": 80, "MILK": 100, "WOOL": 120, "EGG": 20}


def busy_view():
    """Day 12, hour 1: the hour the shipped builder still hires, with four
    mouths to feed so the feed BUY_PRODUCT fires too."""
    return make_view(step=12 * 24 + 1, money=200_000, shed=dict(BUSY_SHED),
                     my_tiles=coop_tiles(4),
                     shops=("BRUNCH_SPOT", "YARN_STORE", "PIZZA_SHOP", "BAKERY"))


def busy_plan(hires):
    """A plan that asks for every kind of buy the shipped builder DEBITS."""
    p = make_plan(BUSY_FLOORS, hires_target=hires,
                  plant_quota={"CARROT": 5, "TOMATO": 3})
    p.land_buys = ["SW"]
    p.animal_buys = {"GOOSE": 2, "SHEEP": 1}
    return p


CFGS = [("default", None), ("RECOMMENDED", selling.RECOMMENDED)]


@pytest.mark.parametrize("cfg_label,cfg", CFGS, ids=[c[0] for c in CFGS])
@pytest.mark.parametrize("hires", [2, 4, 6])
def test_the_slot_reorder_never_evicts_an_order_the_shipped_builder_sent(
        hires, cfg_label, cfg):
    """REGRESSION (rlv3/selling.py build_orders).

    The builder assembled `hires + lead_sells + rest + tail_sells` and cut the
    WHOLE list at MAX_MARKET_ORDERS, while the sells came from
    market._sell_orders unbounded by the slots left over. On a busy day the
    leading sells therefore pushed the shipped builder's BUY_LAND /
    BUY_ANIMAL / BUY_SEED past the cap, and the engine drops anything past 10
    silently (README, "Market Action") -- after the shipped builder has
    already DEBITED them from the DayPlan (`plan.land_buys = remaining_land`,
    `plan.animal_buys[kind] = want - n`). kagent.policy.HeuristicPolicy.act
    builds the DayPlan once at hour 0 and reuses it for the rest of the day,
    so the quadrant and the animals are never re-requested: a paid-for
    quadrant simply never arrives.

    WHY THE ORIGINAL TESTS MISSED IT.
      * test_build_orders_never_exceeds_the_engine_cap asserted only
        `len(got) <= MAX_MARKET_ORDERS` -- the truncation that IS the bug
        satisfies that assertion exactly.
      * test_hire_is_never_displaced guards the HIRE block only, and its plan
        has no land_buys and no animal_buys, so there was nothing behind the
        hires to evict.
      * test_sell_moves_into_the_leading_slots does check that a BUY_SEED
        survives, but on a plan with hires_target unset (0) and no land, where
        the shipped list is far short of 10 and nothing CAN be pushed off the
        end.
    None of them compared the rlv3 list against the shipped list order by
    order on a day that actually fills the slots.

    Both configs are run: the defect was reported against `slots_only`
    (`selling.RECOMMENDED`, shipped by the `recommended` and `full` presets
    and documented as "free: it changes no walking, no allocation, and no
    sizing"), and the bug was in the assembly, which slots_only shares.
    """
    view = busy_view()
    shipped = market.build_orders(view, busy_plan(hires), Memory())
    plan = busy_plan(hires)
    got = selling.build_orders(view, plan, Memory(), cfg)

    assert len(got) <= MAX_MARKET_ORDERS
    ship_buys = sorted(map(repr, (o for o in shipped if o[0] != "SELL")))
    got_buys = sorted(map(repr, (o for o in got if o[0] != "SELL")))
    assert got_buys == ship_buys, (
        "the slot re-order evicted an order the shipped builder chose to send "
        "and had already debited from the DayPlan")
    # ... and the debits the shipped builder made are honoured by what we send.
    assert plan.land_buys == [] and any(o[0] == "BUY_LAND" for o in got)
    for kind, left in plan.animal_buys.items():
        want = busy_plan(hires).animal_buys[kind]
        sent = sum(o[2] for o in got if o[0] == "BUY_ANIMAL" and o[1] == kind)
        assert want - left == sent, f"{kind} debited {want - left} but sent {sent}"


_REPLAY_CASES = None


def replay_cases_at_dawn():
    """Real leader boards at HOUR 1 -- the hour the book actually saturates.

    conftest.mid_season_cases() samples hour 8, and the shipped builder only
    hires while `view.hour <= 1`, so at hour 8 there is no HIRE block, `rest`
    is two or three orders long and nothing can be pushed off the end. That
    is why 18 real mid-season cases plus every synthetic test in this file
    missed the eviction: not one of them was taken at an hour that fills the
    ten slots. Measured over days 2-28 at hours 0/1/8/16 of all 12 leader
    replays, both seats, the pre-fix builder dropped a shipped non-SELL order
    on 1,580 of 8,640 steps (18.3%).
    """
    global _REPLAY_CASES
    if _REPLAY_CASES is None:
        import glob
        import os
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        out = []
        for team in ("mmpq", "boey", "dsm"):
            hits = sorted(glob.glob(os.path.join(repo, "replays", team, "*.json")))
            if not hits:
                continue
            with open(hits[0]) as fh:
                steps = json.load(fh)["steps"]
            for day in (6, 12, 20):
                k = day * 24 + 1
                if k >= len(steps):
                    continue
                for seat in (0, 1):
                    obs = dict(steps[k][seat]["observation"])
                    obs.setdefault("step", k)
                    out.append((f"{team}-d{day}-s{seat}", View(obs)))
        _REPLAY_CASES = out
    return _REPLAY_CASES


def _replay_plan(view):
    """A fresh DayPlan per call: build_orders DEBITS the plan it is given."""
    mem = Memory()
    mem.observe(view)
    return planner.make_day_plan(view, mem, speculative=True)


@pytest.mark.parametrize("cfg_label,cfg", CFGS, ids=[c[0] for c in CFGS])
def test_no_order_is_evicted_on_real_leader_boards(cfg_label, cfg):
    """The same invariant on recorded boards rather than a built one.

    Red without the fix on mmpq day 20 seat 1, where the order lost is the
    animal-feed ['BUY_PRODUCT', 'WHEAT', 15] -- README "Watering / Animal
    Feed": two consecutive unfed days and every animal escapes, unrecoverable.
    """
    cases = replay_cases_at_dawn()
    if not cases:
        pytest.skip("no replays under replays/{mmpq,boey,dsm}")
    for label, view in cases:
        shipped = market.build_orders(view, _replay_plan(view), Memory())
        got = selling.build_orders(view, _replay_plan(view), Memory(), cfg)
        assert len(got) <= MAX_MARKET_ORDERS, label
        lost = (Counter(map(repr, (o for o in shipped if o[0] != "SELL")))
                - Counter(map(repr, (o for o in got if o[0] != "SELL"))))
        assert not lost, f"{label}: the slot re-order deleted {sorted(lost.elements())}"


def test_the_slot_reorder_still_leads_with_sells_when_slots_are_free():
    """The eviction fix must size the sells to the free slots, not switch the
    slot re-order off: whatever room is left after the shipped buys still goes
    to SELLs, quoted ahead of them."""
    view = busy_view()
    got = selling.build_orders(view, busy_plan(2), Memory())
    non_hire = [o for o in got if o[0] != "HIRE"]
    assert non_hire[0][0] == "SELL", "sells must still take the leading slots"
    assert got.index(["HIRE"]) == 0, "HIRE still leads the list"


# ------------------------------------------- the room budget and the books
BUDGET_ITEMS = ("STRAWBERRY", "MILK", "WOOL", "EGG", "WHEAT", "FERTILIZER")


def budget_view():
    """Day 12, HOUR 1, six sellable products, shed total 60.

    Hour 1 and not hour 0: at hour 0 no drain tick has accrued yet, so
    participation_cap is 0 for every product and only the RATCHET products
    get a slice at all. Shed 60 and not 120: `busy_view()` is over
    SHED_PRESSURE (80), which sends plan_slices down the `pressure` bypass.
    """
    return make_view(step=12 * TURNS_PER_DAY + 1, money=200_000,
                     shed={p: 10 for p in BUDGET_ITEMS},
                     my_tiles=coop_tiles(4),
                     shops=("BRUNCH_SPOT", "YARN_STORE", "PIZZA_SHOP", "BAKERY"))


def budget_plan():
    """Enough hires and buys that the shipped builder leaves 2 free slots."""
    p = make_plan({"STRAWBERRY": 80, "MILK": 100, "WOOL": 120, "EGG": 20,
                   "WHEAT": 5, "FERTILIZER": 5},
                  hires_target=3, plant_quota={"CARROT": 5, "TOMATO": 3})
    p.land_buys = ["SW"]
    p.animal_buys = {"GOOSE": 2, "SHEEP": 1}
    return p


def test_the_room_budget_books_only_the_slices_it_actually_sends():
    """REGRESSION (rlv3/selling.py, the room budget).

    The eviction fix stopped the assembled list being truncated and truncated
    `slices` instead -- but it did so in build_orders, AFTER plan_slices had
    already committed every slice it returned to the schedule's books:

        st.last_slice_step[item] = view.step
        st.sold_today[item] += units
        st.carry[item] -= units

    So a slice the budget deleted was still recorded as sold. Three
    consequences, all silent: participation_cap reads `last_slice_step`, so
    the product was gated out for another whole drain tick after an offer
    that was never made; the dawn carry roll-forward compares `sold_today`
    against `day_target`, so the day looked met; and the carry that should
    have rolled the shortfall forward was spent. This is the module's own
    named failure -- an output clipped downstream so the scheduler never
    learns its ask was impossible. Measured on one real game on the `sell`
    preset (rlv3 SELL_MODES["full"]): 38 slices, 362 units across six
    products, booked and never offered.

    WHY THE EXISTING TESTS MISSED IT.
      * test_the_slot_reorder_never_evicts_an_order_the_shipped_builder_sent
        and test_no_order_is_evicted_on_real_leader_boards compare ORDER
        LISTS. Neither ever reads `memory.rlv3_sell`, and a booked-then-
        deleted slice changes no order in either list.
      * Both of them run on `busy_view()`, whose shed holds 120 items --
        above SHED_PRESSURE (80). plan_slices therefore takes the `pressure`
        bypass and returns `_shipped_slices`, which books nothing at all. The
        committing branch had no test that saturated the order book, so the
        budget and the commit loop were never exercised on the same step.
      * test_slice_gap_is_at_least_one_drain_tick and
        test_carry_rolls_yesterdays_shortfall_forward_and_liquidation_clears_it
        do exercise the commit loop, but they call plan_slices DIRECTLY, with
        no caller and therefore no budget.
    """
    cfg = SellConfig()
    view = budget_view()
    mem = Memory()
    got = selling.build_orders(view, budget_plan(), mem, cfg)

    sent = {o[1]: o[2] for o in got if o[0] == "SELL"}
    assert len(got) <= MAX_MARKET_ORDERS
    # If the budget does not bite on this board the test proves nothing.
    assert set(sent) < set(BUDGET_ITEMS), (
        "the shipped buys must leave fewer slots than the schedule wants, "
        f"or there is no truncation to regress on: sent {sorted(sent)}")

    st = selling.state_for(mem)
    for item in BUDGET_ITEMS:
        if item in sent:
            assert st.sold_today.get(item, 0) == sent[item], item
            assert st.last_slice_step.get(item) == view.step, item
        else:
            assert st.sold_today.get(item, 0) == 0, (
                f"{item} was booked as sold but no SELL order was sent")
            assert item not in st.last_slice_step, (
                f"{item} lost its slot, yet the schedule recorded a slice at "
                "this step and will gate it out for another drain tick")
            # ... and that is the harm, stated as the cap the next offer gets.
            assert selling.participation_cap(view, item, st, cfg) > 0, (
                f"{item} is gated to zero by a slice that was never offered")


def test_melons_sold_is_counted_once_from_our_own_orders():
    """The shipped builder increments memory.melons_sold for ITS sizing; we
    replace those orders, so the counter has to be re-derived or the planner
    (planner.py:281) and the feature encoder both read a double count."""
    v = make_view(step=planner.FINAL_STEP - 3, shed={"MELON": 30}, shops=tuple(SHOPS))
    mem = Memory()
    got = selling.build_orders(v, make_plan({"MELON": 1}), mem)
    sold = sum(o[2] for o in got if o[0] == "SELL" and o[1] == "MELON")
    assert mem.melons_sold == sold
    selling.build_orders(make_view(step=planner.FINAL_STEP - 2, shed={"MELON": 30},
                                   shops=tuple(SHOPS)), make_plan({"MELON": 1}), mem)
    assert mem.melons_sold == 2 * sold


def test_disabled_config_keeps_the_shipped_sell_sizes():
    v = make_view(step=8 * 24 + 4, shed={"STRAWBERRY": 40, "WOOL": 20}, money=50_000)
    floors = {"STRAWBERRY": 80, "WOOL": 120}
    base = market.build_orders(v, make_plan(floors), Memory())
    got = selling.build_orders(v, make_plan(floors), Memory(),
                               SellConfig(enabled=False))
    assert sorted((o[1], o[2]) for o in base if o[0] == "SELL") == \
           sorted((o[1], o[2]) for o in got if o[0] == "SELL")


def test_install_is_visible_on_the_dev_path():
    """kagent.policy holds the MODULE object, so patching the attribute is
    reached by every dev-path arm. If policy.py ever switches to a flat import
    this breaks loudly instead of an arm silently running the shipped sizer."""
    import kagent.policy as kpolicy
    assert kpolicy.market is market
    restore = selling.install(SellConfig())
    try:
        assert kpolicy.market.build_orders.__module__.startswith("rlv3")
    finally:
        restore()


def test_install_patches_and_restores_kagent_market():
    original = market.build_orders
    restore = selling.install(SellConfig())
    try:
        assert market.build_orders is not original
    finally:
        restore()
    assert market.build_orders is original


# =============================================================== sell_eval
def _synthetic_trace(tmp_path, steps=48, my_units=6, opp_units=4):
    """A stylised market that never crashes: the recorded inventory oscillates
    around I0 so the crashed-release bypass stays off and the schedule is what
    the test is actually measuring.

    Our recorded SELL sits behind a HIRE at slot 0 and the opponent's leads,
    which is champ-h10's real shape -- it emits HIRE then the buys and lets
    sells "fill the remaining slots", so our sells are quoted last all season.
    """
    out = []
    shed = {"STRAWBERRY": 60}
    for t in range(steps):
        day, hour = t // 24, t % 24
        inv = {p: I0 for p in PRODUCTS_L}
        inv["STRAWBERRY"] = I0 + (t % 8)
        out.append({
            "t": t, "day": day, "hour": hour,
            "market": {"prices": [econ.price(p, inv[p]) for p in PRODUCTS_L],
                       "inv": [inv[p] for p in PRODUCTS_L],
                       "shops": ["BRUNCH_SPOT", "ICE_CREAM_SHOP"]},
            "shed": dict(shed),
            "act": {"me": {"market": [["HIRE"], ["SELL", "STRAWBERRY", my_units]]},
                    "opp": {"market": [["SELL", "STRAWBERRY", opp_units]]}},
        })
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({
        "products": PRODUCTS_L, "final": {"me": 50_000, "opp": 48_000,
                                          "win": True, "margin": 2_000},
        "steps": out}))
    return str(path)


def test_opponent_fills_are_recovered_exactly_from_the_inventory_identity(tmp_path):
    """A recorded order is an OFFER and their shed is not observable, so their
    revenue would otherwise be an upper bound inflated by the over-offer -- and
    the kill rule reads off exactly that column. inv[t+1] = inv[t] + my_fills +
    their_fills - drain closes it for every product that cannot be bought back
    (everything but WHEAT and FERTILIZER).

    FIXTURE CORRECTED 2026-09-27 (`sell-eval-drain-off-by-one`). It used to
    step the inventory with `mm.drain_between("WOOL", t - 1, t, shops)`, the
    same window `attach_opponent_fills` called, so the fixture was a copy of
    the bug and the test was a tautology: it could only ever confirm that the
    code agreed with itself. The path is now stepped by `_brute_drain(t, t+1)`
    -- the engine's own `step % interval == 0` rule, which is what closes
    `inv[t] -> inv[t+1]`.
    """
    shops = ["YARN_STORE"]
    my_fill, their_fill, their_offer = 3, 5, 40      # they over-offer 8x
    inv, out = I0, []
    for t in range(12):
        out.append({
            "t": t, "day": t // 24, "hour": t % 24,
            "market": {"prices": [econ.price(p, I0) for p in PRODUCTS_L],
                       "inv": [inv if p == "WOOL" else I0 for p in PRODUCTS_L],
                       "shops": list(shops)},
            "shed": {"WOOL": 50},
            "act": {"me": {"market": [["SELL", "WOOL", my_fill]]},
                    "opp": {"market": [["SELL", "WOOL", their_offer]]}},
        })
        inv += my_fill + their_fill - int(_brute_drain("WOOL", t, t + 1, shops))
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"products": PRODUCTS_L, "final": {}, "steps": out}))
    steps, _ = sell_eval.load_steps(str(path))
    got = [s["opp_fill"]["WOOL"] for s in steps[:-1]]
    assert got == [their_fill] * (len(out) - 1), got
    assert steps[-1]["opp_fill"]["WOOL"] == their_offer   # no identity left


@pytest.mark.parametrize("t,want_drain", [
    (4, 2),      # plain shop tick: fires DURING step 4, none during step 3
    (24, 3),     # compound tick at hour 0: two shop units plus the town centre
])
def test_opponent_fill_uses_the_tick_that_fires_during_the_step(t, want_drain):
    """REGRESSION, `sell-eval-drain-off-by-one` (rlv3/sell_eval.py:118).

    `attach_opponent_fills` closed `inv[t+1] = inv[t] + fills - drain - buys`
    with `drain_between(t-1, t)` -- the tick that fired during the PREVIOUS
    step -- instead of `drain_between(t, t+1)`, the tick visible between the
    two observations it is differencing. Engine `interpreter` runs
    `_town_consume(env, state, step)` with the pre-increment counter and it
    drains on `step % interval == 0`, so the tick fired during step t is first
    visible at t+1. On 5,814 untraded product-steps of champ_h10_0_b0 the
    shipped window mismatched 1,706 times and `(t, t+1)` mismatched 0 times.

    WHY THE ORIGINAL TEST MISSED IT: the only other test of this identity,
    test_opponent_fills_are_recovered_exactly_from_the_inventory_identity,
    BUILT its inventory path by calling the same `drain_between(t-1, t)` the
    code under test called. Fixture and code shared the error, so the
    assertion held for every offset. This test never names a window: it
    states the fill, states the engine's tick, and lets the solver find it.

    Both steps are chosen where the two conventions disagree -- no tick fires
    during t-1 at either -- so the wrong window drops the whole tick and the
    solved fill comes back short by `want_drain`.
    """
    shops = ["YARN_STORE"]
    their_fill, their_offer = 5, 10               # they over-offer 2x
    assert _brute_drain("WOOL", t, t + 1, shops) == want_drain
    assert _brute_drain("WOOL", t - 1, t, shops) == 0

    def step(k, inv, opp):
        return {"t": k, "day": k // 24, "hour": k % 24, "shops": list(shops),
                "inv": {"WOOL": inv}, "shed": {}, "me": [], "opp": opp,
                "my_buys": {}, "opp_buys": {}}

    steps = [step(t, I0, [("WOOL", their_offer, 0)]),
             step(t + 1, I0 + their_fill - want_drain, [])]
    sell_eval.attach_opponent_fills(steps)
    assert steps[0]["opp_fill"]["WOOL"] == their_fill


def _reference_opp_fills(steps, lag):
    """attach_opponent_fills re-derived from the ENGINE rule via _brute_drain,
    with the drain window shifted by `lag` steps. lag=0 is the interval
    between the two observations; lag=1 is the shipped off-by-one."""
    total = {}
    for k, s in enumerate(steps[:-1]):
        my_fill = {}
        for item, n, _ in s["me"]:
            my_fill[item] = my_fill.get(item, 0) + min(n, int(s["shed"].get(item, 0)))
        offers = {}
        for item, n, _ in s["opp"]:
            offers[item] = offers.get(item, 0) + n
        for item, off in offers.items():
            d_inv = int(steps[k + 1]["inv"][item]) - int(s["inv"][item])
            drain = _brute_drain(item, s["t"] - lag, s["t"] + 1 - lag, s["shops"])
            buys = s["my_buys"].get(item, 0) + s["opp_buys"].get(item, 0)
            exact = d_inv + drain + buys - my_fill.get(item, 0)
            total[item] = total.get(item, 0) + int(max(0, min(off, round(exact))))
    return total


def test_opponent_fill_identity_closes_on_a_recorded_trace():
    """The same regression on real data, where it was demonstrated.

    Two independent checks, neither of which calls `mm.drain_between`:

    1. On every (product, step) at which NEITHER player sent an order for that
       product there is no fill to solve for, so the inventory delta must be
       exactly minus the town's drain. The shipped `(t-1, t)` window failed
       that on 1,706 of 5,814 such rows on this trace; `(t, t+1)` closes all
       5,814.
    2. The solver's own `opp_fill` column must equal a reference solved from
       the engine's tick rule -- and must NOT equal the one the off-by-one
       window produces, which is asserted separately so this test cannot pass
       by both sides being wrong together (the failure mode of the original).
    """
    path = os.path.join(REPO, "rlv3", "eval", "scorecard_traces",
                        "champ_h10_0_b0.json")
    if not os.path.exists(path):
        pytest.skip("no recorded scorecard trace on disk")
    steps, _ = sell_eval.load_steps(path)

    quiet = bad = 0
    for k, s in enumerate(steps[:-1]):
        traded = ({i for i, _, _ in s["me"]} | {i for i, _, _ in s["opp"]}
                  | set(s["my_buys"]) | set(s["opp_buys"]))
        for item in PRODUCTS_L:
            if item in traded:
                continue
            quiet += 1
            d_inv = int(steps[k + 1]["inv"][item]) - int(s["inv"][item])
            if round(d_inv + _brute_drain(item, s["t"], s["t"] + 1,
                                          s["shops"])) != 0:
                bad += 1
    assert quiet > 1000, quiet
    assert bad == 0, f"{bad}/{quiet} untraded product-steps do not close"

    want = _reference_opp_fills(steps, lag=0)
    off_by_one = _reference_opp_fills(steps, lag=1)
    assert want != off_by_one, "this trace does not discriminate the two windows"
    got = {}
    for s in steps[:-1]:
        for item, n in s["opp_fill"].items():
            got[item] = got.get(item, 0) + n
    assert got == want, {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                         if got.get(k) != want.get(k)}
    assert sum(want.values()) > sum(off_by_one.values())   # it UNDER-states them


def test_opponent_fills_are_clipped_to_the_offer():
    """The identity can only ever attribute what they actually asked for."""
    steps = [
        {"t": 4, "day": 0, "hour": 4, "shops": ["YARN_STORE"],
         "inv": {"WOOL": I0}, "shed": {}, "me": [], "opp": [("WOOL", 2, 0)],
         "my_buys": {}, "opp_buys": {}},
        {"t": 5, "day": 0, "hour": 5, "shops": ["YARN_STORE"],
         "inv": {"WOOL": I0 + 900}, "shed": {}, "me": [], "opp": [],
         "my_buys": {}, "opp_buys": {}},
    ]
    sell_eval.attach_opponent_fills(steps)
    assert steps[0]["opp_fill"]["WOOL"] == 2


def test_counterfactual_is_an_identity_under_the_recorded_sizer(tmp_path):
    """If the pre-gate does not read exactly zero when nothing changed, its
    $1,500/game threshold means nothing."""
    path = _synthetic_trace(tmp_path)
    got = sell_eval.counterfactual_revenue(path, sizer=sell_eval.shipped_sizer,
                                           slot_order=False)
    assert got["delta"] == pytest.approx(0.0)
    assert got["opp_delta"] == pytest.approx(0.0)
    assert got["shipped_units"] == got["rlv3_units"] > 0


def test_counterfactual_reports_the_opponent_side(tmp_path):
    """The KILL RULE: the market is ONE shared inventory. Withdrawing supply
    lifts their realised price by the same identity -- that is how evesell
    lost."""
    path = _synthetic_trace(tmp_path)
    got = sell_eval.counterfactual_revenue(path)
    assert "opp_delta" in got and "margin_delta" in got
    assert got["margin_delta"] == pytest.approx(got["delta"] - got["opp_delta"])
    assert got["rlv3_units"] <= got["shipped_units"]


def test_counterfactual_slice_gap_beats_the_shipped_schedule(tmp_path):
    path = _synthetic_trace(tmp_path)
    got = sell_eval.counterfactual_revenue(path)
    assert got["shipped_slice_gap_steps"] == 1
    assert got["slice_gap_steps"] >= 4


def test_slice_gap_steps_is_the_median_over_items():
    assert sell_eval.slice_gap_steps({"A": [0, 4, 8], "B": [1, 5]}) == 4
    assert sell_eval.slice_gap_steps({"A": [0, 1, 2, 3]}) == 1
    assert sell_eval.slice_gap_steps({"A": []}) is None


def test_relative_sell_metrics_are_relative_by_construction(tmp_path):
    path = _synthetic_trace(tmp_path, my_units=6, opp_units=4)
    got = sell_eval.relative_sell_metrics([path])
    m = got["mean"]
    assert 0.0 < m["revenue_share"] < 1.0
    assert m["our_units"] > m["opp_units"]
    assert m["revenue_share"] > 0.5
    assert m["opp_realised_over_mean"] > 0


def test_verdict_kills_a_sub_threshold_delta():
    def row(**kw):
        base = {"delta": 9_000.0, "opp_delta": 0.0, "delta_pct": 0.06,
                "opp_delta_pct": 0.0, "slice_gap_steps": 6, "margin_delta": 9_000.0,
                "slots_only_margin_delta": 0.0}
        base.update(kw)
        return [base]

    assert "margin delta" in sell_eval._verdict(row(margin_delta=200.0))
    assert "collapsed back into sell_pace" in sell_eval._verdict(row(slice_gap_steps=1))
    assert "PASSED" in sell_eval._verdict(row())


def test_verdict_kills_a_shared_curve_lift_even_when_our_bank_rises():
    """THE kill rule, in the form that survives the offer-vs-fill caveat: a
    percentage lift is invariant to how much the opponent over-offers. If their
    realised price rises at least as fast as ours, the schedule moved ONE
    shared curve and they sell into it too. That is how evesell lost."""
    leaky = [{"delta": 40_000.0, "opp_delta": 38_000.0, "margin_delta": 2_000.0,
              "delta_pct": 0.06, "opp_delta_pct": 0.07, "slice_gap_steps": 9,
              "slots_only_margin_delta": 0.0}]
    v = sell_eval._verdict(leaky)
    assert "DO NOT RUN" in v and "ONE shared curve" in v


def test_slot_reorder_is_measured_on_its_own(tmp_path):
    """The slot move withdraws no supply -- same sizes, same volume, same
    walking -- so it is the only part of this component that cannot lift the
    opponent's price. It has to be reported separately from the schedule."""
    path = _synthetic_trace(tmp_path)
    got = sell_eval.counterfactual_revenue(path)
    for k in ("slots_only_delta", "slots_only_opp_delta", "slots_only_margin_delta",
              "delta_pct", "opp_delta_pct"):
        assert k in got
    assert got["slots_only_margin_delta"] == pytest.approx(
        got["slots_only_delta"] - got["slots_only_opp_delta"])
    # Moving ahead of a seller who was quoted first is a pure transfer: we gain
    # and they lose, with no change in total units.
    assert got["slots_only_delta"] > 0 > got["slots_only_opp_delta"]


def test_recommended_config_is_the_one_the_pregate_supports():
    """rlv3/sell_eval.py on 2 champ-h10 traces, re-measured 2026-09-27 with the
    engine's same-slot lockstep: the sizing half is margin +$656/game against
    the slot move's +$772, i.e. it is margin-NEGATIVE on top of the free part.
    RECOMMENDED must carry only the slot move. (The pre-lockstep figures were
    +$891 vs +$1,281; the ordering is what this test pins, and it is the same.)"""
    assert selling.RECOMMENDED.slots_only is True
    v = make_view(step=8 * 24 + 4, shed={"STRAWBERRY": 40, "WOOL": 20},
                  money=200_000, shops=("BRUNCH_SPOT", "YARN_STORE"))
    floors = {"STRAWBERRY": 80, "WOOL": 120}
    base = market.build_orders(v, make_plan(floors, plant_quota={"WHEAT": 6}), Memory())
    got = selling.build_orders(v, make_plan(floors, plant_quota={"WHEAT": 6}),
                               Memory(), selling.RECOMMENDED)
    assert sorted((o[1], o[2]) for o in base if o[0] == "SELL") == \
           sorted((o[1], o[2]) for o in got if o[0] == "SELL"), "sizes must be shipped"
    assert base[0][0] == "BUY_SEED" and got[0][0] == "SELL", "only the slot moves"
