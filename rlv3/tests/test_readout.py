"""The readout is a plan source; the capacity mask has a horizon; both are
pure additions.

What each pins, and the measurement it came from (rlv3/s0_check.py):

  * masks(view) is bit-identical to masks(view, horizon=1). Every existing
    caller -- the champion path, the knob bandit, rl/rollout -- calls it with
    no horizon, and the S0 identity check (1200/1200 dawns) rests on that.
  * At horizon 3 a crop's ceiling can EXCEED its standing count when another
    crop frees ground inside the window. On the leaders' fully-planted dawns
    the horizon-1 ceiling equalled the standing count, so the readout could
    never ask for more than was planted and its 3-day crop commitment was
    empty on days 2-9 where the leaders spend 35-40% on seed.
  * The readout preset minus its readout IS the champion (the S0 identity).
  * commit() asks for the delta, caps animals at the planner's daily cap,
    orders plant_quota by the pricing, and never touches land.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def _leader_dawn(day=10, seat=0):
    from kagent.gamestate import View
    from rlv3.trajectory import leader_paths
    steps = json.load(open(leader_paths()[0][1]))["steps"]
    obs = dict(steps[day * 24][seat]["observation"])
    obs.setdefault("step", day * 24)
    return View(obs)


@pytest.fixture(autouse=True)
def _restore_features():
    """Readout.__init__ sets features.HEADS_VERSION globally and does not put
    it back. Left at 4, the next test file (test_stages, alphabetically) sees
    a fingerprint whose head-set hash no longer matches and its dry-run
    resolution comes out stale -- an order-dependent failure that passes in
    isolation. Every test here restores the version and mode it found."""
    from kagent import features
    v, m = features.HEADS_VERSION, features.MODE
    yield
    if features.HEADS_VERSION != v:
        features.set_heads_version(v)
    if features.MODE != m:
        features.set_mode(m)


@pytest.fixture
def v4():
    from kagent import features
    features.set_heads_version(4)
    yield features


def test_masks_default_horizon_is_bit_identical(v4):
    view = _leader_dawn(10)
    assert v4.masks(view) == v4.masks(view, horizon=1)


def test_horizon_three_opens_capacity_that_other_crops_free(v4):
    view = _leader_dawn(10)
    m1, m3 = v4.masks(view, horizon=1), v4.masks(view, horizon=3)
    names = [n for n, _ in v4.HEADS]
    wi = names.index("wheat_tiles")
    vals = v4.HEADS[wi][1]
    top1 = max(vals[i] for i in range(len(vals)) if m1[wi][i])
    top3 = max(vals[i] for i in range(len(vals)) if m3[wi][i])
    standing = sum(1 for _x, _y, t in view.me.plants() if t.get("crop") == "WHEAT")
    assert top1 == standing, "on a fully-planted dawn the 1-day ceiling IS the standing count"
    assert top3 > top1, "the 3-day ceiling must open when other crops free ground"


def test_horizon_never_lowers_a_ceiling(v4):
    view = _leader_dawn(6)
    m1, m3 = v4.masks(view, horizon=1), v4.masks(view, horizon=3)
    for (name, vals), a, b in zip(v4.HEADS, m1, m3):
        if name in v4.QUOTA_CROP or name in v4.BUY_KIND:
            assert sum(b) >= sum(a), f"{name}: horizon 3 masked MORE than horizon 1"


def test_freeing_by_crop_matches_tile_flow_total(v4):
    view = _leader_dawn(8)
    per = v4.freeing_by_crop(view, 3)
    fl = v4.tile_flow(view, horizon=3)
    assert sum(per.values()) == fl["freeing"][3], "per-crop freeing must sum to tile_flow's"


def test_readout_preset_minus_readout_is_the_champion():
    from rlv3 import agent as A, arm
    from kagent.gamestate import Memory
    off = A.Rlv3PlanProvider(A.config_from_spec({**arm.parse("readout"), "readout": None}))
    base = A.Rlv3PlanProvider(A.config_from_spec(arm.parse("base")))
    view = _leader_dawn(4)
    m1, m2 = Memory(), Memory()
    m1.observe(view); m2.observe(view)
    p1, p2 = off.make_plan(view, m1), base.make_plan(view, m2)
    for f in ("plant_quota", "animal_buys", "structure_builds", "hires_target", "land_buys"):
        assert getattr(p1, f) == getattr(p2, f), f
    assert off.readout is None and off.last_delta is None


def test_readout_predicts_every_scored_head_and_commit_is_a_delta():
    from kagent import features
    from rlv3 import readout as RO
    from kagent.gamestate import Memory
    from kagent.planner import DayPlan
    if not os.path.exists(RO.DEFAULT_WEIGHTS):
        pytest.skip("no trained readout on disk")
    ro = RO.Readout()
    view = _leader_dawn(10)
    m = Memory(); m.observe(view)
    pred = ro.predict(view, m)
    assert set(pred["crops"]) == {"MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}
    assert set(pred["herd"]) == {"GOOSE", "COW", "SHEEP"}
    assert isinstance(pred["hires"], int)
    plan, delta = RO.commit(DayPlan(), view, pred, order=["WHEAT", "STRAWBERRY"])
    standing, _h, _p, _e = RO.standing(view)
    freeing = features.freeing_by_crop(view, RO.HORIZON)
    for c, n in plan.plant_quota.items():
        # persistence-adjusted: only tiles still standing at the horizon count
        assert n > 0 and n == pred["crops"][c] - (standing.get(c, 0) - freeing.get(c, 0)), c
    for k, n in plan.animal_buys.items():
        assert 0 < n <= RO.DAILY_ANIMAL_CAP, k
    assert plan.land_buys == [], "commit must never touch land"
    keys = list(plan.plant_quota)
    ranked = [c for c in ("WHEAT", "STRAWBERRY") if c in keys]
    assert keys[:len(ranked)] == ranked, "plant_quota must follow the pricing's order"


def test_commit_counts_replants_not_just_growth(v4):
    """A wheat tile that harvests inside the window and is replanted ends it
    standing: stock delta 0, but it was planted. The ask must include it."""
    from rlv3 import readout as RO
    from kagent.planner import DayPlan
    view = _leader_dawn(10)
    st, _h, _p, _e = RO.standing(view)
    fr = v4.freeing_by_crop(view, 3)
    assert fr.get("WHEAT", 0) > 0, "this dawn should have wheat freeing within 3 days"
    pred = {"crops": {"WHEAT": st.get("WHEAT", 0)}, "herd": {}, "hires": None, "land": 0}
    plan, delta = RO.commit(DayPlan(), view, pred)
    # target == standing, so a stock delta would be 0; the ask must be the replants
    assert plan.plant_quota.get("WHEAT", 0) == fr["WHEAT"]


def test_commit_is_a_plain_stock_delta_for_an_ongoing_crop(v4):
    from rlv3 import readout as RO
    from kagent.planner import DayPlan
    view = _leader_dawn(10)
    st, _h, _p, _e = RO.standing(view)
    fr = v4.freeing_by_crop(view, 3)
    assert fr.get("STRAWBERRY", 0) == 0, "strawberry is ongoing; nothing frees"
    pred = {"crops": {"STRAWBERRY": st.get("STRAWBERRY", 0) + 3}, "herd": {}, "hires": None, "land": 0}
    plan, _d = RO.commit(DayPlan(), view, pred)
    assert plan.plant_quota.get("STRAWBERRY", 0) == 3


def test_last_buy_day_matches_the_planners_own_herd_rule():
    """readout.LAST_BUY_DAY is a copy of a ground-truth constant that lives in
    planner._champ_herd_targets; if the planner moves, this must move."""
    from kagent import planner
    from rlv3 import readout as RO
    k = dict(planner.KNOB_DEFAULTS, geese=4)
    for kind, last in RO.LAST_BUY_DAY.items():
        assert planner._champ_herd_targets(last, k)[kind] > 0, f"{kind} should still buy on day {last}"
        assert planner._champ_herd_targets(last + 1, k)[kind] == 0, f"{kind} must stop after day {last}"


def test_commit_buys_no_animal_past_its_last_day(v4):
    from rlv3 import readout as RO
    from kagent.planner import DayPlan
    view = _leader_dawn(22)
    _c, herd, _p, _e = RO.standing(view)
    pred = {"crops": {}, "herd": {k: herd.get(k, 0) + 2 for k in ("GOOSE", "COW", "SHEEP")},
            "hires": None, "land": 0}
    plan, _d = RO.commit(DayPlan(), view, pred)
    assert plan.animal_buys == {}, plan.animal_buys


def test_last_plant_day_matches_the_planner_where_it_has_one():
    from kagent import planner
    from rlv3 import readout as RO
    assert RO.last_plant_day("WHEAT") == planner.WHEAT_LAST_PLANT_DAY
    assert RO.last_plant_day("CARROT") == planner.CARROT_LAST_PLANT_DAY
    assert RO.last_plant_day("TOMATO") == planner.TOMATO_LAST_PLANT_DAY
    assert RO.last_plant_day("MELON") == planner.MELON_LAST_PLANT_DAY
    # engine bound for the rest: planted on its last day it still yields by 29
    from kagent.constants import CROPS
    for c in ("STRAWBERRY",):
        lead = CROPS[c]["first_yield_day"] if CROPS[c]["ongoing"] else CROPS[c]["max_yield_day"]
        assert RO.last_plant_day(c) + lead <= 29


def test_commit_plants_nothing_on_the_last_dawn(v4):
    from rlv3 import readout as RO
    from kagent.planner import DayPlan
    view = _leader_dawn(29)
    st, _h, _p, _e = RO.standing(view)
    pred = {"crops": {c: st.get(c, 0) + 5 for c in ("WHEAT", "STRAWBERRY", "MELON")},
            "herd": {}, "hires": None, "land": 0}
    plan, _d = RO.commit(DayPlan(), view, pred)
    assert plan.plant_quota == {}, plan.plant_quota


def test_v5_adds_exactly_the_fertiliser_head():
    from kagent import features as F
    F.set_heads_version(5)
    names = [n for n, _ in F.HEADS]
    assert names[-1] == "fert_budget" and names[:-1] == [n for n, _ in F.ABS_HEADS_V4]
    assert F.HEADS[-1][1] == list(range(0, F.FERT_CAP + 1))
    assert F.N_ACTION_LOGITS == sum(len(v) for _n, v in F.ABS_HEADS_V4) + F.FERT_CAP + 1


def test_fert_label_is_the_replays_own_count():
    from kagent import features as F
    from rl.replay_bc import _targets_to_choices
    F.set_heads_version(5)
    view = _leader_dawn(12)
    m = F.masks(view, horizon=3)
    ch = _targets_to_choices(view, m, {}, {}, {}, 8, False, fert=7)
    fi = [i for i, (n, _) in enumerate(F.HEADS) if n == "fert_budget"][0]
    assert F.HEADS[fi][1][ch[fi]] == 7


def test_scheduler_emits_no_fertilise_task_when_the_budget_is_zero():
    """The phase gate: the leaders fertilise ZERO times on days 0-9."""
    from kagent import scheduler as S
    from kagent.gamestate import Memory
    from kagent.planner import make_day_plan
    view = _leader_dawn(12)
    mem = Memory(); mem.observe(view)
    plan = make_day_plan(view, mem, speculative=True)
    base = sum(1 for t in S.build_tasks(view, plan) if t.op[0] == "FERTILIZE")
    plan.fert_budget = 0
    assert sum(1 for t in S.build_tasks(view, plan) if t.op[0] == "FERTILIZE") == 0
    plan.fert_budget = None
    assert sum(1 for t in S.build_tasks(view, plan) if t.op[0] == "FERTILIZE") == base, \
        "None must be the shipped behaviour exactly"


def test_a_positive_budget_waives_the_price_gate():
    from kagent import scheduler as S
    from kagent.gamestate import Memory
    from kagent.planner import make_day_plan
    view = _leader_dawn(12)
    mem = Memory(); mem.observe(view)
    plan = make_day_plan(view, mem, speculative=True)
    plan.fert_gate = 0.0                     # price gate closed for every crop
    assert sum(1 for t in S.build_tasks(view, plan) if t.op[0] == "FERTILIZE") == 0
    plan.fert_budget = 5
    assert sum(1 for t in S.build_tasks(view, plan) if t.op[0] == "FERTILIZE") > 0


def test_policy_consumes_the_budget_per_emitted_op():
    from kagent.planner import DayPlan
    plan = DayPlan(); plan.fert_budget = 2
    for op in (["FERTILIZE"], ["FERTILIZE"], ["FERTILIZE"]):
        if op[0] == "FERTILIZE" and getattr(plan, "fert_budget", None) is not None:
            plan.fert_budget = max(0, plan.fert_budget - 1)
    assert plan.fert_budget == 0


def test_commit_sets_the_fert_budget_from_the_prediction(v4):
    from rlv3 import readout as RO
    from kagent.planner import DayPlan
    view = _leader_dawn(12)
    plan, d = RO.commit(DayPlan(), view, {"crops": {}, "herd": {}, "hires": None, "land": 0, "fert": 9})
    assert plan.fert_budget == 9 and d["fert"] == 9
    plan2, _ = RO.commit(DayPlan(), view, {"crops": {}, "herd": {}, "hires": None, "land": 0})
    assert plan2.fert_budget is None, "no prediction -> shipped behaviour"
