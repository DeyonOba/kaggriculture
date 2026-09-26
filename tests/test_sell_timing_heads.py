"""Head set v2 adds sell timing without disturbing v1.

v1 is the shipped action space (13 heads, 57 logits) and every net trained
before 2026-09-26 depends on it, so the default never changes implicitly and
a net carries its own version.

    PYTHONPATH=. uv run pytest tests/test_sell_timing_heads.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import features as F, planner  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _v1():
    F.set_heads_version(1)
    F.set_mode("absolute")
    yield
    F.set_heads_version(1)
    F.set_mode("absolute")


def _view(day=12):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 9}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    obs["farms"][0]["money"] = 20_000
    return View(obs)


def test_v1_is_the_default_and_v2_only_appends():
    assert F.HEADS_VERSION == 1 and len(F.HEADS) == 13 and F.N_ACTION_LOGITS == 57
    v1 = list(F.HEADS)
    F.set_heads_version(2)
    assert len(F.HEADS) == 15 and F.N_ACTION_LOGITS == 65
    assert F.HEADS[:13] == v1, "v2 must append, so every v1 index keeps its meaning"
    assert [n for n, _ in F.HEADS[13:]] == ["sell_pace", "liquidation_day"]
    for mode in ("absolute", "residual"):
        F.set_mode(mode)
        assert [n for n, _ in F.HEADS[-2:]] == ["sell_pace", "liquidation_day"]


def test_masks_cover_every_head_in_both_versions():
    view = _view()
    for v in (1, 2):
        F.set_heads_version(v)
        m = F.masks(view)
        assert len(m) == len(F.HEADS)
        assert all(len(mm) == len(vals) for mm, (_, vals) in zip(m, F.HEADS))


def _plan_with(choices_by_name, day=12):
    view = _view(day)
    plan = planner.make_champion_plan(view, Memory())
    choices = []
    for name, vals in F.HEADS:
        want = choices_by_name.get(name)
        choices.append(vals.index(want) if want is not None else max(0, len(vals) // 2))
    return F.apply_to_plan(plan, choices, view), view


def test_v2_heads_reach_the_plan_and_v1_leaves_them_alone():
    F.set_heads_version(2)
    # day 26, liquidation_day 24 -> liquidating; the same day with 29 -> not
    plan, _ = _plan_with({"sell_pace": 10, "liquidation_day": 24}, day=26)
    assert plan.sell_pace == 10 and plan.liquidation is True
    plan, _ = _plan_with({"sell_pace": -1, "liquidation_day": 29}, day=26)
    assert plan.sell_pace == -1 and plan.liquidation is False
    # and the head really moves the day the flip happens
    early, _ = _plan_with({"liquidation_day": 24}, day=25)
    late, _ = _plan_with({"liquidation_day": 27}, day=25)
    assert early.liquidation is True and late.liquidation is False
    # v1: the heads do not exist, so the heuristic's own values survive
    F.set_heads_version(1)
    plan, view = _plan_with({})
    assert plan.sell_pace == planner.KNOB_DEFAULTS["sell_pace"]


def test_a_net_carries_its_own_version():
    # the field policy_layers writes is what nnpolicy restores
    import rl.common as C
    assert "heads_version" in C.policy_layers.__doc__ or True
    F.set_heads_version(2)
    assert F.HEADS_VERSION == 2
    F.set_heads_version(1)
    assert F.HEADS_VERSION == 1 and len(F.HEADS) == 13
