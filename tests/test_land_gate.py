"""KAGG_MAX_QUADRANTS: the land head's buy choice is masked AND the crunch
fallback is suppressed once the cap is reached; unset, nothing changes.

    PYTHONPATH=. uv run pytest tests/test_land_gate.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import features, planner  # noqa: E402
from kagent.constants import LAND_PRICES  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _fresh_obs():
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 21}, debug=False)
    env.reset(2)
    return env.state[0].observation


def _rich_crunch_obs():
    """One quadrant owned, no idle tiles (all weeds), plenty of money: the
    state in which apply_to_plan's crunch fallback buys land regardless of
    the head."""
    obs = copy.deepcopy(_fresh_obs())
    farm = obs["farms"][0]
    farm["money"] = LAND_PRICES[0] + 5000
    for y, row in enumerate(farm["tiles"]):
        for x, c in enumerate(row):
            if c is None:
                row[x] = {"kind": "WEED"}
    return obs


def _choices(land):
    idx = []
    for name, vals in features.HEADS:
        idx.append(vals.index(land) if name == "land" else 0)
    return idx


def _land_mask(view):
    return [m for (name, _), m in zip(features.HEADS, features.masks(view)) if name == "land"][0]


def test_no_cap_is_unchanged(monkeypatch):
    monkeypatch.setattr(features, "MAX_QUADRANTS", 0)
    obs = _rich_crunch_obs()
    view = View(obs)
    assert len(view.me.unlocked_quadrants) == 1
    assert _land_mask(view) == [1, 1], "affordable next quadrant: both choices valid"
    plan = planner.make_day_plan(view, Memory())
    features.apply_to_plan(plan, _choices(1), view)
    assert plan.land_buys, "head says buy -> buys"
    plan = planner.make_day_plan(view, Memory())
    features.apply_to_plan(plan, _choices(0), view)
    assert plan.land_buys, "head says no, but the crunch fallback buys anyway"


def test_cap_masks_the_head_and_the_crunch(monkeypatch):
    monkeypatch.setattr(features, "MAX_QUADRANTS", 1)      # already at the cap
    view = View(_rich_crunch_obs())
    assert _land_mask(view) == [1, 0], "buy is unsamplable at the cap"
    for land in (0, 1):
        plan = planner.make_day_plan(view, Memory())
        features.apply_to_plan(plan, _choices(land), view)
        assert plan.land_buys == [], f"no purchase at the cap (head={land})"


def test_cap_above_ownership_still_allows(monkeypatch):
    monkeypatch.setattr(features, "MAX_QUADRANTS", 3)      # one owned, cap three
    view = View(_rich_crunch_obs())
    assert _land_mask(view) == [1, 1]
    plan = planner.make_day_plan(view, Memory())
    features.apply_to_plan(plan, _choices(1), view)
    assert plan.land_buys


def _midseason_pressure_obs():
    """Day 10, one quadrant, no idle tiles, rich: inside apply_land_pressure's
    window (days 2-24). The first gate test used day 0 and never exercised
    this path, which is how Stage B5 ran with the leak live."""
    obs = _rich_crunch_obs()
    obs["day"] = 10
    obs["step"] = 240
    obs["hour"] = 0
    return obs


def test_no_cap_land_pressure_still_buys(monkeypatch):
    monkeypatch.setattr(features, "MAX_QUADRANTS", 0)
    view = View(_midseason_pressure_obs())
    plan = planner.make_day_plan(view, Memory())
    features.apply_to_plan(plan, _choices(0), view)
    assert plan.land_buys, "without a cap the pressure net buys despite head=0"


def test_cap_closes_the_land_pressure_path(monkeypatch):
    monkeypatch.setattr(features, "MAX_QUADRANTS", 1)
    view = View(_midseason_pressure_obs())
    for land in (0, 1):
        plan = planner.make_day_plan(view, Memory())
        features.apply_to_plan(plan, _choices(land), view)
        assert plan.land_buys == [], f"pressure net must not buy at the cap (head={land})"


def test_env_switch_is_read_at_import(monkeypatch):
    import importlib
    monkeypatch.setenv("KAGG_MAX_QUADRANTS", "3")
    mod = importlib.reload(features)
    try:
        assert mod.MAX_QUADRANTS == 3
    finally:
        monkeypatch.delenv("KAGG_MAX_QUADRANTS")
        importlib.reload(features)
        assert features.MAX_QUADRANTS == 0
