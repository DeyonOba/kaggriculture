"""v3 frames the day plan as a FLOW decision: demand priced at a quantile of
the remaining shop draw, ground that frees within the horizon, our pending
supply against the opponent's, and a quota bounded by real capacity.

v1 and v2 must be untouched — every trained net depends on both the
observation width and the head split.

    PYTHONPATH=. uv run pytest tests/test_flow_framing.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import econ, features as F, planner  # noqa: E402
from kagent.constants import SHOPS  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _v1():
    F.set_heads_version(1); F.set_mode("absolute")
    yield
    F.set_heads_version(1); F.set_mode("absolute")


def _plant(crop, planted_day, mls=-1, held=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day, "watered_today": True,
            "consecutive_unwatered": 0, "yield_units": held, "max_lifespan_step": mls,
            "fertilized_until_day": -1}


def _view(day=12, money=20_000, tiles=None):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 9}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    obs["farms"][0]["money"] = money
    for (x, y), t in (tiles or {}).items():
        obs["farms"][0]["tiles"][y][x] = t
    return View(obs)


# ---------------------------------------------------------------- risk
def test_quantile_is_conservative_and_product_specific():
    shops, day = ["SMOOTHIE_SHOP"], 3
    mean = econ.forecast_drain(shops, day)
    q = econ.forecast_drain(shops, day, quantile=0.35)
    # wool lives in ONE shop of eight, wheat in five: the risk discount must
    # be far harsher on wool
    assert q["WOOL"] / mean["WOOL"] < 0.4
    assert q["WHEAT"] / mean["WHEAT"] > 0.8
    assert all(q[p] <= mean[p] + 1e-9 for p in ("WOOL", "WHEAT", "CARROT", "EGG"))


def test_quantile_converges_on_the_mean_once_the_draws_are_done():
    shops = ["SMOOTHIE_SHOP"] * 8            # cap reached, nothing left to draw
    mean = econ.forecast_drain(shops, 21)
    q = econ.forecast_drain(shops, 21, quantile=0.35)
    assert all(abs(q[p] - mean[p]) < 1e-9 for p in ("WOOL", "WHEAT", "CARROT"))


def test_the_quantile_only_moves_v3():
    view = _view()
    F.set_heads_version(1); a1 = F.anchors(view)
    F.set_heads_version(3); a3 = F.anchors(view)
    assert a3["sheep_target"] <= a1["sheep_target"]
    F.set_heads_version(1)
    assert F.anchors(view) == a1, "v1 anchors must not move"


# ---------------------------------------------------------------- flow
def test_tile_flow_counts_ground_that_frees():
    # a wheat tile at its max-yield day frees today; one planted today does not
    view = _view(day=12, tiles={(0, 0): _plant("WHEAT", 8), (1, 0): _plant("WHEAT", 12)})
    fl = F.tile_flow(view, horizon=3)
    assert fl["freeing"][0] >= 1 and fl["freeing"][3] >= 1
    assert fl["free_now"] >= 0
    assert fl["pending"].get("WHEAT", 0) > 0, "standing wheat is supply already committed"


def test_flow_features_exist_only_in_v3():
    view, mem = _view(), Memory()
    F.set_heads_version(1); n1 = len(F.encode(view, mem))
    F.set_heads_version(2); n2 = len(F.encode(view, mem))
    F.set_heads_version(3); n3 = len(F.encode(view, mem))
    assert n1 == n2 == 168, "v1/v2 observation width is frozen for trained nets"
    assert n3 > n1


def test_quota_cannot_exceed_the_ground_that_will_exist():
    # a full farm: nothing bare, nothing freeing -> nothing plantable
    full = {(x, y): _plant("STRAWBERRY", 2) for y in range(5) for x in range(5)}
    view = _view(day=12, tiles=full)
    F.set_heads_version(3)
    choices = [vals.index(1.5) if name in F.FRACTION_HEADS else 0 for name, vals in F.HEADS]
    plan = F.apply_to_plan(planner.make_champion_plan(view, Memory()), choices, view)
    assert sum(plan.plant_quota.values()) == 0, "asked for the maximum on a full farm"
