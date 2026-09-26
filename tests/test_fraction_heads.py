"""Head set v3: every quantity head is a fraction of a state-dependent
capacity (features.anchors), so resolution follows the operating point.
v1 and v2 are frozen.

    PYTHONPATH=. uv run pytest tests/test_fraction_heads.py -q
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
    F.set_heads_version(1); F.set_mode("absolute")
    yield
    F.set_heads_version(1); F.set_mode("absolute")


def _view(day=12, money=20_000):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 9}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    obs["farms"][0]["money"] = money
    return View(obs)


def test_versions_are_independent():
    v1 = list(F.HEADS)
    F.set_heads_version(3)
    assert len(F.HEADS) == 15
    for name, vals in F.HEADS:
        if name in F.FRACTION_HEADS:
            assert vals == F.FRACTION_FACTORS
    F.set_heads_version(1)
    assert F.HEADS == v1, "v1 must be frozen: trained nets split logits by it"


def test_masks_match_every_head_in_every_version():
    view = _view()
    for v in (1, 2, 3):
        F.set_heads_version(v)
        m = F.masks(view)
        assert len(m) == len(F.HEADS)
        assert all(len(mm) == len(vals) for mm, (_, vals) in zip(m, F.HEADS))


def test_a_fraction_resolves_against_the_anchor_and_respects_the_cap():
    view = _view()
    F.set_heads_version(3)
    a = F.anchors(view)
    for factor, expect_of in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
        choices = [vals.index(factor) if name in F.FRACTION_HEADS else 0
                   for name, vals in F.HEADS]
        plan = F.apply_to_plan(planner.make_champion_plan(view, Memory()), choices, view)
        # hires is the head that lands on the plan unchanged
        assert plan.hires_target == max(0, min(F.ANCHOR_CAPS["hires"],
                                               int(round(expect_of * a["hires"]))))
    # factor 0 really means none
    choices = [vals.index(0.0) if name in F.FRACTION_HEADS else 0 for name, vals in F.HEADS]
    plan = F.apply_to_plan(planner.make_champion_plan(view, Memory()), choices, view)
    assert plan.hires_target == 0 and not plan.animal_buys


def test_resolution_follows_the_anchor():
    # the point of the design: a small anchor gets integer steps, a large one
    # gets coarse steps, from the SAME five factors
    small = [int(round(f * 4)) for f in F.FRACTION_FACTORS]
    large = [int(round(f * 20)) for f in F.FRACTION_FACTORS]
    assert small == [0, 1, 2, 3, 4, 6]
    assert large == [0, 5, 10, 15, 20, 30]


def test_anchors_cover_every_fraction_head():
    a = F.anchors(_view())
    for head in F.FRACTION_HEADS:
        assert head in a, f"{head} has no anchor, so its fraction would always be 0"
        assert a[head] >= 0
