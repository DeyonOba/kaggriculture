"""The egg line we have never run: the `animal_visit` executor flag and the
`geese_day` knob. Both default to today's behaviour.

    PYTHONPATH=. uv run pytest tests/test_geese_line.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import planner, scheduler  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flags():
    scheduler.reset_exec()
    yield
    scheduler.reset_exec()


def _animal(kind, held, fed=True):
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": fed, "cared_today": True, "fertilizer_available": False,
            "yield_units": held, "consecutive_unfed": 0, "placed_day": 4}


def _obs(day=16):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, 6, day * 24 + 6
    obs["farms"][0]["money"] = 20_000
    return obs


def _ops_at(obs, x, y):
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    return {(t.op[0], t.pri) for t in scheduler.build_tasks(view, plan) if (t.x, t.y) == (x, y)}


def test_animal_visit_collects_a_single_unit():
    obs = _obs()
    obs["farms"][0]["tiles"][1][1] = _animal("GOOSE", held=1)
    assert not any(op == "HARVEST" for op, _ in _ops_at(obs, 1, 1))   # shipped: waits for 2
    scheduler.set_exec("animal_visit")
    assert ("HARVEST", scheduler.P_TEND) in _ops_at(obs, 1, 1)
    # two units: harvest priority either way
    obs["farms"][0]["tiles"][1][1] = _animal("GOOSE", held=2)
    assert ("HARVEST", scheduler.P_HARVEST) in _ops_at(obs, 1, 1)
    # nothing held: no harvest
    obs["farms"][0]["tiles"][1][1] = _animal("GOOSE", held=0)
    assert not any(op == "HARVEST" for op, _ in _ops_at(obs, 1, 1))


def _geese_plan(day, knobs):
    mem = Memory(); mem.knobs = knobs
    return planner.make_champion_plan(View(_obs(day)), mem)


def test_geese_day_knob_defaults_to_today_and_opens_the_window():
    assert planner.KNOB_DEFAULTS["geese_day"] == 6 and planner.KNOB_DEFAULTS["geese"] == 0
    assert _geese_plan(3, {}).animal_buys.get("GOOSE", 0) == 0          # geese off by default
    assert _geese_plan(3, {"geese": 5}).animal_buys.get("GOOSE", 0) == 0  # day 3 < window start 6
    assert _geese_plan(8, {"geese": 5}).animal_buys.get("GOOSE", 0) > 0
    # geese_day 2 opens the early window the leaders use
    assert _geese_plan(3, {"geese": 5, "geese_day": 2}).animal_buys.get("GOOSE", 0) > 0


def test_buying_geese_queues_a_coop():
    plan = _geese_plan(8, {"geese": 5})
    assert plan.animal_buys.get("GOOSE", 0) > 0
    assert plan.structure_builds["COOP"] >= plan.animal_buys["GOOSE"]
