"""tend_end (kagent/scheduler.EXEC): liquidation keeps tending. Off by
default; days 27-28 run the everyday harvest and fertiliser rules and the
strip-everything harvest is kept for the final day.
    PYTHONPATH=. uv run pytest tests/test_exec_rules4.py -q
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


def _plant(crop, planted_day, yield_units=0, watered=True, mls=-1, fert=-1):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": watered, "consecutive_unwatered": 0,
            "yield_units": yield_units, "max_lifespan_step": mls, "fertilized_until_day": fert}


def _obs(day, hour):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    farm = obs["farms"][0]
    farm["money"] = 5_000
    farm["farmer"] = [4, 4]
    farm["hands"] = []
    obs["private"]["inventories"] = [{}]
    obs["private"]["seeds"] = {"WHEAT": 20, "CARROT": 20}
    obs["private"]["shed"] = {"WHEAT": 30}
    obs["market"]["prices"]["FERTILIZER"] = 60.0
    return obs


def _liq_plan(obs):
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.plant_quota = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    plan.liquidation = True
    return view, plan


def _at(view, plan, x, y):
    return {(t.op[0], t.pri) for t in scheduler.build_tasks(view, plan) if (t.x, t.y) == (x, y)}


def test_tend_end_keeps_the_everyday_harvest_rules_until_the_final_day():
    # day 27, a strawberry holding 1 unit (max 4), unwatered, past its
    # fertiliser window (age 17): the shipped liquidation strips it at
    # harvest priority, ahead of the water
    obs = _obs(day=27, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=10, yield_units=1, watered=False)
    view, plan = _liq_plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _at(view, plan, 1, 1)
    scheduler.set_exec("tend_end")
    on = _at(view, plan, 1, 1)
    assert not any(op == "HARVEST" for op, _ in on), on      # the everyday rule: not yet
    assert any(op == "WATER" for op, _ in on), on            # the water goes first
    scheduler.set_exec(["tend_end", "harvest_visit"])
    assert ("HARVEST", scheduler.P_TEND) in _at(view, plan, 1, 1)   # taken by the unit already there
    # the final day strips it, flag or not
    obs = _obs(day=29, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=12, yield_units=1, watered=False)
    view, plan = _liq_plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _at(view, plan, 1, 1)
    # three units on day 27: harvest priority either way
    obs = _obs(day=27, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=10, yield_units=3)
    view, plan = _liq_plan(obs)
    scheduler.set_exec("tend_end")
    assert ("HARVEST", scheduler.P_HARVEST) in _at(view, plan, 1, 1)
    # a ripe one-time crop is harvested on day 27 with or without the flag
    obs = _obs(day=27, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=23, yield_units=4)
    view, plan = _liq_plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _at(view, plan, 1, 1)


def test_tend_end_fertilises_through_day_28_and_not_on_the_final_day():
    # strawberry in its fertiliser window (age 12), price 60 < 150: shipped
    # liquidation forbids it
    obs = _obs(day=27, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=15)
    view, plan = _liq_plan(obs)
    assert plan.fert_budget is None
    assert not any(op == "FERTILIZE" for op, _ in _at(view, plan, 1, 1))
    scheduler.set_exec("tend_end")
    assert ("FERTILIZE", scheduler.P_TEND) in _at(view, plan, 1, 1)
    obs = _obs(day=28, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=16)
    view, plan = _liq_plan(obs)
    assert ("FERTILIZE", scheduler.P_TEND) in _at(view, plan, 1, 1)
    # day 29: no production is due after the season, nothing to fertilise for
    obs = _obs(day=29, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=17)
    view, plan = _liq_plan(obs)
    assert not any(op == "FERTILIZE" for op, _ in _at(view, plan, 1, 1))


def test_tend_end_is_off_by_default_and_registered():
    assert "tend_end" in scheduler.EXEC_FLAGS and scheduler.EXEC["tend_end"] is False
    from rlv3 import arm as ARM
    spec = ARM.parse('base:exec_flags=["tend_end"]')
    assert spec["exec"][-1] == "tend_end" and "harvest_visit" in spec["exec"]
