"""The dailywater executor flag: an ongoing crop's off-day water is a tend
task instead of filler, so it is not skipped when the crew is busy.

    PYTHONPATH=. uv run pytest tests/test_dailywater.py -q
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


def _obs(day, hour):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    obs["farms"][0]["money"] = 5_000
    # no FERTILIZE task on the strawberry (a fertilize target skips its water
    # until hour 18), so the WATER task is generated
    obs["market"]["prices"]["FERTILIZER"] = 200.0
    return obs


def _water_pri(obs, x, y):
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    return {t.pri for t in scheduler.build_tasks(view, plan) if (t.x, t.y) == (x, y) and t.op[0] == "WATER"}


def test_off_day_water_becomes_a_tend_task():
    obs = _obs(day=15, hour=6)
    # strawberry planted day 4: produces on ages 10, 12, 14, 16 -> day 15 is an off day
    obs["farms"][0]["tiles"][1][1] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 4,
                                      "watered_today": False, "consecutive_unwatered": 0,
                                      "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1}
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    scheduler.set_exec("dailywater")
    assert _water_pri(obs, 1, 1) == {scheduler.P_TEND}
    # a dry plant is survival either way
    obs["farms"][0]["tiles"][1][1]["consecutive_unwatered"] = 1
    assert _water_pri(obs, 1, 1) == {scheduler.P_SURVIVAL}
    scheduler.reset_exec()
    assert _water_pri(obs, 1, 1) == {scheduler.P_SURVIVAL}


def test_one_time_crops_unchanged():
    obs = _obs(day=12, hour=6)
    # wheat age 1: outside the window (2-4) -> filler, flag or not
    obs["farms"][0]["tiles"][1][1] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 11,
                                      "watered_today": False, "consecutive_unwatered": 0,
                                      "yield_units": 1, "max_lifespan_step": 16 * 24, "fertilized_until_day": -1}
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    scheduler.set_exec("dailywater")
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
