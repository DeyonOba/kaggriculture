"""The endwater executor flag: from ENDWATER_DAY an ongoing crop with a
production still due before the season ends is watered daily at animal
priority; a finished plant, a plant whose next production falls after the
last sellable morning, and any plant before ENDWATER_DAY are unchanged.

    PYTHONPATH=. uv run pytest tests/test_endwater.py -q
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


def _obs(day, hour=6):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    obs["farms"][0]["money"] = 5_000
    obs["market"]["prices"]["FERTILIZER"] = 200.0   # no fertilize target, so the water task exists
    return obs


def _straw(planted_day, mls=-1):
    return {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": planted_day, "watered_today": False,
            "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": mls, "fertilized_until_day": -1}


def _water_pri(obs, x, y):
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    return {t.pri for t in scheduler.build_tasks(view, plan) if (t.x, t.y) == (x, y) and t.op[0] == "WATER"}


def test_endgame_water_for_a_plant_with_a_production_due():
    # planted day 12: productions at ages 10, 12, 14, 16 -> days 22, 24, 26, 28
    obs = _obs(day=23)
    obs["farms"][0]["tiles"][1][1] = _straw(12)
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    scheduler.set_exec("endwater")
    assert _water_pri(obs, 1, 1) == {scheduler.P_ANIMAL}


def test_unchanged_before_the_endgame_when_finished_or_past_the_last_morning():
    scheduler.set_exec("endwater")
    obs = _obs(day=15)                                   # before ENDWATER_DAY
    obs["farms"][0]["tiles"][1][1] = _straw(4)
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    obs = _obs(day=24)                                   # finished: lifespan set
    obs["farms"][0]["tiles"][1][1] = _straw(8, mls=26 * 24)
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    # planted day 14: productions land on the mornings of days 24, 26, 28 and
    # 30 -> on day 28 the next one is after the last sellable morning
    obs = _obs(day=28)
    obs["farms"][0]["tiles"][1][1] = _straw(14)
    assert _water_pri(obs, 1, 1) == {scheduler.P_FILLER}
    # planted day 13: the fourth production lands on the morning of day 29,
    # sellable, so the day-28 water is worth animal priority
    obs["farms"][0]["tiles"][1][1] = _straw(13)
    assert _water_pri(obs, 1, 1) == {scheduler.P_ANIMAL}
