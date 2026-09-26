"""The land4 knob: the champion plan's third-quadrant constant becomes a gated
fourth-quadrant purchase (day window, crew at size, owned ground mostly
planted, working-capital margin), and the wheat engine is lifted once the
fourth quadrant is owned so the ground is planted.

    PYTHONPATH=. uv run pytest tests/test_land4.py -q
"""
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import planner  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _plant(crop, planted_day):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day, "watered_today": True,
            "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1, "fertilized_until_day": -1}


def _animal(kind):
    return {"kind": "PASTURE", "animal": kind, "fed_today": True, "cared_today": True,
            "fertilizer_available": False, "yield_units": 0, "consecutive_unfed": 0}


def _obs(day, bare, quadrants=("NW", "NE", "SW"), money=40_000):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 7}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    farm = obs["farms"][0]
    farm["money"] = money
    farm["unlocked_quadrants"] = list(quadrants)
    origins = {"NW": (0, 0), "NE": (5, 0), "SW": (0, 5), "SE": (5, 5)}
    left = bare
    for q in quadrants:
        x0, y0 = origins[q]
        for i in range(25):
            x, y = x0 + i % 5, y0 + i // 5
            if q == "NE" and i < 16:
                farm["tiles"][y][x] = _animal("COW" if i < 8 else "SHEEP")   # the herd, housed
            elif left > 0:
                farm["tiles"][y][x] = None
                left -= 1
            else:
                # standing strawberries, so the wheat engine's quota is read
                # against zero standing wheat
                farm["tiles"][y][x] = _plant("STRAWBERRY", day - 1)
    return obs


def _plan(obs, knobs):
    mem = Memory()
    mem.knobs = knobs
    return planner.make_champion_plan(View(obs), mem)


def test_off_by_default_three_quadrants_stay():
    assert planner.KNOB_DEFAULTS["land4"] == 0
    assert _plan(_obs(16, bare=4), {}).land_buys == []


def test_gate_buys_the_fourth_inside_its_conditions():
    plan = _plan(_obs(16, bare=4), {"land4": 1})
    assert plan.land_buys == ["SE"] and plan.land_margin == 15000
    assert plan.hires_target >= planner.LAND4_MIN_HIRES
    # too much idle ground, too early, too late, or the fourth already owned: no buy
    assert _plan(_obs(16, bare=planner.LAND4_MAX_IDLE + 1), {"land4": 1}).land_buys == []
    assert _plan(_obs(planner.LAND4_DAY_MIN - 1, bare=4), {"land4": 1}).land_buys == []
    assert _plan(_obs(planner.LAND4_DAY_MAX + 1, bare=4), {"land4": 1}).land_buys == []
    assert _plan(_obs(16, bare=4, quadrants=("NW", "NE", "SW", "SE")), {"land4": 1}).land_buys == []


def test_wheat_engine_lifted_with_the_fourth_quadrant():
    # 25 bare tiles of the SE quadrant: with the engine lifted the wheat quota covers them
    knobs = {"land4": 1, "wheat_engine": 10}
    three = _plan(_obs(17, bare=0), knobs).plant_quota.get("WHEAT", 0)
    four = _plan(_obs(17, bare=25, quadrants=("NW", "NE", "SW", "SE")), knobs).plant_quota.get("WHEAT", 0)
    assert three == 0 and four >= 20
