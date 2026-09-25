"""The fill_idle knob: bare ground is topped up with wheat by the labour that
stood idle yesterday, and never beyond the bare tiles; off by default.

    PYTHONPATH=. uv run pytest tests/test_fill.py -q
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
            "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
            "fertilized_until_day": -1}


def _obs(day, bare):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 7}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    farm = obs["farms"][0]
    farm["money"] = 8_000
    # NW quadrant: `bare` empty tiles, the rest standing strawberries (no
    # wheat, so the wheat engine wants its full cap regardless of the fill).
    # NE quadrant: the full herd already housed, so the plan queues no
    # structure builds (those would take the bare ground first).
    farm["unlocked_quadrants"] = ["NW", "NE"]
    n = 0
    for y in range(5):
        for x in range(5):
            if n < bare:
                farm["tiles"][y][x] = None
            else:
                farm["tiles"][y][x] = _plant("STRAWBERRY", day - 3)
            n += 1
    for i in range(25):
        x, y = 5 + i % 5, i // 5
        if i < 8:
            farm["tiles"][y][x] = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                                   "cared_today": True, "fertilizer_available": False,
                                   "yield_units": 0, "consecutive_unfed": 0}
        elif i < 16:
            farm["tiles"][y][x] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": True,
                                   "cared_today": True, "fertilizer_available": False,
                                   "yield_units": 0, "consecutive_unfed": 0}
        else:
            farm["tiles"][y][x] = _plant("STRAWBERRY", day - 3)
    return obs


def _quota(day, bare, knobs, idle, hands):
    mem = Memory()
    mem.knobs = knobs
    mem.yesterday_idle = idle
    mem.yesterday_hands = hands
    view = View(_obs(day, bare))
    return planner.make_champion_plan(view, mem).plant_quota


DAY = 16  # only the wheat engine plants (strawberries end day 13, carrots start day 25)


def test_off_by_default():
    assert planner.KNOB_DEFAULTS["fill_idle"] == 0.0
    base = _quota(DAY, bare=10, knobs={"wheat_engine": 4}, idle=0.5, hands=9)
    assert sum(base.values()) < 10


def test_fill_tops_wheat_up_to_the_bare_ground_by_idle_labour():
    base = _quota(DAY, bare=10, knobs={"wheat_engine": 4}, idle=0.5, hands=9)
    # 0.5 x 10 units x 24 turns = 120 idle turns x 0.33 -> 39 tiles, capped by the room
    on = _quota(DAY, bare=10, knobs={"wheat_engine": 4, "fill_idle": 0.33}, idle=0.5, hands=9)
    assert sum(on.values()) == 10 and on["WHEAT"] > base.get("WHEAT", 0)
    # little idle labour -> a small top-up: 0.05 x 10 x 24 = 12 turns x 0.33 -> 3
    small = _quota(DAY, bare=10, knobs={"wheat_engine": 4, "fill_idle": 0.33}, idle=0.05, hands=9)
    assert sum(base.values()) < sum(small.values()) < 10
    # no idle labour yesterday -> nothing
    none = _quota(DAY, bare=10, knobs={"wheat_engine": 4, "fill_idle": 0.33}, idle=0.0, hands=9)
    assert none == base


def test_fill_respects_the_planting_window_and_liquidation():
    late = _quota(planner.WHEAT_LAST_PLANT_DAY + 1, bare=10,
                  knobs={"wheat_engine": 4, "fill_idle": 0.33}, idle=0.5, hands=9)
    assert late.get("WHEAT", 0) == 0
