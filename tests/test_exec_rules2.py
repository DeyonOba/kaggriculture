"""Second batch of switchable executor rules: harvest_visit, spent, layout
(kagent/scheduler.EXEC) and the fert_gate knob (plan.fert_gate). Off by
default; each changes exactly the decision it claims.

    PYTHONPATH=. uv run pytest tests/test_exec_rules2.py -q
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


def _animal(kind):
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": True, "cared_today": True, "fertilizer_available": False,
            "yield_units": 0, "consecutive_unfed": 0}


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


def _plan(obs, **quota):
    mem = Memory()
    view = View(obs)
    plan = planner.make_champion_plan(view, mem)
    plan.plant_quota = dict(quota) if quota else {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    plan.liquidation = False
    return view, plan


def _tasks_at(tasks, x, y):
    return {(t.op[0], t.pri) for t in tasks if (t.x, t.y) == (x, y)}


def test_harvest_visit_takes_a_single_unit_at_tend_priority():
    obs = _obs(day=16, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=4, yield_units=1)
    view, plan = _plan(obs)
    assert not any(op == "HARVEST" for op, _ in _tasks_at(scheduler.build_tasks(view, plan), 1, 1))
    scheduler.set_exec("harvest_visit")
    assert ("HARVEST", scheduler.P_TEND) in _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    # three units: harvest priority as shipped, with or without the flag
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=4, yield_units=3)
    view, plan = _plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _tasks_at(scheduler.build_tasks(view, plan), 1, 1)


def test_spent_digs_a_finished_ongoing_crop():
    obs = _obs(day=24, hour=6)
    # last production made last night (max_lifespan_step set two days out)
    # and harvested clean: the plant becomes a weed at that step
    mls = (24 + 2) * 24
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=10, yield_units=0, mls=mls)
    view, plan = _plan(obs)
    assert not any(op == "DIG" for op, _ in _tasks_at(scheduler.build_tasks(view, plan), 1, 1))
    scheduler.set_exec("spent")
    on = _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    assert ("DIG", scheduler.P_TEND) in on and not any(op == "WATER" for op, _ in on)
    # still holding its last unit: harvest at harvest priority, no dig yet
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=10, yield_units=1, mls=mls)
    view, plan = _plan(obs)
    on = _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    assert ("HARVEST", scheduler.P_HARVEST) in on and not any(op == "DIG" for op, _ in on)
    # not finished (no lifespan set): untouched
    obs["farms"][0]["tiles"][1][1] = _plant("STRAWBERRY", planted_day=10, yield_units=0)
    view, plan = _plan(obs)
    assert not any(op == "DIG" for op, _ in _tasks_at(scheduler.build_tasks(view, plan), 1, 1))


def test_layout_puts_wheat_beside_the_animals():
    obs = _obs(day=14, hour=6)
    tiles = obs["farms"][0]["tiles"]
    for y in range(5):
        for x in range(5):
            tiles[y][x] = _plant("STRAWBERRY", planted_day=4)
    tiles[2][2] = _animal("COW")                       # neighbours (1,2) (3,2) (2,1) (2,3)
    for x, y in ((1, 2), (3, 2), (0, 0), (4, 0)):
        tiles[y][x] = None                              # two bare beside the cow, two far away
    view, plan = _plan(obs, WHEAT=2, CARROT=2)
    tasks = scheduler.build_tasks(view, plan)
    base = {(t.x, t.y): t.op[1] for t in tasks if t.op[0] == "PLANT"}
    scheduler.set_exec("layout")
    on = {(t.x, t.y): t.op[1] for t in scheduler.build_tasks(view, plan) if t.op[0] == "PLANT"}
    assert on[(1, 2)] == "WHEAT" and on[(3, 2)] == "WHEAT"
    assert on[(0, 0)] == "CARROT" and on[(4, 0)] == "CARROT"
    assert set(base) == set(on)                          # same tiles get a task


def test_fert_gate_scales_the_price_gates():
    obs = _obs(day=12, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=10, yield_units=1)  # age 2
    obs["market"]["prices"]["FERTILIZER"] = 60.0                                     # shipped gate < 40
    view, plan = _plan(obs)
    assert plan.fert_gate == 1.0
    assert not any(op == "FERTILIZE" for op, _ in _tasks_at(scheduler.build_tasks(view, plan), 1, 1))
    plan.fert_gate = 2.0                                                             # gate < 80
    assert any(op == "FERTILIZE" for op, _ in _tasks_at(scheduler.build_tasks(view, plan), 1, 1))
    mem = Memory()
    mem.knobs = {"fert_gate": 2.0}
    assert planner.make_champion_plan(view, mem).fert_gate == 2.0
    assert planner.KNOB_DEFAULTS["fert_gate"] == 1.0
