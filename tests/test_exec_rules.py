"""Switchable executor rules (kagent/scheduler.EXEC): off by default and
inert; each rule changes exactly the decision it claims.

    PYTHONPATH=. uv run pytest tests/test_exec_rules.py -q
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


def _plant(crop, planted_day, yield_units=1, watered=False, unwatered=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": watered, "consecutive_unwatered": unwatered,
            "yield_units": yield_units, "max_lifespan_step": -1, "fertilized_until_day": -1}


def _animal(kind, fed=False):
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": fed, "cared_today": False, "fertilizer_available": False,
            "yield_units": 0, "consecutive_unfed": 0}


def _obs(day, hour, hands=0):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    farm = obs["farms"][0]
    farm["money"] = 5_000
    farm["farmer"] = [4, 4]
    farm["hands"] = [[4, 4] for _ in range(hands)]
    obs["private"]["inventories"] = [{} for _ in range(hands + 1)]
    obs["private"]["seeds"] = {"WHEAT": 20, "CARROT": 20}
    obs["private"]["shed"] = {"WHEAT": 30}
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


def test_flags_default_off_and_unknown_rejected():
    assert not any(scheduler.EXEC.values())
    with pytest.raises(ValueError):
        scheduler.set_exec("nosuchflag")
    scheduler.set_exec("lastwater,h23")
    assert scheduler.EXEC["lastwater"] and scheduler.EXEC["h23"] and not scheduler.EXEC["chain"]


def test_lastwater_waters_before_harvesting_on_the_windows_last_day():
    # wheat, age 4 (= max_yield_day), 3 units from the age-2/3 waters, dry
    obs = _obs(day=10, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=6, yield_units=3)
    view, plan = _plan(obs)
    base = _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    assert ("HARVEST", scheduler.P_HARVEST) in base          # shipped: harvest at 3 units
    scheduler.set_exec("lastwater")
    on = _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    assert ("WATER", scheduler.P_HARVEST) in on and not any(op == "HARVEST" for op, _ in on)
    # once watered today the harvest is generated
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=6, yield_units=4, watered=True)
    view, plan = _plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _tasks_at(scheduler.build_tasks(view, plan), 1, 1)
    # late in the day an unwatered crop is harvested anyway
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=6, yield_units=3)
    obs["hour"] = scheduler.LASTWATER_HARVEST_HOUR
    view, plan = _plan(obs)
    assert ("HARVEST", scheduler.P_HARVEST) in _tasks_at(scheduler.build_tasks(view, plan), 1, 1)


def test_chain_plants_the_tile_underfoot_before_the_early_shuttle():
    # day 3: a unit at (0,0) holding 3 carrots (>= EARLY_DROP_THRESHOLD)
    # standing on a bare tile the plan wants planted
    obs = _obs(day=3, hour=8)
    obs["farms"][0]["farmer"] = [0, 0]
    obs["private"]["inventories"] = [{"CARROT": 3}]
    assert obs["farms"][0]["tiles"][0][0] is None
    view, plan = _plan(obs, WHEAT=5)
    tasks = scheduler.build_tasks(view, plan)
    assert any(t.op[0] == "PLANT" and (t.x, t.y) == (0, 0) for t in tasks)
    base = scheduler.assign(view, plan, tasks)
    assert base[0][0] in ("EAST", "SOUTH")                    # shipped: walks to the shed
    scheduler.set_exec("chain")
    on = scheduler.assign(view, plan, tasks)
    assert on[0][0] == "PLANT"


def test_weeds_dig_at_planting_priority_when_ground_is_short():
    obs = _obs(day=12, hour=8)
    tiles = obs["farms"][0]["tiles"]
    # NW quadrant: 8 bare tiles, one weed, the rest planted
    bare = [(x, y) for y in range(5) for x in range(5)][:8]
    for y in range(5):
        for x in range(5):
            if (x, y) in bare:
                tiles[y][x] = None
            elif (x, y) == (4, 4):
                tiles[y][x] = {"kind": "WEED"}
            else:
                tiles[y][x] = _plant("STRAWBERRY", planted_day=2, yield_units=0, watered=True)
    view, plan = _plan(obs, WHEAT=10)
    base = _tasks_at(scheduler.build_tasks(view, plan), 4, 4)
    assert ("DIG", scheduler.P_FILLER) in base
    scheduler.set_exec("weeds")
    on = _tasks_at(scheduler.build_tasks(view, plan), 4, 4)
    assert ("DIG", scheduler.P_GROW) in on
    # enough bare ground: unchanged
    view, plan = _plan(obs, WHEAT=3)
    assert ("DIG", scheduler.P_FILLER) in _tasks_at(scheduler.build_tasks(view, plan), 4, 4)


def test_h23_no_moves_on_the_last_turn():
    obs = _obs(day=8, hour=23, hands=1)
    obs["farms"][0]["tiles"][0][0] = _animal("COW")            # unfed cow far from the shed
    obs["private"]["inventories"] = [{"WHEAT": 2}, {"WHEAT": 2}]
    view, plan = _plan(obs)
    tasks = scheduler.build_tasks(view, plan)
    base = scheduler.assign(view, plan, tasks)
    assert any(op[0] in ("NORTH", "SOUTH", "EAST", "WEST") for op in base.values())
    scheduler.set_exec("h23")
    on = scheduler.assign(view, plan, tasks)
    assert not any(op[0] in ("NORTH", "SOUTH", "EAST", "WEST") for op in on.values())
    # a chore underfoot at hour 23 is still done
    obs["farms"][0]["farmer"] = [0, 0]
    view, plan = _plan(obs)
    on = scheduler.assign(view, plan, scheduler.build_tasks(view, plan))
    assert on[0][0] == "FEED"


def test_quadq_allocates_hands_by_pending_work():
    obs = _obs(day=12, hour=6, hands=3)
    farm = obs["farms"][0]
    farm["unlocked_quadrants"] = ["NW", "NE"]
    for y in range(5):
        for x in range(5, 10):
            farm["tiles"][y][x] = None
    for i in range(6):
        farm["tiles"][i % 5][5 + i // 5] = _animal("COW")        # 6 unfed cows in NE
    farm["tiles"][0][0] = None
    view, plan = _plan(obs, WHEAT=1)                            # 1 planting in NW
    tasks = scheduler.build_tasks(view, plan)
    zone_of, n_zones, uz = scheduler.quadrant_queues(view, tasks, plan)
    assert n_zones == 2 and zone_of[(7, 2)] == 1 and zone_of[(2, 2)] == 0
    counts = {}
    for z in uz.values():
        counts[z] = counts.get(z, 0) + 1
    # bare tiles carry one PLANT task each but the quota is 1 planting, so
    # the work is ~1 : 6 -> 1 : 3 units
    assert counts == {0: 1, 1: 3}
    # no work: even split
    zone_of, n_zones, uz = scheduler.quadrant_queues(view, [], plan)
    counts = {}
    for z in uz.values():
        counts[z] = counts.get(z, 0) + 1
    assert counts == {0: 2, 1: 2}
    # the assignment runs end to end with the flag on
    scheduler.set_exec("quadq")
    actions = scheduler.assign(view, plan, tasks)
    assert len(actions) == 4


def test_evesell_walks_produce_in_before_the_hour_0_dump():
    # a unit three tiles from the shed, standing on a dry plant, carrying 4
    # strawberries at hour 21: shipped, it waters; evesell, it walks in
    obs = _obs(day=14, hour=21)
    obs["farms"][0]["farmer"] = [1, 4]
    obs["farms"][0]["tiles"][4][1] = _plant("STRAWBERRY", planted_day=4, yield_units=0, unwatered=1)
    obs["private"]["inventories"] = [{"STRAWBERRY": 4}]
    view, plan = _plan(obs)
    tasks = scheduler.build_tasks(view, plan)
    assert scheduler.assign(view, plan, tasks)[0][0] == "WATER"
    scheduler.set_exec("evesell")
    assert scheduler.assign(view, plan, tasks)[0][0] == "EAST"
    # too early to leave (arrive by 22 from distance 3 -> leave at 19)
    obs["hour"] = 15
    view, plan = _plan(obs)
    assert scheduler.assign(view, plan, scheduler.build_tasks(view, plan))[0][0] == "WATER"
    # not worth the walk with a small load
    obs["hour"] = 21
    obs["private"]["inventories"] = [{"STRAWBERRY": 2}]
    view, plan = _plan(obs)
    assert scheduler.assign(view, plan, scheduler.build_tasks(view, plan))[0][0] == "WATER"
    # at the shed tile it drops
    obs["farms"][0]["farmer"] = [4, 4]
    obs["private"]["inventories"] = [{"STRAWBERRY": 4}]
    view, plan = _plan(obs)
    assert scheduler.assign(view, plan, scheduler.build_tasks(view, plan))[0][0] == "DROP"


def test_full_episode_with_all_rules_holds_the_scheduler_invariants():
    from kaggle_environments import make
    from kagent.policy import HeuristicPolicy

    scheduler.set_exec("lastwater,chain,weeds,h23,quadq,evesell")
    mem = Memory()
    pol = HeuristicPolicy()
    violations = []

    def agent(obs):
        view = View(obs)
        mem.observe(view)
        action = pol.act(view, mem)
        plants = {}
        for op in [action["farmer"]] + action["hands"]:
            if op and op[0] == "PLANT":
                plants[op[1]] = plants.get(op[1], 0) + 1
        for crop, n in plants.items():
            if n > view.seeds.get(crop, 0):
                violations.append((view.step, crop, n))
        if view.hour == 23 and any(op and op[0] in ("NORTH", "SOUTH", "EAST", "WEST")
                                   for op in [action["farmer"]] + action["hands"]):
            violations.append((view.step, "move at hour 23"))
        return action

    env = make("kaggriculture", configuration={"seed": 11}, debug=False)
    env.run([agent, "starter"])
    assert env.steps[-1][0].status == "DONE"
    assert not violations, violations[:10]
