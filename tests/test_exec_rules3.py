"""Third batch of switchable executor rules (kagent/scheduler.EXEC):
handfeed, oneshed, hv_underfoot, nodrift. Off by default; each changes
exactly the decision it claims.

    PYTHONPATH=. uv run pytest tests/test_exec_rules3.py -q
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


def _plant(crop, planted_day, yield_units=0, watered=True, unwatered=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": watered, "consecutive_unwatered": unwatered,
            "yield_units": yield_units, "max_lifespan_step": -1, "fertilized_until_day": -1}


def _animal(kind, fed=False, unfed_days=0):
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": fed, "cared_today": False, "fertilizer_available": False,
            "yield_units": 0, "consecutive_unfed": unfed_days}


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
    obs["private"]["seeds"] = {"WHEAT": 20}
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


def _act(obs, **quota):
    view, plan = _plan(obs, **quota)
    return scheduler.assign(view, plan, scheduler.build_tasks(view, plan))


def _fill_strawberries(tiles):
    for y in range(5):
        for x in range(5):
            tiles[y][x] = _plant("STRAWBERRY", planted_day=4)


def test_handfeed_provisioning_skips_an_animal_with_ripe_wheat_next_door():
    # one unfed cow at (2,2), a harvestable wheat at (2,1); the farmer at the
    # shed with no wheat: shipped, it picks wheat up; handfeed, it does not
    obs = _obs(day=14, hour=1)
    tiles = obs["farms"][0]["tiles"]
    _fill_strawberries(tiles)
    tiles[2][2] = _animal("COW")
    tiles[1][2] = _plant("WHEAT", planted_day=10, yield_units=4)
    base = _act(obs)
    assert base[0][0] == "PICKUP" and base[0][1] == "WHEAT"
    scheduler.set_exec("handfeed")
    assert _act(obs)[0][0] != "PICKUP"
    # a starving animal (unfed yesterday) is provisioned regardless
    tiles[2][2] = _animal("COW", unfed_days=1)
    on = _act(obs)
    assert on[0][0] == "PICKUP" and on[0][1] == "WHEAT"
    # after HANDFEED_UNTIL the shed feeds what the harvest has not
    tiles[2][2] = _animal("COW")
    obs["hour"] = scheduler.HANDFEED_UNTIL
    on = _act(obs)
    assert on[0][0] == "PICKUP" and on[0][1] == "WHEAT"


def test_handfeed_wheat_holder_feeds_the_neighbour_before_its_tile_chore():
    # the farmer on a bare tile (2,1) holding 3 wheat, an unfed cow at (2,2)
    # and a plant quota: shipped, it plants the tile underfoot; handfeed, it
    # steps to the cow first, then feeds it
    obs = _obs(day=14, hour=9)
    tiles = obs["farms"][0]["tiles"]
    _fill_strawberries(tiles)
    tiles[2][2] = _animal("COW")
    tiles[1][2] = None
    obs["farms"][0]["farmer"] = [2, 1]
    obs["private"]["inventories"] = [{"WHEAT": 3}]
    with_layout = _act(obs, WHEAT=5)
    assert with_layout[0][0] == "PLANT"
    scheduler.set_exec("handfeed,layout")
    assert _act(obs, WHEAT=5)[0][0] == "SOUTH"
    obs["farms"][0]["farmer"] = [2, 2]
    assert _act(obs, WHEAT=5)[0][0] == "FEED"


def test_oneshed_consolidates_the_morning_pickups():
    obs = _obs(day=14, hour=0, hands=2)
    tiles = obs["farms"][0]["tiles"]
    _fill_strawberries(tiles)
    for i in range(8):
        tiles[i % 5][i // 5] = _animal("COW")               # 8 unfed cows across the zones
    base = _act(obs)
    base_pick = [op for op in base.values() if op[0] == "PICKUP" and op[1] == "WHEAT"]
    assert base_pick and all(op[2] <= 5 for op in base_pick)
    scheduler.set_exec("oneshed")
    on = _act(obs)
    on_pick = [op for op in on.values() if op[0] == "PICKUP" and op[1] == "WHEAT"]
    # the whole need, 8, in the fewest pickups of five: two units, 5 + 3
    assert sorted(op[2] for op in on_pick) == [3, 5]
    assert sum(1 for op in on.values() if op[0] == "PICKUP") == 2
    # later in the day, no provisioning pickup (the animals are still unfed
    # here only because the fixture never feeds them)
    obs["hour"] = scheduler.ONESHED_LAST_HOUR + 3
    scheduler.reset_exec()
    assert any(op[0] == "PICKUP" for op in _act(obs).values())
    scheduler.set_exec("oneshed")
    assert not any(op[0] == "PICKUP" for op in _act(obs).values())
    # unless an animal is starving
    tiles[0][0] = _animal("COW", unfed_days=1)
    assert any(op[0] == "PICKUP" for op in _act(obs).values())


def test_oneshed_raises_the_early_drop_threshold():
    obs = _obs(day=3, hour=8)
    obs["farms"][0]["farmer"] = [0, 0]
    obs["private"]["inventories"] = [{"CARROT": 3}]
    base = _act(obs)
    assert base[0][0] in ("EAST", "SOUTH")                    # shipped: 3 >= 2, walk to the shed
    scheduler.set_exec("oneshed")
    assert _act(obs)[0][0] not in ("EAST", "SOUTH")            # 3 < 5, keep working
    obs["private"]["inventories"] = [{"CARROT": 6}]
    assert _act(obs)[0][0] in ("EAST", "SOUTH")


def test_hv_underfoot_harvests_only_where_a_unit_stands():
    obs = _obs(day=16, hour=6)
    tiles = obs["farms"][0]["tiles"]
    tiles[1][1] = _plant("STRAWBERRY", planted_day=4, yield_units=1)
    tiles[3][3] = _plant("STRAWBERRY", planted_day=4, yield_units=1)
    obs["farms"][0]["farmer"] = [1, 1]
    view, plan = _plan(obs)
    ops = {(t.x, t.y): t.op[0] for t in scheduler.build_tasks(view, plan) if t.op[0] == "HARVEST"}
    assert not ops
    scheduler.set_exec("hv_underfoot")
    ops = {(t.x, t.y): t.op[0] for t in scheduler.build_tasks(view, plan) if t.op[0] == "HARVEST"}
    assert (1, 1) in ops and (3, 3) not in ops
    scheduler.set_exec("harvest_visit")
    ops = {(t.x, t.y): t.op[0] for t in scheduler.build_tasks(view, plan) if t.op[0] == "HARVEST"}
    assert (1, 1) in ops and (3, 3) in ops


def test_nodrift_idle_unit_passes_instead_of_walking_home():
    obs = _obs(day=14, hour=10, hands=2)
    _fill_strawberries(obs["farms"][0]["tiles"])                # nothing to do anywhere
    base = _act(obs)
    assert any(op[0] in ("NORTH", "SOUTH", "EAST", "WEST") for op in base.values())
    scheduler.set_exec("nodrift")
    on = _act(obs)
    assert all(op[0] == "PASS" for op in on.values())
    # a carrier at the end of the day still stages to the shed
    obs["hour"] = 22
    obs["farms"][0]["farmer"] = [0, 0]
    obs["private"]["inventories"] = [{"STRAWBERRY": 4}, {}, {}]
    on = _act(obs)
    assert on[0][0] in ("EAST", "SOUTH")


def test_full_episode_with_the_new_rules_holds_the_invariants():
    from kaggle_environments import make
    from kagent.policy import HeuristicPolicy

    scheduler.set_exec("lastwater,h23,handfeed,layout,oneshed,hv_underfoot,nodrift,spent")
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
        return action

    env = make("kaggriculture", configuration={"seed": 11}, debug=False)
    env.run([agent, "starter"])
    assert env.steps[-1][0].status == "DONE"
    assert not violations, violations[:10]
