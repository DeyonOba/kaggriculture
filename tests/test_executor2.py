"""executor2 (kagent/executor2.py, EXEC flag `executor2`): assignment with
commitment over the same jobs the shipped executor builds. rlv3/EXECUTOR_V2.md.
    PYTHONPATH=. uv run pytest tests/test_executor2.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import executor2, market, planner, scheduler  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flags():
    scheduler.reset_exec()
    yield
    scheduler.reset_exec()


def _plant(crop, planted_day, yield_units=0, watered=False):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": watered, "consecutive_unwatered": 0,
            "yield_units": yield_units, "max_lifespan_step": -1, "fertilized_until_day": -1}


def _pasture(animal=None, fed=False, cared=False):
    t = {"kind": "PASTURE"}
    if animal:
        t.update({"animal": animal, "fed_today": fed, "cared_today": cared,
                  "consecutive_unfed": 0, "yield_units": 0, "fertilizer_available": False})
    return t


def _obs(day, hour, hands=(), farmer=(4, 4), shed=None, seeds=None, money=5_000):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    farm = obs["farms"][0]
    farm["money"] = money
    farm["farmer"] = list(farmer)
    farm["hands"] = [list(h) for h in hands]
    obs["private"]["inventories"] = [{} for _ in range(1 + len(hands))]
    obs["private"]["seeds"] = seeds or {"WHEAT": 20}
    obs["private"]["shed"] = shed or {"WHEAT": 30}
    for y in range(len(farm["tiles"])):
        for x in range(len(farm["tiles"][y])):
            if farm["tiles"][y][x] is not None and farm["tiles"][y][x] != "LOCKED":
                farm["tiles"][y][x] = None
    return obs


def _run(obs, memory=None, plan=None):
    view = View(obs)
    memory = memory or Memory()
    if plan is None:
        plan = planner.make_champion_plan(view, memory)
        plan.plant_quota = {}
        plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    tasks = scheduler.build_tasks(view, plan)
    return view, plan, executor2.assign2(view, plan, tasks, memory), memory


def test_unit_on_a_job_tile_does_the_job_before_anything_else():
    obs = _obs(day=5, hour=8, farmer=(1, 1))
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=3)          # unwatered underfoot
    obs["farms"][0]["tiles"][1][2] = _plant("WHEAT", planted_day=1, yield_units=4)  # ripe next door
    view, plan, actions, _ = _run(obs)
    assert actions[0] == ["WATER"], actions


def test_commitment_holds_a_unit_on_its_job_across_turns():
    # two dry wheat tiles, one unit: it walks to the nearer one and, once
    # committed, keeps walking to it when an equally near job appears
    obs = _obs(day=5, hour=8, farmer=(4, 4))
    obs["farms"][0]["tiles"][4][1] = _plant("WHEAT", planted_day=3)          # (1,4): 3 steps west
    view, plan, actions, memory = _run(obs)
    assert actions[0] == ["WEST"]
    assert memory.exec2_commit[0] == (1, 4, "WATER")
    obs2 = _obs(day=5, hour=9, farmer=(3, 4))
    obs2["farms"][0]["tiles"][4][1] = _plant("WHEAT", planted_day=3)         # (1,4): 2 steps
    obs2["farms"][0]["tiles"][2][3] = _plant("WHEAT", planted_day=3)         # (3,2): 2 steps
    view2, plan2, actions2, memory = _run(obs2, memory=memory, plan=plan)
    assert actions2[0] == ["WEST"], actions2                                 # the committed job, not the tie
    assert memory.exec2_commit[0] == (1, 4, "WATER")


def test_nearer_job_wins_without_a_commitment():
    obs = _obs(day=5, hour=8, farmer=(4, 4))
    obs["farms"][0]["tiles"][4][1] = _plant("WHEAT", planted_day=3)          # 3 steps
    obs["farms"][0]["tiles"][2][4] = _plant("WHEAT", planted_day=3)          # (4,2): 2 steps
    view, plan, actions, memory = _run(obs)
    assert actions[0] == ["NORTH"], actions


def test_two_units_never_take_one_job():
    obs = _obs(day=5, hour=8, farmer=(2, 2), hands=[(2, 3)])
    obs["farms"][0]["tiles"][2][1] = _plant("WHEAT", planted_day=3)          # (1,2)
    view, plan, actions, _ = _run(obs)
    moving = [i for i, a in actions.items() if a[0] in ("WEST", "NORTH", "SOUTH", "EAST")]
    assert len(moving) == 1, actions
    assert actions[0] == ["WEST"]                                            # the nearer unit


def test_bought_animal_is_fetched_from_the_shed_before_nearer_work():
    # a cow in the shed, an empty pasture, a dry tile underfoot-adjacent: the
    # unit at the shed picks the cow up first (the shipped rule, measured:
    # bought animals otherwise strand in the shed)
    obs = _obs(day=3, hour=6, farmer=(4, 4), shed={"WHEAT": 10, "COW": 1})
    obs["farms"][0]["tiles"][2][2] = _pasture()
    obs["farms"][0]["tiles"][4][3] = _plant("WHEAT", planted_day=1)          # (3,4) one step
    view, plan, actions, memory = _run(obs)
    assert actions[0] == ["PICKUP", "COW", 1], actions
    assert memory.exec2_commit[0] == (2, 2, "PLACE")


def test_morning_provisioning_takes_the_days_wheat_out_of_the_shed():
    obs = _obs(day=6, hour=0, farmer=(4, 4), hands=[(4, 4), (4, 4)], shed={"WHEAT": 30})
    for (x, y) in ((1, 1), (2, 1), (1, 2), (2, 2), (7, 7), (8, 8), (7, 8)):
        obs["farms"][0]["tiles"][y][x] = _pasture("COW")
    view, plan, actions, _ = _run(obs)
    pickups = [a for a in actions.values() if a[0] == "PICKUP" and a[1] == "WHEAT"]
    assert sum(a[2] for a in pickups) == 7, actions                          # exactly the seven feeds
    assert all(a[2] <= 5 for a in pickups)


def test_unit_with_wheat_feeds_and_the_care_follows_underfoot():
    obs = _obs(day=6, hour=8, farmer=(2, 2))
    obs["private"]["inventories"] = [{"WHEAT": 3}]
    obs["farms"][0]["tiles"][2][2] = _pasture("COW")
    view, plan, actions, memory = _run(obs)
    assert actions[0] == ["FEED"], actions
    obs2 = _obs(day=6, hour=9, farmer=(2, 2))
    obs2["private"]["inventories"] = [{"WHEAT": 2}]
    obs2["farms"][0]["tiles"][2][2] = _pasture("COW", fed=True)
    obs2["market"]["prices"]["MILK"] = 150.0
    view2, plan2, actions2, memory = _run(obs2, memory=memory, plan=plan)
    assert actions2[0] == ["CARE"], actions2


def test_plant_jobs_respect_seed_stock_and_the_plan_quota():
    obs = _obs(day=5, hour=8, farmer=(1, 1), seeds={"WHEAT": 1})
    view = View(obs)
    memory = Memory()
    plan = planner.make_champion_plan(view, memory)
    plan.plant_quota = {"WHEAT": 3}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    tasks = scheduler.build_tasks(view, plan)
    obs["farms"][0]["hands"] = [[1, 2], [2, 1]]
    obs["private"]["inventories"] = [{}, {}, {}]
    view = View(obs)
    actions = executor2.assign2(view, plan, tasks, memory)
    plants = [a for a in actions.values() if a[0] == "PLANT"]
    assert len(plants) == 1, actions                                         # one seed, one plant


def test_full_pocket_goes_to_the_shed_first():
    obs = _obs(day=12, hour=8, farmer=(1, 1))
    obs["private"]["inventories"] = [{"MILK": 6, "WOOL": 5}]
    obs["farms"][0]["tiles"][1][1] = _plant("WHEAT", planted_day=10)
    view, plan, actions, _ = _run(obs)
    assert actions[0] in (["EAST"], ["SOUTH"]), actions                      # toward the shed, not the water


def test_approve_buys_the_whole_seed_deficit_and_the_flag_off_keeps_the_ration():
    obs = _obs(day=5, hour=0, farmer=(4, 4), seeds={"WHEAT": 0}, money=500)
    view = View(obs)
    memory = Memory()
    plan = planner.make_champion_plan(view, memory)
    plan.plant_quota = {"WHEAT": 40}
    plan.animal_buys = {}
    plan.land_buys = []
    plan.hires_target = 0
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}

    def seeds(approved):
        plan.approved = approved
        orders = market.build_orders(view, plan, memory)
        return sum(o[2] for o in orders if o[0] == "BUY_SEED" and o[1] == "WHEAT")
    assert seeds(False) == int(500 * 0.6 // 10) == 30                       # the 60%-of-cash ration
    assert seeds(True) == 40                                                 # the whole quota ($400 of $500)


def test_policy_routes_through_executor2_only_under_the_flag(monkeypatch):
    from kagent import policy
    calls = []
    monkeypatch.setattr(executor2, "assign2", lambda *a: calls.append("v2") or {})
    monkeypatch.setattr(scheduler, "assign", lambda *a: calls.append("v1") or {})
    obs = _obs(day=5, hour=8)
    pol = policy.HeuristicPolicy(policy.ChampionPlanProvider())
    mem = Memory()
    view = View(obs)
    mem.observe(view) if hasattr(mem, "observe") else None
    pol.act(view, mem)
    scheduler.set_exec("executor2")
    pol.act(View(_obs(day=5, hour=9)), mem)
    assert calls == ["v1", "v2"], calls
