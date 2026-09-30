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


def test_late_plant_puts_bagged_seeds_in_the_ground_on_day_27_only():
    obs = _obs(day=27, hour=6)
    obs["private"]["seeds"] = {"CARROT": 8, "WHEAT": 3, "STRAWBERRY": 2}
    view, plan = _liq_plan(obs)              # liquidation: the planner asks for nothing
    assert plan.plant_quota == {}
    plants = lambda v, p: [t for t in scheduler.build_tasks(v, p) if t.op[0] == "PLANT"]
    assert plants(view, plan) == []
    scheduler.set_exec("late_plant")
    view, plan = _liq_plan(obs)
    on = plants(view, plan)
    crops = {t.op[1] for t in on}
    assert crops <= {"WHEAT", "CARROT"} and "CARROT" in crops, crops   # never strawberry (10-day lead)
    assert plan.plant_quota == {"WHEAT": 3, "CARROT": 8}
    # day 28: a 2-day crop's first yield lands after the season
    obs = _obs(day=28, hour=6)
    obs["private"]["seeds"] = {"CARROT": 8, "WHEAT": 3}
    view, plan = _liq_plan(obs)
    assert plants(view, plan) == [] and plan.plant_quota == {}


def test_late_plant_harvests_an_unripe_one_time_crop_on_the_final_day():
    # a carrot planted on day 27 is age 2 on day 29 with 2 of 4 units: the
    # shipped ripe rule (max yield or max_yield_day) leaves it on the tile
    obs = _obs(day=29, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("CARROT", planted_day=27, yield_units=2)
    view, plan = _liq_plan(obs)
    assert not any(op == "HARVEST" for op, _ in _at(view, plan, 1, 1))
    scheduler.set_exec("late_plant")
    assert ("HARVEST", scheduler.P_HARVEST) in _at(view, plan, 1, 1)
    # not on day 28, and never an empty tile
    obs = _obs(day=28, hour=6)
    obs["farms"][0]["tiles"][1][1] = _plant("CARROT", planted_day=26, yield_units=2)
    view, plan = _liq_plan(obs)
    assert not any(op == "HARVEST" for op, _ in _at(view, plan, 1, 1))


def test_care_all_emits_care_without_waiting_for_the_feed():
    obs = _obs(day=20, hour=6)
    obs["farms"][0]["tiles"][1][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": False,
                                      "fertilizer_available": False, "yield_units": 0, "consecutive_unfed": 0}
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.plant_quota = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    assert "COW" not in (getattr(plan, "cull_kinds", None) or ())
    on = _at(view, plan, 1, 1)
    assert ("FEED", scheduler.P_ANIMAL) in on and not any(op == "CARE" for op, _ in on)   # shipped: care waits
    scheduler.set_exec("care_all")
    on = _at(view, plan, 1, 1)
    assert ("FEED", scheduler.P_ANIMAL) in on and ("CARE", scheduler.P_ANIMAL) in on
    # already cared: nothing more; day 28: no care either way
    obs["farms"][0]["tiles"][1][1]["cared_today"] = True
    assert not any(op == "CARE" for op, _ in _at(View(obs), plan, 1, 1))
    obs = _obs(day=28, hour=6)
    obs["farms"][0]["tiles"][1][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": False,
                                      "fertilizer_available": False, "yield_units": 0, "consecutive_unfed": 0}
    view = View(obs); plan = planner.make_champion_plan(view, Memory()); plan.plant_quota = {}; plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    assert not any(op == "CARE" for op, _ in _at(view, plan, 1, 1))


def test_radial_plants_from_the_outside_in_and_leaves_the_shed_ring_for_structures():
    obs = _obs(day=5, hour=6)
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.plant_quota = {"WHEAT": 3}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    plan.liquidation = False
    plants = lambda: [(t.x, t.y) for t in scheduler.build_tasks(view, plan) if t.op[0] == "PLANT"]
    first3 = plants()[:3]
    assert first3 and all(scheduler._shed_ring(x, y) <= scheduler.RADIAL_RING for x, y in first3), first3   # shipped: nearest first
    scheduler.set_exec("radial")
    first3 = plants()[:3]
    assert all(scheduler._shed_ring(x, y) > scheduler.RADIAL_RING for x, y in first3), first3          # radial: ring last
    # the ring is still planted when nothing else is bare
    ring = [(x, y) for x, y in plants() if scheduler._shed_ring(x, y) <= scheduler.RADIAL_RING]
    assert ring and plants()[-len(ring):] == ring
    # structures still take the nearest empties
    plan.structure_builds = {"COOP": 0, "PASTURE": 2}
    builds = [(t.x, t.y) for t in scheduler.build_tasks(view, plan) if t.op[0] == "BUILD_PASTURE"]
    assert len(builds) == 2 and all(scheduler._shed_ring(x, y) <= scheduler.RADIAL_RING for x, y in builds), builds


def test_care_price_keeps_caring_when_the_product_has_crashed():
    from kagent.constants import MARKET_PARAMS
    obs = _obs(day=15, hour=6)
    obs["farms"][0]["tiles"][1][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False,
                                       "consecutive_unfed": 0, "yield_units": 0, "fertilizer_available": False}
    obs["market"]["prices"]["MILK"] = 0.2 * MARKET_PARAMS["MILK"]["base"]
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.cull_kinds = set()
    assert not any(op == "CARE" for op, _ in _at(view, plan, 1, 1))          # shipped: the crash-pivot
    scheduler.set_exec("care_price")
    assert ("CARE", scheduler.P_ANIMAL) in _at(view, plan, 1, 1)
    obs["market"]["prices"]["MILK"] = MARKET_PARAMS["MILK"]["base"]
    scheduler.reset_exec()
    assert ("CARE", scheduler.P_ANIMAL) in _at(View(obs), plan, 1, 1)         # a healthy price: cared either way


def test_hold_crash_keeps_a_crashed_product_for_its_floor():
    from kagent import market
    from kagent.constants import MARKET_PARAMS
    obs = _obs(day=15, hour=6)
    obs["private"]["shed"] = {"MILK": 12}
    obs["market"]["prices"]["MILK"] = 0.2 * MARKET_PARAMS["MILK"]["base"]
    p = MARKET_PARAMS["MILK"]
    obs["market"]["inventory"]["MILK"] = int(p["I0"] + p["T"])                  # past the target by the whole ramp: quotes at $1
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    plan.sell_floors["MILK"] = 40
    sells = [o for o in market._sell_orders(view, plan, Memory()) if o[1] == "MILK"]
    assert sells and sells[0][2] == 12                                          # shipped: the crashed-market release dumps it
    scheduler.set_exec("hold_crash")
    assert not [o for o in market._sell_orders(view, plan, Memory()) if o[1] == "MILK"]
    # the shed-pressure valve still dumps
    obs["private"]["shed"] = {"MILK": 90}
    assert [o for o in market._sell_orders(View(obs), plan, Memory()) if o[1] == "MILK"]
