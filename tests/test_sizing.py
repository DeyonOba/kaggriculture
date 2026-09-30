"""kagent/sizing.py (EXEC flags size_budget, size_feed, size_hires, size_ground,
`sized` = all four): the dawn plan is approved against a budget and a crew in
a fixed claim order. rlv3/EXECUTOR_V2.md section 6.
    PYTHONPATH=. uv run pytest tests/test_sizing.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import econ, market, planner, scheduler, sizing  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flags():
    scheduler.reset_exec()
    yield
    scheduler.reset_exec()


def _plant(crop, planted_day, yield_units=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day, "watered_today": False,
            "consecutive_unwatered": 0, "yield_units": yield_units, "max_lifespan_step": -1, "fertilized_until_day": -1}


def _pasture(animal=None):
    t = {"kind": "PASTURE"}
    if animal:
        t.update({"animal": animal, "fed_today": False, "cared_today": False, "consecutive_unfed": 0,
                  "yield_units": 0, "fertilizer_available": False})
    return t


def _obs(day, hour=0, money=500, shed=None, seeds=None, hands=0, wheat_price=None):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    farm = obs["farms"][0]
    farm["money"] = money
    farm["farmer"] = [4, 4]
    farm["hands"] = [[4, 4]] * hands
    farm["hires_today"] = 0
    obs["private"]["inventories"] = [{} for _ in range(1 + hands)]
    obs["private"]["seeds"] = seeds or {}
    obs["private"]["shed"] = shed or {}
    if wheat_price is not None:
        obs["market"]["prices"]["WHEAT"] = wheat_price
    for y in range(len(farm["tiles"])):
        for x in range(len(farm["tiles"][y])):
            if farm["tiles"][y][x] is not None and farm["tiles"][y][x] != "LOCKED":
                farm["tiles"][y][x] = None
    return obs


def _plan(view, quota, animals=None, hires=5, builds=None):
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    plan.plant_quota = dict(quota)
    plan.animal_buys = dict(animals or {})
    plan.structure_builds = dict(builds or {"COOP": 0, "PASTURE": 0})
    plan.hires_target = hires
    plan.land_buys = []
    return plan


def test_fill_covers_the_bare_ground_before_the_expensive_crop():
    # 25 bare tiles, $300 after nothing else: the plan asks 18 strawberries at
    # $100; sized, wheat covers all 25 tiles for $250 and one strawberry is
    # bought with the rest -- no tile waits for the seed it cannot afford
    scheduler.set_exec("sized")
    obs = _obs(day=6, money=300)
    view = View(obs)
    plan = _plan(view, {"STRAWBERRY": 18}, hires=0)
    sizing.approve_plan(view, plan, Memory())
    assert plan.plant_quota["WHEAT"] == 25, plan.plant_quota
    assert plan.plant_quota.get("STRAWBERRY", 0) == 0                        # $50 left buys none
    assert plan.approved
    obs = _obs(day=6, money=400)
    view = View(obs)
    plan = _plan(view, {"STRAWBERRY": 18}, hires=0)
    sizing.approve_plan(view, plan, Memory())
    assert plan.plant_quota == {"WHEAT": 25, "STRAWBERRY": 1}, plan.plant_quota


def test_seeds_on_hand_are_approved_at_no_cost():
    scheduler.set_exec("sized")
    obs = _obs(day=6, money=0, seeds={"STRAWBERRY": 4})
    view = View(obs)
    plan = _plan(view, {"STRAWBERRY": 18}, hires=0)
    sizing.approve_plan(view, plan, Memory())
    assert plan.plant_quota == {"STRAWBERRY": 4}, plan.plant_quota


def test_animals_come_before_the_plans_expensive_crops():
    # $700, 25 bare tiles, a cow asked and 18 strawberries asked: wheat on the
    # ground ($250), the cow ($400), and the $50 left buys no strawberry
    scheduler.set_exec("sized")
    obs = _obs(day=6, money=700)
    view = View(obs)
    plan = _plan(view, {"STRAWBERRY": 18}, animals={"COW": 1}, hires=0, builds={"COOP": 0, "PASTURE": 1})
    sizing.approve_plan(view, plan, Memory())
    assert plan.animal_buys == {"COW": 1}
    assert plan.structure_builds["PASTURE"] == 1
    assert plan.plant_quota == {"WHEAT": 24}, plan.plant_quota                 # one tile is the pasture's
    # an unfunded animal does not get its structure built today
    obs = _obs(day=6, money=300)
    view = View(obs)
    plan = _plan(view, {}, animals={"COW": 1}, hires=0, builds={"COOP": 0, "PASTURE": 1})
    sizing.approve_plan(view, plan, Memory())
    assert plan.animal_buys == {"COW": 0} and plan.structure_builds["PASTURE"] == 0


def test_feed_is_the_first_claim_at_any_price():
    scheduler.set_exec("sized")
    obs = _obs(day=6, money=300, wheat_price=150.0)
    for (x, y) in ((1, 1), (2, 1), (1, 2)):
        obs["farms"][0]["tiles"][y][x] = _pasture("COW")
    view = View(obs)
    plan = _plan(view, {"WHEAT": 25}, hires=0)
    plan.feed_days = 1
    log = sizing.approve_plan(view, plan, Memory())
    assert plan.feed_any
    assert log["feed"] > 0 and log["feed"] == econ.buy_cost("WHEAT", view.market_inv.get("WHEAT", 0), 3)
    assert plan.plant_quota["WHEAT"] < 25                                      # the feed came off the seed money
    # the market then buys the feed above the shipped $120 cap
    orders = market.build_orders(view, plan, Memory())
    assert any(o[0] == "BUY_PRODUCT" and o[1] == "WHEAT" for o in orders), orders
    plan.feed_any = False
    plan.approved = False
    assert not any(o[0] == "BUY_PRODUCT" for o in market.build_orders(view, plan, Memory()))


def test_hires_follow_the_work_over_capacity():
    scheduler.set_exec("sized")
    obs = _obs(day=8, money=5_000)
    farm = obs["farms"][0]
    n = 0
    for y in range(len(farm["tiles"])):
        for x in range(len(farm["tiles"][y])):
            if farm["tiles"][y][x] is None and n < 20:
                farm["tiles"][y][x] = _plant("WHEAT", planted_day=7); n += 1
    for (x, y) in ((0, 0), (1, 0), (2, 0), (3, 0)):
        farm["tiles"][y][x] = _pasture("COW")
    view = View(obs)
    plan = _plan(view, {"WHEAT": 5}, hires=11)                                  # 16 tiles + 5 plantings + 12 = 33 work
    log = sizing.approve_plan(view, plan, Memory())
    import math
    assert log["work"] == 16 + 5 + 12
    assert plan.hires_target == max(sizing.MIN_HANDS, math.ceil(33 / sizing.CAP) - 1)
    # the wage cap: on a $4 dawn with nothing to sell the bill is held to half the budget
    obs = _obs(day=8, money=4)
    view = View(obs)
    plan = _plan(view, {}, hires=11)
    sizing.approve_plan(view, plan, Memory())
    assert plan.hires_target == sizing.MIN_HANDS


def test_budget_counts_yesterdays_realised_income():
    scheduler.set_exec("sized")
    mem = Memory()
    view = View(_obs(day=5, money=100))
    sizing.approve_plan(view, _plan(view, {}, hires=0), mem)
    # day 5 spent $300 on seeds and woke on day 6 with $1,000: income $1,200
    sizing.record_spend(view, mem, [["BUY_SEED", "STRAWBERRY", 3]])
    view6 = View(_obs(day=6, money=1_000))
    log = sizing.approve_plan(view6, _plan(view6, {}, hires=0), mem)
    assert log["forecast"] == pytest.approx(sizing.SALES_DISCOUNT * 1_200)
    assert log["budget"] == pytest.approx(1_000 + sizing.SALES_DISCOUNT * 1_200)


def test_approved_plan_sells_first_at_dawn_and_hires_on_that_cash():
    # the engine resolves orders in list order with the cash updated between
    # them (verified in the env: 8 wheat sold ahead of 3 hires on a $4 dawn
    # paid for the hires); an approved plan puts the dawn sells first
    obs = _obs(day=6, money=4, shed={"WHEAT": 20})
    view = View(obs)
    plan = _plan(view, {}, hires=3)
    plan.wheat_reserve = 0
    plan.approved = True
    orders = market.build_orders(view, plan, Memory())
    kinds = [o[0] for o in orders]
    assert kinds[0] == "SELL" and kinds.count("HIRE") == 3, orders
    plan.approved = False
    orders = market.build_orders(view, plan, Memory())
    kinds = [o[0] for o in orders]
    assert kinds.count("HIRE") <= 1 and (not kinds or kinds[0] != "SELL"), orders   # shipped: 30% of $4


def test_refill_replants_ground_freed_during_the_day():
    scheduler.set_exec("size_ground")
    view = View(_obs(day=8, hour=10, money=100))
    plan = _plan(view, {}, hires=0)
    assert sizing.refill(view, plan, Memory()) == 10                           # $100 buys ten wheat for the 25 bare tiles
    assert plan.plant_quota == {"WHEAT": 10}
    assert sizing.refill(view, plan, Memory()) == 0                            # already asked
    plan.plant_quota = {}
    view = View(_obs(day=8, hour=21, money=100))
    assert sizing.refill(view, plan, Memory()) == 0                            # too late to plant today


def test_flags_off_leaves_the_shipped_plan_and_market_untouched():
    view = View(_obs(day=6, money=700, seeds={"STRAWBERRY": 2}))
    plan = _plan(view, {"STRAWBERRY": 18}, animals={"COW": 1}, hires=5)
    before = (dict(plan.plant_quota), dict(plan.animal_buys), plan.hires_target)
    log = sizing.approve_plan(view, plan, Memory())
    assert (dict(plan.plant_quota), dict(plan.animal_buys), plan.hires_target) == before
    assert not plan.approved and not plan.feed_any
    orders = market.build_orders(view, plan, Memory())
    kinds = [o[0] for o in orders]
    assert kinds.index("BUY_ANIMAL") < kinds.index("BUY_SEED")                 # shipped order: animals before seeds
    assert kinds[0] != "SELL"
