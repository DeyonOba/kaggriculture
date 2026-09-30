"""opening_v2 (planner knob, off by default): a 4-hand crew on days 0-2, crops
that pay first planted first, tomorrow's hire bill kept in the bank, seeds in
lots, animal wants that the market buys as cash arrives while structures
follow what dawn cash affords.
    PYTHONPATH=. uv run pytest tests/test_opening_v2.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import market, planner  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _obs(day=0, money=3000):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, 0, day * 24
    obs["farms"][0]["money"] = money
    return obs


def _plan(obs, **knobs):
    mem = Memory()
    if knobs:
        mem.knobs = dict(knobs)
    view = View(obs)
    return view, planner.make_champion_plan(view, mem), mem


def test_shipped_day_zero_is_unchanged_without_the_knob():
    view, plan, _ = _plan(_obs())
    assert plan.hires_target == 3 and plan.cash_floor == 0.0 and plan.seed_lot == 0
    assert list(plan.plant_quota)[0] == "MELON"


def test_v2_day_zero_crew_order_floor_and_lot():
    view, plan, _ = _plan(_obs(), opening_v2=1)
    assert plan.hires_target == 4
    assert list(plan.plant_quota) == ["WHEAT", "MELON"]         # pays in 2 days before pays in 10
    assert plan.plant_quota == {"WHEAT": 7, "MELON": 12}
    assert plan.cash_floor > 0 and plan.seed_lot == 2 * (1 + plan.hires_target)


def test_v2_wants_stand_but_structures_follow_dawn_cash():
    # day 4, $50 in the bank, fresh farm: the shipped plan wants 4 cows + 2
    # sheep (the market buys them as cash lands) AND builds six pastures for
    # them now -- ground for animals the bank cannot buy
    view, plan, _ = _plan(_obs(day=4, money=50))
    assert plan.animal_buys == {"SHEEP": 2, "COW": 4} and plan.structure_builds["PASTURE"] == 6
    # v2: the same wants, no structure until the cash is there
    view, plan2, _ = _plan(_obs(day=4, money=50), opening_v2=1)
    assert plan2.animal_buys == plan.animal_buys
    assert plan2.structure_builds == {"COOP": 0, "PASTURE": 0}
    # day 4, $2,000: pastures for what dawn cash covers (sheep first: 2 x 500,
    # then cows 2 x 400 fit in the remainder), not for all six
    view, plan3, _ = _plan(_obs(day=4, money=2000), opening_v2=1)
    assert 0 < plan3.structure_builds["PASTURE"] < 6


def test_market_leaves_the_floor_and_buys_seeds_in_lots():
    view, plan, mem = _plan(_obs(), opening_v2=1)
    plan.hires_target = 0                         # isolate the purchases
    plan.cash_floor = 2900.0                      # only $100 to spend
    orders = market.build_orders(view, plan, mem)
    seeds = [o for o in orders if o[0] == "BUY_SEED"]
    animals = [o for o in orders if o[0] == "BUY_ANIMAL"]
    assert animals == []                          # a cow costs 400 > 100
    assert all(o[2] <= plan.seed_lot for o in seeds)
    assert seeds and seeds[0][1] == "WHEAT"       # wheat seeds first


def test_lead_opening_is_the_leaders_day0_basket():
    from kagent import planner, market
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    view = View(obs)
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["lead_opening"] = 1; mem.knobs_decided = True
    plan = planner.make_champion_plan(view, mem)
    assert plan.plant_quota == {"MELON": 6, "WHEAT": 10}, plan.plant_quota
    assert plan.animal_buys == {"SHEEP": 3, "COW": 2}, plan.animal_buys
    assert plan.hires_target == 4 and plan.hire_all_cash
    # day 1 on a $4 dawn: 3 hands for $4, and the four remaining melons
    obs["day"], obs["hour"], obs["step"] = 1, 0, 24
    obs["farms"][0]["money"] = 4
    view = View(obs)
    plan = planner.make_champion_plan(view, mem)
    assert plan.hires_target == 3 and plan.plant_quota["MELON"] == 10
    orders = market.build_orders(view, plan, mem)
    assert [o[0] for o in orders].count("HIRE") == 3, orders
    # shipped: the workload cap holds day 0 at 3 hands and the $4 dawn hires none
    mem.knobs["lead_opening"] = 0
    obs["day"], obs["hour"], obs["step"] = 0, 0, 0
    obs["farms"][0]["money"] = 3000
    plan0 = planner.make_champion_plan(View(obs), mem)
    assert plan0.hires_target == 3 and plan0.plant_quota == {"MELON": 12, "WHEAT": 7} and plan0.animal_buys == {"SHEEP": 2, "COW": 2}
