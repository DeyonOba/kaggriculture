"""Stage 2 sell timing on the exact market (kagent/selltiming.py, EXEC flag sell_dp).
    PYTHONPATH=. uv run pytest tests/test_selltiming.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import econ, market, planner, scheduler, selltiming as S  # noqa: E402
from kagent.constants import MARKET_PARAMS, TOWN_CENTER_SELL_INTERVAL, TOWN_SHOP_SELL_INTERVAL  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flags():
    scheduler.reset_exec()
    yield
    scheduler.reset_exec()


def _obs(day, hour, shed=None, inv=None, shops=None):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    obs["farms"][0]["money"] = 500
    obs["private"]["shed"] = shed or {}
    for item, n in (inv or {}).items():
        obs["market"]["inventory"][item] = n
        obs["market"]["prices"][item] = econ.price(item, n)
    obs["town"]["unlocked_shops"] = list(shops or [])
    return obs


def test_drain_schedule_matches_the_engine():
    assert S.drain_at(0, ["YARN_STORE", "BAKERY"])["WOOL"] == 2 + 1            # shop x2 (single product) + town centre
    assert S.drain_at(TOWN_SHOP_SELL_INTERVAL, ["YARN_STORE"]) == {"WOOL": 2}
    assert S.drain_at(TOWN_SHOP_SELL_INTERVAL + 1, ["YARN_STORE"]) == {}
    assert S.drain_at(TOWN_CENTER_SELL_INTERVAL, [])["MILK"] == 1


def test_inverse_price_is_the_largest_inventory_at_or_above_the_price():
    for item in S.PRODUCTS_DP:
        for lam in (2, 20, 60, 100, MARKET_PARAMS[item]["base"]):
            I = S.inv_at_price(item, lam)
            assert econ.price(item, I) >= lam > econ.price(item, I + 1)


def test_supply_beating_the_drain_sells_now():
    # milk far above target, no shops, the opponent's prior supply on top:
    # the path only falls, so the whole shed goes now
    obs = _obs(day=12, hour=3, shed={"MILK": 12}, inv={"MILK": 10020})
    view = View(obs)
    assert S.sell_now("MILK", 12, view, Memory()) == 12


def test_a_depressed_market_with_a_strong_drain_is_held():
    # wool at $1 (inventory 40 over the cliff) with four yarn stores draining
    # 8 units every 4 turns and the town centre: the price is back near base
    # within the horizon, so nothing is sold into the floor now
    p = MARKET_PARAMS["WOOL"]
    glut = S.inv_at_price("WOOL", 2) + 40
    obs = _obs(day=12, hour=3, shed={"WOOL": 10}, inv={"WOOL": glut}, shops=["YARN_STORE"] * 4)
    view = View(obs)
    mem = Memory()
    mem.st_state = {"prev_step": None, "prev_inv": {}, "ours": {}, "day_units": {}, "day": 12, "daily": {"WOOL": 0.0},
                    "hour_units": {q: [0.0] * 24 for q in S.PRODUCTS_DP}, "hour_n": {q: 0.0 for q in S.PRODUCTS_DP}}
    n = S.sell_now("WOOL", 10, view, mem)
    assert n == 0, n
    # the shipped floor logic would dump it (crashed-market release)
    plan = planner.make_champion_plan(view, mem)
    plan.liquidation = False
    scheduler.set_exec("sell_dp")
    assert not [o for o in market._sell_orders(view, plan, mem) if o[1] == "WOOL"]
    scheduler.reset_exec()
    assert [o for o in market._sell_orders(view, plan, mem) if o[1] == "WOOL"]


def test_their_supply_is_read_from_the_public_inventory():
    mem = Memory()
    v1 = View(_obs(day=12, hour=3, inv={"MILK": 10020}))
    S.observe_market(v1, mem)
    S.note_requests(mem, [["SELL", "MILK", 3]])
    # next step: inventory rose by 3 (ours) + 5 (theirs) - 0 drain (step 291 is not a tick)
    v2 = View(_obs(day=12, hour=4, inv={"MILK": 10028}))
    S.observe_market(v2, mem)
    assert mem.st_state["day_units"]["MILK"] == 5
    assert mem.st_state["hour_units"]["MILK"][3] == 5


def test_flag_off_leaves_the_shipped_sells_untouched():
    obs = _obs(day=12, hour=0, shed={"STRAWBERRY": 8, "MILK": 6}, inv={"STRAWBERRY": 10005, "MILK": 10005})
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    before = market._sell_orders(view, plan, Memory())
    scheduler.set_exec("sell_dp")
    on = market._sell_orders(view, plan, Memory())
    scheduler.reset_exec()
    after = market._sell_orders(view, plan, Memory())
    assert before == after
    assert {o[1] for o in on} <= {o[1] for o in before} | set(S.PRODUCTS_DP)
