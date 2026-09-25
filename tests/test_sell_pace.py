"""Sell pacing (plan.sell_pace / KNOB_DEFAULTS["sell_pace"]): off by default;
on, a steep-curve product with open shop demand is sold per step only as far
as keeps the within-order price drop under `pace` percent, a product the
town does not buy is sold as before, and the shed-pressure valve and the
crashed-market release bypass the cap.

    PYTHONPATH=. uv run pytest tests/test_sell_pace.py -q
"""
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import econ, market, planner  # noqa: E402
from kagent.constants import MARKET_I0  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _view(shed, inv, shops, day=14):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 3}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24 + 2, 2
    obs["private"]["shed"] = dict(shed)
    for item, n in inv.items():
        obs["market"]["inventory"][item] = n
        obs["market"]["prices"][item] = econ.price(item, n)
    obs["town"] = {"unlocked_shops": list(shops)}
    return View(obs)


def _plan(pace, floors):
    plan = planner.DayPlan()
    plan.sell_floors = dict(floors)
    plan.sell_pace = pace
    return plan


def _sells(view, plan):
    return {o[1]: o[2] for o in market._sell_orders(view, plan, Memory())}


def test_off_by_default_and_shipped_sizing_unchanged():
    assert planner.KNOB_DEFAULTS["sell_pace"] == -1 and planner.DayPlan().sell_pace == -1
    view = _view({"STRAWBERRY": 30}, {"STRAWBERRY": MARKET_I0 - 20}, ["SMOOTHIE_SHOP"])
    assert _sells(view, _plan(-1, {"STRAWBERRY": 40}))["STRAWBERRY"] == 30


def test_pace_bounds_the_within_order_price_drop():
    floors = {"STRAWBERRY": 40}
    view = _view({"STRAWBERRY": 30}, {"STRAWBERRY": MARKET_I0 - 20}, ["SMOOTHIE_SHOP"])
    first = econ.price("STRAWBERRY", MARKET_I0 - 20)
    for pace in (5, 10, 25):
        n = _sells(view, _plan(pace, floors))["STRAWBERRY"]
        assert 0 < n < 30
        last = econ.sell_marginal_prices("STRAWBERRY", MARKET_I0 - 20, n)[-1]
        assert last >= first * (1 - pace / 100.0)
        # one more unit would break the bound
        nxt = econ.sell_marginal_prices("STRAWBERRY", MARKET_I0 - 20, n + 1)[-1]
        assert nxt < first * (1 - pace / 100.0)
    assert _sells(view, _plan(5, floors))["STRAWBERRY"] < _sells(view, _plan(25, floors))["STRAWBERRY"]


def test_products_without_shop_demand_are_not_paced():
    # wool with no yarn store open: the market never drains, sell as shipped
    view = _view({"WOOL": 10}, {"WOOL": MARKET_I0 - 5}, ["SMOOTHIE_SHOP"])
    assert _sells(view, _plan(5, {"WOOL": 60}))["WOOL"] == 10
    # with a yarn store it is paced
    view = _view({"WOOL": 10}, {"WOOL": MARKET_I0 - 5}, ["YARN_STORE"])
    assert _sells(view, _plan(5, {"WOOL": 60}))["WOOL"] < 10


def test_pressure_and_collapse_bypass_the_cap():
    # shed over 80 units: the valve dumps at any price
    view = _view({"STRAWBERRY": 85}, {"STRAWBERRY": MARKET_I0 - 20}, ["SMOOTHIE_SHOP"])
    assert _sells(view, _plan(5, {"STRAWBERRY": 40}))["STRAWBERRY"] > 30
    # a collapsed market (price <= 0.3 x base) releases everything
    view = _view({"STRAWBERRY": 30}, {"STRAWBERRY": MARKET_I0 + 90}, ["SMOOTHIE_SHOP"])
    assert view.market_prices["STRAWBERRY"] <= 0.3 * 120
    assert _sells(view, _plan(5, {"STRAWBERRY": 40}))["STRAWBERRY"] == 30
