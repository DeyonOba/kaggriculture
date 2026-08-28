"""Parity between kagent's mirrored economics and the installed engine."""

import random

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from kagent import constants, econ


def test_constant_tables_match_engine():
    assert constants.CROPS == engine.CROPS
    assert constants.ANIMALS == engine.ANIMALS
    assert constants.PRODUCTS == engine.PRODUCTS
    assert constants.MARKET_PARAMS == engine.MARKET_PARAMS
    assert constants.SHOPS == engine.SHOPS
    assert constants.MARKET_I0 == engine.MARKET_I0
    assert constants.PRICE_FLOOR == engine.PRICE_FLOOR
    assert constants.HINGE_GAIN == engine.HINGE_GAIN
    assert constants.LAND_ORDER == engine.LAND_ORDER
    assert constants.LAND_PRICES == engine.LAND_PRICES
    assert constants.MAX_SHOP_INSTANCES == engine.MAX_SHOP_INSTANCES
    assert constants.TOWN_CENTER_PRODUCTS == engine.TOWN_CENTER_PRODUCTS


def test_price_parity_dense_range():
    for item in constants.PRODUCTS:
        for inv in range(constants.MARKET_I0 - 1500, constants.MARKET_I0 + 1501):
            assert econ.price(item, inv) == engine.market_price(item, inv), (
                item,
                inv,
            )


def test_price_parity_extremes():
    rng = random.Random(0)
    for item in constants.PRODUCTS:
        for _ in range(200):
            inv = rng.randint(0, 40000)
            assert econ.price(item, inv) == engine.market_price(item, inv)


def test_fib_matches_engine():
    for n in range(25):
        assert constants.fib(n) == engine._fib(n)


def test_hire_cost_closed_form():
    for k in range(1, 20):
        assert econ.hire_total_cost(k) == constants.fib(k + 1) - 1
