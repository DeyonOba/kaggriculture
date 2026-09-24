"""rl/phase_potential.py: bounded, phase-mixed, seat-antisymmetric, and a
potential in the PBRS sense (its shaping telescopes to a start-state constant).

    PYTHONPATH=. uv run pytest tests/test_phase_potential.py -q
"""
import math
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl import phase_potential as pp  # noqa: E402
from tools import shaping  # noqa: E402


def test_bounded_whatever_the_world_does():
    # tanh saturates to exactly 1.0 in float64 past ~19 scale units, so the
    # bound is reached at absurd leads, never exceeded.
    for lead in (-1e9, -1e5, -1.0, 0.0, 1.0, 1e5, 1e9):
        for day in range(0, 30):
            v = pp.phase_potential(lead, lead, day)
            assert abs(v) <= pp.W
    assert abs(pp.phase_potential(60_000.0, 60_000.0, 5)) < pp.W   # a real lead stays strictly inside
    assert pp.phase_potential(0.0, 0.0, 5) == 0.0


def test_mix_endpoints_and_midpoint():
    # crossover 11, ramp 4: pure asset through day 9, pure money from day 13
    assert pp.phase_mix(0) == 0.0
    assert pp.phase_mix(9) == 0.0
    assert pp.phase_mix(11) == pytest.approx(0.5)
    assert pp.phase_mix(13) == 1.0
    assert pp.phase_mix(29) == 1.0
    # ramp 0 is a step
    assert pp.phase_mix(10.99, ramp=0) == 0.0 and pp.phase_mix(11, ramp=0) == 1.0


def test_early_reads_assets_only_and_late_reads_money_only():
    early_a = pp.phase_potential(10_000.0, 0.0, day=3)
    early_m = pp.phase_potential(0.0, 10_000.0, day=3)
    late_a = pp.phase_potential(10_000.0, 0.0, day=20)
    late_m = pp.phase_potential(0.0, 10_000.0, day=20)
    assert early_a > 0 and early_m == 0.0
    assert late_m > 0 and late_a == 0.0
    # and monotone in the lead that is live
    assert pp.phase_potential(20_000.0, 0.0, 3) > early_a
    assert pp.phase_potential(0.0, 20_000.0, 20) > late_m


def test_is_a_potential_shaping_telescopes():
    """Any Phi telescopes under gamma * Phi(s') - Phi(s); this one included."""
    rng = random.Random(0)
    gamma = 0.997
    for _ in range(50):
        n = rng.randint(1, 30)
        phis = [pp.phase_potential(rng.uniform(-6e4, 6e4), rng.uniform(-6e4, 6e4), d)
                for d in range(n)]
        terms = shaping.shaping_terms(phis, gamma)
        total = shaping.discounted_sum(terms, gamma)
        assert total == pytest.approx(shaping.telescope_sum(phis, gamma), abs=1e-9)
        assert total == pytest.approx(-phis[0], abs=1e-9)      # terminal Phi is 0


def test_symmetric_start_and_seat_antisymmetry():
    """On a fresh world both leads are zero, and the same state seen from the
    other seat negates them -- the potential is a lead, not a level."""
    from kaggle_environments import make

    from kagent.gamestate import View

    env = make("kaggriculture", configuration={"seed": 11}, debug=False)
    env.reset(2)
    v0 = View(env.state[0].observation)
    v1 = View(env.state[1].observation)
    i0, i1 = pp.ingredients(v0), pp.ingredients(v1)
    assert i0["asset_lead"] == 0.0 and i0["money_lead"] == 0.0
    assert pp.potential_of(v0) == 0.0
    assert i1["asset_lead"] == -i0["asset_lead"]
    assert i1["money_lead"] == -i0["money_lead"]
    # advance the world with the built-in agents and check antisymmetry holds
    # on a populated board too
    env.run(["starter", "starter"])
    mid = env.steps[240]                                   # dawn of day 10
    a, b = View(mid[0].observation), View(mid[1].observation)
    ia, ib = pp.ingredients(a), pp.ingredients(b)
    assert ia["day"] == 10 and ib["day"] == 10
    assert ia["asset_lead"] == pytest.approx(-ib["asset_lead"])
    assert ia["money_lead"] == pytest.approx(-ib["money_lead"])
    assert pp.potential_of(a) == pytest.approx(-pp.potential_of(b))
    assert math.isfinite(pp.potential_of(a))


def test_board_assets_matches_rollout_portfolio_arithmetic():
    """Same crop/herd pricing as rl.rollout.portfolio_value, for our seat:
    portfolio_value - money - shed value == board_assets(me)."""
    from kaggle_environments import make

    from kagent import econ
    from kagent.constants import ANIMALS
    from kagent.gamestate import View
    from rl import rollout

    env = make("kaggriculture", configuration={"seed": 12}, debug=False)
    env.run(["starter", "starter"])
    v = View(env.steps[360][0].observation)               # dawn of day 15
    shed_v = 0.0
    for item, n in v.shed.items():
        if n <= 0:
            continue
        if item in ANIMALS:
            shed_v += 0.5 * ANIMALS[item]["cost"] * n
            continue
        inv = v.market_inv.get(item)
        if inv is not None:
            shed_v += econ.sell_revenue(item, inv, min(n, 40))
    assert pp.board_assets(v.me, v) == pytest.approx(
        rollout.portfolio_value(v) - v.me.money - shed_v)
