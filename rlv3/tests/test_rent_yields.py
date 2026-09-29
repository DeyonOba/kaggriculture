"""`rent.crop_block`'s one-time unit count, checked against the ENGINE.

WHY THE ORIGINAL TESTS MISSED IT. Every existing rent test in
`test_allocation.py` is a SHAPE test, never a LEVEL test:
`test_marginal_rent_is_concave_in_n` asserts the marginals decrease,
`test_water_fill_is_equal_marginal` asserts no candidate beats the last
accepted step, `test_relative_rent_exceeds_own_rent_...` asserts an ordering
between three numbers computed from the same `crop_block`. Scaling wheat's
units from 4 to 6 leaves every one of those properties intact -- the curve is
still concave, the fill is still equal-marginal, the opponent term is still
positive -- so a 50% error in the absolute yield passed all of them. Nothing
ever compared a number out of this module to a number out of the engine, and
`sell_revenue` was the only quantity pinned to a verified constant
(`test_sell_revenue_matches_the_measured_ladder`).

So these tests do the one thing the shape tests cannot: they PLAY the tile.
`engine_watered_yield` plants a seed through `kaggriculture._apply_unit_action`,
waters it every day through the real day refresh, harvests it at the RIPE age
measured off the engine, and reads the units out of the farmer's inventory. No
game is played and nothing is reasoned about -- README "Object Types" line 28 says
wheat and carrot reach 6 and 4 only WITH fertilizer, the engine's WATER branch
says `bonus = 2 if tile["fertilized_until_day"] >= day else 1`, and
`block_revenue` never buys a fertilizer, so the fill must be priced at the
watered-only total.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.kaggriculture import kaggriculture as ENG

from kagent import features
from rlv3 import rent
from rlv3.tests.conftest import mid_season_cases
from rlv3.tests.test_rent_melon_last_plant import engine_ripe_age, engine_tile

ONE_TIME = ("WHEAT", "CARROT", "MELON")


def engine_watered_yield(crop, *, fertilize=False):
    """(units harvested, days occupied) for one tile driven by the ENGINE.

    Plant on day 0, water every day, harvest at the RIPE age -- the age
    watering stops adding units, measured off the engine by `engine_ripe_age`.

    This used to harvest on `max_yield_day` and report `max_yield_day + 1` as
    the occupancy, i.e. it took `crop_block`'s convention as its own
    definition. That is why the melon gate and the melon rate could be wrong
    here for a season without a test noticing: the only file pinning this
    module to the engine agreed with it by construction. See
    `test_rent_melon_last_plant`.
    """
    ripe = engine_ripe_age(crop)
    _h, units, _c = engine_tile(crop, 0, harvest_age=ripe, fertilize=fertilize)
    return units, ripe + 1


@pytest.mark.parametrize("crop", ONE_TIME)
def test_crop_block_units_are_the_engines_watered_only_yield(crop):
    """The defect: `units = CROPS[crop]["max_yield"]` is the FERTILISED cap.

    The engine hands back 4 wheat and 3 carrot on water alone (README's
    "Yield / tile / day" column, 0.80 and 0.75); `crop_block` claimed 6 and 4,
    a 1.50x and 1.33x overstatement of the quantity `marginal_rent` ranks, with
    no fertilizer ever charged in `block_revenue`.
    """
    watered, occupies = engine_watered_yield(crop)
    spec = rent.crop_block(crop, 0, horizon=14)
    assert spec is not None
    _k, units, occ = spec
    assert units == float(watered), (crop, units, watered)
    assert occ == float(occupies), (crop, occ, occupies)

    # And the fertilised cap really is the higher number, so the fix is not
    # just renaming a coincidence: max_yield is reachable, but only by paying.
    fertilised, _ = engine_watered_yield(crop, fertilize=True)
    assert fertilised == ENG.CROPS[crop]["max_yield"], (crop, fertilised)
    if crop in ("WHEAT", "CARROT"):
        assert fertilised > watered, (crop, fertilised, watered)


@pytest.mark.parametrize("crop", ONE_TIME)
def test_crop_block_rate_agrees_with_per_tile_day(crop):
    """The module was internally inconsistent about the SAME tile.

    `allocation.demand_caps` sizes its cap with `rent.per_tile_day` (wheat
    0.800 out of `features._PER_TILE_DAY`) while the water-fill it feeds scored
    that tile through `crop_block` at 6/5 = 1.200 -- the cap and the score
    disagreed by 50%. MELON used to be excluded here, its own 18% disagreement
    (6/13 = 0.462 against 0.545) recorded in this docstring as "a difference in
    occupancy convention"; it was a second defect, and it is in the
    parametrisation now. README divides melon's 6 units by the 11 days to the
    cap, and so does `crop_block`.
    """
    _k, units, occ = rent.crop_block(crop, 0, horizon=14)
    assert units / occ == pytest.approx(rent.per_tile_day(crop), abs=1e-3)
    assert units / occ == pytest.approx(features._PER_TILE_DAY[crop], abs=1e-3)


def test_block_revenue_prices_the_watered_units_on_real_boards():
    """The priced path, not just the constant: on the recorded mid-season
    Views, `block_revenue`'s total units for n tiles is n x the engine's
    watered-only yield, so nothing downstream reintroduces the fertilised cap.
    """
    cases = mid_season_cases()
    if not cases:
        pytest.skip("no replays under replays/{mmpq,boey,dsm}")
    truth = {c: engine_watered_yield(c)[0] for c in ONE_TIME}
    checked = 0
    for label, view, _plan in cases:
        path = rent.inventory_path(view, horizon=14)
        for crop in ONE_TIME:
            if rent.crop_block(crop, view.day, horizon=14) is None:
                continue
            _rev, _occ, total, _inv = rent.block_revenue(view, crop, 7, path)
            assert total == 7 * truth[crop], (label, crop, total, truth[crop])
            checked += 1
    assert checked > 0
