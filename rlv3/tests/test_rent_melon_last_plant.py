"""`rent.crop_block`'s one-time RIPE AGE, checked by playing the tile.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_rent_melon_last_plant.py -q

THE DEFECT. `crop_block` gated one-time crops on `day + max_yield_day >
SEASON_END` and charged them `max_yield_day + 1` tile-days. `max_yield_day` is
the end of the bonus WINDOW, not the age the yield caps. For MELON the window
runs to age 12 but base 1 plus one unit per watered day hits the cap of 6 at
age 10, so the last plantable melon day came out 17 where the engine (and
`features.LAST_PLANT`) allow 19 -- `allocation.demand_caps` zeroed MELON on
days 18 and 19, refusing two days of the highest-value one-time crop -- and
every melon tile was priced over 13 tile-days instead of 11, understating
melon $/tile-day by 18% (147.45 -> 124.77 at day 6) all season.

WHY THE EXISTING TESTS MISSED IT.

  * `test_rent_yields.py` is the file that pinned `crop_block` to the engine,
    and it pinned only the UNIT COUNT. Its engine driver harvests at
    `cd["max_yield_day"]` and returns `cd["max_yield_day"] + 1` as occupancy --
    it takes the defective convention as its own definition, so it could never
    disagree with the module about it. Melon's unit count is 6 either way, so
    melon passed.

  * `test_crop_block_rate_agrees_with_per_tile_day` is the one test that would
    have caught it -- 6/13 = 0.462 against `per_tile_day("MELON") = 0.545` --
    and MELON was explicitly excluded from its parametrisation, with the gap
    recorded in the docstring as "a difference in occupancy convention that
    predates this defect". A known disagreement between two numbers in the
    same module was written down as a convention instead of measured.

  * Nothing anywhere asked the engine WHICH DAYS a crop can still be planted
    on. The gate is a hard zero in `allocation.demand_caps`; every rent test
    in `test_allocation.py` is a SHAPE test (concavity, equal-marginal,
    orderings) and shapes survive a crop being dropped from the candidate set
    entirely -- the fill just splits the capacity among the crops that are
    left, still concave and still equal-marginal.

So these tests do the thing none of those do: they PLAY the tile through the
engine on the days in question. `engine_tile` plants through
`kaggriculture._apply_unit_action`, waters every day, runs the real per-step
`_decay_plants` and per-day `_daily_refresh_plants`, harvests, and reads the
units out of the farmer's inventory. Nothing is reasoned about and no game is
played.
"""

from __future__ import annotations

import pytest

from kaggle_environments.envs.kaggriculture import kaggriculture as ENG

from kagent import features
from rlv3 import allocation, rent
from rlv3.tests.test_allocation import make_view

ONE_TIME = ("WHEAT", "CARROT", "MELON")
TURNS_PER_DAY = 24
SEASON_END = rent.SEASON_END          # 29, the last day a harvest can land


# --- the engine, driven ------------------------------------------------------

def _fresh(crop, fertilize):
    farm = {"tiles": [[None]], "farmer": [0, 0], "hands": [], "money": 10 ** 6,
            "unlocked_quadrants": ["NW"], "hires_today": 0}
    private = {"seeds": {crop: 1}, "shed": {},
               "inventories": [{"FERTILIZER": 99} if fertilize else {}]}
    return farm, private


def engine_tile(crop, plant_day, *, harvest_age=None, last_day=SEASON_END,
                fertilize=False):
    """Plant `crop` on `plant_day`, water every day, and harvest at
    `harvest_age` (default: the age watering stops adding units).

    -> (harvest_day, units, curve) where `curve` is [(age, yield_units)] after
    each day's watering, and `harvest_day` is None when the harvest cannot land
    on or before `last_day`. The per-step `_decay_plants` and the end-of-day
    `_daily_refresh_plants` are both run, so a tile that rots or weeds before
    it can be picked reports it.

    ONE DRIVER FOR THE WHOLE SUITE. `test_rent_yields` imports this rather than
    keeping its own: its private copy hard-coded `max_yield_day` as both the
    harvest age and the occupancy, which is the defect this file exists for,
    and a second driver that shares the module's assumption can never
    contradict it.
    """
    farm, private = _fresh(crop, fertilize)
    ENG._apply_unit_action(farm, private, 0, ["PLANT", crop], 1, plant_day,
                           TURNS_PER_DAY)
    curve, harvest_day = [], None
    for day in range(plant_day, last_day + 1):
        age = day - plant_day
        if fertilize:
            ENG._apply_unit_action(farm, private, 0, ["FERTILIZE"], 1, day,
                                   TURNS_PER_DAY)
        ENG._apply_unit_action(farm, private, 0, ["WATER"], 1, day, TURNS_PER_DAY)
        tile = farm["tiles"][0][0]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            curve.append((age, tile["yield_units"]))
        if harvest_age is not None and age == harvest_age:
            ENG._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day,
                                   TURNS_PER_DAY)
            harvest_day = day
            break
        for step in range(day * TURNS_PER_DAY, (day + 1) * TURNS_PER_DAY):
            ENG._decay_plants(farm, step)
        ENG._daily_refresh_plants(farm, day, TURNS_PER_DAY)
    return harvest_day, private["inventories"][0].get(crop, 0), curve


def engine_ripe_age(crop):
    """The AGE at which watering stops adding units -- measured, not read off
    `max_yield_day`. Floored by `first_yield_day`, because the engine's HARVEST
    returns early while `day - planted_day < first_yield_day`."""
    _h, _u, curve = engine_tile(crop, 0)
    peak = max(y for _a, y in curve)
    return max(ENG.CROPS[crop]["first_yield_day"],
               min(a for a, y in curve if y >= peak))


def engine_last_plant_day(crop):
    """The last day a tile of `crop` can be planted and still be harvested by
    `SEASON_END`, found by planting it on every day and picking it."""
    ripe = engine_ripe_age(crop)
    last = None
    for day in range(0, SEASON_END + 1):
        harvest_day, units, _c = engine_tile(crop, day, harvest_age=ripe)
        if harvest_day is not None and harvest_day <= SEASON_END and units > 0:
            last = day
    return last


# --- the regression ----------------------------------------------------------

@pytest.mark.parametrize("crop", ONE_TIME)
def test_crop_block_is_offered_on_every_day_the_engine_still_allows(crop):
    """THE DEFECT, at the gate. `crop_block` returned None from day 18 for
    MELON; the engine plants it on 18 and 19 and hands back 6 units on days 28
    and 29. `features.LAST_PLANT` said 19 all along, and `demand_caps` takes
    the AND of the two, so the module's own table could not rescue it."""
    last = engine_last_plant_day(crop)
    assert last is not None
    assert last == features.LAST_PLANT[crop], (crop, last)

    for day in range(0, last + 1):
        assert rent.crop_block(crop, day, horizon=14) is not None, (crop, day)
    for day in range(last + 1, SEASON_END + 1):
        assert rent.crop_block(crop, day, horizon=14) is None, (crop, day)


@pytest.mark.parametrize("crop", ONE_TIME)
def test_crop_block_occupancy_and_lag_are_the_engines_ripe_age(crop):
    """THE DEFECT, at the price. `occupies` is what `block_rent` divides by and
    `k` is the day of the inventory path the block is priced at; both were the
    end of the bonus window. Melon's block is picked at age 10, so it holds the
    ground for 11 days and meets the market on day+10, not 13 and day+12."""
    ripe = engine_ripe_age(crop)
    _h, units, _c = engine_tile(crop, 0, harvest_age=ripe)

    k, block_units, occupies = rent.crop_block(crop, 0, horizon=14)
    assert block_units == float(units), (crop, block_units, units)
    assert occupies == float(ripe + 1), (crop, occupies, ripe + 1)
    assert k == min(14, ripe), (crop, k, ripe)


@pytest.mark.parametrize("crop", ONE_TIME)
def test_crop_block_rate_is_the_readme_column_for_melon_too(crop):
    """The module was internally inconsistent about the same tile, and the
    inconsistency was written down rather than measured: `demand_caps` sizes
    MELON's cap with `per_tile_day` = 0.545 (README "Yield / tile / day"
    0.55 = 6/11) while the water-fill it feeds scored that tile at 6/13 =
    0.462, an 18% disagreement."""
    _k, units, occupies = rent.crop_block(crop, 0, horizon=14)
    assert units / occupies == pytest.approx(rent.per_tile_day(crop), abs=1e-3)
    assert units / occupies == pytest.approx(features._PER_TILE_DAY[crop],
                                             abs=1e-3)


def test_demand_caps_still_offer_melon_on_days_18_and_19():
    """The consequence the agent actually pays: `demand_caps` gates on
    `rent.crop_block(crop, day) is None` and zeroes the crop, so under any
    lam > 0 preset MELON's cap was 0 on days 18 and 19 -- two days of a $250
    base, 6-units-a-tile one-time crop refused on days the engine allows."""
    last = features.LAST_PLANT["MELON"]
    for day in range(15, SEASON_END + 1):
        view = make_view(day=day, quads=("NW", "NE"), crops={}, opp_crops={})
        crops, _herd, dropped = allocation.demand_caps(view, "LEVEL")
        if day <= last:
            assert crops["MELON"] > 0, (day, crops["MELON"], dropped)
            assert "MELON" not in dropped, (day, dropped)
        else:
            assert crops["MELON"] == 0, (day, crops["MELON"])
            assert "MELON" in dropped, (day, dropped)


def test_melon_rent_is_the_eleven_day_rate():
    """The 18% understatement, on a priced board: a solo melon tile at day 6
    was worth 124.77 $/tile-day over 13 days and is worth ~147 over the 11 the
    engine charges. Asserted as a floor against the wrong denominator, not as
    a magic constant -- the revenue itself moves with the inventory path."""
    view = make_view(day=6, quads=("NW", "NE"), crops={}, opp_crops={})
    path = rent.inventory_path(view, horizon=14)
    revenue, occupies, units, _inv = rent.block_revenue(view, "MELON", 1, path,
                                                        day=6)
    assert units == 6
    assert occupies == 11.0
    rate = rent.block_rent(view, "MELON", 1, path, day=6)
    assert rate == pytest.approx(revenue / 11.0, rel=1e-9)
    assert rate > revenue / 13.0 * 1.15, (rate, revenue)
