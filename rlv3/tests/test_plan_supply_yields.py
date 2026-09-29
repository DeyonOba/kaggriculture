"""`objective._plan_supply`'s one-time unit count, checked against the ENGINE.

THE DEFECT. A planted one-time quota was credited `n * CROPS[crop]["max_yield"]`
-- the FERTILISED cap (wheat 6, carrot 4) -- while the engine's watered-only
peak is 4 and 3 and nothing in `_plan_supply` ever buys a fertilizer. Every
dawn `Rlv3PlanProvider.make_plan` ranks candidate doses with `better()` on
exactly this margin, so wheat quotas were over-credited 50% and carrot 33%,
both in the revenue booked and in `_overlay_path` -- a pure bias toward
one-time-crop quotas inside a comparator that is meant to be fitted-free
engine arithmetic.

WHY THE ORIGINAL TESTS MISSED IT.

  * `test_rent_yields.py` pinned the same constant, but only in `rlv3.rent`.
    `crop_block` was fixed there and `_plan_supply` -- a second, independent
    transcription of the same engine rule, in a different module -- was never
    looked at. A per-module fix leaves a per-module hole.
  * Every `_plan_supply` test in `test_objective.py` is a SHAPE test:
    `test_plan_overlay_adds_supply_and_charges_seeds` asserts the overlay
    raises our WHEAT part and moves our bank, and
    `test_our_own_plan_moves_the_OPPONENTS_bank_too` asserts the opponent's
    bank moves at all. Scaling wheat from 4 units to 6 leaves both properties
    intact -- the part still rises, the opponent still moves -- so a 50% error
    in the absolute yield passed them. The one LEVEL assertion in that file
    (`horizon_end=view.day + 1`) is the seed bill, $60, which the defect does
    not touch because the harvest falls outside that horizon.
  * Nothing ever checked the docstring's own claim, that `_plan_supply`
    "mirrors `kagent.planner._rent_board_supply`'s yield arithmetic exactly".
    It did not: the same three tiles came back 6/4/6 from one and 4/3/6 from
    the other.

So these tests do the one thing the shape tests cannot: they PLAY the tile
through `kaggriculture._apply_unit_action` (via `test_rent_yields`'s
`engine_watered_yield`, the one engine-grounded truth in the suite), and they
assert the two transcriptions agree both as counts and, through `project()`,
as dollars. No game is played.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_plan_supply_yields.py -q
"""

from __future__ import annotations

import pytest

from kagent.constants import BOARD_SIZE, CROPS, PRODUCTS
from kagent.gamestate import View
from kagent import planner as kplanner
from kagent.planner import DayPlan

from rlv3 import objective as O
from rlv3 import rent
from rlv3.tests.test_rent_yields import engine_watered_yield

ONE_TIME = ("WHEAT", "CARROT", "MELON")
HORIZON = 14
DAY = 10


# ------------------------------------------------------------------ fixtures

def _plant(crop, planted_day):
    """A tile as the ENGINE leaves it the instant the seed goes in.

    `kaggriculture._new_plant` starts a plant at `yield_units = 1`; that unit
    is the base the watering bonus is added to, and it is what makes a
    freshly planted tile the right board-side twin of a plan quota.
    """
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": False, "consecutive_unwatered": 0,
            "yield_units": 1, "max_lifespan_step": -1,
            "fertilized_until_day": -1}


def _obs(day=DAY, my_tiles=()):
    def board(spec):
        tiles = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for x, y, t in spec:
            tiles[y][x] = t
        return tiles

    def farm(money, spec):
        return {"money": money, "tiles": board(spec), "farmer": [4, 4],
                "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0}

    return {"player": 0, "step": day * 24, "day": day, "hour": 0,
            "farms": [farm(5000.0, my_tiles), farm(5000.0, ())],
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
            "market": {"inventory": {p: 10_000 for p in PRODUCTS},
                       "prices": {p: 100 for p in PRODUCTS}},
            "town": {"unlocked_shops": ["BAKERY"]}}


def _quota_supply(quota, day=DAY, horizon=HORIZON):
    plan = DayPlan()
    plan.plant_quota = dict(quota)
    return O._plan_supply(plan, View(_obs(day)), horizon)


def _tiles_for(quota, day=DAY):
    out, x, y = [], 0, 0
    for crop, n in quota.items():
        for _ in range(int(n)):
            out.append((x, y, _plant(crop, day)))
            x += 1
            if x >= BOARD_SIZE:
                x, y = 0, y + 1
    return out


# --------------------------------------------------------------------- tests

@pytest.mark.parametrize("crop", ONE_TIME)
@pytest.mark.parametrize("n", (1, 7))
def test_plan_supply_credits_the_engines_watered_only_yield(crop, n):
    """The defect itself: `n * cd["max_yield"]` is the FERTILISED cap.

    The engine hands back 4 wheat, 3 carrot and 6 melon on water alone
    (README "Object Types": "Wheat and Carrot only reach their listed Max
    Yield of 6 and 4 with fertilizer; watering alone peaks at 4 and 3").
    Pre-fix this asserted 6 / 4 / 6.
    """
    watered, _occ = engine_watered_yield(crop)
    supply, _cost = _quota_supply({crop: n})
    k = CROPS[crop]["max_yield_day"]
    assert supply[k].get(crop, 0) == n * watered, (crop, n, supply[k], watered)
    assert sum(row.get(crop, 0) for row in supply) == n * watered

    # and the fertilised cap really is the higher number for wheat/carrot, so
    # the fix is not a coincidence of renaming: max_yield is reachable, but
    # only by paying for a fertilizer this function never charges.
    if crop in ("WHEAT", "CARROT"):
        assert n * watered < n * CROPS[crop]["max_yield"]


@pytest.mark.parametrize("crop", ONE_TIME)
def test_plan_supply_agrees_with_rent_crop_block_on_UNITS(crop):
    """One engine rule, two modules. `rent.crop_block` was fixed and this was
    not, so the comparator and the allocator priced the same tile differently.

    UNITS only, deliberately. `crop_block`'s `k` is the RIPE age
    (`rent.one_time_shape`), which for melon is 10 rather than
    `max_yield_day` 12; `_plan_supply`'s `k` is `max_yield_day`, because this
    function's contract is to mirror `planner._rent_board_supply`, which is
    what the rest of `project()` prices the board with. That divergence is a
    separate question from the one this file is about.
    """
    spec = rent.crop_block(crop, DAY, horizon=HORIZON)
    assert spec is not None
    _k, units, _occ = spec
    supply, _cost = _quota_supply({crop: 1})
    assert sum(row.get(crop, 0) for row in supply) == units, (crop, units)


def test_plan_supply_mirrors_rent_board_supply_exactly():
    """`_plan_supply`'s docstring claims it "Mirrors
    `kagent.planner._rent_board_supply`'s yield arithmetic exactly". Pre-fix
    the SAME three tiles came back 6/4/6 from one and 4/3/6 from the other.
    """
    quota = {c: 1 for c in ONE_TIME}
    supply, _cost = _quota_supply(quota)
    tiles = View(_obs(my_tiles=_tiles_for(quota))).me.tiles
    board = kplanner._rent_board_supply(tiles, DAY, HORIZON)
    assert [dict(r) for r in supply] == [dict(r) for r in board]


def test_project_overlay_equals_the_same_tiles_standing_on_the_board():
    """The DOLLARS, not just the counts: a quota is worth exactly what the
    tiles it plants are worth, less the seed bill.

    This is the assertion `better()` actually consumes. Pre-fix the overlay
    booked 24 phantom wheat units and 8 phantom carrot units, inflating
    `our_bank` and depressing the inventory path they were priced against,
    with no matching change on the opponent's side -- a bias toward one-time
    quotas that no shape test could see.
    """
    quota = {"WHEAT": 12, "CARROT": 8}
    plan = DayPlan()
    plan.plant_quota = dict(quota)
    seed_bill = sum(CROPS[c]["seed"] * n for c, n in quota.items())

    overlay = O.project(View(_obs()), plan=plan)
    standing = O.project(View(_obs(my_tiles=_tiles_for(quota))))

    assert overlay.our_bank + seed_bill == pytest.approx(standing.our_bank)
    assert overlay.opp_bank == pytest.approx(standing.opp_bank)
    assert overlay.our_parts.keys() == standing.our_parts.keys()
    for item, v in standing.our_parts.items():
        assert overlay.our_parts[item] == pytest.approx(v), item
