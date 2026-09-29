"""Test fixtures for the rlv3 executor seam.

MID-SEASON STATES, NOT FRESH ONES. `kaggriculture-land-purchase-paths` says
to write gate tests on mid-season states: Stage B5 ran three hours with two
of three land paths gated and the capped control still held 3.8 quadrants,
because the test had only ever looked at a fresh board. The views here come
out of the three leaders' recorded replays at days 6, 12 and 20, both seats,
so a rewrite is exercised against real boards with animals standing, plants
part-grown, weeds, locked quadrants and full sheds.

Replays are used rather than played episodes on purpose: a replay carries
the complete per-step observation including `private`, so a mid-season View
costs a JSON read instead of a game. The machine is busy and no test here
plays anything.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from kagent.gamestate import Memory, View
from kagent import planner

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAMS = ("mmpq", "boey", "dsm")
DAYS = (6, 12, 20)
HOUR = 8


def _replay_paths():
    out = []
    for team in TEAMS:
        hits = sorted(glob.glob(os.path.join(REPO, "replays", team, "*.json")))
        if hits:
            out.append((team, hits[0]))
    return out


def _view_at(steps, k, seat):
    obs = dict(steps[k][seat]["observation"])
    obs.setdefault("step", k)
    return View(obs)


def _plan_for(view):
    memory = Memory()
    memory.observe(view)
    return planner.make_day_plan(view, memory, speculative=True)


def _load_cases():
    cases = []
    for team, path in _replay_paths():
        with open(path) as fh:
            steps = json.load(fh)["steps"]
        for day in DAYS:
            k = day * 24 + HOUR
            if k >= len(steps):
                continue
            for seat in (0, 1):
                try:
                    view = _view_at(steps, k, seat)
                    cases.append((f"{team}-d{day}-s{seat}", view, _plan_for(view)))
                except Exception:
                    continue
    return cases


_CASES = None


def mid_season_cases():
    """-> [(label, View, DayPlan)] -- cached, three recorded sessions."""
    global _CASES
    if _CASES is None:
        _CASES = _load_cases()
    return _CASES


@pytest.fixture(scope="session")
def cases():
    cs = mid_season_cases()
    if not cs:
        pytest.skip("no replays under replays/{mmpq,boey,dsm}")
    return cs


# ------------------------------------------------------- synthetic boards
def blank_obs(player=0, day=12, hour=8, quadrants=("NW", "NE", "SW", "SE"),
              n_hands=3, money=50_000, opp_money=50_000):
    """A 10x10 observation with every unlocked tile bare -- the controlled
    board the placement tests put structures and plants onto by hand."""
    def farm(m):
        tiles = []
        for y in range(10):
            row = []
            for x in range(10):
                q = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
                row.append(None if q in quadrants else "LOCKED")
            tiles.append(row)
        return {"money": m, "tiles": tiles, "farmer": [4, 4],
                "hands": [[4, 5]] * n_hands, "hires_today": 0,
                "unlocked_quadrants": list(quadrants)}

    farms = [farm(money), farm(opp_money)]
    if player == 1:
        farms = [farm(opp_money), farm(money)]
    return {
        "player": player, "step": day * 24 + hour, "day": day, "hour": hour,
        "farms": farms,
        "private": {"shed": {"WHEAT": 40, "FERTILIZER": 20},
                    "seeds": {"WHEAT": 40, "STRAWBERRY": 40, "CARROT": 40},
                    "inventories": [{} for _ in range(n_hands + 1)]},
        "market": {"inventory": {k: 10_000 for k in
                                 ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                                  "EGG", "MILK", "WOOL", "FERTILIZER")},
                   "prices": {}},
        "town": {"unlocked_shops": ["BAKERY", "PET_CAFE"]},
    }


def plant_tile(crop="STRAWBERRY", planted_day=7, **kw):
    t = {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
         "yield_units": 0, "watered_today": True, "consecutive_unwatered": 0,
         "fertilized_until_day": -1, "max_lifespan_step": -1}
    t.update(kw)
    return t


def animal_tile(kind="PASTURE", animal="COW", placed_day=2, **kw):
    t = {"kind": kind, "animal": animal, "placed_day": placed_day,
         "yield_units": 0, "fed_today": False, "cared_today": False,
         "consecutive_unfed": 0, "fertilizer_available": False,
         "pending_care_bonus": 0}
    t.update(kw)
    return t


def simple_plan(quota=None, **kw):
    plan = planner.DayPlan()
    plan.plant_quota = dict(quota or {})
    for k, v in kw.items():
        setattr(plan, k, v)
    return plan
