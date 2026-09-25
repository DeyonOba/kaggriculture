"""Cull knobs on the champion plan: off by default (the shipped plan is
unchanged), cows culled on a milk collapse, sheep on a wool collapse or in
the endgame, geese never; the scheduler then issues no FEED/CARE for culled
kinds, and the plan stops buying them.

    PYTHONPATH=. uv run pytest tests/test_cull.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import planner, scheduler  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _animal(kind):
    # FarmView.animals() yields OCCUPIED STRUCTURES: kind COOP/PASTURE + animal
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": False, "cared_today": False,
            "fertilizer_available": False, "yield_units": 0, "consecutive_unfed": 0}


def _obs(day, milk, wool):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 31}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["step"], obs["hour"] = day, day * 24, 0
    farm = obs["farms"][0]
    farm["money"] = 20_000
    farm["tiles"][0][0] = _animal("COW"); farm["tiles"][0][1] = _animal("COW")
    farm["tiles"][1][0] = _animal("SHEEP"); farm["tiles"][1][1] = _animal("SHEEP")
    farm["tiles"][2][0] = _animal("GOOSE")
    obs["market"]["prices"]["MILK"] = milk
    obs["market"]["prices"]["WOOL"] = wool
    return obs


def _plan(obs, knobs):
    mem = Memory(); mem.knobs = knobs
    view = View(obs)
    return view, planner.make_champion_plan(view, mem)


def _fed_kinds(view, plan):
    kinds = set()
    for t in scheduler.build_tasks(view, plan):
        if t.op[0] in ("FEED", "CARE"):
            tile = view.me.tiles[t.y][t.x]
            kinds.add(tile.get("animal"))
    return kinds


def test_default_knobs_cull_nothing():
    view, plan = _plan(_obs(16, milk=10.0, wool=20.0), {})
    assert plan.cull_kinds == set()
    assert {"COW", "SHEEP", "GOOSE"} <= _fed_kinds(view, plan)


def test_milk_collapse_culls_cows_only_and_stops_buying_them():
    view, plan = _plan(_obs(16, milk=20.0, wool=200.0), {"cull_milk_ratio": 0.35})
    assert plan.cull_kinds == {"COW"}
    fed = _fed_kinds(view, plan)
    assert "COW" not in fed and {"SHEEP", "GOOSE"} <= fed
    assert "COW" not in plan.animal_buys
    # not below the ratio -> no cull
    view, plan = _plan(_obs(16, milk=100.0, wool=200.0), {"cull_milk_ratio": 0.35})
    assert plan.cull_kinds == set()


def test_price_culls_wait_for_day_12():
    view, plan = _plan(_obs(8, milk=10.0, wool=10.0), {"cull_milk_ratio": 0.35, "cull_wool_ratio": 0.5})
    assert plan.cull_kinds == set()


def test_wool_collapse_culls_sheep():
    view, plan = _plan(_obs(14, milk=160.0, wool=60.0), {"cull_wool_ratio": 0.5})
    assert plan.cull_kinds == {"SHEEP"}
    assert "SHEEP" not in _fed_kinds(view, plan)


def test_endgame_cull_releases_ruminants_keeps_geese():
    view, plan = _plan(_obs(26, milk=160.0, wool=200.0), {"endgame_cull_day": 25})
    assert plan.cull_kinds == {"COW", "SHEEP"}
    fed = _fed_kinds(view, plan)
    assert fed == {"GOOSE"}
    # the wheat reserve counts only the animals still fed (1 goose)
    assert plan.wheat_reserve == max(1 + 2, plan.feed_days * 1)
    view, plan = _plan(_obs(24, milk=160.0, wool=200.0), {"endgame_cull_day": 25})
    assert plan.cull_kinds == set()
