"""Chore order: it must reorder chores ON A TILE and change nothing about
where anyone walks.

The per-quadrant priority queue lost 11.6 points because it scored tasks
priority-before-distance. `scheduler.pick` scores `(urgent, d, pri)` -- two
tasks on the SAME tile have identical `d` from every unit, so a `pri`
rewrite between them is a pure tie-break. These tests enforce that
structurally rather than trusting the docstring: the survival band is never
touched, no task crosses a band edge, and the priority multiset per tile is
PERMUTED rather than replaced.
"""

from __future__ import annotations

import pytest

from kagent.gamestate import View
from kagent.scheduler import (
    P_ANIMAL, P_FILLER, P_HARVEST, P_SURVIVAL, P_TEND, Task,
)
from rlv3 import choreorder, executor
from rlv3.tests.conftest import animal_tile, blank_obs, plant_tile, simple_plan

ALL_OFF = {f: False for f in executor.RLV3_FLAGS}


def ctx_for(flags, **kw):
    f = dict(ALL_OFF)
    f.update(flags)
    out = {"flags": f, "params": dict(executor.PARAMS), "lead": 0.0,
           "regime": "even", "opp_standing": {}, "stats": None}
    out.update(kw)
    return out


def animal_view(**kw):
    obs = blank_obs(day=12, quadrants=("NW",))
    obs["farms"][0]["tiles"][2][2] = animal_tile("PASTURE", "COW", **kw)
    return View(obs)


def crop_view(**kw):
    obs = blank_obs(day=12, quadrants=("NW",))
    obs["farms"][0]["tiles"][2][2] = plant_tile("WHEAT", **kw)
    return View(obs)


def pri_at(tasks, op):
    return next(t.pri for t in tasks if t.op[0] == op)


# ------------------------------------------------------------- pure parts
def test_tile_groups_preserve_order():
    tasks = [Task(2, 1, 1, ["WATER"]), Task(1, 2, 2, ["FEED"]),
             Task(2, 1, 1, ["HARVEST"])]
    g = choreorder.tile_groups(tasks)
    assert [t.op[0] for t in g[(1, 1)]] == ["WATER", "HARVEST"]
    assert [t.op[0] for t in g[(2, 2)]] == ["FEED"]


def test_chain_order_animal_is_the_engine_dependency_order():
    view = animal_view()
    group = [Task(P_TEND, 2, 2, ["HARVEST"]),
             Task(P_TEND, 2, 2, ["COLLECT_FERTILIZER"]),
             Task(P_ANIMAL, 2, 2, ["FEED"], need="WHEAT")]
    order = choreorder.chain_order(view, simple_plan(), (2, 2), group)
    assert [group[i].op[0] for i in order] == ["FEED", "COLLECT_FERTILIZER", "HARVEST"]


def test_chain_order_crop_waters_before_it_harvests():
    """The same engine fact that made `lastwater` worth $2.1k a game: an
    in-window WATER adds a yield unit to the crop you are about to take."""
    view = crop_view()
    group = [Task(P_TEND, 2, 2, ["HARVEST"]), Task(P_TEND, 2, 2, ["WATER"])]
    order = choreorder.chain_order(view, simple_plan(), (2, 2), group)
    assert [group[i].op[0] for i in order] == ["WATER", "HARVEST"]


# ------------------------------------------------------------- reordering
def test_co_chain_tile_reorders_an_animal_tile():
    view = animal_view()
    tasks = [Task(P_TEND, 2, 2, ["HARVEST"]),
             Task(P_TEND, 2, 2, ["COLLECT_FERTILIZER"]),
             Task(P_ANIMAL, 2, 2, ["FEED"], need="WHEAT")]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True}))
    assert pri_at(out, "FEED") < pri_at(out, "COLLECT_FERTILIZER") < pri_at(out, "HARVEST")
    # the multiset is permuted, not replaced
    assert sorted(round(t.pri, 3) for t in out) == sorted(t.pri for t in tasks)


def test_co_chain_tile_reorders_a_crop_tile_water_before_harvest():
    view = crop_view()
    tasks = [Task(P_TEND, 2, 2, ["HARVEST"]), Task(P_TEND, 2, 2, ["WATER"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True}))
    assert pri_at(out, "WATER") < pri_at(out, "HARVEST")


def test_co_animal_first_chains_animal_tiles_only():
    """The separable half: animal chores per visit is 1.617 against 2.773,
    the crop half 1.267 against 1.764. Gating them apart is what says which
    half carries an effect."""
    obs = blank_obs(day=12, quadrants=("NW",))
    obs["farms"][0]["tiles"][2][2] = animal_tile("PASTURE", "COW")
    obs["farms"][0]["tiles"][4][4] = plant_tile("WHEAT")
    view = View(obs)
    tasks = [Task(P_TEND, 2, 2, ["HARVEST"]),
             Task(P_ANIMAL, 2, 2, ["FEED"], need="WHEAT"),
             Task(P_TEND, 4, 4, ["HARVEST"]), Task(P_TEND, 4, 4, ["WATER"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_animal_first": True}))
    animal = [t for t in out if (t.x, t.y) == (2, 2)]
    crop = [t for t in out if (t.x, t.y) == (4, 4)]
    assert next(t.pri for t in animal if t.op[0] == "FEED") < \
        next(t.pri for t in animal if t.op[0] == "HARVEST")
    # the crop tile is untouched by this flag
    assert [t.pri for t in crop] == [P_TEND, P_TEND]


def test_all_flags_off_is_an_exact_pass_through():
    view = animal_view()
    tasks = [Task(P_TEND, 2, 2, ["HARVEST"]), Task(P_ANIMAL, 2, 2, ["FEED"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks, ctx_for({}))
    assert [(t.pri, t.op) for t in out] == [(t.pri, t.op) for t in tasks]


# ---------------------------------------------------- structural defences
def test_the_survival_band_is_never_touched():
    view = animal_view(consecutive_unfed=1)
    tasks = [Task(P_SURVIVAL, 2, 2, ["FEED"], need="WHEAT"),
             Task(P_SURVIVAL, 2, 2, ["WATER"]),
             Task(P_TEND, 2, 2, ["HARVEST"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True, "co_value": True}))
    surv = [t for t in out if t.pri == P_SURVIVAL]
    assert len(surv) == 2
    assert all(t.pri == P_SURVIVAL for t in out if t.op[0] in ("FEED", "WATER"))


def test_no_task_crosses_a_band_edge():
    view = crop_view()
    tasks = [Task(P_HARVEST, 2, 2, ["HARVEST"]), Task(P_FILLER, 2, 2, ["WATER"]),
             Task(P_FILLER + 1, 2, 2, ["DIG"]), Task(P_TEND, 2, 2, ["FERTILIZE"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True}))
    for t in out:
        if t.op[0] in ("HARVEST", "FERTILIZE"):
            assert P_SURVIVAL < t.pri < P_FILLER
        else:
            assert t.pri >= P_FILLER


def test_a_lone_task_on_a_tile_is_never_rewritten():
    view = crop_view()
    tasks = [Task(P_TEND, 2, 2, ["WATER"]), Task(P_TEND, 3, 3, ["WATER"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True, "co_value": True}))
    assert [t.pri for t in out] == [P_TEND, P_TEND]


def test_the_epsilon_is_small_enough_to_only_break_exact_ties():
    view = animal_view()
    tasks = [Task(P_TEND, 2, 2, ["HARVEST"]),
             Task(P_TEND, 2, 2, ["COLLECT_FERTILIZER"])]
    out = choreorder.reprioritise(view, simple_plan(), tasks,
                                  ctx_for({"co_chain_tile": True}))
    assert all(abs(t.pri - P_TEND) < 1e-3 for t in out)
    assert out[0].pri != out[1].pri


def test_invariants_hold_on_mid_season_states(cases):
    from kagent import scheduler
    for flag in ("co_chain_tile", "co_animal_first", "co_value"):
        for label, view, plan in cases:
            tasks = scheduler.build_tasks(view, plan)
            out = choreorder.reprioritise(view, plan, tasks, ctx_for({flag: True}))
            executor.check_invariants(view, tasks, out, strict=True)
            assert len(out) == len(tasks), f"{flag} changed the count on {label}"


# --------------------------------------------- the lexicographic objective
def test_value_weight_reads_the_market_and_the_opponents_board():
    view = animal_view()
    plan = simple_plan()
    task = Task(P_HARVEST, 2, 2, ["HARVEST"])
    clean = choreorder.value_weight(view, plan, task, 0.0, "even", {})
    flooding = choreorder.value_weight(view, plan, task, 0.0, "even", {"MILK": 20})
    assert flooding < clean, "a product they are about to flood must be tended later"


def test_contest_regime_flips_the_sign_toward_their_supply():
    """Behind and late, the market is ONE shared inventory and the score is
    relative: race them to the products they hold standing supply in. This
    is a deliberate sign flip against the `even` branch."""
    view = animal_view()
    plan = simple_plan()
    task = Task(P_HARVEST, 2, 2, ["HARVEST"])
    clean = choreorder.value_weight(view, plan, task, -30_000, "contest", {})
    contested = choreorder.value_weight(view, plan, task, -30_000, "contest", {"MILK": 20})
    assert contested > clean


def test_ahead_regime_suppresses_the_value_tilt():
    view = animal_view()
    task = Task(P_HARVEST, 2, 2, ["HARVEST"])
    assert choreorder.value_weight(view, simple_plan(), task, 5_000, "ahead", {}) == 0.0


def test_salvage_regime_prices_only_realisable_cash():
    """Objective 3: minimise the loss margin. A WATER that pays on day 31 is
    worth nothing; a HARVEST holding units is worth its units."""
    plan = simple_plan()
    empty = animal_view(yield_units=0)
    full = animal_view(yield_units=4)
    task = Task(P_HARVEST, 2, 2, ["HARVEST"])
    water = Task(P_TEND, 2, 2, ["CARE"])
    assert choreorder.value_weight(empty, plan, task, -40_000, "salvage", {}) == 0.0
    assert choreorder.value_weight(full, plan, task, -40_000, "salvage", {}) > 0.0
    assert choreorder.value_weight(full, plan, water, -40_000, "salvage", {}) == 0.0


@pytest.mark.parametrize("lead,day,want", [
    (-40_000, 26, "salvage"),
    (-5_000, 20, "contest"),
    (-5_000, 10, "even"),
    (5_000, 20, "ahead"),
    (0, 20, "even"),
])
def test_regime_selection_is_lexicographic(lead, day, want):
    from rlv3 import ledger
    led = ledger.ExecLedger()
    led.lead = lead
    view = animal_view()
    view.day = day
    assert led.regime(view, executor.PARAMS) == want


# ----------------------------------------------- the guard's comparison key
# REGRESSION: the guard compared the desired order against `band`, the tasks'
# position in the TASK LIST, instead of the order they will EXECUTE in.
#
# WHY EVERY TEST ABOVE MISSED IT. They all hand-build the group, and they
# hand-build it in an order that is NOT the chain -- `[HARVEST,
# COLLECT_FERTILIZER, FEED]` -- so `order != band` and the guard falls
# through by accident. `scheduler.build_tasks` appends a real animal tile
# FEED, COLLECT_FERTILIZER, HARVEST, which IS the chain order, at priorities
# 1.5 / 2.0 / 1.0. `assign._underfoot` selects with `_t.pri < _best.pri` and
# `pick` scores `(urgent, d, task.pri)`, so that tile executes HARVEST, FEED,
# COLLECT_FERTILIZER -- wrong -- while the guard, reading append order, said
# "nobody asked for a different order here" and skipped it. The levers were
# therefore inert on precisely the tiles they exist to fix: 0 of 179 animal
# groups rewritten across the 18 recorded leader mid-season boards, 79 of
# which executed out of chain order. So these tests build the tasks with the
# real `scheduler.build_tasks` and check the REORDER COUNTER and the action
# the real `scheduler.assign` emits, never a hand-built list.

def cow_view_with_chores():
    """One COW holding milk with fertilizer waiting, farmer standing on it.

    `build_tasks` yields FEED (P_ANIMAL), COLLECT_FERTILIZER (P_TEND) and
    HARVEST (P_HARVEST) on this tile -- append order == chain order, exec
    order != chain order, which is the shape the guard got wrong."""
    obs = blank_obs(day=12, quadrants=("NW",), n_hands=0)
    obs["farms"][0]["tiles"][2][2] = animal_tile(
        "PASTURE", "COW", yield_units=2, fertilizer_available=True)
    obs["farms"][0]["farmer"] = [2, 2]
    obs["private"]["inventories"] = [{"WHEAT": 5}]
    return View(obs)


def test_build_tasks_appends_an_animal_tile_in_chain_order_at_non_chain_pri():
    """The premise of the regression, asserted against the shipped scheduler
    so the test fails loudly if `build_tasks` ever changes shape."""
    from kagent import scheduler
    view = cow_view_with_chores()
    tile = [t for t in scheduler.build_tasks(view, simple_plan())
            if (t.x, t.y) == (2, 2)]
    assert [t.op[0] for t in tile] == ["FEED", "COLLECT_FERTILIZER", "HARVEST"]
    assert [t.pri for t in tile] == [P_ANIMAL, P_TEND, P_HARVEST]
    # append order IS the chain; execution order (ascending pri) is NOT
    assert sorted(tile, key=lambda t: t.pri)[0].op[0] == "HARVEST"


@pytest.mark.parametrize("flag", ["co_chain_tile", "co_animal_first"])
def test_chain_lever_fires_on_a_scheduler_built_animal_tile(flag):
    """Non-zero `reordered` counter on tasks from `scheduler.build_tasks`.

    Pre-fix this counter was 0 for `co_animal_first` on every animal tile in
    the repo, which is the DEAD-not-null failure this package exists to
    prevent."""
    import collections
    from kagent import scheduler
    view = cow_view_with_chores()
    plan = simple_plan()
    tasks = scheduler.build_tasks(view, plan)
    stats = collections.Counter()
    out = choreorder.reprioritise(view, plan, tasks, ctx_for({flag: True}, stats=stats))
    assert stats[f"{flag}.reordered"] > 0, f"{flag} rewrote nothing on a real animal tile"
    assert pri_at(out, "FEED") < pri_at(out, "COLLECT_FERTILIZER") < pri_at(out, "HARVEST")
    assert sorted(round(t.pri, 3) for t in out) == sorted(t.pri for t in tasks)


def test_the_lever_changes_the_action_the_real_assign_emits():
    """Run the code, do not reason about it: the shipped executor takes
    HARVEST off this tile, the chained one takes FEED."""
    from kagent import scheduler
    view = cow_view_with_chores()
    plan = simple_plan()
    tasks = scheduler.build_tasks(view, plan)
    assert scheduler.assign(view, plan, tasks)[0] == ["HARVEST"]
    out = choreorder.reprioritise(view, plan, tasks, ctx_for({"co_chain_tile": True}))
    assert scheduler.assign(view, plan, out)[0] == ["FEED"], \
        "co_chain_tile did not change what the unit standing on the tile does"


@pytest.mark.parametrize("flag", ["co_chain_tile", "co_animal_first"])
def test_chain_levers_are_not_inert_on_the_recorded_mid_season_boards(flag, cases):
    """The measurement that exposed it: across the 18 recorded leader boards
    `co_animal_first` rewrote 0 of 179 animal tile groups while 79 executed
    out of chain order. Counts the groups whose EXECUTION order differs from
    the engine chain and requires the lever to fire on every one of them."""
    import collections
    from kagent import scheduler
    wrong = fired = 0
    for label, view, plan in cases:
        tasks = scheduler.build_tasks(view, plan)
        stats = collections.Counter()
        choreorder.reprioritise(view, plan, tasks, ctx_for({flag: True}, stats=stats))
        fired += stats[f"{flag}.reordered"]
        f = dict(ALL_OFF)
        f[flag] = True
        for xy, idxs in choreorder.tile_group_indices(tasks).items():
            group = [tasks[i] for i in idxs]
            if len(group) < 2 or not choreorder._chains_here(view, xy, f):
                continue
            work = [i for i in range(len(group)) if P_SURVIVAL < group[i].pri < P_FILLER]
            if len(work) < 2:
                continue
            chain = [i for i in choreorder.chain_order(view, plan, xy, group) if i in set(work)]
            if chain != sorted(work, key=lambda i: (group[i].pri, i)):
                wrong += 1
    assert wrong > 0, "no board in the fixture exercises an out-of-chain tile"
    assert fired >= wrong, f"{flag} fired on {fired} of {wrong} out-of-chain tile groups"
