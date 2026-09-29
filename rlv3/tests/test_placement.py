"""Placement: does the lever fire, does it fire only where the data says
there is a gap, and does it leave the planner's allocation alone.

Check (c) of the risk list is a BEHAVIOURAL trace, not a metric: confirm the
intended switch actually fires before spending an evaluation. These tests
are that check at the unit level, on a controlled board where the answer is
known by construction, so a 995-game arm is never spent on a lever that was
silently inert.
"""

from __future__ import annotations

import pytest

from kagent.gamestate import View
from kagent.scheduler import Task, P_GROW
from rlv3 import executor, ledger, placement
from rlv3.tests.conftest import animal_tile, blank_obs, plant_tile, simple_plan

ALL_OFF = {f: False for f in executor.RLV3_FLAGS}


def ctx_for(view, plan, tasks, flags, **params):
    p = dict(executor.PARAMS)
    p.update(params)
    f = dict(ALL_OFF)
    f.update(flags)
    return placement.make_ctx(view, plan, tasks, f, p)


def wheat_board():
    """A board with two structures in the NW corner, everything else bare,
    and a wheat + strawberry quota. Only four tiles are orthogonal to a
    structure, so `pl_wheat_adj` has exactly one right answer."""
    obs = blank_obs(day=12, quadrants=("NW",), n_hands=3)
    tiles = obs["farms"][0]["tiles"]
    tiles[0][0] = animal_tile("PASTURE", "COW")      # (0, 0)
    tiles[3][3] = animal_tile("COOP", "GOOSE")       # (3, 3)
    view = View(obs)
    plan = simple_plan({"WHEAT": 2, "STRAWBERRY": 6})
    return view, plan


def growth_tiles(tasks, op):
    return [(t.x, t.y) for t in tasks if t.op[0] == op]


def crop_of(tasks):
    return {(t.x, t.y): t.op[1] for t in tasks if t.op[0] == "PLANT"}


# ------------------------------------------------------------- pure parts
def test_split_growth_and_crop_slots():
    tasks = [Task(2, 1, 1, ["WATER"]), Task(P_GROW, 2, 2, ["PLANT", "WHEAT"]),
             Task(P_GROW, 3, 3, ["PLANT", "STRAWBERRY"]),
             Task(P_GROW, 4, 4, ["BUILD_COOP"]), Task(2, 5, 5, ["FEED"], need="WHEAT")]
    growth, rest = placement.split_growth(tasks)
    assert [t.op[0] for t in growth] == ["PLANT", "PLANT", "BUILD_COOP"]
    assert [t.op[0] for t in rest] == ["WATER", "FEED"]
    assert placement.crop_slots(growth) == ["WHEAT", "STRAWBERRY"]
    assert placement.candidate_tiles(growth, "PLANT") == [(2, 2), (3, 3)]
    assert placement.candidate_tiles(growth, "BUILD_COOP") == [(4, 4)]


def test_zero_weights_reproduce_the_scheduler_exactly():
    """The identity property assign_crops is built around: every weight at
    zero and the greedy must return the original pairing, tile for tile."""
    view, plan = wheat_board()
    tasks = executor._sched_mod.build_tasks(view, plan) if executor.is_installed() \
        else __import__("kagent.scheduler", fromlist=["x"]).build_tasks(view, plan)
    tiles = growth_tiles(tasks, "PLANT")
    crops = [t.op[1] for t in tasks if t.op[0] == "PLANT"]
    ctx = ctx_for(view, plan, tasks, {"pl_wheat_adj": True},
                  w_wheat_adj=0.0, w_cluster=0.0, w_water_first=0.0)
    pairs = placement.assign_crops(view, plan, tiles, crops, ctx)
    assert pairs == list(zip(tiles, crops))


def test_adjacency_sets_read_live_structures():
    view, plan = wheat_board()
    adj = placement.adjacency_sets(view, [])
    assert (1, 0) in adj["near_structure"] and (0, 1) in adj["near_structure"]
    assert (3, 2) in adj["near_structure"] and (2, 3) in adj["near_structure"]
    assert (2, 2) not in adj["near_structure"]


def test_near_pending_only_counts_non_filler_work():
    from kagent.scheduler import P_FILLER
    view, _ = wheat_board()
    tasks = [Task(2, 5, 5, ["WATER"]), Task(P_FILLER, 8, 8, ["WATER"])]
    adj = placement.adjacency_sets(view, tasks)
    assert (5, 4) in adj["near_pending"] and (5, 4) in adj["near_water_work"]
    assert (8, 7) not in adj["near_pending"]


# ------------------------------------------------------------ the levers
def test_pl_wheat_adj_moves_wheat_beside_a_structure():
    """THE placement gap the data supports: wheat beside an animal is 0.09
    for us against 0.36-0.46 for the leaders."""
    from kagent import scheduler
    view, plan = wheat_board()
    tasks = scheduler.build_tasks(view, plan)
    near = placement.adjacency_sets(view, tasks)["near_structure"]

    base = crop_of(tasks)
    ctx = ctx_for(view, plan, tasks, {"pl_wheat_adj": True})
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    new = crop_of(out)

    bare_near = [xy for xy in new if xy in near]
    assert bare_near, "the board has no structure-adjacent bare tile to claim"
    # every structure-adjacent bare tile goes to WHEAT, and no other crop
    # takes one while a wheat slot sits off one -- the reservation must not
    # leak through the greedy's fallback
    assert all(new[xy] == "WHEAT" for xy in bare_near), \
        f"a non-wheat crop took a structure tile: " \
        f"{[(xy, new[xy]) for xy in bare_near if new[xy] != 'WHEAT']}"
    was = sum(1 for xy in base if xy in near and base[xy] == "WHEAT")
    now = sum(1 for xy in bare_near if new[xy] == "WHEAT")
    assert now > was, f"the lever was inert on a board built for it ({was} -> {now})"
    # and the mix is untouched
    assert sorted(new.values()) == sorted(base.values())


def test_a_counter_never_names_a_flag_that_is_off():
    """Regression. A traced game reported `pl_water_first.moved: 143` on an
    arm where pl_water_first was OFF: the knock-on moves of the wheat
    repulsion were filed under an unrelated lever. The activation check is
    "if the counter is zero the arm is dead", so a counter that credits the
    wrong flag makes an arm unreadable in the other direction."""
    import collections
    from kagent import scheduler
    view, plan = wheat_board()
    tasks = scheduler.build_tasks(view, plan)
    for flag in ("pl_wheat_adj", "pl_crop_cluster", "pl_water_first"):
        stats = collections.Counter()
        ctx = ctx_for(view, plan, tasks, {flag: True})
        ctx["stats"] = stats
        placement.replace_growth_tasks(view, plan, tasks, None, ctx)
        for key in stats:
            named = key.split(".")[0]
            assert named in (flag, "placement"), \
                f"arm {flag} credited {key}, but {named} is off"
    # and the wheat arm's knock-on moves land in their own bucket
    stats = collections.Counter()
    ctx = ctx_for(view, plan, tasks, {"pl_wheat_adj": True})
    ctx["stats"] = stats
    placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    assert stats["pl_wheat_adj.moved"] > 0
    assert set(stats) <= {"pl_wheat_adj.moved", "pl_wheat_adj.displaced"}


def test_pl_wheat_adj_is_inert_when_every_animal_is_fed():
    """It claims bare tiles for wheat only while there are unfed animals for
    that wheat to serve -- otherwise it is paying for nothing."""
    from kagent import scheduler
    obs = blank_obs(day=12, quadrants=("NW",))
    obs["farms"][0]["tiles"][0][0] = animal_tile("PASTURE", "COW", fed_today=True,
                                                 cared_today=True)
    obs["farms"][0]["tiles"][3][3] = animal_tile("COOP", "GOOSE", fed_today=True,
                                                 cared_today=True)
    view = View(obs)
    plan = simple_plan({"WHEAT": 2, "STRAWBERRY": 6})
    tasks = scheduler.build_tasks(view, plan)
    ctx = ctx_for(view, plan, tasks, {"pl_wheat_adj": True})
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    assert crop_of(out) == crop_of(tasks)


def test_pl_crop_cluster_seats_kin_together():
    from kagent import scheduler
    obs = blank_obs(day=12, quadrants=("NW", "NE"))
    obs["farms"][0]["tiles"][0][0] = plant_tile("STRAWBERRY")     # (0, 0)
    obs["farms"][0]["tiles"][1][1] = plant_tile("STRAWBERRY")     # (1, 1)
    view = View(obs)
    plan = simple_plan({"STRAWBERRY": 3})
    tasks = scheduler.build_tasks(view, plan)
    ctx = ctx_for(view, plan, tasks, {"pl_crop_cluster": True}, cluster_radius=2.0)
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)

    kin = [(0, 0), (1, 1)]
    placed = [xy for xy, c in crop_of(out).items() if c == "STRAWBERRY"]
    near = [xy for xy in placed
            if min(abs(xy[0] - k[0]) + abs(xy[1] - k[1]) for k in kin) <= 2]
    base = [xy for xy, c in crop_of(tasks).items() if c == "STRAWBERRY"]
    base_near = [xy for xy in base
                 if min(abs(xy[0] - k[0]) + abs(xy[1] - k[1]) for k in kin) <= 2]
    assert len(near) >= len(base_near)
    assert len(near) > 0


def test_cluster_radius_zero_makes_the_lever_inert():
    """The within-episode kill switch: at radius 0 the flag does nothing."""
    from kagent import scheduler
    obs = blank_obs(day=12, quadrants=("NW", "NE"))
    obs["farms"][0]["tiles"][0][0] = plant_tile("STRAWBERRY")
    view = View(obs)
    plan = simple_plan({"STRAWBERRY": 3})
    tasks = scheduler.build_tasks(view, plan)
    ctx = ctx_for(view, plan, tasks, {"pl_crop_cluster": True}, cluster_radius=0.0)
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    assert crop_of(out) == crop_of(tasks)


def test_pl_water_first_seats_plants_beside_pending_work():
    from kagent import scheduler
    obs = blank_obs(day=12, quadrants=("NW", "NE"))
    # a thirsty plant far from the centre creates real pending work at (9, 0)
    obs["farms"][0]["tiles"][0][9] = plant_tile("STRAWBERRY", watered_today=False,
                                                consecutive_unwatered=0)
    view = View(obs)
    plan = simple_plan({"WHEAT": 4})
    tasks = scheduler.build_tasks(view, plan)
    pend = placement.adjacency_sets(view, tasks)["near_pending"]
    ctx = ctx_for(view, plan, tasks, {"pl_water_first": True})
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    got = [xy for xy in crop_of(out) if xy in pend]
    was = [xy for xy in crop_of(tasks) if xy in pend]
    assert len(got) >= len(was)
    assert got, "no plant landed beside pending work on a board built for it"


# -------------------------------------------------------- the negatives
def test_pl_pasture_ring_pushes_structures_away_from_the_shed():
    """The DIAGNOSTIC NEGATIVE. It exists to reproduce the known −$11.3k and
    prove the seam can move placement at all. It must never ship."""
    from kagent import scheduler
    obs = blank_obs(day=6, quadrants=("NW", "NE", "SW", "SE"))
    view = View(obs)
    plan = simple_plan({"WHEAT": 4}, structure_builds={"COOP": 1, "PASTURE": 2})
    tasks = scheduler.build_tasks(view, plan)
    base = [placement._shed_dist(xy) for xy in growth_tiles(tasks, "BUILD_PASTURE")]
    ctx = ctx_for(view, plan, tasks, {"pl_pasture_ring": True})
    out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
    got = [placement._shed_dist(xy) for xy in growth_tiles(out, "BUILD_PASTURE")]
    assert got and sum(got) > sum(base), f"ring inert: {base} -> {got}"


def test_animal_slots_reproduces_the_schedulers_own_pairing():
    """PLACE tiles are never repooled; this proves the pass-through IS what
    build_tasks would have done."""
    from kagent import scheduler
    obs = blank_obs(day=6, quadrants=("NW",))
    obs["farms"][0]["tiles"][1][1] = {"kind": "COOP"}
    obs["farms"][0]["tiles"][2][2] = {"kind": "PASTURE"}
    obs["private"]["shed"].update({"GOOSE": 1, "COW": 1})
    view = View(obs)
    plan = simple_plan({})
    tasks = scheduler.build_tasks(view, plan)
    want = [((t.x, t.y), t.op[1]) for t in tasks if t.op[0] == "PLACE"]
    got = placement.animal_slots(view, plan, placement.available_animals(view))
    assert got == want


# --------------------------------------------------------- the tripwires
def test_volume_tripwires_cannot_move(cases):
    """one_time_units and plant_ops are PLANNER-owned. A placement lever
    that lowered plant count would be changing the allocation it does not
    own, so the count is asserted equal on every mid-season case."""
    from kagent import scheduler
    for flag in ("pl_wheat_adj", "pl_crop_cluster", "pl_water_first", "pl_pasture_ring"):
        for label, view, plan in cases:
            tasks = scheduler.build_tasks(view, plan)
            ctx = ctx_for(view, plan, tasks, {flag: True})
            out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
            n_before = sum(1 for t in tasks if t.op[0] == "PLANT")
            n_after = sum(1 for t in out if t.op[0] == "PLANT")
            assert n_before == n_after, f"{flag} changed plant count on {label}"
            assert sorted(t.op[1] for t in tasks if t.op[0] == "PLANT") == \
                sorted(t.op[1] for t in out if t.op[0] == "PLANT")


def test_tiles_stay_inside_the_original_pool(cases):
    from kagent import scheduler
    for flag in ("pl_wheat_adj", "pl_crop_cluster", "pl_water_first"):
        for label, view, plan in cases:
            tasks = scheduler.build_tasks(view, plan)
            ctx = ctx_for(view, plan, tasks, {flag: True})
            out = placement.replace_growth_tasks(view, plan, tasks, None, ctx)
            executor.check_invariants(view, tasks, out, strict=True)


def test_ledger_kill_switch_removes_the_flag_from_active_flags():
    from kagent import scheduler
    scheduler.reset_exec()
    executor.register(scheduler)
    ledger.LEDGER.reset_state()
    try:
        scheduler.set_exec(["pl_wheat_adj"])
        assert executor.active_flags()["pl_wheat_adj"] is True
        ledger.LEDGER.off.add("pl_wheat_adj")
        assert executor.active_flags()["pl_wheat_adj"] is False
    finally:
        ledger.LEDGER.reset_state()
        scheduler.reset_exec()


@pytest.mark.parametrize("op0,cls", [("PLANT", "PLANT"), ("BUILD_COOP", "BUILD"),
                                     ("BUILD_PASTURE", "BUILD"), ("PLACE", "PLACE"),
                                     ("WATER", None)])
def test_tile_class(op0, cls):
    assert placement.tile_class(op0) == cls


# ------------------------------------------------------------------ pl_net
def test_pl_net_prior_is_flat_and_therefore_inert(cases):
    """The pre-arm check the risk list demands, kept as a test so it is
    re-run if the checkpoint changes.

    The op head is supervised AT THE LABEL TILE ONLY, and its off-label
    cells turn out to echo the request planes -- which are input channels
    identical on every candidate tile. So the prior comes out flat and
    carries no ranking information. Measured: cv 0.018-0.019 over 25
    candidates. If this ever rises above 0.25, pl_net is worth an arm.
    """
    from kagent import scheduler
    placenet = pytest.importorskip("rlv3.placenet")
    pytest.importorskip("torch")
    import os
    if not os.path.exists(os.path.join(placenet.REPO, placenet.DEFAULT_PT)):
        pytest.skip("no execnet2_cond checkpoint")

    triples = []
    for _, view, plan in cases:
        triples.append((view, plan, scheduler.build_tasks(view, plan)))
    r = placenet.prior_dispersion(triples)
    if r["n_cases"] == 0:
        pytest.skip("no mid-season case had >= 2 candidate tiles")
    assert r["mean_cv"] < 0.25, (
        f"the pl_net prior is no longer flat ({r}); re-read placenet.py's "
        "verdict -- an arm may now be worth running")
