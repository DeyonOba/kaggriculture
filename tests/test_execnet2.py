"""execnet v2 (destination vocabulary): label derivation over a unit's future
actions, the request-map planes built from the scheduler's task list, and the
legality mask over destinations.

    PYTHONPATH=. uv run pytest tests/test_execnet2.py -q
"""
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import planner, scheduler  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402
from rl.execnet import features2 as F2  # noqa: E402


def _obs(day=12, hour=6):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    obs["farms"][0]["money"] = 5_000
    return obs


def _view_tasks(obs):
    view = View(obs)
    plan = planner.make_champion_plan(view, Memory())
    plan.liquidation = False
    return view, scheduler.build_tasks(view, plan)


def test_destination_labels_walk_back_to_the_next_tile_op():
    # the unit walks from (1,1) to (1,0), waters there, then goes to the shed
    ops = [None, ["NORTH"], ["WATER"], ["SOUTH"], ["PICKUP", "WHEAT", 5]]
    pos = [(1, 1), (1, 1), (1, 0), (4, 4), (4, 4)]
    days = [3] * 5
    dest, opc = F2.destinations(ops, pos, days, 5)
    assert dest[0] == dest[1] == F2.tile_index(1, 0)      # walking to the water
    assert dest[2] == F2.DEST_STAY                        # acting where it stands
    assert dest[3] == dest[4] == F2.DEST_SHED
    assert opc[0] == opc[1] == opc[2] == F2.REQ_CLASSES.index("WATER")
    assert opc[3] == -1


def test_day_boundary_ends_the_horizon():
    # nothing happens on day 3; the op at step 2 belongs to day 4
    ops = [None, None, ["HARVEST"]]
    pos = [(2, 2), (2, 2), (7, 7)]
    dest, _ = F2.destinations(ops, pos, [3, 3, 4], 3)
    assert dest[0] == dest[1] == F2.DEST_STAY
    assert dest[2] == F2.DEST_STAY                        # it acts where it stands


def test_request_planes_carry_priority_and_class():
    obs = _obs()
    view, tasks = _view_tasks(obs)
    planes, tiles = F2.request_planes(tasks)
    assert len(planes) == F2.N_REQ and all(len(p) == F2.N_TILES for p in planes)
    assert tiles, "the champion plan should request something on day 12"
    for t in tasks:
        c = F2.REQ_OF_OP.get(t.op[0])
        if c is None:
            continue
        v = planes[c][F2.tile_index(t.x, t.y)]
        assert 0.5 <= v <= 1.0
        if t.pri == scheduler.P_SURVIVAL:
            assert v == 1.0                                # survival reads 1.0
    # every requested tile is non-zero in at least one class
    for (x, y) in tiles:
        assert any(p[F2.tile_index(x, y)] > 0 for p in planes)


def test_dest_mask_is_the_request_set_plus_shed_and_stay():
    obs = _obs()
    obs["farms"][0]["farmer"] = [1, 1]
    view, tasks = _view_tasks(obs)
    _, tiles = F2.request_planes(tasks)
    m = F2.dest_mask(view, 0, tiles)
    assert len(m) == F2.N_DEST
    assert m[F2.DEST_STAY] and m[F2.DEST_SHED]
    assert not m[F2.tile_index(1, 1)]                      # own tile -> STAY encodes it
    for i in range(F2.N_TILES):
        x, y = i % F2.BOARD, i // F2.BOARD
        assert m[i] == ((x, y) in tiles and (x, y) != (1, 1))
    # standing at the shed masks SHED out
    obs["farms"][0]["farmer"] = [4, 4]
    view, tasks = _view_tasks(obs)
    _, tiles = F2.request_planes(tasks)
    assert not F2.dest_mask(view, 0, tiles)[F2.DEST_SHED]


def test_board_planes_shape_and_scheduler_destination_classes():
    obs = _obs()
    view, _ = _view_tasks(obs)
    planes = F2.board_planes(view)
    assert len(planes) == 24 and all(len(p) == F2.N_TILES for p in planes)
    assert F2.scheduler_destination(view, 0, ["WATER"]) == "STAY"
    assert F2.scheduler_destination(view, 0, ["PICKUP", "WHEAT", 5]) == "SHED"
    assert F2.scheduler_destination(view, 0, ["NORTH"]) == "MOVE"
    assert F2.scheduler_destination(view, 0, ["PASS"]) == "STAY"
