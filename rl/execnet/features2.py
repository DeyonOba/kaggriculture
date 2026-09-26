"""execnet v2 (Maia-style): the vocabulary is a DESTINATION over our own
board, not a per-step direction.

For each (step, unit) the label is where that unit performs its next tile op:
  0..99  the tile index (y * 10 + x)
  100    SHED   -- its next op is a shed op (PICKUP / DROP / PLACE at the shed)
  101    STAY   -- it acts on the tile it is standing on, or nothing happens
                  before the day ends (units respawn at the shed at hour 0,
                  so the day boundary ends the horizon)
A second head predicts the op CLASS at that destination.

The candidate set is the scheduler's own request map: the tiles our
`kagent.scheduler.build_tasks` has a pending task on, plus the shed, plus
stay. So the model answers "given these requests, which one does a top
player serve next" -- the ranker-over-our-tasks hybrid, not a free-form
policy. A recorded destination with no pending request cannot be expressed
and its row is dropped (the dataset reports that rate).

Planes (all 10x10, values in [0, 1]):
  0..23   the tile channels of rl/execnet/features.tile_channels
  24      the unit's position
  25..32  the request map, one plane per op class, 0 where no task:
          value = (1 + (P_FILLER - pri) / P_FILLER) / 2, so a survival task
          reads 1.0 and a filler task 0.5
Scalars: the same N_SCALAR vector as v1 (rl/execnet/features.unit_features).
"""
from __future__ import annotations

from kagent.constants import SHED_ACCESS
from kagent.scheduler import P_FILLER
from rl.execnet import features as F

BOARD = 10
N_TILES = BOARD * BOARD
DEST_SHED = N_TILES
DEST_STAY = N_TILES + 1
N_DEST = N_TILES + 2

REQ_CLASSES = ["HARVEST", "FEED", "CARE", "WATER", "PLANT", "FERTILIZE", "DIG", "BUILD"]
REQ_OF_OP = {"HARVEST": 0, "FEED": 1, "CARE": 2, "WATER": 3, "PLANT": 4, "FERTILIZE": 5,
             "DIG": 6, "BUILD_COOP": 7, "BUILD_PASTURE": 7, "PLACE": 7,
             "COLLECT_FERTILIZER": 0}
N_REQ = len(REQ_CLASSES)
N_PLANE = F.TILE_C + 1 + N_REQ          # 24 board + 1 unit + 8 request = 33
SHED_SET = set(SHED_ACCESS)
SHED_OPS = {"PICKUP", "DROP"}
TILE_OPS = {"WATER", "HARVEST", "PLANT", "FEED", "CARE", "FERTILIZE", "DIG",
            "BUILD_COOP", "BUILD_PASTURE", "PLACE", "COLLECT_FERTILIZER"}
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}


def tile_index(x, y):
    return y * BOARD + x


def board_planes(view):
    """-> list[F.TILE_C][N_TILES] floats, shared by every unit this step."""
    planes = [[0.0] * N_TILES for _ in range(F.TILE_C)]
    for x, y, t in view.me.iter_tiles():
        v = F.tile_channels(t, view.day)
        i = tile_index(x, y)
        for c, val in enumerate(v):
            if val:
                planes[c][i] = val
    return planes


def request_planes(tasks):
    """-> (list[N_REQ][N_TILES] floats, set of tiles with a request)."""
    planes = [[0.0] * N_TILES for _ in range(N_REQ)]
    tiles = set()
    for t in tasks:
        c = REQ_OF_OP.get(t.op[0])
        if c is None:
            continue
        i = tile_index(t.x, t.y)
        val = (1.0 + (P_FILLER - t.pri) / P_FILLER) / 2.0
        if val > planes[c][i]:
            planes[c][i] = val
        tiles.add((t.x, t.y))
    return planes, tiles


def dest_mask(view, ui, req_tiles):
    """-> list[N_DEST] bools. The unit's own tile is never a tile label (STAY
    encodes it) and SHED is masked out while the unit stands at the shed."""
    _, x, y = view.me.units[ui]
    m = [False] * N_DEST
    for (tx, ty) in req_tiles:
        if (tx, ty) != (x, y):
            m[tile_index(tx, ty)] = True
    m[DEST_SHED] = (x, y) not in SHED_SET
    m[DEST_STAY] = True
    return m


def unit_plane(view, ui):
    _, x, y = view.me.units[ui]
    p = [0.0] * N_TILES
    p[tile_index(x, y)] = 1.0
    return p


def op_class(op):
    """A recorded op -> request class index, or -1 when it is not a tile op."""
    if not op or not isinstance(op, (list, tuple)):
        return -1
    return REQ_OF_OP.get(op[0], -1)


def destinations(ops_by_step, positions, days, n_steps):
    """Destination label per step for ONE unit, from its recorded play.

    ops_by_step[k]  the unit's op at step k (list or None)
    positions[k]    (x, y) at step k       days[k] the day at step k
    -> (dest[k], op_class[k]) with op_class -1 for SHED / STAY / nothing.
    """
    dest = [DEST_STAY] * n_steps
    opc = [-1] * n_steps
    # scan backwards: the answer at k is the answer at k+1 unless the unit
    # acts at k, and it resets at a day boundary
    nxt = (DEST_STAY, -1, None)          # (dest, op class, tile) for step k+1
    for k in range(n_steps - 1, -1, -1):
        op = ops_by_step[k]
        name = op[0] if op and isinstance(op, (list, tuple)) else None
        if name in TILE_OPS:
            tile = positions[k]
            nxt = (tile_index(*tile), op_class(op), tile)
        elif name in SHED_OPS or (name == "PLACE" and positions[k] in SHED_SET):
            nxt = (DEST_SHED, -1, None)
        elif k + 1 < n_steps and days[k + 1] != days[k]:
            nxt = (DEST_STAY, -1, None)   # the day ends here: nothing is due
        d, c, tile = nxt
        if tile is not None and tile == positions[k]:
            dest[k], opc[k] = DEST_STAY, c
        else:
            dest[k], opc[k] = d, c
    return dest, opc


def scheduler_destination(view, ui, action):
    """Our scheduler's destination CLASS for one unit, from the action it
    emitted: STAY when it acts where it stands, SHED when it does a shed op,
    MOVE otherwise. (assign() does not expose the tile a moving unit is
    walking to, so the comparison is three-way.)"""
    if not action or not isinstance(action, (list, tuple)):
        return "STAY"
    name = action[0]
    if name in SHED_OPS:
        return "SHED"
    if name in TILE_OPS:
        _, x, y = view.me.units[ui]
        return "SHED" if (name == "PLACE" and (x, y) in SHED_SET) else "STAY"
    if name in MOVE_OPS:
        return "MOVE"
    return "STAY"
