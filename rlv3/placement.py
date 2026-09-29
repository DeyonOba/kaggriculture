"""WHICH TILE a growth task lands on. Never which unit, never a move.

This module rewrites the `(x, y)` of the growth tasks `kagent.scheduler.
build_tasks` already produced. It never adds a task, never removes one, and
never changes an op or a crop: the planner owns the allocation (how many
tiles of each crop), and the rewrite is a permutation of TILES among the
slots the planner asked for. That is the whole reason it is safe to bolt on
to a shipped agent -- the day's volume, the crop mix and the plant quota
come out identical, so the one-time-units and plant-ops tripwires cannot
move down because of anything in here.

THE ONE PLACEMENT GAP THE DATA SUPPORTS is wheat adjacency:
docs/results/top3_tile_metrics.txt has wheat tiles beside an animal at 0.09
for us against 0.46 / 0.36 / 0.42 for M&M&P&Q / Boey / DSM. That is the
target of `pl_wheat_adj`.

THE ONE THE DATA REFUTES is pasture distance. The same file has animal shed
distance 2.45 ours against 3.53 / 2.07 / 2.10 -- leader MEAN 2.57, with two
of three leaders sitting CLOSER to the shed than we do. "Move the animals
away from the shed" is a one-leader artifact, and a rule of exactly that
shape has already been built and lost $11.3k. `pl_pasture_ring` is here to
reproduce that loss on purpose, as proof the seam can move placement at
all; it is permanently off and is the only lever allowed to widen its tile
pool (see TILE POOLS below).

TILE POOLS. Invariant (b) of the seam says a growth task may only land on a
tile the ORIGINAL build_tasks offered "for that op class". Two readings are
possible and they differ, so this module fixes one and makes the other
testable:

  STRICT  -- PLANT may only take a tile the original offered a PLANT on,
             BUILD only a BUILD tile, PLACE only a PLACE tile. Every lever
             that ships is strict, and `rlv3.executor` asserts it.
  WIDE    -- PLANT and BUILD_* share one pool, because the original draws
             both from the single `empties` list and splits it at an
             arbitrary cursor; a bare tile accepts either op. PLACE is
             never widened: it is only legal on a matching empty structure.

Only `pl_pasture_ring` sets WIDE, and only because a ring that cannot leave
the four centre-most bare tiles is not the rule that lost the $11.3k and so
would not reproduce it.
"""

from __future__ import annotations

from kagent.constants import ANIMALS, SHED_ACCESS
from kagent.scheduler import (
    P_FILLER,
    PLANT_LAST_HOUR,
    Task,
    _dist,
)

GROWTH_OPS = frozenset(("PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE"))
BUILD_OPS = frozenset(("BUILD_COOP", "BUILD_PASTURE"))
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def tile_class(op0):
    """The tile POOL an op draws from. See TILE POOLS in the docstring."""
    if op0 == "PLANT":
        return "PLANT"
    if op0 in BUILD_OPS:
        return "BUILD"
    if op0 == "PLACE":
        return "PLACE"
    return None


# --------------------------------------------------------------- splitting
def split_growth(tasks):
    """-> (growth, rest), both preserving the original relative order."""
    growth, rest = [], []
    for t in tasks:
        (growth if t.op[0] in GROWTH_OPS else rest).append(t)
    return growth, rest


def crop_slots(growth):
    """The crop multiset the planner chose, in the original slot order."""
    return [t.op[1] for t in growth if t.op[0] == "PLANT"]


def candidate_tiles(growth, op):
    """The tiles the original offered for `op`'s class, in original order."""
    cls = tile_class(op)
    out = []
    for t in growth:
        if tile_class(t.op[0]) == cls and (t.x, t.y) not in out:
            out.append((t.x, t.y))
    return out


# ------------------------------------------------------------- adjacencies
def adjacency_sets(view, tasks=None):
    """Live adjacency structure of the farm, read fresh every turn.

    near_structure   bare-tile neighbours of a COOP or PASTURE
    near_water_work  bare-tile neighbours of a tile with a pending WATER
    near_pending     bare-tile neighbours of any pending non-filler task
    """
    near_structure, near_water, near_pending = set(), set(), set()
    for sx, sy, t in view.me.iter_tiles():
        if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
            for dx, dy in ORTHO:
                near_structure.add((sx + dx, sy + dy))
    for t in tasks or ():
        if t.pri >= P_FILLER:
            continue
        for dx, dy in ORTHO:
            near_pending.add((t.x + dx, t.y + dy))
        if t.op[0] == "WATER":
            for dx, dy in ORTHO:
                near_water.add((t.x + dx, t.y + dy))
    return {"near_structure": near_structure,
            "near_water_work": near_water,
            "near_pending": near_pending}


def _kin(view):
    """crop -> [(x, y)] of every standing plant of that crop."""
    out = {}
    for x, y, t in view.me.plants():
        out.setdefault(t.get("crop"), []).append((x, y))
    return out


def _unfed_animals(view):
    return sum(1 for _, _, t in view.me.animals() if not t.get("fed_today"))


def make_ctx(view, plan, tasks, flags, params, regime="even", stats=None):
    """Everything score_tile reads, computed once per turn."""
    return {
        "flags": flags,
        "params": params,
        "adj": adjacency_sets(view, tasks),
        "kin": _kin(view),
        "unfed": _unfed_animals(view),
        "regime": regime,
        "stats": stats,
        "wide_pool": bool(flags.get("pl_pasture_ring")),
    }


# ------------------------------------------------------------------ scoring
def score_tile(view, plan, crop, xy, ctx, prior=None):
    """Score of seating `crop` on `xy`. Zero weights -> zero, always.

    Every term is a live read: structures from `iter_tiles`, kin from
    `plants()`, pending work from this turn's own task list. Nothing here is
    a schedule and nothing is cached across a turn.
    """
    flags, p = ctx["flags"], ctx["params"]
    s = 0.0

    if flags.get("pl_wheat_adj") and ctx["unfed"] > 0:
        # Claim the bare tiles beside a structure for the wheat quota, and
        # only while there are unfed animals for that wheat to serve. The
        # ledger switches this off after two days below handfeed_target.
        #
        # The REPULSION is not decoration. The greedy runs in slot order, so
        # a strawberry slot whose own tile a wheat slot just took falls back
        # to the lowest-index free tile -- which is a structure neighbour,
        # and the reservation leaks. This is the same two-sided rule the
        # shipped `layout` flag implements ("bare tiles adjacent to a
        # pasture or coop take the wheat quota first, the rest of the quota
        # goes to the other bare tiles").
        w = float(p.get("w_wheat_adj", 3.0))
        if xy in ctx["adj"]["near_structure"]:
            s += w if crop == "WHEAT" else -0.5 * w

    if flags.get("pl_crop_cluster"):
        radius = float(p.get("cluster_radius", 2))
        if radius > 0:
            kin = ctx["kin"].get(crop) or ()
            if kin:
                d = min(_dist(xy[0], xy[1], kx, ky) for kx, ky in kin)
                if d <= radius:
                    s += float(p.get("w_cluster", 2.0)) * (radius - d + 1.0) / radius

    if flags.get("pl_water_first") and view.hour <= PLANT_LAST_HOUR:
        # A fresh plant needs a second WATER the same day. Seating it beside
        # work a unit is already walking to makes that water one step.
        w = float(p.get("w_water_first", 1.0))
        if p.get("water_tight"):
            # the ledger saw yesterday's plants go dry: insist on the water
            # cluster itself rather than any pending work
            if xy in ctx["adj"]["near_water_work"]:
                s += w * 2.0
        elif xy in ctx["adj"]["near_pending"]:
            s += w

    if prior:
        s += float(p.get("w_net", 1.0)) * float(prior.get(xy, 0.0))
    return s


# ------------------------------------------------------------- assignments
def assign_crops(view, plan, tiles, crops, ctx, prior=None):
    """Deterministic greedy in SLOT order -> [((x, y), crop)].

    Slot j keeps its crop and chooses a tile. Ties are broken by the slot's
    OWN original tile first, then by the original scheduler index, so a
    configuration with every weight at zero reproduces build_tasks exactly.
    """
    n = min(len(tiles), len(crops))
    if n == 0:
        return []
    index_of = {xy: i for i, xy in enumerate(tiles)}
    table = {}
    for crop in set(crops[:n]):
        for xy in tiles:
            table[(crop, xy)] = score_tile(view, plan, crop, xy, ctx, prior)
    free = list(tiles)
    out = []
    for j in range(n):
        crop = crops[j]
        own = tiles[j] if j < len(tiles) else None
        best, best_key = None, None
        for xy in free:
            key = (table[(crop, xy)], 1 if xy == own else 0, -index_of[xy])
            if best_key is None or key > best_key:
                best, best_key = xy, key
        free.remove(best)
        out.append((best, crop))
    return out


def structure_tiles(view, plan, n_coop, n_pasture, tiles):
    """-> [((x, y), "COOP" | "PASTURE")] over the BUILD candidate tiles.

    Identity unless `pl_pasture_ring` is on: it is a DIAGNOSTIC NEGATIVE
    that pushes pastures AWAY from the shed, reproducing the known −$11.3k
    so the seam is proved able to move placement before any positive claim
    is made. It never ships.
    """
    kinds = ["COOP"] * n_coop + ["PASTURE"] * n_pasture
    n = min(len(kinds), len(tiles))
    return [(tiles[i], kinds[i]) for i in range(n)]


def pasture_ring_tiles(view, kinds, tiles):
    """`pl_pasture_ring`: the pasture slots take the bare tiles FARTHEST
    from the shed, the coops keep the rest, nearest first."""
    order = sorted(tiles, key=lambda xy: (-_shed_dist(xy), xy[1], xy[0]))
    near = sorted(tiles, key=lambda xy: (_shed_dist(xy), xy[1], xy[0]))
    out, used = [], set()
    for kind in kinds:
        pool = order if kind == "PASTURE" else near
        pick = next((xy for xy in pool if xy not in used), None)
        if pick is None:
            break
        used.add(pick)
        out.append((pick, kind))
    return out


def _shed_dist(xy):
    return min(_dist(xy[0], xy[1], sx, sy) for sx, sy in SHED_ACCESS)


def _move_key(crop, flags):
    """Which counter a moved plant belongs to.

    A counter must never name a flag that is OFF. The first version of this
    fell through to `pl_water_first.moved` for any non-wheat move and a
    traced game reported 143 of them on an arm where `pl_water_first` was
    off -- the knock-on moves of the wheat REPULSION, filed under an
    unrelated lever. A miscounted lever is how an arm becomes unreadable,
    so displaced plants get their own bucket under the flag that displaced
    them.
    """
    if flags.get("pl_wheat_adj"):
        return "pl_wheat_adj.moved" if crop == "WHEAT" else "pl_wheat_adj.displaced"
    if flags.get("pl_crop_cluster"):
        return "pl_crop_cluster.moved"
    if flags.get("pl_water_first"):
        return "pl_water_first.moved"
    if flags.get("pl_net"):
        return "pl_net.moved"
    return "placement.moved"


def animal_slots(view, plan, available):
    """The PLACE pairing build_tasks itself derives, re-derived here.

    A COOP takes a GOOSE and a PASTURE a COW or a SHEEP, so the only freedom
    is which PASTURE gets which of the two -- no measured gap asks for that,
    and PLACE tiles are never repooled. `replace_growth_tasks` therefore
    passes PLACE tasks through untouched; this function exists so the tests
    can prove that the pass-through IS the scheduler's own pairing.
    """
    avail = dict(available)
    out = []
    for x, y, t in view.me.empty_structures():
        kinds = ("GOOSE",) if t["kind"] == "COOP" else ("COW", "SHEEP")
        for kind in kinds:
            if avail.get(kind, 0) > 0:
                avail[kind] -= 1
                out.append(((x, y), kind))
                break
    return out


def available_animals(view):
    out = {}
    for kind in ANIMALS:
        out[kind] = view.shed.get(kind, 0) + sum(
            view.unit_inv(i).get(kind, 0) for i, _, _ in view.me.units
        )
    return out


# ---------------------------------------------------------------- rewriting
def replace_growth_tasks(view, plan, tasks, prior=None, ctx=None):
    """-> a NEW task list with the growth tiles re-seated.

    Positions in the list are preserved exactly: the task at growth slot j
    is replaced in place, so the op multiset, the crop multiset and the
    ordering build_tasks produced all survive untouched.
    """
    if ctx is None:
        return list(tasks)
    flags = ctx["flags"]
    stats = ctx.get("stats")
    growth, _ = split_growth(tasks)
    if not growth:
        return list(tasks)

    plant_idx = [i for i, t in enumerate(tasks) if t.op[0] == "PLANT"]
    build_idx = [i for i, t in enumerate(tasks) if t.op[0] in BUILD_OPS]
    plant_tiles = [(tasks[i].x, tasks[i].y) for i in plant_idx]
    build_tiles = [(tasks[i].x, tasks[i].y) for i in build_idx]

    out = list(tasks)

    # --- BUILD: identity unless the diagnostic negative is on -------------
    if flags.get("pl_pasture_ring") and build_idx:
        kinds = ["COOP" if tasks[i].op[0] == "BUILD_COOP" else "PASTURE" for i in build_idx]
        pool = build_tiles + ([t for t in plant_tiles if t not in build_tiles]
                              if ctx.get("wide_pool") else [])
        pairs = pasture_ring_tiles(view, kinds, pool)
        taken = set()
        for j, i in enumerate(build_idx):
            if j >= len(pairs):
                break
            (nx, ny), _kind = pairs[j]
            taken.add((nx, ny))
            if (nx, ny) != (tasks[i].x, tasks[i].y) and stats is not None:
                stats["pl_pasture_ring.moved"] += 1
            out[i] = Task(tasks[i].pri, nx, ny, list(tasks[i].op), tasks[i].need)
        # a plant slot whose tile a pasture just took swaps onto the tile the
        # pasture vacated, so the tile multiset over all growth ops is fixed
        vacated = [t for t in build_tiles if t not in taken]
        vi = 0
        for k, i in enumerate(plant_idx):
            xy = (tasks[i].x, tasks[i].y)
            if xy in taken and vi < len(vacated):
                nx, ny = vacated[vi]
                vi += 1
                plant_tiles[k] = (nx, ny)
                out[i] = Task(tasks[i].pri, nx, ny, list(tasks[i].op), tasks[i].need)

    # --- PLANT: permute the tiles among the planner's crop slots ----------
    if plant_idx and any(flags.get(f) for f in
                         ("pl_wheat_adj", "pl_crop_cluster", "pl_water_first", "pl_net")):
        crops = [out[i].op[1] for i in plant_idx]
        pairs = assign_crops(view, plan, plant_tiles, crops, ctx, prior)
        for j, i in enumerate(plant_idx):
            if j >= len(pairs):
                break
            (nx, ny), crop = pairs[j]
            base = out[i]
            if (nx, ny) != (base.x, base.y):
                if stats is not None:
                    stats[_move_key(crop, flags)] += 1
                out[i] = Task(base.pri, nx, ny, list(base.op), base.need)
    return out
