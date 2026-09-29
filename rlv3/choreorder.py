"""PRIORITY ONLY, and only BETWEEN TASKS ON THE SAME TILE.

WHY THE SEAM IS THIS NARROW. The per-quadrant priority queue
(`kagent.scheduler.EXEC["quadq"]`) lost 11.6 points because it scored tasks
priority-before-distance and walked hands across the farm. `scheduler.pick`
scores a task `(urgent, d, pri)`: distance first, priority only as a
tie-break. Two tasks on the SAME tile have the same `d` from every unit, so
rewriting `pri` between them changes which chore a unit takes while it is
standing there, and changes nothing about where anyone walks. That is the
structural defence, and `rlv3.executor`'s invariant check enforces it.

WHAT IT BUYS. `chores_per_visit` is 1.453 for us against 2.076 / 1.898 /
1.913 for the three leaders, and `visits_ge2_share` 0.32 against
0.60 / 0.52 / 0.54. A unit that is already on an animal tile with a FEED, a
COLLECT_FERTILIZER and a HARVEST pending should take all three before it
leaves. Which it takes FIRST is decided by `pri`, and on a tile where two
pending ops share a priority the shipped order is whatever `build_tasks`
happened to append.

WHAT IT MUST NOT DO, enforced here and asserted by the seam:
  * no task may become, or stop being, P_SURVIVAL -- the survival band is
    left completely untouched, never permuted and never epsilon-shifted,
    because `assign`'s survival override and `pick`'s `urgent` flag both
    test `pri == P_SURVIVAL` exactly;
  * no task may cross the P_FILLER band edge;
  * the multiset of priorities on a tile is PERMUTED, not replaced, so the
    global priority histogram is unchanged up to the epsilon tie-break.
"""

from __future__ import annotations

from kagent import econ
from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS
from kagent.scheduler import P_FILLER, P_SURVIVAL, Task

CHAIN_ANIMAL = ("FEED", "CARE", "COLLECT_FERTILIZER", "HARVEST")
CHAIN_CROP = ("WATER", "HARVEST", "FERTILIZE", "PLANT", "DIG")
CO_FLAGS = ("co_chain_tile", "co_animal_first", "co_value")

# Small enough that it only decides EXACT ties -- two tasks at equal
# distance and equal shipped priority, which `build_tasks` resolved by
# append order and `pick` by first-encountered.
EPS = 1e-6
MAX_RANK = 64


def tile_groups(tasks):
    """-> {(x, y): [Task]} preserving the original order inside each tile."""
    out = {}
    for t in tasks:
        out.setdefault((t.x, t.y), []).append(t)
    return out


def tile_group_indices(tasks):
    """-> {(x, y): [index into tasks]}. Index-keyed rather than id-keyed so a
    list that happens to hold the same Task object twice cannot alias."""
    out = {}
    for i, t in enumerate(tasks):
        out.setdefault((t.x, t.y), []).append(i)
    return out


def _tile_kind(view, xy):
    t = view.me.tile(*xy) if 0 <= xy[0] < view.me.board_size and 0 <= xy[1] < view.me.board_size else None
    if isinstance(t, dict):
        kind = t.get("kind")
        if kind in ("COOP", "PASTURE"):
            return "animal", t
        if kind == "PLANT":
            return "crop", t
    return "other", t if isinstance(t, dict) else None


def chain_order(view, plan, xy, group):
    """The group in the ENGINE's dependency order.

    Animal: FEED -> CARE -> COLLECT_FERTILIZER -> HARVEST (CARE only banks
    the 3x bonus on a fed day). Crop: WATER -> HARVEST -> FERTILIZE ->
    PLANT -> DIG (an in-window WATER adds a yield unit to the crop you are
    about to take -- the same fact that made `lastwater` worth $2.1k).
    """
    kind, _ = _tile_kind(view, xy)
    chain = CHAIN_ANIMAL if kind == "animal" else CHAIN_CROP
    pos = {op: i for i, op in enumerate(chain)}
    return sorted(range(len(group)),
                  key=lambda i: (pos.get(group[i].op[0], len(chain)), i))


def _product_of(view, task):
    """The market product a task's output lands in, or None."""
    op = task.op[0]
    if op == "COLLECT_FERTILIZER":
        return "FERTILIZER"
    kind, t = _tile_kind(view, (task.x, task.y))
    if kind == "animal" and t is not None:
        return ANIMALS.get(t.get("animal"), {}).get("product")
    if kind == "crop" and t is not None:
        return t.get("crop")
    if op == "PLANT" and len(task.op) > 1:
        return task.op[1]
    return None


def _units_now(view, task):
    """Units this op puts in a pocket THIS turn (0 when it is not a take)."""
    if task.op[0] != "HARVEST":
        return 0
    _, t = _tile_kind(view, (task.x, task.y))
    return int((t or {}).get("yield_units", 0) or 0)


def _opp_standing(view):
    """The opponent's standing supply per product: the board read that makes
    this the one place opponent state enters the executor."""
    out = {}
    for _, _, t in view.opp.plants():
        crop = t.get("crop")
        if crop in CROPS:
            out[crop] = out.get(crop, 0) + 1
    for _, _, t in view.opp.animals():
        p = ANIMALS.get(t.get("animal"), {}).get("product")
        if p:
            out[p] = out.get(p, 0) + 1
    return out


def value_weight(view, plan, task, lead, regime="even", opp=None):
    """Higher = take this chore first. Recomputed every turn from today's
    inventory and their CURRENT standing counts; nothing is cached."""
    product = _product_of(view, task)
    if not product or product not in MARKET_PARAMS:
        return 0.0
    inv = view.market_inv.get(product, MARKET_PARAMS[product]["I0"])
    p = float(econ.price(product, inv))
    base = float(MARKET_PARAMS[product]["base"]) or 1.0
    rel = p / base
    theirs = (opp if opp is not None else _opp_standing(view)).get(product, 0)

    if regime == "salvage":
        # Objective 3: minimise the loss margin. Only realisable cash per
        # turn counts -- a WATER that pays on day 31 is worth nothing.
        return rel * _units_now(view, task)
    if regime == "contest":
        # Objective 1, behind and late: the market is ONE shared inventory.
        # Reach the products they hold standing supply in BEFORE they do --
        # strawberry, milk and wool are 97% of the shared-inventory cost.
        # NOTE the deliberate sign flip against the `even` branch below.
        return rel * (1.0 + 0.1 * theirs)
    if regime == "ahead":
        # Objective 2: protect the lead. Chain order alone; no value tilt,
        # so a unit finishes the tile underfoot and takes no new work.
        return 0.0
    # Even or early: a product they are about to flood is tended later.
    return rel * (1.0 / (1.0 + 0.1 * theirs))


def _chains_here(view, xy, flags):
    """Does a chain apply to this tile, given the flags?

    `co_chain_tile` chains every tile. `co_animal_first` chains ANIMAL tiles
    only, and exists as its own flag because the animal half of the gap is
    much the larger one: chores per visit on animal tiles is 1.617 for us
    against 2.773 / 1.824 / 1.978, where the crop half is 1.267 against
    1.764 / 1.949 / 1.846. Gating the two separately is what tells us which
    half carries an effect instead of bundling them into one verdict.
    """
    if flags.get("co_chain_tile"):
        return True
    if flags.get("co_animal_first"):
        return _tile_kind(view, xy)[0] == "animal"
    return False


def _rank(view, plan, xy, group, ctx):
    """-> desired order as a list of indices into `group`."""
    flags = ctx["flags"]
    n = len(group)
    chained = _chains_here(view, xy, flags)
    chain = chain_order(view, plan, xy, group) if chained else list(range(n))
    chain_pos = {i: k for k, i in enumerate(chain)}
    if not flags.get("co_value"):
        return chain
    lead = ctx.get("lead", 0.0)
    regime = ctx.get("regime", "even")
    opp = ctx.get("opp_standing")
    val = {i: value_weight(view, plan, group[i], lead, regime, opp) for i in range(n)}
    if chained:
        # dependency is a hard constraint, value only breaks its ties
        return sorted(range(n), key=lambda i: (chain_pos[i], -val[i], i))
    return sorted(range(n), key=lambda i: (-val[i], i))


def reprioritise(view, plan, tasks, ctx=None):
    """-> a NEW list; only `pri` differs, and only between same-tile tasks."""
    if ctx is None or not any(ctx["flags"].get(f) for f in CO_FLAGS):
        return list(tasks)
    stats = ctx.get("stats")
    out = list(tasks)
    for xy, idxs in tile_group_indices(tasks).items():
        if len(idxs) < 2:
            continue
        group = [tasks[i] for i in idxs]
        # the survival band is untouchable: `assign` and `pick` both test
        # `pri == P_SURVIVAL` exactly
        work = [i for i in range(len(group)) if P_SURVIVAL < group[i].pri < P_FILLER]
        fill = [i for i in range(len(group)) if group[i].pri >= P_FILLER]
        changed = False
        for band in (work, fill):
            if len(band) < 2:
                continue
            inband = set(band)
            order = [i for i in _rank(view, plan, xy, group, ctx) if i in inband]
            # Compare against the order the band will actually EXECUTE in,
            # not its position in the task list. `assign._underfoot` takes
            # the lowest `pri` and `pick` scores `(urgent, d, pri)`, so on
            # one tile execution runs in ASCENDING `pri`; `build_tasks`
            # appends an animal tile FEED, COLLECT_FERTILIZER, HARVEST at
            # pri 1.5 / 2.0 / 1.0, i.e. already in chain order but executed
            # HARVEST first. Comparing against the append order declared
            # those tiles unchanged and made the lever inert on exactly the
            # tiles whose execution order is wrong.
            execo = sorted(band, key=lambda i: (group[i].pri, i))
            if order == execo:
                # no lever asked for a different order here, so the band is
                # left EXACTLY as shipped -- the epsilon never appears on a
                # tile nothing reordered. This is what keeps `co_animal_first`
                # off crop tiles and keeps the side-effect surface equal to
                # the set of tiles a lever actually touched.
                continue
            values = sorted(group[i].pri for i in band)
            for k, i in enumerate(order):
                new = values[k] + EPS * min(k, MAX_RANK)
                if new != group[i].pri:
                    changed = True
                t = group[i]
                out[idxs[i]] = Task(new, t.x, t.y, list(t.op), t.need)
        if changed and stats is not None:
            f = ctx["flags"]
            key = ("co_chain_tile.reordered" if f.get("co_chain_tile")
                   else "co_animal_first.reordered" if _chains_here(view, xy, f)
                   else "co_value.reordered")
            stats[key] += 1
    return out
