"""Per-unit features, legality mask and label vocabulary shared by the
dataset builder (training) and the policy (play), so they cannot drift.

A sample is one (step, unit): the unit's 5x5 neighbourhood as tile channels
(TILE_C x 25, stored as uint8 0..255 = value x 255), a vector of scalars
(unit, pocket, shed, farm counts, clock), the legality mask over OPS from the
verified engine rules (kaggle_environments kaggriculture.py `_apply_unit_action`),
and the label = index of the op the recorded unit chose.
"""
from __future__ import annotations

from kagent.constants import ANIMALS, CROPS, SHED_ACCESS

OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "PLANT", "FEED",
       "CARE", "FERTILIZE", "DIG", "PICKUP", "DROP", "PLACE", "COLLECT_FERTILIZER",
       "BUILD_COOP", "BUILD_PASTURE"]
OP_INDEX = {o: i for i, o in enumerate(OPS)}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_L = ["GOOSE", "COW", "SHEEP"]
WINDOW = 5
HALF = WINDOW // 2
TILE_C = 24
N_TILE = TILE_C * WINDOW * WINDOW
N_SCALAR = 31
SHED_SET = set(SHED_ACCESS)


def tile_channels(t, day, off_board=False):
    """-> list of TILE_C floats in [0, 1] for one tile."""
    v = [0.0] * TILE_C
    if off_board:
        v[0] = 1.0
        v[1] = 1.0
        return v
    if t == "LOCKED":
        v[0] = 1.0
        return v
    if t is None:
        v[2] = 1.0
        return v
    if not isinstance(t, dict):
        v[2] = 1.0
        return v
    kind = t.get("kind")
    if kind == "WEED":
        v[3] = 1.0
        return v
    if kind == "PLANT":
        v[6] = 1.0
        crop = t.get("crop")
        if crop in CROPS_L:
            v[7 + CROPS_L.index(crop)] = 1.0
        v[12] = min(1.0, max(0.0, (day - t.get("planted_day", day)) / 12.0))
        v[13] = min(1.0, t.get("yield_units", 0) / 6.0)
        v[14] = 1.0 if t.get("watered_today") else 0.0
        v[15] = 1.0 if t.get("consecutive_unwatered", 0) >= 1 else 0.0
        return v
    if kind == "COOP":
        v[4] = 1.0
    elif kind == "PASTURE":
        v[5] = 1.0
    if t.get("animal"):
        v[16] = 1.0
        a = t.get("animal")
        if a in ANIMALS_L:
            v[17 + ANIMALS_L.index(a)] = 1.0
        v[20] = 1.0 if t.get("fed_today") else 0.0
        v[21] = 1.0 if t.get("cared_today") else 0.0
        v[22] = 1.0 if t.get("fertilizer_available") else 0.0
        v[23] = min(1.0, t.get("yield_units", 0) / 6.0)
    return v


def shed_dist(x, y):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in SHED_ACCESS)


def farm_counts(view):
    """Farm-level scalars shared by every unit this step (computed once)."""
    me = view.me
    per_crop = [0] * 5
    fed = unfed = bare = weeds = estruct = 0
    for x, y, t in me.iter_tiles():
        if t is None:
            bare += 1
        elif isinstance(t, dict):
            k = t.get("kind")
            if k == "PLANT":
                c = t.get("crop")
                if c in CROPS_L:
                    per_crop[CROPS_L.index(c)] += 1
            elif k == "WEED":
                weeds += 1
            elif k in ("COOP", "PASTURE"):
                if t.get("animal"):
                    if t.get("fed_today"):
                        fed += 1
                    else:
                        unfed += 1
                else:
                    estruct += 1
    shed_wheat = view.shed.get("WHEAT", 0)
    shed_fert = view.shed.get("FERTILIZER", 0)
    quads = [1.0 if q in me.unlocked_quadrants else 0.0 for q in ("NW", "NE", "SW", "SE")]
    return {
        "per_crop": [min(1.0, n / 25.0) for n in per_crop],
        "unfed": min(1.0, unfed / 16.0), "fed": min(1.0, fed / 16.0),
        "bare": min(1.0, bare / 25.0), "weeds": min(1.0, weeds / 10.0),
        "estruct": min(1.0, estruct / 8.0),
        "shed_wheat": min(1.0, shed_wheat / 50.0), "shed_fert": min(1.0, shed_fert / 30.0),
        "shed_total": min(1.0, view.shed_total / 100.0),
        "seeds": min(1.0, sum(view.seeds.values()) / 25.0),
        "money": min(1.0, max(0.0, me.money) / 200_000.0),
        "n_units": min(1.0, len(me.units) / 13.0),
        "quads": quads,
    }


def unit_features(view, ui, counts=None):
    """-> (tiles: list[N_TILE] floats, scalars: list[N_SCALAR] floats) for unit ui."""
    counts = counts or farm_counts(view)
    me = view.me
    idx, x, y = me.units[ui]
    inv = view.unit_inv(idx)
    tiles = []
    for dy in range(-HALF, HALF + 1):
        for dx in range(-HALF, HALF + 1):
            tx, ty = x + dx, y + dy
            if 0 <= tx < me.board_size and 0 <= ty < me.board_size:
                tiles.extend(tile_channels(me.tile(tx, ty), view.day))
            else:
                tiles.extend(tile_channels(None, view.day, off_board=True))
    produce = sum(n for k, n in inv.items() if k not in ("WHEAT", "FERTILIZER") and k not in ANIMALS)
    carrying_animal = 1.0 if any(inv.get(a, 0) for a in ANIMALS) else 0.0
    scalars = [
        view.hour / 24.0, view.day / 30.0, x / 9.0, y / 9.0, min(1.0, shed_dist(x, y) / 10.0),
        1.0 if (x, y) in SHED_SET else 0.0, 1.0 if idx == 0 else 0.0, counts["n_units"],
        min(1.0, inv.get("WHEAT", 0) / 10.0), min(1.0, inv.get("FERTILIZER", 0) / 10.0),
        min(1.0, produce / 20.0), carrying_animal, counts["seeds"],
        counts["shed_wheat"], counts["shed_fert"], counts["shed_total"], counts["money"],
    ] + counts["per_crop"] + [counts["unfed"], counts["fed"], counts["bare"], counts["weeds"],
                              counts["estruct"]] + counts["quads"]
    assert len(scalars) == N_SCALAR, len(scalars)
    return tiles, scalars


def legal_mask(view, ui):
    """-> list[len(OPS)] of bools: which ops the engine would not discard for
    unit ui (verified rules: shed ops resolve before the LOCKED guard; tile
    ops need an owned tile; PLANT needs a seed; FEED needs wheat in the
    pocket; HARVEST needs yield and, for plants, maturity)."""
    me = view.me
    idx, x, y = me.units[ui]
    inv = view.unit_inv(idx)
    t = me.tile(x, y)
    m = [False] * len(OPS)
    m[OP_INDEX["NORTH"]] = y > 0
    m[OP_INDEX["SOUTH"]] = y < me.board_size - 1
    m[OP_INDEX["EAST"]] = x < me.board_size - 1
    m[OP_INDEX["WEST"]] = x > 0
    m[OP_INDEX["PASS"]] = True
    shed_adj = (x, y) in SHED_SET
    has_inv = sum(inv.values()) > 0
    m[OP_INDEX["PICKUP"]] = shed_adj and view.shed_total > 0
    m[OP_INDEX["DROP"]] = shed_adj and has_inv
    carrying = [a for a in ANIMALS if inv.get(a, 0) > 0]
    if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and not t.get("animal"):
        m[OP_INDEX["PLACE"]] = any(ANIMALS[a]["structure"] == t.get("kind") for a in carrying)
    m[OP_INDEX["PLACE"]] = m[OP_INDEX["PLACE"]] or (shed_adj and has_inv)
    if t == "LOCKED":
        return m
    if t is None:
        m[OP_INDEX["PLANT"]] = any(n > 0 for n in view.seeds.values())
        m[OP_INDEX["BUILD_COOP"]] = True
        m[OP_INDEX["BUILD_PASTURE"]] = True
        return m
    if not isinstance(t, dict):
        return m
    kind = t.get("kind")
    if kind == "PLANT":
        crop = CROPS.get(t.get("crop"))
        m[OP_INDEX["WATER"]] = not t.get("watered_today")
        age = view.day - t.get("planted_day", view.day)
        m[OP_INDEX["HARVEST"]] = (t.get("yield_units", 0) > 0 and crop is not None
                                  and age >= crop["first_yield_day"])
        m[OP_INDEX["FERTILIZE"]] = inv.get("FERTILIZER", 0) > 0
        m[OP_INDEX["DIG"]] = True
    elif kind == "WEED":
        m[OP_INDEX["DIG"]] = True
    elif kind in ("COOP", "PASTURE"):
        if t.get("animal"):
            m[OP_INDEX["FEED"]] = (not t.get("fed_today")) and inv.get("WHEAT", 0) > 0
            m[OP_INDEX["CARE"]] = not t.get("cared_today")
            m[OP_INDEX["HARVEST"]] = t.get("yield_units", 0) > 0
            m[OP_INDEX["COLLECT_FERTILIZER"]] = bool(t.get("fertilizer_available"))
        else:
            m[OP_INDEX["DIG"]] = True
    return m


def label_of(op):
    """Recorded unit action -> class index, or -1 when not in the vocabulary."""
    if not op or not isinstance(op, (list, tuple)):
        return OP_INDEX["PASS"]
    return OP_INDEX.get(op[0], -1)


def unit_ops(action, n_units):
    """A recorded farm action {'farmer': op, 'hands': [op...]} -> ops per unit."""
    if not isinstance(action, dict):
        return [None] * n_units
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    ops = ops[:n_units] + [None] * max(0, n_units - len(ops))
    return ops
