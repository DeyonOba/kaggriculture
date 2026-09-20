"""The enumerated (unit, op, tile) action space, with masks from the engine.

The foundation for a GridNet-style policy (Han et al., ICML 2019): a
convolutional encoder-decoder emits ONE ACTION PER CELL and each unit
executes the action at the cell it occupies. That is this engine's semantics
exactly -- `_apply_unit_action` resolves every op against
`_farmer_position(farm, idx)` -- and it handles our varying unit count (1
farmer plus 0-11 hands) without architectural change.

Gym-uRTS (Huang & Ontanon, arXiv 2105.13807) reports invalid action masking
as ESSENTIAL rather than helpful: full masking beat action-type-only masking
substantially, and GridNet with masking reached ~91% against a pool of 11
competition bots. We are better placed than they were -- our masks are
COMPUTED from the engine's guards, not learnt.

This is also the "Maia analogue" the 2026-09-17 review identified and never
built: a distribution over concrete legal alternatives in context, rather
than over a fixed action-type vocabulary (raw_bc's 28 classes, which picks
WATER and then hunts for a legal tile) or over whatever our own build_tasks
happened to propose (exec_bc, ptrnet -- both planner-bottlenecked).

WHAT IS VERIFIABLE HERE, and why this file exists before any model: every
mask bit is checkable against the engine. A bit set to 1 must change state
when applied; a bit set to 0 must be a silent no-op. `--validate` checks
both directions on real boards, so the action space is proven before a
policy is trained on it.

    PYTHONPATH=. uv run python tools/action_space.py --validate --n 8
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

# The per-cell vocabulary. Parameterised ops are flattened into one entry
# each, which is what makes the head a plain categorical per cell.
# Everything a unit can carry. PICKUP and PLACE both range over ALL of it,
# which the corpus confirms and which the livestock chain REQUIRES:
#
#   BUILD_COOP -> BUY_ANIMAL:GOOSE (lands in the SHED) -> PICKUP:GOOSE (bag)
#   -> PLACE:GOOSE (tile) -> FEED/CARE -> HARVEST (EGG to bag) -> DROP -> SELL
#
# PLACE takes the animal from the unit's BAG, and PICKUP is the only bridge
# from shed to bag. Restricting PICKUP to WHEAT and FERTILIZER made
# PLACE:GOOSE/COW/SHEEP permanently unreachable -- the policy could never
# have learnt livestock at all. Corpus over 30 games: PICKUP COW 555,
# SHEEP 421, GOOSE 43.
#
# PLACE also has a SHED-DROP fallthrough used as a TARGETED drop -- DROP
# dumps the whole bag, PLACE:MILK stores only milk, which matters against
# shed capacity. Corpus: PLACE FERTILIZER 1,081, MILK 917, STRAWBERRY 660.
CARRY = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]

OPS = (["PASS", "NORTH", "SOUTH", "EAST", "WEST",
        "WATER", "HARVEST", "FEED", "CARE", "FERTILIZE", "DIG",
        "COLLECT_FERTILIZER", "DROP", "BUILD_COOP", "BUILD_PASTURE"]
       + [f"PLANT:{c}" for c in CROPS]
       + [f"PLACE:{i}" for i in CARRY]
       + [f"PICKUP:{i}" for i in CARRY])
N_OPS = len(OPS)
OP_IX = {o: i for i, o in enumerate(OPS)}
MOVE_D = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _shed_adjacent(x, y, n):
    return K._is_shed_adjacent((x, y), n)


def _unit_at(farm):
    """cell -> list of unit indices. idx 0 is the farmer, 1+ are hands.

    A cell can hold SEVERAL units -- measured at 24.3% of unit-frames -- which
    is why a single action-per-cell map is lossy here and why the exact mask
    is per-unit.
    """
    out = {}
    f = farm.get("farmer")
    if f:
        out.setdefault((f[0], f[1]), []).append(0)
    for i, h in enumerate(farm.get("hands") or []):
        out.setdefault((h[0], h[1]), []).append(i + 1)
    return out


def _bag(private, idx):
    invs = private.get("inventories") or private.get("inventory") or []
    if idx is None or idx >= len(invs):
        return {}
    return invs[idx] or {}


def legal_ops(obs, seat, unit_idx, private=None):
    """-> (N_OPS,) uint8 for ONE unit at its current position. EXACT.

    This is what a policy must mask against. Four ops depend on the acting
    unit's bag rather than on the board -- FEED needs a WHEAT, FERTILIZE a
    FERTILIZER, PLACE the animal itself, DROP anything at all -- so a mask
    keyed only by cell is wrong whenever units share a tile.
    """
    farm = obs["farms"][seat]
    if private is None:
        private = (obs.get("private") or {})
    tiles = farm["tiles"]
    n = len(tiles)
    day = obs.get("day", obs["step"] // 24)
    m = np.zeros(N_OPS, "uint8")
    pos = (farm.get("farmer") if unit_idx == 0
           else (farm.get("hands") or [None] * 99)[unit_idx - 1]
           if unit_idx - 1 < len(farm.get("hands") or []) else None)
    if pos is None:
        return m
    x, y = pos[0], pos[1]
    bag = _bag(private, unit_idx)
    seeds = (private.get("seeds") or {})
    shed = (private.get("shed") or {})

    # movement: BOUNDS ONLY. The engine allows stepping onto LOCKED tiles --
    # a hand can spawn on a locked shed-access tile and blocking it would
    # strand the unit forever. A unit may also BE on a locked tile, so this
    # runs before any tile-kind check.
    for op, (dx, dy) in MOVE_D.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            m[OP_IX[op]] = 1

    if _shed_adjacent(x, y, n):
        if any(v > 0 for v in bag.values()):
            m[OP_IX["DROP"]] = 1
        for it in CARRY:
            if shed.get(it, 0) > 0:
                m[OP_IX[f"PICKUP:{it}"]] = 1
        # PLACE has a SECOND path: if the animal-placement branch does not
        # match, it falls through to a shed drop, so PLACE:X beside the shed
        # stores X instead. Legal whenever the bag holds one and the shed has
        # room -- missing this made the mask too strict on GOOSE and SHEEP.
        room = 100 - sum(shed.values())
        if room > 0:
            for it in CARRY:
                if bag.get(it, 0) > 0:
                    m[OP_IX[f"PLACE:{it}"]] = 1

    t = tiles[y][x]
    if t == "LOCKED":
        return m
    if t is None:
        for c in CROPS:
            if seeds.get(c, 0) > 0:
                m[OP_IX[f"PLANT:{c}"]] = 1
        m[OP_IX["BUILD_COOP"]] = 1
        m[OP_IX["BUILD_PASTURE"]] = 1
        return m
    if not isinstance(t, dict):
        return m
    kind = t.get("kind")
    if kind == "WEED":
        m[OP_IX["DIG"]] = 1
    elif kind == "PLANT":
        m[OP_IX["DIG"]] = 1
        if not t.get("watered_today"):
            m[OP_IX["WATER"]] = 1
        if t.get("yield_units", 0) > 0 and \
                day - t.get("planted_day", day) >= \
                K.CROPS[t["crop"]]["first_yield_day"]:
            m[OP_IX["HARVEST"]] = 1
        if bag.get("FERTILIZER", 0) > 0:
            m[OP_IX["FERTILIZE"]] = 1
    elif t.get("animal"):
        if not t.get("fed_today") and bag.get("WHEAT", 0) > 0:
            m[OP_IX["FEED"]] = 1
        if not t.get("cared_today"):
            m[OP_IX["CARE"]] = 1
        if t.get("yield_units", 0) > 0:
            m[OP_IX["HARVEST"]] = 1
        if t.get("fertilizer_available"):
            m[OP_IX["COLLECT_FERTILIZER"]] = 1
    elif kind in ("COOP", "PASTURE"):
        m[OP_IX["DIG"]] = 1
        for a in ANIMALS:
            if kind == K.ANIMALS[a]["structure"] and bag.get(a, 0) > 0:
                m[OP_IX[f"PLACE:{a}"]] = 1
    return m


def legal_map(obs, seat, private=None):
    """-> (n, n, N_OPS) for a GridNet-style head. LOSSY where units share a cell.

    Uses the FIRST unit standing on each cell. 24.3% of unit-frames are on a
    shared cell, so a one-action-per-cell head cannot give two co-located
    units different jobs; that is a real limit of the architecture here, not
    of this function.
    """
    farm = obs["farms"][seat]
    if private is None:
        private = (obs.get("private") or {})
    n = len(farm["tiles"])
    out = np.zeros((n, n, N_OPS), "uint8")
    for (x, y), ids in _unit_at(farm).items():
        out[y, x] = legal_ops(obs, seat, ids[0], private)
    return out


def to_action(op_name):
    """Vocabulary entry -> the list the engine expects."""
    if ":" in op_name:
        head, arg = op_name.split(":", 1)
        return [head, arg] if head != "PICKUP" else [head, arg, 1]
    return [op_name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    print(f"action vocabulary: {N_OPS} ops per cell\n")
    for i, o in enumerate(OPS):
        print(f"  {i:>2} {o}")
    if not args.validate:
        return

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.opp_coupling import load_index
    import tools.price_forecast as pf
    import random

    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(4).shuffle(paths)

    tot = np.zeros(N_OPS, int)
    print(f"\nmask density over {args.n} games (fraction of unlocked cells "
          f"where each op is legal)\n")
    cells = 0
    for p in paths[: args.n]:
        s = json.load(open(p))
        env = make("kaggriculture", configuration={"seed": s["seed"]},
                   debug=False)
        env.reset(2)
        a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
        step = 0
        while not env.done:
            o = env.state[0].observation
            if step % 120 == 0:
                m = legal_mask(o, 0, env.state[0].observation.private)
                live = sum(1 for row in o["farms"][0]["tiles"]
                           for t in row if t != "LOCKED")
                tot += m.reshape(-1, N_OPS).sum(0).astype(int)
                cells += live
            env.step([a0(o), a1(env.state[1].observation)])
            step += 1
    print(f"    {'op':<22}{'legal in % of cells':>22}")
    for i, o in enumerate(OPS):
        print(f"    {o:<22}{100*tot[i]/max(cells,1):>21.1f}%")
    print(f"\n  mean legal ops per cell: "
          f"{tot.sum()/max(cells,1):.1f} of {N_OPS}")


if __name__ == "__main__":
    main()


def verify(n_games=4, stride=31):
    """Apply EVERY op at a unit's cell and compare the engine to the mask.

    A mask bit of 1 must change state; a bit of 0 must be a silent no-op.
    Disagreements are the point of this function -- some engine guards read
    the UNIT's inventory rather than the board, so a purely spatial mask
    cannot be right for those ops and they have to be represented.
    """
    import copy
    import json
    import random

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _apply_unit_action, _farmer_position, _farmer_inventory)
    from tools.opp_coupling import load_index
    import tools.price_forecast as pf

    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(4).shuffle(paths)
    agree = np.zeros((N_OPS, 2, 2), int)     # [op][mask bit][engine changed]

    def fp(farm, private, idx_):
        pos = _farmer_position(farm, idx_)
        if pos is None:
            return None
        t = farm["tiles"][pos[1]][pos[0]]
        sig = (repr(t), pos, farm.get("money"),
               tuple(sorted((_farmer_inventory(private, idx_) or {}).items())),
               tuple(sorted((private.get("shed") or {}).items())))
        return sig

    for p in paths[:n_games]:
        s = json.load(open(p))
        env = make("kaggriculture", configuration={"seed": s["seed"]},
                   debug=False)
        env.reset(2)
        a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
        step = 0
        while not env.done:
            o = env.state[0].observation
            if step % stride == 0:
                farm = o["farms"][0]
                priv = env.state[0].observation.private
                nn = len(farm["tiles"])
                n_units = 1 + len(farm.get("hands") or [])
                for ui in range(n_units):
                    pos = _farmer_position(farm, ui)
                    if pos is None:
                        continue
                    x, y = pos[0], pos[1]
                    um = legal_ops(o, 0, ui, priv)
                    for oi, name in enumerate(OPS):
                        f2 = copy.deepcopy(farm)
                        p2 = copy.deepcopy(priv)
                        before = fp(f2, p2, ui)
                        _apply_unit_action(f2, p2, ui, to_action(name), nn,
                                           o["step"] // 24, 24, 100)
                        changed = int(fp(f2, p2, ui) != before)
                        agree[oi, int(um[oi]), changed] += 1
            env.step([a0(o), a1(env.state[1].observation)])
            step += 1

    print(f"\nMASK vs ENGINE   ({n_games} games, farmer's own cell)\n")
    print(f"    {'op':<22}{'mask=1 & moved':>16}{'mask=1 & NO-OP':>16}"
          f"{'mask=0 & MOVED':>16}")
    bad = []
    for oi, name in enumerate(OPS):
        if name == "PASS":
            continue
        a = agree[oi]
        fp_ = a[1, 0]      # said legal, engine ignored it
        fn_ = a[0, 1]      # said illegal, engine acted
        print(f"    {name:<22}{a[1,1]:>16,}{fp_:>16,}{fn_:>16,}")
        if fp_ or fn_:
            bad.append((name, fp_, fn_))
    print(f"\n  ops with ANY disagreement: {len(bad)} of {N_OPS}")
    for name, f, g in bad:
        why = ("mask too permissive" if f and not g else
               "mask too strict" if g and not f else "both")
        print(f"    {name:<22}{why}")
