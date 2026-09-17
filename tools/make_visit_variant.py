"""A new executor rule: finish the tile you are standing on before you walk.

`scheduler.assign` is BAND-MAJOR. Pass 1 serves P_SURVIVAL..P_FILLER inside the
worker's zone, pass 2 helps within HELP_RADIUS, pass 3 takes filler. Distance is
only a tie-break INSIDE a band, so a worker standing on a tile that still has a
P_FILLER task can be sent across its zone by pass 1 for a P_GROW task. A free
0-move action is traded for an N-move one.

Measured on champ-h8:

    moves                        3,864/game
    ...with work underfoot       2,846/game   (73.6% of moves)
    tasks abandoned underfoot    4,110/game

and the consequence is the whole efficiency gap: 38.2% of our actions follow
zero moves, against ahmed's 49.6%; 1.45 moves per productive action against
ahmed's 0.84 and the tape's 0.85.

This is not the eleven earlier routing attempts. Every one of those changed
which tasks a worker could REACH (walk caps, zone sizing, priority bands,
idle-help scoring, prize-collecting value, pickup batching, dawn tours, optimal
matching, crop clustering) while leaving the band-major order intact. This
changes the ORDER: a pass 0 runs first and lets every worker take any doable
task on its own tile, whatever band it is in.

Survival is not sacrificed. The pass is skipped for a worker once the day is
late enough that `pick` starts prioritising dies-tonight work, and the existing
survival override at the end of `assign` still runs.

    PYTHONPATH=. uv run python tools/make_visit_variant.py --name v-tile
    PYTHONPATH=. uv run python tools/make_visit_variant.py --name v-tile-late --when-late
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.'''

REPL = '''    # Pass 0 — VISIT-MAJOR: finish the tile you are standing on.
    # The rest of assign() is band-major, which walks workers away from free
    # work underfoot on 73.6% of their moves. A same-tile action costs one turn
    # and yields one action; the same action after an N-tile walk costs N+1
    # turns for the same yield, so underfoot work wins on any band unless
    # something is dying elsewhere.
    if SAME_TILE_FIRST and (WHEN_LATE or not late):
        for idx in sorted(free):
            _ux, _uy = units[idx]
            _inv = view.unit_inv(idx)
            _best = None
            for _t in all_tasks + global_tasks:
                _d = _dist(_ux, _uy, _t.x, _t.y)
                if _d > VISIT_RADIUS:
                    continue
                if _t.pri > SAME_TILE_MAX_PRI:
                    continue
                if (_t.x, _t.y, _t.op[0]) in claimed:
                    continue
                if _t.need and _inv.get(_t.need, 0) < 1:
                    continue
                # Distance first, band second -- the inversion of assign()'s
                # ordering, and the whole point of the pass.
                _k = (_d, _t.pri)
                if _best is None or _k < _best[0]:
                    _best = (_k, _t)
            if _best is not None and try_execute(idx, _best[1]):
                free.discard(idx)
    _count_moves(actions, _seen, "sametile")
    _seen = set(actions)

    # Pass 1 — own zone, real work (below filler): animal deliveries first.'''

CONSTS = '''SAME_TILE_FIRST = __ON__
SAME_TILE_MAX_PRI = __MAXPRI__
WHEN_LATE = __WHENLATE__
VISIT_RADIUS = __RADIUS__

'''


def build(base, name, on=True, max_pri=99.0, when_late=False, radius=0):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if FIND not in text:
                raise RuntimeError("Pass 1 anchor not found")
            text = text.replace(FIND, REPL, 1)
            text = (CONSTS.replace("__ON__", repr(bool(on)))
                          .replace("__MAXPRI__", repr(float(max_pri)))
                          .replace("__WHENLATE__", repr(bool(when_late)))
                          .replace("__RADIUS__", repr(int(radius)))) + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("scheduler module not found")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-pri", type=float, default=99.0,
                    help="only take underfoot tasks at or below this priority")
    ap.add_argument("--when-late", action="store_true",
                    help="also run the pass after the sweep deadline hour")
    ap.add_argument("--radius", type=int, default=0,
                    help="0 = own tile only; 1 = tile plus neighbours, etc.")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, True, a.max_pri, a.when_late, a.radius)}  "
          f"(radius={a.radius}, max_pri={a.max_pri}, when_late={a.when_late})")
