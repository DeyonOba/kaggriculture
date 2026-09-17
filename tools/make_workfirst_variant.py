"""Do the work under your feet before running goods to the shed.

Localised from the decision diff against M & M & P & Q (3,093, reactive). On the
23,900 unit-turns where they WORK and champ-h8 would MOVE:

    a doable task on our unit's OWN tile        94.9%
    distance to our nearest task                mean 0.05, median 0
    our unit is carrying goods                  81.7%  (mean load 3.6)
    ...and is mid-trip to the shed              76.9%

So it is not a routing problem -- there is nothing to route, the work is
underfoot. `scheduler.assign` section 1 ("Liquidation / heavy-load shuttles")
runs BEFORE every work pass and calls free.discard(idx), so a unit standing on a
ripe crop while carrying 3 items is sent to the shed before any task pass sees
it. Hence PICKUP 348/game against their 194, and 1.45 moves per productive
action against their 0.71.

This defers the shuttle for one turn when the unit is standing on work it can
actually do. The unit performs the action; the shuttle fires on a later turn
when its tile is exhausted. Guards kept: a unit at/over `--max-load` shuttles
regardless, and the endgame/aggressive paths are untouched, so nothing strands.

Note this is NOT the earlier v-tile attempt, which added a same-tile pass AFTER
the shuttle and therefore never saw these units.

    PYTHONPATH=. uv run python tools/make_workfirst_variant.py --name w-first
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = """        if total < threshold:
            continue
        tx, ty = _nearest_shed_tile(x, y)"""

REPL = """        if total < threshold:
            continue
        # Work under your feet first. The shuttle used to preempt every task
        # pass, so a unit standing on a doable task walked to the shed instead:
        # measured at 94.9% of the turns where a 3,093-rated agent worked and we
        # would have moved. Defer one turn; the shuttle still fires once the
        # tile is exhausted, and a full unit (>= WORKFIRST_MAX_LOAD) always goes.
        if WORKFIRST and total < WORKFIRST_MAX_LOAD and not aggressive:
            _inv_here = view.unit_inv(idx)
            _underfoot = False
            for _t in tasks:
                if _t.x != x or _t.y != y:
                    continue
                if _t.need and _inv_here.get(_t.need, 0) < 1:
                    continue
                _underfoot = True
                break
            if _underfoot:
                continue
        tx, ty = _nearest_shed_tile(x, y)"""

CONSTS = """WORKFIRST = True
WORKFIRST_MAX_LOAD = __MAXLOAD__

"""


def build(base, name, max_load=12):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, REPL, 1)
            text = CONSTS.replace("__MAXLOAD__", repr(int(max_load))) + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("shuttle block not found")
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
    ap.add_argument("--max-load", type=int, default=12,
                    help="always shuttle at or above this carried total")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.max_load)}  (max_load={a.max_load})")
