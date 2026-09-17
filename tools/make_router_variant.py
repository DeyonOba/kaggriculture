"""Value-driven task router: replace nearest-first with prize-collecting.

The current `scheduler.pick` scores a task `(urgent, d, task.pri)` -- a
lexicographic tuple that sorts by DISTANCE first and priority last. A worker
therefore walks to the nearest doable task whatever it is worth, which is how
champ-h8 ends up at 1.37 moves per productive action against the tape's 0.84.
976 excess moves a game at the LP's ~$23/action is ~$22k, essentially the whole
gap that allocation and labour supply do not explain.

This replaces the ordering with an economic one:

    score(task) = dollar_value(task) - LAMBDA * distance

maximised, where the value is computed from the tile and the live market price,
and LAMBDA is the opportunity cost of spending a turn walking. That is a greedy
prize-collecting assignment (the standard heuristic for team orienteering).

Why this is not the five failed routing attempts again: those were all
RELAXATIONS of the zone scheme -- walk caps, zone sizing, priority-band tweaks,
idle-help scoring. None changed what a worker is trying to maximise. This one
does, and it is the specific fix the chaining result demands: chaining added 9%
more actions and lost $6,385 because it optimised action COUNT; a prize-
collecting router optimises action VALUE net of travel.

    PYTHONPATH=. uv run python tools/make_router_variant.py --name route-v1 --lam 23
"""

import argparse
import ast
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

VALUE_FN = '''

# --- value-driven routing (tools/make_router_variant.py) ---------------------
# Dollar value of doing a task now, from the tile state and the live price.
ROUTE_LAMBDA = __LAM__          # $ forgone by spending one turn walking
_ANIMAL_PROD = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _tile_at(view, x, y):
    try:
        return view.me.tiles[y][x]
    except Exception:
        return None


def task_value(view, task):
    """What this task is worth in dollars if done this turn."""
    op = task.op[0]
    px = view.market_prices
    t = _tile_at(view, task.x, task.y)
    if not isinstance(t, dict):
        # PLANT on bare ground: expected yield of the crop, less the seed.
        if op == "PLANT" and len(task.op) > 1:
            crop = task.op[1]
            cd = CROPS.get(crop) or {}
            return max(0.0, cd.get("max_yield", 3) * 0.7 * px.get(crop, 25)
                       - cd.get("seed", 10))
        return 1.0

    kind = t.get("kind")
    animal = t.get("animal")
    if op == "HARVEST":
        n = float(t.get("yield_units", 0) or 0)
        if kind == "PLANT":
            return n * px.get(t.get("crop"), 25)
        prod = _ANIMAL_PROD.get(animal)
        return n * px.get(prod, 50) if prod else 0.0

    if op == "WATER" and kind == "PLANT":
        crop = t.get("crop")
        cd = CROPS.get(crop) or {}
        p = float(px.get(crop, 25))
        dry = t.get("consecutive_unwatered", 0) or 0
        # A second dry day turns the tile into a weed: the whole remaining
        # yield of the plant is at stake, not one unit.
        if dry >= 1:
            return p * max(1.0, cd.get("max_yield", 3))
        if cd.get("ongoing"):
            return p * 0.6          # keeps the production schedule on track
        age = view.day - t.get("planted_day", view.day)
        ws = (cd.get("max_yield_day", 4) + 1) // 2
        if ws <= age <= cd.get("max_yield_day", 4):
            return p * (2.0 if t.get("fertilized_until_day", -1) >= view.day else 1.0)
        return p * 0.2

    if op == "FEED" and animal:
        prod = _ANIMAL_PROD.get(animal, "MILK")
        p = float(px.get(prod, 50))
        # unfed twice and the animal is gone for good
        return p * (12.0 if (t.get("consecutive_unfed", 0) or 0) >= 1 else 1.5)

    if op == "CARE" and animal:
        return float(px.get(_ANIMAL_PROD.get(animal, "MILK"), 50))

    if op == "COLLECT_FERTILIZER":
        return float(px.get("FERTILIZER", 100))

    if op == "FERTILIZE" and kind == "PLANT":
        return 2.0 * float(px.get(t.get("crop"), 25))

    if op == "DIG":
        return 15.0
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return 120.0
    return 5.0
'''

PICK_FIND = '''            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            score = (urgent, d, task.pri)
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None'''

PICK_REPL = '''            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            # Prize-collecting: maximise value net of the travel it costs.
            # Urgency stays lexicographic so a dies-tonight task still wins.
            score = (urgent, -(task_value(view, task) - ROUTE_LAMBDA * d))
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lam", type=float, default=23.0)
    ap.add_argument("--mode", choices=("value", "batch"), default="value",
                    help="value = prize-collecting pick(); batch = bigger shed "
                         "pickups, fewer round trips")
    ap.add_argument("--wheat-batch", type=int, default=12)
    ap.add_argument("--fert-batch", type=int, default=8)
    args = ap.parse_args()

    src = open(os.path.join(AGENTS, args.base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, patched = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if args.mode == "value":
                if PICK_FIND not in text:
                    raise RuntimeError("pick() scoring block not found")
                text = text.replace(PICK_FIND, PICK_REPL, 1)
                anchor = "def _dist("
                k = text.index(anchor)
                text = text[:k] + VALUE_FN.replace("__LAM__", repr(args.lam)) + "\n\n" + text[k:]
            else:
                # Each shed pickup is a round trip. champ-h8 makes 131 more of
                # them than the tape; carrying more per trip is the direct fix.
                old = ('batch = 5 if item == "WHEAT" else '
                       '(3 if item == "FERTILIZER" else 2)')
                if old not in text:
                    raise RuntimeError("global pickup batch not found")
                text = text.replace(old, f'batch = {args.wheat_batch} if item == "WHEAT" '
                                         f'else ({args.fert_batch} if item == "FERTILIZER" '
                                         f'else 2)', 1)
                zold = 'n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)'
                if zold not in text:
                    raise RuntimeError("zone pickup batch not found")
                text = text.replace(zold, f'n = min(need, shed_left[item], '
                                          f'{args.wheat_batch} if item == "WHEAT" '
                                          f'else {args.fert_batch})', 1)
            patched = True
        out.append((mod, text))
    if not patched:
        raise RuntimeError("scheduler module not found in bundle")

    d = os.path.join(AGENTS, args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    tag = (f"lambda ${args.lam:g}/turn" if args.mode == "value"
           else f"pickup batches wheat {args.wheat_batch} / fert {args.fert_batch}")
    print(f"wrote {p}  ({args.mode}: {tag})")


if __name__ == "__main__":
    main()
