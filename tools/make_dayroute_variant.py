"""Whole-day route planning: commit each worker to a tour at dawn.

Measured defect. Per worker-day champ-h8 and the tape touch the SAME number of
tiles (5.4 vs 5.2) but spend 12.4 moves against 9.4 -- three wasted moves per
worker-day, ~990 a season, which matches the 976 excess moves that account for
essentially the whole unexplained gap to the tape.

It is not task SELECTION. Two value-driven selection attempts failed (a
prize-collecting pick() made it much worse; bigger shed pickups did not
replicate), and the priority bands already approximate the right thing. It is
route QUALITY: `scheduler.pick` chooses the nearest doable task EVERY TURN and
re-chooses as tasks get claimed, so a worker oscillates instead of following a
tour.

The tape reaches 0.85 moves/action with a FIXED SCRIPT that makes no decisions
at all -- a recording out-routes every adaptive scheduler here, because it was
optimised as a whole day offline. This does the same thing online: at dawn each
unit's zone tasks are ordered into a tour (nearest-neighbour seeded, then
2-opt), and the unit follows it instead of re-deciding each turn.

What is deliberately kept: the shuttle/drop pass, zone provisioning, animal
deliveries, the help and filler passes for units whose tour is exhausted, and
the survival override -- so nothing urgent is ever stranded by a committed
route. Tasks that appear later in the day (a crop ripening) are picked up by
the existing greedy passes, so this is a hybrid, not a replacement.

    PYTHONPATH=. uv run python tools/make_dayroute_variant.py --name route-day
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

ROUTE_FN = '''

# --- whole-day route planning (tools/make_dayroute_variant.py) ---------------
# Per-seat, per-day committed tours. Module state is safe here the same way
# MOVE_STATS is: each bundle exec gets its own globals dict, and the state is
# keyed by seat and invalidated when the day changes.
_ROUTES = {}
_ROUTE_2OPT_GUARD = 24


def _tour_len(start, order):
    cx, cy = start
    total = 0
    for p in order:
        total += _dist(cx, cy, p[0], p[1])
        cx, cy = p[0], p[1]
    return total


def _build_tour(start, pts, budget):
    """Nearest-neighbour seed, TRUNCATE to what the day can reach, then 2-opt.

    Truncating before the 2-opt is the whole point. A full TSP tour minimises
    the cost of visiting EVERY stop, so 2-opt will happily move a distant stop
    to the front to shorten the overall loop -- but a worker completes only
    about five of its zone's stops in a day, so it then spends the day walking
    there. Measured: touring the full set cost 532 productive actions and took
    moves/action from 1.37 to 1.63. With a time budget this is a prize-
    collecting tour over a subset, not a TSP.
    """
    if not pts:
        return []
    rem = list(pts)
    order = []
    cx, cy = start
    while rem and len(order) < budget:
        k = min(range(len(rem)),
                key=lambda i: _dist(cx, cy, rem[i][0], rem[i][1]))
        p = rem.pop(k)
        order.append(p)
        cx, cy = p[0], p[1]
    best = _tour_len(start, order)
    improved = True
    guard = 0
    while improved and guard < _ROUTE_2OPT_GUARD:
        improved = False
        guard += 1
        for a in range(len(order) - 1):
            for b in range(a + 2, len(order) + 1):
                cand = order[:a] + order[a:b][::-1] + order[b:]
                lc = _tour_len(start, cand)
                if lc < best - 1e-9:
                    order, best, improved = cand, lc, True
    return order


def _plan_day_routes(view, tasks, zone_of, n_zones, units):
    """One committed tour per unit over its own zone's tasks."""
    by_zone = {}
    for t in tasks:
        z = zone_of.get((t.x, t.y))
        if z is None:
            continue
        by_zone.setdefault(z, []).append(t)
    routes = {}
    for idx, pos in units.items():
        pool = by_zone.get(idx % n_zones, [])
        # Survival first, then a spatial tour over the rest: a dies-tonight
        # task must not wait for the tour to reach it.
        urgent = [(t.x, t.y, t.op[0]) for t in pool if t.pri == P_SURVIVAL]
        rest = [(t.x, t.y, t.op[0]) for t in pool if t.pri != P_SURVIVAL]
        # A stop costs roughly one action plus the steps to reach it; measured
        # ~5.4 tiles per worker-day. Budget the tour to what the remaining
        # hours can actually serve.
        budget = max(3, view.hours_left_today // 3)
        seen = set()
        uniq = []
        for p in urgent + _build_tour(pos, rest, budget):
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        routes[idx] = uniq
    return routes
'''

PASS1_FIND = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)'''

PASS1_REPL = '''    # Pass 1 — follow the day's committed tour. Animal deliveries still jump
    # the queue (a carried cow is placed anywhere, immediately).
    _st = _ROUTES.setdefault(view.player, {"day": -1, "routes": {}})
    if _st["day"] != view.day:
        _st["day"] = view.day
        _st["routes"] = _plan_day_routes(view, tasks, zone_of, n_zones, units)
    _live = {}
    for t in tasks:
        _live.setdefault((t.x, t.y, t.op[0]), t)
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        route = _st["routes"].get(idx)
        if task is None and route:
            # Drop stops that are done, claimed by someone else, or no longer
            # real, then aim at the head of what is left.
            while route:
                head = route[0]
                cand = _live.get(head)
                if cand is None or head in claimed or cand.pri >= P_FILLER:
                    route.pop(0)
                    continue
                task = cand
                break
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            # Advance the tour only when the stop was actually performed, not
            # when the unit merely stepped toward it.
            if (route and route[0] == (task.x, task.y, task.op[0])
                    and actions.get(idx) == task.op):
                route.pop(0)
            free.discard(idx)'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    src = open(os.path.join(AGENTS, args.base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if PASS1_FIND not in text:
                raise RuntimeError("Pass 1 block not found")
            text = text.replace(PASS1_FIND, PASS1_REPL, 1)
            k = text.index("def _dist(")
            text = text[:k] + ROUTE_FN + "\n\n" + text[k:]
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("scheduler module not found")

    d = os.path.join(AGENTS, args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
