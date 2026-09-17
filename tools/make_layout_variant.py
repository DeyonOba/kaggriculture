"""Cluster crops spatially instead of scattering them. Allocation held constant.

`scheduler.build_tasks` assigns crops to ground round-robin:

    wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
    for i, (x, y) in enumerate(empties[cursor:]):
        tasks.append(Task(P_GROW, x, y, ["PLANT", wanted[i % len(wanted)]]))

so every crop type is smeared across the whole farm. The comment defends this
("so every zone plants its own ground instead of all hands converging on the
few center-most empties") and that reasoning is right about PLANTING day -- but
planting is one action per tile, while watering, fertilising and harvesting are
many, and those come due by crop. Tiles planted the same day with the same crop
ripen the same day. A scattered layout therefore guarantees that each day's due
work is spread over the entire farm.

That is why ten routing attempts returned null. Routing decides the order of
visits to a fixed set of stops; it cannot make a scattered stop set compact.
The 976 excess moves are a property of the LAYOUT, and the layout is chosen
here, by one modulo.

This changes only WHICH tile gets WHICH crop. The quota per crop is copied from
the same `wanted` multiset, so the allocation -- the thing the LP gate already
proved flat -- is bit-for-bit identical. Only the spatial arrangement moves.

    block  contiguous runs over the distance-sorted empties, so each crop
           occupies one annulus around the farm centre.
    zone   whole zones get a single crop, so a worker's own zone ripens
           together and its day is one crop's worth of work in one place.

    PYTHONPATH=. uv run python tools/make_layout_variant.py --name lay-block --mode block
    PYTHONPATH=. uv run python tools/make_layout_variant.py --name lay-zone  --mode zone
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''        wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
        if wanted:
            for i, (x, y) in enumerate(empties[cursor:]):
                tasks.append(Task(P_GROW, x, y, ["PLANT", wanted[i % len(wanted)]]))'''

BLOCK = '''        wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
        if wanted:
            # Contiguous blocks instead of `i % len(wanted)`. `empties` is
            # already sorted by distance from the farm centre, so each crop
            # lands in one annulus and same-crop tiles sit next to each other.
            # Identical crop multiset, identical counts -- only the tile each
            # crop is assigned to changes.
            free = empties[cursor:]
            n = len(free)
            for i, (x, y) in enumerate(free):
                crop = wanted[min(len(wanted) - 1, i * len(wanted) // max(1, n))]
                tasks.append(Task(P_GROW, x, y, ["PLANT", crop]))'''

ZONE = '''        wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
        if wanted:
            # One crop per zone. Workers are partitioned by `idx % n_zones`, so
            # giving a zone a single crop means that worker's ground ripens
            # together: its whole day is one crop's work in one place, which is
            # the shape the round-robin destroys.
            _zof, _nz = zone_map(view)
            free = empties[cursor:]
            buckets = {}
            for p in free:
                buckets.setdefault(_zof.get(p, -1), []).append(p)
            order = sorted(buckets)
            # Walk zones in order, drawing from the crop multiset as we go, so
            # each crop covers a run of whole zones in proportion to the ground
            # they hold and the quotas stay reachable.
            total = len(free)
            i = 0
            for z in order:
                for (x, y) in buckets[z]:
                    crop = wanted[min(len(wanted) - 1,
                                      i * len(wanted) // max(1, total))]
                    tasks.append(Task(P_GROW, x, y, ["PLANT", crop]))
                    i += 1'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("block", "zone"), default="block")
    args = ap.parse_args()

    src = open(os.path.join(AGENTS, args.base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, BLOCK if args.mode == "block" else ZONE, 1)
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("plant-assignment block not found")

    d = os.path.join(AGENTS, args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    print(f"wrote {p}  (mode={args.mode})")


if __name__ == "__main__":
    main()
