"""Pick up a FULL batch of wheat, not just the unit's own zone's need.

Measured defect. `scheduler.assign` step 2 sizes a wheat pickup by the ZONE:

    need = zone_needs[z]["WHEAT"] - zone_carry[z]["WHEAT"]
    n = min(need, shed_left["WHEAT"], 5)

With ~11 zones and ~14 animals a zone holds 1-2 unfed animals, so a unit makes a
shed round-trip to collect 2 wheat. We issue 348 PICKUPs a game; a 1750-rated
tape issues 194, and the gap is concentrated in the opening (PICKUP is 15.5% of
our day-6 actions against the tape's 7.9%).

That also explains the tape's day-12 step change. Its PLACE goes 12.1% -> 0 and
PICKUP -> 0 once the herd is built, and its moves/action halves 1.23 -> 0.66.
Ours never falls: PICKUP is still 14.8% of actions on day 20. Our logistics phase
never ends.

Why this is not the refuted farm-wide courier. The code comment records that
experiment ("150-200 zone moves/day for 40-60 actions"), and the engine has no
unit-to-unit transfer, so whoever carries wheat must stand on each animal. But
the herd is CLUSTERED: measured at 14.4 animals, mean 2.20 tiles from the shed,
with 10.6 of them within 3 tiles of one another. Feeding five of those from one
pickup is a short walk, not a farm crossing.

So: top the pickup up to a full batch whenever the FARM still has unfed animals,
and let the existing pass-2 help radius (HELP_RADIUS=5) place the surplus. Zone
provisioning is otherwise untouched, and fertilizer is left alone -- fertilize
targets are crop tiles, which are spread out.

    PYTHONPATH=. uv run python tools/make_hauler_variant.py --name h-batch
    PYTHONPATH=. uv run python tools/make_hauler_variant.py --name h-batch8 --batch 8
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = """    for idx in sorted(free):
        z = idx % n_zones
        for item in ("WHEAT", "FERTILIZER"):
            need = zone_needs.get(z, {}).get(item, 0) - zone_carry.get(z, {}).get(item, 0)
            n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)"""

REPL = """    # Farm-wide unfed count, so a wheat pickup can be sized to what the UNIT
    # can carry rather than to what its own zone happens to need. The herd is
    # clustered (14.4 animals, mean 2.20 tiles from the shed, 10.6 of them
    # within 3 tiles of each other), so a loaded unit feeding five of them is a
    # short walk -- not the farm-wide courier the zone scheme was built to
    # avoid. Pass 2's HELP_RADIUS places the surplus.
    _farm_wheat_need = 0
    for _t in tasks:
        if _t.need == "WHEAT":
            _farm_wheat_need += 1
    _farm_wheat_need = max(0, _farm_wheat_need
                           - sum(view.unit_inv(i).get("WHEAT", 0)
                                 for i, _, _ in me.units))
    for idx in sorted(free):
        z = idx % n_zones
        for item in ("WHEAT", "FERTILIZER"):
            need = zone_needs.get(z, {}).get(item, 0) - zone_carry.get(z, {}).get(item, 0)
            if item == "WHEAT" and need > 0 and _farm_wheat_need > need:
                # top the trip up to a full load while the farm still needs it
                need = min(_farm_wheat_need, HAULER_BATCH)
            n = min(need, shed_left[item], HAULER_BATCH if item == "WHEAT" else 3)"""

TRACK = """            shed_left[item] -= n
            zone_carry[z][item] = zone_carry[z].get(item, 0) + n"""
TRACK_REPL = """            shed_left[item] -= n
            zone_carry[z][item] = zone_carry[z].get(item, 0) + n
            if item == "WHEAT":
                _farm_wheat_need = max(0, _farm_wheat_need - n)"""

CONST = """HAULER_BATCH = __BATCH__
DELIVER_RADIUS = __RADIUS__

"""


SUPPRESS_REPL = """    # Farm-wide wheat accounting. zone_carry only counts wheat held by units
    # whose (idx % n_zones) == z, so wheat in ANOTHER zone's hands is invisible
    # and zone 7 sends its worker to the shed while zone 3's worker stands there
    # holding five. Measured: 43.6% of wheat pickups are issued while the farm
    # already holds enough for every unfed animal (mean 8.33 held vs 11.87
    # unfed), and we haul 595 wheat to deliver 333 feeds.
    _farm_wheat = sum(view.unit_inv(i).get("WHEAT", 0) for i, _, _ in me.units)
    _farm_need = sum(1 for _t in tasks if _t.need == "WHEAT")
    for idx in sorted(free):
        z = idx % n_zones
        for item in ("WHEAT", "FERTILIZER"):
            need = zone_needs.get(z, {}).get(item, 0) - zone_carry.get(z, {}).get(item, 0)
            if item == "WHEAT" and _farm_wheat >= _farm_need:
                continue          # the farm already carries enough; do not go
            n = min(need, shed_left[item], HAULER_BATCH if item == "WHEAT" else 3)"""

SUPPRESS_TRACK = """            shed_left[item] -= n
            zone_carry[z][item] = zone_carry[z].get(item, 0) + n
            if item == "WHEAT":
                _farm_wheat += n"""


# Mode 3: make carried wheat get DELIVERED.
# Diagnosis chain: per-zone provisioning is 43.6% redundant (the farm already
# holds enough when the trip is issued), but suppressing on farm-wide stock
# collapsed FEED 333 -> 208 because the stock is in the WRONG HANDS -- zone
# rounds only offer a unit its own zone's tasks, so unit 3's five wheat cannot
# reach unit 7's animals. Wheat is fungible; the zone partition makes it not.
# The herd is clustered (14.4 animals, 10.6 within 3 tiles of one another), so a
# carrier can serve several. This adds one pass, before the zone rounds, where
# any unit already holding WHEAT feeds the nearest unfed animal in reach.
DELIVER_FIND = """    # Pass 1 \u2014 own zone, real work (below filler): animal deliveries first."""

DELIVER_REPL = """    # ---- 2.5 Deliver what is already carried --------------------------------
    # A unit holding wheat feeds the nearest unfed animal within reach, whatever
    # zone it is in. Without this, carried wheat idles while another zone's
    # worker walks to the shed for more.
    for idx in sorted(free):
        if view.unit_inv(idx).get("WHEAT", 0) < 1:
            continue
        ux, uy = units[idx]
        best = None
        for _t in tasks:
            if _t.need != "WHEAT" or (_t.x, _t.y, _t.op[0]) in claimed:
                continue
            d = _dist(ux, uy, _t.x, _t.y)
            if d > DELIVER_RADIUS:
                continue
            if best is None or d < best[0]:
                best = (d, _t)
        if best is not None and try_execute(idx, best[1]):
            free.discard(idx)
    _count_moves(actions, _seen, "deliver")
    _seen = set(actions)

    # Pass 1 \u2014 own zone, real work (below filler): animal deliveries first."""


def build(base, name, batch=5, mode="batch", radius=5):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text and TRACK in text:
            if mode == "deliver":
                if DELIVER_FIND not in text:
                    raise RuntimeError("zone-rounds anchor not found")
                text = text.replace(DELIVER_FIND, DELIVER_REPL, 1)
                text = text.replace(FIND, SUPPRESS_REPL, 1).replace(TRACK, SUPPRESS_TRACK, 1)
            elif mode == "suppress":
                text = text.replace(FIND, SUPPRESS_REPL, 1).replace(TRACK, SUPPRESS_TRACK, 1)
            else:
                text = text.replace(FIND, REPL, 1).replace(TRACK, TRACK_REPL, 1)
            text = (CONST.replace("__BATCH__", repr(int(batch)))
                         .replace("__RADIUS__", repr(int(radius)))) + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("zone-provisioning anchors not found")
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
    ap.add_argument("--batch", type=int, default=5,
                    help="max wheat carried per shed trip (shipped: 5)")
    ap.add_argument("--radius", type=int, default=5,
                    help="how far a carrier will walk to feed (HELP_RADIUS is 5)")
    ap.add_argument("--mode", choices=("batch", "suppress", "deliver"), default="batch",
                    help="batch = bigger loads (REFUTED: PICKUP rose, FEED flat); "
                         "suppress = skip the trip when the FARM already holds "
                         "enough wheat for every unfed animal")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.batch, a.mode, a.radius)}  "
          f"(mode={a.mode}, batch={a.batch}, radius={a.radius})")
