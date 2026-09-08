"""Turn-level task generation and ZONE-BASED unit assignment.

Field measurement (episode 106290169 vs a $175k opponent) showed the old
greedy nearest-task assignment walking 3.2 steps per productive action vs the
leaders' 1.2, starving distant quadrants entirely (SW: 72 actions all game).
This executor gives each worker a contiguous serpentine ZONE for the day and
a fixed round inside it — fertilize > water > feed > care > harvest > plant —
so travel is intra-zone, every unlocked quadrant gets dedicated labor, and
the care regime (the 3x animal-output multiplier) is guaranteed rather than
priority-contested. Survival tasks keep a global override; logistics
(pickups, deliveries, drops, liquidation shuttles) stay global.
"""

import math

from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, SHED_ACCESS, TURNS_PER_DAY
from kagent.planner import FINAL_STEP

# Task priorities (lower = more urgent).
P_SURVIVAL = 0
P_HARVEST = 1
P_TEND = 2
P_GROW = 3
P_FILLER = 5

PLANT_LAST_HOUR = 21  # leave >= 2 turns to water the fresh plant
DROP_THRESHOLD = 10   # mid-game inventory size that triggers a shed trip
MIN_ZONE_TILES = 5


class Task:
    __slots__ = ("pri", "x", "y", "op", "need")

    def __init__(self, pri, x, y, op, need=None):
        self.pri = pri
        self.x = x
        self.y = y
        self.op = op
        self.need = need  # item that must be in the unit's inventory


def _dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def _step_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    if dx != 0:
        return "EAST" if dx > 0 else "WEST"
    return None


def _nearest_shed_tile(x, y):
    return min(SHED_ACCESS, key=lambda t: _dist(x, y, t[0], t[1]))


def _unlocked(view, x, y):
    return view.me.tile(x, y) != "LOCKED"


def zone_map(view):
    """Serpentine partition of unlocked tiles into contiguous worker zones.

    Returns (zone_of: {(x, y): zone_id}, n_zones). Deterministic given the
    farm state, so a unit's zone (unit_index % n_zones) is sticky within a
    day without any cross-turn state.
    """
    quads = [("NW", 0, 0), ("NE", 5, 0), ("SW", 0, 5), ("SE", 5, 5)]
    unlocked = view.me.unlocked_quadrants
    path = []
    for name, x0, y0 in quads:
        if name not in unlocked:
            continue
        for r in range(5):
            cols = range(5) if r % 2 == 0 else range(4, -1, -1)
            for c in cols:
                path.append((x0 + c, y0 + r))
    n_units = max(1, len(view.me.units))
    n_zones = max(1, min(n_units, len(path) // MIN_ZONE_TILES))
    chunk = math.ceil(len(path) / n_zones)
    zone_of = {tile: min(i // chunk, n_zones - 1) for i, tile in enumerate(path)}
    return zone_of, n_zones


def build_tasks(view, plan):
    """Generate this turn's pending tile tasks."""
    tasks = []
    day = view.day
    me = view.me

    fert_targets = set()

    # --- Plants: water / harvest / fertilize ---------------------------------
    for x, y, t in me.plants():
        crop = CROPS.get(t["crop"])
        if crop is None:
            continue
        age = day - t["planted_day"]
        yields = t.get("yield_units", 0)

        decaying = 0 <= t.get("max_lifespan_step", -1) <= view.step + 24
        if crop["ongoing"]:
            if yields >= crop["max_yield"] - 1 or (yields > 0 and (decaying or plan.liquidation)):
                tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))
        else:
            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= crop["max_yield_day"]
            )
            if ripe or (yields > 0 and decaying and age >= crop["first_yield_day"]):
                tasks.append(Task(P_HARVEST if not decaying else P_SURVIVAL, x, y, ["HARVEST"]))
                continue

        if t.get("fertilized_until_day", -1) < day and not plan.liquidation:
            want_fert = (
                (t["crop"] == "TOMATO" and age in (8, 11))
                or (t["crop"] == "WHEAT" and age == 2)
                or (t["crop"] == "CARROT" and age == 2)
            )
            if want_fert:
                tasks.append(Task(P_TEND, x, y, ["FERTILIZE"], need="FERTILIZER"))
                fert_targets.add((x, y))

    for x, y, t in me.plants():
        if t.get("watered_today"):
            continue
        crop = CROPS.get(t["crop"])
        if crop is None:
            continue
        age = day - t["planted_day"]
        if (x, y) in fert_targets and t.get("consecutive_unwatered", 0) == 0 and view.hour < 18:
            continue
        if t.get("consecutive_unwatered", 0) >= 1:
            pri = P_SURVIVAL  # dies tonight if unwatered
        elif not crop["ongoing"] and (crop["max_yield_day"] + 1) // 2 <= age <= crop["max_yield_day"]:
            pri = P_TEND
        elif crop["ongoing"]:
            pri = P_TEND if t.get("fertilized_until_day", -1) >= day else P_FILLER
        else:
            pri = P_FILLER
        tasks.append(Task(pri, x, y, ["WATER"]))

    # --- Animals: the care regime is first-class (3x output multiplier) --------
    feeding_day = day < 29
    for x, y, t in me.animals():
        if feeding_day and not t.get("fed_today"):
            # Feed EVERY animal EVERY day: the CARE bonus (banked only on
            # fed+cared days, paid only on fed production days) is what turns
            # 0.5 milk/cow/day into ~1.5. Survival cases stay hard-urgent.
            pri = P_SURVIVAL if t.get("consecutive_unfed", 0) >= 1 else P_TEND
            tasks.append(Task(pri, x, y, ["FEED"], need="WHEAT"))
        if feeding_day and t.get("fed_today") and not t.get("cared_today") and day < 28:
            # Crash-pivot: caring boosts output we cannot profitably sell when
            # the product's market has collapsed — release those actions.
            product = ANIMALS.get(t.get("animal"), {}).get("product")
            price_ok = (
                not product
                or view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product]["base"]
            )
            if price_ok:
                tasks.append(Task(P_TEND, x, y, ["CARE"]))
        if t.get("fertilizer_available"):
            tasks.append(Task(P_TEND, x, y, ["COLLECT_FERTILIZER"]))
        held = t.get("yield_units", 0)
        max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
        # Harvest every production so yields never clip at max_held.
        if held >= 2 or (held > 0 and (plan.liquidation or day >= 27)):
            tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))

    # --- Growth: builds, animal placement, planting ------------------------------
    empties = sorted(
        me.empty_tiles(), key=lambda p: _dist(p[0], p[1], 4, 4) + _dist(p[0], p[1], 5, 5)
    )
    cursor = 0
    for structure, op in (("COOP", "BUILD_COOP"), ("PASTURE", "BUILD_PASTURE")):
        for _ in range(plan.structure_builds.get(structure, 0)):
            if cursor >= len(empties):
                break
            x, y = empties[cursor]
            cursor += 1
            tasks.append(Task(P_GROW, x, y, [op]))

    available = {}
    for kind in ("GOOSE", "COW", "SHEEP"):
        available[kind] = view.shed.get(kind, 0) + sum(
            view.unit_inv(i).get(kind, 0) for i, _, _ in me.units
        )
    for x, y, t in me.empty_structures():
        kinds = ("GOOSE",) if t["kind"] == "COOP" else ("COW", "SHEEP")
        for kind in kinds:
            if available.get(kind, 0) > 0:
                available[kind] -= 1
                tasks.append(Task(P_TEND, x, y, ["PLACE", kind], need=kind))
                break

    if view.hour <= PLANT_LAST_HOUR:
        for crop, n in plan.plant_quota.items():
            for _ in range(n):
                if cursor >= len(empties):
                    break
                x, y = empties[cursor]
                cursor += 1
                tasks.append(Task(P_GROW, x, y, ["PLANT", crop]))

    weed_pri = P_GROW if len(empties) - cursor < 6 else P_FILLER
    for x, y, _ in me.weeds():
        tasks.append(Task(weed_pri, x, y, ["DIG"]))

    return tasks


def _feed_pickup_needs(view, zone_of, n_zones):
    """Wheat each unit should carry for the unfed animals in ITS zone."""
    needs = {}
    for x, y, t in view.me.animals():
        if view.day >= 29 or t.get("fed_today"):
            continue
        z = zone_of.get((x, y))
        if z is not None:
            needs[z] = needs.get(z, 0) + 1
    return needs


def assign(view, plan, tasks):
    """Zone-based assignment. Returns {unit_idx: op_list}."""
    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    seeds_left = dict(view.seeds)
    shed_room = max(0, 100 - view.shed_total)
    zone_of, n_zones = zone_map(view)

    final_push = view.step >= FINAL_STEP - 10
    aggressive = view.day >= 29 or final_push

    # ---- 1. Liquidation / heavy-load shuttles (global) ----------------------
    for idx, x, y in me.units:
        inv = view.unit_inv(idx)
        total = sum(
            n
            for item, n in inv.items()
            if not (
                item in ("GOOSE", "COW", "SHEEP")
                or (not aggressive and item == "FERTILIZER" and n <= 3)
                or (not aggressive and item == "WHEAT" and n <= 5)
            )
        )
        if total == 0:
            continue
        threshold = 1 if (aggressive and view.hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if plan.liquidation and not aggressive:
            threshold = 8
        if total < threshold:
            continue
        tx, ty = _nearest_shed_tile(x, y)
        if (x, y) == (tx, ty):
            if shed_room >= total or final_push:
                actions[idx] = ["DROP"]
                shed_room = max(0, shed_room - total)
            else:
                item = max(inv, key=inv.get)
                n = min(inv[item], shed_room)
                if n <= 0:
                    continue
                actions[idx] = ["PLACE", item, n]
                shed_room -= n
        else:
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else ["PASS"]
        free.discard(idx)

    # ---- 1.5 Global need-provisioning: animals, fertilizer, and any wheat the
    # morning zone pickups missed. Without this, bought animals strand in the
    # shed forever (measured: 47 structures built, zero animals ever placed).
    wants = {}
    carried = {}
    for t in tasks:
        if t.need:
            wants[t.need] = wants.get(t.need, 0) + 1
    for idx, _, _ in me.units:
        for item, n in view.unit_inv(idx).items():
            carried[item] = carried.get(item, 0) + n
    for item, want in sorted(wants.items()):
        shortage = min(want - carried.get(item, 0), view.shed.get(item, 0))
        batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
        while shortage > 0 and free:
            idx = min(
                free,
                key=lambda i: _dist(units[i][0], units[i][1],
                                    *_nearest_shed_tile(units[i][0], units[i][1])),
            )
            x, y = units[idx]
            tx, ty = _nearest_shed_tile(x, y)
            n = min(batch, shortage)
            if (x, y) == (tx, ty):
                actions[idx] = ["PICKUP", item, n]
            else:
                step = _step_toward(x, y, tx, ty)
                if not step:
                    break
                actions[idx] = [step]
            free.discard(idx)
            shortage -= n

    # ---- 2. Morning feed provisioning: stock YOUR zone's wheat at the shed ---
    if view.hour <= 2 and view.day < 29:
        needs = _feed_pickup_needs(view, zone_of, n_zones)
        for idx in sorted(free):
            z = idx % n_zones
            need = needs.get(z, 0)
            if need <= 0:
                continue
            carried = view.unit_inv(idx).get("WHEAT", 0)
            deficit = min(need - carried, view.shed.get("WHEAT", 0))
            if deficit <= 0:
                continue
            x, y = units[idx]
            if (x, y) in SHED_ACCESS or _dist(x, y, *_nearest_shed_tile(x, y)) == 0:
                actions[idx] = ["PICKUP", "WHEAT", deficit]
                free.discard(idx)

    # ---- 3. Zone rounds: each unit serves its own zone -----------------------
    zone_tasks = {}
    global_tasks = []
    for t in tasks:
        z = zone_of.get((t.x, t.y))
        if z is None:
            global_tasks.append(t)
        else:
            zone_tasks.setdefault(z, []).append(t)
    for z in zone_tasks:
        zone_tasks[z].sort(key=lambda t: (t.pri, t.y, t.x))

    plant_requests = {}
    claimed = set()  # (x, y, op0) already assigned this turn

    def try_execute(idx, task):
        x, y = units[idx]
        key = (task.x, task.y, task.op[0])
        if key in claimed:
            return False
        if task.need and view.unit_inv(idx).get(task.need, 0) < 1:
            return False
        if task.op[0] == "PLANT":
            crop = task.op[1]
            if plant_requests.get(crop, 0) + 1 > seeds_left.get(crop, 0):
                return False
        if (x, y) == (task.x, task.y):
            if not _unlocked(view, task.x, task.y):
                return False
            actions[idx] = task.op
            if task.op[0] == "PLANT":
                plant_requests[task.op[1]] = plant_requests.get(task.op[1], 0) + 1
        else:
            step = _step_toward(x, y, task.x, task.y)
            if not step:
                return False
            actions[idx] = [step]
        claimed.add(key)
        return True

    for idx in sorted(free):
        z = idx % n_zones
        pool = zone_tasks.get(z, [])
        x, y = units[idx]
        best = None
        for task in pool:
            key = (task.x, task.y, task.op[0])
            if key in claimed:
                continue
            if task.need and view.unit_inv(idx).get(task.need, 0) < 1:
                continue
            d = _dist(x, y, task.x, task.y)
            score = (task.pri, d)
            if best is None or score < best[0]:
                best = (score, task)
        if best and try_execute(idx, best[1]):
            free.discard(idx)

    # ---- 4. Idle units help: nearest unclaimed task anywhere ----------------
    all_tasks = sorted(
        (t for pool in zone_tasks.values() for t in pool),
        key=lambda t: (t.pri, t.y, t.x),
    )
    for idx in sorted(free):
        x, y = units[idx]
        best = None
        for task in all_tasks + global_tasks:
            key = (task.x, task.y, task.op[0])
            if key in claimed:
                continue
            if task.need and view.unit_inv(idx).get(task.need, 0) < 1:
                continue
            d = _dist(x, y, task.x, task.y)
            score = (task.pri, d)
            if best is None or score < best[0]:
                best = (score, task)
        if best and try_execute(idx, best[1]):
            free.discard(idx)

    # ---- 5. Survival override: nothing urgent goes unserved -----------------
    urgent = [
        t for t in all_tasks
        if t.pri == P_SURVIVAL and (t.x, t.y, t.op[0]) not in claimed
    ]
    if urgent:
        movable = [
            idx for idx, op in actions.items()
            if op and op[0] in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")
        ] + list(free)
        for task in urgent:
            if not movable:
                break
            eligible = [
                i for i in movable
                if not task.need or view.unit_inv(i).get(task.need, 0) >= 1
            ]
            if not eligible:
                continue
            idx = min(eligible, key=lambda i: _dist(units[i][0], units[i][1], task.x, task.y))
            if try_execute(idx, task):
                movable.remove(idx)
                free.discard(idx)

    # ---- 6. Whoever is left: stage toward own zone / end-of-day shed --------
    for idx in list(free):
        x, y = units[idx]
        if sum(view.unit_inv(idx).values()) > 0 and view.hour >= TURNS_PER_DAY - 3:
            tx, ty = _nearest_shed_tile(x, y)
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else (["DROP"] if shed_room > 0 else ["PASS"])
        else:
            z = idx % n_zones
            home = [tile for tile, zz in zone_of.items() if zz == z]
            if home and (x, y) not in home:
                hx, hy = min(home, key=lambda p: _dist(x, y, p[0], p[1]))
                step = _step_toward(x, y, hx, hy)
                actions[idx] = [step] if step else ["PASS"]
            elif not _unlocked(view, x, y):
                step = _step_toward(x, y, 4, 4)
                actions[idx] = [step] if step else ["PASS"]
            else:
                actions[idx] = ["PASS"]
        free.discard(idx)

    return actions
