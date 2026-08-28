"""Turn-level task generation and greedy unit assignment.

Recomputed from scratch every turn (no cross-turn path state): the midnight
respawn of all units at the shed needs no special-casing.
"""

from kagent.constants import ANIMALS, CROPS, SHED_ACCESS, TURNS_PER_DAY
from kagent.planner import FINAL_STEP

# Task priorities (lower = more urgent).
P_SURVIVAL = 0
P_HARVEST = 1
P_TEND = 2
P_GROW = 3
P_FILLER = 5

PLANT_LAST_HOUR = 21  # leave >= 2 turns to water the fresh plant
DROP_THRESHOLD = 10   # mid-game inventory size that triggers a shed trip


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

        # Harvest.
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
                continue  # will be gone; no watering needed

        # Fertilize (before watering the same tile): tomato production days
        # and the start of wheat/carrot bonus windows.
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
        # Suppress the bonus water until the fertilizer lands (early in the day).
        if (x, y) in fert_targets and t.get("consecutive_unwatered", 0) == 0 and view.hour < 18:
            continue
        if t.get("consecutive_unwatered", 0) >= 1:
            pri = P_SURVIVAL  # dies tonight if unwatered
        elif not crop["ongoing"] and (crop["max_yield_day"] + 1) // 2 <= age <= crop["max_yield_day"]:
            pri = P_TEND      # bonus-window water
        elif crop["ongoing"]:
            pri = P_TEND if t.get("fertilized_until_day", -1) >= day else P_FILLER
        else:
            pri = P_FILLER    # keeps consecutive_unwatered at 0; cheap insurance
        tasks.append(Task(pri, x, y, ["WATER"]))

    # --- Animals ---------------------------------------------------------------
    # Day 29 has no end-of-day refresh: feeding then is pure waste.
    feeding_day = day < 29
    for x, y, t in me.animals():
        ruminant = t.get("animal") in ("COW", "SHEEP")
        if feeding_day and not t.get("fed_today"):
            # cu >= 1 means the animal escapes tonight if unfed: hard deadline.
            # Cows/sheep are fed daily so CARE can bank (+1 product each
            # fed+cared day, paid on the next production: ~$125+/action).
            if t.get("consecutive_unfed", 0) >= 1:
                pri = P_SURVIVAL
            else:
                pri = P_TEND if ruminant else P_FILLER
            tasks.append(Task(pri, x, y, ["FEED"], need="WHEAT"))
        if feeding_day and t.get("fed_today") and not t.get("cared_today") and day < 28:
            tasks.append(Task(P_TEND if ruminant else P_FILLER, x, y, ["CARE"]))
        if t.get("fertilizer_available"):
            tasks.append(Task(P_TEND, x, y, ["COLLECT_FERTILIZER"]))
        held = t.get("yield_units", 0)
        max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
        if held >= max_held - 1 or (held > 0 and plan.liquidation):
            tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))

    # --- Growth: builds, goose placement, planting ------------------------------
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

    # --- Weeds: keep the farm clear (idle hands are cheap) ------------------------
    weed_pri = P_GROW if len(empties) - cursor < 6 else P_FILLER
    for x, y, _ in me.weeds():
        tasks.append(Task(weed_pri, x, y, ["DIG"]))

    return tasks


def _provision_tasks(view, plan, tasks):
    """PICKUP tasks that stock units for need-tasks (feed wheat, fertilizer,
    goose delivery), placed at the shed."""
    carried = {}
    for idx, _, _ in view.me.units:
        for item, n in view.unit_inv(idx).items():
            carried[item] = carried.get(item, 0) + n

    wants = {}
    for t in tasks:
        if t.need:
            wants[t.need] = wants.get(t.need, 0) + 1

    out = []
    for item, want in wants.items():
        shortage = want - carried.get(item, 0)
        available = max(0, view.shed.get(item, 0))
        take = min(shortage, available)
        if take <= 0:
            continue
        pri = P_SURVIVAL if item == "WHEAT" else P_TEND
        # Split large pickups across units: one carrier cannot visit 20 tiles
        # in a day, and a starved animal escapes.
        batch = 5 if item == "WHEAT" else 3
        while take > 0:
            n = min(batch, take)
            out.append(Task(pri, -1, -1, ["PICKUP", item, n], need=None))
            take -= n
    return out


def _assign_batch(view, plan, ordered, units, free, actions, plant_requests, seeds_left=None):
    """Greedy nearest-unit assignment of a task batch (mutates free/actions).

    Returns the tasks that could not be assigned this turn (no free/eligible
    unit, or unreachable) so idle units can pre-position toward them.
    """
    seeds_left = seeds_left if seeds_left is not None else {}
    leftover = []
    for task in ordered:
        if not free:
            leftover.append(task)
            continue

        if task.op[0] == "PICKUP":
            idx = min(
                free,
                key=lambda i: _dist(
                    units[i][0], units[i][1], *_nearest_shed_tile(units[i][0], units[i][1])
                ),
            )
            x, y = units[idx]
            tx, ty = _nearest_shed_tile(x, y)
            if (x, y) == (tx, ty):
                actions[idx] = task.op
            else:
                step = _step_toward(x, y, tx, ty)
                actions[idx] = [step] if step else ["PASS"]
            free.discard(idx)
            continue

        if task.op[0] == "PLANT":
            crop = task.op[1]
            if plant_requests.get(crop, 0) + 1 > seeds_left.get(crop, 0):
                continue  # atomic-PLANT rule: never over-request

        candidates = [
            i for i in free if not task.need or view.unit_inv(i).get(task.need, 0) >= 1
        ]
        if not candidates:
            leftover.append(task)
            continue
        idx = min(candidates, key=lambda i: _dist(units[i][0], units[i][1], task.x, task.y))
        x, y = units[idx]
        if (x, y) == (task.x, task.y):
            if not _unlocked(view, task.x, task.y):
                continue
            actions[idx] = task.op
            if task.op[0] == "PLANT":
                plant_requests[task.op[1]] = plant_requests.get(task.op[1], 0) + 1
        else:
            d = _dist(x, y, task.x, task.y)
            if d > view.hours_left_today + 4:
                leftover.append(task)
                continue  # unreachable today; idle units can stage toward it
            step = _step_toward(x, y, task.x, task.y)
            actions[idx] = [step] if step else ["PASS"]
        free.discard(idx)
    return leftover


def assign(view, plan, tasks):
    """Greedy nearest-unit assignment. Returns {unit_idx: op_list}."""
    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    seeds_left = dict(view.seeds)
    shed_room = max(0, 100 - view.shed_total)

    # Provisioning first (positions resolve to nearest shed tile per unit).
    pickups = _provision_tasks(view, plan, tasks)

    # Liquidation shuttles: get inventories into the shed so market can sell.
    drop_tasks = []
    final_push = view.step >= FINAL_STEP - 10
    aggressive = view.day >= 29 or final_push
    for idx, x, y in me.units:
        inv = view.unit_inv(idx)
        # Working stock is not cargo: geese are being delivered to coops,
        # fertilizer is bound for crops, small wheat loads are feed runs.
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
        if total >= threshold:
            drop_tasks.append((idx, total))

    ordered = sorted(pickups + tasks, key=lambda t: (t.pri, t.y, t.x))
    survival = [t for t in ordered if t.pri == P_SURVIVAL]
    ordered = [t for t in ordered if t.pri != P_SURVIVAL]

    # Survival tasks (unwatered plants, unfed animals) claim units before
    # anything else, including liquidation shuttles.
    leftover = _assign_batch(view, plan, survival, units, free, actions, {}, seeds_left)

    # Forced drops next for units that must shuttle.
    for idx, total in drop_tasks:
        if idx not in free:
            continue
        x, y = units[idx]
        tx, ty = _nearest_shed_tile(x, y)
        if (x, y) == (tx, ty):
            if shed_room >= total or final_push:
                actions[idx] = ["DROP"]
                shed_room = max(0, shed_room - total)
            else:
                # Partial drop of the single biggest stack.
                inv = view.unit_inv(idx)
                item = max(inv, key=inv.get)
                n = min(inv[item], shed_room)
                if n > 0:
                    actions[idx] = ["PLACE", item, n]
                    shed_room -= n
                else:
                    continue  # no room; let the unit do normal work
        else:
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else ["PASS"]
        free.discard(idx)

    plant_requests = {}
    leftover += _assign_batch(view, plan, ordered, units, free, actions, plant_requests, seeds_left)

    # Idle units pre-position instead of passing: end-of-day carriers head for
    # the shed; everyone else stages toward the nearest unassigned task, and
    # units stranded on locked land walk back to the farm.
    for idx in list(free):
        x, y = units[idx]
        if sum(view.unit_inv(idx).values()) > 0 and view.hour >= TURNS_PER_DAY - 3:
            tx, ty = _nearest_shed_tile(x, y)
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else (["DROP"] if shed_room > 0 else ["PASS"])
            free.discard(idx)
            continue
        target = None
        best = 10**9
        # Hands vanish at midnight and the farmer teleports home, so staging
        # toward future work only pays while there is day left to use it.
        if view.hours_left_today > 6:
            for t in leftover:
                # Only stage toward tasks this unit could actually perform on
                # arrival; a need-task (feed/fertilize/place) without the item
                # in hand would strand the unit next to work it can't do.
                if t.x < 0 or (t.need and view.unit_inv(idx).get(t.need, 0) < 1):
                    continue
                d = _dist(x, y, t.x, t.y)
                if d < best:
                    best, target = d, t
        if target is not None:
            step = _step_toward(x, y, target.x, target.y)
            actions[idx] = [step] if step else ["PASS"]
        elif not _unlocked(view, x, y):
            step = _step_toward(x, y, 4, 4)
            actions[idx] = [step] if step else ["PASS"]
        else:
            actions[idx] = ["PASS"]
        free.discard(idx)

    return actions
