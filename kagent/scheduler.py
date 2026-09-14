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

from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, SHED_ACCESS, TURNS_PER_DAY
from kagent.planner import FINAL_STEP

# Task priorities (lower = more urgent).
P_SURVIVAL = 0
P_HARVEST = 1
P_ANIMAL = 1.5   # feed/care: a fed+cared production is 3-4x a bare one
P_TEND = 2
P_GROW = 3
P_FILLER = 5

PLANT_LAST_HOUR = 21  # leave >= 2 turns to water the fresh plant
DROP_THRESHOLD = 10   # mid-game inventory size that triggers a shed trip
EARLY_DROP_THRESHOLD = 2  # days 0-9: every fertilizer unit is the day's cash
MIN_ZONE_TILES = 5
# Executor variant: rank a distant task by how far FORWARD it lies along the
# zone's serpentine path instead of by Manhattan distance, so a hand sweeps its
# zone in one direction. SWEEP_BACKSTEP is what stepping backwards is allowed to
# cost, so a task one tile behind is not walked around the whole tour.
SWEEP_ORDER = False
SWEEP_BACKSTEP = 2
SWEEP_DEADLINE_HOUR = 16  # from here survival tasks (dies tonight) jump the sweep
HELP_RADIUS = 5           # cross-zone help only within this many steps


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
    # Even split (sizes differ by at most one): a ceil'd chunk left the last
    # zone EMPTY (75 tiles / 12 hands -> 11 zones), so one hand roamed the
    # whole farm every day.
    zone_of = {tile: i * n_zones // len(path) for i, tile in enumerate(path)}
    # Position of each tile along the serpentine, for the sweep executor: two
    # consecutive indices are always physically adjacent, so a hand that always
    # takes the next task FORWARD along the path walks one tile between actions.
    order_of = {tile: i for i, tile in enumerate(path)}
    return zone_of, n_zones, order_of


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

        # Fertilizer is cash (~$100/unit at base): only spend it on a crop
        # when the market has glutted it below what the extra yield returns.
        fert_price = view.market_prices.get("FERTILIZER", 100)
        if t.get("fertilized_until_day", -1) < day and not plan.liquidation:
            # Strawberries: a fertilized production yields 2 instead of 1
            # (~$165 each) for 3 days of cover — champions run ~270 units off
            # 33 tiles this way versus ~130 unfertilized.
            want_fert = (
                (t["crop"] == "STRAWBERRY" and 9 <= age <= 15 and fert_price < 150)
                or (t["crop"] == "TOMATO" and age in (8, 11) and fert_price < 80)
                or (t["crop"] == "WHEAT" and age == 2 and fert_price < 40)
                or (t["crop"] == "CARROT" and age == 2 and fert_price < 50)
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
        # Engine rules: water adds yield only inside a one-shot crop's growth
        # window, or on a fertilized ongoing crop's production night; any
        # other day it is pure survival, and a plant survives one dry day.
        # Watering everything daily is what starved planting of labor.
        if t.get("consecutive_unwatered", 0) >= 1:
            pri = P_SURVIVAL  # dies tonight if unwatered
        elif not crop["ongoing"]:
            in_window = (crop["max_yield_day"] + 1) // 2 <= age <= crop["max_yield_day"]
            pri = P_TEND if in_window else P_FILLER
        else:
            since_first = day + 1 - t["planted_day"] - crop["first_yield_day"]
            produces_tonight = since_first >= 0 and since_first % crop["interval"] == 0
            fertilized = t.get("fertilized_until_day", -1) >= day
            pri = P_TEND if (produces_tonight and fertilized) else P_FILLER
        tasks.append(Task(pri, x, y, ["WATER"]))

    # --- Animals: the care regime is first-class (3x output multiplier) --------
    feeding_day = day < 29
    for x, y, t in me.animals():
        if feeding_day and not t.get("fed_today"):
            # Feed EVERY animal EVERY day: the CARE bonus (banked only on
            # fed+cared days, paid only on fed production days) is what turns
            # 0.5 milk/cow/day into ~1.5. Survival cases stay hard-urgent.
            pri = P_SURVIVAL if t.get("consecutive_unfed", 0) >= 1 else P_ANIMAL
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
                tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))
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
        # One PLANT task per empty tile, crops interleaved by remaining quota,
        # so every zone plants its own ground instead of all hands converging
        # on the few center-most empties. The daily quota is enforced at
        # execution time (assign.try_execute).
        wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
        if wanted:
            for i, (x, y) in enumerate(empties[cursor:]):
                tasks.append(Task(P_GROW, x, y, ["PLANT", wanted[i % len(wanted)]]))

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


MOVE_STATS = {}  # debug: moves per assignment step (filled only when enabled)
ASSIGN_LABEL = {}  # debug: unit idx -> step that assigned its action this turn
MOVE_STATS_ENABLED = False


def _count_moves(actions, before, label):
    if not MOVE_STATS_ENABLED:
        return
    for idx, op in actions.items():
        if idx in before or not op:
            continue
        ASSIGN_LABEL[idx] = label
        key = label if op[0] in ("NORTH", "SOUTH", "EAST", "WEST") else "act:" + label
        MOVE_STATS[key] = MOVE_STATS.get(key, 0) + 1


def assign(view, plan, tasks):
    """Zone-based assignment. Returns {unit_idx: op_list}."""
    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    seeds_left = dict(view.seeds)
    shed_room = max(0, 100 - view.shed_total)
    zone_of, n_zones, order_of = zone_map(view)

    final_push = view.step >= FINAL_STEP - 10
    aggressive = view.day >= 29 or final_push

    # ---- 1. Liquidation / heavy-load shuttles (global) ----------------------
    # Fertilizer in a pocket is only working stock while a FERTILIZE task
    # wants it; otherwise it is unsold cash and goes to the shed.
    fert_wanted = any(t.need == "FERTILIZER" for t in tasks)
    for idx, x, y in me.units:
        inv = view.unit_inv(idx)
        total = sum(
            n
            for item, n in inv.items()
            if not (
                item in ("GOOSE", "COW", "SHEEP")
                or (not aggressive and fert_wanted and item == "FERTILIZER" and n <= 3)
                or (not aggressive and item == "WHEAT" and n <= 5)
            )
        )
        if total == 0:
            continue
        threshold = 1 if (aggressive and view.hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if view.day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)
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

    _count_moves(actions, set(), "shuttle")
    _seen = set(actions)

    # ---- 1.5 Global need-provisioning: animals, fertilizer, and any wheat the
    # morning zone pickups missed. Without this, bought animals strand in the
    # shed forever (measured: 47 structures built, zero animals ever placed).
    # Only ANIMALS are provisioned globally here (rare, high-value, any unit
    # may carry one to its pasture). Wheat and fertilizer are fetched by the
    # zone that needs them (step 2) so hands never turn into farm-wide
    # couriers — measured: 150-200 zone moves/day for 40-60 actions.
    wants = {}
    carried = {}
    for t in tasks:
        if t.need in ("GOOSE", "COW", "SHEEP"):
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

    _count_moves(actions, _seen, "provision")
    _seen = set(actions)

    # ---- 2. Zone provisioning: each zone's hand fetches ITS zone's wheat and
    # fertilizer from the shed (any hour), so deliveries stay intra-zone.
    zone_needs = {}
    for t in tasks:
        if t.need in ("WHEAT", "FERTILIZER"):
            z = zone_of.get((t.x, t.y))
            if z is not None:
                zone_needs.setdefault(z, {}).setdefault(t.need, 0)
                zone_needs[z][t.need] += 1
    zone_carry = {}
    for idx, _, _ in me.units:
        z = idx % n_zones
        for item in ("WHEAT", "FERTILIZER"):
            zone_carry.setdefault(z, {}).setdefault(item, 0)
            zone_carry[z][item] += view.unit_inv(idx).get(item, 0)
    shed_left = {item: view.shed.get(item, 0) for item in ("WHEAT", "FERTILIZER")}
    for idx in sorted(free):
        z = idx % n_zones
        for item in ("WHEAT", "FERTILIZER"):
            need = zone_needs.get(z, {}).get(item, 0) - zone_carry.get(z, {}).get(item, 0)
            n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)
            if n <= 0:
                continue
            x, y = units[idx]
            tx, ty = _nearest_shed_tile(x, y)
            if (x, y) == (tx, ty):
                actions[idx] = ["PICKUP", item, n]
            else:
                step = _step_toward(x, y, tx, ty)
                if not step:
                    continue
                actions[idx] = [step]
            shed_left[item] -= n
            zone_carry[z][item] = zone_carry[z].get(item, 0) + n
            free.discard(idx)
            break

    _count_moves(actions, _seen, "morning")
    _seen = set(actions)

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
            allowed = min(seeds_left.get(crop, 0), plan.plant_quota.get(crop, 0))
            if plant_requests.get(crop, 0) + 1 > allowed:
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

    # A unit carrying an ANIMAL places it FIRST, anywhere on the farm. Without
    # this the provisioning carrier fell back into its own zone's rounds and
    # hauled a cow around for days while the pasture sat empty (traced on
    # day 0-3). Wheat/fertilizer deliveries stay inside the zone round.
    need_tasks = [t for t in tasks if t.need in ("GOOSE", "COW", "SHEEP")]

    # SWEEP scoring: nearest task first, priority only as a tie-break. The
    # old (priority, distance) order walked a hand four tiles to a harvest
    # and four back to a water next door: 1.7 moves per action against the
    # leaders' ~1.0 (measured on our five worst worlds: 71-105 productive
    # actions a day vs their 105-159 with the same crew). Late in the day
    # survival tasks (dies tonight) jump the queue so the sweep never
    # strands them.
    late = view.hour >= SWEEP_DEADLINE_HOUR

    def pick(idx, pool, lo, hi, max_dist=None):
        """Nearest executable task in pool with lo <= pri < hi."""
        x, y = units[idx]
        inv = view.unit_inv(idx)
        best = None
        for task in pool:
            if not (lo <= task.pri < hi):
                continue
            key = (task.x, task.y, task.op[0])
            if key in claimed:
                continue
            if task.need and inv.get(task.need, 0) < 1:
                continue
            d = _dist(x, y, task.x, task.y)
            if max_dist is not None and d > max_dist:
                continue
            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            if SWEEP_ORDER and d > 0:
                # Forward-only serpentine sweep: among tasks not under our feet,
                # take the next one ALONG the path rather than the nearest in
                # Manhattan terms. Nearest-task is a greedy nearest-neighbour
                # tour, which zig-zags; sweeping the zone in one direction does
                # every tile's work as it passes. Anything at d == 0 still wins
                # outright, so same-tile chains (harvest -> plant -> water) are
                # never broken.
                ti = order_of.get((task.x, task.y))
                hi_ = order_of.get((x, y))
                if ti is not None and hi_ is not None:
                    fwd = (ti - hi_) % max(1, len(order_of))
                    score = (urgent, min(fwd, d + SWEEP_BACKSTEP), task.pri)
                else:
                    score = (urgent, d, task.pri)
            else:
                score = (urgent, d, task.pri)
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None

    all_tasks = sorted(
        (t for pool in zone_tasks.values() for t in pool),
        key=lambda t: (t.pri, t.y, t.x),
    )

    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "zone")
    _seen = set(actions)

    # Pass 2 — real work anywhere: animal-heavy zones (pastures cluster near
    # the shed) get help before anyone waters a plant that is not due.
    for idx in sorted(free):
        task = pick(idx, all_tasks + global_tasks, P_SURVIVAL, P_FILLER, max_dist=HELP_RADIUS)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "idlehelp")
    _seen = set(actions)

    # Pass 3 — filler work: own zone, then anywhere.
    for idx in sorted(free):
        task = pick(idx, zone_tasks.get(idx % n_zones, []), P_FILLER, 99)
        if task is None:
            task = pick(idx, all_tasks + global_tasks, P_FILLER, 99)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "filler")
    _seen = set(actions)

    # ---- 5. Survival override: nothing urgent goes unserved -----------------
    urgent = [
        t for t in all_tasks
        if t.pri == P_SURVIVAL and (t.x, t.y, t.op[0]) not in claimed
    ]
    if urgent:
        # Only genuinely idle units are pulled: re-targeting hands that are
        # already walking to their zone's work made every unfed animal or dry
        # plant a farm-wide emergency and hands paced between quadrants
        # (traced: 200 moves for 60 actions a day). Zone rounds already
        # serve P_SURVIVAL first inside each zone.
        movable = [idx for idx, op in actions.items() if op and op[0] == "PASS"] + list(free)
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

    _count_moves(actions, _seen, "survival")
    _seen = set(actions)

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
    _count_moves(actions, _seen, "stage")

    return actions
