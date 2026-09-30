"""Executor v2: the plan is executed by ASSIGNMENT WITH COMMITMENT, not by the
zone sweep (rlv3/EXECUTOR_V2.md).

Per turn: the derived jobs come from `scheduler.build_tasks` (the engine-rule
knowledge is kept: when water adds yield, the fertiliser windows, the
harvest-before-last-water rule, the shipped flags). Each unit is matched to a
job by a min-cost assignment (Hungarian) where

    cost = PRI_STEP * priority + travel + fetch (a shed round trip when the job
           needs wheat/fertiliser the unit does not carry) - COMMIT_BONUS if the
           unit was on this job last turn  (+ SWITCH_COST if it was on another)

so a unit keeps its job unless a strictly better one appears -- commitment
won self-play (+$3,233) where optimal re-matching every turn lost (-$3,636).
Execution is one step toward the job, the op on arrival, claims so two units
never take one job, and the plan's quotas enforced at the op as before.
Logistics keep the shipped rules: drop at the shed when the pocket is full,
place animals from the pocket, the last-hour drop.

Purchases: with `plan.approved` the market controller buys the plan's seeds in
full (no 60%-of-cash ration) above `plan.cash_floor`; animals as cash lands.
"""
from __future__ import annotations

from kagent import scheduler
from kagent.scheduler import (
    DROP_THRESHOLD, EARLY_DROP_THRESHOLD, FINAL_STEP, P_FILLER, P_SURVIVAL, SWEEP_DEADLINE_HOUR,
    TURNS_PER_DAY, _dist, _nearest_shed_tile, _step_toward, _unlocked,
)

PRI_STEP = 0.5        # one priority level is worth half a step: nearest first, priority as the tie-break (the shipped sweep, measured)
COMMIT_BONUS = 1.0    # a unit's current job is this many steps "closer"
SWITCH_COST = 1.0     # and any other job this many "farther"
URGENT = 12.0         # a survival job late in the day jumps the queue
RADIUS = 99           # jobs farther than this are not offered (survival always is)
PROVISION_UNTIL = 1   # hours 0-1: the crew is at the shed, take the day's wheat and fertiliser out
ANIMAL_BONUS = 10.0   # a bought animal is placed before anything nearer: it strands in the shed otherwise (measured: 47 structures, 0 placed)
ANIMALS = ("GOOSE", "COW", "SHEEP")
FETCH_BATCH = {"WHEAT": 5, "FERTILIZER": 3, "GOOSE": 2, "COW": 2, "SHEEP": 2}   # the shipped trip sizes


def _solve(cost):
    """Min-cost assignment on a rectangular cost matrix -> list of (row, col)."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(np.asarray(cost, dtype=float))
        return list(zip(r.tolist(), c.tolist()))
    except Exception:                      # no scipy in the sandbox: greedy
        pairs, used = [], set()
        order = sorted(((cost[i][j], i, j) for i in range(len(cost)) for j in range(len(cost[0]))))
        seen_r = set()
        for _, i, j in order:
            if i in seen_r or j in used:
                continue
            pairs.append((i, j)); seen_r.add(i); used.add(j)
        return pairs


def assign2(view, plan, tasks, memory):
    """-> {unit_idx: op}. Same contract as scheduler.assign."""
    me = view.me
    day, hour = view.day, view.hour
    units = {idx: (x, y) for idx, x, y in me.units}
    commit = getattr(memory, "exec2_commit", None)
    if commit is None or getattr(memory, "exec2_day", -1) != day:
        commit = {}
        memory.exec2_commit, memory.exec2_day = commit, day
    actions, claimed = {}, set()
    seeds_left = dict(view.seeds)
    plant_requests = {}
    shed_room = max(0, 100 - view.shed_total)
    final_push = view.step >= FINAL_STEP - 10
    aggressive = day >= 29 or final_push
    shed_stock = dict(view.shed)

    # ---- 1. logistics first: full pockets go to the shed (the shipped rule)
    for idx, (x, y) in list(units.items()):
        inv = view.unit_inv(idx)
        total = sum(n for item, n in inv.items()
                    if not (item in ANIMALS or (not aggressive and item == "WHEAT" and n <= 5)
                            or (not aggressive and item == "FERTILIZER" and n <= 3)))
        threshold = 1 if (aggressive and hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)
        if plan.liquidation and not aggressive:
            threshold = 8
        if total < threshold and not (hour >= TURNS_PER_DAY - 1 and total > 0):
            continue
        tx, ty = _nearest_shed_tile(x, y)
        if (x, y) == (tx, ty):
            if shed_room >= total or final_push:
                actions[idx] = ["DROP"]; shed_room = max(0, shed_room - total)
            else:
                cargo = {k: v for k, v in inv.items() if k not in ANIMALS}
                if cargo:
                    item = max(cargo, key=cargo.get); n = min(cargo[item], shed_room)
                    if n > 0:
                        actions[idx] = ["PLACE", item, n]; shed_room -= n
        else:
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else ["PASS"]
        commit.pop(idx, None)

    free = [idx for idx in units if idx not in actions]
    late = hour >= SWEEP_DEADLINE_HOUR

    # ---- 1a. morning provisioning: units spawn at the shed, so the day's
    # wheat and fertiliser come out with them (the shipped morning rule,
    # farm-wide). Later in the day a job's fetch trip is priced in the matrix.
    carried = {}
    for idx in units:
        for item, n in view.unit_inv(idx).items():
            carried[item] = carried.get(item, 0) + n
    if hour <= PROVISION_UNTIL:
        for item in ("WHEAT", "FERTILIZER"):
            need = sum(1 for t in tasks if t.need == item and t.pri < P_FILLER) - carried.get(item, 0)
            batch = FETCH_BATCH[item]
            for idx in sorted(free, key=lambda i: _dist(*units[i], *_nearest_shed_tile(*units[i]))):
                if need <= 0 or shed_stock.get(item, 0) <= 0:
                    break
                x, y = units[idx]
                sx, sy = _nearest_shed_tile(x, y)
                if (x, y) != (sx, sy):
                    continue
                n = min(batch, need, shed_stock[item])
                actions[idx] = ["PICKUP", item, n]
                shed_stock[item] -= n
                carried[item] = carried.get(item, 0) + n
                need -= n
                free.remove(idx)
    # a fetch trip is priced only while the crew's pockets cannot cover the
    # item's jobs (the shipped zone-carry accounting): otherwise a unit without
    # the item leaves those jobs to the units that carry it
    uncovered = {item: sum(1 for t in tasks if t.need == item and t.pri < P_FILLER) - carried.get(item, 0)
                 for item in ("WHEAT", "FERTILIZER")}

    def can_plant(t):
        return t.op[0] != "PLANT" or (
            plant_requests.get(t.op[1], 0) < min(seeds_left.get(t.op[1], 0), plan.plant_quota.get(t.op[1], 0)))

    # ---- 1b. the tile underfoot, every unit, before anyone is matched: work
    # is underfoot 95% of the time and a unit that walks off a job tile to its
    # committed job pays twice (the shipped Pass 0, measured on the DIG residue)
    underfoot = {}
    for t in tasks:
        if t.pri < P_FILLER:
            underfoot.setdefault((t.x, t.y), []).append(t)
    for idx in list(free):
        x, y = units[idx]
        inv = view.unit_inv(idx)
        here = [t for t in underfoot.get((x, y), [])
                if (t.x, t.y, t.op[0]) not in claimed and can_plant(t)
                and (not t.need or inv.get(t.need, 0) >= 1) and _unlocked(view, x, y)]
        if not here:
            continue
        t = min(here, key=lambda t: t.pri)
        actions[idx] = t.op
        if t.op[0] == "PLANT":
            plant_requests[t.op[1]] = plant_requests.get(t.op[1], 0) + 1
        claimed.add((t.x, t.y, t.op[0]))
        commit.pop(idx, None)
        free.remove(idx)

    jobs = [t for t in tasks if t.pri <= P_FILLER and (t.x, t.y, t.op[0]) not in claimed]
    if not free or not jobs:
        for idx in free:
            actions[idx] = ["PASS"]
        return actions

    # ---- 2. the cost matrix: units x jobs
    keys = [(t.x, t.y, t.op[0]) for t in jobs]
    cost = []
    for idx in free:
        x, y = units[idx]
        inv = view.unit_inv(idx)
        row = []
        for t, key in zip(jobs, keys):
            d = _dist(x, y, t.x, t.y)
            if t.need and inv.get(t.need, 0) < 1:
                if shed_stock.get(t.need, 0) < 1 or uncovered.get(t.need, 1) <= 0:
                    row.append(1e6); continue       # nothing to fetch, or the crew already carries enough
                sx, sy = _nearest_shed_tile(x, y)
                d = _dist(x, y, sx, sy) + 1 + _dist(sx, sy, t.x, t.y)
            if t.pri > P_SURVIVAL and d > RADIUS:
                row.append(1e6); continue
            if not can_plant(t):
                row.append(1e6); continue
            c = PRI_STEP * t.pri + d
            if late and t.pri == P_SURVIVAL:
                c -= URGENT
            if t.need in ANIMALS:
                c -= ANIMAL_BONUS
            cur = commit.get(idx)
            if cur == key:
                c -= COMMIT_BONUS
            elif cur is not None:
                c += SWITCH_COST
            row.append(c)
        cost.append(row)

    # ---- 3. solve and execute
    for r, cidx in _solve(cost):
        idx, t = free[r], jobs[cidx]
        if cost[r][cidx] >= 1e6:
            continue
        key = keys[cidx]
        if key in claimed:
            continue
        x, y = units[idx]
        inv = view.unit_inv(idx)
        if t.need and inv.get(t.need, 0) < 1:
            sx, sy = _nearest_shed_tile(x, y)
            if (x, y) == (sx, sy):
                batch = FETCH_BATCH.get(t.need, 2)
                n = max(1, min(batch, shed_stock.get(t.need, 0)))
                actions[idx] = ["PICKUP", t.need, n]
                shed_stock[t.need] = max(0, shed_stock.get(t.need, 0) - n)
            else:
                step = _step_toward(x, y, sx, sy)
                actions[idx] = [step] if step else ["PASS"]
            commit[idx] = key; claimed.add(key)
            continue
        if not can_plant(t):
            continue
        if (x, y) == (t.x, t.y):
            if not _unlocked(view, t.x, t.y):
                continue
            actions[idx] = t.op
            if t.op[0] == "PLANT":
                plant_requests[t.op[1]] = plant_requests.get(t.op[1], 0) + 1
            commit.pop(idx, None)               # done: free to take the next job
        else:
            step = _step_toward(x, y, t.x, t.y)
            if not step:
                continue
            actions[idx] = [step]
            commit[idx] = key
        claimed.add(key)
    for idx in free:
        if idx not in actions:
            actions[idx] = ["PASS"]
            commit.pop(idx, None)
    return actions
