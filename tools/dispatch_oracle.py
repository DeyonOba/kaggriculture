"""How much of the day does DISPATCH cost us? A Team Orienteering bound.

The proposed design, stated precisely: at dawn every hand gets an ANCHOR
coordinate, walks there spending N of its 24 turns, then works the region
around that anchor with the remaining 24-N, with neighbour broadcast so two
hands do not chase the same tile.

That is not tile layout. It is the **Team Orienteering Problem** (Chao,
Golden & Wasil): m vehicles, one depot, a per-vehicle time budget, scored
nodes, each node collected at most once. Ours is the OPEN variant -- hands
never walk back, they drop inventory and respawn at the shed -- and
CAPACITATED, because FEED needs wheat carried from the depot.

TOP is NP-hard, so this does not solve it. It brackets it, which is the
only number that decides whether the design is worth building:

    ACTUAL        work-actions the executor really completes in a day
    GREEDY-TOP    a cheap anchored dispatch on the SAME task set: send each
                  unit to the highest-priority task, then repeatedly take
                  the nearest remaining task it can still afford
    CEILING       units x 24, every turn a work action, travel free

If GREEDY-TOP lands near ACTUAL, dispatch is not the constraint and the
anchor design cannot pay. If it lands near CEILING, the executor is leaving
most of the day on the floor.

    PYTHONPATH=. uv run python tools/dispatch_oracle.py --n 20
"""
import argparse
import collections
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

TPD = 24
N = 10
WORK_OPS = {"WATER", "HARVEST", "FEED", "CARE", "FERTILIZE", "DIG",
            "COLLECT_FERTILIZER", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
            "PLACE", "PICKUP", "DROP"}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def shed_tiles(n=N):
    h = n // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def tasks_at_dawn(farm, private, day):
    """The day's work, derived from the board with the VERIFIED rules.

    Score is the cost of NOT doing it today:
      6  an animal that escapes tonight, or a plant that weeds tonight
      4  a yield ready to collect, or fertilizer that expires at the refresh
      2  ordinary upkeep
      1  filler
    """
    out = []
    for y in range(len(farm["tiles"])):
        for x in range(len(farm["tiles"])):
            t = farm["tiles"][y][x]
            if not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "PLANT":
                if not t.get("watered_today"):
                    dying = t.get("consecutive_unwatered", 0) >= 1
                    out.append((x, y, 6 if dying else 2, "WATER"))
                if t.get("yield_units", 0) > 0:
                    cd = K.CROPS[t["crop"]]
                    if day - t.get("planted_day", day) >= cd["first_yield_day"]:
                        out.append((x, y, 4, "HARVEST"))
            elif k == "WEED":
                out.append((x, y, 1, "DIG"))
            elif t.get("animal"):
                if not t.get("fed_today"):
                    dying = t.get("consecutive_unfed", 0) >= 1
                    out.append((x, y, 6 if dying else 2, "FEED"))
                if not t.get("cared_today"):
                    out.append((x, y, 2, "CARE"))
                if t.get("fertilizer_available"):
                    # use-it-or-lose-it, gone at the end-of-day refresh
                    out.append((x, y, 4, "COLLECT_FERTILIZER"))
                if t.get("yield_units", 0) > 0:
                    out.append((x, y, 4, "HARVEST"))
    return out


def greedy_top(tasks, n_units, budget=TPD, start=None):
    """Open, anchored TOP heuristic. -> (tasks done, score, travel turns).

    Each unit is ANCHORED on the best remaining task it can still reach --
    the user's "most distant and highest priority work" -- then works
    outward by nearest-neighbour. Claimed tiles are removed globally, which
    is the neighbour broadcast.
    """
    if start is None:
        start = shed_tiles()[0]
    pool = list(tasks)
    done = score = travel = 0
    for _ in range(n_units):
        if not pool:
            break
        px, py = start
        left = budget
        # ANCHOR: highest priority, breaking ties toward the FAR one, since
        # near work stays cheap for whoever comes next
        pool.sort(key=lambda t: (-t[2], -(abs(t[0] - px) + abs(t[1] - py))))
        first = pool[0]
        d = abs(first[0] - px) + abs(first[1] - py)
        if d + 1 > left:
            continue
        pool.pop(0)
        left -= d + 1
        travel += d
        done += 1
        score += first[2]
        px, py = first[0], first[1]
        # then nearest-neighbour, priority-weighted, while budget allows
        while pool and left > 0:
            best, bi = None, -1
            for i, t in enumerate(pool):
                dd = abs(t[0] - px) + abs(t[1] - py)
                if dd + 1 > left:
                    continue
                v = t[2] / (dd + 1.0)
                if best is None or v > best:
                    best, bi = v, i
            if bi < 0:
                break
            t = pool.pop(bi)
            dd = abs(t[0] - px) + abs(t[1] - py)
            left -= dd + 1
            travel += dd
            done += 1
            score += t[2]
            px, py = t[0], t[1]
    return done, score, travel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--agent", default="champ-h8")
    args = ap.parse_args()

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]

    orig = K._apply_unit_action
    cur = {"work": 0, "move": 0, "seat0": None, "m0": None}

    def hooked(farm, private, idx, action, board_size, day, tpd, cap=100):
        if cur["seat0"] is None and cur["m0"] is not None \
                and abs(float(farm.get("money", -1)) - cur["m0"]) < 1e-9:
            cur["seat0"] = id(farm)
        if cur["seat0"] is not None and id(farm) == cur["seat0"]:
            op = action[0] if action else "PASS"
            if op in WORK_OPS:
                cur["work"] += 1
            elif op in MOVES:
                cur["move"] += 1
        return orig(farm, private, idx, action, board_size, day, tpd, cap)

    K._apply_unit_action = hooked
    rows = []
    try:
        for i in range(args.n):
            a0, a1 = load(args.agent), load(args.agent)
            e = make("kaggriculture", configuration={"seed": 940000 + i},
                     debug=False)
            e.reset(2)
            pend = None
            while not e.done:
                o = e.state[0].observation
                step, day = o["step"], o["step"] // TPD
                if step % TPD == 0:
                    if pend is not None:
                        pend["work"] = cur["work"]
                        pend["move"] = cur["move"]
                        rows.append(pend)
                    cur["work"] = cur["move"] = 0
                    pend = None
                if step % TPD == 12 and 3 <= day <= 26:
                    farm = o["farms"][0]
                    units = 1 + len(farm.get("hands") or [])
                    tk = tasks_at_dawn(farm, o.private, day)
                    gd, gs, gt = greedy_top(tk, units)
                    pend = dict(day=day, units=units, avail=len(tk),
                                greedy=gd, gscore=gs, gtravel=gt,
                                ceiling=units * TPD)
                m0 = float(o["farms"][0]["money"])
                m1 = float(o["farms"][1]["money"])
                cur["m0"] = m0 if abs(m0 - m1) > 1e-9 else None
                cur["seat0"] = None
                e.step([a0(o), a1(e.state[1].observation)])
    finally:
        K._apply_unit_action = orig

    R = [r for r in rows if "work" in r]
    if not R:
        print("no days captured")
        return
    A = lambda k: float(np.mean([r[k] for r in R]))
    print(f"\n{args.agent}, {args.n} games, {len(R)} agent-days "
          f"(days 3-26)\n")
    print("=" * 66)
    print("  THE DAY'S BUDGET, per agent-day")
    print("=" * 66)
    print(f"  {'units on the farm':<34}{A('units'):>10.1f}")
    print(f"  {'turn ceiling (units x 24)':<34}{A('ceiling'):>10.1f}")
    print(f"  {'tasks available at dawn':<34}{A('avail'):>10.1f}")
    print()
    print(f"  {'ACTUAL work actions':<34}{A('work'):>10.1f}"
          f"{100*A('work')/A('ceiling'):>9.1f}% of ceiling")
    print(f"  {'ACTUAL move actions':<34}{A('move'):>10.1f}"
          f"{100*A('move')/A('ceiling'):>9.1f}% of ceiling")
    print()
    print(f"  {'GREEDY-TOP tasks completed':<34}{A('greedy'):>10.1f}"
          f"{100*A('greedy')/A('ceiling'):>9.1f}% of ceiling")
    print(f"  {'GREEDY-TOP travel turns':<34}{A('gtravel'):>10.1f}")
    print()
    gap = A('greedy') - A('work')
    print("=" * 66)
    print(f"  DISPATCH GAP  {gap:+.1f} work-actions per agent-day"
          f"   ({100*gap/max(A('work'),1):+.0f}%)")
    print("=" * 66)
    if gap <= 0.5:
        print("  A cheap anchored dispatch does NOT beat the shipped executor.")
        print("  Dispatch is not the constraint; the anchor design cannot pay.")
    else:
        print(f"  A cheap anchored dispatch completes {gap:.1f} more work")
        print(f"  actions per day on the SAME task set. Over ~24 scored days")
        print(f"  that is ~{gap*24:.0f} extra actions per game.")
    print()
    print("  NOTE: greedy-TOP is a HEURISTIC, so it is a lower bound on the")
    print("  true optimum -- the real gap is at least this. It also ignores")
    print("  the wheat-reload constraint on FEED, so it is optimistic there.")
    print("  Both effects are named rather than netted.")


if __name__ == "__main__":
    main()
