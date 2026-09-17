"""Diff the TASK SET, not the actions: what work do they do that we never propose?

The decision diff established that the gap is task VALUE, not task ORDER -- on
94.9% of the disputed turns the work is already underfoot, twelve routing
mechanisms gated null, and four separate throughput fixes all LOST money because
the extra work our executor finds is filler. So the question moves upstream:
standing in their position, what does our planner even put on the board?

For every unit-turn where M & M & P & Q performs real work, this asks:

  * COVERED   -- our `build_tasks` proposes that exact (tile, op). We had the
                 idea and the executor simply did not get to it.
  * WRONG OP  -- we propose something at that tile, but a different operation.
  * ABSENT    -- we propose nothing at that tile at all. Our planner cannot see
                 the work, so no executor change could ever reach it.

and reports the priority band we assign to the covered ones, because a task we
rank P_FILLER is one our passes reach last.

The reverse direction matters just as much: tasks we propose that they never
touch all season are work our planner invents and theirs does not.

Our plan is computed from THEIR state with `make_champion_plan`, which is the
right counterfactual -- what would we want to do from here.

    PYTHONPATH=. uv run python tools/taskset_diff.py --n 6
"""

import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent.gamestate import View            # noqa: E402
from kagent.planner import make_champion_plan  # noqa: E402
from kagent import scheduler                 # noqa: E402

TEAM = "M & M & P & Q"
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
BANDS = [(0, "P_SURVIVAL"), (1, "P_HARVEST"), (1.5, "P_ANIMAL"),
         (2, "P_TEND"), (3, "P_GROW"), (5, "P_FILLER")]


def band_name(pri):
    best = "?"
    for v, n in BANDS:
        if pri >= v:
            best = n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="replays/mmpq")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--every", type=int, default=2, help="sample every Nth turn")
    args = ap.parse_args()

    cover = collections.Counter()
    by_op = collections.defaultdict(collections.Counter)   # their op -> outcome
    band_of_covered = collections.Counter()
    ours_proposed = collections.Counter()                  # our op -> count proposed
    theirs_done = collections.Counter()                    # their op -> count done
    absent_ops = collections.Counter()
    turns = 0

    for p in sorted(glob.glob(os.path.join(REPO, args.dir, "*.json")))[:args.n]:
        rep = json.load(open(p))
        names = (rep.get("info") or {}).get("TeamNames") or []
        if TEAM not in names:
            continue
        seat = names.index(TEAM)
        steps = rep["steps"]
        for t in range(0, len(steps) - 1, args.every):
            obs = steps[t][seat].get("observation")
            act = steps[t + 1][seat].get("action") or {}
            if not obs:
                continue
            try:
                v = View(obs)
                plan = make_champion_plan(v, None)
                tasks = scheduler.build_tasks(v, plan)
            except Exception:
                continue
            turns += 1
            at = collections.defaultdict(list)
            for k in tasks:
                at[(k.x, k.y)].append(k)
                ours_proposed[k.op[0]] += 1
            units = list(v.me.units)
            acts = [act.get("farmer")] + list(act.get("hands") or [])
            for i, a in enumerate(acts):
                if i >= len(units):
                    break
                op = a[0] if isinstance(a, list) and a else "PASS"
                if op in MOVES or op == "PASS":
                    continue
                theirs_done[op] += 1
                _, ux, uy = units[i]
                here = at.get((ux, uy), [])
                same = [k for k in here if k.op[0] == op]
                if same:
                    cover["covered"] += 1
                    by_op[op]["covered"] += 1
                    band_of_covered[band_name(min(k.pri for k in same))] += 1
                elif here:
                    cover["wrong_op"] += 1
                    by_op[op]["wrong_op"] += 1
                else:
                    cover["absent"] += 1
                    by_op[op]["absent"] += 1
                    absent_ops[op] += 1

    n = sum(cover.values()) or 1
    print(f"{turns:,} sampled turns, {n:,} of their work-actions\n")
    print(f"  COVERED   we propose that exact (tile, op)   {cover['covered']/n*100:>5.1f}%")
    print(f"  WRONG OP  we propose something else there    {cover['wrong_op']/n*100:>5.1f}%")
    print(f"  ABSENT    we propose nothing at that tile    {cover['absent']/n*100:>5.1f}%")

    print(f"\n  priority we assign to the COVERED work:")
    cv = sum(band_of_covered.values()) or 1
    for _, nm in BANDS:
        if band_of_covered[nm]:
            print(f"    {nm:<12} {band_of_covered[nm]/cv*100:>5.1f}%")

    print(f"\n  by their operation:")
    print(f"    {'op':<20} {'n':>7} {'covered':>8} {'wrong op':>9} {'absent':>8}")
    for op, c in sorted(by_op.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        print(f"    {op:<20} {tot:>7,} {c['covered']/tot*100:>7.1f}% "
              f"{c['wrong_op']/tot*100:>8.1f}% {c['absent']/tot*100:>7.1f}%")

    print(f"\n  what WE propose vs what THEY do (per sampled turn):")
    print(f"    {'op':<20} {'we propose':>11} {'they do':>9}")
    for op in sorted(set(ours_proposed) | set(theirs_done),
                     key=lambda o: -(ours_proposed[o] + theirs_done[o]))[:12]:
        print(f"    {op:<20} {ours_proposed[op]/max(1,turns):>11.1f} "
              f"{theirs_done[op]/max(1,turns):>9.2f}")


if __name__ == "__main__":
    main()
