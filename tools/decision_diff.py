"""Turn-by-turn diff against a 3,093-rated REACTIVE agent on identical states.

Every comparison so far has been of AGGREGATES -- their 0.71 moves per productive
action against our 1.45 -- which says a gap exists but not what produces it.
This asks the pointed question instead: standing in exactly the position
M & M & P & Q was in, what would champ-h8 have done?

No re-simulation is needed and none is done. The replay records the full
observation at every step, so our agent is called on THEIR recorded states and
its choice is compared with theirs. Our agent's own memory evolves along their
trajectory, which is the right counterfactual: "what would we do from here".

Alignment matters and is easy to get wrong (`tools/build_sessions.py` documents
it): `steps[t]["action"]` is the action that PRODUCED state t, i.e. the decision
taken while observing state t-1. So the action chosen while observing state t is
`steps[t+1]["action"]`.

The headline cell is THEY WORK / WE MOVE -- the moments where they are producing
and we would still be walking. That is the 1.45-vs-0.71 gap, localised to
specific turns and days instead of averaged away.

    PYTHONPATH=. uv run python tools/decision_diff.py --n 8
    PYTHONPATH=. uv run python tools/decision_diff.py --n 8 --agent match-all
"""

import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TEAM = "M & M & P & Q"
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def load_agent(name):
    """Exec a frozen bundle and hand back its `agent` callable."""
    path = os.path.join(REPO, "arena", "agents", name, "main.py")
    src = open(path).read()
    g = {"__name__": f"bundle_{name}"}
    exec(compile(src, path, "exec"), g)
    return g["agent"]


def op_of(a):
    return a[0] if isinstance(a, (list, tuple)) and a else "PASS"


def klass(op):
    return "MOVE" if op in MOVES else ("PASS" if op == "PASS" else "WORK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--dir", default="replays/mmpq")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    agent = load_agent(args.agent)
    paths = sorted(glob.glob(os.path.join(REPO, args.dir, "*.json")))[:args.n]

    conf = collections.Counter()          # (their class, our class)
    op_conf = collections.Counter()       # (their op, our op) for unit 0
    by_day = collections.defaultdict(lambda: collections.Counter())
    exact = total = 0
    failures = 0
    games = 0

    for p in paths:
        rep = json.load(open(p))
        names = (rep.get("info") or {}).get("TeamNames") or []
        if TEAM not in names:
            continue
        seat = names.index(TEAM)
        steps = rep["steps"]
        games += 1
        for t in range(len(steps) - 1):
            obs = steps[t][seat].get("observation")
            theirs = (steps[t + 1][seat].get("action") or {})
            if not obs or not isinstance(theirs, dict):
                continue
            try:
                ours = agent(obs, rep.get("configuration"))
            except Exception:
                failures += 1
                continue
            if not isinstance(ours, dict):
                failures += 1
                continue
            day = obs.get("day", 0)
            t_units = [theirs.get("farmer")] + list(theirs.get("hands") or [])
            o_units = [ours.get("farmer")] + list(ours.get("hands") or [])
            for i in range(min(len(t_units), len(o_units))):
                to, oo = op_of(t_units[i]), op_of(o_units[i])
                tk, ok = klass(to), klass(oo)
                conf[(tk, ok)] += 1
                by_day[day][(tk, ok)] += 1
                total += 1
                exact += t_units[i] == o_units[i]
                if i == 0:
                    op_conf[(to, oo)] += 1

    if not total:
        sys.exit("no comparable turns -- check the replay directory")

    print(f"{args.agent} vs {TEAM}, {games} replays, {total:,} unit-decisions"
          f"{f' ({failures} agent errors)' if failures else ''}\n")
    print(f"exact action match: {exact/total*100:.1f}%\n")

    kinds = ("WORK", "MOVE", "PASS")
    print(f"{'':>12}" + "".join(f"{'we ' + k:>12}" for k in kinds))
    for tk in kinds:
        row = f"{'they ' + tk:>12}"
        for ok in kinds:
            n = conf[(tk, ok)]
            row += f"{n:>7,} {n/total*100:>4.1f}%"
        print(row)

    tw_om = conf[("WORK", "MOVE")]
    tw = sum(conf[("WORK", k)] for k in kinds)
    mw = sum(conf[("MOVE", k)] for k in kinds)
    print(f"\nTHEY WORK / WE MOVE: {tw_om:,} of {tw:,} of their working turns "
          f"({tw_om/max(1,tw)*100:.1f}%)")
    print(f"THEY MOVE / WE WORK: {conf[('MOVE','WORK')]:,} of {mw:,} "
          f"({conf[('MOVE','WORK')]/max(1,mw)*100:.1f}%)")
    print(f"they WORK on {tw/total*100:.1f}% of unit-turns, we would on "
          f"{sum(conf[(k,'WORK')] for k in kinds)/total*100:.1f}%")

    print(f"\nby season third -- share of THEIR working turns where we would move:")
    for lo, hi in ((0, 9), (10, 19), (20, 29)):
        a = sum(by_day[d][("WORK", "MOVE")] for d in range(lo, hi + 1))
        b = sum(by_day[d][("WORK", k)] for d in range(lo, hi + 1) for k in kinds)
        print(f"   days {lo:>2}-{hi:<2} {a:>7,} / {b:>7,}   {a/max(1,b)*100:>5.1f}%")

    print(f"\nwhat they do when we would MOVE (farmer only, top 10):")
    rows = [(n, to, oo) for (to, oo), n in op_conf.items() if klass(oo) == "MOVE"]
    for n, to, oo in sorted(rows, reverse=True)[:10]:
        print(f"   they {to:<20} we {oo:<8} {n:>7,}")


if __name__ == "__main__":
    main()
