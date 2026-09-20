"""Rollout policy improvement over any of the planner's per-day schedules.

`tools/rollout_hires.py` proved the pattern on hiring: the SAME lever is refuted
as a constant (+2 hands -$5,229, +4 -$19,344, monotone) and pays as a per-day
SCHEDULE (+$3,287 +/- 1,528 over 192 fresh-world games). The knob space was
never the plan space.

Every other crop target in the planner is still a constant. This generalises
the rollout to them by rewriting the FROZEN bundle's embedded planner source --
`_SOURCES` is `ast.literal_eval`-able, so the module source can be extracted,
edited as plain text and re-emitted with `repr`, which avoids hand-escaping the
bundle's string literals.

Why the frozen bundle and not `kagent/`: the dev tree can no longer reproduce
the champion ($25,417 against the bundle's $59,656 on the same world, and
injecting the shipped BANDIT_RULES does not close it), so monkeypatching
`kagent/` optimises a $34k-weaker agent.

Schedules:
  hires   day 0-29  hands hired that day          (already rolled out)
  straw   day 5-13  standing strawberry tiles
  wheat   day 2-25  standing wheat tiles

    PYTHONPATH=. uv run python tools/rollout_plan.py --schedule straw --worlds 3
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OPPS = ["arena/agents/pub-ahmed-v38/main.py",
        "arena/agents/pub-boatlee-v16/main.py"]
BASE_AGENT = "arena/agents/champ-h8/main.py"
AGENT_DIR = "/tmp/rollout_plan_agents"
DAYS = 30

# name -> (base schedule, planner-source edit). Each edit replaces the scalar
# decision with a lookup into a per-day array injected next to CHAMP_HIRES.
SCHEDULES = {
    # --- OPENING schedules -------------------------------------------------
    # Relative ANIMAL count at day 3 is the strongest early signal in the
    # 21,054-game corpus (r = +0.237 against the final margin) -- stronger
    # than crops (+0.005) and far stronger than money (+0.008). It is also
    # taken with ZERO shop information: the first shop opens on day 3.
    # The herd targets were the one opening lever with no schedule.
    "cow": {
        "base": [2, 2, 2, 3, 4, 5, 6, 7, 8] + [8] * 10 + [0] * 11,
        "days": (0, 6),
        "array": "COW_SCHED",
        "find": '    cow = 2 if day < 3 else min(knobs["cows"], 2 + (day - 2))',
        "repl": '    cow = (COW_SCHED[day] if day < len(COW_SCHED) else '
                '(2 if day < 3 else min(knobs["cows"], 2 + (day - 2))))',
    },
    "sheep": {
        "base": [2] * 7 + [3, 4, 5, 6, 7, 8] + [8] * 8 + [0] * 9,
        "days": (0, 6),
        "array": "SHEEP_SCHED",
        "find": '    sheep = 2 if day < 7 else min(knobs["sheep"], '
                '2 + (day - 6))',
        "repl": '    sheep = (SHEEP_SCHED[day] if day < len(SHEEP_SCHED) '
                'else (2 if day < 7 else min(knobs["sheep"], 2 + (day - 6))))',
    },

    "hires": {
        "base": [5, 4, 4, 5, 4, 5, 7, 9, 9, 10] + [11] * 16 + [10, 10, 10, 8],
        "days": (0, 29),
        "array": None,   # hires already IS a literal array
    },
    "straw": {
        # days 5-9 use min(straw_early 18, straw_target 33) = 18; days 10-13 use 33
        "base": [0] * 5 + [18] * 5 + [33] * 4 + [0] * 16,
        "days": (5, 13),
        "array": "STRAW_SCHED",
        "find": 'straw_target = knobs["straw_target"] if day >= 10 else '
                'min(knobs["straw_early"], knobs["straw_target"])',
        "repl": 'straw_target = (STRAW_SCHED[day] if day < len(STRAW_SCHED) '
                'else knobs["straw_target"])',
    },
    "land": {
        # base: no land before day 5, then a standing wish for 2 quadrants
        # through day 22. A per-day CAP on quadrants owned expresses timing
        # directly, and lets the rollout reach the $4,000 third quadrant.
        "base": [0] * 5 + [2] * 18 + [0] * 7,
        "days": (4, 20),
        "array": "LAND_SCHED",
        "find": "    if next_idx < CHAMP_LAND_BUYS and CHAMP_LAND_DAY <= day <= 22:",
        "repl": "    _lcap = (LAND_SCHED[day] if day < len(LAND_SCHED) "
                "else CHAMP_LAND_BUYS)\n"
                "    if next_idx < _lcap and day <= 22:",
    },
    "wheat": {
        # day<10 -> 8 standing; day>=10 -> wheat_engine 32
        "base": [0, 0] + [8] * 8 + [32] * 16 + [0] * 4,
        "days": (2, 25),
        "array": "WHEAT_SCHED",
        "find": "            if day < 10:\n                wheat_target = 8\n"
                "            else:\n                wheat_target = min(knobs"
                '["wheat_engine"], crops.get("WHEAT", 0) + free)',
        # The early branch has NO free-tile cap; applying one there changed the
        # base behaviour (-$31,488 against champ-h8's -$27,602), so the two
        # branches are kept distinct and only the target is read from the array.
        "repl": "            if day < 10:\n"
                "                wheat_target = (WHEAT_SCHED[day] "
                "if day < len(WHEAT_SCHED) else 8)\n"
                "            else:\n"
                "                _wt = (WHEAT_SCHED[day] if day < len(WHEAT_SCHED) "
                'else knobs["wheat_engine"])\n'
                '                wheat_target = min(_wt, crops.get("WHEAT", 0) + free)',
    },
}


def _split_bundle(src):
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    return src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]


def bundle_for(name, schedule):
    key = hashlib.sha1(f"{name}:{list(schedule)}".encode()).hexdigest()[:12]
    d = os.path.join(AGENT_DIR, key)
    p = os.path.join(d, "main.py")
    if os.path.exists(p):
        return p
    spec = SCHEDULES[name]
    src = open(os.path.join(REPO, BASE_AGENT)).read()

    if spec["array"] is None:                       # hires: swap the literal
        head, mods, tail = _split_bundle(src)
        out = []
        for mod, text in mods:
            if mod == "planner":
                text, n = re.subn(r"CHAMP_HIRES = \[[0-9,\s\n]*?\]",
                                  "CHAMP_HIRES = " + repr(list(schedule)), text, count=1)
                if n != 1:
                    raise RuntimeError("CHAMP_HIRES literal not found")
            out.append((mod, text))
    else:                                            # crops: inject + rewrite
        head, mods, tail = _split_bundle(src)
        out = []
        for mod, text in mods:
            if mod == "planner":
                if spec["find"] not in text:
                    raise RuntimeError(f"anchor for {name} not found in planner source")
                text = text.replace(spec["find"], spec["repl"], 1)
                anchor = "CHAMP_HIRES = "
                k = text.index(anchor)
                text = (text[:k] + f"{spec['array']} = {list(schedule)!r}\n" + text[k:])
            out.append((mod, text))

    os.makedirs(d, exist_ok=True)
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


def _run_one(task):
    agent_path, opp, seed = task
    try:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
                   debug=False)
        env.run([agent_path, opp])
        f = [env.steps[-1][s]["reward"] for s in (0, 1)]
        if f[0] is None or f[1] is None:
            return None
        return float(f[0] - f[1])
    except Exception:
        return None


def evaluate(name, schedule, worlds, pool):
    path = bundle_for(name, schedule)
    vals = [v for v in pool.map(_run_one, [(path, o, s) for o, s in worlds]) if v is not None]
    return sum(vals) / len(vals) if vals else -1e18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True, choices=sorted(SCHEDULES))
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=51000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--days", default="",
                    help="restrict the search window, e.g. 0-6 for the opening")
    ap.add_argument("--deltas", default="-4,-2,0,2,4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = SCHEDULES[args.schedule]
    base = list(spec["base"])
    lo, hi = spec["days"]
    if args.days:
        # --days 0-6 restricts the search to the opening without
        # editing the schedule definition
        a, _, b = args.days.partition("-")
        lo, hi = int(a), int(b or a)
    deltas = [int(d) for d in args.deltas.split(",")]
    worlds = [(o, args.base_seed + k) for o in OPPS for k in range(args.worlds)]
    out_path = args.out or f"rl/datasets/rollout_{args.schedule}.json"

    print(f"schedule '{args.schedule}' base {base}")
    print(f"days {lo}-{hi}, deltas {deltas}, {len(worlds)} worlds\n", flush=True)

    t0 = time.time()
    with Pool(args.procs) as pool:
        cur = list(base)
        best = evaluate(args.schedule, cur, worlds, pool)
        print(f"base margin {best:+,.0f}\n", flush=True)
        history = []
        for day in range(lo, hi + 1):
            scores = {}
            for d in deltas:
                v = max(0, cur[day] + d)
                if v not in scores:
                    cand = list(cur)
                    cand[day] = v
                    scores[v] = evaluate(args.schedule, cand, worlds, pool)
            bv = max(scores, key=lambda v: scores[v])
            if bv != cur[day] and scores[bv] > best:
                print(f"  day {day:>2}: {cur[day]} -> {bv}  margin {scores[bv]:+,.0f} "
                      f"({scores[bv]-best:+,.0f})", flush=True)
                cur[day], best = bv, scores[bv]
            else:
                print(f"  day {day:>2}: keep {cur[day]}", flush=True)
            history.append({"day": day, "chosen": cur[day], "scores": scores, "running": best})

    print(f"\nbase     {base}\nrollout  {cur}\nmargin   -> {best:+,.0f}   "
          f"({(time.time()-t0)/60:.1f} min)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"schedule": args.schedule, "base": base, "rollout": cur,
               "margin": best, "history": history}, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
