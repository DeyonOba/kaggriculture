"""Rollout policy improvement over the per-day HIRING schedule.

Why hiring, and why per-day. The LP bound says labour binds on 29 of 30 days
and tiles on 16, so labour is the constraint worth spending compute on. The
constant `hires_delta` knob is already refuted (+2 hands -$5,229, +4 -$19,344,
monotone), but that knob can only shift the WHOLE season by a fixed amount.
`planner.CHAMP_HIRES` is a 30-entry per-day schedule, and the space of
schedules is vastly larger than the space of constant offsets. That is exactly
the gap a rollout is for.

Method (Bertsekas rollout / one-step lookahead policy improvement): walk the
days in order; at each day try every candidate value for that day, evaluate the
resulting FULL schedule by simulating whole seasons with the base policy, keep
the best, move on. Because the incumbent value is always among the candidates,
the result is never worse than the base schedule ON THE EVALUATION WORLDS --
the cost-improvement property. Whether it generalises to fresh worlds is a
separate question and is exactly what the gate is for.

Mid-game state resume is NOT used: `Environment(steps=...)` does not faithfully
restore engine state (measured -- resuming a scripted replay at step 24 scored
$1,047 against a $68,748 reference), so every evaluation re-simulates from step
0. At ~2.6 s a game that is affordable: 30 days x k candidates x w worlds.

    PYTHONPATH=. uv run python tools/rollout_hires.py --worlds 4 --procs 6
"""

import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OPPS = ["arena/agents/pub-ahmed-v38/main.py",
        "arena/agents/pub-boatlee-v16/main.py"]


# The dev kagent/ tree can NO LONGER reproduce the frozen champion: with the
# shipped rules injected it still scores $25,417 where the frozen bundle scores
# $59,656 on the same world, because the planner source has moved on since
# champ-h8 was frozen. So the rollout must operate on the FROZEN bundle, which
# cannot be monkeypatched -- it is patched textually instead, the same way
# tools/make_knob_variant.py patches baked-in knob rules.
# The schedule is only meaningful for the stack it was optimised on: the
# rent planner changed what a marginal hand is worth per day. Override with
# --base so a rollout is never silently fitted to a superseded agent.
BASE_AGENT = "arena/agents/champ-h8/main.py"
AGENT_DIR = "/tmp/rollout_agents"
_HIRES_RE = None


def _bundle_for(schedule):
    """Write (once) a frozen bundle whose embedded CHAMP_HIRES is `schedule`."""
    import hashlib
    import re
    key = hashlib.sha1(repr(list(schedule)).encode()).hexdigest()[:12]
    d = os.path.join(AGENT_DIR, key)
    p = os.path.join(d, "main.py")
    if os.path.exists(p):
        return p
    src = open(os.path.join(REPO, BASE_AGENT)).read()
    pat = re.compile(r"CHAMP_HIRES = \[[0-9,\s\\n]*?\]")
    new, n = pat.subn("CHAMP_HIRES = " + repr(list(schedule)), src, count=1)
    if n != 1:
        raise RuntimeError(f"CHAMP_HIRES literal not found in {BASE_AGENT} (matched {n})")
    os.makedirs(d, exist_ok=True)
    open(p, "w").write(new)
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


def evaluate(schedule, worlds, pool):
    path = _bundle_for(schedule)
    vals = [v for v in pool.map(_run_one, [(path, o, s) for o, s in worlds]) if v is not None]
    return sum(vals) / len(vals) if vals else -1e18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=4, help="seeds per opponent")
    ap.add_argument("--base-seed", type=int, default=51000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--deltas", default="-2,-1,0,1,2")
    ap.add_argument("--days", default="0-29")
    ap.add_argument("--out", default="rl/datasets/rollout_hires.json")
    ap.add_argument("--base", default=None,
                    help="agent whose CHAMP_HIRES is rolled out (default "
                         "champ-h8); use the CURRENT best stack")
    args = ap.parse_args()
    if args.base:
        global BASE_AGENT
        BASE_AGENT = f"arena/agents/{args.base}/main.py"
        print(f"rolling out over {BASE_AGENT}")

    import re
    src = open(os.path.join(REPO, BASE_AGENT)).read()
    m = re.search(r"CHAMP_HIRES = \[([0-9,\s\\n]*?)\]", src)
    base = [int(x) for x in re.findall(r"\d+", m.group(1))]
    deltas = [int(d) for d in args.deltas.split(",")]
    lo, hi = (int(x) for x in args.days.split("-"))
    worlds = [(o, args.base_seed + k) for o in OPPS for k in range(args.worlds)]

    print(f"base schedule: {base}")
    print(f"{len(worlds)} worlds, deltas {deltas}, days {lo}-{hi}\n", flush=True)

    t0 = time.time()
    with Pool(args.procs) as pool:
        cur = list(base)
        best = evaluate(cur, worlds, pool)
        print(f"base margin {best:+,.0f}\n", flush=True)
        history = []
        for day in range(lo, min(hi + 1, len(cur))):
            scores = {}
            for d in deltas:
                v = max(0, cur[day] + d)
                if v in scores:
                    continue
                cand = list(cur)
                cand[day] = v
                scores[v] = evaluate(cand, worlds, pool)
            bestv = max(scores, key=lambda v: scores[v])
            gain = scores[bestv] - best
            if bestv != cur[day] and gain > 0:
                print(f"  day {day:>2}: {cur[day]} -> {bestv}  margin {scores[bestv]:+,.0f} "
                      f"({gain:+,.0f})", flush=True)
                cur[day] = bestv
                best = scores[bestv]
            else:
                print(f"  day {day:>2}: keep {cur[day]}  (best alt {bestv} "
                      f"{scores[bestv]:+,.0f})", flush=True)
            history.append({"day": day, "chosen": cur[day], "scores": scores,
                            "running": best})

    dt = time.time() - t0
    print(f"\nbase     {base}")
    print(f"rollout  {cur}")
    print(f"margin   {evaluate(base, worlds, Pool(args.procs)):+,.0f} -> {best:+,.0f}")
    print(f"{dt/60:.1f} min")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"base": base, "rollout": cur, "margin": best, "history": history,
               "worlds": [[o, s] for o, s in worlds]}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
