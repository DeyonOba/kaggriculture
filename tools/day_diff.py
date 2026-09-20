"""Same world, same day, same pending work: what did each agent DO with it?

The scorecard says our executor is out of band on all four axes -- less work,
more walking. It does not say WHERE in the day that happens. This puts a top
player and our agent on the SAME seed against the SAME opponent tape, and for
each day prints the work that was standing at dawn next to the actions each
one actually spent.

"Work available at dawn" is the engine's own obligations:
    dry      crops not watered today  (2 unwatered days kills the plant)
    ripe     tiles holding yield_units
    hungry   animals not fed today    (FEED costs 1 WHEAT each)
    uncared  animals not cared today  (CARE doubles that day's output)
    empty    plantable tiles

    PYTHONPATH=. uv run python tools/day_diff.py --n 3 --days 12
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def pending(farm):
    """Standing obligations on a board at this instant."""
    d = {"dry": 0, "ripe": 0, "hungry": 0, "uncared": 0, "empty": 0}
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                d["empty"] += 1
            elif not isinstance(t, dict):
                continue
            elif t.get("kind") == "PLANT":
                if not t.get("watered_today"):
                    d["dry"] += 1
                if t.get("yield_units", 0) > 0:
                    d["ripe"] += 1
            elif t.get("animal"):
                if not t.get("fed_today"):
                    d["hungry"] += 1
                if not t.get("cared_today"):
                    d["uncared"] += 1
                if t.get("yield_units", 0) > 0:
                    d["ripe"] += 1
    return d


def tally(actions):
    c = {"move": 0, "pass": 0, "water": 0, "harvest": 0, "feed": 0,
         "care": 0, "plant": 0, "haul": 0, "other": 0}
    for a in actions:
        if not a:
            continue
        op = a[0]
        if op in MOVES:
            c["move"] += 1
        elif op == "PASS":
            c["pass"] += 1
        elif op == "WATER":
            c["water"] += 1
        elif op == "HARVEST":
            c["harvest"] += 1
        elif op == "FEED":
            c["feed"] += 1
        elif op == "CARE":
            c["care"] += 1
        elif op == "PLANT":
            c["plant"] += 1
        elif op in ("PICKUP", "DROP", "COLLECT_FERTILIZER"):
            c["haul"] += 1
        else:
            c["other"] += 1
    return c


def run(seed, actor, opp_actions, days):
    """actor is (kind, payload): ('tape', actions) or ('agent', name)."""
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.price_forecast as pf

    if actor[0] == "tape":
        a0 = pf._script(actor[1])
    else:
        g = {}
        exec(open(f"{REPO}/arena/agents/{actor[1]}/main.py").read(), g)
        a0 = g["agent"]
    a1 = pf._script(opp_actions)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    rows = []
    cur, pend = None, None
    while not env.done:
        o = env.state[0].observation
        d, hr = o["step"] // 24, o["step"] % 24
        if hr == 0:
            if cur is not None:
                rows.append((len(rows), pend, cur))
            pend = pending(o["farms"][0])
            cur = tally([])
        act = a0(o)
        t = tally([act.get("farmer")] + list(act.get("hands") or []))
        for k in cur:
            cur[k] += t[k]
        if d >= days:
            break
        env.step([act, a1(env.state[1].observation)])
    if cur is not None:
        rows.append((len(rows), pend, cur))
    return rows, env.state[0].reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="r-mrg100")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--min-rating", type=float, default=2900.0)
    args = ap.parse_args()

    from tools.opp_coupling import load_index
    import random
    idx = load_index()
    paths = [v for v in idx.values()
             if v.get("scores") and all(x is not None for x in v["scores"])
             and min(v["scores"]) >= args.min_rating]
    random.Random(3).shuffle(paths)

    for v in paths[: args.n]:
        s = json.load(open(v["path"]))
        theirs, ours_r = run(s["seed"], ("tape", s["actions"][0]),
                             s["actions"][1], args.days)
        ours, ours_m = run(s["seed"], ("agent", args.agent),
                           s["actions"][1], args.days)
        print(f"\n{'='*92}")
        print(f"episode {s['episode']}  seed {s['seed']}  "
              f"rated {min(v['scores']):.0f}/{max(v['scores']):.0f}")
        print(f"  seat 0: TOP PLAYER vs OUR AGENT ({args.agent}), "
              f"identical opponent tape in seat 1")
        print(f"{'='*92}")
        print(f"{'':>4}{'pending at dawn':^34}"
              f"{'TOP PLAYER spent':^26}{'OURS spent':^26}")
        print(f"{'day':>4}{'dry':>6}{'ripe':>6}{'hungry':>8}{'empty':>7}"
              f"{'|':>3}{'wat':>5}{'har':>5}{'fd/cr':>7}{'mv':>6}"
              f"{'|':>3}{'wat':>5}{'har':>5}{'fd/cr':>7}{'mv':>6}")
        for (d, p, tc), (_, p2, oc) in zip(theirs, ours):
            print(f"{d:>4}{p['dry']:>6}{p['ripe']:>6}{p['hungry']:>8}"
                  f"{p['empty']:>7}{'|':>3}"
                  f"{tc['water']:>5}{tc['harvest']:>5}"
                  f"{tc['feed']+tc['care']:>7}{tc['move']:>6}{'|':>3}"
                  f"{oc['water']:>5}{oc['harvest']:>5}"
                  f"{oc['feed']+oc['care']:>7}{oc['move']:>6}")
        tw = sum(c["water"] + c["harvest"] + c["feed"] + c["care"] + c["plant"]
                 + c["haul"] + c["other"] for _, _, c in theirs)
        ow = sum(c["water"] + c["harvest"] + c["feed"] + c["care"] + c["plant"]
                 + c["haul"] + c["other"] for _, _, c in ours)
        tm = sum(c["move"] for _, _, c in theirs)
        om = sum(c["move"] for _, _, c in ours)
        print(f"\n  through day {args.days}:   work {tw} vs {ow}"
              f"   moves {tm} vs {om}"
              f"   moves/work {tm/max(tw,1):.2f} vs {om/max(ow,1):.2f}")


if __name__ == "__main__":
    main()
