"""WHERE do crops and livestock go, and where SHOULD they go?

57.4% of our action budget is movement (tools/executor_audit.py), and the
shed is fixed: it sits at the board centre and every unit spawns beside it
each morning. So tile placement is not decoration -- it sets the travel bill
for the whole season.

The verified rules split the ops by whether they need a CARRIED CONSUMABLE:

    needs a shed round-trip        FEED (1 wheat/animal/day), FERTILIZE
    produces into the bag          HARVEST, COLLECT_FERTILIZER
    needs presence only            WATER, CARE, PLANT, DIG, BUILD_*

Animals therefore demand a wheat delivery EVERY DAY for their whole life,
and an unfed animal escapes after two days. Crops need watering, which
carries nothing. That predicts a radial layout: livestock close to the shed,
crops pushed out. This measures whether anyone actually plays that way.

Distance is Manhattan to the nearest of the four shed-access tiles
(4,4) (5,4) (4,5) (5,5) at boardSize 10 -- verified in readme_audit.

    PYTHONPATH=. uv run python tools/placement_heatmap.py --n 40
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "rl/datasets/placement.npz")
TPD = 24
N = 10


def shed_dist_grid(n=N):
    half = n // 2
    access = [(half - 1, half - 1), (half, half - 1),
              (half - 1, half), (half, half)]
    g = np.zeros((n, n), "int16")
    for y in range(n):
        for x in range(n):
            g[y, x] = min(abs(x - ax) + abs(y - ay) for ax, ay in access)
    return g


DIST = shed_dist_grid()


def _grids_from_farm(farm):
    """-> (crop, livestock, structure, weed) occupancy grids for one snapshot."""
    crop = np.zeros((N, N), "float32")
    live = np.zeros((N, N), "float32")
    struct = np.zeros((N, N), "float32")
    weed = np.zeros((N, N), "float32")
    for y in range(N):
        for x in range(N):
            t = farm["tiles"][y][x]
            if not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "PLANT":
                crop[y, x] = 1
            elif k == "WEED":
                weed[y, x] = 1
            elif t.get("animal"):
                live[y, x] = 1
            elif k in ("COOP", "PASTURE"):
                struct[y, x] = 1
    return crop, live, struct, weed


def job_agent(spec):
    seed, agent, opp, days = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load(agent), load(opp)
    e = make("kaggriculture", configuration={"seed": seed}, debug=False)
    e.reset(2)
    acc = np.zeros((4, N, N), "float32")
    hits = 0
    while not e.done:
        o = e.state[0].observation
        if o["step"] % TPD == 12 and (o["step"] // TPD) in days:
            for i, g in enumerate(_grids_from_farm(o["farms"][0])):
                acc[i] += g
            hits += 1
        e.step([a0(o), a1(e.state[1].observation)])
    return acc, hits, float(e.state[0].reward - e.state[1].reward)


def job_corpus(spec):
    """Replay a RECORDED session and read the stronger seat's board."""
    path, days = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.price_forecast as pf
    try:
        s = json.load(open(path))
    except Exception:
        return None
    e = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    e.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    fin = s.get("finals") or [0, 0]
    seat = 0 if fin[0] >= fin[1] else 1        # the WINNING seat's layout
    acc = np.zeros((4, N, N), "float32")
    hits = 0
    while not e.done:
        o = e.state[0].observation
        if o["step"] % TPD == 12 and (o["step"] // TPD) in days:
            for i, g in enumerate(_grids_from_farm(o["farms"][seat])):
                acc[i] += g
            hits += 1
        e.step([a0(o), a1(e.state[1].observation)])
    return acc, hits, 0.0


def build(args):
    days = set(range(args.day_lo, args.day_hi))
    print(f"sampling days {args.day_lo}-{args.day_hi - 1} at hour 12\n")

    print(f"  A. {args.agent} self-play, {args.n} games ...", flush=True)
    specs = [(700500 + i, args.agent, args.opp, days) for i in range(args.n)]
    with mp.Pool(args.procs, maxtasksperchild=20) as pool:
        got = [r for r in pool.map(job_agent, specs) if r]
    ours = sum(r[0] for r in got)
    ours_hits = sum(r[1] for r in got)

    from tools.opp_coupling import load_index
    idx = load_index()
    rows = []
    for v in idx.values():
        sc = [x for x in (v.get("scores") or []) if isinstance(x, (int, float))]
        if sc:
            rows.append((max(sc), v["path"]))
    rows.sort()
    top = [p for _, p in rows[-int(len(rows) * 0.10):]]
    random.Random(5).shuffle(top)
    top = top[: args.n]
    print(f"  B. top-10% corpus, {len(top)} recorded games ...", flush=True)
    with mp.Pool(args.procs, maxtasksperchild=20) as pool:
        gotc = [r for r in pool.map(job_corpus, [(p, days) for p in top]) if r]
    theirs = sum(r[0] for r in gotc)
    theirs_hits = sum(r[1] for r in gotc)

    np.savez_compressed(OUT, ours=ours, ours_hits=ours_hits,
                        theirs=theirs, theirs_hits=theirs_hits, dist=DIST)
    print(f"\n-> {OUT}")
    report(args)


def _radial(grid, hits):
    """Occupancy per tile, bucketed by Manhattan distance from the shed."""
    out = {}
    for d in range(int(DIST.max()) + 1):
        m = DIST == d
        if not m.any():
            continue
        out[d] = float(grid[m].sum() / (hits * m.sum())) if hits else 0.0
    return out


def report(args):
    d = np.load(OUT)
    ours, oh = d["ours"], int(d["ours_hits"])
    theirs, th = d["theirs"], int(d["theirs_hits"])
    names = ["crop", "livestock", "structure", "weed"]

    def show(grid, hits, title):
        print(f"\n  {title}  (occupancy per tile, x10)")
        print("     " + "".join(f"{x:>5}" for x in range(N)))
        for y in range(N):
            row = "".join(f"{10 * grid[y, x] / max(hits, 1):>5.1f}"
                          for x in range(N))
            print(f"  {y:>2} {row}")

    print("\n" + "=" * 72)
    print("  RADIAL PROFILE -- occupancy by Manhattan distance from the shed")
    print("=" * 72)
    print(f"  {'dist':>5}{'tiles':>7}"
          f"{'OUR crop':>11}{'OUR live':>11}"
          f"{'TOP crop':>11}{'TOP live':>11}")
    oc, ol = _radial(ours[0], oh), _radial(ours[1] + ours[2], oh)
    tc, tl = _radial(theirs[0], th), _radial(theirs[1] + theirs[2], th)
    for dd in sorted(oc):
        ntile = int((DIST == dd).sum())
        print(f"  {dd:>5}{ntile:>7}{oc[dd]:>11.3f}{ol[dd]:>11.3f}"
              f"{tc.get(dd, 0):>11.3f}{tl.get(dd, 0):>11.3f}")

    # the decisive single number: mean distance of each class from the shed
    print("\n" + "=" * 72)
    print("  MEAN SHED DISTANCE  (livestock SHOULD be closer: FEED needs a")
    print("  daily wheat delivery; WATER carries nothing)")
    print("=" * 72)

    def meand(grid):
        tot = grid.sum()
        return float((grid * DIST).sum() / tot) if tot > 0 else float("nan")
    for lbl, g in (("ours", ours), ("top 10%", theirs)):
        c = meand(g[0])
        l = meand(g[1] + g[2])
        verdict = ("livestock CLOSER  <- as the rules imply" if l < c - 0.15
                   else "livestock FARTHER <- backwards"
                   if l > c + 0.15 else "no separation")
        print(f"  {lbl:<10} crop {c:>5.2f}   livestock {l:>5.2f}"
              f"   gap {c - l:>+5.2f}   {verdict}")

    show(ours[0], oh, "OURS - crops")
    show(ours[1] + ours[2], oh, "OURS - livestock + structures")
    show(theirs[0], th, "TOP 10% - crops")
    show(theirs[1] + theirs[2], th, "TOP 10% - livestock + structures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report"], nargs="?",
                    default="build")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--day-lo", type=int, default=10)
    ap.add_argument("--day-hi", type=int, default=26)
    args = ap.parse_args()
    (build if args.cmd == "build" else report)(args)


if __name__ == "__main__":
    main()
