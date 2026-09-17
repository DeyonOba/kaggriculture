"""The relationship between the metrics -- which is what every failed
optimisation this project ran was actually about, and what none of them
measured.

THE PROBLEM, stated properly. Every lever tested here improved its own metric
and lost money:

  geese          egg supply up          labour displaced      all 6 ladder bins worse
  tomato         tomato supply up       tiles displaced       -$1,171 +/- 618
  triage keep18  crop deaths 75%->60%   P_TEND work displaced -$1,028 +/- 511
  crops book     carrots 37->126        wheat 181->100        +$231 +/- 3,186
  v-water        watering re-priced     -                     -$534 +/- 556

Each was recorded as a separate null. They are not separate: they are one
phenomenon seen five times. Improving any metric costs another because the
constraints couple them, and NOBODY EVER MEASURED THE COUPLING. A per-lever
experiment cannot see it -- only a joint measurement can.

WHAT THIS DOES. Runs a set of agent variants over identical seeds, computes the
FULL metric vector per game per DAY-BIN (opening 0-6, contest 7-17, consolidate
18-25, endgame 26-29, matching docs/game_phases.md), and records the outcome
alongside. Then it reports:

  1. the correlation matrix among metrics  -- which ones cannot move alone
  2. the correlation of each metric with the OUTCOME, per bin -- which ones
     actually predict winning, and where
  3. a partial view: metric vs outcome after removing the metric it is most
     entangled with

This is the covariance structure of the executor. It is the thing that says
"stop tuning A, it is 0.9-correlated with B and B is what pays".

Cheap by design: 12 agents x 12 seeds is 144 games, a few minutes, and the
variants already exist from every rejected experiment.

    PYTHONPATH=. uv run python tools/metric_matrix.py --n 12
"""

import argparse
import collections
import itertools
import math
import os
import statistics as st
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

BINS = [("open", 0, 6), ("contest", 7, 17), ("consol", 18, 25), ("end", 26, 29)]
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
WORK_OPS = ["WATER", "HARVEST", "FEED", "CARE", "COLLECT_FERTILIZER",
            "FERTILIZE", "PLANT", "PICKUP", "DROP", "DIG", "PLACE"]


def _row(task):
    agent, opp, seed = task
    from kaggle_environments import make
    from kagent.constants import CROPS
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
             os.path.join(REPO, "arena", "agents", opp, "main.py")])

    per = {b[0]: collections.Counter() for b in BINS}
    empty = {b[0]: [] for b in BINS}
    weeds = {b[0]: [] for b in BINS}
    cash = {b[0]: [] for b in BINS}
    prev_grid = None

    def binof(d):
        for name, lo, hi in BINS:
            if lo <= d <= hi:
                return name
        return None

    for t in range(1, len(env.steps)):
        o = env.steps[t][0].observation
        b = binof(o["day"])
        if b is None:
            continue
        a = env.steps[t][0].get("action") or {}
        for x in [a.get("farmer")] + list(a.get("hands") or []):
            op = x[0] if isinstance(x, list) and x else "PASS"
            if op in MOVES:
                per[b]["move"] += 1
            elif op == "PASS":
                per[b]["pass"] += 1
            else:
                per[b]["work"] += 1
                per[b]["op_" + op] += 1
        for m in (a.get("market") or []):
            if isinstance(m, list) and m:
                per[b]["mk_" + m[0]] += 1
        if o["hour"] == 23:
            tiles = o["farms"][0]["tiles"]
            grid = {}
            e = w = 0
            for y, row in enumerate(tiles):
                for x, c in enumerate(row):
                    if c is None:
                        e += 1
                    elif isinstance(c, dict):
                        k = c.get("kind")
                        if k == "WEED":
                            w += 1
                        grid[(x, y)] = c.get("crop") if k == "PLANT" else k
            empty[b].append(e); weeds[b].append(w)
            cash[b].append(float(o["farms"][0].get("money", 0)))
            if prev_grid is not None:
                for k2, v in grid.items():
                    p = prev_grid.get(k2)
                    if v == "WEED" and p in CROPS:
                        per[b]["died"] += 1
                    elif v in CROPS and p not in CROPS:
                        per[b]["planted"] += 1
            prev_grid = grid

    fin = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if fin[0] is None or fin[1] is None:
        return None
    out = {"agent": agent, "seed": seed,
           "margin": float(fin[0] - fin[1]), "win": 1.0 if fin[0] > fin[1] else 0.0}
    for name, _, _ in BINS:
        c = per[name]
        tot = c["work"] + c["move"] + c["pass"] or 1
        out[f"{name}.work_share"] = c["work"] / tot
        out[f"{name}.move_share"] = c["move"] / tot
        out[f"{name}.pass_share"] = c["pass"] / tot
        out[f"{name}.mv_per_act"] = c["move"] / max(1, c["work"])
        out[f"{name}.actions"] = c["work"]
        for op in WORK_OPS:
            out[f"{name}.{op.lower()}"] = c["op_" + op]
        out[f"{name}.sells"] = c["mk_SELL"]
        out[f"{name}.hires"] = c["mk_HIRE"]
        out[f"{name}.died"] = c["died"]
        out[f"{name}.planted"] = c["planted"]
        out[f"{name}.empty"] = st.mean(empty[name]) if empty[name] else 0.0
        out[f"{name}.weeds"] = st.mean(weeds[name]) if weeds[name] else 0.0
        out[f"{name}.cash"] = st.mean(cash[name]) if cash[name] else 0.0
    return out


def corr(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = st.mean(a), st.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=("champ-h8,g-goose4,g-tom9,g-tom14,"
                                         "t-keep12,t-keep18,v-water9,j-tile,"
                                         "j-crops,j-crops-tile,bo-best,hill-best"))
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=94000)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    agents = [a for a in args.agents.split(",")
              if os.path.isdir(os.path.join(REPO, "arena", "agents", a))]
    seeds = [args.base_seed + k for k in range(args.n)]
    tasks = [(a, args.opp, s) for a in agents for s in seeds]
    print(f"{len(agents)} agents x {len(seeds)} seeds = {len(tasks)} games\n", flush=True)

    import json
    cache = os.path.join(REPO, "rl", "metric_rows.json")
    if args.reuse and os.path.exists(cache):
        rows = json.load(open(cache))
        print(f"{len(rows)} rows (cached)\n")
    else:
        with Pool(args.procs) as pool:
            rows = [r for r in pool.map(_row, tasks) if r]
        json.dump(rows, open(cache, "w"))
        print(f"{len(rows)} rows -> {cache}\n")

    keys = [k for k in rows[0] if k not in ("agent", "seed", "margin", "win")]
    col = {k: [r[k] for r in rows] for k in keys}
    y = [r["margin"] for r in rows]

    # 1. which metrics track the OUTCOME
    with_out = sorted(((corr(col[k], y), k) for k in keys),
                      key=lambda t: -abs(t[0]))
    print("METRIC vs OUTCOME (margin), across all agents and seeds\n")
    print(f"  {'metric':<28}{'r':>8}")
    for c, k in with_out[:args.top]:
        print(f"  {k:<28}{c:>+8.2f}")

    # 2. which metrics cannot move alone
    print(f"\nENTANGLEMENT -- pairs that move together (|r| >= 0.8)\n")
    top_keys = [k for _, k in with_out[:args.top * 2]]
    pairs = []
    for a, b in itertools.combinations(top_keys, 2):
        c = corr(col[a], col[b])
        if abs(c) >= 0.8:
            pairs.append((abs(c), c, a, b))
    pairs.sort(reverse=True)
    if not pairs:
        print("  none above 0.8")
    for _, c, a, b in pairs[:args.top]:
        print(f"  {a:<26} {b:<26} {c:>+6.2f}")

    # 3. the tell: a metric correlated with the outcome ONLY through another
    print(f"\nPARTIAL -- metric vs outcome after removing its strongest partner\n")
    print(f"  {'metric':<26}{'r raw':>9}{'partner':>26}{'r partial':>11}")
    for c0, k in with_out[:args.top]:
        best = None
        for o in keys:
            if o == k:
                continue
            cc = abs(corr(col[k], col[o]))
            if best is None or cc > best[0]:
                best = (cc, o)
        if not best or best[0] < 0.5:
            continue
        p = best[1]
        rkp, ryp = corr(col[k], col[p]), corr(col[p], y)
        den = math.sqrt(max(1e-9, (1 - rkp ** 2) * (1 - ryp ** 2)))
        print(f"  {k:<26}{c0:>+9.2f}{p:>26}{(c0 - rkp * ryp) / den:>+11.2f}")

    print("\n  A metric whose PARTIAL collapses toward zero does not pay on its "
          "own --\n  it was only ever tracking its partner. That is the coupling "
          "every\n  single-lever experiment here ran into and none of them could see.")


if __name__ == "__main__":
    main()
