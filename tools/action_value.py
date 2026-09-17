"""What is ONE action actually worth, by day -- and does the LP's shadow price
predict it?

The design question this settles: counterfactual labels cost 1.09 s each and
are ground truth; LP duals are free and are an ESTIMATE from a relaxed model
(eff 0.414, movement abstracted away, fractional tiles). So the expensive
source should be spent validating the free one. If the LP's labour price
predicts the realised value of an action, we distil the LP for nothing. If it
does not, distilling it would teach the wrong teacher -- and there is already a
warning on record: the LP's provably optimal allocation, executed by our own
scheduler, tied champ-h8 at +$828 +/- 2,317.

METHOD. On a fixed world with a scripted opponent the game is deterministic
(verified: three identical runs returned $132,321.0). Replace ONE work action
with PASS and replay: the margin lost is exactly what that action was worth.

Replacing work with PASS is the clean perturbation. PASS does not move the
unit, and neither did the work action, so every later scripted action still
lands on the tile it was written for. That is what contaminated
tools/turn_mutate.py, which substituted arbitrary ops including moves and
measured mostly desync (83.4% worse, median -$1,438).

    PYTHONPATH=. uv run python tools/action_value.py --per-day 8
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import statistics as st
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

TURNS = 24
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def _eval(task):
    actions, seed, opp = task
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([make_script_agent(actions), make_script_agent(opp)])
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return None
    return float(f[0] - f[1])


def blank(actions, turn, unit):
    """One work action replaced by PASS. Position unchanged, so no desync."""
    out = list(actions)
    a = dict(out[turn])
    if unit == 0:
        a["farmer"] = ["PASS"]
    else:
        h = list(a.get("hands") or [])
        if unit - 1 < len(h):
            h[unit - 1] = ["PASS"]
        a["hands"] = h
    out[turn] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=1,
                    help="replicate across N worlds and pool")
    ap.add_argument("--arm", type=int, default=129)
    ap.add_argument("--per-day", type=int, default=8)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tools.tape_agent import _load_lib
    from tools.raw_bc import cluster_episodes
    eps = cluster_episodes(3029.3)
    files = [f for f in sorted(glob.glob(os.path.join(
        REPO, "rl", "datasets", "sessions", "session-*.json")))
        if int(f.split("-")[-1][:-5]) in eps]
    lib = _load_lib()
    base = list(lib[args.arm]["actions"])
    worlds = files[args.world:args.world + args.worlds]

    # index every work action by day (same for every world: same arm)
    byday = collections.defaultdict(list)
    for t, turn in enumerate(base):
        acts = [turn.get("farmer")] + list(turn.get("hands") or [])
        for u, a in enumerate(acts):
            op = a[0] if isinstance(a, list) and a else "PASS"
            if op not in MOVES and op != "PASS":
                byday[t // TURNS].append((t, u))

    rng = random.Random(args.seed)
    picks = []
    for d, lst in byday.items():
        lst = list(lst); rng.shuffle(lst)
        picks += [(d, t, u) for t, u in lst[:args.per_day]]
    n_eval = (len(picks) + 1) * len(worlds)
    print(f"{len(worlds)} worlds x {len(picks)} deletions = {n_eval} evaluations "
          f"(~{n_eval*1.09/args.procs/60:.0f} min)\n", flush=True)

    per = collections.defaultdict(list)
    with Pool(args.procs) as pool:
        for wi, f in enumerate(worlds):
            sess = json.load(open(f))
            seed, opp = sess["seed"], sess["actions"][1]
            b = pool.map(_eval, [(base, seed, opp)])[0]
            if b is None:
                continue
            vals = pool.map(_eval, [(blank(base, t, u), seed, opp)
                                    for _, t, u in picks])
            for (d, t, u), v in zip(picks, vals):
                if v is not None:
                    per[d].append(b - v)
            print(f"  world {wi+1}/{len(worlds)} ep {sess['episode']} "
                  f"base {b:>+10,.0f}", flush=True)

    from tools.lp_bound import build_and_solve
    print("solving the LP for its labour dual...", flush=True)
    lp = build_and_solve(eff=0.414, shops=None)
    ld = lp["labour_duals"]

    print(f"\n  {'phase':<16}{'n':>6}{'REALISED $/action':>20}{'se':>10}{'LP dual':>10}{'ratio':>9}")
    PH = [("opening 0-6",0,6),("contest 7-17",7,17),("consol 18-25",18,25),
          ("endgame 26-29",26,29)]
    xs, ys = [], []
    for name, lo, hi in PH:
        vv = [v for d in range(lo,hi+1) for v in per.get(d,[])]
        if not vv: continue
        se = st.stdev(vv)/len(vv)**0.5 if len(vv)>1 else 0
        lpm = st.mean(ld[lo:hi+1])
        print(f"  {name:<16}{len(vv):>6}{st.mean(vv):>20,.0f}{se:>10,.0f}"
              f"{lpm:>10,.1f}{st.mean(vv)/max(lpm,1e-9):>8.0f}x")
    for d in sorted(per):
        if per[d]:
            xs.append(ld[d]); ys.append(st.mean(per[d]))
    early=[v for d in range(0,12) for v in per.get(d,[])]
    late=[v for d in range(12,30) for v in per.get(d,[])]
    if early and late:
        ese=st.stdev(early)/len(early)**0.5; lse=st.stdev(late)/len(late)**0.5
        print(f"\n  days 0-11   {st.mean(early):>+9,.0f} +/- {ese:,.0f}  (n={len(early)})")
        print(f"  days 12-29  {st.mean(late):>+9,.0f} +/- {lse:,.0f}  (n={len(late)})")
        diff=st.mean(early)-st.mean(late); dse=(ese**2+lse**2)**0.5
        print(f"  difference  {diff:>+9,.0f} +/- {dse:,.0f}   z = {diff/dse if dse else 0:.2f}")

    ma, mb = st.mean(xs), st.mean(ys)
    va = sum((x - ma) ** 2 for x in xs); vb = sum((y - mb) ** 2 for y in ys)
    c = (sum((x - ma) * (y - mb) for x, y in zip(xs, ys)) / math.sqrt(va * vb)
         if va > 0 and vb > 0 else 0.0)
    print(f"\n  mean realised {mb:>8,.1f}    mean LP dual {ma:>8,.1f}"
          f"    ratio {mb/ma if ma else 0:.2f}")
    print(f"  correlation between the LP's price and the realised price: "
          f"r = {c:+.2f}")
    print(f"\n  r near 0 means the LP is the wrong teacher: distilling it would "
          f"train a\n  model to price actions in a way our executor does not "
          f"experience.")


if __name__ == "__main__":
    main()
