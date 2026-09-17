"""W1: is mean V(s) over the trajectory a lower-variance estimator than the
terminal bank?

A game runs 720 steps and we score it with one number. Per-world sd of that
number is $34,749, which is why resolving a $3,000 effect needs ~50,000 games
and why eight experiments in one day came back unmeasurable. The fitted value
head (`rl/value_weights.json`, V(s) = P(win | s), one model per day) turns the
same game into 30 observations.

This is the evaluation analogue of an actor-critic baseline: the terminal
outcome is unbiased but maximally noisy; a fitted value function is biased but
far tighter. For RANKING candidates, bias that is common to all candidates
cancels and only the variance matters.

The test is not "does V look sensible" -- it is whether, ON THE SAME GAMES, the
V-score separates a candidate from the base with a larger |z| than money does.
Same games, so this is a pure estimator comparison with no extra compute.

    PYTHONPATH=. uv run python tools/vscore.py --cand g-tom9 g-tom14 --n 40
"""

import argparse
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from kaggle_environments import make                        # noqa: E402
from tools.value_head import _load_model, value, N_DAYS     # noqa: E402


def play(agent, opp, seed, model):
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
             os.path.join(REPO, "arena", "agents", opp, "main.py")])
    vs = []
    for d in range(N_DAYS):
        i = d * 24
        if i >= len(env.steps):
            break
        vs.append(value(model, env.steps[i][0]["observation"], 0, d))
    fin = [env.steps[-1][i]["reward"] for i in (0, 1)]
    return (st.mean(vs) if vs else 0.0,          # mean win probability held
            float(fin[0] - fin[1]),               # terminal margin
            1.0 if fin[0] > fin[1] else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=91000)
    args = ap.parse_args()

    model = _load_model()
    seeds = [args.base_seed + k for k in range(args.n)]
    runs = {}
    for name in [args.base] + args.cand:
        runs[name] = [play(name, args.opp, s, model) for s in seeds]
        v = st.mean(r[0] for r in runs[name])
        m = st.mean(r[1] for r in runs[name])
        print(f"  {name:<14} meanV {v:.3f}   margin {m:>+10,.0f}   "
              f"win {st.mean(r[2] for r in runs[name]):.3f}", flush=True)

    print(f"\npaired against {args.base} on the same {args.n} worlds\n")
    print(f"  {'candidate':<14}{'money diff':>12}{'se':>8}{'|z|':>7}   "
          f"{'V diff':>9}{'se':>8}{'|z|':>7}   {'z ratio':>8}")
    ratios = []
    for name in args.cand:
        dm = [c[1] - b[1] for c, b in zip(runs[name], runs[args.base])]
        dv = [c[0] - b[0] for c, b in zip(runs[name], runs[args.base])]
        mm, mse = st.mean(dm), st.stdev(dm) / len(dm) ** 0.5
        vm, vse = st.mean(dv), st.stdev(dv) / len(dv) ** 0.5
        zm, zv = abs(mm / mse) if mse else 0, abs(vm / vse) if vse else 0
        ratios.append(zv / zm if zm else 0)
        print(f"  {name:<14}{mm:>+12,.0f}{mse:>8,.0f}{zm:>7.2f}   "
              f"{vm:>+9.4f}{vse:>8.4f}{zv:>7.2f}   {ratios[-1]:>8.2f}")

    print(f"\n  z ratio > 1 means the V-score separates this candidate from the "
          f"base more\n  sharply than money does, on the SAME games. Mean "
          f"ratio {st.mean(ratios):.2f}.")
    print(f"  Equivalent game saving is the ratio SQUARED: "
          f"{st.mean(ratios)**2:.1f}x fewer games for the same power.")
    if len(args.cand) > 1:
        xs = []
        for name in args.cand:
            dm = st.mean(c[1] - b[1] for c, b in zip(runs[name], runs[args.base]))
            dv = st.mean(c[0] - b[0] for c, b in zip(runs[name], runs[args.base]))
            xs.append((dm, dv))
        mx, my = st.mean(x for x, _ in xs), st.mean(y for _, y in xs)
        num = sum((x - mx) * (y - my) for x, y in xs)
        den = (sum((x - mx) ** 2 for x, _ in xs) *
               sum((y - my) ** 2 for _, y in xs)) ** 0.5
        print(f"  agreement between the two estimators across "
              f"{len(xs)} candidates: r = {num/den if den else 0:+.2f}")


if __name__ == "__main__":
    main()
