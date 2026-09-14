"""Joint plan search, scored by the win-probability value head.

Every lever this project can reach has been measured one at a time and none of
them holds the $30-40k gap to a 2500+ agent (see CHANGELOG). The value head says
why that can be true: farm composition is worth <0.01 win probability while cash
is worth +0.099, so the gap lives in the joint trajectory, not in any single
parameter. This searches CONFIGURATIONS -- several knobs moved together -- and
scores them by mean V(s) mid-season rather than by final bank.

Why mid-season V and not money: final money carries ~$20k of per-world noise, so
a $2k effect needs ~50 games to resolve. V(s) is a probability in [0,1] fitted on
3,000 real games, and the head is most informative exactly where the season is
decided (days 10-17). `validate` measures whether that is actually a better
objective before any search runs -- if it is not, this tool should not be used.

    PYTHONPATH=. uv run python tools/plan_search.py validate --n 24
    PYTHONPATH=. uv run python tools/plan_search.py search --iters 40 --games 8
"""

import argparse
import itertools
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.value_head import _load_model, features  # noqa: E402

OPPONENTS = ["arena/agents/pub-ahmed-v38/main.py",
             "arena/agents/pub-tschinkel-psr/main.py",
             "arena/agents/pub-boatlee-v16/main.py"]
BASE_RULES = "arena/agents/champ-h8/rules.json"
SCORE_DAYS = (12, 14, 16)

# Joint space. Each entry is the set of values that knob may take; a candidate
# is one value from each. Centred on champ-h8 so the incumbent is reachable.
SPACE = {
    "cows": [4, 6, 8, 10],
    "sheep": [5, 8, 10, 12],
    "geese": [0, 2, 4],
    "straw_target": [24, 29, 33, 36],
    "wheat_engine": [24, 28, 32, 36],
    "carrot_standing": [0, 6, 12],
    "premium_scale": [0.2, 0.35, 0.6],
    "feed_days": [1, 2, 3],
}


def _agent(rules):
    from kagent.banditpolicy import KnobBanditPolicy
    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy
    pol = HeuristicPolicy(KnobBanditPolicy(rules))
    mem = Memory()

    def fn(obs, config=None):
        view = View(obs)
        mem.observe(view)
        return pol.act(view, mem)
    return fn


def _play(job):
    """Run one game; return (V by day, final margin, win)."""
    rules, opp, seed = job
    from kaggle_environments import make
    model = _load_model()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    try:
        env.run([_agent(rules), opp])
    except Exception:  # noqa: BLE001
        return None
    vs = {}
    for d in SCORE_DAYS:
        i = d * 24
        if i < len(env.steps):
            obs = env.steps[i][0]["observation"]
            if "farms" in obs:
                mu, sd, w = model[d] if d in model else model[max(model)]
                x = (np.array(features(obs, 0)) - mu) / sd
                vs[d] = float(1.0 / (1.0 + np.exp(-np.clip(np.r_[1.0, x] @ w, -30, 30))))
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return None
    return vs, float(f[0] - f[1]), 1.0 if f[0] > f[1] else 0.0


def _run(rules, seeds, procs, opponents=None):
    opponents = opponents or OPPONENTS
    jobs = [(rules, opp, s) for s in seeds for opp in opponents]
    out = []
    with Pool(procs) as pool:
        for r in pool.imap_unordered(_play, jobs, chunksize=1):
            if r:
                out.append(r)
    return out


def _rules_with(overrides):
    base = json.load(open(BASE_RULES))
    conds = [c for c in base["conditions"] if c["knob"] not in overrides]
    for k, v in overrides.items():
        conds.append({"knob": k, "value": v})
    base["conditions"] = conds
    base["max_knobs"] = 12
    return base


def validate(args):
    """Is mid-season V a better objective than final money? Compare how many
    games each needs to separate two agents we KNOW differ."""
    seeds = [60000 + i for i in range(args.n)]
    print("running incumbent...", flush=True)
    a = _run(_rules_with({}), seeds, args.procs)
    print("running a knob we know is bad (+4 hands)...", flush=True)
    b = _run(_rules_with({"hires_delta": 4}), seeds, args.procs)
    print(f"\n{len(a)} and {len(b)} games\n")
    print(f"{'metric':<14} {'incumbent':>12} {'bad variant':>12} {'diff':>10} {'se':>8} {'|t|':>6}")
    rows = []
    for d in SCORE_DAYS:
        rows.append((f"V day {d}", [x[0].get(d, 0.5) for x in a], [x[0].get(d, 0.5) for x in b]))
    rows.append(("final margin", [x[1] for x in a], [x[1] for x in b]))
    rows.append(("win rate", [x[2] for x in a], [x[2] for x in b]))
    for name, xs, ys in rows:
        xs, ys = np.array(xs), np.array(ys)
        diff = xs.mean() - ys.mean()
        se = float(np.sqrt(xs.var(ddof=1) / len(xs) + ys.var(ddof=1) / len(ys)))
        t = abs(diff) / max(se, 1e-9)
        fmt = ",.0f" if "margin" in name else ".4f"
        print(f"{name:<14} {xs.mean():>12{fmt}} {ys.mean():>12{fmt}} {diff:>10{fmt}} {se:>8{fmt}} {t:>6.2f}")
    # correlation of each V with the actual outcome
    print("\ncorrelation with the game's own outcome (incumbent games):")
    win = np.array([x[2] for x in a])
    for d in SCORE_DAYS:
        v = np.array([x[0].get(d, 0.5) for x in a])
        print(f"  V day {d} vs win: {np.corrcoef(v, win)[0,1]:+.3f}   vs margin: "
              f"{np.corrcoef(v, np.array([x[1] for x in a]))[0,1]:+.3f}")
    print("\nA higher |t| for V than for final margin means V separates the same two agents")
    print("with fewer games, and the search below is worth running.")


def _score(results):
    """Objective = mean final margin.

    This was meant to be mid-season V, on the theory that a probability fitted on
    3,000 real games would separate agents with fewer games than a $20k-noisy
    bank. `validate` refuted that outright: separating the incumbent from a knob
    known to be bad (+4 hands) over 66 games a side gives |t| = 6.53 for final
    margin against 1.41 for the best V (day 16) and 0.08 at day 12. V predicts a
    single game's own outcome well (r = +0.90 with margin at day 16) but squashes
    into [0,1], compressing the tails where the between-agent signal lives. The
    head stays a diagnostic; the objective is margin."""
    if not results:
        return -1e9
    return float(np.mean([r[1] for r in results]))


def _vscore(results):
    if not results:
        return 0.0
    return float(np.mean([np.mean([r[0].get(d, 0.5) for d in SCORE_DAYS]) for r in results]))


def search(args):
    rng = random.Random(args.seed)
    seeds = [70000 + i for i in range(args.games)]
    keys = sorted(SPACE)
    incumbent = {}
    best = _run(_rules_with(incumbent), seeds, args.procs)
    best_score = _score(best)
    print(f"incumbent margin={best_score:+,.0f} V={_vscore(best):.3f} "
          f"win={np.mean([r[2] for r in best]):.2f}  ({len(best)} games)\n", flush=True)
    tried = {tuple(sorted(incumbent.items())): best_score}
    history = []
    for it in range(args.iters):
        # Coordinate move: perturb 1-3 knobs off the current best.
        cand = dict(incumbent)
        for k in rng.sample(keys, rng.randint(1, 3)):
            cand[k] = rng.choice(SPACE[k])
        key = tuple(sorted(cand.items()))
        if key in tried:
            continue
        res = _run(_rules_with(cand), seeds, args.procs)
        sc = _score(res)
        tried[key] = sc
        se = float(np.std([r[1] for r in res], ddof=1) / max(1, len(res) ** 0.5)) if len(res) > 1 else 0.0
        flag = ""
        if sc > best_score:
            incumbent, best_score, flag = cand, sc, "  <-- new best"
        history.append((sc, _vscore(res), dict(cand)))
        print(f"it {it:>3} margin={sc:>+9,.0f} (se {se:>6,.0f}) V={_vscore(res):.3f} "
              f"{ {k: v for k, v in cand.items()} }{flag}", flush=True)
    print(f"\nbest margin={best_score:+,.0f}: {incumbent}")
    history.sort(key=lambda h: -h[0])
    print("\ntop 5 by margin:")
    for sc, vv, cfg in history[:5]:
        print(f"  margin={sc:>+9,.0f} V={vv:.3f}  {cfg}")
    if history:
        v = np.array([h[0] for h in history])
        m = np.array([h[1] for h in history])
        print(f"\ncorrelation across configs between mean V and final margin: {np.corrcoef(m, v)[0,1]:+.3f}")
    out = args.out or "rl/datasets/plan_search.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"best": incumbent, "best_score": best_score,
               "history": [(s, float(m), c) for s, m, c in history]}, open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "search"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    {"validate": validate, "search": search}[args.cmd](args)


if __name__ == "__main__":
    main()
