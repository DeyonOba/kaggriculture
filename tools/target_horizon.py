"""How far back must we look to predict the OPPONENT'S TARGETS?

`kaggriculture-state-is-markov` closed the history question for PRICE: k days
of board history bought $1 of $29 and depth was flat. That memo says not to
re-open it without a different prediction target. This is a different target.

Price is Markov because the engine integrates history into the observation
(`consecutive_unwatered`, `yield_units`, cumulative market inventory). An
opponent's TARGET is not in the observation at all -- `straw_target`, `cows`,
`sheep`, `wheat_engine` are numbers inside their code. What is observable is
the ramp toward them, and a ramp is only visible over time. So history should
matter here even though it did not there, and this measures how much.

The label is what the forward model actually needs: the opponent's standing
count of each crop and animal H days ahead, which determines their supply.

    PYTHONPATH=. uv run python tools/target_horizon.py --n 80 --horizon 10
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")

KINDS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "COW", "SHEEP", "GOOSE"]
DEPTHS = (0, 1, 2, 3, 5, 10)      # days of history; 0 = current frame only



def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def _script(actions):
    st = {"t": 0}

    def agent(obs):
        t = obs.get("step", st["t"])
        st["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def standing(farm):
    c = dict.fromkeys(KINDS, 0)
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                c[t["crop"]] = c.get(t["crop"], 0) + 1
            elif t.get("animal"):
                c[t["animal"]] = c.get(t["animal"], 0) + 1
    return c


def job(path):
    """Per-day standing counts for both seats, plus context."""
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([_script(s["actions"][0]), _script(s["actions"][1])])
    days = []
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24:
            continue
        days.append({
            "seat": [standing(o["farms"][i]) for i in (0, 1)],
            "money": [o["farms"][i]["money"] for i in (0, 1)],
            "quads": [len(o["farms"][i]["unlocked_quadrants"]) for i in (0, 1)],
            "shops": len(o["town"]["unlocked_shops"]),
            "inv": dict(o["market"]["inventory"]),
        })
    return {"episode": s["episode"], "days": days,
            "scores": s.get("scores"), "finals": s["finals"]}


def build(rows, depth, horizon, obs_seat=0):
    """X, y for predicting the OTHER seat's standing counts `horizon` ahead."""
    X, Y, G = [], [], []
    tgt = 1 - obs_seat
    for gi, r in enumerate(rows):
        d = r["days"]
        for t in range(depth, len(d) - horizon):
            f = []
            # their board now and at each earlier lag
            for lag in range(depth + 1):
                f += [d[t - lag]["seat"][tgt][k] for k in KINDS]
            # context available to us at time t
            f += [d[t]["money"][tgt] / 1000.0, float(d[t]["quads"][tgt]),
                  float(d[t]["shops"]), float(t)]
            f.append(1.0)
            X.append(f)
            Y.append([d[t + horizon]["seat"][tgt][k] for k in KINDS])
            G.append(gi)
    return np.array(X, float), np.array(Y, float), np.array(G)


def ridge_cv(X, Y, G, lam=1.0, folds=5):
    """R^2 held out BY GAME, per target column."""
    gids = np.unique(G)
    pred = np.zeros_like(Y)
    for f in range(folds):
        te = np.isin(G, gids[f::folds])
        tr = ~te
        if te.sum() == 0 or tr.sum() < X.shape[1]:
            continue
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, X[tr].T @ Y[tr])
        pred[te] = X[te] @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-9), np.abs(Y - pred).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--cache", default="")
    args = ap.parse_args()

    import random
    if args.cache and os.path.exists(args.cache):
        rows = json.load(open(args.cache))
        print(f"{len(rows)} sessions from cache\n", flush=True)
    else:
        idx = json.load(open(INDEX))
        paths = [v["path"] for v in idx.values()
                 if _rating_ok(v, args.min_rating)]
        random.Random(args.seed).shuffle(paths)
        paths = paths[: args.n]
        print(f"replaying {len(paths)} sessions rated >= {args.min_rating:.0f}"
              f"...\n", flush=True)
        with mp.Pool(args.procs) as pool:
            rows = pool.map(job, paths)
        if args.cache:
            json.dump(rows, open(args.cache, "w"))

    for H in args.horizon:
        print(f"=== predicting THEIR standing counts {H} days ahead "
              f"({H*24} turns) ===\n")
        print(f"    {'history':<22}{'mean R2':>9}{'  per-kind R2 '
                                                 '(wheat carrot tom straw melon cow sheep goose)'}")
        base = None
        for depth in DEPTHS:
            X, Y, G = build(rows, depth, H)
            r2, mae = ridge_cv(X, Y, G)
            m = r2.mean()
            if base is None:
                base = m
            lbl = ("current frame only" if depth == 0
                   else f"+ {depth} day{'s' if depth > 1 else ''} back")
            print(f"    {lbl:<22}{m:>9.3f}   "
                  + " ".join(f"{v:>5.2f}" for v in r2)
                  + (f"   {100*(m-base)/max(1e-9,abs(base)):+.0f}%" if depth else ""))
        print()


if __name__ == "__main__":
    main()
