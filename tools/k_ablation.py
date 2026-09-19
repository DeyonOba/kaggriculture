"""How far back to look, in TURNS and in DAYS, for board AND market.

`tools/target_horizon.py` swept history in DAYS only, because it sampled at
dawn. Turn-level lags (5, 10, 24, 48) are a different and untested question:
24 turns is one day, so 5 and 10 are INTRA-day, which dawn sampling cannot
see at all.

Two history sources, swept independently so their contributions separate:

    BOARD    the opponent's standing counts per crop and animal at each lag
    MARKET   inventory deviation from I0 per product at each lag -- the
             forward model uses only the CURRENT inventory, so its trajectory
             has never been an input to anything

Target is what the forward model consumes: the opponent's standing counts
H days ahead. Held out BY GAME, ridge, so the comparison is about information
content rather than model capacity.

Everything is sampled EVERY TURN and lagged afterwards, so turn-lags and
day-lags come from one pass and are exactly comparable.

    PYTHONPATH=. uv run python tools/k_ablation.py --n 60 --horizon 5 10
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")

KINDS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "COW", "SHEEP", "GOOSE"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

# Lag sets, all expressed in TURNS. 24 turns = 1 day.
#
# Swept as two GROUPS, not one table. A row's usable window starts at its
# deepest lag, so scoring every row over one window would cost the shallow
# turn-lags 10 of their 25 samples per game to accommodate d10. Each group is
# instead scored over the window its own deepest member allows, and carries
# its OWN no-history baseline computed on that same window -- so comparisons
# are exact within a group, and across groups only the deltas are comparable.
GROUPS = [
    ("TURNS  (5, 10, 24, 48 -- 24 turns = 1 day)", {
        "none": (), "t5": (5,), "t10": (10,), "t24": (24,), "t48": (48,),
        "t5+10": (5, 10), "t5+10+24": (5, 10, 24), "t24+48": (24, 48),
        "t5+10+24+48": (5, 10, 24, 48)}),
    ("DAYS   (1, 3, 5, 10)", {
        "none": (), "d1": (24,), "d3": (72,), "d5": (120,), "d10": (240,),
        "d1+3": (24, 72), "d1+3+10": (24, 72, 240),
        "d1+3+5+10": (24, 72, 120, 240)}),
]


def _rating_ok(v, floor):
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


def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([_script(s["actions"][0]), _script(s["actions"][1])])

    board, market = [], []
    for st_ in env.steps:
        o = st_[0].observation
        row = []
        for seat in (0, 1):
            c = dict.fromkeys(KINDS, 0)
            for r in o["farms"][seat]["tiles"]:
                for t in r:
                    if not isinstance(t, dict):
                        continue
                    if t.get("kind") == "PLANT":
                        c[t["crop"]] = c.get(t["crop"], 0) + 1
                    elif t.get("animal"):
                        c[t["animal"]] = c.get(t["animal"], 0) + 1
            row += [c[k] for k in KINDS]
        board.append(row)
        inv = o["market"]["inventory"]
        market.append([(inv[p] - 10000) / 600.0 for p in PRODUCTS])
    return np.asarray(board, "float32"), np.asarray(market, "float32")


def design(games, lags, horizon_days, back, use_board=True, use_market=True):
    """X, Y, G over every DAWN turn from `back` on.

    The BASE features -- their counts now, market now, day -- are present in
    every arm. The only thing the flags change is which HISTORY is appended,
    so a column difference is attributable to that history and nothing else,
    and the `none` row is identical across all three columns by construction.
    """
    H = horizon_days * 24
    K = len(KINDS)
    X, Y, G = [], [], []
    for gi, (b, m) in enumerate(games):
        for t in range(back, len(b) - H, 24):     # dawn turns only
            f = list(b[t][K:]) + list(m[t]) + [t / 720.0, 1.0]
            if use_board:
                for lag in lags:
                    f += list(b[t - lag][K:])
            if use_market:
                for lag in lags:
                    f += list(m[t - lag])
            X.append(f)
            Y.append(list(b[t + H][K:]))
            G.append(gi)
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(G)


def ridge_by_game(X, Y, G, lam=1.0, folds=5):
    gids = np.unique(G)
    pred = np.zeros_like(Y)
    for f in range(folds):
        te = np.isin(G, gids[f::folds])
        tr = ~te
        if te.sum() == 0 or tr.sum() < X.shape[1]:
            continue
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        pred[te] = X[te] @ np.linalg.solve(A, X[tr].T @ Y[tr])
    ss = ((Y - pred) ** 2).sum(0)
    tt = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss / np.maximum(tt, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values() if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"replaying {len(paths)} sessions, sampling EVERY turn ...\n",
          flush=True)
    with mp.Pool(args.procs) as pool:
        games = pool.map(job, paths)

    for Hd in args.horizon:
        print(f"{'='*78}\n=== target: THEIR standing counts {Hd} days "
              f"({Hd*24} turns) ahead\n{'='*78}")
        for title, lagsets in GROUPS:
            back = max((max(v) for v in lagsets.values() if v), default=0)
            n_s = len(design(games, (), Hd, back)[0])
            print(f"\n  {title}      window starts turn {back}, "
                  f"{n_s} samples\n")
            print(f"    {'history':<14}{'k':>3}{'+board hist':>13}"
                  f"{'+market hist':>14}{'+both':>10}{'vs none':>10}")
            rows, per_kind = {}, {}
            for name, lags in lagsets.items():
                cells = []
                for ub, um in ((True, False), (False, True), (True, True)):
                    X, Y, G = design(games, lags, Hd, back, ub, um)
                    if len(X) < 150:
                        cells.append(float("nan"))
                        continue
                    r2 = ridge_by_game(X, Y, G)
                    if (ub and um) or not lags:
                        per_kind[name] = r2
                    cells.append(float(r2.mean()))
                rows[name] = cells
                base = rows["none"][0]
                d = "" if name == "none" else f"{cells[2]-base:>+10.3f}"
                print(f"    {name:<14}{len(lags):>3}"
                      + "".join(f"{c:>13.3f}" if i < 2 else f"{c:>10.3f}"
                                for i, c in enumerate(cells))
                      + d, flush=True)
            base = rows["none"][0]
            best = max(rows.items(), key=lambda kv: kv[1][2])
            print(f"\n    no-history baseline {base:.3f}  ->  best is "
                  f"'{best[0]}' at {best[1][2]:.3f} "
                  f"({100*(best[1][2]-base)/max(1e-9,abs(base)):+.1f}%)")
            print(f"    per-kind R2   "
                  + "  ".join(f"{k[:4].lower()}" for k in KINDS))
            for nm in ("none", best[0]):
                if nm in per_kind:
                    print(f"      {nm:<11} "
                          + "  ".join(f"{v:>4.2f}" for v in per_kind[nm]))
        print()


if __name__ == "__main__":
    main()
