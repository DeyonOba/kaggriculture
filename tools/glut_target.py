"""Day-of-glut: does the threshold-crossing target have signal before we fit it?

S4's lesson is that a better PATH did not make a better decision, because
`_rent_pick` reads one threshold off that path and `econ.sell_revenue` is
convex -- a smoothed path is biased exactly at the crossing. So predict the
crossing directly.

Target, per product, from day d: how many days until market inventory first
exceeds I0 (the point where price equals base and past which it falls),
right-censored at CAP.

Two things have to be true before a head is worth building, and both are
cheap to check:

  1. VARIANCE. If a product gluts on the same day every game, or never, there
     is nothing to learn and the rent rule can hard-code it.
  2. PREDICTABILITY ABOVE A TRIVIAL BASELINE. Ridge on standing counts is the
     same bar S2 used. A CNN only earns its place by beating it.

    PYTHONPATH=. uv run python tools/glut_target.py --n 200
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
OUT = os.path.join(REPO, "rl/datasets/glut.npz")

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
I0 = 10000
CAP = 15                 # right-censoring horizon, in days




def _load_index():
    """See tools/opp_coupling.load_index -- the index is generated, not
    tracked, so it does not exist on a fresh Kaggle kernel."""
    from tools.opp_coupling import load_index
    return load_index()

def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.board_tensor import board_pair_full
    import tools.price_forecast as pf

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    env.run([a0, a1])

    dawn, mid = {}, {}
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
        elif o["step"] % 24 == 12:
            mid[o["step"] // 24] = o
    days = sorted(dawn)
    inv = {p: [dawn[d]["market"]["inventory"][p] for d in days] for p in PRODUCTS}

    T, Y, C, D = [], [], [], []
    for i, d in enumerate(days):
        if d + CAP > days[-1]:
            break
        T.append(board_pair_full(dawn[d], 0, prev_mid=mid.get(d - 1)))
        y, c = [], []
        for p in PRODUCTS:
            hit = CAP
            for j in range(1, CAP + 1):
                if inv[p][i + j] > I0:
                    hit = j
                    break
            y.append(float(hit))
            c.append(1.0 if hit < CAP else 0.0)      # observed, not censored
        Y.append(y)
        C.append(c)
        D.append(d)
    if not T:
        return None
    return (np.stack(T), np.asarray(Y, "float32"), np.asarray(C, "float32"),
            np.asarray(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    idx = _load_index()
    paths = [v["path"] for v in idx.values()
             if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"replaying {len(paths)} sessions ...", flush=True)
    with mp.Pool(args.procs) as pool:
        out = [o for o in pool.map(job, paths) if o]
    T = np.concatenate([o[0] for o in out])
    Y = np.concatenate([o[1] for o in out])
    C = np.concatenate([o[2] for o in out])
    D = np.concatenate([o[3] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    np.savez_compressed(OUT, T=T, Y=Y, C=C, D=D, G=G)
    print(f"{len(T):,} samples from {len(out)} games -> {OUT}\n")

    # --- 1. does the target vary at all? ---------------------------------
    print("  days until glut, right-censored at 15 (a value of 15 means "
          "'not within 15 days')\n")
    print(f"    {'product':<12}{'mean':>7}{'sd':>7}{'p10':>6}{'p50':>6}"
          f"{'p90':>6}{'% observed':>12}{'% already glutted':>19}")
    for pi, p in enumerate(PRODUCTS):
        y = Y[:, pi]
        obs = C[:, pi].mean()
        zero = (y == 0).mean()
        print(f"    {p:<12}{y.mean():>7.1f}{y.std():>7.1f}"
              f"{np.percentile(y,10):>6.0f}{np.percentile(y,50):>6.0f}"
              f"{np.percentile(y,90):>6.0f}{100*obs:>11.0f}%{100*zero:>18.0f}%")

    # --- 2. is it predictable above a trivial baseline? -------------------
    from tools.board_tensor import arm_channels
    cols = arm_channels("E_full_83")
    # ridge on pooled channel means -- the cheap "counts" equivalent
    X = T[:, cols].mean(axis=(2, 3))
    X = np.hstack([X, D[:, None].astype("float32"), np.ones((len(X), 1), "float32")])
    gids = np.unique(G)
    pred = np.zeros_like(Y)
    for f in range(5):
        te = np.isin(G, gids[f::5])
        tr = ~te
        A = X[tr].T @ X[tr] + 1.0 * np.eye(X.shape[1], dtype="float32")
        W = np.linalg.solve(A, X[tr].T @ Y[tr])
        pred[te] = X[te] @ W
    print(f"\n  ridge on pooled channel means + day, held out BY GAME:\n")
    print(f"    {'product':<12}{'R2':>8}{'MAE (days)':>12}"
          f"{'MAE of always-predicting-the-mean':>36}")
    for pi, p in enumerate(PRODUCTS):
        y, q = Y[:, pi], pred[:, pi]
        ss = ((y - q) ** 2).sum()
        tt = ((y - y.mean()) ** 2).sum()
        print(f"    {p:<12}{1 - ss/max(tt,1e-9):>8.3f}"
              f"{np.abs(y-q).mean():>12.2f}"
              f"{np.abs(y-y.mean()).mean():>36.2f}")


if __name__ == "__main__":
    main()
