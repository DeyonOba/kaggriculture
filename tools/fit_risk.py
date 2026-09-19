"""Fit the LEARNT half of the risk comparison: realised $/unit and its variance.

`tools/volatility.py` COMPUTES per-product variance from the engine: the shop
draw is eight uniform draws, price is closed form on inventory, so the
distribution is exact. This fits the same two quantities from the corpus
instead, so "computed" and "learnt" can be compared at identical functional
form and identical consumer.

Per product, from a day-d state, two ridges:

    mean model      realised $/unit over the rest of the game
    variance model  a second ridge on the SQUARED RESIDUAL of the first --
                    heteroscedastic risk, i.e. which states are unpredictable

The features are what a planner can see at decision time: day, shops open (and
how many draws remain), market inventory deviation, our standing count and the
opponent's, our shed.

    PYTHONPATH=. uv run python tools/fit_risk.py --n 300 --out /tmp/risk.py
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]



def _load_index():
    """See tools/opp_coupling.load_index -- the index is generated, not
    tracked, so it does not exist on a fresh Kaggle kernel."""
    from tools.opp_coupling import load_index
    return load_index()

def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, MARKET_PARAMS, SHOPS)
    import tools.price_forecast as pf

    s = json.load(open(path))
    pf.FLOW.clear()
    ref = pf._patch_flow()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    dawn = {}
    rev = collections.defaultdict(lambda: [0.0, 0])   # day -> [$ , units]
    while not env.done:
        o = env.state[0].observation
        d = o["step"] // 24
        ref["day"] = d
        if o["step"] % 24 == 0:
            dawn[d] = o
        env.step([a0(o), a1(env.state[1].observation)])

    # realised $/unit for SEAT 0 from day d onward, per product
    flow = pf.FLOW.get(0, {})
    X, Y, P, D = [], [], [], []
    for d in sorted(dawn):
        o = dawn[d]
        inv = o["market"]["inventory"]
        shops = list(o["town"]["unlocked_shops"])
        for pi, p in enumerate(PRODUCTS):
            units = revs = 0
            for dd, items in flow.items():
                if dd < d:
                    continue
                n = items.get(p, 0)
                if n > 0:
                    units += n
                    lvl = dawn.get(dd, o)["market"]["inventory"][p]
                    revs += n * __import__("kagent.econ", fromlist=["econ"]).price(p, lvl)
            if units < 3:
                continue
            base = MARKET_PARAMS[p]["base"]
            n_shop = sum(1 for sh in shops if p in SHOPS.get(sh, ()))
            mine = theirs = 0
            for seat, acc in ((0, "mine"), (1, "theirs")):
                c = 0
                for row in o["farms"][seat]["tiles"]:
                    for t in row:
                        if not isinstance(t, dict):
                            continue
                        if t.get("kind") == "PLANT" and t["crop"] == p:
                            c += 1
                        elif t.get("animal") and ANIMALS[t["animal"]]["product"] == p:
                            c += 1
                if seat == 0:
                    mine = c
                else:
                    theirs = c
            X.append([d / 29.0, len(shops) / 8.0, (8 - len(shops)) / 8.0,
                      n_shop / 4.0, (inv[p] - 10000) / 600.0,
                      mine / 30.0, theirs / 30.0,
                      (o.get("private", {}) or {}).get("shed", {}).get(p, 0) / 50.0,
                      1.0])
            Y.append(revs / units / base)         # realised, in units of base
            P.append(pi)
            D.append(d)
    if not X:
        return None
    return (np.asarray(X, "float32"), np.asarray(Y, "float32"),
            np.asarray(P), np.asarray(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random
    idx = _load_index()
    paths = [v["path"] for v in idx.values()
             if v["scores"] and all(x is not None for x in v["scores"])]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"replaying {len(paths)} sessions ...", flush=True)
    with mp.Pool(args.procs) as pool:
        out = [o for o in pool.map(job, paths) if o]
    X = np.concatenate([o[0] for o in out])
    Y = np.concatenate([o[1] for o in out])
    P = np.concatenate([o[2] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    print(f"{len(X):,} (state, product) samples from {len(out)} games\n")

    W, V = {}, {}
    gids = np.unique(G)
    print(f"    {'product':<12}{'n':>7}{'mean R2':>10}{'sd R2':>9}"
          f"{'E[ratio]':>10}{'sd[ratio]':>11}")
    for pi, p in enumerate(PRODUCTS):
        m = P == pi
        if m.sum() < 200:
            continue
        x, y, g = X[m], Y[m], G[m]
        pred = np.zeros_like(y)
        for f in range(5):
            te = np.isin(g, gids[f::5])
            tr = ~te
            if te.sum() == 0 or tr.sum() < x.shape[1]:
                continue
            A = x[tr].T @ x[tr] + 1.0 * np.eye(x.shape[1], dtype="float32")
            pred[te] = x[te] @ np.linalg.solve(A, x[tr].T @ y[tr])
        r2 = 1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())
        # variance model on squared residuals
        r = (y - pred) ** 2
        pv = np.zeros_like(r)
        for f in range(5):
            te = np.isin(g, gids[f::5])
            tr = ~te
            if te.sum() == 0 or tr.sum() < x.shape[1]:
                continue
            A = x[tr].T @ x[tr] + 1.0 * np.eye(x.shape[1], dtype="float32")
            pv[te] = x[te] @ np.linalg.solve(A, x[tr].T @ r[tr])
        r2v = 1 - ((r - pv) ** 2).sum() / max(1e-9, ((r - r.mean()) ** 2).sum())
        A = x.T @ x + 1.0 * np.eye(x.shape[1], dtype="float32")
        W[p] = np.linalg.solve(A, x.T @ y).tolist()
        V[p] = np.linalg.solve(A, x.T @ r).tolist()
        print(f"    {p:<12}{int(m.sum()):>7}{r2:>10.3f}{r2v:>9.3f}"
              f"{y.mean():>10.2f}{y.std():>11.2f}")

    src = ['"""Learnt risk model: realised $/unit and its variance (generated)."""\n',
           "_RISK_W = %r" % (W,), "_RISK_V = %r" % (V,)]
    open(args.out, "w").write("\n".join(src))
    print(f"\n  -> {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
