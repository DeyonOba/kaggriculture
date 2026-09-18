"""S2: does the BOARD beat plain COUNTS at predicting the opponent's targets?

The kill criterion for the whole board-encoder premise. If a CNN over the
two-farm tensor cannot beat a ridge over standing counts, then spatial
structure carries nothing the counts do not, and no amount of channels or
depth will change that.

Both models see the SAME MOMENT -- one frame, no history -- so the comparison
isolates representation, not information. Label is the opponent's standing
count of each crop and animal H days ahead, which is what the forward market
model needs (`tools/price_forecast.py` L4).

Held out BY GAME, never by sample: consecutive days of one game are almost
the same board, so a random split would leak and report a fantasy.

    PYTHONPATH=. uv run python tools/s2_target_cnn.py --build --n 300
    PYTHONPATH=. uv run python tools/s2_target_cnn.py --train --horizon 5 10
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
OUT = os.path.join(REPO, "rl/datasets/s2_targets.npz")

KINDS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "COW", "SHEEP", "GOOSE"]


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
    from tools.board_tensor import board_pair, counts

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([_script(s["actions"][0]), _script(s["actions"][1])])

    T, C, L = [], [], []
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24:
            continue
        T.append(board_pair(o, 0))
        C.append(counts(o, 0))
        farm = o["farms"][1]
        c = dict.fromkeys(KINDS, 0)
        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    c[t["crop"]] = c.get(t["crop"], 0) + 1
                elif t.get("animal"):
                    c[t["animal"]] = c.get(t["animal"], 0) + 1
        L.append([c[k] for k in KINDS])
    return (np.stack(T).astype("float32"), np.stack(C).astype("float32"),
            np.asarray(L, dtype="float32"))


def build(args):
    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if v["scores"] and all(x is not None for x in v["scores"])
             and min(v["scores"]) >= args.min_rating]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"replaying {len(paths)} sessions rated >= {args.min_rating:.0f} ...",
          flush=True)
    with mp.Pool(args.procs) as pool:
        out = pool.map(job, paths)
    T = np.concatenate([o[0] for o in out])
    C = np.concatenate([o[1] for o in out])
    L = np.concatenate([o[2] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    D = np.concatenate([np.arange(len(o[0])) for o in out])
    np.savez_compressed(OUT, T=T, C=C, L=L, G=G, D=D)
    print(f"{T.shape[0]:,} day-samples from {len(out)} games   "
          f"tensor {T.shape}   -> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")


def ridge_by_game(X, Y, G, lam=1.0, folds=5):
    gids = np.unique(G)
    pred = np.zeros_like(Y)
    Xb = np.hstack([X, np.ones((len(X), 1), "float32")])
    for f in range(folds):
        te = np.isin(G, gids[f::folds])
        tr = ~te
        A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1], dtype="float32")
        W = np.linalg.solve(A, Xb[tr].T @ Y[tr])
        pred[te] = Xb[te] @ W
    return pred


def cnn_by_game(T, Y, G, folds=5, epochs=40, seed=0):
    """Targets are z-scored PER KIND before the loss.

    Without it the joint MSE is dominated by the high-variance crop columns
    (standing counts reach 33) and the animal heads (0-12, and near-constant
    for agents running a fixed herd knob) are starved: the first run had the
    CNN beating ridge on all five crops and collapsing to R2 0.17/0.24/0.18 on
    cow/sheep/goose. That was a loss-balance defect, not a representation one.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    gids = np.unique(G)
    pred = np.zeros_like(Y)
    C = T.shape[1]
    for f in range(folds):
        te = np.isin(G, gids[f::folds])
        tr = ~te
        mu = Y[tr].mean(0)
        sd = np.maximum(Y[tr].std(0), 1e-3)
        net = nn.Sequential(
            nn.Conv2d(C, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, Y.shape[1]))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        Xtr = torch.from_numpy(T[tr])
        Ytr = torch.from_numpy(((Y[tr] - mu) / sd).astype("float32"))
        n = len(Xtr)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xtr[b]), Ytr[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred[te] = net(torch.from_numpy(T[te])).numpy() * sd + mu
    return pred


def r2(Y, P):
    ss = ((Y - P) ** 2).sum(0)
    tt = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss / np.maximum(tt, 1e-9)


def train(args):
    d = np.load(OUT)
    T, C, L, G, D = d["T"], d["C"], d["L"], d["G"], d["D"]
    print(f"{len(T):,} day-samples, {len(np.unique(G))} games, "
          f"tensor {T.shape[1:]}\n")
    for H in args.horizon:
        # pair day d with day d+H inside the same game
        keep, tgt = [], []
        pos = {(g, dd): i for i, (g, dd) in enumerate(zip(G, D))}
        for i, (g, dd) in enumerate(zip(G, D)):
            j = pos.get((g, dd + H))
            if j is not None:
                keep.append(i)
                tgt.append(j)
        keep = np.asarray(keep)
        Y = L[np.asarray(tgt)]
        Ti, Ci, Gi = T[keep], C[keep], G[keep]
        print(f"=== H = {H} days ({H*24} turns), {len(keep):,} samples ===")
        pr = ridge_by_game(Ci, Y, Gi)
        pc = cnn_by_game(Ti, Y, Gi, epochs=args.epochs)
        rr, rc = r2(Y, pr), r2(Y, pc)
        print(f"    {'model':<22}{'mean R2':>9}   " +
              " ".join(f"{k[:5]:>6}" for k in KINDS))
        print(f"    {'ridge on counts':<22}{rr.mean():>9.3f}   " +
              " ".join(f"{v:>6.2f}" for v in rr))
        print(f"    {'CNN on board pair':<22}{rc.mean():>9.3f}   " +
              " ".join(f"{v:>6.2f}" for v in rc))
        delta = rc.mean() - rr.mean()
        print(f"\n    CNN - ridge = {delta:+.3f}"
              f"   -> {'CNN WINS' if delta > 0.02 else 'KILL: no spatial gain'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horizon", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()
    if args.build:
        build(args)
    if args.train:
        train(args)


if __name__ == "__main__":
    main()
