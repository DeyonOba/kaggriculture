"""S3: does a learned opponent-supply model move the FORECAST?

S2 showed the board beats counts at predicting the opponent's standing counts.
That is a proxy. The quantity the forward market model actually consumes is
their NET MARKET FLOW per product per day -- exactly what the L8 oracle
supplies, and the oracle bounds what any model can win:

    board-projected supply (today)   k=5 MAE 13.0
    PERFECT opponent supply          k=5 MAE 10.4

So S3 predicts the oracle's own quantity and is scored on the same axis.

    KILL unless the model closes at least HALF of 13.0 -> 10.4, i.e. reaches
    11.7 at k=5. Below that it cannot matter downstream.

Targets: 9 products x H days of net flow (sells minus buys), z-scored per
output. Auxiliary: the opponent's per-tile crop occupancy at day t+H -- dense
spatial supervision, 100 gradients a sample instead of 90 scalars.

    PYTHONPATH=. uv run python tools/s3_flow.py --build --n 400
    PYTHONPATH=. uv run python tools/s3_flow.py --train
    PYTHONPATH=. uv run python tools/s3_flow.py --eval --n 40
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
DATA = os.path.join(REPO, "rl/datasets/s3_flow.npz")
MODEL = os.path.join(REPO, "rl/datasets/boardnet_flow.npz")

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
H = 10                      # days of flow predicted
EVAL_K = (1, 3, 5, 10)


def _pick(n, min_rating, seed, skip=0):
    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if v["scores"] and all(x is not None for x in v["scores"])
             and min(v["scores"]) >= min_rating]
    random.Random(seed).shuffle(paths)
    return paths[skip: skip + n]


def job(path):
    """Tensors, true opponent flow for the next H days, and the aux mask."""
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.board_tensor import board_pair
    import tools.price_forecast as pf

    s = json.load(open(path))
    pf.FLOW.clear()
    ref = pf._patch_flow()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    dawn = {}
    while not env.done:
        o = env.state[0].observation
        ref["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
        env.step([a0(o), a1(env.state[1].observation)])

    flow = pf.FLOW.get(1, {})
    T, F, A, D = [], [], [], []
    for day in sorted(dawn):
        if day + H not in dawn:
            continue
        T.append(board_pair(dawn[day], 0))
        f = np.zeros((H, len(PRODUCTS)), dtype="float32")
        for j in range(1, H + 1):
            d = flow.get(day + j, {})
            for pi, p in enumerate(PRODUCTS):
                f[j - 1, pi] = d.get(p, 0)
        F.append(f.reshape(-1))
        fut = dawn[day + H]["farms"][1]["tiles"]
        A.append(np.asarray(
            [[1.0 if (isinstance(t, dict) and t.get("kind") == "PLANT") else 0.0
              for t in row] for row in fut], dtype="float32"))
        D.append(day)
    if not T:
        return None
    return (np.stack(T), np.stack(F), np.stack(A), np.asarray(D))


def build(args):
    paths = _pick(args.n, args.min_rating, args.seed)
    print(f"replaying {len(paths)} sessions ...", flush=True)
    with mp.Pool(args.procs) as pool:
        out = [o for o in pool.map(job, paths) if o]
    T = np.concatenate([o[0] for o in out])
    F = np.concatenate([o[1] for o in out])
    A = np.concatenate([o[2] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    np.savez_compressed(DATA, T=T, F=F, A=A, G=G)
    print(f"{len(T):,} samples from {len(out)} games   tensor {T.shape[1:]}   "
          f"flow {F.shape[1]} outputs ({H} days x {len(PRODUCTS)} products)   "
          f"-> {DATA} ({os.path.getsize(DATA)/1e6:.0f} MB)")


def train(args):
    from tools.boardnet import fit, predict, save
    d = np.load(DATA)
    T, F, A, G = d["T"], d["F"], d["A"], d["G"]
    print(f"{len(T):,} samples, {len(np.unique(G))} games\n")
    net, stats, hist = fit(T, F, G, A=A, epochs=args.epochs, seed=args.seed,
                           width=args.width, blocks=args.blocks)
    save(MODEL, net, stats)
    print(f"    best val MSE {stats['val_mse']:.4f} after {len(hist)} epochs")
    print(f"    -> {MODEL} ({os.path.getsize(MODEL)/1e6:.1f} MB)")

    # held-out R2 per product, summed over the H days (what the forecast uses)
    gids = np.unique(G)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(gids)
    va = np.isin(G, gids[:max(1, len(gids) // 5)])
    P = predict(net, stats, T[va])
    Yv = F[va]
    print(f"\n    held-out R2 on TOTAL {H}-day flow, per product:")
    n_p = len(PRODUCTS)
    for pi, p in enumerate(PRODUCTS):
        cols = [j * n_p + pi for j in range(H)]
        y = Yv[:, cols].sum(1)
        q = P[:, cols].sum(1)
        r2 = 1 - ((y - q) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())
        print(f"      {p:<12}{r2:>7.3f}")


def eval_job(spec):
    """Forecast MAE with board-projected / model / oracle opponent supply."""
    path, use_model = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, CROPS, PRODUCTS as ENGP, SHOPS, TOWN_CENTER_PRODUCTS,
        market_price)
    import tools.price_forecast as pf
    from tools.board_tensor import board_pair
    K = {"SHOPS": SHOPS, "TOWN": TOWN_CENTER_PRODUCTS, "CROPS": CROPS,
         "ANIMALS": ANIMALS, "PRODUCTS": ENGP, "market_price": market_price}

    net = stats = None
    if use_model:
        from tools.boardnet import load, predict
        net, stats = load(MODEL)

    s = json.load(open(path))
    pf.FLOW.clear()
    ref = pf._patch_flow()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    dawn = {}
    while not env.done:
        o = env.state[0].observation
        ref["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
        env.step([a0(o), a1(env.state[1].observation)])
    flow = pf.FLOW.get(1, {})

    err = {m: {k: [] for k in EVAL_K} for m in ("board", "model", "oracle")}
    n_p = len(PRODUCTS)
    for day in sorted(dawn):
        if day + H > max(dawn):
            break
        truth = {p: [dawn[day + k]["market"]["prices"][p] for k in range(H + 1)]
                 for p in ENGP}
        oracle = [{}] + [dict(flow.get(day + j, {})) for j in range(1, H + 1)]
        runs = {"board": None, "oracle": oracle}
        if net is not None:
            from tools.boardnet import predict
            t = board_pair(dawn[day], 0)[None].astype("float32")
            q = predict(net, stats, t)[0].reshape(H, n_p)
            runs["model"] = [{}] + [
                {p: float(q[j - 1, pi]) for pi, p in enumerate(PRODUCTS)}
                for j in range(1, H + 1)]
        for name, oo in runs.items():
            f = pf.forecast(dawn[day], 0, H, 5, K, opp_oracle=oo)
            for k in EVAL_K:
                for p in ENGP:
                    err[name][k].append(abs(f[p][k] - truth[p][k]))
    return {m: {k: (sum(v), len(v)) for k, v in dd.items()}
            for m, dd in err.items()}


def evaluate(args):
    # held-out games: skip the block used for training
    paths = _pick(args.n, args.min_rating, args.seed, skip=args.skip)
    print(f"{len(paths)} HELD-OUT sessions (skipping the first {args.skip} "
          f"used for training)\n", flush=True)
    with mp.Pool(args.procs) as pool:
        rows = pool.map(eval_job, [(p, True) for p in paths])
    print(f"    {'opponent supply from':<28}"
          + "".join(f"{'k='+str(k):>9}" for k in EVAL_K))
    got = {}
    for m, lbl in (("board", "their board (arithmetic)"),
                   ("model", "BoardNet"),
                   ("oracle", "PERFECT (the bound)")):
        cells = []
        for k in EVAL_K:
            tot = sum(r[m][k][0] for r in rows)
            n = sum(r[m][k][1] for r in rows)
            cells.append(tot / max(1, n))
        got[m] = cells
        print(f"    {lbl:<28}" + "".join(f"{c:>9.1f}" for c in cells))
    i = EVAL_K.index(5)
    b, mo, o = got["board"][i], got["model"][i], got["oracle"][i]
    frac = (b - mo) / max(1e-9, b - o)
    print(f"\n    at k=5 the model closes {100*frac:.0f}% of the "
          f"{b:.1f} -> {o:.1f} gap"
          f"   -> {'PASS' if frac >= 0.5 else 'KILL (needs >= 50%)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--skip", type=int, default=400)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    args = ap.parse_args()
    if args.build:
        build(args)
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
