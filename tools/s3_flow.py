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

Targets: 9 products x H days, z-scored per output. `F` is net flow as before;
`C` and `B` are the gross SELL and BUY series, kept apart because netting them
erased 18% of all market commits (see price_forecast.SIDES). The buy series is
live for WHEAT and FERTILIZER only -- the other seven products are never
bought -- so a buy head must be read per product and dead columns excluded
from any headline mean.

All three come from the replay oracle and none is observable at play time.
They are PRIVILEGED TARGETS: the model is trained to predict them from public
state and at inference simply runs that prediction, so nothing has to be
switched off.

Auxiliary: the opponent's per-tile crop occupancy at day t+H -- dense
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

# Capacity is measured down to 60% of base, matching market_capacity.py.
#
# Scaling is LOG, not linear, and the smoke test is why. Capacity spans 33
# units (milk) to unbounded (wheat, egg) in the same frame, so a linear scale
# put seven products inside [0.03, 0.2] and pinned two at the ceiling with
# zero variance -- six of the nine channels were numerically dead.
#
# The saturation is real rather than an artifact: units_sellable_above walks
# one unit at a time and wheat's price never falls below 60% of base at any
# inventory, so its capacity is genuinely infinite. That is the annuity/trap
# split the market-structure work already found. CAP_MAX is therefore a
# SEARCH ceiling, not a clamp on a real number -- hitting it means unbounded,
# and log1p maps that to 1.0 while giving the crowded low end real
# resolution. Kept at 2000 because the search is O(cap) and an unbounded
# product runs it to completion on every call.
CAP_FRAC = 0.6
CAP_MAX = 2000
REGEN_SCALE = 20.0
EVAL_K = (1, 3, 5, 10)



def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def _pick(n, min_rating, seed, skip=0):
    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if _rating_ok(v, min_rating)]
    random.Random(seed).shuffle(paths)
    return paths[skip: skip + n]


def job(path):
    """Tensors, true opponent flow for the next H days, and the aux mask."""
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.price_forecast as pf

    s = json.load(open(path))
    pf.FLOW.clear()
    ref = pf._patch_flow()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    dawn, mid, oshed = {}, {}, {}
    while not env.done:
        o = env.state[0].observation
        ref["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
            # their shed: hidden at play time, available here as an AUX LABEL
            sh = (env.state[1].observation.get("private") or {}).get("shed") or {}
            oshed[o["step"] // 24] = [float(sh.get(p, 0)) for p in PRODUCTS]
        elif o["step"] % 24 == 12:
            mid[o["step"] // 24] = o
        env.step([a0(o), a1(env.state[1].observation)])

    flow = pf.FLOW.get(1, {})
    sides = pf.SIDES.get(1, {})

    # --- market capacity per product per dawn, the L2 sell-timing input ----
    # Exact, not estimated: units sellable before price falls below the floor.
    from kagent import econ
    from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
    from tools.market_clock import daily_drain
    from tools.board_tensor import LAGS_DEEP, board_pair_all
    floors = {p: max(1, int(CAP_FRAC * MARKET_PARAMS[p]["base"]))
              for p in PRODUCTS}
    lg = np.log1p(CAP_MAX)
    capday = {}
    for d, o in dawn.items():
        inv = o["market"]["inventory"]
        capday[d] = np.log1p(np.asarray(
            [econ.units_sellable_above(p, inv[p], floors[p], cap=CAP_MAX)
             for p in PRODUCTS], dtype="float32")) / lg
    last = dawn[max(dawn)]
    regen = np.asarray([daily_drain(last["town"]["unlocked_shops"], p)
                        for p in PRODUCTS], dtype="float32") / REGEN_SCALE

    T, F, A, D, S, C, B = [], [], [], [], [], [], []
    for day in sorted(dawn):
        if day + H not in dawn:
            continue
        # most-recent-first list of PAST dawns, for the temporal opponent stack
        past = [dawn[d] for d in range(day - 1, -1, -1) if d in dawn]
        # capacity at each lag, clamped to day 0 rather than zero-padded --
        # same convention as temporal_opp, so early game reads "unchanged"
        cl = np.stack([capday[max(0, day - L)] for L in LAGS_DEEP])
        T.append(board_pair_all(dawn[day], 0, prev_mid=mid.get(day - 1),
                                hist=past, cap_now=capday[day],
                                cap_lags=cl, regen=regen))
        S.append(oshed.get(day, [0.0] * len(PRODUCTS)))
        f = np.zeros((H, len(PRODUCTS)), dtype="float32")
        b = np.zeros((H, len(PRODUCTS)), dtype="float32")
        c = np.zeros((H, len(PRODUCTS)), dtype="float32")
        for j in range(1, H + 1):
            d = flow.get(day + j, {})
            e = sides.get(day + j, {})
            for pi, p in enumerate(PRODUCTS):
                f[j - 1, pi] = d.get(p, 0)                    # net, as before
                c[j - 1, pi] = e.get(p, {}).get("sell", 0)    # gross sells
                b[j - 1, pi] = e.get(p, {}).get("buy", 0)     # gross buys
        F.append(f.reshape(-1))
        C.append(c.reshape(-1))
        B.append(b.reshape(-1))
        fut = dawn[day + H]["farms"][1]["tiles"]
        A.append(np.asarray(
            [[1.0 if (isinstance(t, dict) and t.get("kind") == "PLANT") else 0.0
              for t in row] for row in fut], dtype="float32"))
        D.append(day)
    if not T:
        return None
    return (np.stack(T), np.stack(F), np.stack(A), np.asarray(D),
            np.asarray(S, dtype="float32"), np.stack(C), np.stack(B))


def build(args):
    paths = _pick(args.n, args.min_rating, args.seed)
    print(f"replaying {len(paths)} sessions ...", flush=True)
    with mp.Pool(args.procs) as pool:
        out = [o for o in pool.map(job, paths) if o]
    T = np.concatenate([o[0] for o in out])
    F = np.concatenate([o[1] for o in out])
    A = np.concatenate([o[2] for o in out])
    S = np.concatenate([o[4] for o in out])
    C = np.concatenate([o[5] for o in out])
    B = np.concatenate([o[6] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    dest = args.data_out or DATA
    np.savez_compressed(dest, T=T, F=F, A=A, G=G, S=S, C=C, B=B)
    print(f"{len(T):,} samples from {len(out)} games   tensor {T.shape[1:]}   "
          f"flow {F.shape[1]} outputs ({H} days x {len(PRODUCTS)} products)   "
          f"-> {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")

    # Which BUY columns carry any signal at all. Seven of nine products had
    # exactly zero buys in the 6-game probe; at this scale that claim is
    # either confirmed or corrected, and a dead column must never be averaged
    # into a headline R2 as though the model had predicted it.
    nb = B.reshape(len(B), H, len(PRODUCTS)).sum((0, 1))
    ns = C.reshape(len(C), H, len(PRODUCTS)).sum((0, 1))
    print(f"\n    {'product':<12}{'gross sells':>13}{'gross buys':>12}"
          f"{'net':>10}{'buy share':>11}{'buy head':>11}")
    live = []
    for pi, p in enumerate(PRODUCTS):
        sh = nb[pi] / max(1.0, ns[pi] + nb[pi])
        if nb[pi] > 0:
            live.append(p)
        print(f"    {p:<12}{ns[pi]:>13,.0f}{nb[pi]:>12,.0f}"
              f"{ns[pi]-nb[pi]:>10,.0f}{100*sh:>10.1f}%"
              f"{('LIVE' if nb[pi] > 0 else 'dead'):>11}")
    print(f"\n    buy head is live for {len(live)}/{len(PRODUCTS)} products: "
          f"{', '.join(live) if live else 'NONE'}")


def train(args):
    from tools.boardnet import fit, predict, save
    from tools.board_tensor import arm_channels
    d = np.load(args.data or DATA)
    T, F, A, G = d["T"], d["F"], d["A"], d["G"]
    if args.arm:
        cols = arm_channels(args.arm)
        T = np.ascontiguousarray(T[:, cols])
        print(f"arm {args.arm}: {len(cols)} of {d['T'].shape[1]} channels")
    S = d["S"] if (args.shed_aux and "S" in d.files) else None
    print(f"{len(T):,} samples, {len(np.unique(G))} games\n")
    net, stats, hist = fit(T, F, G, A=A, S=S, epochs=args.epochs, seed=args.seed,
                           width=args.width, blocks=args.blocks)
    out = args.out or MODEL
    save(out, net, stats)
    print(f"    best val MSE {stats['val_mse']:.4f} after {len(hist)} epochs")
    print(f"    -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")

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


def ablate(args):
    """Every v2 change attributed, as pure channel slices of ONE dataset.

    The dead-channel arm (A vs B) is the one asked for directly. Its outcome
    is predictable -- a structurally-zero channel contributes zero to the conv
    sum and gets zero gradient, so A and B can differ only through the shape of
    the random init -- which is exactly why it doubles as the SEED-NOISE FLOOR
    for reading every other arm.
    """
    from tools.board_tensor import ARMS, arm_channels
    from tools.boardnet import fit, predict
    d = np.load(args.data or DATA)
    T, F, A, G = d["T"], d["F"], d["A"], d["G"]
    S = d["S"] if "S" in d.files else None
    n_p = len(PRODUCTS)
    print(f"{len(T):,} samples, {len(np.unique(G))} games, superset "
          f"{T.shape[1]} channels\n")
    print(f"    {'arm':<18}{'ch':>4}{'seed':>6}{'val MSE':>10}"
          f"{'held-out R2 on 10-day flow':>28}")
    res = {}
    for arm in ARMS:
        cols = arm_channels(arm)
        if max(cols) >= T.shape[1]:
            # An arm indexes ABSOLUTE channels, so a dataset built before that
            # arm existed would silently hand it the wrong planes rather than
            # fail. Skip loudly instead.
            print(f"    {arm:<18}{len(cols):>4}   SKIPPED -- needs "
                  f"{max(cols)+1} channels, dataset has {T.shape[1]}; "
                  f"rebuild with --build")
            continue
        Ti = np.ascontiguousarray(T[:, cols])
        for seed in args.seeds:
            net, stats, hist = fit(Ti, F, G, A=A,
                                   S=(S if arm.startswith("F_") else None),
                                   epochs=args.epochs, seed=seed,
                                   width=args.width, blocks=args.blocks,
                                   verbose=False)
            gids = np.unique(G)
            rng = np.random.default_rng(seed)
            rng.shuffle(gids)
            va = np.isin(G, gids[:max(1, len(gids) // 5)])
            P = predict(net, stats, Ti[va])
            Yv = F[va]
            r2s = []
            for pi in range(n_p):
                c = [j * n_p + pi for j in range(H)]
                y, q = Yv[:, c].sum(1), P[:, c].sum(1)
                r2s.append(1 - ((y - q) ** 2).sum()
                           / max(1e-9, ((y - y.mean()) ** 2).sum()))
            m = float(np.mean(r2s))
            res.setdefault(arm, []).append(m)
            print(f"    {arm:<18}{len(cols):>4}{seed:>6}{stats['val_mse']:>10.4f}"
                  f"{m:>28.4f}", flush=True)
    print(f"\n    {'arm':<18}{'mean R2':>10}{'spread':>9}{'vs A':>9}")
    base = float(np.mean(res[ARMS[0]]))
    for arm in ARMS:
        v = res[arm]
        print(f"    {arm:<18}{np.mean(v):>10.4f}{(max(v)-min(v)):>9.4f}"
              f"{np.mean(v)-base:>+9.4f}")
    noise = max(res[ARMS[0]]) - min(res[ARMS[0]])
    print(f"\n    seed-noise floor from arm A: {noise:.4f}. Read any 'vs A'"
          f" smaller than this as nothing.")


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
    from tools.board_tensor import board_pair2
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
    dawn, mid = {}, {}
    while not env.done:
        o = env.state[0].observation
        ref["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
        elif o["step"] % 24 == 12:
            mid[o["step"] // 24] = o
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
            from tools.board_tensor import board_pair_full, arm_channels
            t = board_pair_full(dawn[day], 0, prev_mid=mid.get(day - 1))
            t = t[arm_channels(os.environ.get('ARM', 'E_full_83'))][None]
            t = np.ascontiguousarray(t, dtype='float32')
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
    # Bootstrap BY GAME. Days inside one game are nearly the same board, so a
    # per-sample interval would be far too tight -- the honest unit is a game.
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(2000):
        pick = rng.integers(0, len(rows), len(rows))
        vals = {}
        for m in ("board", "model", "oracle"):
            tot = sum(rows[j][m][5][0] for j in pick)
            n = sum(rows[j][m][5][1] for j in pick)
            vals[m] = tot / max(1, n)
        fr.append((vals["board"] - vals["model"])
                  / max(1e-9, vals["board"] - vals["oracle"]))
    lo, hi = np.percentile(fr, [2.5, 97.5])
    print(f"\n    at k=5 the model closes {100*frac:.0f}% of the "
          f"{b:.1f} -> {o:.1f} gap"
          f"   [95% by-game CI {100*lo:.0f}-{100*hi:.0f}%]"
          f"\n    -> {'PASS' if lo >= 0.5 else ('PASS (point est) but CI crosses the bar' if frac >= 0.5 else 'KILL (needs >= 50%)')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default="",
                    help="model destination for --train")
    ap.add_argument("--data", default="",
                    help="dataset to READ for --train/--ablate")
    ap.add_argument("--data-out", default="",
                    help="dataset destination for --build; kept separate from --out so\n"
                         "--build --train cannot write both to one path")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--skip", type=int, default=400)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--shed-aux", action="store_true",
                    help="add the opponent-shed auxiliary head")
    args = ap.parse_args()
    if args.build:
        build(args)
    if args.train:
        train(args)
    if args.ablate:
        ablate(args)
    if args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
