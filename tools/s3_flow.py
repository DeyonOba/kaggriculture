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

BANDS = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 1e9)]


def _pick(n, min_rating, seed, skip=0, bands=False):
    """Session paths. `bands` draws EQUALLY per rating band instead of using a
    floor.

    The floor is the default only for reproducing old builds. 14,211 of the
    corpus's 21,054 sessions sit in the 2500+ band, so `--min-rating 2000` is
    in practice "top games only" -- and the 2500+ band is the HARDEST to
    predict of all of them (R^2 -0.100 against +0.334 at <1000). Every arm
    this project ever compared was drawn that way.
    """
    import random
    idx = _load_index()
    if not bands:
        paths = [v["path"] for v in idx.values() if _rating_ok(v, min_rating)]
        random.Random(seed).shuffle(paths)
        return paths[skip: skip + n]

    pool = {i: [] for i in range(len(BANDS))}
    for v in idx.values():
        sc = [x for x in (v.get("scores") or []) if x is not None]
        if not sc or not v.get("path"):
            continue
        r = max(sc)
        b = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= r < hi)
        pool[b].append(v["path"])
    rng = random.Random(seed)
    per, out = max(1, n // len(BANDS)), []
    for b in range(len(BANDS)):
        rng.shuffle(pool[b])
        take = pool[b][skip: skip + per]
        out += take
        print(f"  band {BANDS[b][0]}-{BANDS[b][1]:.0f}: "
              f"{len(pool[b]):5d} available, {len(take)} drawn")
    return out


_RATING_BY_PATH = None


def _session_rating(path):
    """max rating of the two seats, or nan -- stored per row for per-band eval.

    From the INDEX, not from the session file: `scores` sits after the two
    720-step action scripts, i.e. megabytes into the JSON, so reading a 4 KB
    head returns nothing and every rating comes back NaN.
    """
    import math
    global _RATING_BY_PATH
    if _RATING_BY_PATH is None:
        _RATING_BY_PATH = {}
        for v in _load_index().values():
            sc = [x for x in (v.get("scores") or []) if x is not None]
            if v.get("path"):
                _RATING_BY_PATH[v["path"]] = max(sc) if sc else math.nan
    return _RATING_BY_PATH.get(path, math.nan)


def cap_series(dawn):
    """-> ({day: (9,) log-scaled capacity}, (9,) scaled daily regen).

    Everything here is PUBLIC -- market inventory is in the observation and
    econ is the engine's own price curve -- so this runs identically at build
    time and at play time. Shared by job() and eval_job() precisely so the
    tensor the model is scored on cannot drift from the one it was trained
    on; they used different encoders before and the eval could not even
    construct the capacity arms.
    """
    from kagent import econ
    from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS
    from tools.market_clock import daily_drain
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
    return capday, regen


def encode_day(dawn, mid, day, capday, regen):
    """The ONE place a scoreable tensor is built, for build and eval alike."""
    from tools.board_tensor import LAGS_DEEP, board_pair_all
    past = [dawn[d] for d in range(day - 1, -1, -1) if d in dawn]
    cl = np.stack([capday[max(0, day - L)] for L in LAGS_DEEP])
    return board_pair_all(dawn[day], 0, prev_mid=mid.get(day - 1),
                          hist=past, cap_now=capday[day], cap_lags=cl,
                          regen=regen)


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

    capday, regen = cap_series(dawn)
    from tools.board_tensor import LAGS_DEEP, board_pair_all

    T, F, A, D, S, C, B = [], [], [], [], [], [], []
    for day in sorted(dawn):
        if day + H not in dawn:
            continue
        T.append(encode_day(dawn, mid, day, capday, regen))
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
    # float16: T is 177x10x10 of normalised values and indicator bits, and at
    # 1,000+ sessions a float32 return buffered by pool.map is ~1.5 GB in the
    # PARENT, which is what broke the pipe on the first band-stratified build.
    return (np.stack(T).astype("float16"), np.stack(F), np.stack(A), np.asarray(D),
            np.asarray(S, dtype="float32"), np.stack(C), np.stack(B))


def build(args):
    paths = _pick(args.n, args.min_rating, args.seed, bands=args.bands)
    print(f"replaying {len(paths)} sessions ...", flush=True)
    raw = []
    with mp.Pool(args.procs) as pool:
        for i, o in enumerate(pool.imap(job, paths), 1):
            raw.append(o)
            if i % 100 == 0:
                print(f"    {i}/{len(paths)} replayed", flush=True)
    # Keep the path alongside its result: `job` returns None for a session
    # that does not run a full season, so filtering first and zipping after
    # silently pairs each rating with the wrong game.
    pairs = [(p_, o) for p_, o in zip(paths, raw) if o]
    out = [o for _, o in pairs]
    T = np.concatenate([o[0] for o in out])
    F = np.concatenate([o[1] for o in out])
    A = np.concatenate([o[2] for o in out])
    S = np.concatenate([o[4] for o in out])
    C = np.concatenate([o[5] for o in out])
    B = np.concatenate([o[6] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    from tools.board_tensor import ENCODER_VERSION
    R = np.concatenate([np.full(len(o[0]), _session_rating(p_), dtype="float32")
                        for p_, o in pairs])
    dest = args.data_out or DATA
    np.savez_compressed(dest, T=T, F=F, A=A, G=G, S=S, C=C, B=B, R=R,
                        ENC=np.asarray([ENCODER_VERSION]))
    print(f"-> {dest}  encoder v{ENCODER_VERSION}, {len(T):,} rows, "
          f"{len(set(G.tolist()))} sessions")
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
    T, F, A, G = d["T"].astype("float32"), d["F"], d["A"], d["G"]
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
    T, F, A, G = d["T"].astype("float32"), d["F"], d["A"], d["G"]
    S = d["S"] if "S" in d.files else None
    n_p = len(PRODUCTS)

    # ARMS[0] is the reference for both the baseline and the seed-noise floor,
    # so a subset always keeps it however the user spells the request.
    if args.arms:
        # The reference used to be ARMS[0] unconditionally, because the old
        # unpaired floor was derived from it. The paired statistic needs only
        # the --base arm, so forcing A_v1_72 in just burns a third of the fits.
        want, ref = set(args.arms), (args.base if args.base in ARMS else ARMS[0])
        want.add(ref)
        unknown = want - set(ARMS)
        if unknown:
            raise SystemExit(f"unknown arm(s): {sorted(unknown)}\n"
                             f"available: {ARMS}")
        ARMS = [a for a in ARMS if a in want or a == ref]

    print(f"{len(T):,} samples, {len(np.unique(G))} games, superset "
          f"{T.shape[1]} channels, {len(ARMS)} arms x {len(args.seeds)} "
          f"seeds\n")
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
    # PAIRED, because the design is paired and reading it unpaired both
    # overstates and erases real effects.
    #
    # fit() takes `seed` and the eval split is default_rng(seed) over the same
    # game ids, so arm X seed s and arm Y seed s are scored on the IDENTICAL
    # held-out games. The seed effect is large and SHARED -- one seed drags
    # every arm down together -- so differencing within a seed cancels it. On
    # the capacity run the per-arm sd was 0.0107 while the sd of the
    # difference was 0.0044, 2.4x tighter, and the old unpaired "spread"
    # floor of 0.0198 would have killed a +0.0187 effect at p = 0.0001.
    #
    # Baseline is the LAST arm requested, not ARMS[0]: arms are supersets of
    # one another, and every new arm here is a superset of E rather than of
    # the v1 tensor, so comparing to ARMS[0] flatters anything that merely
    # rediscovers what E already had.
    ref = args.base if args.base in res else ARMS[0]
    nseed = len(args.seeds)
    print(f"\n    baseline = {ref}   PAIRED over {nseed} seed"
          f"{'s' if nseed > 1 else ''}")
    print(f"\n    {'arm':<18}{'mean R2':>10}{'sd/seed':>9}"
          f"{'vs base':>10}{'se':>9}{'t':>8}")
    b = np.asarray(res[ref], dtype=float)
    for arm in ARMS:
        v = np.asarray(res[arm], dtype=float)
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        d = v - b
        if arm == ref or len(d) < 2:
            print(f"    {arm:<18}{v.mean():>10.4f}{sd:>9.4f}"
                  f"{d.mean():>+10.4f}{'':>9}{'':>8}")
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("inf")
        print(f"    {arm:<18}{v.mean():>10.4f}{sd:>9.4f}"
              f"{d.mean():>+10.4f}{se:>9.4f}{t:>+8.2f}")
    if nseed < 3:
        print(f"\n    {nseed} seeds gives no usable se. Re-run with "
              f"--seeds 0 1 2 3 4 5 before believing any of it.")
    else:
        print(f"\n    |t| > ~2.6 is p < 0.05 at {nseed-1} df. The unpaired "
              f"spread column is FYI only -- judge on se and t.")


def eval_job(spec):
    """Forecast MAE with board-projected / model / oracle opponent supply."""
    # The model path travels WITH the spec. It used to be the module-level
    # MODEL constant, so --train --out <somewhere> trained one model and
    # --eval silently scored a different, stale one.
    path, use_model, model_path = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, CROPS, PRODUCTS as ENGP, SHOPS, TOWN_CENTER_PRODUCTS,
        market_price)
    import tools.price_forecast as pf
    K = {"SHOPS": SHOPS, "TOWN": TOWN_CENTER_PRODUCTS, "CROPS": CROPS,
         "ANIMALS": ANIMALS, "PRODUCTS": ENGP, "market_price": market_price}

    net = stats = None
    if use_model:
        from tools.boardnet import load, predict
        net, stats = load(model_path)

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
    capday, regen = cap_series(dawn)

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
            from tools.board_tensor import arm_channels
            # encode_day, not board_pair_full: the latter stops at 91 channels
            # so any arm carrying temporal or capacity planes indexed past the
            # end of it. TM_full_157 could not be evaluated at all.
            t = encode_day(dawn, mid, day, capday, regen)
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
    mp_ = args.model or MODEL
    if not os.path.exists(mp_):
        raise SystemExit(f"no model at {mp_}; train one first with "
                         f"--train --arm <arm> --out <path>")
    # ARM must match what the model was trained on -- a mismatch silently
    # feeds the net a different channel slice of the same width in the worst
    # case, so print it and let a wrong pairing be visible in the output.
    arm = os.environ.get("ARM", "E_full_83")
    print(f"model {os.path.basename(mp_)}   arm {arm}")
    print(f"{len(paths)} HELD-OUT sessions (skipping the first {args.skip} "
          f"used for training)\n", flush=True)
    with mp.Pool(args.procs) as pool:
        rows = pool.map(eval_job, [(p, True, mp_) for p in paths])
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
    ap.add_argument("--bands", action="store_true",
                    help="draw EQUALLY per rating band instead of using a "
                         "rating floor (see _pick)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default="",
                    help="model destination for --train")
    ap.add_argument("--model", default="",
                    help="model to score for --eval; defaults to MODEL")
    ap.add_argument("--base", default="E_full_83",
                    help="baseline arm for the paired comparison; arms are\n"
                         "supersets, so this must be the one they extend")
    ap.add_argument("--arms", nargs="+", default=[],
                    help="subset of arms for --ablate; the reference arm is\n"
                         "always kept for the baseline and noise floor")
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
