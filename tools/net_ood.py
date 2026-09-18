"""Does BoardNet still predict when the opponent is NOT a recorded top player?

The nets were trained on 400 recorded sessions where both seats were rated
2800+, so the thing they learned to predict is "what a strong human-ish agent
will put on the market". S3 evaluated them on HELD-OUT sessions of the same
kind -- in-distribution -- and they closed 67% of the oracle gap.

Then the round-robin played them against OUR agents, and they lost. That is a
distribution shift, not necessarily a modelling failure: champ-h8 and r-mrg100
plant and sell nothing like a 2800 seat. If the prediction collapses
out-of-distribution, the net is not wrong, it is being asked a question it was
never trained on -- and the fix is self-play data, not a better architecture.

Three regimes, same three-way comparison as S3 (arithmetic / model / oracle):

    replay      opponent is a held-out recorded 2800+ seat   IN-DISTRIBUTION
    champ-h8    opponent is our heuristic                    OOD
    r-mrg100    opponent is our best agent                   OOD

    PYTHONPATH=. uv run python tools/net_ood.py --model rl/datasets/boardnet_E.npz \
        --arm E_full_83 --n 20
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
H = 10
EVAL_K = (1, 3, 5, 10)


def job(spec):
    path, seed, opp_name, model, arm = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, CROPS, PRODUCTS as ENGP, SHOPS, TOWN_CENTER_PRODUCTS,
        market_price)
    import tools.price_forecast as pf
    from tools.board_tensor import board_pair_full, arm_channels
    from tools.boardnet import load, predict
    K = {"SHOPS": SHOPS, "TOWN": TOWN_CENTER_PRODUCTS, "CROPS": CROPS,
         "ANIMALS": ANIMALS, "PRODUCTS": ENGP, "market_price": market_price}
    net, stats = load(model)
    cols = arm_channels(arm)

    pf.FLOW.clear()
    ref = pf._patch_flow()
    if opp_name == "replay":
        s = json.load(open(path))
        seed = s["seed"]
        a0 = pf._script(s["actions"][0])
        a1 = pf._script(s["actions"][1])
    else:
        def ld(n):
            g = {}
            exec(open(os.path.join(REPO, "arena", "agents", n,
                                   "main.py")).read(), g)
            return g["agent"]
        a0 = ld(opp_name)
        a1 = ld(opp_name)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
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
    for day in sorted(dawn):
        if day + H > max(dawn):
            break
        truth = {p: [dawn[day + k]["market"]["prices"][p] for k in range(H + 1)]
                 for p in ENGP}
        oracle = [{}] + [dict(flow.get(day + j, {})) for j in range(1, H + 1)]
        t = board_pair_full(dawn[day], 0, prev_mid=mid.get(day - 1))
        x = np.ascontiguousarray(t[cols], dtype="float32")[None]
        q = predict(net, stats, x)[0].reshape(H, len(PRODUCTS))
        mdl = [{}] + [{p: float(q[j - 1, i]) for i, p in enumerate(PRODUCTS)}
                      for j in range(1, H + 1)]
        for name, oo in (("board", None), ("model", mdl), ("oracle", oracle)):
            f = pf.forecast(dawn[day], 0, H, 5, K, opp_oracle=oo)
            for k in EVAL_K:
                for p in ENGP:
                    err[name][k].append(abs(f[p][k] - truth[p][k]))
    return {m: {k: (sum(v), len(v)) for k, v in dd.items()}
            for m, dd in err.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rl/datasets/boardnet_E.npz")
    ap.add_argument("--arm", default="E_full_83")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip", type=int, default=400)
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if v["scores"] and all(x is not None for x in v["scores"])
             and min(v["scores"]) >= 2800]
    random.Random(args.seed).shuffle(paths)
    paths = paths[args.skip: args.skip + args.n]
    rng = random.Random(99)
    seeds = [300_000 + rng.randrange(10**6) for _ in range(args.n)]

    print(f"model {os.path.basename(args.model)}   arm {args.arm}   "
          f"{args.n} games per regime\n", flush=True)
    print(f"    {'regime':<30}" + "".join(f"{'k='+str(k):>9}" for k in EVAL_K))
    for label, opp in (("replay (held-out 2800+)  IN-DIST", "replay"),
                       ("vs champ-h8              OOD", "champ-h8"),
                       ("vs r-mrg100              OOD", "r-mrg100")):
        specs = [(paths[i] if opp == "replay" else None, seeds[i], opp,
                  os.path.join(REPO, args.model), args.arm)
                 for i in range(args.n)]
        with mp.Pool(args.procs) as pool:
            rows = pool.map(job, specs)
        got = {}
        for m in ("board", "model", "oracle"):
            got[m] = [sum(r[m][k][0] for r in rows) /
                      max(1, sum(r[m][k][1] for r in rows)) for k in EVAL_K]
        i5 = EVAL_K.index(5)
        b, mo, o = got["board"][i5], got["model"][i5], got["oracle"][i5]
        frac = (b - mo) / max(1e-9, b - o)
        print(f"    {label:<30}")
        for m, nm in (("board", "  arithmetic"), ("model", "  BoardNet"),
                      ("oracle", "  PERFECT")):
            print(f"    {nm:<30}" + "".join(f"{c:>9.1f}" for c in got[m]))
        print(f"      -> closes {100*frac:>4.0f}% of the gap at k=5\n", flush=True)


if __name__ == "__main__":
    main()
