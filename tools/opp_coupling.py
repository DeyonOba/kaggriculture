"""How much money does the OPPONENT cost us through the shared market?

The two farms never touch. Separate 10x10 grids, separate units, separate
sheds. The only coupling in the entire engine is `market["inventory"]`, a
single global dict that both players sell into, and price is a deterministic
closed-form function of it (`market_price`, no randomness at all).

That coupling is steep. STRAWBERRY has T=100 and above_target=1.6, so 100
units above I0 takes the price from $120 to $1. WOOL is steeper still
(T=105, above=3.2).

So before designing any opponent model, measure the prize. This runs a
deterministic counterfactual on recorded sessions:

    baseline   both recorded scripts, unmodified
    muted      the OPPONENT's SELLs still pay them, still drain their shed,
               but do not add to market inventory

Muting the inventory line rather than deleting their orders keeps their cash,
their shed and therefore every one of their later actions bit-identical. Only
the price channel is cut. The delta is exactly what their supply costs us --
an upper bound on what any opponent model could ever recover, because a model
can predict their selling but cannot stop it.

    PYTHONPATH=. uv run python tools/opp_coupling.py --n 40 --min-rating 2800
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS_DIR = os.path.join(REPO, "rl/datasets/sessions")
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")

# Filled in the worker: (seat -> {item: units}) attribution of inventory moves.
TRACE = {}


def build_index():
    """episode -> {seed, finals, scores, path} for every recorded session.

    Sessions are ~270 KB each and there are 21k of them, so this reads only
    the first 400 and last 300 bytes: `episode`/`seed`/`finals` lead the file
    and `scores` trails it. 8.8 s for the whole corpus against minutes for
    json.load. `scores` are the two seats' leaderboard ratings.

    NOTE rl/datasets/ratings/agents.csv covers a different episode range --
    2 of 21,517 sessions overlap -- so it cannot be used to rate these games.
    """
    import glob
    import re

    head_re = re.compile(
        rb'"episode":\s*(\d+),\s*"seed":\s*(-?\d+),.*?"finals":\s*\[([^\]]*)\]', re.S)
    score_re = re.compile(rb'"scores":\s*\[([^\]]*)\]')
    idx, bad = {}, 0
    for p in glob.glob(os.path.join(SESS_DIR, "session-*.json")):
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(400)
            f.seek(max(0, size - 300))
            tail = f.read()
        mh, ms = head_re.search(head), score_re.search(tail)
        if not mh or not ms:
            bad += 1
            continue
        idx[int(mh.group(1))] = {
            "seed": int(mh.group(2)),
            "finals": [float(x) for x in mh.group(3).split(b",")],
            "scores": [None if x.strip() == b"null" else float(x)
                       for x in ms.group(1).split(b",")],
            "path": p,
        }
    json.dump(idx, open(INDEX, "w"))
    print(f"indexed {len(idx)} sessions ({bad} unparsed) -> {INDEX}", flush=True)
    return idx


def _script(actions):
    st = {"t": 0}

    def agent(obs):
        t = obs.get("step", st["t"])
        st["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def _patch(mute_seat, trace=False, only=None):
    """Mute `mute_seat`'s supply into market inventory. Returns nothing.

    `only` restricts the muting to a single product, which isolates where the
    congestion actually costs money.

    Tagging happens in `_initialize`, which the interpreter looks up as a
    module global on the first step, so a monkeypatch takes effect.
    """
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    orig_init = getattr(K, "_opp_orig_init", None) or K._initialize
    K._opp_orig_init = orig_init
    orig_commit = getattr(K, "_opp_orig_commit", None) or K._commit_unit
    K._opp_orig_commit = orig_commit

    def init(state, env):
        orig_init(state, env)
        for i, s in enumerate(state):
            s.observation.private["_seat"] = i

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        seat = private.get("_seat", -1)
        if op == "SELL" and seat == mute_seat and (only is None or item == only):
            # Same as the engine, minus the one inventory line.
            if private["shed"].get(item, 0) <= 0:
                return False
            private["shed"][item] -= 1
            farm["money"] += price
            if trace:
                TRACE.setdefault(seat, {}).setdefault(item, 0)
                TRACE[seat][item] += 1
            return True
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and trace and seat >= 0:
            d = TRACE.setdefault(seat, {})
            if op == "SELL" and price > 1:
                d[item] = d.get(item, 0) + 1
            elif op == "BUY_PRODUCT":
                d[item] = d.get(item, 0) - 1
        return ok

    K._initialize = init
    K._commit_unit = commit


def _run(seed, actions, mute_seat=-1, trace=False, only=None):
    import tools.fast_env as fast_env

    fast_env.install()
    TRACE.clear()
    _patch(mute_seat, trace=trace, only=only)
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([_script(actions[0]), _script(actions[1])])
    last = env.steps[-1]
    finals = [last[0].reward or 0.0, last[1].reward or 0.0]
    inv = dict(last[0].observation["market"]["inventory"])
    return finals, inv, {k: dict(v) for k, v in TRACE.items()}


def job(path):
    sys.path.insert(0, REPO)
    s = json.load(open(path))
    seed, acts = s["seed"], s["actions"]
    try:
        base, inv, tr = _run(seed, acts, mute_seat=-1, trace=True)
        # Seat 0's world with seat 1's supply removed, and vice versa.
        m1, _, _ = _run(seed, acts, mute_seat=1)
        m0, _, _ = _run(seed, acts, mute_seat=0)
    except Exception as e:  # a handful of sessions use a stale module version
        return {"episode": s["episode"], "error": f"{type(e).__name__}: {e}"}
    return {
        "episode": s["episode"],
        "seed": seed,
        "recorded": s["finals"],
        "base": base,
        # gain to seat 0 when seat 1's supply never reaches the market
        "gain0": m1[0] - base[0],
        "gain1": m0[1] - base[1],
        "inv_end": inv,
        "supply": tr,
    }


def job_product(path):
    """Mute the opponent's supply of ONE product at a time.

    Nine extra replays per session, but it answers the question that decides
    what a model should predict: which market is the congestion in?
    """
    sys.path.insert(0, REPO)
    from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS

    s = json.load(open(path))
    seed, acts = s["seed"], s["actions"]
    try:
        base, _, _ = _run(seed, acts, mute_seat=-1)
        out = {"episode": s["episode"], "base": base, "by_product": {}}
        for p in PRODUCTS:
            m1, _, _ = _run(seed, acts, mute_seat=1, only=p)
            m0, _, _ = _run(seed, acts, mute_seat=0, only=p)
            out["by_product"][p] = [m1[0] - base[0], m0[1] - base[1]]
    except Exception as e:
        return {"episode": s["episode"], "error": f"{type(e).__name__}: {e}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--by-product", action="store_true",
                    help="decompose the gain product by product (10x cost)")
    args = ap.parse_args()

    idx = json.load(open(INDEX)) if os.path.exists(INDEX) else build_index()
    paths = [
        v["path"] for v in idx.values()
        if v["scores"] and all(s is not None for s in v["scores"])
        and min(v["scores"]) >= args.min_rating
    ]
    n_pool = len(paths)
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"{n_pool} episodes with both seats >= {args.min_rating:.0f}; "
          f"{len(paths)} sampled\n", flush=True)

    with mp.Pool(args.procs) as pool:
        rows = pool.map(job_product if args.by_product else job, paths)

    ok = [r for r in rows if "error" not in r]

    if args.by_product:
        from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS

        print(f"{len(ok)} sessions, {2 * len(ok)} seat-observations\n")
        print(f"  gain to a seat when the OPPONENT's supply of one product"
              f" never reaches the market:\n")
        print(f"    {'product':<12}{'gain':>11}{'se':>9}{'z':>8}")
        tot = 0.0
        for p in PRODUCTS:
            g = [v for r in ok for v in r["by_product"][p]]
            m = sum(g) / len(g)
            sd = (sum((x - m) ** 2 for x in g) / max(1, len(g) - 1)) ** 0.5
            se = sd / len(g) ** 0.5
            tot += m
            print(f"    {p:<12}{m:>11,.0f}{se:>9,.0f}"
                  f"{(m / se if se else 0):>8.2f}")
        print(f"    {'sum':<12}{tot:>11,.0f}    (vs the joint mute; the"
              f" difference is interaction)")
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=1)
            print(f"\nwrote {args.out}")
        return
    bad = len(rows) - len(ok)
    gains = [g for r in ok for g in (r["gain0"], r["gain1"])]
    bases = [b for r in ok for b in r["base"]]
    if not gains:
        print("no usable replays")
        return
    n = len(gains)
    mean = sum(gains) / n
    sd = (sum((g - mean) ** 2 for g in gains) / max(1, n - 1)) ** 0.5
    se = sd / n**0.5
    base_mean = sum(bases) / len(bases)

    print(f"{len(ok)} sessions ({bad} unreplayable), {n} seat-observations\n")
    print(f"  baseline final bank            ${base_mean:>12,.0f}")
    print(f"  with opponent supply MUTED     ${base_mean + mean:>12,.0f}")
    print(f"  gain                           ${mean:>12,.0f}  +/- {se:,.0f}"
          f"   z = {mean / se if se else 0:.2f}")
    print(f"  as a share of the bank          {100 * mean / base_mean:>11.1f}%\n")

    q = sorted(gains)
    print("  distribution of the per-seat gain:")
    for lbl, v in [("min", q[0]), ("p25", q[n // 4]), ("median", q[n // 2]),
                   ("p75", q[3 * n // 4]), ("max", q[-1])]:
        print(f"    {lbl:>6}  ${v:>12,.0f}")

    # Which product carries the coupling: end-of-season inventory tells us
    # which market the two of them actually crowded.
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MARKET_PARAMS, PRODUCTS, market_price)

    print("\n  end-of-season market inventory vs I0 (mean over sessions), and"
          "\n  the price that deviation implies:")
    print(f"    {'product':<12}{'inv-I0':>9}{'price':>8}{'base':>7}"
          f"{'mine':>8}{'theirs':>8}")
    for p in PRODUCTS:
        I0 = MARKET_PARAMS[p]["I0"]
        devs = [r["inv_end"][p] - I0 for r in ok]
        d = sum(devs) / len(devs)
        mine = sum(r["supply"].get(0, {}).get(p, 0) for r in ok) / len(ok)
        theirs = sum(r["supply"].get(1, {}).get(p, 0) for r in ok) / len(ok)
        print(f"    {p:<12}{d:>9,.0f}{market_price(p, int(I0 + d)):>8}"
              f"{MARKET_PARAMS[p]['base']:>7}{mine:>8.0f}{theirs:>8.0f}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
