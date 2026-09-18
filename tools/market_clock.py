"""The temporal side of market inventory: when does each product glut?

`tools/opp_coupling.py` measured the opponent's price damage as a TERMINAL
counterfactual -- final bank with and without their supply. That says how much
the congestion costs but nothing about when, and a sell floor is a timing rule.
This supplies the missing clock.

Inventory is an exact accounting identity, no randomness in it at all:

    inv_p(t) = I0_p + sold_p(t) - bought_p(t) - drained_p(t)

    drained_p per day = 6 * SUM over unlocked shop instances holding p of
                        (2 if the shop sells one product else 1)
                      + 1 if p is a TOWN_CENTER product

6 because townShopSellInterval=4 against turnsPerDay=24. The only exogenous
term is WHICH shops unlocked: one uniform draw from the 8 shops every
townShopUnlockInterval=3 days, with replacement, capped at MAX_SHOP_INSTANCES=8
-- so the demand curve for the whole season is set by 8 coin flips revealed
between day 3 and day 24.

    PYTHONPATH=. uv run python tools/market_clock.py --n 24 --min-rating 2800
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS_DIR = os.path.join(REPO, "rl/datasets/sessions")
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")

# day -> per-product {seat: units sold, "buy": units bought}
FLOW = {}


def daily_drain(shops, product):
    """Analytic: units of `product` the town removes per day, given `shops`."""
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        SHOPS, TOWN_CENTER_PRODUCTS)

    per_event = sum(
        (2 if len(SHOPS[s]) == 1 else 1)
        for s in shops if product in SHOPS[s]
    )
    return 6 * per_event + (1 if product in TOWN_CENTER_PRODUCTS else 0)


def _patch():
    """Record every market commit by day, seat and product."""
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    orig_init = getattr(K, "_mc_orig_init", None) or K._initialize
    K._mc_orig_init = orig_init
    orig_commit = getattr(K, "_mc_orig_commit", None) or K._commit_unit
    K._mc_orig_commit = orig_commit

    def init(state, env):
        orig_init(state, env)
        for i, s in enumerate(state):
            s.observation.private["_seat"] = i

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and op in ("SELL", "BUY_PRODUCT"):
            seat = private.get("_seat", -1)
            d = FLOW.setdefault(item, {})
            # `price > 1` sales do not add supply; record what the engine did.
            if op == "SELL":
                if price > 1:
                    d[seat] = d.get(seat, 0) + 1
                d[f"s{seat}_all"] = d.get(f"s{seat}_all", 0) + 1
                d[f"s{seat}_rev"] = d.get(f"s{seat}_rev", 0) + price
            else:
                d["buy"] = d.get("buy", 0) + 1
        return ok

    K._initialize = init
    K._commit_unit = commit


def _script(actions):
    st = {"t": 0}

    def agent(obs):
        t = obs.get("step", st["t"])
        st["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def job_agent(spec):
    """Same trajectory, but for OUR agent instead of a recorded script.

    The recorded table shows top players moving volume 2-4x with the shop
    draw. This asks whether ours moves at all.
    """
    seed, cand, opp = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env

    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS

    def load(name):
        g = {}
        exec(open(os.path.join(REPO, "arena", "agents", name, "main.py")).read(), g)
        return g["agent"]

    FLOW.clear()
    _patch()
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([load(cand), load(opp)])
    traj = {p: [] for p in PRODUCTS}
    shops_at = []
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24 != 0:
            continue
        shops_at.append(list(o["town"]["unlocked_shops"]))
        for p in PRODUCTS:
            traj[p].append([o["market"]["inventory"][p], o["market"]["prices"][p]])
    return {
        "episode": seed,
        "shops": env.steps[-1][0].observation["town"]["unlocked_shops"],
        "traj": traj,
        "flow": {p: dict(v) for p, v in FLOW.items()},
        "check": {p: [0, 0] for p in PRODUCTS},
        "finals": [env.steps[-1][i].reward or 0.0 for i in (0, 1)],
    }


def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env

    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MARKET_PARAMS, PRODUCTS)

    s = json.load(open(path))
    FLOW.clear()
    _patch()
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([_script(s["actions"][0]), _script(s["actions"][1])])

    # Per-day inventory and price, read off the first step of each day.
    traj = {p: [] for p in PRODUCTS}
    shops_at = []
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24 != 0:
            continue
        shops_at.append(list(o["town"]["unlocked_shops"]))
        for p in PRODUCTS:
            traj[p].append([o["market"]["inventory"][p], o["market"]["prices"][p]])

    # Did the analytic drain model reproduce the engine? Check the last day:
    # inv = I0 + sold - bought - drained, summed over the season.
    final = env.steps[-1][0].observation["market"]["inventory"]
    check = {}
    for p in PRODUCTS:
        I0 = MARKET_PARAMS[p]["I0"]
        f = FLOW.get(p, {})
        sold = f.get(0, 0) + f.get(1, 0)
        bought = f.get("buy", 0)
        drained = sum(daily_drain(sh, p) for sh in shops_at)
        check[p] = [final[p], I0 + sold - bought - drained]

    return {
        "episode": s["episode"],
        "shops": s["shops"],
        "traj": traj,
        "flow": {p: dict(v) for p, v in FLOW.items()},
        "check": check,
        "finals": [env.steps[-1][i].reward or 0.0 for i in (0, 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--agent", default="",
                    help="measure OUR agent instead of recorded scripts")
    ap.add_argument("--opp", default="champ-h8")
    args = ap.parse_args()

    import random

    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MARKET_PARAMS, PRODUCTS, market_price)

    if args.agent:
        rng = random.Random(args.seed)
        specs = [(300_000 + rng.randrange(10**6), args.agent, args.opp)
                 for _ in range(args.n)]
        print(f"{args.n} worlds, {args.agent} vs {args.opp}\n", flush=True)
        with mp.Pool(args.procs) as pool:
            rows = pool.map(job_agent, specs)
        mean_bank = sum(r["finals"][0] for r in rows) / len(rows)
        print(f"mean bank ${mean_bank:,.0f}\n")
    else:
        idx = json.load(open(INDEX))
        paths = [v["path"] for v in idx.values()
                 if v["scores"] and all(x is not None for x in v["scores"])
                 and min(v["scores"]) >= args.min_rating]
        random.Random(args.seed).shuffle(paths)
        paths = paths[: args.n]
        print(f"{len(paths)} sessions, both seats >= {args.min_rating:.0f}\n",
              flush=True)
        with mp.Pool(args.procs) as pool:
            rows = pool.map(job, paths)

        # --- 1. does the analytic drain model reproduce the engine? --------
        err = [abs(r["check"][p][0] - r["check"][p][1])
               for r in rows for p in PRODUCTS]
        exact = sum(1 for e in err if e == 0)
        print(f"analytic inventory identity: {exact}/{len(err)} products exact, "
              f"max error {max(err)}")
        print("  inv = I0 + sold - bought - 6*shop_instances - town_centre\n")

    n_days = min(len(r["traj"]["WHEAT"]) for r in rows)

    # --- 2. the clock: when does each product cross into glut? --------------
    print("per-day mean PRICE, and the day inventory first crosses I0 (glut):\n")
    hdr = "  ".join(f"d{d:<2}" for d in range(0, n_days, 3))
    print(f"    {'product':<12}{'base':>5}  {hdr}   glut day")
    for p in PRODUCTS:
        I0 = MARKET_PARAMS[p]["I0"]
        cells, glut = [], None
        for d in range(0, n_days, 3):
            pr = sum(r["traj"][p][d][1] for r in rows) / len(rows)
            cells.append(f"{pr:>3.0f}")
        for d in range(n_days):
            over = sum(1 for r in rows if r["traj"][p][d][0] > I0)
            if over > len(rows) / 2:
                glut = d
                break
        g = f"{glut}" if glut is not None else "never"
        print(f"    {p:<12}{MARKET_PARAMS[p]['base']:>5}  " +
              "   ".join(cells) + f"   {g:>6}")

    # --- 3. realised price vs base, per seat --------------------------------
    print("\nvolume and realised $/unit (both seats pooled):\n")
    print(f"    {'product':<12}{'units':>8}{'realised':>10}{'base':>7}"
          f"{'ratio':>8}{'E drain/day':>13}")
    for p in PRODUCTS:
        units = sum(r["flow"].get(p, {}).get(f"s{s}_all", 0) for r in rows for s in ((0,) if args.agent else (0,1)))
        rev = sum(r["flow"].get(p, {}).get(f"s{s}_rev", 0) for r in rows for s in ((0,) if args.agent else (0,1)))
        if not units:
            print(f"    {p:<12}{0:>8}")
            continue
        real = rev / units
        base = MARKET_PARAMS[p]["base"]
        dr = sum(daily_drain(r["shops"], p) for r in rows) / len(rows)
        print(f"    {p:<12}{units / len(rows):>8.0f}{real:>10.0f}{base:>7}"
              f"{real / base:>8.2f}{dr:>13.1f}")

    # --- 4. how much does the shop draw move the outcome? -------------------
    print("\nshop draw is the only exogenous economic variable. Its spread:\n")
    # For an agent run, report SEAT 0 only -- that is the agent under test.
    seats = (0,) if args.agent else (0, 1)
    for p in ("WOOL", "STRAWBERRY", "MILK"):
        by = {}
        for r in rows:
            k = daily_drain(r["shops"], p)
            units = sum(r["flow"].get(p, {}).get(f"s{s}_all", 0) for s in seats)
            rev = sum(r["flow"].get(p, {}).get(f"s{s}_rev", 0) for s in seats)
            by.setdefault(k, []).append((units, rev))
        print(f"    {p}:")
        for k in sorted(by):
            v = by[k]
            u = sum(x[0] for x in v) / len(v)
            rr = sum(x[1] for x in v) / max(1e-9, sum(x[0] for x in v))
            print(f"      drain {k:>3}/day  n={len(v):>3}  units {u:>5.0f}"
                  f"  realised ${rr:>5.0f}")

    if args.out:
        json.dump(rows, open(args.out, "w"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
