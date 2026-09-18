"""Forward market model: predict the price of every product k days ahead.

This is the one part of the world that does not need to be learned. Market
inventory is an exact identity and price is a closed form on top of it:

    inv_p(t+1) = inv_p(t) + sold_p - bought_p - drain_p(shops)
    price_p    = market_price(inv_p)                 (deterministic)

`tools/market_clock.py` verified the identity at 36/36 products, zero error.
What is NOT known at time t is the future of the three flow terms, and each
one is observable to a different degree:

    drain      known exactly for shops already unlocked; the remaining draws
               of the 8 are uniform over 8 shops, so their contribution is a
               computable EXPECTATION
    supply     both farms' boards are public (observation.farms is shared),
               and yield is mechanical -- a non-ongoing crop's units are the
               number of times it is watered inside
               [(max_yield_day+1)//2, max_yield_day], an ongoing crop accrues
               1/interval days, an animal 1/interval days
    buys       feed purchases, which scale with herd size

So the forecast is an ABLATION, not a fit. Each layer adds one observable and
the error reduction says what that observable is worth:

    L0  persistence            price(t+k) = price(t)
    L1  + drain from the shops already unlocked
    L2  + expected drain from the shop draws still to come
    L3  + our own board's projected supply
    L4  + the opponent's board's projected supply

L4 minus L3 is the value of looking at their farm at all.

    PYTHONPATH=. uv run python tools/price_forecast.py --n 24 --min-rating 2800
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")

N_SHOP_DRAWS = 8          # MAX_SHOP_INSTANCES
SHOP_EVERY = 3            # townShopUnlockInterval, first draw lands on day 3
HORIZONS = (1, 3, 5, 10)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def shop_drain(shops, product, SHOPS, TOWN):
    per_event = sum((2 if len(SHOPS[s]) == 1 else 1)
                    for s in shops if product in SHOPS[s])
    return 6 * per_event + (1 if product in TOWN else 0)


def expected_extra_drain(n_future_draws, product, SHOPS):
    """Mean drain per day added by draws not yet revealed."""
    names = sorted(SHOPS)
    mu = sum((2 if len(SHOPS[s]) == 1 else 1) if product in SHOPS[s] else 0
             for s in names) / len(names)
    return 6 * n_future_draws * mu


def board_supply(farm, day, horizon, CROPS, ANIMALS):
    """Units of each product this board puts on the market, by day offset.

    Mechanical, not fitted. Three corrections measured from the first version's
    per-product bias, each traced to an engine rule it was ignoring:

    * ANIMAL CARE DOUBLES OUTPUT. `_daily_refresh_animals` adds `base + bonus`
      where the bonus is a pending care credit, so a cared-and-fed animal
      yields 2 per production day. Ignoring it under-predicted milk and wool
      supply ~2x, which showed up as a +$125 / +$88 price bias at k=10.
    * ANIMALS PRODUCE FERTILIZER DAILY. `fertilizer_available = True` every
      night, one unit per animal. The first version predicted fertilizer never
      moved (bias +$31, error identical to persistence).
    * FEED CONSUMES WHEAT. `FEED` takes one WHEAT per animal per day, so a
      herd of 18 eats 540 units a season. That is a demand sink, and omitting
      it over-predicted wheat supply (bias -$7.4).

    Crop yield is also read off the tile rather than assumed: a non-ongoing
    crop carries `yield_units` now and gains one per remaining watering day
    inside [(max_yield_day+1)//2, max_yield_day], capped at max_yield.
    """
    out = [{} for _ in range(horizon + 1)]

    def add(k, item, n):
        if 0 <= k <= horizon and n:
            out[k][item] = out[k].get(item, 0) + n

    n_animals = 0
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                held = t.get("yield_units", 0)
                if not cd["ongoing"]:
                    ws = (cd["max_yield_day"] + 1) // 2
                    remaining = max(0, cd["max_yield_day"] - max(age, ws - 1))
                    units = min(cd["max_yield"], held + remaining)
                    add(cd["max_yield_day"] - age, t["crop"], units)
                else:
                    add(0, t["crop"], held)
                    n_done = ((age - cd["first_yield_day"]) // cd["interval"] + 1
                              if age >= cd["first_yield_day"] else 0)
                    nxt = cd["first_yield_day"] + n_done * cd["interval"] - age
                    while n_done < cd["max_yield"] and nxt <= horizon:
                        add(nxt, t["crop"], 1)
                        n_done += 1
                        nxt += cd["interval"]
            elif t.get("animal"):
                n_animals += 1
                a = ANIMALS[t["animal"]]
                item = a["product"]
                add(0, item, t.get("yield_units", 0))
                # cared animals bank a bonus and yield 2 on a fed day
                rate = 1 + (1 if (t.get("pending_care_bonus", 0)
                                  or t.get("cared_today")) else 0)
                age = day - t["placed_day"]
                nxt = (a["interval"] - max(0, age - a["first_yield_day"]) % a["interval"]
                       if age >= a["first_yield_day"] else a["first_yield_day"] - age)
                while nxt <= horizon:
                    add(nxt, item, rate)
                    nxt += a["interval"]

    for k in range(1, horizon + 1):
        add(k, "FERTILIZER", n_animals)     # one per animal per night
        add(k, "WHEAT", -n_animals)         # FEED eats one WHEAT per animal
    return out


def growth_supply(farm, day, horizon, CROPS, ANIMALS):
    """Supply from tiles that are EMPTY NOW and will be planted.

    board_supply sees only what is already in the ground, so through a ramp
    it under-predicts badly -- our own agent goes from 3 to 33 standing
    strawberries between day 6 and day 13. This assumes every empty unlocked
    tile is filled with the farm's CURRENT crop mix and delivers on the normal
    clock from the day it is planted.
    """
    counts, empty = {}, 0
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                empty += 1
            elif isinstance(t, dict) and t.get("kind") == "PLANT":
                counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    total = sum(counts.values())
    out = [{} for _ in range(horizon + 1)]
    if not total or not empty:
        return out
    for crop, n in counts.items():
        cd = CROPS[crop]
        tiles = empty * n / total
        if cd["ongoing"]:
            # planted ~now, first yield at first_yield_day then every interval
            k = cd["first_yield_day"]
            i = 0
            while k <= horizon and i < cd["max_yield"]:
                out[k][crop] = out[k].get(crop, 0) + tiles
                k += cd["interval"]
                i += 1
        else:
            k = cd["max_yield_day"]
            while k <= horizon:
                out[k][crop] = out[k].get(crop, 0) + tiles * cd["max_yield"]
                k += cd["max_yield_day"] + 1      # replant cycle
    return out


def forecast(obs, seat, horizon, layer, K, opp_shed=None):
    """Price path per product, days t..t+horizon, at ablation `layer` (0-4)."""
    SHOPS, TOWN, CROPS, ANIMALS = K["SHOPS"], K["TOWN"], K["CROPS"], K["ANIMALS"]
    PRODUCTS, market_price = K["PRODUCTS"], K["market_price"]

    day = obs["day"]
    inv = dict(obs["market"]["inventory"])
    shops = list(obs["town"]["unlocked_shops"])
    n_future = max(0, N_SHOP_DRAWS - len(shops))

    mine = board_supply(obs["farms"][seat], day, horizon, CROPS, ANIMALS) \
        if layer >= 3 else None
    theirs = board_supply(obs["farms"][1 - seat], day, horizon, CROPS, ANIMALS) \
        if layer >= 4 else None
    # Shed stock: already harvested, waiting to be sold. Ours is in
    # observation.private; theirs is the ONE hidden variable in the game, so
    # layer 6 is an ORACLE -- an upper bound on what a belief model could buy.
    shed = {}
    if layer >= 5:
        for it, n in (obs.get("private", {}).get("shed", {}) or {}).items():
            shed[it] = shed.get(it, 0) + n
    if layer >= 6 and opp_shed:
        for it, n in opp_shed.items():
            shed[it] = shed.get(it, 0) + n

    path = {p: [market_price(p, inv[p])] for p in PRODUCTS}
    cur = dict(inv)
    for k in range(1, horizon + 1):
        for p in PRODUCTS:
            d = 0.0
            if layer >= 1:
                d -= shop_drain(shops, p, SHOPS, TOWN)
            if layer >= 2 and n_future:
                # draws land every SHOP_EVERY days; count how many by now
                landed = min(n_future, max(0, (day + k - 3) // SHOP_EVERY + 1
                                           - len(shops)))
                d -= expected_extra_drain(landed, p, SHOPS)
            if mine is not None:
                d += mine[k].get(p, 0)
            if theirs is not None:
                d += theirs[k].get(p, 0)
            if k == 1 and shed:
                # sheds clear over the next few days; release evenly over 3
                d += shed.get(p, 0) / 3.0
            elif k in (2, 3) and shed:
                d += shed.get(p, 0) / 3.0
            cur[p] = cur[p] + d
            path[p].append(market_price(p, int(round(cur[p]))))
    return path


# --------------------------------------------------------------------------
# accuracy harness
# --------------------------------------------------------------------------

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
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, CROPS, PRODUCTS, SHOPS, TOWN_CENTER_PRODUCTS, market_price)
    K = {"SHOPS": SHOPS, "TOWN": TOWN_CENTER_PRODUCTS, "CROPS": CROPS,
         "ANIMALS": ANIMALS, "PRODUCTS": PRODUCTS, "market_price": market_price}

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([_script(s["actions"][0]), _script(s["actions"][1])])

    dawn, dawn1 = {}, {}
    for st_ in env.steps:
        o = st_[0].observation
        if o["step"] % 24 == 0:
            dawn[o["step"] // 24] = o
            dawn1[o["step"] // 24] = st_[1].observation

    H = max(HORIZONS)
    err = {L: {k: [] for k in HORIZONS} for L in range(7)}
    # per product, layer 4 only: (abs error, signed error, persistence error)
    byp = {p: {k: [0.0, 0.0, 0.0, 0] for k in HORIZONS} for p in PRODUCTS}
    for day, o in sorted(dawn.items()):
        if day + H > 29:
            break
        truth = {p: [dawn[day + k]["market"]["prices"][p] for k in range(H + 1)]
                 for p in PRODUCTS}
        for L in range(7):
            f = forecast(o, 0, H, L, K,
                         opp_shed=dict(dawn1[day].get("private", {})
                                       .get("shed", {}) or {}))
            for k in HORIZONS:
                for p in PRODUCTS:
                    err[L][k].append(abs(f[p][k] - truth[p][k]))
                    if L == 6:
                        c = byp[p][k]
                        c[0] += abs(f[p][k] - truth[p][k])
                        c[1] += f[p][k] - truth[p][k]
                        c[2] += abs(truth[p][0] - truth[p][k])
                        c[3] += 1
    return {"episode": s["episode"],
            "err": {L: {k: (sum(v), len(v)) for k, v in d.items()}
                    for L, d in err.items()},
            "byp": byp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if v["scores"] and all(x is not None for x in v["scores"])
             and min(v["scores"]) >= args.min_rating]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"{len(paths)} sessions, both seats >= {args.min_rating:.0f}\n", flush=True)

    with mp.Pool(args.procs) as pool:
        rows = pool.map(job, paths)

    names = {0: "L0 persistence", 1: "L1 + known shops",
             2: "L2 + expected future draws", 3: "L3 + our board",
             4: "L4 + THEIR board", 5: "L5 + our shed",
             6: "L6 + their shed (ORACLE)"}
    print("mean absolute price error ($), pooled over all nine products\n")
    print(f"    {'layer':<30}" + "".join(f"{'k='+str(k):>9}" for k in HORIZONS))
    base = {}
    for L in range(7):
        cells = []
        for k in HORIZONS:
            tot = sum(r["err"][L][k][0] for r in rows)
            n = sum(r["err"][L][k][1] for r in rows)
            mae = tot / max(1, n)
            if L == 0:
                base[k] = mae
            cells.append(mae)
        print(f"    {names[L]:<30}" + "".join(f"{c:>9.1f}" for c in cells))
    print("\n  layer 4, per product: mean |error|, mean signed error (+ = we")
    print("  predict too HIGH), and persistence for comparison\n")
    print(f"    {'product':<12}" + "".join(f"{'k='+str(k)+' mae':>10}{'bias':>8}{'pers':>7}"
                                           for k in (3, 10)))
    for p in sorted(rows[0]["byp"]):
        cells = []
        for k in (3, 10):
            a = sum(r["byp"][p][k][0] for r in rows)
            b = sum(r["byp"][p][k][1] for r in rows)
            c = sum(r["byp"][p][k][2] for r in rows)
            n = max(1, sum(r["byp"][p][k][3] for r in rows))
            cells.append(f"{a/n:>10.1f}{b/n:>+8.1f}{c/n:>7.1f}")
        print(f"    {p:<12}" + "".join(cells))
    print(f"\n    {'improvement vs persistence':<30}"
          + "".join(f"{100*(1-  (sum(r['err'][6][k][0] for r in rows)/max(1,sum(r['err'][6][k][1] for r in rows)))/base[k]):>8.0f}%"
                    for k in HORIZONS))


if __name__ == "__main__":
    main()
