"""One traversal, every metric. See docs/metric_harvest_spec.md.

Fifteen-plus tools each replay the corpus to compute one slice of the same
state. Every number in the metric hierarchy was taken on 40-400 games, which
is 0.2-1.9% of the 21,054 sessions indexed. This records everything once so
each of those becomes a query on a table instead of a 2.4-hour replay.

WHAT IT RECORDS AND HOW, because attribution is the whole difficulty:

  emitted actions   from the recorded action list, classified EXACTLY as
                    tools/haul_probe.py does (NORTH/SOUTH/EAST/WEST -> move,
                    PASS -> pass, else work). Matching that classification is
                    what makes moves/work comparable to the published 1.43.

  plant / harvest   `_apply_unit_action` receives `private`, which carries
                    `_seat`, so the seat is exact. The tile state is compared
                    before and after, because an illegal action is a silent
                    no-op and an emitted PLANT that did nothing must not count
                    as a planting. HARVEST is detected by `yield_units`
                    FALLING rather than by the tile clearing -- the engine
                    only removes the tile for crops that are not `ongoing`,
                    and the same branch collects milk, wool and egg from
                    animals. See _patch for what that cost to learn.

  crop deaths       split by CAUSE, and the two causes live in different
                    places:
                      DROUGHT  consecutive_unwatered >= 2, in _end_of_day
                      EXPIRY   lifespan run out, in _decay_plants -- which
                               the engine calls in the per-STEP loop
                    Both are pure waste and nothing currently measures
                    either. Note consecutive_unwatered starts at 1 on the
                    planting day and death is at >= 2, so a crop planted and
                    not watered before the next rollover dies immediately.

  market            `_commit_unit`, the hook price_forecast already proves,
                    with SELL and BUY_PRODUCT kept APART rather than netted.

  livestock removal is not recorded because it cannot happen: no engine path
  clears a tile's animal. Once bought it stands for the rest of the game --
  which is also why k_ablation found animal counts gain +0.00 from history.

    PYTHONPATH=. uv run python tools/metric_harvest.py --n 20 --validate
    PYTHONPATH=. uv run python tools/metric_harvest.py --shard 0/8 --out X.npz
"""

import argparse
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
CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_L = ["GOOSE", "COW", "SHEEP"]
KINDS = CROPS_L + ANIMALS_L
VERBS = ["WATER", "FEED", "CARE", "PICKUP", "PLANT", "HARVEST", "SELL",
         "HIRE", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND"]
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")

D, S = 31, 2
P, K, C = len(PRODUCTS), len(KINDS), len(CROPS_L)
NV = len(VERBS) + 2                       # + _MOVE, _PASS
I0 = 10000
CAP_FRAC, CAP_MAX = 0.6, 2000


def _rating_ok(v, floor):
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0


def _script(actions):
    st = {"t": 0}

    def agent(obs):
        t = obs.get("step", st["t"])
        st["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def _tile_kind(t):
    """-> index into KINDS, or -1."""
    if not isinstance(t, dict):
        return -1
    if t.get("kind") == "PLANT":
        c = t.get("crop")
        return KINDS.index(c) if c in KINDS else -1
    a = t.get("animal")
    return KINDS.index(a) if a in KINDS else -1


def _grid(farm):
    return [[_tile_kind(t) for t in row] for row in farm["tiles"]]


def _patch(rec, state):
    """Install every hook. `rec` is mutated in place; `state` carries the day.

    Two things here were wrong in the first cut and the validation pass caught
    both, which is what it exists for.

    HARVEST is detected by `yield_units` FALLING, not by the tile clearing.
    The engine sets yield_units to 0 and only removes the tile when the crop
    is not `ongoing`, so a grid diff misses every strawberry and tomato
    harvest -- the first run reported 67.3 strawberries planted, 0 harvested,
    and 401.5 sold. The same branch handles `elif "animal" in tile`, so
    reading yield_units also catches milk, wool and egg collection, which a
    grid diff could never see at all.

    DEATH BY LIFESPAN is hooked on `_decay_plants`, which the engine calls in
    the per-STEP loop, not inside `_end_of_day`. Hooking only end-of-day
    reported zero expiries for every crop. Its call site is
    `for farm in obs0.farms`, with no seat index, so the seat comes from a
    farm-identity registry that `unit` populates -- the engine calls
    _apply_unit_action for each seat's farmer every step, so the registry is
    always warm before decay runs.

    With decay hooked separately, every PLANT removal seen at end of day is
    DROUGHT (`_daily_refresh_plants`); `_spawn_weeds` only touches empty
    tiles, so it filters out on `before < 0`.
    """
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _farmer_position)

    o_init = getattr(K, "_mh_init", None) or K._initialize
    o_unit = getattr(K, "_mh_unit", None) or K._apply_unit_action
    o_commit = getattr(K, "_mh_commit", None) or K._commit_unit
    o_eod = getattr(K, "_mh_eod", None) or K._end_of_day
    o_decay = getattr(K, "_mh_decay", None) or K._decay_plants
    K._mh_init, K._mh_unit, K._mh_commit = o_init, o_unit, o_commit
    K._mh_eod, K._mh_decay = o_eod, o_decay
    reg = state.setdefault("reg", {})

    def init(st, env):
        o_init(st, env)
        reg.clear()
        for i, s_ in enumerate(st):
            s_.observation.private["_seat"] = i

    def unit(farm, private, idx, action, board_size, day, turns_per_day,
             shed_capacity=100):
        seat = private.get("_seat", -1)
        if seat >= 0:
            reg[id(farm)] = seat
        op = action[0] if isinstance(action, list) and action else None
        pre = None
        if op in ("PLANT", "HARVEST") and seat >= 0:
            pos = _farmer_position(farm, idx)
            if pos is not None:
                fx, fy = pos[0], pos[1]
                t = farm["tiles"][fy][fx]
                pre = (fx, fy, _tile_kind(t),
                       t.get("yield_units", 0) if isinstance(t, dict) else 0)
        o_unit(farm, private, idx, action, board_size, day, turns_per_day,
               shed_capacity)
        if pre is None:
            return
        fx, fy, k0, y0 = pre
        t = farm["tiles"][fy][fx]
        k1 = _tile_kind(t)
        y1 = t.get("yield_units", 0) if isinstance(t, dict) else 0
        d = min(day, D - 1)
        # An illegal action is a silent no-op, so require the state to move.
        if op == "PLANT" and k1 >= 0 and k0 < 0:
            rec["add"][d, seat, k1] += 1
        elif op == "HARVEST" and k0 >= 0 and y0 > y1:
            rec["harv"][d, seat, k0] += 1
            rec["harvu"][d, seat, k0] += (y0 - y1)

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = o_commit(op, item, price, farm, private, market, shed_capacity)
        if not ok:
            return ok
        seat = private.get("_seat", -1)
        d = min(state["day"], D - 1)
        if seat < 0:
            return ok
        if op == "SELL" and price > 1 and item in PRODUCTS:
            pi = PRODUCTS.index(item)
            rec["sell"][d, seat, pi] += 1
            rec["rev"][d, seat, pi] += price
        elif op == "BUY_PRODUCT" and item in PRODUCTS:
            rec["buy"][d, seat, PRODUCTS.index(item)] += 1
        elif op == "BUY_ANIMAL" and item in KINDS:
            rec["add"][d, seat, KINDS.index(item)] += 1
        return ok

    def decay(farm, step):
        seat = reg.get(id(farm), -1)
        before = _grid(farm) if seat >= 0 else None
        o_decay(farm, step)
        if before is None:
            return
        after = _grid(farm)
        d = min(step // 24, D - 1)
        for y in range(len(after)):
            for x in range(len(after[y])):
                b = before[y][x]
                if 0 <= b < C and after[y][x] != b:
                    rec["expire"][d, seat, b] += 1

    def eod(st, env, day):
        obs0 = st[0].observation
        before = [_grid(f) for f in obs0.farms]
        o_eod(st, env, day)
        after = [_grid(f) for f in obs0.farms]
        d = min(day, D - 1)
        for seat in range(min(S, len(after))):
            for y in range(len(after[seat])):
                for x in range(len(after[seat][y])):
                    b = before[seat][y][x]
                    if 0 <= b < C and after[seat][y][x] != b:
                        rec["drought"][d, seat, b] += 1

    K._initialize, K._apply_unit_action = init, unit
    K._commit_unit, K._end_of_day, K._decay_plants = commit, eod, decay


def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kagent import econ
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        MARKET_PARAMS)
    from tools.market_clock import daily_drain

    try:
        s = json.load(open(path))
    except Exception:
        return None

    rec = {k: np.zeros((D, S, n), "float32") for k, n in
           (("add", K), ("harv", K), ("harvu", K), ("drought", C),
            ("expire", C),
            ("sell", P), ("buy", P), ("rev", P))}
    st = {"day": 0}
    _patch(rec, st)

    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = _script(s["actions"][0]), _script(s["actions"][1])

    act = np.zeros((S, NV), "float32")
    for seat in range(S):
        for turn in s["actions"][seat] or []:
            if not turn:
                continue
            for x in [turn.get("farmer")] + list(turn.get("hands") or []):
                if not x:
                    continue
                op = x[0]
                if op in MOVES:
                    act[seat, len(VERBS)] += 1
                elif op == "PASS":
                    act[seat, len(VERBS) + 1] += 1
                elif op in VERBS:
                    act[seat, VERBS.index(op)] += 1
            for m in (turn.get("market") or []):
                if m and m[0] in VERBS:
                    act[seat, VERBS.index(m[0])] += 1

    day_mkt = np.zeros((D, P), "float32")
    day_pri = np.zeros((D, P), "float32")
    day_cap = np.zeros((D, P), "float32")
    day_shop = np.zeros(D, "float32")
    seat_day = np.zeros((D, S, 4), "float32")       # money, quads, hands, shed_tot
    seat_kind = np.zeros((D, S, K), "float32")
    seat_shed = np.zeros((D, S, P), "float32")
    seen = np.zeros(D, "float32")
    floors = {p: max(1, int(CAP_FRAC * MARKET_PARAMS[p]["base"]))
              for p in PRODUCTS}
    lastshops = []

    while not env.done:
        o = env.state[0].observation
        st["day"] = o["step"] // 24
        if o["step"] % 24 == 0:
            d = min(st["day"], D - 1)
            seen[d] = 1.0
            inv, pri = o["market"]["inventory"], o["market"]["prices"]
            for pi, p in enumerate(PRODUCTS):
                day_mkt[d, pi] = inv[p]
                day_pri[d, pi] = pri[p]
                day_cap[d, pi] = econ.units_sellable_above(
                    p, inv[p], floors[p], cap=CAP_MAX)
            lastshops = o["town"]["unlocked_shops"]
            day_shop[d] = len(lastshops)
            for seat in range(S):
                f = o["farms"][seat]
                seat_day[d, seat, 0] = f["money"]
                seat_day[d, seat, 1] = len(f["unlocked_quadrants"])
                cnt = np.zeros(K, "float32")
                for row in f["tiles"]:
                    for t in row:
                        ki = _tile_kind(t)
                        if ki >= 0:
                            cnt[ki] += 1
                seat_kind[d, seat] = cnt
                sh = (env.state[seat].observation.get("private") or {}
                      ).get("shed") or {}
                v = [float(sh.get(p, 0)) for p in PRODUCTS]
                seat_shed[d, seat] = v
                seat_day[d, seat, 3] = sum(v)
        env.step([a0(o), a1(env.state[1].observation)])

    regen = np.asarray([daily_drain(lastshops, p) for p in PRODUCTS], "float32")
    fin = s.get("finals") or [0, 0]
    sc = s.get("scores")
    rated = 1.0 if (sc and all(x is not None for x in sc)) else 0.0
    game = np.asarray([
        float(s.get("seed") or 0), rated,
        float(min(sc)) if rated else 0.0, float(max(sc)) if rated else 0.0,
        float(fin[0]), float(fin[1]),
        1.0 if fin[0] > fin[1] else 0.0, float(fin[0] - fin[1]),
    ], "float32")

    return dict(game=game, day_mkt=day_mkt, day_pri=day_pri, day_cap=day_cap,
                day_regen=regen, day_shop=day_shop, seen=seen,
                seat_day=seat_day, seat_kind=seat_kind, seat_shed=seat_shed,
                seat_act=act, **rec)


def _paths(args):
    idx = json.load(open(INDEX))
    items = sorted(idx.items())
    paths = [v["path"] for _, v in items if _rating_ok(v, args.min_rating)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        paths = paths[i::n]
    return paths[: args.n] if args.n else paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = the whole shard")
    ap.add_argument("--shard", default="", help="i/n")
    ap.add_argument("--min-rating", type=float, default=0.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    paths = _paths(args)
    print(f"traversing {len(paths):,} sessions ...", flush=True)
    with mp.Pool(args.procs, maxtasksperchild=50) as pool:
        rows = [r for r in pool.imap_unordered(job, paths, chunksize=4) if r]
    print(f"{len(rows):,} succeeded\n", flush=True)

    out = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    if args.out:
        np.savez_compressed(args.out, **out)
        print(f"-> {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")
    if args.validate:
        report(out)


def report(o):
    """Recompute published numbers. If these do not match, the harvest is wrong."""
    n = len(o["game"])
    seen = o["seen"]
    print(f"=== {n} games ===\n")

    print("L6 action census (compare: corpus work 2925-3374, moves 3075-3498,")
    print("    moves/work 0.91-1.20, pickup/feed 0.52-0.63)\n")
    a = o["seat_act"].reshape(-1, NV)
    work = a[:, :len(VERBS)].sum(1) - a[:, VERBS.index("BUY_LAND")]
    mv, pk, fd = a[:, len(VERBS)], a[:, VERBS.index("PICKUP")], a[:, VERBS.index("FEED")]
    print(f"    work {work.mean():>8.0f}   moves {mv.mean():>8.0f}   "
          f"moves/work {mv.sum()/max(1,work.sum()):>5.2f}   "
          f"pickup/feed {pk.sum()/max(1,fd.sum()):>5.2f}\n")

    print("L2 capacity + glut day (compare: fertilizer d2, wool d7,")
    print("    melon/milk d11, strawberry d23, wheat/carrot/tomato/egg never)\n")
    print(f"    {'product':<12}{'cap d0':>9}{'cap d20':>9}{'regen':>8}"
          f"{'glut day':>10}{'% games':>9}{'utilisation':>13}")
    for pi, p in enumerate(PRODUCTS):
        c0 = o["day_cap"][:, 0, pi].mean()
        c20 = o["day_cap"][:, min(20, D - 1), pi].mean()
        rg = o["day_regen"][:, pi].mean()
        over = (o["day_mkt"][:, :, pi] > I0) & (seen > 0)
        gl = [np.argmax(r) if r.any() else np.nan for r in over]
        g = np.nanmean(gl) if not np.all(np.isnan(gl)) else np.nan
        sold = o["sell"][:, :, :, pi].sum((1, 2)).mean()
        season = c0 + 30 * rg
        frac = float(np.mean([bool(r.any()) for r in over]))
        print(f"    {p:<12}{c0:>9.0f}{c20:>9.0f}{rg:>8.1f}"
              + (f"{g:>10.1f}" if not np.isnan(g) else f"{'never':>10}")
              + f"{100*frac:>9.0f}%"
              + f"{100*sold/max(1.0,season):>12.0f}%")

    print("\nNEW — crop exits per game per seat, by cause\n")
    print("    ONGOING crops (strawberry, tomato) yield repeatedly and EXPIRY")
    print("    is their natural end of life, not waste. For one-shot crops an")
    print("    expiry is a plant that was never harvested -- pure loss.\n")
    print(f"    {'crop':<12}{'planted':>9}{'harvests':>10}{'units':>9}"
          f"{'DROUGHT':>9}{'EXPIRY':>8}{'lost/planted':>14}")
    for ci, c in enumerate(CROPS_L):
        pl = o["add"][:, :, :, ci].sum((1, 2)).mean()
        hv = o["harv"][:, :, :, ci].sum((1, 2)).mean()
        hu = o["harvu"][:, :, :, ci].sum((1, 2)).mean()
        dr = o["drought"][:, :, :, ci].sum((1, 2)).mean()
        ex = o["expire"][:, :, :, ci].sum((1, 2)).mean()
        ongoing = hv > pl * 1.2
        lost = dr if ongoing else dr + ex
        print(f"    {c:<12}{pl:>9.1f}{hv:>10.1f}{hu:>9.1f}{dr:>9.1f}{ex:>8.1f}"
              f"{100*lost/max(1e-9, pl):>13.0f}%"
              + ("  (ongoing)" if ongoing else ""))
    print(f"\n    animal collection (units): "
          + "  ".join(
              f"{k.lower()} {o['harvu'][:, :, :, K - 3 + i].sum((1,2)).mean():.0f}"
              for i, k in enumerate(ANIMALS_L)))

    print(f"\n    livestock bought/game/seat: "
          + "  ".join(f"{k.lower()} {o['add'][:, :, :, K - 3 + i].sum((1,2)).mean():.1f}"
                      for i, k in enumerate(ANIMALS_L))
          + "   (never removed -- no engine path clears an animal)")

    print("\nL5 gross sells vs buys (netting these erased 18% of commits)\n")
    print(f"    {'product':<12}{'sells':>10}{'buys':>10}{'buy share':>11}")
    for pi, p in enumerate(PRODUCTS):
        sl = o["sell"][:, :, :, pi].sum((1, 2)).mean()
        by = o["buy"][:, :, :, pi].sum((1, 2)).mean()
        print(f"    {p:<12}{sl:>10.1f}{by:>10.1f}"
              f"{100*by/max(1e-9,sl+by):>10.1f}%")

    print("\nL4 quadrant timing (unexplored)\n")
    # seat_day is (N, D, S, 4): reduce over D, the DAY axis. Reducing over S
    # instead returned argmax of a length-2 axis, i.e. "day 0" for everything.
    q = o["seat_day"][:, :, :, 1]                      # (N, D, S)
    marg = o["game"][:, 7]
    for target in (2, 3, 4):
        hit = q >= target                              # (N, D, S)
        ever = hit.any(1)                              # (N, S)
        when = np.where(ever, hit.argmax(1), np.nan)   # (N, S)
        # seat 0 is the seat `margin` is signed for, so correlate that one
        w = when[:, 0]
        ok = ~np.isnan(w)
        r = (np.corrcoef(w[ok], marg[ok])[0, 1]
             if ok.sum() > 2 and np.ptp(w[ok]) > 0 else np.nan)
        print(f"    day reaching {target} quadrants: "
              f"{100*ok.mean():>3.0f}% of seats, mean day "
              f"{np.nanmean(w):>5.1f}   corr with OUR margin {r:>+6.3f}")


if __name__ == "__main__":
    main()
