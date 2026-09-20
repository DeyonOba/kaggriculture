"""The three phases have three different objectives. Measure each on its own.

We instrumented the OPENING all season and never built the other two. The
game is not one problem:

    early  days 0-8    place capital        investment r=+0.304, money -0.021
    mid    days 8-20   service it, and do   nothing built
                       not commit capital
                       that cannot pay back
    end    days 20-30  LIQUIDATE            nothing built

The endgame is the one we have never once measured, and the rule that makes
it matter is verified: **unsold inventory counts for zero** (README "Reward",
confirmed in tools/readme_audit.py). Every unit sitting in the shed at the
final bell is a total loss. And the premium lines cannot simply be dumped --
MELON above_func `sq` at target 3.6 and WOOL `sq` at 3.2 reach the $1 floor
by I0+T, so a large dump pays pennies.

Two engine facts shape the endgame and pull in opposite directions:

  AGAINST dumping  selling ADDS to market inventory, which drives the price
                   down along above_func. Premium lines hit $1 fast.
  FOR spreading    the town DRAINS inventory every day (town centre 1/product
                   /day, each shop instance 6/day, single-product shops 12),
                   so a price you crashed today partly recovers tomorrow.

Therefore the endgame is a scheduling problem with a known optimum shape,
and "how far from it are we" is a measurable number in dollars.

    PYTHONPATH=. uv run python tools/phase_metrics.py --n 24
"""
import argparse
import collections
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

CRASHY = {p for p, m in K.MARKET_PARAMS.items() if m["above_target"] > 1.0}
TPD = 24


# ----------------------------------------------------------------- mid phase
MID_NAMES = ["dead_capital", "dead_frac", "board_value", "board_roi",
             "payback_days", "wheat_days_left", "service_burden",
             "capital_at_risk", "marginal_tile_days"]


def _first_yield_left(t, day):
    """Days until this tile's FIRST product. Negative once it is producing."""
    if t.get("kind") == "PLANT":
        cd = K.CROPS[t["crop"]]
        return cd["first_yield_day"] - (day - t.get("planted_day", day))
    a = K.ANIMALS[t["animal"]]
    return a["first_yield_day"] - (day - t.get("placed_day", day))


def mid_features(obs, seat, private=None, season=30):
    """Investment RISK: capital that cannot pay back before the bell."""
    v = np.zeros(len(MID_NAMES), "float32")
    ix = {k: i for i, k in enumerate(MID_NAMES)}
    f = obs["farms"][seat]
    if private is None:
        private = (obs.get("private") or {})
    day = obs.get("day", obs["step"] // 24)
    left = max(0, season - day)
    prices = (obs.get("market") or {}).get("prices") or {}

    dead = invested = board = risk = 0.0
    horizons = []
    animals = 0
    for row in f["tiles"]:
        for t in row:
            if not isinstance(t, dict) or t.get("kind") == "WEED":
                continue
            if t.get("kind") == "PLANT":
                cost = K.CROPS[t["crop"]]["seed"]
                prod = t["crop"]
            elif t.get("animal"):
                cost = K.ANIMALS[t["animal"]]["cost"]
                prod = K.ANIMALS[t["animal"]]["product"]
                animals += 1
            else:
                continue
            invested += cost
            fy = _first_yield_left(t, day)
            horizons.append(fy)
            px = float(prices.get(prod, K.MARKET_PARAMS[prod]["base"]))
            # DEAD CAPITAL: the first product lands after the season ends, so
            # this money can never come back. Exactly computable, and the
            # single clearest mid-game error available.
            if fy > left:
                dead += cost
            else:
                board += t.get("yield_units", 0) * px
            if prod in CRASHY:
                risk += cost

    v[ix["dead_capital"]] = min(dead, 4000) / 4000.0
    v[ix["dead_frac"]] = dead / max(invested, 1.0)
    v[ix["board_value"]] = min(board, 20000) / 20000.0
    v[ix["board_roi"]] = min(board / max(invested, 1.0), 4.0) / 4.0
    v[ix["payback_days"]] = (min(float(np.mean(horizons)), 12) / 12.0
                             if horizons else 0.0)
    # SERVICE: animals eat 1 WHEAT each per day and escape after 2 unfed days.
    wheat = ((private.get("shed") or {}).get("WHEAT", 0)
             + sum((b or {}).get("WHEAT", 0)
                   for b in (private.get("inventories") or [])))
    v[ix["wheat_days_left"]] = (min(wheat / animals, 10) / 10.0
                                if animals else 1.0)
    v[ix["service_burden"]] = min(animals, 20) / 20.0
    v[ix["capital_at_risk"]] = risk / max(invested, 1.0)
    v[ix["marginal_tile_days"]] = min(left, 30) / 30.0
    return v


# ----------------------------------------------------------------- end phase
END_NAMES = ["unsold_units", "unsold_at_base", "unsold_if_dumped",
             "dump_loss_frac", "floor_frac", "days_left", "drain_cover",
             "liquidation_pressure"]


def town_drain(obs):
    """Units per DAY the town removes from market inventory, per product.

    Town centre: 1 of every product except fertilizer every
    townCenterSellInterval (24) turns = 1/day. Each unlocked shop INSTANCE:
    1 of each product it demands every townShopSellInterval (4) turns =
    6/day, and single-product shops consume 2x = 12/day.
    """
    d = collections.Counter()
    for p in K.PRODUCTS:
        if p != "FERTILIZER":
            d[p] += 1
    shops = ((obs.get("town") or {}).get("unlocked_shops") or [])
    for s in shops:
        items = K.SHOPS.get(s, [])
        rate = 12 if len(items) == 1 else 6
        for it in items:
            d[it] += rate
    return d


def sell_revenue(item, units, inv0, drain_per_day, days, schedule):
    """Walk the real price curve. `schedule` is units sold on each day.

    Mirrors the engine: price is quoted at PRE-sell inventory, the unit is
    paid for, and inventory rises by one ONLY if the price cleared $1 --
    at the floor the sale does not add supply (verified in readme_audit).
    """
    inv = inv0
    rev = 0.0
    floored = 0
    for t in range(days):
        for _ in range(int(schedule[t])):
            p = K.market_price(item, inv, None)
            rev += p
            if p > 1:
                inv += 1
            else:
                floored += 1
        inv = max(0, inv - drain_per_day)
    return rev, floored


def liquidation_gap(item, units, inv0, drain, days):
    """-> (dump_now, spread_even, best, best_name). Money left on the table."""
    if units <= 0 or days <= 0:
        return 0.0, 0.0, 0.0, "none"
    cand = {}
    s = np.zeros(days); s[-1] = units
    cand["dump_last"] = s.copy()
    s = np.zeros(days); s[0] = units
    cand["dump_now"] = s.copy()
    even = np.full(days, units / days)
    cand["spread_even"] = even
    # front-load and back-load, in case drain timing matters
    w = np.linspace(2.0, 0.5, days); cand["front"] = units * w / w.sum()
    w = np.linspace(0.5, 2.0, days); cand["back"] = units * w / w.sum()
    out = {}
    for name, sch in cand.items():
        sch = np.floor(sch).astype(int)
        sch[-1] += units - sch.sum()
        out[name] = sell_revenue(item, units, inv0, drain, days, sch)[0]
    best = max(out, key=out.get)
    return out["dump_now"], out["spread_even"], out[best], best


def end_features(obs, seat, private=None, season=30):
    v = np.zeros(len(END_NAMES), "float32")
    ix = {k: i for i, k in enumerate(END_NAMES)}
    if private is None:
        private = (obs.get("private") or {})
    day = obs.get("day", obs["step"] // 24)
    left = max(1, season - day)
    shed = (private.get("shed") or {})
    minv = ((obs.get("market") or {}).get("inventory") or {})
    drain = town_drain(obs)

    units = base = dumped = 0.0
    floored = 0
    for item, n in shed.items():
        if n <= 0 or item not in K.MARKET_PARAMS:
            continue
        units += n
        base += n * K.MARKET_PARAMS[item]["base"]
        inv0 = int(minv.get(item, K.MARKET_PARAMS[item]["I0"]))
        r, fl = sell_revenue(item, n, inv0, 0, 1, [n])
        dumped += r
        floored += fl
    v[ix["unsold_units"]] = min(units, 100) / 100.0
    v[ix["unsold_at_base"]] = min(base, 20000) / 20000.0
    v[ix["unsold_if_dumped"]] = min(dumped, 20000) / 20000.0
    v[ix["dump_loss_frac"]] = 1.0 - (dumped / base if base > 0 else 1.0)
    v[ix["floor_frac"]] = floored / max(units, 1)
    v[ix["days_left"]] = left / 30.0
    v[ix["drain_cover"]] = min(sum(drain.values()) * left / max(units, 1),
                               5.0) / 5.0
    v[ix["liquidation_pressure"]] = min(units / max(left, 1), 50) / 50.0
    return v


NAMES = MID_NAMES + END_NAMES


def features(obs, seat, private=None):
    return np.concatenate([mid_features(obs, seat, private),
                           end_features(obs, seat, private)])


def sale_timing(n_games=16, agent="champ-h8", base_seed=700001):
    """WHEN do we sell, against when each product is actually worth most?

    Hooks `_commit_unit` to record every completed sale, and samples the
    public price path at hour 12 each day. Both seats are recorded: this is
    self-play, so both are the same policy and it doubles the sample.

    THE CAVEAT, stated before the numbers. The gap here is NOT
    (peak - VWAP) x units. Two reasons, both recorded mistakes:
      - marginal is not average: our own VWAP already beats the price at our
        median sale day, because sales are spread and the early ones catch
        the better price.
      - the path is partly OUR doing. Moving every unit of wool to day 5
        would crash day-5 wool. The peak is not a price we could all get.
    So this measures DIRECTION -- which side of a cliff we sell on -- and the
    magnitude needs a paired gate, not a multiplication.
    """
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    sales, day, price = [], [0], collections.defaultdict(list)
    orig = K._commit_unit

    def hooked(op, item, pr, farm, private, market, shed_capacity=100):
        ok = orig(op, item, pr, farm, private, market, shed_capacity)
        if ok and op == "SELL":
            sales.append((day[0], item, float(pr)))
        return ok
    K._commit_unit = hooked
    try:
        def load(nm):
            g = {}
            exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
            return g["agent"]
        for s in range(base_seed, base_seed + n_games):
            a, b = load(agent), load(agent)
            e = make("kaggriculture", configuration={"seed": s}, debug=False)
            e.reset(2)
            while not e.done:
                o = e.state[0].observation
                day[0] = o["step"] // TPD
                if o["step"] % TPD == 12:
                    for pr_ in K.PRODUCTS:
                        price[(pr_, day[0])].append(o["market"]["prices"][pr_])
                e.step([a(o), b(e.state[1].observation)])
    finally:
        K._commit_unit = orig

    by = collections.defaultdict(list)
    for d, it, pr in sales:
        by[it].append((d, pr))
    print(f"\n{len(sales):,} sales over {n_games} games (both seats)\n")
    print(f"  {'item':<12}{'peak day':>9}{'peak $':>8}{'our med day':>13}"
          f"{'$ then':>8}{'our VWAP':>10}   timing")
    late = []
    for it in sorted(by, key=lambda k: -len(by[k])):
        path = {d: float(np.mean(price[(it, d)]))
                for d in range(30) if price[(it, d)]}
        if not path:
            continue
        pk = max(path, key=path.get)
        d = np.array([x[0] for x in by[it]])
        pr = np.array([x[1] for x in by[it]])
        med = int(np.median(d))
        off = med - pk
        tag = ("AT PEAK" if abs(off) <= 2
               else f"{abs(off)}d {'LATE' if off > 0 else 'EARLY'}")
        if off > 2:
            late.append((off, it))
        print(f"  {it:<12}{pk:>9}{path[pk]:>8.0f}{med:>13}"
              f"{path.get(med, float('nan')):>8.0f}{pr.mean():>10.0f}   {tag}")
    print(f"\n  price path (mean, hour 12)")
    cols = (3, 6, 9, 12, 15, 18, 21, 24, 27, 29)
    print(f"  {'item':<12}" + "".join(f"{'d'+str(c):>7}" for c in cols))
    for it in sorted(by, key=lambda k: -len(by[k])):
        path = {d: float(np.mean(price[(it, d)]))
                for d in range(30) if price[(it, d)]}
        print(f"  {it:<12}" + "".join(f"{path.get(c, 0):>7.0f}" for c in cols))
    if late:
        print(f"\n  SOLD LATE, worst first: "
              + ", ".join(f"{i} ({o}d)" for o, i in sorted(late, reverse=True)))
    return by, price


# ------------------------------------------------------------- measurement
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--sales", action="store_true",
                    help="measure WHEN we sell against each product's path")
    args = ap.parse_args()
    if args.sales:
        sale_timing(args.n, args.agent)
        return

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]

    rows = []
    for i in range(args.n):
        a0, a1 = load(args.agent), load(args.agent)
        e = make("kaggriculture", configuration={"seed": 700000 + i},
                 debug=False)
        e.reset(2)
        snap = {}
        while not e.done:
            o = e.state[0].observation
            # HOUR 12, not hour 0: at dawn hands and bags read 0 because
            # units respawn empty at the shed. Sampling at dawn is how a
            # previous pass concluded we never hire.
            if o["step"] % TPD == 12:
                d = o["step"] // TPD
                if d in (10, 15, 20, 25, 28, 29):
                    snap[d] = (mid_features(o, 0, o.private),
                               end_features(o, 0, o.private),
                               dict(o.private.get("shed") or {}),
                               dict((o.get("market") or {}).get("inventory")
                                    or {}),
                               town_drain(o))
            e.step([a0(o), a1(e.state[1].observation)])
        rows.append((snap, e.state[0].reward, e.state[1].reward))

    print(f"\n{len(rows)} games, {args.agent} self-play\n")
    print("=" * 74)
    print("  MID-GAME INVESTMENT RISK -- capital that cannot pay back")
    print("=" * 74)
    print(f"  {'day':>5}{'dead capital':>15}{'dead % of invested':>21}"
          f"{'board ROI':>12}{'wheat days':>12}")
    for d in (10, 15, 20, 25):
        v = [s[d][0] for s, _, _ in rows if d in s]
        if not v:
            continue
        V = np.stack(v)
        print(f"  {d:>5}{V[:,0].mean()*4000:>14,.0f}$"
              f"{V[:,1].mean()*100:>20.1f}%"
              f"{V[:,3].mean()*4:>12.2f}{V[:,5].mean()*10:>12.1f}")

    print()
    print("=" * 74)
    print("  END-GAME LIQUIDATION -- unsold inventory counts for ZERO")
    print("=" * 74)
    print(f"  {'day':>5}{'units held':>13}{'value at base':>16}"
          f"{'if dumped now':>16}{'dump loss':>12}")
    for d in (20, 25, 28, 29):
        v = [s[d][1] for s, _, _ in rows if d in s]
        if not v:
            continue
        V = np.stack(v)
        print(f"  {d:>5}{V[:,0].mean()*100:>13.0f}"
              f"{V[:,1].mean()*20000:>15,.0f}$"
              f"{V[:,2].mean()*20000:>15,.0f}$"
              f"{V[:,3].mean()*100:>11.1f}%")

    # the headline: what is actually thrown away, and what a schedule buys
    print()
    print("=" * 74)
    print("  WHAT THE ENDGAME COSTS US")
    print("=" * 74)
    tot_dump = tot_best = tot_base = 0.0
    per_item = collections.Counter()
    per_item_best = collections.Counter()
    for snap, r0, r1 in rows:
        if 25 not in snap:
            continue
        _, _, shed, minv, drain = snap[25]
        for item, n in shed.items():
            if n <= 0 or item not in K.MARKET_PARAMS:
                continue
            inv0 = int(minv.get(item, 10000))
            dump, even, best, name = liquidation_gap(item, int(n), inv0,
                                                     drain[item], 5)
            tot_dump += dump
            tot_best += best
            tot_base += n * K.MARKET_PARAMS[item]["base"]
            per_item[item] += dump
            per_item_best[item] += best
    ng = sum(1 for s, _, _ in rows if 25 in s)
    if ng:
        print(f"  holdings at day 25, liquidated over the last 5 days"
              f"   ({ng} games)\n")
        print(f"    dump it all at once        ${tot_dump/ng:>10,.0f} /game")
        print(f"    best simple schedule       ${tot_best/ng:>10,.0f} /game")
        print(f"    gap                        ${(tot_best-tot_dump)/ng:>10,.0f}"
              f" /game   <- scheduling alone")
        print(f"    (value at base price       ${tot_base/ng:>10,.0f} -- the"
              f" number a demand table would quote)")
        print(f"\n  {'item':<14}{'dumped':>12}{'scheduled':>12}{'gain':>10}")
        for it in sorted(per_item, key=lambda k: -(per_item_best[k]
                                                   - per_item[k])):
            g = (per_item_best[it] - per_item[it]) / ng
            if abs(g) < 1:
                continue
            print(f"    {it:<12}{per_item[it]/ng:>11,.0f}$"
                  f"{per_item_best[it]/ng:>11,.0f}${g:>9,.0f}$")

    fin = np.array([r0 for _, r0, _ in rows])
    print(f"\n  final bank: ${fin.mean():,.0f} +/- ${fin.std():,.0f}")


if __name__ == "__main__":
    main()
