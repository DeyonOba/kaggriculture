"""Who holds stock at the final bell, and what does it cost them?

Observed in a live game: both seats held strawberries and tomatoes back in the
closing turns. Unsold stock scores ZERO -- the reward is `farm["money"]` -- so
holding to the end is strictly a loss unless the price is expected to recover.
That makes it a war of attrition: whoever dumps first crashes the price and the
other sells into the wreckage, and if neither blinks they both burn the stock.

This measures the phenomenon across the recorded corpus instead of one game.

What is observable about the opponent's crop lifecycle, verified against the
engine:

    PLANTED    visible the turn it lands -- tiles[y][x] becomes a PLANT dict
               carrying `crop` and `planted_day`. The SEED PURCHASE before it
               is not: BUY_SEED touches neither market inventory nor the board.
    RIPENED    visible -- `yield_units` on the tile, plus the deterministic
               clock from planted_day.
    HARVESTED  inferable by differencing frames: a non-ongoing crop's tile
               becomes None (kaggriculture.py:468), an ongoing crop's
               yield_units drops to 0.
    SOLD       visible -- market inventory rises, minus our own sales.
    HELD       the shed, and the ONE hidden link in the chain.

So plant -> ripen -> harvest -> sell is fully trackable from public state; only
the seed buy and the shed balance are hidden, and the shed is
harvested-minus-sold, i.e. the residual of two observable series.

    PYTHONPATH=. uv run python tools/endgame_hold.py --n 120
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]



def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
    import tools.price_forecast as pf

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([pf._script(s["actions"][0]), pf._script(s["actions"][1])])

    last = env.steps[-1]
    inv = last[0].observation["market"]["inventory"]
    out = {"episode": s["episode"], "scores": s.get("scores"),
           "finals": [last[i].reward or 0.0 for i in (0, 1)], "seats": []}
    for i in (0, 1):
        shed = dict((last[i].observation.get("private") or {}).get("shed") or {})
        # value the stock at the price it would have fetched, one unit at a time
        # would be exact; the final marginal price is the optimistic bound.
        val = {p: shed.get(p, 0) * market_price(p, inv[p]) for p in PRODUCTS}
        # also: units still standing on the board, never harvested
        standing = 0
        for row in last[i].observation["farms"][i]["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    standing += t.get("yield_units", 0)
        out["seats"].append({"shed": shed, "value": val, "standing": standing})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--min-rating", type=float, default=0.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    rows0 = [v for v in idx.values()
             if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(rows0)
    paths = [v["path"] for v in rows0[: args.n]]
    print(f"replaying {len(paths)} sessions ...\n", flush=True)
    with mp.Pool(args.procs) as pool:
        rows = [r for r in pool.map(job, paths) if r]

    n = 2 * len(rows)
    print(f"{len(rows)} games, {n} seats. UNSOLD at the final bell:\n")
    print(f"    {'product':<12}{'units/seat':>12}{'$ stranded':>13}"
          f"{'% of seats holding any':>24}")
    tot = 0.0
    for p in PRODUCTS:
        u = sum(s["shed"].get(p, 0) for r in rows for s in r["seats"]) / n
        v = sum(s["value"].get(p, 0) for r in rows for s in r["seats"]) / n
        h = sum(1 for r in rows for s in r["seats"]
                if s["shed"].get(p, 0) > 0) / n
        tot += v
        print(f"    {p:<12}{u:>12.1f}{v:>13,.0f}{100*h:>23.0f}%")
    bank = sum(f for r in rows for f in r["finals"]) / n
    print(f"    {'TOTAL':<12}{'':>12}{tot:>13,.0f}"
          f"   = {100*tot/max(1,bank):.1f}% of a {bank:,.0f} bank")
    st = sum(s["standing"] for r in rows for s in r["seats"]) / n
    print(f"\n  plus {st:.1f} units/seat still STANDING on the board, never "
          f"harvested")

    # does holding track skill?
    band = {}
    for r in rows:
        for i, s in enumerate(r["seats"]):
            k = int((r["scores"][i] or 0) // 500 * 500)
            band.setdefault(k, []).append(sum(s["value"].values()))
    print(f"\n  stranded $ by rating band:")
    for k in sorted(band):
        v = band[k]
        print(f"    {k:>5}-{k+499:<5} n={len(v):>4}   ${sum(v)/len(v):>9,.0f}")


if __name__ == "__main__":
    main()
