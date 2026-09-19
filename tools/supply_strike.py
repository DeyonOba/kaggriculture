"""Oracle bound on MARKET DENIAL: suppressing a price recovery they are waiting on.

Episode 110472097 (QQ 124,552 vs Sida Zuo 122,435) showed the pattern exactly.
Strawberry fell 178 -> 66 -> 18 -> 1 by day 23. Both seats then STOPPED
selling and sat on 42 and 41 units while the town drain pulled inventory back
down; the price recovered 1 -> 11 -> 30 -> 64 and both sold into it, ending
with nothing stranded. `tools/endgame_hold.py` missed this because it looked at
the final bell: holding is not stranding.

It is a supply strike and it is individually rational -- selling at the $1
floor is worthless, so waiting dominates. Which is exactly what makes it
attackable: their held stock is only worth anything IF the price recovers, and
a recovery is just the drain outrunning supply. Keep selling through the
trough and the recovery never arrives. Our own units were worth ~$1 either way,
so the cost is near zero and their entire held position stays at the floor.

This measures the prize with an ORACLE on their shed -- the same discipline
that retired the shed-as-a-feature branch. Per game and product:

    trough      the price falls below `--floor-frac` x base
    recovery    it later rises back above `--recover-frac` x base
    exposure    units each seat holds at the trough (shed + ripe on board)
    prize       their recovery value, which denial removes
    cost        ours, which denial also removes -- we sell at the floor instead

    margin swing = their recovery value - our recovery value

Positive means denial wins the GAME even if it loses us money, which is the
objective that matters: a win is our bank above theirs.

    PYTHONPATH=. uv run python tools/supply_strike.py --n 120
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

def job(spec):
    path, floor_frac, rec_frac = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        ANIMALS, MARKET_PARAMS, market_price)
    import tools.price_forecast as pf

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([pf._script(s["actions"][0]), pf._script(s["actions"][1])])

    dawn = [(st[0].observation, st[1].observation) for st in env.steps
            if st[0].observation["step"] % 24 == 0]

    def held(o_self, o_any, seat):
        """shed (ORACLE) + ripe units standing on that seat's board."""
        sh = dict((o_self.get("private") or {}).get("shed") or {})
        out = dict.fromkeys(PRODUCTS, 0)
        for p in PRODUCTS:
            out[p] += sh.get(p, 0)
        for row in o_any["farms"][seat]["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                n = t.get("yield_units", 0)
                if not n:
                    continue
                if t.get("kind") == "PLANT":
                    out[t["crop"]] = out.get(t["crop"], 0) + n
                elif t.get("animal"):
                    out[ANIMALS[t["animal"]]["product"]] += n
        return out

    events = []
    for p in PRODUCTS:
        base = MARKET_PARAMS[p]["base"]
        px = [market_price(p, o0["market"]["inventory"][p]) for o0, _ in dawn]
        i = 0
        while i < len(px):
            if px[i] <= floor_frac * base:
                j = i
                while j < len(px) and px[j] <= rec_frac * base:
                    j += 1
                if j < len(px):                     # a recovery actually happened
                    trough = min(range(i, j), key=lambda k: px[k])
                    o0, o1 = dawn[trough]
                    h0 = held(o0, o0, 0)[p]
                    h1 = held(o1, o0, 1)[p]
                    events.append({
                        "product": p, "trough_day": trough, "rec_day": j,
                        "px_trough": px[trough], "px_rec": px[j],
                        "ours": h0, "theirs": h1,
                        "our_gain": h0 * (px[j] - px[trough]),
                        "their_gain": h1 * (px[j] - px[trough]),
                    })
                i = j + 1
            else:
                i += 1
    return {"episode": s["episode"], "scores": s.get("scores"),
            "finals": [env.steps[-1][i].reward or 0.0 for i in (0, 1)],
            "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--floor-frac", type=float, default=0.25)
    ap.add_argument("--recover-frac", type=float, default=0.55)
    args = ap.parse_args()

    import random
    idx = _load_index()
    rows0 = [v for v in idx.values()
             if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(rows0)
    paths = [v["path"] for v in rows0[: args.n]]
    print(f"replaying {len(paths)} sessions rated >= {args.min_rating:.0f}; "
          f"trough <= {args.floor_frac:.0%} of base, recovery >= "
          f"{args.recover_frac:.0%}\n", flush=True)
    with mp.Pool(args.procs) as pool:
        rows = [r for r in pool.map(job, [(p, args.floor_frac,
                                           args.recover_frac) for p in paths])]

    ev = [e for r in rows for e in r["events"]]
    n = len(rows)
    print(f"{len(ev)} crash-and-recover events across {n} games "
          f"({len(ev)/n:.2f} per game)\n")
    print(f"    {'product':<12}{'events':>8}{'trough$':>9}{'recov$':>8}"
          f"{'days':>6}{'our units':>11}{'their units':>13}"
          f"{'margin swing':>14}")
    by = {}
    for e in ev:
        by.setdefault(e["product"], []).append(e)
    tot = 0.0
    for p in PRODUCTS:
        v = by.get(p)
        if not v:
            continue
        sw = sum(e["their_gain"] - e["our_gain"] for e in v) / n
        tot += sw
        print(f"    {p:<12}{len(v):>8}"
              f"{sum(e['px_trough'] for e in v)/len(v):>9.0f}"
              f"{sum(e['px_rec'] for e in v)/len(v):>8.0f}"
              f"{sum(e['rec_day']-e['trough_day'] for e in v)/len(v):>6.1f}"
              f"{sum(e['ours'] for e in v)/len(v):>11.1f}"
              f"{sum(e['theirs'] for e in v)/len(v):>13.1f}"
              f"{sw:>+14,.0f}")
    print(f"    {'TOTAL':<12}{len(ev):>8}{'':>9}{'':>8}{'':>6}{'':>11}{'':>13}"
          f"{tot:>+14,.0f}  per game")

    margins = [abs(r["finals"][0] - r["finals"][1]) for r in rows]
    margins.sort()
    med = margins[len(margins) // 2]
    print(f"\n  median winning margin in these games: ${med:,.0f}")
    print(f"  denial prize as a share of it: {100*abs(tot)/max(1,med):.0f}%")
    flip = sum(1 for r in rows
               if abs(r["finals"][0] - r["finals"][1]) <
               sum(e["their_gain"] - e["our_gain"] for e in r["events"]))
    print(f"  games where the swing EXCEEDS the winning margin: "
          f"{flip}/{n} = {100*flip/n:.0f}%")


if __name__ == "__main__":
    main()
