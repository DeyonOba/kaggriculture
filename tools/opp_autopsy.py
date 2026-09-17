"""Where do the public agents' extra $40k come from? Paired, same-world autopsy.

We have been decomposing our gap against the TAPE, which is a recording. The
larger and better-posed gap is against `pub-ahmed-v38`, a live agent that scores
~$111-134k on the same worlds where champ-h8 scores ~$72-84k. Against the LP
season bound ($147k competitive) ahmed runs at roughly 80% and we run at ~50%.

Both farms are fully public every step, so this needs no inference at all -- it
reads the two farms side by side out of the same episode and reports where the
trajectories separate. Unlike the tape comparison, this is a live closed-loop
opponent, so anything it does is reachable by a policy.

Reported per day, ours vs theirs:

    money       cash on hand
    land        unlocked tiles (the capital decision)
    hands       crew size (the labour decision)
    plants      standing crops / animals (the production base)
    sold        units sold that day and the revenue they fetched
    work        productive actions, moves, and PASSes

    PYTHONPATH=. uv run python tools/opp_autopsy.py --n 6
    PYTHONPATH=. uv run python tools/opp_autopsy.py --n 6 --opp pub-boatlee-v16
"""

import argparse
import collections
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from kaggle_environments import make  # noqa: E402

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def autopsy(agent, opp, seeds):
    """Per-day series for both seats, averaged over seeds."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    finals = [[], []]
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        # money_by_day[p][day] = last observed money that day
        day_money = [{}, {}]
        day_land = [{}, {}]
        day_hands = [{}, {}]
        day_plants = [{}, {}]
        day_animals = [{}, {}]
        work = [collections.Counter(), collections.Counter()]
        prev_money = [None, None]
        sold = [collections.Counter(), collections.Counter()]

        for stp in env.steps[:-1]:
            o = stp[0].observation
            day = o["day"]
            for p in (0, 1):
                farm = o["farms"][p]
                day_money[p][day] = farm["money"]
                tiles = farm["tiles"]
                day_land[p][day] = sum(
                    1 for row in tiles for t in row
                    if isinstance(t, dict) and t.get("unlocked")
                )
                day_hands[p][day] = 1 + len(farm.get("hands") or [])
                day_plants[p][day] = sum(
                    1 for row in tiles for t in row
                    if isinstance(t, dict) and t.get("kind") == "PLANT"
                )
                day_animals[p][day] = sum(
                    1 for row in tiles for t in row
                    if isinstance(t, dict) and t.get("animal")
                )
                a = stp[p].get("action") or {}
                for x in [a.get("farmer")] + list(a.get("hands") or []):
                    op = x[0] if isinstance(x, list) and x else "PASS"
                    work[p]["move" if op in MOVES else
                            ("pass" if op == "PASS" else "work")] += 1
                for order in (a.get("market") or []):
                    if isinstance(order, list) and order and order[0] == "SELL":
                        it = order[1] if len(order) > 1 else "?"
                        # Units AND orders: the price ladder means one 50-unit
                        # order walks the price down, while five 10-unit orders
                        # across days let town drain lift it back between sales.
                        # Lot size is the whole question, so count both.
                        sold[p][it] += int(order[2]) if len(order) > 2 else 0
                        sold[p]["#" + it] += 1

        last = env.steps[-1]
        for p in (0, 1):
            finals[p].append(last[p]["reward"])
            for day in sorted(day_money[p]):
                acc[p]["money"].append((day, day_money[p][day]))
                acc[p]["land"].append((day, day_land[p][day]))
                acc[p]["hands"].append((day, day_hands[p][day]))
                acc[p]["plants"].append((day, day_plants[p][day]))
                acc[p]["animals"].append((day, day_animals[p][day]))
            for k, v in work[p].items():
                acc[p]["_" + k].append((0, v))
            for it, n in sold[p].items():
                acc[p]["sell:" + it].append((0, n))
    return acc, finals


def series(acc, p, key):
    by_day = collections.defaultdict(list)
    for day, v in acc[p].get(key, []):
        by_day[day].append(v)
    return {d: st.mean(v) for d, v in by_day.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=71000)
    ap.add_argument("--selfplay", action="store_true",
                    help="also run each agent against a copy of itself")
    args = ap.parse_args()

    seeds = [args.base_seed + k for k in range(args.n)]
    acc, finals = autopsy(args.agent, args.opp, seeds)

    print(f"{args.agent} vs {args.opp}, {len(seeds)} seeds, same worlds\n")
    print(f"final: {args.agent} ${st.mean(finals[0]):,.0f}   "
          f"{args.opp} ${st.mean(finals[1]):,.0f}   "
          f"gap ${st.mean(finals[1]) - st.mean(finals[0]):+,.0f}\n")

    if args.selfplay:
        # Both farms sell into ONE market inventory, so part of their surplus is
        # taken FROM us rather than produced. Self-play separates the two: solo
        # capability is what each earns against a copy of itself; the rest of
        # the head-to-head gap is exploitation.
        _, mine = autopsy(args.agent, args.agent, seeds)
        _, theirs = autopsy(args.opp, args.opp, seeds)
        us_solo = st.mean(mine[0] + mine[1])
        them_solo = st.mean(theirs[0] + theirs[1])
        print(f"self-play: {args.agent} ${us_solo:,.0f}   "
              f"{args.opp} ${them_solo:,.0f}   "
              f"capability gap ${them_solo - us_solo:+,.0f}")
        h2h = st.mean(finals[1]) - st.mean(finals[0])
        print(f"           head-to-head gap ${h2h:+,.0f}, of which "
              f"${them_solo - us_solo:+,.0f} is capability and "
              f"${h2h - (them_solo - us_solo):+,.0f} is exploitation\n")

    keys = ["money", "land", "hands", "plants", "animals"]
    s = {(p, k): series(acc, p, k) for p in (0, 1) for k in keys}
    days = sorted(s[(0, "money")])
    hdr = "".join(f"{k:>21}" for k in keys)
    print(f"{'day':>4}{hdr}")
    print(f"{'':>4}" + "".join(f"{'us':>10}{'them':>11}" for _ in keys))
    for d in days:
        if d % 3 and d != days[-1]:
            continue
        row = f"{d:>4}"
        for k in keys:
            row += f"{s[(0,k)].get(d,0):>10,.0f}{s[(1,k)].get(d,0):>11,.0f}"
        print(row)

    print()
    for k in ("_work", "_move", "_pass"):
        a = sum(v for _, v in acc[0].get(k, [])) / len(seeds)
        b = sum(v for _, v in acc[1].get(k, [])) / len(seeds)
        print(f"{k[1:]:>8} per game: us {a:>8,.0f}   them {b:>8,.0f}   "
              f"{b - a:>+9,.0f}")

    print()
    items = sorted({k[5:] for p in (0, 1) for k in acc[p]
                    if k.startswith("sell:") and not k.startswith("sell:#")})
    print(f"{'item':>12} {'units us':>9} {'them':>8}   {'orders us':>10} {'them':>6}"
          f"   {'lot us':>7} {'them':>7}")
    for it in items:
        u = [sum(v for _, v in acc[p].get("sell:" + it, [])) / len(seeds)
             for p in (0, 1)]
        o = [sum(v for _, v in acc[p].get("sell:#" + it, [])) / len(seeds)
             for p in (0, 1)]
        lot = [u[p] / o[p] if o[p] else 0 for p in (0, 1)]
        print(f"{it:>12} {u[0]:>9,.0f} {u[1]:>8,.0f}   {o[0]:>10,.0f} {o[1]:>6,.0f}"
              f"   {lot[0]:>7,.1f} {lot[1]:>7,.1f}")


if __name__ == "__main__":
    main()
