"""Endgame & banking scorecard: what the S0 stage table does not see.

Per day over a window (default 20-29), for the LEADERS (winner seat of a
replay glob) and for OUR agent (an rlv3 arm string played in-process on
held-out tape worlds, the paired instrument's own draw):
    ops        PLANT by crop, WATER, FERTILIZE, HARVEST, SELL orders
    cash flow  money at the next dawn minus this dawn
    held       units in shed + pockets at dawn
    sell units SELL units REQUESTED (requests, not fills: the engine fills
               unit by unit at the running price)
and at the end: seeds left unplanted, units left unsold valued at final
prices, and the final day's SELL requests by hour block and by product.

    PYTHONPATH=. uv run python tools/endgame_scorecard.py \\
        --replays "replays/now/*/*.json" --games 4 --arm base \\
        --flags tend_end,late_plant

Measured 2026-09-29 (memory: kaggriculture-endgame-flags-null): the leaders
fertilise 16/14/1 times on days 27/28/29 to the champion's 0/0/0, their
harvests rise 33/39/37 to our 30/23/9, they strand 0.7 seeds to our 11.8,
and their days-20-29 cash flow is $52.8k to our $45.8k. Two liquidation
switches (tend_end, late_plant) gated NULL against that: the difference is
their late wheat/carrot pipeline and the labour to tend it.
"""
from __future__ import annotations
import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OPS = ["PLANT:WHEAT", "PLANT:CARROT", "PLANT:STRAWBERRY", "WATER", "FERTILIZE", "HARVEST", "SELL"]


def _new():
    return {"ops": collections.defaultdict(collections.Counter), "flow": collections.Counter(),
            "held": collections.Counter(), "sell_units": collections.Counter(),
            "seeds": collections.Counter(), "left_units": 0.0, "left_value": 0.0,
            "byhour": collections.Counter(), "byprod": collections.Counter(), "n": 0}


def scan(steps, seat, acc, days):
    """Fold one game's steps (list of [agent0, agent1] dicts) for `seat` into acc."""
    dawn_money = {}
    for t, st in enumerate(steps):
        day, hour = t // 24, t % 24
        obs = st[seat].get("observation") or {}
        priv = obs.get("private") or {}
        farms = obs.get("farms") or []
        me = farms[seat] if len(farms) > seat else {}
        if hour == 0 and me:
            dawn_money[day] = me.get("money", 0)
            shed = sum((priv.get("shed") or {}).values())
            pockets = sum(sum(i.values()) for i in (priv.get("inventories") or []) if isinstance(i, dict))
            acc["held"][day] += shed + pockets
        a = st[seat].get("action") or {}
        if day in days:
            for u in [a.get("farmer") or []] + list(a.get("hands") or []):
                if u:
                    acc["ops"][day][u[0] + (":" + str(u[1]) if u[0] == "PLANT" and len(u) > 1 else "")] += 1
            for m in a.get("market") or []:
                if m and m[0] == "SELL" and len(m) > 2:
                    acc["ops"][day]["SELL"] += 1
                    acc["sell_units"][day] += int(m[2])
                    if day == 29:
                        acc["byhour"][hour // 6] += int(m[2])
                        acc["byprod"][m[1]] += int(m[2])
    last = steps[-1][seat].get("observation") or {}
    farms = last.get("farms") or []
    final_money = (farms[seat] or {}).get("money", 0) if len(farms) > seat else 0
    for d in days:
        nxt = dawn_money.get(d + 1, final_money if d == 29 else None)
        if d in dawn_money and nxt is not None:
            acc["flow"][d] += nxt - dawn_money[d]
    priv = last.get("private") or {}
    acc["seeds"].update(priv.get("seeds") or {})
    left = collections.Counter(priv.get("shed") or {})
    for i in priv.get("inventories") or []:
        if isinstance(i, dict):
            left.update(i)
    prices = (last.get("market") or {}).get("prices") or {}
    acc["left_units"] += sum(n for k, n in left.items() if k != "FERTILIZER")
    acc["left_value"] += sum(n * prices.get(k, 0) for k, n in left.items() if k != "FERTILIZER")
    acc["n"] += 1


def report(title, acc, days):
    n = max(1, acc["n"])
    print(f"\n=== {title} (mean per game, n={acc['n']}) ===")
    print(f"{'day':>4s} " + " ".join(f"{o[:9]:>9s}" for o in OPS) + f" {'cash flow':>10s} {'held@dawn':>10s} {'sell units':>11s}")
    for d in days:
        c = acc["ops"][d]
        print(f"{d:4d} " + " ".join(f"{c[o] / n:9.1f}" for o in OPS)
              + f" {acc['flow'][d] / n:10.0f} {acc['held'][d] / n:10.1f} {acc['sell_units'][d] / n:11.1f}")
    tot = sum(acc["flow"][d] for d in days) / n
    late = sum(acc["flow"][d] for d in days if d >= 28) / n
    print(f"  cash flow over the window: {tot:,.0f}; on days 28-29: {late:,.0f} ({100 * late / max(1, tot):.0f}%)")
    print(f"  seeds left unplanted: {sum(acc['seeds'].values()) / n:.1f}  ("
          + ", ".join(f"{k} {v / n:.1f}" for k, v in acc["seeds"].most_common(4)) + ")")
    print(f"  units left unsold: {acc['left_units'] / n:.1f} valued ${acc['left_value'] / n:,.0f}")
    print("  day-29 SELL units by hour block 0-5/6-11/12-17/18-23: "
          + " / ".join(f"{acc['byhour'][b] / n:.0f}" for b in range(4)))
    print("  day-29 SELL units by product: "
          + ", ".join(f"{k} {v / n:.0f}" for k, v in acc["byprod"].most_common(8)))


def leaders(pattern, days):
    acc = _new()
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        steps = r["steps"]
        last = steps[-1]
        w = 0 if (last[0].get("reward") or 0) >= (last[1].get("reward") or 0) else 1
        scan(steps, w, acc, days)
    return acc


def ours(arm, flags, games, seed, per_bin, days):
    from kaggle_environments import make
    from rl import rollout, session_eval as SE
    from rlv3 import arm as ARM
    from kagent import weights as W
    entries = SE.held_out_entries(os.path.join(REPO, "rl", "datasets", "session_index.json"), seed, per_bin)
    acc = _new()
    for k in range(games):
        p, ss, _b, _r = entries[k]
        sess = rollout.load_session(os.path.join(REPO, p))
        rollout._set_exec(list(W.EXEC_RULES) + [f for f in flags.split(",") if f])
        learner, _ = rollout._make_dev_policy_agent(ARM.parse(arm))
        opp = rollout._make_opponent(("session", os.path.join(REPO, p), ss))
        side = 1 - ss
        env = make("kaggriculture", configuration={"seed": sess["seed"]}, debug=False)
        env.run([learner, opp] if side == 0 else [opp, learner])
        scan(env.steps, side, acc, days)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", default="replays/now/*/*.json", help="leader replay glob (winner seat is scored)")
    ap.add_argument("--games", type=int, default=4, help="held-out tape worlds to play our arm on (0 = skip)")
    ap.add_argument("--arm", default="base", help="rlv3 arm string, e.g. base or readout:apply=herd")
    ap.add_argument("--flags", default="", help="extra executor flags on top of the shipped rules, comma-separated")
    ap.add_argument("--days", default="20-29")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-bin", type=int, default=24)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.days.split("-"))
    days = list(range(lo, hi + 1))
    report(f"LEADERS, winner seat of {args.replays}", leaders(args.replays, days), days)
    if args.games > 0:
        label = f"OURS: rlv3 {args.arm} + shipped rules" + (f" + {args.flags}" if args.flags else "")
        report(label, ours(args.arm, args.flags, args.games, args.seed, args.per_bin, days), days)


if __name__ == "__main__":
    main()
