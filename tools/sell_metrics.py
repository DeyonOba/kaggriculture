"""Sell-timing metrics from traces (tools/game_trace.py) and Kaggle replays:
per steep-curve product the units per order, the within-step price impact,
the 24-step recovery, the realised price over the product's game-mean
price, the evening share of units sold, and money.

    PYTHONPATH=. uv run python tools/sell_metrics.py \\
        --trace ours=/tmp/base.json --replay-glob mmpq="replays/mmpq/*.json" \\
        --replay-glob boey="replays/boey/*.json:team=Boey"

The engine fills a SELL order unit by unit at the running price, both
players interleaved, so a large order walks down the curve.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import statistics as st
import sys

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
SHOW = ("STRAWBERRY", "MILK", "WOOL", "MELON")


def load_trace(path):
    T = json.load(open(path))
    steps = [(s["day"], s["hour"], dict(zip(PRODUCTS, s["market"]["prices"])),
              dict(zip(PRODUCTS, s["market"]["inv"])), (s["act"]["me"] or {}).get("market"))
             for s in T["steps"]]
    return steps, T["final"]["me"], T["final"]["opp"]


def load_replay(path, team="M & M & P & Q"):
    d = json.load(open(path))
    tn = d["info"].get("TeamNames")
    tn = ast.literal_eval(tn) if isinstance(tn, str) else tn
    seat = list(tn).index(team)
    raw = d["steps"]
    steps = []
    for k in range(len(raw) - 1):
        obs = raw[k][0]["observation"]
        act = raw[k + 1][seat].get("action") or {}
        steps.append((int(obs["day"]), int(obs["hour"]),
                      {p: float(obs["market"]["prices"].get(p, 0)) for p in PRODUCTS},
                      {p: int(obs["market"]["inventory"].get(p, 0)) for p in PRODUCTS}, act.get("market")))
    fin = [float(raw[-1][s].get("reward") or 0.0) for s in range(2)]
    return steps, fin[seat], fin[1 - seat]


def analyse(steps):
    gm = collections.defaultdict(list)
    for d, h, px, inv, orders in steps:
        for p, v in px.items():
            gm[p].append(v)
    gm = {p: st.mean(v) for p, v in gm.items()}
    per = collections.defaultdict(list)
    for k, (d, h, px, inv, orders) in enumerate(steps[:-25]):
        for o in orders or []:
            if o and isinstance(o, (list, tuple)) and o[0] == "SELL" and len(o) >= 3 and o[1] in px and px[o[1]] > 1:
                p = o[1]
                per[p].append((int(o[2]), steps[k + 1][2][p] / px[p], steps[k + 24][2][p] / px[p],
                               px[p] / gm[p] if gm[p] else 0, h, inv[p] - 10000))
    out = {}
    for p, v in per.items():
        units = sum(x[0] for x in v)
        out[p] = {"orders": len(v), "units": units, "units_per_order": units / len(v),
                  "impact": st.mean(x[1] for x in v), "recovery24": st.mean(x[2] for x in v),
                  "realized_over_mean": sum(x[0] * x[3] for x in v) / units,
                  "evening_share": sum(x[0] for x in v if x[4] >= 20) / units,
                  "inv_minus_I0_at_sale": st.mean(x[5] for x in v)}
    tot = sum(x[0] for v in per.values() for x in v)
    out["_all"] = {"units": tot, "orders": sum(len(v) for v in per.values()),
                   "evening_share": (sum(x[0] for v in per.values() for x in v if x[4] >= 20) / tot) if tot else None,
                   "realized_over_mean": (sum(x[0] * x[3] for v in per.values() for x in v) / tot) if tot else None}
    return out


def _avg(dicts):
    out = {}
    keys = set().union(*[d.keys() for d in dicts])
    for p in keys:
        ms = [d[p] for d in dicts if p in d]
        out[p] = {k: st.mean([m[k] for m in ms if m.get(k) is not None]) for k in ms[0]
                  if any(m.get(k) is not None for m in ms)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", action="append", default=[], help="label=path")
    ap.add_argument("--replay", action="append", default=[], help="label=path[:team=NAME]")
    ap.add_argument("--replay-glob", action="append", default=[], help="label=glob[:team=NAME], averaged")
    args = ap.parse_args()
    results = {}

    def spec(rest):
        team = "M & M & P & Q"
        if ":team=" in rest:
            rest, team = rest.split(":team=", 1)
        return rest, team

    for a in args.trace:
        label, path = a.split("=", 1)
        steps, me, opp = load_trace(path)
        results[label] = (analyse(steps), me, opp, 1)
    for a in args.replay:
        label, rest = a.split("=", 1)
        path, team = spec(rest)
        steps, me, opp = load_replay(path, team)
        results[label] = (analyse(steps), me, opp, 1)
    for a in args.replay_glob:
        label, rest = a.split("=", 1)
        pattern, team = spec(rest)
        ds, mes, opps = [], [], []
        for f in sorted(glob.glob(pattern)):
            try:
                steps, me, opp = load_replay(f, team)
            except (ValueError, KeyError) as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            ds.append(analyse(steps))
            mes.append(me)
            opps.append(opp)
        if not ds:
            ap.error(f"no replay with team {team!r} matches {pattern}")
        results[label] = (_avg(ds), st.mean(mes), st.mean(opps), len(ds))
    if not results:
        ap.error("give at least one input")
    labels = list(results)
    w = max(len(l) for l in labels) + 2
    print(f"{'metric':<36}" + "".join(f"{l:>{w}}" for l in labels))
    print(f"{'games':<36}" + "".join(f"{results[l][3]:>{w}}" for l in labels))
    print(f"{'final money':<36}" + "".join(f"{results[l][1]:>{w},.0f}" for l in labels))
    print(f"{'opponent money':<36}" + "".join(f"{results[l][2]:>{w},.0f}" for l in labels))
    for k, fmt in (("units", "{:.0f}"), ("orders", "{:.0f}"), ("realized_over_mean", "{:.3f}"), ("evening_share", "{:.2f}")):
        cells = [("-" if results[l][0].get("_all", {}).get(k) is None else fmt.format(results[l][0]["_all"][k])) for l in labels]
        print(f"{'all products ' + k:<36}" + "".join(f"{c:>{w}}" for c in cells))
    for p in SHOW:
        for k, fmt in (("units", "{:.0f}"), ("units_per_order", "{:.1f}"), ("impact", "{:.3f}"), ("recovery24", "{:.3f}"),
                       ("realized_over_mean", "{:.3f}"), ("evening_share", "{:.2f}"), ("inv_minus_I0_at_sale", "{:.0f}")):
            cells = []
            for l in labels:
                m = results[l][0].get(p)
                cells.append("-" if not m or m.get(k) is None else fmt.format(m[k]))
            print(f"{p.lower() + ' ' + k:<36}" + "".join(f"{c:>{w}}" for c in cells))


if __name__ == "__main__":
    main()
