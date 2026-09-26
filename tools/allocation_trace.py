"""Resource allocation, DAY BY DAY, for our agent against the three leaders.

The scorecard samples the crop mix at days 6, 12 and 20. That is enough to say
the allocation is wrong; it is not enough to watch a model LEARN to allocate.
This prints the whole curve -- every day, every resource -- so a candidate can
be compared against the leaders' trajectory rather than three snapshots, and so
a learned planner's day-to-day corrections are visible.

Per day, per farm:
    tiles under each crop, animals of each kind, hands, quadrants owned,
    ground free at dawn, ground that frees that day, and units brought to
    market that day.

    PYTHONPATH=. uv run python tools/allocation_trace.py \
        --trace ours=/tmp/kagg/exec/b4_ship2.json \
        --replay-glob mmpq='replays/mmpq/*.json':team='M & M & P & Q' \
        [--csv out.csv]
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from tile_metrics import load_replay, load_trace  # noqa: E402

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
FIELDS = CROPS + ANIMALS + ["hands", "quads", "bare", "freeing", "sold"]


def per_day(steps):
    """-> {day: {field: value}} for one game."""
    out = {}
    sold_by_day = collections.Counter()
    for s in steps:
        for m in s.get("market") or []:
            if m and isinstance(m, (list, tuple)) and m[0] == "SELL" and len(m) > 2:
                sold_by_day[s["day"]] += int(m[2])
    prev_counts = {}
    for s in steps:
        if s["hour"] != 0:
            continue
        d = s["day"]
        c = collections.Counter()
        bare = 0
        for row in s["tiles"]:
            for t in row:
                if t == 1 or t == 2:
                    bare += 1
                elif isinstance(t, list) and t[0] == 4:
                    c[CROPS[t[1]]] += 1
                elif isinstance(t, list) and t[0] == 5:
                    c[ANIMALS[t[1]]] += 1
        rec = {k: c.get(k, 0) for k in CROPS + ANIMALS}
        rec["hands"] = len(s["units"]) - 1
        rec["quads"] = max(1, round(sum(1 for r in s["tiles"] for t in r if t != 0) / 25))
        rec["bare"] = bare
        rec["sold"] = sold_by_day.get(d, 0)
        # ground that frees today = crops that vanish by tomorrow's dawn
        rec["freeing"] = 0
        out[d] = rec
        if prev_counts:
            gone = sum(max(0, prev_counts.get(k, 0) - rec.get(k, 0)) for k in CROPS)
            out[d - 1]["freeing"] = gone
        prev_counts = rec
    return out


def average(games):
    days = sorted({d for g in games for d in g})
    return {d: {f: st.mean([g[d][f] for g in games if d in g] or [0]) for f in FIELDS}
            for d in days}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", action="append", default=[], help="label=path")
    ap.add_argument("--replay-glob", action="append", default=[],
                    help="label=glob:team=NAME")
    ap.add_argument("--days", default="0-29")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    series = {}
    for spec in args.trace:
        label, path = spec.split("=", 1)
        series[label] = average([per_day(load_trace(path)[0])])
    for spec in args.replay_glob:
        label, rest = spec.split("=", 1)
        pat, team = rest.split(":team=", 1)
        games = []
        for f in sorted(glob.glob(pat.strip("'\""))):
            try:
                games.append(per_day(load_replay(f, team=team)[0]))
            except Exception:
                continue
        if games:
            series[label] = average(games)
    if not series:
        ap.error("give at least one --trace or --replay-glob")

    lo, hi = (int(x) for x in args.days.split("-"))
    labels = list(series)
    show = [f for f in FIELDS if any(
        any(s.get(d, {}).get(f, 0) for d in range(lo, hi + 1)) for s in series.values())]
    for f in show:
        print(f"\n== {f} ==")
        print("day   " + "".join(f"{l:>12}" for l in labels))
        for d in range(lo, hi + 1):
            row = "".join(f"{series[l].get(d, {}).get(f, 0):>12.1f}" for l in labels)
            if row.strip():
                print(f"{d:>3}   {row}")
    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("label,day," + ",".join(FIELDS) + "\n")
            for l in labels:
                for d in sorted(series[l]):
                    fh.write(f"{l},{d}," + ",".join(f"{series[l][d].get(f,0):.2f}" for f in FIELDS) + "\n")
        print("\n->", args.csv)


if __name__ == "__main__":
    main()
