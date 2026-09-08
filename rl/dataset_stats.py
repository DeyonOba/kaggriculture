"""Answer the dataset questions over a streamed field harvest:
win rate, winning/losing profit margins, and which day-plans win
(per-head choice vs outcome).

Usage:
    PYTHONPATH=. uv run python rl/dataset_stats.py --data rl/datasets/field
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagent import features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rl/datasets/field")
    args = ap.parse_args()

    stats = [json.loads(l) for l in open(os.path.join(args.data, "stats.jsonl")) if l.strip()]
    games = {s["episode"] for s in stats}
    wins = [s for s in stats if s["win"]]
    losses = [s for s in stats if not s["win"]]

    print(f"=== field dataset: {len(games)} games / {len(stats)} agent-sides ===")
    print(f"win rate (by construction ~50%): {len(wins)/max(1,len(stats)):.2%}")
    for label, rows in (("WINNERS", wins), ("LOSERS", losses)):
        money = [r["money"] for r in rows]
        margin = [abs(r["margin"]) for r in rows]
        if money:
            print(f"{label:>8}: money mean=${statistics.mean(money):>9,.0f} "
                  f"median=${statistics.median(money):>9,.0f} "
                  f"| margin mean=${statistics.mean(margin):>9,.0f} "
                  f"median=${statistics.median(margin):>9,.0f}")
    hi = sorted(stats, key=lambda s: -s["money"])[:5]
    print("top finals:", ", ".join(f"${s['money']:,.0f}" for s in hi))

    # Which plans win: per-head, win rate by chosen value (day 0-13 decisions,
    # where plans differ most), min 80 samples per cell.
    path = os.path.join(args.data, "samples.jsonl")
    per_head = [dict() for _ in features.HEADS]
    n = 0
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if "win" not in s:
                continue
            day = s["x"][0] * 30.0  # feature 0 is day/30
            if day > 13:
                continue
            n += 1
            for h, c in enumerate(s["choices"]):
                val = features.HEADS[h][1][c]
                cell = per_head[h].setdefault(val, [0, 0])
                cell[0] += s["win"]
                cell[1] += 1
    print(f"\n=== plan vs outcome (early-game decisions, n={n}) ===")
    for (name, _), cells in zip(features.HEADS, per_head):
        parts = []
        for val, (w, t) in sorted(cells.items()):
            if t >= 80:
                parts.append(f"{val}:{w/t:.0%}({t})")
        if len(parts) > 1:
            print(f"  {name:>20}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
