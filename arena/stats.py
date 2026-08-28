"""Aggregate arena results: final-money stats, win rate, paired sign test."""

import json
import math
import statistics
import sys


def _load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sign_test_p(wins, losses):
    """Two-sided binomial sign test p-value (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(path):
    records = _load(path)
    if not records:
        print("no records")
        return

    names = records[0]["players"]
    a, b = sorted(set(names))
    finals = {a: [], b: []}
    bad = []
    a_wins = b_wins = ties = 0

    for rec in records:
        for name in rec["players"]:
            r = rec["final"][name]
            finals[name].append(r if r is not None else float("nan"))
        for name, status in zip(rec["players"], rec["statuses"]):
            if status != "DONE" or rec["final"][name] is None:
                bad.append((rec["seed"], rec["swap"], name, status))
        fa, fb = rec["final"][a], rec["final"][b]
        if fa is None or fb is None:
            continue
        if fa > fb:
            a_wins += 1
        elif fb > fa:
            b_wins += 1
        else:
            ties += 1

    print(f"\n=== {path} ({len(records)} games) ===")
    for name in (a, b):
        vals = [v for v in finals[name] if not math.isnan(v)]
        if not vals:
            print(f"{name:>12}: no valid finals")
            continue
        print(
            f"{name:>12}: mean={statistics.mean(vals):9.1f}  "
            f"median={statistics.median(vals):9.1f}  "
            f"std={statistics.pstdev(vals):8.1f}  "
            f"min={min(vals):9.1f}  max={max(vals):9.1f}"
        )
    p = _sign_test_p(a_wins, b_wins)
    print(f"head-to-head: {a} {a_wins}W / {b} {b_wins}W / {ties}T  (sign test p={p:.4f})")
    if bad:
        print(f"!! {len(bad)} bad statuses/rewards: {bad[:10]}")
    else:
        print("statuses: all DONE, no null rewards")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(arg)
