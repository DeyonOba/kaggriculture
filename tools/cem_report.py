"""Read a CEM run: which coordinates the search actually moved, and how far.

    PYTHONPATH=. uv run python tools/cem_report.py rl/runs/cem_theta_s0

WHAT THIS IS AND IS NOT. Every line here is OBSERVATIONAL -- it correlates a
coordinate's value with the paired delta ACROSS the trials the search happened
to draw, and the trials are not a designed experiment: CEM concentrates its
sampling where it already thinks the good region is, so a coordinate can look
decisive because the elites carried it, not because it caused anything. The
project's most expensive recurring error has exactly this shape: behaviour
cloning at 69.4% accuracy, the market demand table, the value head at 4.8x
tighter fit -- all correct about the data and wrong about the effect of a
change (memory: observational vs interventional).

So this report GENERATES hypotheses and settles none. The verdict is the
holdout gate the run prints at the end, and anything read here that matters
gets its own paired arm. What it is good for: seeing whether the search moved
at all, which coordinates it moved TOGETHER (the whole reason for a joint
search), and whether the point it landed on is a basin or one lucky draw.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the --out directory of a cem_theta run")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    from kagent import theta as T

    trials = []
    with open(os.path.join(args.run, "trials.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    if not trials:
        raise SystemExit("no trials yet")

    champ = T.champion()
    deltas = [t["score"]["delta"] for t in trials]
    print(f"{len(trials)} trials   delta range {min(deltas):+.3f} .. {max(deltas):+.3f}"
          f"   median {sorted(deltas)[len(deltas)//2]:+.3f}")

    # --- is the search moving? -------------------------------------------
    gens = sorted({t["gen"] for t in trials})
    print(f"\n{'gen':>4}{'n':>4}{'best':>9}{'mean':>9}{'worst':>9}   "
          f"{'money':>9}{'opp':>9}")
    for g in gens:
        d = [t["score"]["delta"] for t in trials if t["gen"] == g]
        m = [t["score"]["money"] for t in trials if t["gen"] == g]
        o = [t["score"]["opp"] for t in trials if t["gen"] == g]
        print(f"{g:>4}{len(d):>4}{max(d):>+9.3f}{sum(d)/len(d):>+9.3f}{min(d):>+9.3f}"
              f"   {sum(m)/len(m):>9,.0f}{sum(o)/len(o):>9,.0f}")

    # --- per-coordinate association ---------------------------------------
    print(f"\nCOORDINATE ASSOCIATION  (observational -- see the module docstring)")
    print(f"{'coordinate':<18}{'champ':>9}{'best':>9}{'r':>8}   "
          f"{'off/low':>9}{'on/high':>9}{'gap':>8}")
    best_t = max(trials, key=lambda t: (t["score"]["delta"], t["score"]["margin"]))
    rows = []
    for name in T.NAMES:
        xs, ys = [], []
        for t in trials:
            v = t["theta"].get(name)
            xs.append(float(bool(v)) if isinstance(v, bool) else float(v))
            ys.append(t["score"]["delta"])
        r = _corr(xs, ys)
        mid = sorted(xs)[len(xs) // 2]
        lo = [y for x, y in zip(xs, ys) if x <= mid]
        hi = [y for x, y in zip(xs, ys) if x > mid]
        if not lo or not hi:
            gap = float("nan")
            mlo = sum(lo) / len(lo) if lo else float("nan")
            mhi = sum(hi) / len(hi) if hi else float("nan")
        else:
            mlo, mhi = sum(lo) / len(lo), sum(hi) / len(hi)
            gap = mhi - mlo
        rows.append((abs(gap) if gap == gap else -1, name, r, mlo, mhi, gap))
    rows.sort(reverse=True)
    for _k, name, r, mlo, mhi, gap in rows:
        c, b = champ.get(name), best_t["theta"].get(name)
        cs = f"{int(c) if isinstance(c, bool) else c}"
        bs = f"{int(b) if isinstance(b, bool) else b}"
        print(f"{name:<18}{cs:>9}{bs:>9}{r:>8.2f}   "
              f"{mlo:>+9.3f}{mhi:>+9.3f}{gap:>+8.3f}")

    # --- the trials themselves --------------------------------------------
    print(f"\nTOP {args.top} TRIALS")
    top = sorted(trials, key=lambda t: (t["score"]["delta"], t["score"]["margin"]),
                 reverse=True)[:args.top]
    for t in top:
        s = t["score"]
        print(f"  g{t['gen']} {s['delta']:+.3f} +/-{s['delta_se']:.3f}  win {s['win']:.3f}  "
              f"money {s['money']:>8,.0f}  opp {s['opp']:>8,.0f}")
        print(f"      {t['diff']}")

    st = os.path.join(args.run, "state.json")
    if os.path.exists(st):
        with open(st) as fh:
            state = json.load(fh)
        print(f"\nCEM centre after gen {state['gen']} "
              f"({state['elapsed_min']:.0f} min):")
        print(f"  {T.describe(state['centre'], champ)}")
        print(f"\nBERNOULLI PROBABILITIES the search has settled on:")
        cells = [f"{T.NAMES[i]}={state['mu'][i]:.2f}" for i in T.BOOL_AT]
        for i in range(0, len(cells), 4):
            print("  " + "  ".join(f"{c:<24}" for c in cells[i:i + 4]))


if __name__ == "__main__":
    main()
