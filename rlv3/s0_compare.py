"""Read several S0 runs side by side: per stage, per day, per asset.

    PYTHONPATH=. uv run python rlv3/s0_compare.py \
        A=rl/runs/headfix/s0_check.json B=rl/runs/headfix/s0_check_D.json C=rl/runs/headfix/s0_check_C.json
"""
from __future__ import annotations

import json
import statistics
import sys

from kagent.constants import ANIMALS, CROPS

STAGES = (("opening", 0, 8), ("contest", 9, 26), ("consolidation", 27, 29))


def load(spec):
    name, _, path = spec.partition("=")
    with open(path) as fh:
        return name, json.load(fh)


def stage_tvd(run, lo, hi):
    ds = [d for d in run["day_tvd"] if lo <= int(d) <= hi]
    return (statistics.mean(run["day_tvd"][d] for d in ds),
            statistics.mean(run["day_floor"][d] for d in ds),
            statistics.mean(run["day_in"][d] for d in ds))


def main():
    runs = [load(a) for a in sys.argv[1:]]
    print(f"{'stage':<14}{'floor':>7}  " + "".join(f"{n:>9}" for n, _ in runs) + "   (TVD to the leader; at/under floor = agreement)")
    for st, lo, hi in STAGES:
        floor = stage_tvd(runs[0][1], lo, hi)[1]
        cells = "".join(f"{stage_tvd(r, lo, hi)[0]:>9.3f}" for _n, r in runs)
        print(f"{st:<14}{floor:>7.3f}  {cells}")

    print(f"\nTILES, mean 3-day commitment per dawn   (leader stock-delta | leader PLANT flow)")
    print(f"{'asset':<12}{'leader':>16}  " + "".join(f"{n:>9}" for n, _ in runs))
    for c in list(CROPS) + list(ANIMALS):
        rows0 = runs[0][1]["rows"]
        if c in CROPS:
            ld = statistics.mean(r["lead_dcrops"].get(c, 0) for r in rows0)
            lf = statistics.mean(r.get("lead_flow", {}).get(c, 0) for r in rows0) if "lead_flow" in rows0[0] else None
            lead = f"{ld:>6.2f} | {lf:>6.2f}" if lf is not None else f"{ld:>6.2f} |   -   "
        else:
            ld = statistics.mean(r["lead_dherd"].get(c, 0) for r in rows0)
            lead = f"{ld:>6.2f} |   -   "
        cells = ""
        for _n, r in runs:
            v = statistics.mean((x["our_crops"] if c in CROPS else x["our_herd"]).get(c, 0) for x in r["rows"])
            cells += f"{v:>9.2f}"
        print(f"{c:<12}{lead:>16}  {cells}")

    print(f"\nDOLLARS by day, seed / livestock / hands   (leader median first)")
    print(f"{'day':>4}  {'leader':^18}  " + "".join(f"{n:^20}" for n, _ in runs))
    for d in range(0, 30):
        by = [[x for x in r["rows"] if x["day"] == d] for _n, r in runs]
        if not by[0]:
            continue
        def sh(rows, key):
            out = []
            for x in rows:
                sp = x[key]; tot = sum(sp.values()) or 1.0
                out.append({k: v / tot for k, v in sp.items()})
            return {k: statistics.median(o[k] for o in out) for k in ("seed", "livestock", "hands")}
        L = sh(by[0], "lead_spend")
        line = f"{d:>4}  {L['seed']:>5.2f} {L['livestock']:>5.2f} {L['hands']:>5.2f}  "
        for rows in by:
            O = sh(rows, "our_spend")
            line += f"  {O['seed']:>5.2f} {O['livestock']:>5.2f} {O['hands']:>5.2f}   "
        print(line)


if __name__ == "__main__":
    main()
