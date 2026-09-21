"""Do knob results measured on ONE executor transfer to another?

The standing claim is that every knob and every BC run was fitted against
champ-h8's executor, so none of it transfers and every closed experiment has
to be reopened. That is a testable claim, not a given, and reopening ~20
closed experiments is expensive enough to be worth testing first.

The test: take K knobs with known effects, apply each to THREE executors that
share one planner and differ only in `scheduler.py` --

    E0  champ-h8    index-order greedy assignment (the shipped executor)
    E1  match-all   global-greedy matching over the same candidate set
    E2  a-commit    cross-turn commitment to a target

-- gate every arm against ITS OWN executor on the same held-out ladder worlds,
and compare the effect vectors.

    agreement  -> knob effects are a property of the PLAN, not the executor,
                  and the closed experiments stay closed
    divergence -> the claim holds and the corpus of negatives is void

    PYTHONPATH=. uv run python tools/knob_transfer.py --n 60 --seed 0
"""
import argparse
import itertools
import json
import math
import os
import statistics as st
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
CACHE = os.path.join(REPO, "rl", "bakeoff_runs")

EXECUTORS = [("E0", "champ-h8"), ("E1", "match-all"), ("E2", "a-commit"),
             ("E3", "h11-uf")]   # E3: the underfoot fix, obeys 92% vs 85%
KNOBS = ["geese", "wheat_engine", "straw_target", "melon_opening",
         "hires_delta", "carrot_standing"]


def load(agent, tag):
    p = os.path.join(CACHE, f"ladder_{agent}_{tag}.json")
    if not os.path.exists(p):
        return None
    return {tuple(json.loads(k)): tuple(v) for k, v in json.load(open(p)).items()}


def effect(arm, base):
    ks = sorted(set(arm) & set(base))
    if len(ks) < 30:
        return None
    d = [arm[k][0] - base[k][0] for k in ks]
    b_only = sum(1 for k in ks if base[k][1] and not arm[k][1])
    c_only = sum(1 for k in ks if arm[k][1] and not base[k][1])
    return {
        "n": len(ks),
        "money": st.mean(d),
        "se": st.stdev(d) / math.sqrt(len(d)),
        "win_delta": c_only - b_only,
        "base_win": sum(1 for k in ks if base[k][1]) / len(ks),
        "arm_win": sum(1 for k in ks if arm[k][1]) / len(ks),
    }


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return float("nan")
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def run_gates(n, seed, procs):
    tag = f"{n}_{seed}"
    for ex, base in EXECUTORS:
        cands = [f"t{ex}-{k}" for k in KNOBS]
        missing = [c for c in cands
                   if not os.path.isfile(os.path.join(REPO, "arena", "agents",
                                                      c, "main.py"))]
        if missing:
            sys.exit(f"missing arms {missing}; build them with "
                     f"tools/make_knob_variant.py --base {base}")
        need = [c for c in cands if load(c, tag) is None] or []
        if not need and load(base, tag) is not None:
            print(f"  {ex} ({base}): cached")
            continue
        print(f"  {ex} ({base}): gating {len(cands)} arms", flush=True)
        subprocess.run(
            ["uv", "run", "python", "tools/ladder_gate.py", "--cand", *cands,
             "--base", base, "--n", str(n), "--seed", str(seed),
             "--procs", str(procs)],
            cwd=REPO, env={**os.environ, "PYTHONPATH": REPO}, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    tag = f"{args.n}_{args.seed}"

    if not args.report_only:
        run_gates(args.n, args.seed, args.procs)

    bases = {ex: load(base, tag) for ex, base in EXECUTORS}
    table = {}
    print(f"\nknob effect vs its OWN executor, paired on the same ladder worlds "
          f"(n={args.n}/bin, seed {args.seed})\n")
    hdr = f"{'knob':18s}"
    for ex, base in EXECUTORS:
        hdr += f"{ex + ' ' + base:>26s}"
    print(hdr)
    print("-" * len(hdr))
    for k in KNOBS:
        row = f"{k:18s}"
        for ex, base in EXECUTORS:
            arm = load(f"t{ex}-{k}", tag)
            e = effect(arm, bases[ex]) if arm and bases[ex] else None
            table.setdefault(ex, {})[k] = e
            row += (f"{e['money']:>+15,.0f} ±{e['se']:>5,.0f}"
                    f"{e['win_delta']:>+4d}" if e else f"{'—':>26s}")
        print(row)

    print()
    for (xa, _), (xb, _) in itertools.combinations(EXECUTORS, 2):
        ks = [k for k in KNOBS if table.get(xa, {}).get(k) and table.get(xb, {}).get(k)]
        if len(ks) < 3:
            continue
        a = [table[xa][k]["money"] for k in ks]
        b = [table[xb][k]["money"] for k in ks]
        agree = sum(1 for i in range(len(ks))
                    if (a[i] > 0) == (b[i] > 0))
        # would a decision made on xa have been the same on xb?
        big = [i for i in range(len(ks))
               if abs(a[i]) > 2 * table[xa][ks[i]]["se"]]
        big_agree = sum(1 for i in big if (a[i] > 0) == (b[i] > 0))
        print(f"{xa} vs {xb}:  Spearman {spearman(a, b):+.2f}   "
              f"sign agreement {agree}/{len(ks)}   "
              f"sign agreement on arms {xa} resolves (>2se) "
              f"{big_agree}/{len(big)}")

    print("\nA knob whose sign is the same on every executor is a property of "
          "the PLAN.\nA knob that flips sign is evidence the executor was "
          "carrying the result.")


if __name__ == "__main__":
    main()
