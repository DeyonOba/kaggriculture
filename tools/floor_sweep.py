"""Is the sell-floor surface flat? Paired sweep of the market-timing heads.

This is the decisive test for the RL plan. The policy's action space is thirteen
heads; ten are allocation quantities, and allocation is measured flat (the LP's
optimal allocation, executed, tied champ-h8 at +$828 +/- 2,317). The other three
scale the sell floors. If the floor surface is ALSO flat, then every dimension a
trained policy can move is flat, and no amount of data, architecture or compute
changes that -- the RL plan dies on the merits rather than on effort. If it is
not flat, we have a live lever that reacts to price, which is exactly what a
fixed tape cannot do.

Everything is paired: every variant plays the identical (opponent, seed, swap)
worlds, and `f-base` is the same textual patch at scale 1.0, so the reported
difference isolates the knob from the patching. Games run through
`arena/run_games.py` (which has tools/fast_env.py wired in) rather than a
private process pool.

    PYTHONPATH=. uv run python tools/floor_sweep.py --axis premium --n 16
    PYTHONPATH=. uv run python tools/floor_sweep.py --axis all --n 16 --procs 6
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "arena", "results")
sys.path.insert(0, REPO)

from tools.make_floor_variant import build  # noqa: E402

OPPS = ["pub-ahmed-v38", "pub-tschinkel-psr", "pub-boatlee-v16"]

GRID = {
    "premium": [("premium", v) for v in (0.5, 0.75, 1.25, 1.6)],
    "staple":  [("staple", v) for v in (0.5, 0.75, 1.4, 2.0)],
    "melon":   [("melon", v) for v in (0.5, 0.75, 1.5, 2.5)],
}


def play(agent, opp, n, base_seed, procs):
    """Run one agent vs one opponent, return {(seed, swap): reward}."""
    path = os.path.join(RESULTS, f"{agent}_vs_{opp}.jsonl")
    if os.path.exists(path):
        os.remove(path)
    r = subprocess.run(
        ["uv", "run", "python", "arena/run_games.py", "--a", agent, "--b", opp,
         "--n", str(n), "--base-seed", str(base_seed), "--procs", str(procs)],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    if not os.path.exists(path):
        print(f"    FAILED {agent} vs {opp}: {(r.stderr or r.stdout)[-300:]}")
        return {}
    out = {}
    for line in open(path):
        row = json.loads(line)
        if agent not in row["players"]:
            continue
        i = row["players"].index(agent)
        if row.get("statuses") and any(s != "DONE" for s in row["statuses"]):
            continue
        out[(row["seed"], row["swap"])] = row["rewards"][i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("premium", "staple", "melon", "all"),
                    default="all")
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--n", type=int, default=16, help="seeds (x2 swaps x3 opponents)")
    ap.add_argument("--base-seed", type=int, default=71000)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    axes = ["premium", "staple", "melon"] if args.axis == "all" else [args.axis]
    settings = [("base", {})]
    for ax in axes:
        for knob, v in GRID[ax]:
            settings.append((f"{knob[:4]}-{v:g}", {knob: v}))

    names = []
    for tag, kw in settings:
        name = f"f-{tag}"
        build(args.base, name, **kw)
        names.append(name)
    print(f"built {len(names)} variants from {args.base}; "
          f"{len(names) * len(OPPS) * args.n * 2} games\n", flush=True)

    res = {}
    for name in names:
        got = 0
        for opp in OPPS:
            r = play(name, opp, args.n, args.base_seed, args.procs)
            for k, v in r.items():
                res[(name, opp) + k] = v
            got += len(r)
        print(f"   {name:<14} {got:>4} games", flush=True)

    base = "f-base"
    keys = [k[1:] for k in res if k[0] == base]
    bm = st.mean(res[(base,) + k] for k in keys)
    print(f"\npaired vs {base} (same worlds, same opponents)\n")
    print(f"{'variant':<14} {'n':>4} {'mean':>11} {'paired diff':>13} {'se':>8} "
          f"{'t':>6} {'better':>9}")
    print(f"{base:<14} {len(keys):>4} {bm:>11,.0f} {'--':>13}")

    rows = []
    for name in names:
        if name == base:
            continue
        pairs = [(res[(name,) + k], res[(base,) + k]) for k in keys
                 if (name,) + k in res]
        if len(pairs) < 3:
            print(f"{name:<14} {len(pairs):>4}  insufficient")
            continue
        d = [c - b for c, b in pairs]
        m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        rows.append((name, m, se, t))
        print(f"{name:<14} {len(d):>4} {st.mean(c for c, _ in pairs):>11,.0f} "
              f"{m:>+13,.0f} {se:>8,.0f} {t:>6.2f} "
              f"{sum(x > 0 for x in d):>5}/{len(d):<3}")

    if rows:
        best = max(rows, key=lambda r: r[1])
        span = max(r[1] for r in rows) - min(r[1] for r in rows)
        print(f"\nbest: {best[0]} {best[1]:+,.0f} +/- {best[2]:,.0f} (t={best[3]:.2f})")
        print(f"spread across settings: ${span:,.0f}")
        print("\nflat surface -- no setting clears 2 se" if abs(best[3]) < 2
              else "\nNOT flat -- a setting clears 2 se; gate it on a fresh block")


if __name__ == "__main__":
    main()
