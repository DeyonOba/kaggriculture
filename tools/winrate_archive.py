"""Re-score every experiment we have ever run by WIN RATE, not margin.

The gate measured paired margin against our own baseline; the rating is
win-based. champ-h8 loses 96.5% of its games at a mean margin of -$31,049, so
every $1-5k lever moves the size of a loss. This asks a different question of
the same archive: did anything ever move WINS?

Three outputs:

  1. win rate per agent with a Wilson 95% interval, so underpowered comparisons
     are visible as such rather than read as signal;
  2. the same paired against champ-h8 on identical (seed, swap) worlds, which is
     the only fair comparison;
  3. the FLIPPABLE FRONTIER -- the distribution of losing margins. A game lost by
     $2k is one lever away; a game lost by $43k is not. This is what says where
     a win is even purchasable, and it is the sampling distribution the search
     should target.

    PYTHONPATH=. uv run python tools/winrate_archive.py
    PYTHONPATH=. uv run python tools/winrate_archive.py --frontier
"""

import argparse
import collections
import glob
import json
import math
import os
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "arena", "results")


def wilson(w, n, z=1.96):
    """Wilson score interval — honest at the tiny win rates we actually have."""
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path, agent):
    """-> {(seed, swap): (won, margin)} for `agent` in that file."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if agent not in r["players"]:
            continue
        if r.get("statuses") and any(s != "DONE" for s in r["statuses"]):
            continue
        i = r["players"].index(agent)
        mine, theirs = r["rewards"][i], r["rewards"][1 - i]
        if mine is None or theirs is None:
            continue
        out[(r["seed"], r["swap"])] = (mine > theirs, mine - theirs)
    return out


def collect():
    data = collections.defaultdict(dict)   # agent -> {(opp, seed, swap): (won, marg)}
    for f in sorted(glob.glob(os.path.join(R, "*_vs_pub-*.jsonl"))):
        base = os.path.basename(f)[:-6]
        agent, opp = base.split("_vs_")
        if agent.startswith("pub-"):
            continue
        for k, v in load(f, agent).items():
            data[agent][(opp,) + k] = v
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--min-n", type=int, default=48)
    ap.add_argument("--frontier", action="store_true",
                    help="show the losing-margin distribution instead")
    args = ap.parse_args()

    data = collect()
    base = data.get(args.base, {})
    if not base:
        raise SystemExit(f"no results for {args.base}")

    if args.frontier:
        print("FLIPPABLE FRONTIER — how far from a win is each loss?\n")
        by_opp = collections.defaultdict(list)
        for (opp, _, _), (won, m) in base.items():
            by_opp[opp].append((won, m))
        print(f"{'opponent':<22} {'n':>4} {'wins':>5} "
              f"{'losses within $5k':>18} {'$10k':>7} {'$20k':>7} {'median loss':>13}")
        for opp, rows in sorted(by_opp.items()):
            losses = [-m for w, m in rows if not w]
            n = len(rows)
            w = sum(1 for x, _ in rows if x)
            f5 = sum(1 for x in losses if x <= 5000)
            f10 = sum(1 for x in losses if x <= 10000)
            f20 = sum(1 for x in losses if x <= 20000)
            med = st.median(losses) if losses else 0
            print(f"{opp:<22} {n:>4} {w:>5} {f5:>18} {f10:>7} {f20:>7} "
                  f"{med:>13,.0f}")
        allrows = [v for v in base.values()]
        losses = sorted(-m for w, m in allrows if not w)
        print(f"\nlosing margins, all opponents (n={len(losses)}):")
        for q in (0.05, 0.10, 0.25, 0.50):
            i = int(len(losses) * q)
            print(f"   p{int(q*100):<3} {losses[i]:>10,.0f}")
        print(f"\n{sum(1 for x in losses if x <= 5000)} of {len(losses)} losses are "
              f"within $5k — those are the only games a $1-5k lever can flip.")
        return

    print("WIN RATE vs the public agents (Wilson 95%)\n")
    print(f"{'agent':<16} {'n':>5} {'wins':>5} {'win%':>7} {'95% CI':>16} "
          f"{'mean margin':>13}  {'paired vs ' + args.base:>22}")
    rows = []
    for agent, d in data.items():
        n = len(d)
        if n < args.min_n:
            continue
        w = sum(1 for won, _ in d.values() if won)
        lo, hi = wilson(w, n)
        marg = st.mean(m for _, m in d.values())
        shared = sorted(set(d) & set(base))
        pair = ""
        if agent != args.base and len(shared) >= 20:
            cw = sum(1 for k in shared if d[k][0])
            bw = sum(1 for k in shared if base[k][0])
            pair = f"{bw}->{cw} on {len(shared)}"
        rows.append((w / n, agent, n, w, lo, hi, marg, pair))
    for p, agent, n, w, lo, hi, marg, pair in sorted(rows, reverse=True):
        print(f"{agent:<16} {n:>5} {w:>5} {p*100:>6.1f}% "
              f"[{lo*100:>5.1f},{hi*100:>5.1f}] {marg:>13,.0f}  {pair:>22}")

    print(f"\nNote: at these win rates a difference of 1-2 games is inside the "
          f"interval.\nDetecting 3.5% -> 7.0% at 95% needs roughly 700 games per arm.")


if __name__ == "__main__":
    main()
