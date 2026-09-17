"""E-E: fit the day-6 arm selector on LADDER worlds, cross-validated.

The shipped `tape-sel` picks between two continuations at step 144 using one
hand-written rule (a yarn store among the first two shops). This searches the
choice instead: every cluster arm is played on the same held-out ladder worlds,
the per-world winner is the SEARCH RESULT, and a selector is fitted to predict
it from day-6 state. That is expert iteration with the search made tractable by
the splice -- every tape in the 3029.3 cluster has byte-identical field actions
for days 0-5, so switching at step 144 is lossless.

Why the ladder and not the panel: every previous selector fit used the
3-opponent panel, which is a refuted instrument (tape-112 won 70.8% there and
rated 191 points BELOW the arm it beat). Ladder money correlates +0.94 with
rating. Both prior selector refits looked good locally and rated worse; this is
the first fit on the instrument that tracks the leaderboard.

Two numbers matter and they are different:

    ORACLE      per-world best arm, with hindsight. An upper bound, not a
                target -- no rule can see the future.
    CV          leave-fold-out accuracy of a rule using only day-6 state. This
                is what a shipped selector would actually get.

Reporting only the oracle would repeat the mistake this project keeps making:
an in-sample number that does not survive contact with held-out worlds.

    PYTHONPATH=. uv run python tools/day6_select.py oracle
    PYTHONPATH=. uv run python tools/day6_select.py fit --folds 6
"""

import argparse
import collections
import glob
import json
import os
import random
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CACHE = os.path.join(REPO, "rl", "bakeoff_runs")


def load(tag="30_0"):
    """-> {agent: {(episode, seat): (margin, win)}} for every laddered arm."""
    out = {}
    for f in sorted(glob.glob(os.path.join(CACHE, f"ladder_*_{tag}.json"))):
        name = os.path.basename(f)[len("ladder_"):-len(f"_{tag}.json")]
        out[name] = {tuple(json.loads(k)): tuple(v)
                     for k, v in json.load(open(f)).items()}
    return out


def arms_and_worlds(D, prefix="cl-"):
    arms = sorted([a for a in D if a.startswith(prefix)],
                  key=lambda a: -sum(1 for v in D[a].values() if v[1]))
    if not arms:
        return [], []
    ks = sorted(set.intersection(*[set(D[a]) for a in arms]))
    return arms, ks


def oracle(args):
    D = load(args.tag)
    arms, ks = arms_and_worlds(D)
    if not arms:
        sys.exit("no cl-* arms laddered yet")
    print(f"{len(arms)} cluster arms on {len(ks)} shared held-out ladder worlds\n")
    print(f"  {'arm':<12}{'win':>8}{'$/game':>12}")
    for a in arms:
        w = sum(1 for k in ks if D[a][k][1]) / len(ks)
        print(f"  {a:<12}{w:>8.3f}{st.mean(D[a][k][0] for k in ks):>12,.0f}")
    for extra in ("tape-sel",):
        if extra in D and set(ks) <= set(D[extra]):
            w = sum(1 for k in ks if D[extra][k][1]) / len(ks)
            print(f"  {extra:<12}{w:>8.3f}"
                  f"{st.mean(D[extra][k][0] for k in ks):>12,.0f}   (shipped)")
    ow = sum(1 for k in ks if any(D[a][k][1] for a in arms)) / len(ks)
    om = st.mean(max(D[a][k][0] for a in arms) for k in ks)
    best = arms[0]
    bw = sum(1 for k in ks if D[best][k][1]) / len(ks)
    print(f"\n  {'ORACLE':<12}{ow:>8.3f}{om:>12,.0f}   <- hindsight, not reachable")
    print(f"\n  oracle over best single arm: {100*(ow-bw):+.1f} points of win rate")
    if "tape-sel" in D and set(ks) <= set(D["tape-sel"]):
        sw = sum(1 for k in ks if D["tape-sel"][k][1]) / len(ks)
        print(f"  oracle over tape-sel:        {100*(ow-sw):+.1f} points")
    # How concentrated is the choice? If one arm wins nearly every world the
    # oracle is unreachable AND uninteresting -- just ship that arm.
    who = collections.Counter()
    for k in ks:
        who[max(arms, key=lambda a: D[a][k][0])] += 1
    print(f"\n  which arm is best, per world (top 6):")
    for a, n in who.most_common(6):
        print(f"    {a:<12}{n:>5} / {len(ks)}  = {100*n/len(ks):.0f}%")
    print(f"\n  a selector can only pay if this is SPREAD. If one arm takes most "
          f"worlds,\n  the oracle is just that arm plus noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oracle", "fit"])
    ap.add_argument("--tag", default="30_0")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    if args.mode == "oracle":
        oracle(args)
    else:
        sys.exit("fit: run `oracle` first -- if the per-world winner is not "
                 "spread across arms there is nothing to fit")


if __name__ == "__main__":
    main()
