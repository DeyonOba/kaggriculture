"""What does the knob archive already say about this knob?

Run this BEFORE gating a knob change. 30,272 counterfactual-bandit samples sit
in `rl/datasets/bandit/knob_*.jsonl` and nothing reads them back. On
2026-09-17 a `geese` variant was built, gated on the panel (~45 min) and
laddered (~40 min) and rejected -- and the archive already held
`geese 2: -$2,160 +/- 700 (n=236)` and `geese 3: -$3,185 +/- 1,434`. The same
check also retired a planned `melon_opening` sweep in one second:
8 is null (+386 +/- 987), 16 is negative (-658 +/- 443).

Each sample is ONE world played twice -- the champion twin and the same world
with a single (day, head, offset) deviation -- so `gain_margin` is a paired
counterfactual, and the win columns are that world's win for each side.

Caveats worth carrying: the base was champ-h4/h5, not champ-h8; the deviation
is applied on one day rather than season-long; and the objective is margin,
which [[gate on WIN RATE]] says moves the size of a loss. So a negative here is
a strong prior against, not a proof -- but it is free.

    PYTHONPATH=. uv run python tools/knob_archive.py               # everything
    PYTHONPATH=. uv run python tools/knob_archive.py geese tomato  # filter
"""

import collections
import glob
import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = os.path.join(REPO, "rl", "datasets", "bandit", "knob_*.jsonl")


def load():
    gain = collections.defaultdict(list)
    wins = collections.defaultdict(list)
    for f in glob.glob(PAT):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not d.get("applied") or d.get("noop"):
                continue
            head, off, g = d.get("head"), d.get("offset"), d.get("gain_margin")
            if head is None or g is None:
                continue
            gain[(head, off)].append(float(g))
            if d.get("final") is not None and d.get("base_final") is not None:
                wins[(head, off)].append(
                    (d["final"] > d["opp"], d["base_final"] > d["base_opp"]))
    return gain, wins


def main():
    want = [a.lower() for a in sys.argv[1:]]
    gain, wins = load()
    rows = []
    for (head, off), g in gain.items():
        if want and not any(w in head.lower() for w in want):
            continue
        n = len(g)
        se = st.stdev(g) / n ** 0.5 if n > 1 else 0.0
        w = wins[(head, off)]
        bw = sum(b for _, b in w) / len(w) if w else 0.0
        cw = sum(c for c, _ in w) / len(w) if w else 0.0
        rows.append((head, off, st.mean(g), se, n, bw, cw))
    if not rows:
        print("no samples" + (f" matching {want}" if want else ""))
        return
    print(f"{'knob':<30}{'val':>8}{'margin':>10}{'se':>8}{'n':>7}"
          f"{'win base':>10}{'win cand':>10}{'':>5}")
    for head, off, m, se, n, bw, cw in sorted(rows, key=lambda r: (r[0], str(r[1]))):
        flag = ""
        if n >= 40:
            if m + 2 * se < 0:
                flag = "REFUTED"
            elif m - 2 * se > 0 and cw >= bw:
                flag = "positive"
        print(f"{head:<30}{str(off):>8}{m:>10,.0f}{se:>8,.0f}{n:>7}"
              f"{bw:>10.3f}{cw:>10.3f}  {flag}")
    print(f"\n  {len(rows)} arms. REFUTED = mean + 2se < 0 at n >= 40: do not "
          f"re-gate without a reason\n  the archive does not cover (different "
          f"base agent, season-long instead of one-day, win rate).")


if __name__ == "__main__":
    main()
