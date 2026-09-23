"""Attach the final margin and win to a flow dataset, without replaying it.

`tools/s3_flow.py --build` stores the OPPONENT-FLOW target and nothing about
who won. Testing whether the temporal board planes predict *margin* rather
than *flow* needs the outcome as a label -- and it does not need another 25
minutes of replay, because `finals` is already in the session index and
`encode_day` builds every tensor from seat 0's perspective
(`board_pair_all(dawn[day], 0, ...)`), so

    margin = finals[0] - finals[1]        win = finals[0] > finals[1]

The only hard part is the row -> session mapping. `G` is `enumerate` over the
sessions that replayed successfully, in the order `_pick` returned them, so
re-deriving `_pick` with the same arguments reproduces it -- PROVIDED no
session was dropped. That is checked rather than assumed: the reconstructed
per-row rating vector must equal the `R` the build stored, or this refuses to
write anything.

    PYTHONPATH=. uv run python tools/attach_outcome.py \\
        --data rl/datasets/s3_flow_v2band.npz --n 1200 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=1200)
    # MUST match the seed the dataset was built with. s3_flow's --seed
    # defaults to 7, not 0, and a build run without the flag used 7 -- so 0
    # here silently reconstructs a different 1,075 sessions. The rating
    # assertion below is what turns that into a refusal instead of a wrong
    # margin column.
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bands", action="store_true", default=True)
    args = ap.parse_args()

    from tools.s3_flow import _pick, _session_rating

    d = dict(np.load(args.data))
    if "M" in d and "W" in d:
        print(f"{args.data} already carries M/W -- nothing to do")
        return
    G = d["G"].astype(int)
    n_sess = int(G.max()) + 1

    paths = _pick(args.n, 0.0, args.seed, bands=args.bands)
    if len(paths) != n_sess:
        raise SystemExit(
            f"_pick returned {len(paths)} sessions but the dataset holds "
            f"{n_sess}: some session was dropped at build time, so the "
            f"row -> session mapping cannot be reconstructed this way")

    # The check that makes the mapping trustworthy rather than plausible.
    want = np.asarray([_session_rating(p) for p in paths], dtype="float32")
    got_per_sess = np.asarray([d["R"][G == i][0] for i in range(n_sess)],
                              dtype="float32")
    if not np.allclose(want, got_per_sess, equal_nan=True):
        bad = int((~np.isclose(want, got_per_sess, equal_nan=True)).sum())
        raise SystemExit(
            f"rating mismatch on {bad}/{n_sess} sessions -- the reconstructed "
            f"order is NOT the build order, refusing to attach outcomes")

    with open(os.path.join(REPO, "rl", "datasets", "session_index.json")) as fh:
        idx = json.load(fh)
    by_path = {v["path"]: v for v in idx.values() if v.get("path")}

    margin = np.zeros(n_sess, dtype="float32")
    win = np.zeros(n_sess, dtype="float32")
    missing = 0
    for i, p in enumerate(paths):
        rel = p[len(REPO) + 1:] if p.startswith(REPO) else p
        rec = by_path.get(rel) or by_path.get(p)
        fin = (rec or {}).get("finals") or []
        if len(fin) != 2:
            missing += 1
            margin[i], win[i] = np.nan, np.nan
            continue
        margin[i] = float(fin[0]) - float(fin[1])
        win[i] = 1.0 if fin[0] > fin[1] else 0.0

    d["M"] = margin[G]
    d["W"] = win[G]
    np.savez_compressed(args.data, **d)
    live = int((~np.isnan(margin)).sum())
    print(f"-> {args.data}")
    print(f"   rating check PASS on all {n_sess} sessions")
    print(f"   margin attached for {live}/{n_sess} sessions"
          + (f" ({missing} without finals)" if missing else ""))
    m = margin[~np.isnan(margin)]
    print(f"   margin mean ${m.mean():,.0f}  sd ${m.std():,.0f}  "
          f"seat-0 win rate {np.nanmean(win):.3f}")


if __name__ == "__main__":
    main()
