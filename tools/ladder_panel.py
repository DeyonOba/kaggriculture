"""Per-day metric panel from REAL recorded games, not self-play.

`rl/datasets/propagation_perturb.npz` -- the panel Φ was fitted on -- is
champ-h8 against champ-h8 with randomised openings. Every number in
`docs/shaping_spec.md` §4 therefore describes how well Φ reads *our own*
trajectories. Deployment is not that. `ladder_gate` draws recorded seats
across six rating bins, and `tools/net_ood.py` exists because a net trained
on a 21%-of-corpus slice closed +71% in-distribution and **−13%** against
champ-h8.

So this builds the same 48-metric panel from sessions replayed exactly:
recorded seed, both seats' recorded 720-step scripts, no agent in the loop at
all. The trajectories are real players', at real ratings, and nothing about
them was produced by anything we wrote.

Two rules that the self-play panel did not need:

  BOTH SEATS, GROUPED BY SESSION. Each seat of a game is its own trajectory
  with its own margin and win, which doubles the rows for free. But the two
  seats share a world, so they must never be split across train and test --
  the npz carries an explicit `G` (group) column and `tools/fit_potential.py`
  splits on it.

  STRATIFIED BY RATING. Sampling the corpus by frequency would hand back the
  rating band that plays most, which is exactly the mistake net_ood was
  written to catch. `--bins` draws equally from the six ladder_gate bands.

    PYTHONPATH=. uv run python tools/ladder_panel.py --n 400 --procs 5 \
        --out rl/datasets/ladder_panel.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TPD = 24
SAMPLE_HOUR = 12          # same hour as tools/propagation.py -- at dawn the
                          # hands and bags read 0 because units respawn empty
EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000),
         (2000, 2500), (2500, 99999)]


def rating_bin(r: Optional[float]) -> int:
    if r is None:
        return -1
    return next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)


def replay(spec):
    """Replay one recorded session and sample both seats every day.

    Returns (rows, days, seats, margins, wins, ratings) or None if the
    session does not run to a full season.
    """
    path, = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.early_metrics as em
    import tools.phase_metrics as pm

    try:
        with open(path) as fh:
            s = json.load(fh)
    except Exception:
        return None
    scripts = s.get("actions") or []
    if len(scripts) != 2 or not scripts[0] or not scripts[1]:
        return None

    # Scripts are indexed by the GLOBAL step, which only seat 0's
    # observation carries -- seat 1 gets day+hour instead. So the step is
    # passed in rather than read off the observation.
    def at(script, t):
        if t < len(script) and script[t]:
            return script[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}
    e = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    e.reset(2)
    rows: List[np.ndarray] = []
    days: List[int] = []
    seats: List[int] = []
    while not e.done:
        o0, o1 = e.state[0].observation, e.state[1].observation
        if o0["step"] % TPD == SAMPLE_HOUR:
            d = o0["step"] // TPD
            # The public observation is SHARED -- `farms` carries both farms
            # and the metric modules take the seat as an argument. Only
            # `private` (shed, seeds, unit inventories) differs, and seat 1's
            # observation is missing `step` entirely (it gets day+hour), so
            # seat 1 is read as (seat-0 public state, side=1, seat-1 private).
            for side, pv in ((0, o0.private), (1, o1.private)):
                rows.append(np.concatenate([em.features(o0, side, pv),
                                            pm.mid_features(o0, side, pv),
                                            pm.end_features(o0, side, pv)]))
                days.append(d)
                seats.append(side)
        t = o0["step"]
        e.step([at(scripts[0], t), at(scripts[1], t)])

    fin = [e.state[0].reward or 0.0, e.state[1].reward or 0.0]
    sc = s.get("scores") or [None, None]
    margins = [fin[0] - fin[1], fin[1] - fin[0]]
    wins = [float(fin[0] > fin[1]), float(fin[1] > fin[0])]
    return (np.stack(rows).astype("float32"), np.asarray(days, "int16"),
            np.asarray(seats, "int8"), margins, wins, sc,
            int(s.get("episode", 0)))


def names() -> List[str]:
    import tools.early_metrics as em
    import tools.phase_metrics as pm
    return (["E:" + n for n in em.NAMES]
            + ["M:" + n for n in pm.MID_NAMES]
            + ["D:" + n for n in pm.END_NAMES])


def pick_sessions(n: int, stratify: bool, seed: int,
                  team: Optional[str] = None) -> List[str]:
    idx_path = os.path.join(REPO, "rl", "datasets", "session_index.json")
    with open(idx_path) as fh:
        idx = json.load(fh)
    rng = np.random.default_rng(seed)
    rows = [(ep, r) for ep, r in idx.items() if r.get("path")]

    if team:
        # `teams` sits in the first few hundred bytes of the record, ahead of
        # the two 720-step action scripts. Parsing 5.8 GB of JSON to read a
        # name would take minutes; a byte scan of the head takes seconds.
        needle = json.dumps(team)[1:-1].encode()
        keep = []
        for ep, r in rows:
            try:
                with open(os.path.join(REPO, r["path"]), "rb") as fh:
                    if needle in fh.read(4096):
                        keep.append((ep, r))
            except Exception:
                continue
        rows = keep
        print(f"  {len(rows)} sessions for team {team!r}")

    if not stratify or team:
        order = rng.permutation(len(rows))[:n]
        return [os.path.join(REPO, rows[i][1]["path"]) for i in order]

    # equal draw from each ladder_gate rating band, by the MAX of the two
    # seats -- a 2,900 vs 800 game is a top-band game for our purposes
    banded: Dict[int, List[str]] = {}
    for ep, r in rows:
        sc = [x for x in (r.get("scores") or []) if x is not None]
        if not sc:
            continue
        banded.setdefault(rating_bin(max(sc)), []).append(
            os.path.join(REPO, r["path"]))
    live = sorted(b for b, v in banded.items() if v and b >= 0)
    per = max(1, n // max(1, len(live)))
    out: List[str] = []
    for b in live:
        pool = banded[b]
        take = min(per, len(pool))
        out += [pool[i] for i in rng.permutation(len(pool))[:take]]
        print(f"  bin {b} ({EDGES[b][0]}-{EDGES[b][1]}): "
              f"{len(pool):5d} available, {take} drawn")
    return out[:n]


def build(args) -> None:
    from multiprocessing import Pool
    paths = pick_sessions(args.n, not args.no_stratify, args.seed, args.team)
    print(f"replaying {len(paths)} recorded sessions on {args.procs} procs ...",
          flush=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        got = [r for r in pool.imap_unordered(replay, [(p,) for p in paths])
               if r is not None]
    print(f"  {len(got)}/{len(paths)} replayed in {time.time()-t0:.0f}s")
    if not got:
        raise SystemExit("nothing replayed")

    n_days = max(int(r[1].max()) + 1 for r in got)
    X, D, S, M, W, G, R, EP = [], [], [], [], [], [], [], []
    for gi, (rows, days, seats, margins, wins, sc, ep) in enumerate(got):
        if int(days.max()) + 1 != n_days:
            continue                       # short game: drop, keep the panel rectangular
        X.append(rows); D.append(days); S.append(seats)
        M.append(np.array([margins[s] for s in seats], "float32"))
        W.append(np.array([wins[s] for s in seats], "float32"))
        G.append(np.full(len(days), gi, "int32"))
        R.append(np.array([(sc[s] if s < len(sc) and sc[s] is not None else np.nan)
                           for s in seats], "float32"))
        EP.append(np.full(len(days), ep, "int64"))
    out = dict(X=np.concatenate(X), D=np.concatenate(D), S=np.concatenate(S),
               M=np.concatenate(M), W=np.concatenate(W), G=np.concatenate(G),
               R=np.concatenate(R), EP=np.concatenate(EP),
               names=np.array(names()))
    np.savez_compressed(args.out, **out)
    ng = len(np.unique(out["G"]))
    rt = out["R"][~np.isnan(out["R"])]
    print(f"-> {args.out}  {len(out['X']):,} rows, {ng} sessions x 2 seats "
          f"x {n_days} days")
    print(f"   ratings {np.nanmin(rt):.0f}-{np.nanmax(rt):.0f}, "
          f"median {np.median(rt):.0f}; margin sd ${out['M'].std():,.0f}, "
          f"win {out['W'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--team", default=None,
                    help="only sessions containing this team name")
    ap.add_argument("--no-stratify", action="store_true",
                    help="sample the corpus as it is instead of equally per "
                         "rating band (which is the net_ood mistake)")
    ap.add_argument("--out", default=os.path.join(
        REPO, "rl", "datasets", "ladder_panel.npz"))
    build(ap.parse_args())


if __name__ == "__main__":
    main()
