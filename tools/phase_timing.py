"""When does a player COMMENCE each phase, and does the timing rank?

`docs/game_phases.md` defines four phases by day number -- 0-7, 7-17, 18-25,
26-29 -- taken from where our own P(win) curve bends. That is a description
of champ-h8, not a law of the game. A stronger player may place capital
faster, contest longer, or start liquidating later, and none of that is
visible in a fixed calendar.

So this reads the phase boundaries off each SEAT's own trajectory, as
landmarks in the per-day metric panel, and then asks where a player's
landmarks sit in the ladder-wide distribution.

THE LANDMARKS, and why each is the boundary it claims to be.

  build_done   first day `E:occ_all` reaches 90% of that seat's season max.
               The opening's job is to convert cash into placed capital
               (`docs/game_phases.md`), so the opening ends when placement
               stops.
  peak_board   the day `M:board_value` peaks. Capital on the board is an
               asset early and a liability late -- correlation +0.366 with
               the outcome at day 4 and -0.200 at day 20 -- so the turn is
               the contest/consolidation boundary.
  peak_stock   the day `E:shed_frac` peaks, i.e. the last day stock is still
               accumulating. After it the shed only drains: liquidation.
  first_sell   first day `E:shed_frac` falls while `M:board_value` is still
               within 10% of its peak -- selling before the board turns.

All four are computed per seat from that seat's own curve, so a player who
runs the whole season faster is measured as faster rather than as noisy.

    PYTHONPATH=. uv run python tools/phase_timing.py \\
        --panel rl/datasets/ladder_panel.npz \\
        --focus rl/datasets/mmpq_panel.npz --label "M & M & P & Q"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.fit_potential import Panel  # noqa: E402

LANDMARKS = ("build_done", "peak_board", "peak_stock", "first_sell")


def seat_curves(p: Panel) -> List[Tuple[int, int, Dict[str, np.ndarray], float, float, float]]:
    """One entry per (group, seat): its per-day curves plus outcome+rating."""
    out = []
    for g in np.unique(p.game):
        for s in np.unique(p.seat[p.game == g]):
            m = (p.game == g) & (p.seat == s)
            order = np.argsort(p.day[m])
            idx = np.flatnonzero(m)[order]
            if len(idx) < p.n_days:
                continue
            cur = {n: p.X[idx, i] for i, n in enumerate(p.names)}
            out.append((int(g), int(s), cur, float(p.margin[idx[0]]),
                        float(p.win[idx[0]]), float(p.rating[idx[0]])))
    return out


def landmarks(cur: Dict[str, np.ndarray]) -> Dict[str, float]:
    occ = cur["E:occ_all"]
    board = cur["M:board_value"]
    stock = cur["E:shed_frac"]
    n = len(occ)

    lo, hi = float(occ.min()), float(occ.max())
    thr = lo + 0.9 * (hi - lo)
    build = int(np.argmax(occ >= thr)) if hi > lo else float("nan")

    pk_board = int(np.argmax(board))
    pk_stock = int(np.argmax(stock))

    # first sustained drawdown of stock while the board is still near peak
    first = float("nan")
    bmax = float(board.max()) or 1.0
    for d in range(1, n):
        if stock[d] < stock[d - 1] and board[d] >= 0.9 * bmax:
            first = d
            break
    return {"build_done": float(build), "peak_board": float(pk_board),
            "peak_stock": float(pk_stock), "first_sell": first}


def table(p: Panel, label: str) -> Dict[str, np.ndarray]:
    rows = seat_curves(p)
    acc: Dict[str, List[float]] = {k: [] for k in LANDMARKS}
    meta = {"margin": [], "win": [], "rating": []}
    for _, _, cur, mg, wn, rt in rows:
        lm = landmarks(cur)
        for k in LANDMARKS:
            acc[k].append(lm[k])
        meta["margin"].append(mg); meta["win"].append(wn); meta["rating"].append(rt)
    out = {k: np.array(v, float) for k, v in acc.items()}
    out.update({k: np.array(v, float) for k, v in meta.items()})
    print(f"  {label}: {len(rows)} seat-trajectories")
    return out


def pct_rank(value: float, pool: np.ndarray) -> float:
    pool = pool[~np.isnan(pool)]
    if not len(pool) or np.isnan(value):
        return float("nan")
    return float((pool < value).mean() + 0.5 * (pool == value).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default=os.path.join(REPO, "rl", "datasets",
                                                    "ladder_panel.npz"))
    ap.add_argument("--focus", default=None, help="a second panel to rank")
    ap.add_argument("--label", default="focus")
    ap.add_argument("--self-play", default=os.path.join(
        REPO, "rl", "datasets", "propagation_perturb.npz"))
    args = ap.parse_args()

    print("reading panels ...")
    base = table(Panel(args.panel), "ladder (all bands)")
    focus = table(Panel(args.focus), args.label) if args.focus else None
    ours = None
    if args.self_play and os.path.isfile(args.self_play):
        ours = table(Panel(args.self_play), "champ-h8 self-play")

    print(f"\nPHASE COMMENCEMENT, day of season (median [IQR])\n")
    hdr = f"  {'landmark':>12} {'ladder all':>18}"
    if focus is not None:
        hdr += f" {args.label:>20} {'pct-rank':>9}"
    if ours is not None:
        hdr += f" {'champ-h8':>14}"
    print(hdr)
    for k in LANDMARKS:
        b = base[k][~np.isnan(base[k])]
        line = (f"  {k:>12} {np.median(b):6.1f} "
                f"[{np.percentile(b,25):4.1f}-{np.percentile(b,75):4.1f}]")
        if focus is not None:
            f = focus[k][~np.isnan(focus[k])]
            r = pct_rank(float(np.median(f)), b)
            line += (f" {np.median(f):8.1f} "
                     f"[{np.percentile(f,25):4.1f}-{np.percentile(f,75):4.1f}]"
                     f" {r:9.0%}")
        if ours is not None:
            o = ours[k][~np.isnan(ours[k])]
            line += f" {np.median(o):14.1f}"
        print(line)

    # Does the timing predict anything? Rank-correlate each landmark with the
    # seat's own rating and its own result, on the ladder panel.
    print(f"\nDOES TIMING TRACK STRENGTH?  (Spearman on {len(base['margin'])} "
          f"ladder seat-trajectories)\n")
    print(f"  {'landmark':>12} {'vs rating':>10} {'vs margin':>10} {'vs win':>8}")
    for k in LANDMARKS:
        v = base[k]
        for tgt, nm in (("rating", "r"), ):
            pass
        cells = []
        for tgt in ("rating", "margin", "win"):
            y = base[tgt]
            ok = ~np.isnan(v) & ~np.isnan(y)
            if ok.sum() < 10:
                cells.append(float("nan")); continue
            rv = np.argsort(np.argsort(v[ok])).astype(float)
            ry = np.argsort(np.argsort(y[ok])).astype(float)
            cells.append(float(np.corrcoef(rv, ry)[0, 1]))
        print(f"  {k:>12} {cells[0]:+10.3f} {cells[1]:+10.3f} {cells[2]:+8.3f}")

    # Top band only -- the comparison that matters for a 3,000-rated player.
    top = base["rating"] >= 2500
    if top.sum() > 20 and focus is not None:
        print(f"\nAGAINST THE 2500+ BAND ONLY ({int(top.sum())} seat-trajectories)\n")
        print(f"  {'landmark':>12} {'2500+ median':>13} {args.label:>20} {'pct-rank':>9}")
        for k in LANDMARKS:
            b = base[k][top & ~np.isnan(base[k])]
            f = focus[k][~np.isnan(focus[k])]
            if not len(b):
                continue
            print(f"  {k:>12} {np.median(b):13.1f} {np.median(f):20.1f} "
                  f"{pct_rank(float(np.median(f)), b):9.0%}")


if __name__ == "__main__":
    main()
