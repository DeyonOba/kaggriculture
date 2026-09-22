"""Does history help, band by band? The cheap version, before the 10 GB build.

`docs/state_capture_spec.md` proposes a three-frame state capture. Before
spending a 10 GB tensor pipeline on it, the same question can be asked of the
48-metric panels that already exist: take the per-day metrics, append their
one- and two-day differences, and see whether the extra columns buy anything.

This is not the same model as the board CNN and it cannot settle the CNN's
case. It settles something cheaper and logically prior: **is there rate
information in the state at all, and is it in every band or only at the top?**
If a linear model cannot find it in any band, a convolution over the same
underlying quantities is a long shot; if it finds it in four bands, the tensor
build is justified.

Two protocol rules, both from things that went wrong before:

  SPLIT BY SESSION. The two seats of a game share a world and its shop draw,
  so seat 0 in train and seat 1 in test is a leak, not a held-out game.

  REPORT PER BAND, never pooled first. Every board-tensor arm to date was
  drawn with a rating FLOOR, and 14,211 of 21,054 corpus sessions sit in the
  2500+ band -- so those arms were effectively top-games-only, which is how a
  net that closed +71% in-distribution went -13% against champ-h8.

    PYTHONPATH=. uv run python tools/temporal_bands.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.fit_potential import (Panel, ridge, r2, auc, _zscore,  # noqa: E402
                                 live_columns, pick_alpha)

BANDS = [("<1000", 0, 1000), ("1000-1500", 1000, 1500), ("1500-2000", 1500, 2000),
         ("2000-2500", 2000, 2500), ("2500+", 2500, 1e9)]
LAGS = (1, 2)


def lagged(p: Panel) -> Tuple[np.ndarray, np.ndarray]:
    """X with day-1 and day-2 differences appended, and a validity mask.

    A row's lags come from the SAME (session, seat) trajectory. Rows whose
    lag does not exist -- days 0 and 1 -- are marked invalid rather than
    zero-filled, because a zero difference is a real and different statement
    from "unknown".
    """
    n, c = p.X.shape
    out = np.zeros((n, c * (1 + len(LAGS))), dtype=np.float64)
    out[:, :c] = p.X
    ok = np.ones(n, bool)
    key = p.game.astype(np.int64) * 8 + p.seat.astype(np.int64)
    index: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        index[(int(key[i]), int(p.day[i]))] = i
    for j, lag in enumerate(LAGS):
        for i in range(n):
            src = index.get((int(key[i]), int(p.day[i]) - lag))
            if src is None:
                ok[i] = False
                continue
            out[i, c * (1 + j):c * (2 + j)] = p.X[i] - p.X[src]
    return out, ok


def per_day_fit(X: np.ndarray, y: np.ndarray, day: np.ndarray, game: np.ndarray,
                rows_ok: np.ndarray, tr: np.ndarray, te: np.ndarray,
                folds: int, seed: int) -> np.ndarray:
    """Fit one ridge per day on train sessions; return predictions for all rows."""
    pred = np.full(len(X), np.nan)
    in_tr = np.isin(game, tr) & rows_ok
    in_te = np.isin(game, te) & rows_ok
    for d in np.unique(day):
        m_tr, m_te = in_tr & (day == d), in_te & (day == d)
        if m_tr.sum() < 20 or m_te.sum() == 0:
            continue
        cols = live_columns(X[m_tr])
        if not len(cols):
            continue
        Xtr, Xte = X[m_tr][:, cols], X[m_te][:, cols]
        a = pick_alpha(Xtr, y[m_tr], game[m_tr], folds, seed + int(d))
        Ztr, Zte = _zscore(Xtr, Xte)
        w, b = ridge(Ztr, y[m_tr], a)
        # Clamp to the range the fit actually SAW, widened by half. A linear
        # model extrapolates without limit, and a column that barely moved in
        # training can arrive far outside that range at test time -- measured
        # once at 3,677 sigmas, and again here as R^2 -1.683 in the smallest
        # band. Without this the harness reports a feature as catastrophic
        # when the catastrophe is the harness.
        tr_pred = Ztr @ w + b
        lo, hi = float(tr_pred.min()), float(tr_pred.max())
        pad = 0.5 * (hi - lo)
        pred[m_te] = np.clip(Zte @ w + b, lo - pad, hi + pad)
    return pred


# The audited-but-unencoded channels (tools/missing_features.py), grouped by
# the reason each was expected to matter. Ablating the groups says WHICH gap
# pays, which a single all-or-nothing arm cannot.
GROUPS = {
    "clocks": ("X:lifespan_left", "X:dying_3d", "X:unfed_streak",
               "X:starving", "X:animal_age"),
    "banked": ("X:pending_care", "X:care_cover", "X:fert_active",
               "X:fert_days_left"),
    "opp": ("X:opp_money", "X:hires_today", "X:opp_hires_today", "X:opp_hands"),
    "seeds": tuple(f"X:seed_{c}" for c in
                   ("wheat", "carrot", "tomato", "strawberry", "melon")),
    "shops": None,       # every X:shop_* column
    "minv": None,        # every X:minv_* column
}


def feature_arms(p: Panel):
    base = [i for i, n in enumerate(p.names) if not n.startswith("X:")]
    idx = {n: i for i, n in enumerate(p.names)}
    arms = {f"base ({len(base)})": np.array(base)}
    for g, cols in GROUPS.items():
        if cols is None:
            sel = [i for i, n in enumerate(p.names) if n.startswith(f"X:{g}_")]
        else:
            sel = [idx[c] for c in cols if c in idx]
        if sel:
            arms[f"+{g} ({len(base)+len(sel)})"] = np.array(sorted(base + sel))
    allx = [i for i, n in enumerate(p.names) if n.startswith("X:")]
    arms[f"+ALL ({len(base)+len(allx)})"] = np.array(sorted(base + allx))
    return arms


def report(p: Panel, args) -> None:
    if args.mode == "features":
        ok = np.ones(len(p.X), bool)
        Xl = p.X.astype(np.float64)
        arms = {k: (v, ok) for k, v in feature_arms(p).items()}
    else:
        Xl, ok = lagged(p)
        c = p.X.shape[1]
        arms = {"now only (48)": (np.arange(c), ok),
                "now + d1 (96)": (np.arange(2 * c), ok),
                "now + d1 + d2 (144)": (np.arange(3 * c), ok)}
    band_of = np.full(len(p.X), -1)
    for bi, (_, lo, hi) in enumerate(BANDS):
        band_of[(p.rating >= lo) & (p.rating < hi)] = bi

    acc: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
        k: {bi: [] for bi in range(len(BANDS))} for k in arms}
    day10: Dict[str, List[Tuple[float, float]]] = {k: [] for k in arms}

    for rep in range(args.repeats):
        seed = args.seed + 997 * rep
        tr, te = p.split(args.train_frac, seed)
        for k, (cols, rows) in arms.items():
            pred = per_day_fit(Xl[:, cols], p.margin, p.day, p.game, rows,
                               tr, te, args.folds, seed)
            got = ~np.isnan(pred)
            for bi in range(len(BANDS)):
                m = got & (band_of == bi)
                if m.sum() < 30:
                    continue
                acc[k][bi].append((r2(p.margin[m], pred[m]),
                                   auc(p.win[m], pred[m])))
            m = got & (p.day == 10)
            if m.sum() >= 30:
                day10[k].append((r2(p.margin[m], pred[m]), auc(p.win[m], pred[m])))

    print(f"\nPER BAND, held-out sessions, {args.repeats} splits "
          f"(mean R^2 / AUC vs the final margin & win)\n")
    hdr = f"  {'band':>11} {'n rows':>8} " + " ".join(f"{k:>21}" for k in arms)
    print(hdr)
    for bi, (nm, lo, hi) in enumerate(BANDS):
        nrow = int(((p.rating >= lo) & (p.rating < hi)).sum())
        cells = []
        for k in arms:
            v = acc[k][bi]
            if not v:
                cells.append(f"{'--':>21}"); continue
            r = np.mean([x[0] for x in v]); a = np.mean([x[1] for x in v])
            cells.append(f"{r:+9.3f} / {a:8.3f}")
        print(f"  {nm:>11} {nrow:8,d} " + " ".join(cells))

    print(f"\n  {'DAY 10 only':>11} {'':8} " + " ".join(
        (f"{np.mean([x[0] for x in day10[k]]):+9.3f} / "
         f"{np.mean([x[1] for x in day10[k]]):8.3f}") if day10[k] else f"{'--':>21}"
        for k in arms))

    base = list(arms)[0]
    print(f"\nDELTA against `{base}` -- the kill criteria in "
          f"docs/state_capture_spec.md S6\n")
    print(f"  {'band':>11} " + " ".join(f"{k:>21}" for k in arms if k != base))
    wins = {k: 0 for k in arms if k != base}
    for bi, (nm, _, _) in enumerate(BANDS):
        if not acc[base][bi]:
            continue
        b_r = np.mean([x[0] for x in acc[base][bi]])
        b_a = np.mean([x[1] for x in acc[base][bi]])
        cells = []
        for k in arms:
            if k == base or not acc[k][bi]:
                if k != base:
                    cells.append(f"{'--':>21}")
                continue
            dr = np.mean([x[0] for x in acc[k][bi]]) - b_r
            da = np.mean([x[1] for x in acc[k][bi]]) - b_a
            if dr > 0.01:
                wins[k] += 1
            cells.append(f"{dr:+9.3f} / {da:+8.3f}")
        print(f"  {nm:>11} " + " ".join(cells))
    print()
    for k, w in wins.items():
        verdict = ("PASSES the >=4-band bar" if w >= 4 else
                   f"FAILS -- gains >0.01 R^2 in only {w}/{len(BANDS)} bands")
        print(f"  {k:>21}: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default=os.path.join(
        REPO, "rl", "datasets", "ladder_panel.npz"))
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--mode", choices=("lags", "features"), default="lags",
                    help="`features` ablates the audited missing channels "
                         "instead of appending temporal lags")
    args = ap.parse_args()
    p = Panel(args.panel)
    print(f"{args.panel}: {p.n_games} sessions, {len(p.X):,} rows, "
          f"{p.X.shape[1]} metrics")
    report(p, args)


if __name__ == "__main__":
    main()
