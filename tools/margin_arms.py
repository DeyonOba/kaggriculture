"""Do the temporal board planes predict MARGIN, where they failed on flow?

`--ablate` closed the temporal stack on its own target and only on that
target. Its loss is

    MSE(flow, opponent's next 10 days x 9 products)
      + 0.3 * BCE(aux, opponent's plant mask at day+10)
      + 0.3 * MSE(shed, opponent's hidden shed)

and the reported R^2 is on the 10-day sum per product. The final margin is
not in that loss, is not a head, and was not even a column in the dataset
until `tools/attach_outcome.py`.

WHY THE TWO TARGETS SHOULD BEHAVE DIFFERENTLY, which is the whole reason this
is worth a second run rather than a rescue of the first.

  Ten-day flow is close to a deterministic function of the CURRENT board.
  What is planted, how mature it is, what stands in the pasture -- the
  engine's yield rules are known, so the present state screens off the past.
  That is the same thing `docs/...state-is-markov` found on price. History
  has little to add, and the six-seed table says exactly that: T_temporal_95
  negative in 5 of 5 bands.

  The margin at day 29 is a TRAJECTORY. How fast they built, whether they are
  accelerating or stalling, whether liquidation has started. Rates matter for
  extrapolating twenty days out in a way they do not for ten-day yield
  mechanics -- and rates are precisely what a lag plane carries and a single
  frame cannot.

The counter-evidence, stated up front: `tools/temporal_bands.py` already
tested day-1 and day-2 deltas against margin, per band, and they failed --
0 of 5 bands and 1 of 5. That was a LINEAR model on 48 aggregate scalars, not
a convolution on board planes, so it lowers the prior without closing it.

ON REUSING THE TRAINED TRUNK. A frozen probe -- take the flow-trained trunk,
fit only a new margin head -- costs seconds instead of hours, and it is
ONE-SIDED. A representation optimised for flow has already decided what to
discard from its input channels; if the temporal planes carry margin signal
that flow training threw away, the probe reads nothing, which is a false
negative on exactly the hypothesis under test. A positive probe would be
strong evidence. A null probe would be uninformative. So the arms are trained
against margin directly, and `--probe` exists only as the cheap screen.

    PYTHONPATH=. uv run python tools/margin_arms.py \\
        --data rl/datasets/s3_flow_v2band.npz --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ARMS = ["E_full_83", "T_temporal_95", "T_crop_103"]


def r2(y, p):
    d = float(((y - y.mean()) ** 2).sum())
    return float("nan") if d <= 0 else 1 - float(((y - p) ** 2).sum()) / d


def auc(y, s):
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def main() -> None:
    from tools.board_tensor import arm_channels
    from tools.boardnet import fit, predict
    from tools.s3_flow import BANDS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(
        REPO, "rl", "datasets", "s3_flow_v2band.npz"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--base", default="E_full_83")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    d = np.load(args.data)
    for k in ("M", "W"):
        if k not in d.files:
            raise SystemExit(f"{args.data} has no {k} column -- run "
                             f"tools/attach_outcome.py on it first")
    T, G, R = d["T"], d["G"].astype(int), d["R"]
    M = d["M"].astype("float32").reshape(-1, 1)
    W = d["W"].astype("float32")
    enc = int(d["ENC"][0]) if "ENC" in d.files else 1
    print(f"    encoder v{enc}   target: FINAL MARGIN (seat 0), "
          f"{len(T):,} rows, {len(np.unique(G))} games")
    print(f"    margin sd ${M.std():,.0f}   win rate {W.mean():.3f}\n")
    print(f"    {'arm':<18}{'ch':>4}{'seed':>6}{'val MSE':>10}"
          f"{'held-out R2 (margin)':>22}{'AUC (win)':>11}")

    band_of = np.full(len(T), -1)
    for bi, (lo, hi) in enumerate(BANDS):
        band_of[(R >= lo) & (R < hi)] = bi

    pooled: Dict[str, List[float]] = {}
    per_band: Dict[str, Dict[int, List[float]]] = {}
    per_auc: Dict[str, Dict[int, List[float]]] = {}
    for arm in args.arms:
        cols = arm_channels(arm)
        if max(cols) >= T.shape[1]:
            print(f"    {arm:<18} SKIPPED -- needs {max(cols)+1} channels")
            continue
        Ti = np.ascontiguousarray(T[:, cols].astype("float32"))
        for seed in args.seeds:
            net, stats, _ = fit(Ti, M, G, epochs=args.epochs, seed=seed,
                                verbose=False)
            gids = np.unique(G)
            rng = np.random.default_rng(seed)
            rng.shuffle(gids)
            va = np.isin(G, gids[:max(1, len(gids) // 5)])
            P = predict(net, stats, Ti[va]).reshape(-1)
            y, w, b = M[va].reshape(-1), W[va], band_of[va]
            m, a = r2(y, P), auc(w, P)
            pooled.setdefault(arm, []).append(m)
            for bi in range(len(BANDS)):
                sel = b == bi
                if sel.sum() < 30:
                    continue
                per_band.setdefault(arm, {}).setdefault(bi, []).append(
                    r2(y[sel], P[sel]))
                per_auc.setdefault(arm, {}).setdefault(bi, []).append(
                    auc(w[sel], P[sel]))
            print(f"    {arm:<18}{len(cols):>4}{seed:>6}{stats['val_mse']:>10.4f}"
                  f"{m:>22.4f}{a:>11.4f}", flush=True)

    ref = args.base if args.base in pooled else args.arms[0]
    nb = len(args.seeds)
    print(f"\n    baseline = {ref}   PAIRED over {nb} seeds\n")
    print(f"    {'arm':<18}{'mean R2':>10}{'vs base':>10}{'se':>9}{'t':>8}")
    base = np.asarray(pooled[ref])
    for arm in pooled:
        v = np.asarray(pooled[arm])
        dd = v - base
        if arm == ref or len(dd) < 2:
            print(f"    {arm:<18}{v.mean():>10.4f}{dd.mean():>+10.4f}")
            continue
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"    {arm:<18}{v.mean():>10.4f}{dd.mean():>+10.4f}{se:>9.4f}"
              f"{(dd.mean()/se if se > 0 else float('inf')):>+8.2f}")

    for title, store, fmt in (("R2 (margin)", per_band, "+13.4f"),
                              ("AUC (win)", per_auc, "+13.4f")):
        print(f"\n    PER BAND, {title}, paired vs {ref}\n")
        print(f"    {'arm':<18}" + "".join(
            f"{f'{lo}-{hi:.0f}' if hi < 1e8 else '2500+':>13}"
            for lo, hi in BANDS))
        for arm in store:
            cells = []
            for bi in range(len(BANDS)):
                v = store[arm].get(bi)
                bb = store[ref].get(bi)
                if not v or not bb:
                    cells.append(f"{'--':>13}")
                elif arm == ref:
                    cells.append(f"{np.mean(v):>13.4f}")
                else:
                    cells.append(f"{np.mean(v) - np.mean(bb):{fmt}}")
            print(f"    {arm:<18}" + "".join(cells))
        print(f"    {ref} row is absolute; the rest are deltas in the same band.")

    print(f"\n    H-margin: an arm passes on >= +0.01 R2 in >= 4 of "
          f"{len(BANDS)} bands. Selection is reported on win AUC alongside, "
          f"because the rating is win-based.")


if __name__ == "__main__":
    main()
