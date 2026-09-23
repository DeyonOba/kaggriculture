"""Score the flow arms PER DAY, in dollars -- the test the stop rule should have been.

WHY THIS EXISTS. `--ablate` reports

    R2_p = 1 - sum_b ( sum_j yhat - sum_j y )^2 / sum_b ( sum_j y - mean )^2

with the **sum over horizon days inside the residual**. Timing error cancels
there. A forecast that gets the ten-day total right and every individual day
wrong scores perfectly -- and *when* supply arrives is precisely what a lag
plane would improve, while *how much over ten days* is close to a
deterministic function of the standing board.

The evidence that this mattered was already in the six-seed table and I read
past it. Paired within seed, `T_crop_103` against `E_full_83`:

    val MSE (per day x product, lower better)   -0.0108   se 0.0016   t -6.75   6/6 seeds
    R2      (10-day SUM, higher better)         -0.0012                         2/6 seeds

The strongest paired effect in this line of work, in the opposite direction
from the number the stop rule fired on. The training loss and the early-stop
criterion both see all 90 cells, so the NETS were never blind -- only the
statistic they were judged by.

THE SCORE AND THE LOSS, one weight for both:

    alpha[j,p] = gamma^(j-1) * (sd[j,p] * price[p])^2 / Z      mean(alpha) = 1

    sd * price converts a z-space residual back to DOLLARS, because the target
    is z-scored per column.

    gamma^(j-1) decays the horizon. The day plan is rebuilt EVERY day at hour
    0, so a day-10 error is re-forecast nine times before it binds; only the
    near horizon is ever acted on irrevocably. gamma is NOT measured, so it is
    swept with gamma = 1 in the sweep as the null.

    loss  = mean_b mean_jp alpha * (yhat - z)^2
    score = sum_jp (alpha / sum alpha) * R2[j,p],  R2 computed PER CELL

`fit` takes `col_w`, and `mse_loss(out*w, y*w)` is `mean((out-y)^2 w^2)`, so
the tensor passed is sqrt(alpha), not alpha.

The legacy summed-R2 is reported in every table alongside. A new metric that
only ever flatters the new arms is not evidence.

    PYTHONPATH=. uv run python tools/perday_arms.py --block A
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def cell_r2(Y, P, alpha, n_p, H, rows=None):
    """alpha-weighted mean of PER-CELL R^2 over cells that vary."""
    num = den = 0.0
    live = 0
    for j in range(H):
        for pi in range(n_p):
            k = j * n_p + pi
            y, q = (Y[:, k], P[:, k]) if rows is None else (Y[rows, k], P[rows, k])
            d = float(((y - y.mean()) ** 2).sum())
            if d <= 1e-6 * max(1.0, float((y ** 2).sum())) or len(y) < 2:
                continue
            a = float(alpha[k])
            num += a * (1 - float(((y - q) ** 2).sum()) / d)
            den += a
            live += 1
    return (num / den if den > 0 else float("nan")), live


def summed_r2(Y, P, n_p, H, rows=None):
    """The legacy statistic: sum the horizon, then R^2. Kept as the guard."""
    out = []
    for pi in range(n_p):
        c = [j * n_p + pi for j in range(H)]
        y, q = Y[:, c].sum(1), P[:, c].sum(1)
        if rows is not None:
            y, q = y[rows], q[rows]
        d = float(((y - y.mean()) ** 2).sum())
        if d <= 1e-6 * max(1.0, float((y ** 2).sum())) or len(y) < 2:
            continue
        out.append(1 - float(((y - q) ** 2).sum()) / d)
    return (float(np.mean(out)) if out else float("nan")), len(out)


def make_alpha(F, price, n_p, H, gamma, dollars):
    sd = np.maximum(F.std(0), 1e-3)                      # (90,)
    a = np.ones(n_p * H, dtype="float64")
    for j in range(H):
        for pi in range(n_p):
            k = j * n_p + pi
            a[k] = (gamma ** j) * ((sd[k] * price[pi]) ** 2 if dollars else 1.0)
    return a / a.mean()


def main() -> None:
    from kagent.constants import MARKET_PARAMS
    from tools.board_tensor import arm_channels
    from tools.boardnet import fit, predict
    from tools.s3_flow import BANDS, PRODUCTS, H

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(
        REPO, "rl", "datasets", "s3_flow_v2band.npz"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--block", choices=("A", "B"), default="A",
                    help="A: all 3 arms, CURRENT loss, new score -- the "
                         "re-test of the arms the stop rule killed. "
                         "B: the loss sweep (dollar weight x gamma) on E and "
                         "the best temporal arm.")
    args = ap.parse_args()

    d = np.load(args.data)
    T, F, A, G, R = d["T"], d["F"], d["A"], d["G"].astype(int), d["R"]
    n_p = len(PRODUCTS)
    price = np.asarray([MARKET_PARAMS[p]["base"] for p in PRODUCTS], "float64")

    # THREE scores, not two, because per-cell scoring and dollar weighting
    # are two separate changes against the legacy statistic and Block A must
    # be able to attribute a pass to one of them:
    #
    #   summed      legacy: sum the horizon inside the residual   (the guard)
    #   perday eq   per cell, UNWEIGHTED  -> isolates the metric change alone
    #   perday $    per cell, dollar-weighted -> adds the weighting
    #
    # gamma is absent from all three. gamma = 1 is the null and belongs in
    # Block B; an unmeasured parameter has no place inside the decisive test.
    alpha_eq = np.ones(n_p * H, dtype="float64")
    alpha_usd = make_alpha(F, price, n_p, H, gamma=1.0, dollars=True)

    if args.block == "A":
        configs = [(arm, "equal (current)", None)
                   for arm in ("E_full_83", "T_temporal_95", "T_crop_103")]
    else:
        configs = []
        for arm in ("E_full_83", "T_crop_103"):
            for g in (1.0, 0.9):
                a = make_alpha(F, price, n_p, H, gamma=g, dollars=True)
                configs.append((arm, f"dollar g={g}", np.sqrt(a).astype("float32")))

    print(f"    block {args.block}   {len(T):,} rows, {len(np.unique(G))} games, "
          f"{len(configs)} configs x {len(args.seeds)} seeds")
    print(f"    scores: per-cell equal | per-cell dollar | legacy summed "
          f"(no gamma in any of them)\n")
    print(f"    {'arm':<16}{'loss':<16}{'seed':>5}{'perday eq':>11}"
          f"{'perday $':>10}{'summed':>10}")

    per: Dict[str, Dict[int, List[float]]] = {}
    peq: Dict[str, Dict[int, List[float]]] = {}
    leg: Dict[str, Dict[int, List[float]]] = {}
    band_of = np.full(len(T), -1)
    for bi, (lo, hi) in enumerate(BANDS):
        band_of[(R >= lo) & (R < hi)] = bi

    for arm, label, cw in configs:
        cols = arm_channels(arm)
        Ti = np.ascontiguousarray(T[:, cols].astype("float32"))
        key = f"{arm}|{label}"
        for seed in args.seeds:
            net, stats, _ = fit(Ti, F, G, A=A, S=None, epochs=args.epochs,
                                seed=seed, col_w=cw, verbose=False)
            gids = np.unique(G)
            rng = np.random.default_rng(seed)
            rng.shuffle(gids)
            va = np.isin(G, gids[:max(1, len(gids) // 5)])
            P, Yv, bd = predict(net, stats, Ti[va]), F[va], band_of[va]
            pd_, _ = cell_r2(Yv, P, alpha_usd, n_p, H)
            pe_, _ = cell_r2(Yv, P, alpha_eq, n_p, H)
            sm, _ = summed_r2(Yv, P, n_p, H)
            per.setdefault(key, {}).setdefault(-1, []).append(pd_)
            peq.setdefault(key, {}).setdefault(-1, []).append(pe_)
            leg.setdefault(key, {}).setdefault(-1, []).append(sm)
            for bi in range(len(BANDS)):
                sel = bd == bi
                if sel.sum() < 30:
                    continue
                v, n1 = cell_r2(Yv, P, alpha_usd, n_p, H, rows=sel)
                e2, _ = cell_r2(Yv, P, alpha_eq, n_p, H, rows=sel)
                s2, n2 = summed_r2(Yv, P, n_p, H, rows=sel)
                if n1:
                    per[key].setdefault(bi, []).append(v)
                    peq[key].setdefault(bi, []).append(e2)
                if n2:
                    leg[key].setdefault(bi, []).append(s2)
            print(f"    {arm:<16}{label:<16}{seed:>5}{pe_:>11.4f}{pd_:>10.4f}"
                  f"{sm:>10.4f}", flush=True)

    ref = list(per)[0]
    print(f"\n    baseline = {ref}   PAIRED over {len(args.seeds)} seeds\n")
    print(f"    {'config':<34}{'perday $R2':>12}{'vs base':>10}{'se':>9}{'t':>8}")
    base = np.asarray(per[ref][-1])
    for key in per:
        v = np.asarray(per[key][-1])
        dd = v - base
        if key == ref or len(dd) < 2:
            print(f"    {key:<34}{v.mean():>12.4f}{dd.mean():>+10.4f}")
            continue
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"    {key:<34}{v.mean():>12.4f}{dd.mean():>+10.4f}{se:>9.4f}"
              f"{(dd.mean()/se if se > 0 else float('inf')):>+8.2f}")

    for title, store in (("PER-DAY EQUAL R2 (metric change alone)", peq),
                         ("PER-DAY DOLLAR R2 (metric + weighting)", per),
                         ("LEGACY SUMMED R2 (guard)", leg)):
        print(f"\n    PER BAND, {title}, paired vs base\n")
        print(f"    {'config':<34}" + "".join(
            f"{f'{lo}-{hi:.0f}' if hi < 1e8 else '2500+':>13}" for lo, hi in BANDS))
        for key in store:
            cells = []
            for bi in range(len(BANDS)):
                x, y = store[key].get(bi), store[ref].get(bi)
                if not x or not y:
                    cells.append(f"{'--':>13}")
                elif key == ref:
                    cells.append(f"{np.mean(x):>13.4f}")
                else:
                    cells.append(f"{np.mean(x) - np.mean(y):>+13.4f}")
            print(f"    {key:<34}" + "".join(cells))
        print(f"    base row is absolute; the rest are deltas in the same band.")

    print(f"\n    H-perday: an arm passes on >= +0.01 per-day dollar R2 in >= 4 "
          f"of {len(BANDS)} bands, without losing more than 0.02 legacy summed "
          f"R2 in any band.")


if __name__ == "__main__":
    main()
