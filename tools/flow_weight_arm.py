"""Weight the flow loss in DOLLARS, because the current one has no money in it.

THE MISALIGNMENT. `tools/boardnet.fit` z-scores the target per column and then
takes an unweighted MSE over all 90 outputs (10 days x 9 products). So every
(day, product) cell contributes equally **in units of its own standard
deviation**. Measured in this corpus:

    WHEAT       4,481,777 gross sells    base $25
    CARROT        138,841               base $35

-- a 32x difference in volume, and after per-column z-scoring the two count
the same. The scoring does the same thing again: R^2 per product on the 10-day
sum, then an unweighted mean over products.

The forecast exists to price a planting decision. It feeds the opponent-supply
term inside `_rent_inventory`, which converts predicted supply into a price.
An error there costs money in proportion to the DOLLARS of flow it misplaces,
and the objective currently cannot tell a product worth $4.5M of flow from one
worth $139k.

THE FIX. Y is z-scored, so a residual in z-space maps to dollars as

    dollar residual = (pred_z - y_z) * sd_col * base_price(product)

Passing `col_w = sd_col * base_price` to `fit` therefore makes the loss an MSE
on dollar-denominated flow error. Normalised to mean square 1 so the overall
loss scale, and hence the learning rate, is unchanged.

MEASURING IT WITHOUT RIGGING IT. Scoring a dollar-trained arm on a
dollar-weighted metric is circular -- it wins by construction. So both metrics
are reported for both arms:

    equal-weight R^2    the incumbent metric, unweighted mean over products
    dollar-weighted R^2 the metric that matches what the planner spends

and the pre-registered claim is on the dollar metric, which is the one the
money depends on, with the equal metric reported so a collapse there is
visible rather than hidden.

    H-dollar: training with the dollar-weighted loss improves dollar-weighted
    held-out R^2 by >= +0.01 in >= 4 of 5 rating bands against the
    equal-weight loss, on identical channels and seeds, without losing more
    than 0.02 equal-weight R^2 in any band.

    PYTHONPATH=. uv run python tools/flow_weight_arm.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def product_r2(Yv, P, n_p, H, rows=None, weights=None):
    """(mean R^2 over products that vary, n_live). `weights` is per product."""
    num, den, live = 0.0, 0.0, 0
    for pi in range(n_p):
        c = [j * n_p + pi for j in range(H)]
        y, q = Yv[:, c].sum(1), P[:, c].sum(1)
        if rows is not None:
            y, q = y[rows], q[rows]
        d = float(((y - y.mean()) ** 2).sum())
        if d <= 1e-6 * max(1.0, float((y ** 2).sum())) or len(y) < 2:
            continue
        w = 1.0 if weights is None else float(weights[pi])
        num += w * (1 - float(((y - q) ** 2).sum()) / d)
        den += w
        live += 1
    return (num / den if den > 0 else float("nan")), live


def main() -> None:
    from kagent.constants import MARKET_PARAMS
    from tools.board_tensor import arm_channels
    from tools.boardnet import fit, predict
    from tools.s3_flow import BANDS, PRODUCTS, H

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(
        REPO, "rl", "datasets", "s3_flow_v2band.npz"))
    ap.add_argument("--arm", default="E_full_83")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    d = np.load(args.data)
    T, F, A, G, R = d["T"], d["F"], d["A"], d["G"].astype(int), d["R"]
    S = d["S"] if "S" in d.files else None
    n_p = len(PRODUCTS)
    cols = arm_channels(args.arm)
    Ti = np.ascontiguousarray(T[:, cols].astype("float32"))

    price = np.asarray([MARKET_PARAMS[p]["base"] for p in PRODUCTS], "float64")
    # sd of each of the 90 columns, over ALL rows -- the split is per seed, so
    # a single global scale keeps the weight identical across arms and seeds.
    sd_col = np.maximum(F.std(0), 1e-3)
    w = sd_col * np.repeat(price[None, :], H, axis=0).reshape(-1)
    w = (w / np.sqrt((w ** 2).mean())).astype("float32")     # mean square 1
    # per-product weight for the dollar-weighted SCORE: dollars of 10-day flow
    pw = price * np.asarray([F[:, [j * n_p + pi for j in range(H)]].sum(1).std()
                             for pi in range(n_p)])
    pw = pw / pw.sum()

    print(f"    arm {args.arm}, {len(Ti):,} rows, {len(np.unique(G))} games")
    print(f"\n    {'product':<12}{'base $':>8}{'sd(10d flow)':>14}"
          f"{'score weight':>14}")
    for pi, p in enumerate(PRODUCTS):
        sdp = F[:, [j * n_p + pi for j in range(H)]].sum(1).std()
        print(f"    {p:<12}{price[pi]:>8.0f}{sdp:>14,.0f}{pw[pi]:>14.3f}")
    print(f"\n    loss weight range {w.min():.3f}-{w.max():.3f} "
          f"(mean square 1)\n")
    print(f"    {'loss':<16}{'seed':>6}{'equal R2':>11}{'dollar R2':>12}")

    acc: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for label, cw in (("equal (current)", None), ("dollars", w)):
        for seed in args.seeds:
            net, stats, _ = fit(Ti, F, G, A=A, S=None, epochs=args.epochs,
                                seed=seed, col_w=cw, verbose=False)
            gids = np.unique(G)
            rng = np.random.default_rng(seed)
            rng.shuffle(gids)
            va = np.isin(G, gids[:max(1, len(gids) // 5)])
            P, Yv, band = predict(net, stats, Ti[va]), F[va], R[va]
            eq, _ = product_r2(Yv, P, n_p, H)
            do, _ = product_r2(Yv, P, n_p, H, weights=pw)
            acc.setdefault(label, {}).setdefault("equal", {}).setdefault(-1, []).append(eq)
            acc[label].setdefault("dollar", {}).setdefault(-1, []).append(do)
            for bi, (lo, hi) in enumerate(BANDS):
                sel = (band >= lo) & (band < hi)
                if sel.sum() < 30:
                    continue
                e, n1 = product_r2(Yv, P, n_p, H, rows=sel)
                v, n2 = product_r2(Yv, P, n_p, H, rows=sel, weights=pw)
                if n1:
                    acc[label]["equal"].setdefault(bi, []).append(e)
                if n2:
                    acc[label]["dollar"].setdefault(bi, []).append(v)
            print(f"    {label:<16}{seed:>6}{eq:>11.4f}{do:>12.4f}", flush=True)

    ref, alt = "equal (current)", "dollars"
    print(f"\n    PAIRED over {len(args.seeds)} seeds, {alt} vs {ref}\n")
    for metric in ("dollar", "equal"):
        a = np.asarray(acc[alt][metric][-1]) - np.asarray(acc[ref][metric][-1])
        if len(a) < 2:
            print(f"    pooled {metric:<8} R2 delta {a.mean():+8.4f}   "
                  f"(one seed: no se)")
            continue
        se = a.std(ddof=1) / np.sqrt(len(a))
        print(f"    pooled {metric:<8} R2 delta {a.mean():+8.4f}   se {se:.4f}"
              f"   t {(a.mean()/se if se > 0 else float('inf')):+6.2f}")

    for metric, bar in (("dollar", 0.01), ("equal", -0.02)):
        print(f"\n    PER BAND, {metric}-weighted R2: {alt} minus {ref}\n")
        print("    " + "".join(
            f"{f'{lo}-{hi:.0f}' if hi < 1e8 else '2500+':>13}" for lo, hi in BANDS))
        cells, hits = [], 0
        for bi in range(len(BANDS)):
            x, y = acc[alt][metric].get(bi), acc[ref][metric].get(bi)
            if not x or not y:
                cells.append(f"{'--':>13}"); continue
            dd = np.mean(x) - np.mean(y)
            if metric == "dollar" and dd >= bar:
                hits += 1
            if metric == "equal" and dd <= bar:
                hits += 1
            cells.append(f"{dd:>+13.4f}")
        print("    " + "".join(cells))
        if metric == "dollar":
            print(f"\n    H-dollar: >= +0.01 in >= 4 of {len(BANDS)} bands "
                  f"-> {hits}/{len(BANDS)}  "
                  f"{'PASS' if hits >= 4 else 'FAIL'}")
        else:
            print(f"\n    guard: no band may lose more than 0.02 equal-weight "
                  f"R2 -> {hits} band(s) breach  "
                  f"{'FAIL' if hits else 'ok'}")


if __name__ == "__main__":
    main()
