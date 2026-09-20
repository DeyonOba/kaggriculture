"""Where does the spatial value head actually work? By DAY, on held-out games.

A pooled +15.4% over the base rate can be earned entirely on day 28, where
the winner is nearly determined and any reader of the bank is right. That
number would be useless for our purpose: branch points live in days 2-26,
and the recorded finding is that the relative money lead predicts NOTHING
before ~day 16 (r = -0.021 at day 10) while investment does (+0.304).

So the pooled figure is not the read. The read is the CURVE, plus the day
at which it first clears the base rate.

    PYTHONPATH=. uv run python tools/value_spatial_eval.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    import torch
    import tools.value_spatial as vs

    d = np.load(vs.DATA)
    U, Y, G, D = d["U"], d["Y"], d["G"], d["D"]
    T = np.load(vs.TENS, mmap_mode="r")[: int(d["n"])]

    # reproduce fit()'s split EXACTLY -- same seed, same rng call order
    gids = np.unique(G)
    rng = np.random.default_rng(11)
    rng.shuffle(gids)
    va = np.isin(G, gids[: max(1, len(gids) // 5)])
    va_ix = np.where(va)[0]
    print(f"held-out: {len(va_ix):,} samples from "
          f"{len(np.unique(G[va_ix]))} games\n")

    net = vs.load()
    p = []
    with torch.no_grad():
        for i in range(0, len(va_ix), 512):
            k = va_ix[i: i + 512]
            x = torch.from_numpy(np.asarray(T[k], dtype="float32"))
            u = torch.from_numpy(U[k].astype("float32"))
            p.append(torch.sigmoid(net(x, u)).numpy())
    p = np.concatenate(p)
    y, day = Y[va_ix], D[va_ix]

    def ll(pp, yy):
        pp = np.clip(pp, 1e-6, 1 - 1e-6)
        return float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())

    base = float(Y[~va].mean())
    print(f"  pooled   logloss {ll(p, y):.4f}   base {ll(np.full_like(p, base), y):.4f}"
          f"   acc {((p > 0.5) == (y > 0.5)).mean():.3f}\n")

    print(f"  {'days':>8}{'n':>8}{'acc':>8}{'logloss':>10}{'base':>9}"
          f"{'gain':>8}")
    first = None
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
                   (24, 31)):
        m = (day >= lo) & (day < hi)
        if m.sum() < 50:
            continue
        b = ll(np.full(int(m.sum()), base), y[m])
        l = ll(p[m], y[m])
        g = 100 * (b - l) / b
        if first is None and g > 2:
            first = lo
        print(f"  {str(lo)+'-'+str(hi):>8}{m.sum():>8,}"
              f"{((p[m] > 0.5) == (y[m] > 0.5)).mean():>8.3f}"
              f"{l:>10.4f}{b:>9.4f}{g:>7.1f}%")

    print(f"\n  first day-bin clearing the base rate by >2%: "
          f"{'day ' + str(first) if first is not None else 'NONE'}")

    # calibration: a head used for ORDERING must at least be monotone
    print(f"\n  calibration (is a higher score actually a higher win rate?)\n")
    print(f"  {'predicted':>12}{'n':>8}{'actual':>9}")
    qs = np.quantile(p, np.linspace(0, 1, 9))
    mono = True
    prev = -1
    for i in range(len(qs) - 1):
        m = (p >= qs[i]) & (p <= qs[i + 1] if i == len(qs) - 2 else p < qs[i + 1])
        if m.sum() < 30:
            continue
        a = float(y[m].mean())
        print(f"  {p[m].mean():>12.3f}{m.sum():>8,}{a:>9.3f}")
        if a < prev - 0.02:
            mono = False
        prev = a
    print(f"\n  monotone: {mono}")


if __name__ == "__main__":
    main()
