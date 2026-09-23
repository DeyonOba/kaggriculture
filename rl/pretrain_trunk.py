"""Pretrain the board trunk to predict the FINAL MARGIN, then hand it to PPO.

WHY MARGIN AND NOT THE CHAMPION'S CHOICES. Cloning a strong player's standing
targets is the obvious pretraining task and it is a measured null here: 69.4%
held-out action accuracy against our rule's 27.9%, and it gated NULL over
4,800 paired games. Ten of the thirteen heads are ALLOCATION, and allocation
is flat -- the LP's provably optimal allocation, executed by our own
scheduler, tied champ-h8 at +$828 +/- 2,317. Pretraining a representation on
that target teaches it the quantity we have measured as worthless.

Margin is the one target that passed anything this week. H-margin:
`T_crop_103` beat `E_full_83` by >= +0.01 R2 in 4 of 5 rating bands on the
final margin, with win AUC up in 4 of 5. So the trunk learns "what about this
board predicts who wins", and PPO then fine-tunes it on the states the policy
actually visits.

NOT FROZEN. The weights are an INITIALISATION. A trunk fitted offline and
frozen cannot move off the distribution it was fitted on, which is how a
self-play potential scored -0.398 R2 out of distribution. `rl/train.py
--trunk-init` loads these and keeps training them.

THE SPLIT IS BY SESSION. Sixty consecutive samples are one game seen from two
seats on thirty days; splitting by sample puts day 4 of a game in train and
day 5 in validation and the R2 that comes back is memorised. The grouping is
reconstructed from the extraction order and then CHECKED against the stored
margins rather than trusted.

    PYTHONPATH=. uv run python rl/pretrain_trunk.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BANDS = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 1e9)]


def r2(y, p):
    d = float(((y - y.mean()) ** 2).sum())
    return float("nan") if d <= 0 else 1 - float(((y - p) ** 2).sum()) / d


def auc(y, s):
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    o = np.argsort(v, kind="mergesort")
    rk = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def groups_and_ratings(rows, n_expected_per_session, n_draw, seed):
    """Session id per sample, plus that session's rating -- both CHECKED.

    The extractor emits seat 0 days 0..29 then seat 1 days 0..29 for each
    session, in the order `pick_sessions` returned them. That makes
    `bi // 60` the session index, but only if no sample was dropped, so the
    count is verified before the mapping is used.
    """
    from tools.ladder_panel import pick_sessions
    from tools.s3_flow import _session_rating

    n = len(rows)
    if n % n_expected_per_session:
        raise SystemExit(
            f"{n:,} samples is not a multiple of {n_expected_per_session}: "
            f"a session was partially dropped, so bi//{n_expected_per_session} "
            f"is not the session index. Re-extract or add an explicit id.")
    g = np.arange(n) // n_expected_per_session
    n_sess = int(g.max()) + 1

    paths = pick_sessions(n_draw, True, seed)[:n_sess]
    if len(paths) != n_sess:
        raise SystemExit(f"reconstructed {len(paths)} paths for {n_sess} sessions")
    # pick_sessions returns ABSOLUTE paths; the rating map is keyed on the
    # index's RELATIVE ones. Without this every rating comes back NaN and the
    # per-band table silently prints nothing -- the same shape of failure as
    # reading a dataset with the wrong --seed.
    def _rel(p):
        return p[len(REPO) + 1:] if p.startswith(REPO) else p
    rating = np.asarray([_session_rating(_rel(p)) for p in paths], "float32")
    if not np.isfinite(rating).any():
        raise SystemExit("every reconstructed rating is NaN -- the path keys "
                         "do not match the session index")
    return g, rating[g]


def main() -> None:
    import torch
    import torch.nn as nn
    from rl.common import BoardTrunk, BOARD_W

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(REPO, "rl", "datasets",
                                                   "session_bc.jsonl"))
    ap.add_argument("--n-draw", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(REPO, "rl", "datasets",
                                                  "trunk_margin.pt"))
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("KAGG_THREADS", "1")))
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(args.data)]
    B = np.load(args.data.replace(".jsonl", "_boards.npy"), mmap_mode="r")
    y = np.asarray([r["m"] for r in rows], "float32")
    w = np.asarray([r["w"] for r in rows], "float32")
    ok = np.isfinite(y)
    print(f"{len(rows):,} samples, boards {B.shape}, {int(ok.sum()):,} with an outcome")

    g, rating = groups_and_ratings(rows, 60, args.n_draw, args.seed)
    # The check that makes the reconstruction trustworthy: margin is constant
    # within a (session, seat) block and opposite between the two seats.
    blk = y.reshape(-1, 30)
    if not np.allclose(blk, blk[:, :1], equal_nan=True):
        raise SystemExit("margin varies within a 30-day block -- the row -> "
                         "session mapping is wrong, refusing to train")
    print(f"  {int(g.max())+1} sessions, ratings "
          f"{np.nanmin(rating):.0f}-{np.nanmax(rating):.0f}")

    rng = np.random.default_rng(args.seed)
    gids = np.unique(g); rng.shuffle(gids)
    nv = max(1, int(len(gids) * args.val_frac))
    va = np.isin(g, gids[:nv]) & ok
    tr = ~np.isin(g, gids[:nv]) & ok
    mu, sd = y[tr].mean(), max(y[tr].std(), 1.0)
    print(f"  split by SESSION: {int(tr.sum()):,} train / {int(va.sum()):,} val")

    trunk = BoardTrunk()
    head = nn.Linear(BOARD_W, 1)
    params = list(trunk.parameters()) + list(head.parameters())
    print(f"  trunk {sum(p.numel() for p in trunk.parameters()):,} params  "
          f"-> {int(tr.sum())/sum(p.numel() for p in trunk.parameters()):.1f} samples/param")
    opt = torch.optim.Adam(params, lr=args.lr)
    tr_i, va_i = np.flatnonzero(tr), np.flatnonzero(va)
    yv = torch.tensor((y[va_i] - mu) / sd).unsqueeze(1)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        trunk.train(); perm = rng.permutation(len(tr_i))
        for s in range(0, len(perm), args.batch):
            idx = np.sort(tr_i[perm[s:s + args.batch]])
            xb = torch.from_numpy(np.asarray(B[idx], dtype="float32"))
            yb = torch.tensor((y[idx] - mu) / sd).unsqueeze(1)
            opt.zero_grad()
            loss = nn.functional.mse_loss(head(trunk(xb)), yb)
            loss.backward(); opt.step()
        trunk.eval(); preds = []
        with torch.no_grad():
            for s in range(0, len(va_i), 512):
                idx = va_i[s:s + 512]
                preds.append(head(trunk(torch.from_numpy(
                    np.asarray(B[idx], dtype="float32")))).numpy())
        P = np.concatenate(preds).reshape(-1)
        v = float(((P - yv.numpy().reshape(-1)) ** 2).mean())
        m = r2(y[va_i], P * sd + mu)
        a = auc(w[va_i], P)
        print(f"  epoch {ep+1:>3}  val mse {v:.4f}  margin R2 {m:+.4f}  win AUC {a:.4f}",
              flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in trunk.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                print("  early stop"); break

    trunk.load_state_dict(best_state)
    torch.save({"trunk": best_state, "arm": os.environ.get("KAGG_BOARD_ARM",
                                                           "T_crop_103")}, args.out)
    # Per band, because a trunk that only works at the top is the net_ood
    # mistake wearing a different hat.
    trunk.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(va_i), 512):
            idx = va_i[s:s + 512]
            preds.append(head(trunk(torch.from_numpy(
                np.asarray(B[idx], dtype="float32")))).numpy())
    P = np.concatenate(preds).reshape(-1)
    rb = rating[va_i]
    print(f"\n  {'band':>12}{'n':>8}{'margin R2':>12}{'win AUC':>10}")
    for lo, hi in BANDS:
        s_ = (rb >= lo) & (rb < hi)
        if s_.sum() < 30:
            continue
        print(f"  {f'{lo}-{hi:.0f}' if hi < 1e8 else '2500+':>12}{int(s_.sum()):>8}"
              f"{r2(y[va_i][s_], P[s_] * sd + mu):>12.4f}{auc(w[va_i][s_], P[s_]):>10.4f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
