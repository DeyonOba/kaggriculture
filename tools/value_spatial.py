"""V(s) on the SPATIAL trunk: the value head refit so it can see an action.

`tools/value_head.py` reads day-level aggregates -- money, crop and herd
counts, shed, prices -- and is therefore BLIND on 75.8% of one-ply branch
points: a single WATER, HARVEST, MOVE, FEED or CARE changes none of them.
`tools/tau_derive.py` measured that directly, and it is why no override
threshold can be derived from that head: coverage binds before any tau does.

Measured before building this, ceiling first:

    value_head FEATURES identical   91/120 = 75.8%
    spatial TENSOR identical         3/120 =  2.5%
    blind to features, VISIBLE in the tensor  73.3%

So the information is in the representation and the old head simply could
not reach it. This refits V on board_pair_units (181 channels) with the SAME
corpus labels, changing exactly one variable. That isolates the ENCODER fix
from the LABEL fix -- the spec also calls for search outcomes instead of
co-occurrence, and mixing both changes at once would make neither readable.

    PYTHONPATH=. uv run python tools/value_spatial.py build --n 300
    PYTHONPATH=. uv run python tools/value_spatial.py fit
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "rl/datasets/value_spatial.npz")
TENS = os.path.join(REPO, "rl/datasets/value_spatial_T.npy")
MODEL = os.path.join(REPO, "rl/datasets/value_spatial_model.npz")


def job(spec):
    """Sparse turn-level samples from one recorded game, stored float16."""
    path, stride = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.board_tensor as bt
    import tools.price_forecast as pf

    try:
        s = json.load(open(path))
    except Exception:
        return None
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
    T, U, D, S = [], [], [], []
    while not env.done:
        o = env.state[0].observation
        st = o["step"]
        if st % stride == 0 and st > 0:
            # BOTH seats. Same replay cost, and it hands the net the position
            # from each side with exactly opposite labels, so the classes
            # balance by construction. It does NOT double the independent
            # label count -- the two seats of one game are the same game --
            # which is why both carry the same group id and land on the same
            # side of the by-game split.
            for seat in (0, 1):
                ob = env.state[seat].observation
                pv = ob.private
                T.append(bt.board_pair_units(ob, seat, private=pv)
                         .astype("float16"))
                U.append(bt.unit_state(ob, seat, 0, pv).astype("float16"))
                D.append(st // 24)
                S.append(seat)
        env.step([a0(o), a1(env.state[1].observation)])
    if not T:
        return None
    fin = [env.state[i].reward for i in (0, 1)]
    sa = np.asarray(S, "int8")
    y = np.where(sa == 0, float(fin[0] > fin[1]),
                 float(fin[1] > fin[0])).astype("float32")
    m = np.where(sa == 0, fin[0] - fin[1], fin[1] - fin[0]).astype("float32")
    return (np.stack(T), np.stack(U), np.asarray(D, "int16"), y, m)


def build(args):
    """Stream to a MEMMAP, never holding the corpus in RAM.

    The label is per-GAME, so N samples from G games carry G independent
    observations, not N. Games are therefore the quantity to buy, and samples
    per game the one to spend sparingly -- consecutive turns are nearly
    identical anyway. The aggregate head this is measured against was fit on
    3,000 games; matching that label count is what makes the comparison about
    the ENCODER rather than about dataset size.

    36.2 KB per sample makes that impossible to hold in memory, so the tensor
    is written straight to an uncompressed .npy and mmapped at fit time. The
    small arrays still go in the npz.
    """
    from tools.opp_coupling import load_index
    import random
    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    cap = 2 * (760 // args.stride)    # 30 days x 24h x 2 seats
    import tools.board_tensor as bt
    shape = (len(paths) * cap, bt.N_PAIR_U, 10, 10)
    gb = np.prod(shape) * 2 / 1e9
    print(f"replaying {len(paths)} sessions, one sample every {args.stride} "
          f"turns\n  reserving {shape[0]:,} x {shape[1]}x10x10 float16"
          f" = {gb:.2f} GB on disk (mmap, not RAM) ...", flush=True)
    T = np.lib.format.open_memmap(TENS, mode="w+", dtype="float16",
                                  shape=shape)
    U, D, Y, M, G, cur, ng = [], [], [], [], [], 0, 0
    with mp.Pool(args.procs, maxtasksperchild=20) as pool:
        for o in pool.imap_unordered(job, [(p, args.stride) for p in paths]):
            if not o:
                continue
            k = len(o[0])
            T[cur: cur + k] = o[0]
            U.append(o[1]); D.append(o[2]); Y.append(o[3]); M.append(o[4])
            G.append(np.full(k, ng, "int32"))
            cur += k; ng += 1
            if ng % 200 == 0:
                print(f"  {ng}/{len(paths)} games, {cur:,} samples",
                      flush=True)
    T.flush()
    del T
    np.savez_compressed(DATA, U=np.concatenate(U), D=np.concatenate(D),
                        Y=np.concatenate(Y), M=np.concatenate(M),
                        G=np.concatenate(G), n=cur)
    y = np.concatenate(Y)
    print(f"\n{cur:,} samples from {ng} games   "
          f"{os.path.getsize(TENS)/1e9:.2f} GB   win rate {y.mean():.3f}")
    print(f"  {ng} INDEPENDENT labels (one per game), "
          f"{cur/max(ng,1):.0f} correlated samples each")


def build_net(C, nu, width=64, blocks=3):
    """The trunk + value head. Module level so `fit` and inference agree.

    Same stem/ResBlock/mean+amax shape as `boardnet.build`, which cleared its
    own kill criterion on this tensor; ResBlock is nested inside that function
    so it cannot be imported, and a silent copy-drift between train and
    inference is exactly the kind of bug that produces a confident wrong
    number. One definition, used by both.
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(c)
            self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(c)

        def forward(self, x):
            h = torch.relu(self.b1(self.c1(x)))
            return torch.relu(x + self.b2(self.c2(h)))

    class VNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(C, width, 3, padding=1,
                                                bias=False),
                                      nn.BatchNorm2d(width), nn.ReLU())
            self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
            self.head = nn.Sequential(nn.Linear(2 * width + nu, 128), nn.ReLU(),
                                      nn.Linear(128, 1))

        def forward(self, x, u):
            h = self.body(self.stem(x))
            g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
            return self.head(torch.cat([g, u], 1)).squeeze(1)

    return VNet()


_CACHE = {}


def load():
    """The fitted head, cached per process."""
    if "net" in _CACHE:
        return _CACHE["net"]
    import torch
    d = np.load(MODEL)
    meta = ("C", "nu", "width", "blocks")
    net = build_net(*[int(d[k]) for k in meta])
    net.load_state_dict({k: torch.from_numpy(d[k]) for k in d.files
                         if k not in meta})
    net.eval()
    _CACHE["net"] = net
    return net


def value(net, obs, seat, private, unit_idx=0):
    """-> P(this seat finishes with more money). Same encoder as training."""
    import torch
    import tools.board_tensor as bt
    t = bt.board_pair_units(obs, seat, private=private)[None]
    u = bt.unit_state(obs, seat, unit_idx, private)[None]
    with torch.no_grad():
        return float(torch.sigmoid(net(torch.from_numpy(t.astype("float32")),
                                       torch.from_numpy(u.astype("float32")))))


def fit(args):
    import torch
    import torch.nn as nn

    d = np.load(DATA)
    U, Y, G, D = d["U"], d["Y"], d["G"], d["D"]
    # mmap: batches are read from disk, so RAM holds one batch, not 1+ GB.
    # fit() sorts each batch's indices, which keeps these reads near-sequential.
    T = np.load(TENS, mmap_mode="r")[: int(d["n"])]
    gids = np.unique(G)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(gids)
    va = np.isin(G, gids[: max(1, len(gids) // 5)])
    tr = ~va
    print(f"{len(T):,} samples, {len(gids)} games   "
          f"train {tr.sum():,} / val {va.sum():,}  (split BY GAME)")
    print(f"  {len(gids)} INDEPENDENT labels; val carries "
          f"{len(gids) // 5} of them, NOT {va.sum():,}")

    C = T.shape[1]
    nu = U.shape[1]
    net = build_net(C, nu, args.width, args.blocks)
    # weight decay: 345k params against a few thousand game-level labels is
    # the regime the first attempt memorised in six epochs.
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss()
    # keep the corpus in float16 and cast PER BATCH: a full float32 copy of
    # the training tensor is ~2x the dataset and the dataset is already the
    # largest thing here.
    # T stays a MEMMAP. `T[tr]` with a boolean mask would fancy-index and copy
    # the entire split into RAM, which is exactly what the memmap exists to
    # avoid -- so the split is carried as INDEX ARRAYS and only a batch is
    # ever read from disk. U and Y are kilobytes; materialise those.
    tr_ix, va_ix = np.where(tr)[0], np.where(va)[0]
    Utr_, Ytr_ = U[tr_ix], Y[tr_ix]
    Uva_, Yva_ = U[va_ix], Y[va_ix]
    Yva = torch.from_numpy(Yva_)

    def batch(ix, Ub, k):
        """k must be ASCENDING: ix is sorted, so ascending k gives ascending
        file offsets (near-sequential reads) and keeps Ub aligned to it."""
        return (torch.from_numpy(np.asarray(T[ix[k]], dtype="float32")),
                torch.from_numpy(Ub[k].astype("float32")))

    def val_logits():
        outs = []
        for i in range(0, len(va_ix), 512):
            k = np.arange(i, min(i + 512, len(va_ix)))
            xb, ub = batch(va_ix, Uva_, k)
            outs.append(net(xb, ub))
        return torch.cat(outs)

    best, best_state, bad = 1e9, None, 0
    n = len(tr_ix)
    Ytr_t = torch.from_numpy(Ytr_)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(n)
        for i in range(0, n, args.batch):
            j = np.sort(perm[i: i + args.batch])
            xb, ub = batch(tr_ix, Utr_, j)
            opt.zero_grad()
            loss = lossf(net(xb, ub), Ytr_t[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lg = val_logits()
            p = torch.sigmoid(lg)
            vl = float(lossf(lg, Yva))
            acc = float(((p > 0.5).float() == Yva).float().mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad >= args.patience:
            print(f"  ep {ep:>3}  val logloss {vl:.4f}  acc {acc:.3f}",
                  flush=True)
        if bad >= args.patience:
            break
    net.load_state_dict(best_state)
    np.savez_compressed(MODEL, C=C, nu=nu, width=args.width,
                        blocks=args.blocks,
                        **{k: v.numpy() for k, v in net.state_dict().items()})
    # the baseline any value head must beat: predict the base rate
    base = Ytr_.mean()
    bl = -(Yva_ * np.log(base) + (1 - Yva_) * np.log(1 - base)).mean()
    print(f"\n  best val logloss {best:.4f}   base-rate baseline {bl:.4f}"
          f"   -> {100*(bl-best)/bl:+.1f}%")
    print(f"  -> {MODEL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fit"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()
    (build if args.cmd == "build" else fit)(args)


if __name__ == "__main__":
    main()
