"""BoardNet: the opponent-supply model, designed rather than improvised.

S2's network was a deliberate throwaway -- three convs and a global average
pool, defined inside a cross-validation loop and discarded. It answered one
question (does spatial structure beat plain counts: yes, +0.116/+0.137 R2) and
was the right instrument for that, because a win with an unremarkable net
cannot be confounded with a clever net.

This is the model S3 actually needs, and five things change.

1. AVG POOL ALONE THROWS AWAY THE POINT OF A GRID. Reducing 10x10 to a mean
   tells the head HOW MUCH and never WHERE. We concatenate average and max
   pooling: the mean carries mass (how many strawberries), the max carries
   existence and extremes (is there a tile about to die). Both matter and
   neither implies the other.

2. RESIDUAL BLOCKS WITH NORMALISATION. The input mixes one-hot masks, ratios
   and broadcast globals on wildly different scales; BatchNorm after the stem
   makes the depth trainable at all. Three blocks is deep for a 10x10 board --
   receptive field after the stem plus three 3x3 blocks is 9x9, which already
   spans the farm.

3. A PER-TILE AUXILIARY HEAD. The label S2 used was 8 scalars, so the grid was
   used on the input side only. A 1x1 conv head predicting per-tile "this tile
   yields within H days" gives the encoder dense spatial supervision -- 100
   gradients per sample instead of 8 -- and costs one conv.

4. TARGETS Z-SCORED PER OUTPUT. S2's first run reported a KILL that was purely
   loss imbalance: unnormalised joint MSE let high-variance crop columns starve
   the animal heads. That is now structural, not a patch.

5. IT SAVES. Weights, normalisation stats and the channel count go to one npz
   so an agent can load it. S2 saved nothing.

    from tools.boardnet import BoardNet, fit, save, load
"""

import numpy as np


def _torch():
    import torch
    return torch


class _Res:
    pass


def build(in_ch, n_out, width=64, blocks=3, aux=True):
    """-> nn.Module mapping (B, in_ch, 10, 10) to (flow (B,n_out), aux (B,1,10,10))."""
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
            h = self.b2(self.c2(h))
            return torch.relu(x + h)

    class BoardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU())
            self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
            # avg carries mass, max carries existence -- keep both
            self.head = nn.Sequential(
                nn.Linear(2 * width, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, n_out))
            self.aux = nn.Conv2d(width, 1, 1) if aux else None

        def forward(self, x):
            h = self.blocks(self.stem(x))
            g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
            a = self.aux(h) if self.aux is not None else None
            return self.head(g), a

    return BoardNet()


def fit(T, Y, G, A=None, *, width=64, blocks=3, epochs=60, lr=2e-3,
        batch=256, val_frac=0.2, patience=8, seed=0, aux_w=0.3, verbose=True):
    """Train on games outside the validation split, early-stopped on it.

    Splitting BY GAME, not by sample: consecutive days of one game are nearly
    the same board, so a sample-wise split leaks and every number becomes a
    fantasy. Returns (model, stats, history).
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    gids = np.unique(G)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * val_frac))
    val_g = set(gids[:n_val].tolist())
    va = np.array([g in val_g for g in G])
    tr = ~va

    mu, sd = Y[tr].mean(0), np.maximum(Y[tr].std(0), 1e-3)
    Yz = ((Y - mu) / sd).astype("float32")

    net = build(T.shape[1], Y.shape[1], width, blocks, aux=A is not None)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=3)
    Xtr, Ytr = torch.from_numpy(T[tr]), torch.from_numpy(Yz[tr])
    Xva, Yva = torch.from_numpy(T[va]), torch.from_numpy(Yz[va])
    Atr = torch.from_numpy(A[tr]) if A is not None else None

    best, best_state, bad, hist = float("inf"), None, 0, []
    n = len(Xtr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            out, a = net(Xtr[b])
            loss = nn.functional.mse_loss(out, Ytr[b])
            if Atr is not None:
                loss = loss + aux_w * nn.functional.binary_cross_entropy_with_logits(
                    a.squeeze(1), Atr[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(Xva)[0], Yva).item()
        sched.step(v)
        hist.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"      early stop at epoch {ep} (best val {best:.4f})")
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"mu": mu, "sd": sd, "in_ch": T.shape[1], "n_out": Y.shape[1],
                 "width": width, "blocks": blocks, "val_mse": best}, hist


def predict(net, stats, T, batch=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(T), batch):
            out.append(net(torch.from_numpy(T[i:i + batch]))[0].numpy())
    return np.concatenate(out) * stats["sd"] + stats["mu"]


def save(path, net, stats):
    import torch
    blob = {k: v.numpy() for k, v in net.state_dict().items()}
    blob["__mu"] = stats["mu"]
    blob["__sd"] = stats["sd"]
    blob["__meta"] = np.array([stats["in_ch"], stats["n_out"],
                               stats["width"], stats["blocks"]], dtype="int64")
    np.savez_compressed(path, **blob)


def load(path):
    import torch
    d = np.load(path)
    in_ch, n_out, width, blocks = [int(x) for x in d["__meta"]]
    net = build(in_ch, n_out, width, blocks, aux=any(
        k.startswith("aux.") for k in d.files))
    sd = {k: torch.from_numpy(d[k]) for k in d.files if not k.startswith("__")}
    net.load_state_dict(sd)
    net.eval()
    return net, {"mu": d["__mu"], "sd": d["__sd"], "in_ch": in_ch,
                 "n_out": n_out, "width": width, "blocks": blocks}
