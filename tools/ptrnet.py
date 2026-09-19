"""A real pointer network for unit->task assignment, and its training loop.

`tools/make_assign_variant.py` tested optimal matching (scipy Hungarian) under
two costs and found the MATCHING was fine and the COST was wrong -- but both
costs were fixed functions. A pointer network differs in kind: it encodes the
whole candidate set, lets every unit attend over every task AND over the other
units, and decodes a permutation autoregressively, so a unit's choice can
depend on what the other units are about to take. Neither a linear ranker nor
Hungarian-on-a-fixed-cost can represent that.

Architecture, sized for <=12 units and <=64 candidate tasks:

    task encoder    per-task features -> d, then SELF-ATTENTION across tasks so
                    a task knows what else is available nearby
    unit encoder    per-unit features -> d
    glimpse         units attend over tasks (multi-head), producing a context
    pointer         a single-head compatibility layer emitting logits over
                    tasks for each unit, masked for already-taken tasks
    decode          greedy over units in order of decreasing max-logit, each
                    pointing at its best remaining task

Trained by imitation on M & M & P & Q's joint assignments -- the same labels
`tools/exec_bc.py` used to reach 69.4% per-unit accuracy, but scored JOINTLY:
cross-entropy over which task each unit took, with the mask updated as units
are assigned, so the model is penalised for two units pointing at one tile.

    PYTHONPATH=. uv run python tools/ptrnet.py --build --n 12
    PYTHONPATH=. uv run python tools/ptrnet.py --train --epochs 60
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "rl/datasets/ptrnet.npz")
MODEL = os.path.join(REPO, "rl/datasets/ptrnet.npz.model")

MAX_U, MAX_T = 12, 64
# Task feature width is whatever tools/exec_bc.FEATS defines (25 today);
# read it from the data rather than hardcoding, so the two models stay
# trained on identical features.
F_U = 6


def build_net(f_t, d=64, heads=4):
    import torch
    import torch.nn as nn

    class PtrNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.te = nn.Sequential(nn.Linear(f_t, d), nn.ReLU(), nn.Linear(d, d))
            self.ue = nn.Sequential(nn.Linear(F_U, d), nn.ReLU(), nn.Linear(d, d))
            self.tsa = nn.MultiheadAttention(d, heads, batch_first=True)
            self.tln = nn.LayerNorm(d)
            self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
            self.gln = nn.LayerNorm(d)
            self.wq = nn.Linear(d, d)
            self.wk = nn.Linear(d, d)
            self.scale = d ** 0.5
            # learned bias on unit-task distance -- without it the
            # shared task encoding cannot tell units apart at all
            self.dbias = nn.Parameter(torch.tensor(-5.0))

        def forward(self, U, T, umask, tmask, D):
            """U (B,MAX_U,F_U)  T (B,MAX_T,F_T)  masks True = VALID."""
            h = self.te(T)
            pad = ~tmask
            sa, _ = self.tsa(h, h, h, key_padding_mask=pad)
            h = self.tln(h + sa)                       # tasks see each other
            u = self.ue(U)
            g, _ = self.glimpse(u, h, h, key_padding_mask=pad)
            u = self.gln(u + g)                        # units see the task set
            logits = (self.wq(u) @ self.wk(h).transpose(1, 2)) / self.scale
            logits = logits + self.dbias * D
            logits = logits.masked_fill(~tmask.unsqueeze(1), -1e9)
            logits = logits.masked_fill(~umask.unsqueeze(2), -1e9)
            return logits                              # (B, MAX_U, MAX_T)

    return PtrNet()


def fit(U, T, UM, TM, Y, G, D, *, epochs=60, d=64, lr=1e-3, batch=64,
        val_frac=0.2, patience=10, seed=0):
    """Y (B,MAX_U) = index of the task each unit took, -1 where none."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gids = np.unique(G)
    rng.shuffle(gids)
    val = set(gids[: max(1, len(gids) // 5)].tolist())
    va = np.array([g in val for g in G])
    tr = ~va

    net = build_net(T.shape[2], d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    def pack(m):
        return (torch.from_numpy(U[m]), torch.from_numpy(T[m]),
                torch.from_numpy(UM[m]), torch.from_numpy(TM[m]),
                torch.from_numpy(D[m]), torch.from_numpy(Y[m]).long())

    Xtr, Xva = pack(tr), pack(va)
    best, best_state, bad = 1e9, None, 0
    n = len(Xtr[0])
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lg = net(Xtr[0][b], Xtr[1][b], Xtr[2][b], Xtr[3][b], Xtr[4][b])
            loss = nn.functional.cross_entropy(
                lg.reshape(-1, MAX_T), Xtr[5][b].reshape(-1), ignore_index=-1)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lg = net(*Xva[:5])
            v = nn.functional.cross_entropy(
                lg.reshape(-1, MAX_T), Xva[5].reshape(-1),
                ignore_index=-1).item()
            pred = lg.argmax(-1)
            sel = Xva[5] >= 0
            acc = float((pred[sel] == Xva[5][sel]).float().mean()) if sel.any() else 0.0
        sched.step(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
            best_acc = acc
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0:
            print(f"      ep {ep:>3}  val loss {v:.4f}  top-1 {100*acc:.1f}%",
                  flush=True)
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"val": best, "acc": best_acc, "d": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d", type=int, default=64)
    args = ap.parse_args()

    if args.build:
        from tools.ptrnet_data import build
        build(args.n)
    if args.train:
        z = np.load(DATA)
        print(f"{len(z['U']):,} turns, "
              f"{int(z['UM'].sum()):,} assigned units\n", flush=True)
        net, st = fit(z["U"], z["T"], z["UM"], z["TM"], z["Y"], z["G"], z["D"],
                      epochs=args.epochs, d=args.d)
        import torch
        torch.save(net.state_dict(), MODEL)
        print(f"\n  best val loss {st['val']:.4f}   joint top-1 "
              f"{100*st['acc']:.1f}%   -> {MODEL}")


if __name__ == "__main__":
    main()
