"""A CNN over the board that predicts, per tile, which crops are about to die.

WHY A PREDICTOR AND NOT A POLICY. Raw-action behaviour cloning reached 75.5%
per-action accuracy and scored $35, because 719 steps of closed-loop control
compound small errors and there is no correction path -- DAgger's fix needs a
queryable expert and ours are recordings. A per-tile PREDICTOR has no loop: one
board in, one map out, no accumulation. It is the part of the idea that the
compounding failure does not touch.

WHY PER-TILE AND NOT A SCALAR. "How many crops will die" is a count any model
can fit. "WHICH tiles will die" is a map, and a map is actionable: it is the
input a triage rule, a planting rule, or a labour allocator would need. It is
also the first thing we have ever asked that requires knowing where anything
is -- adjacency, clustering, distance to the shed.

TARGET. For each tile holding a crop at the start of day d, will that crop
become a WEED within the next 3 days? That is the engine's
consecutive_unwatered >= 2 rule, and it is mechanical: a fact about the
trajectory, not a model's opinion.

HOW IT WILL BE JUDGED. Not by accuracy. A classifier that predicts "no death"
everywhere scores ~90% here because deaths are rare, and five fitted models
this project trusted on held-out accuracy went on to fail at the only thing
that matters. The honest measures are PR-AUC against the base rate, and
recall at a precision a rule could actually act on.

    PYTHONPATH=. uv run python tools/board_cnn.py build --n 40 --procs 5
    PYTHONPATH=. uv run python tools/board_cnn.py fit --epochs 25
"""

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

DATA = os.path.join(REPO, "rl", "datasets", "boardcnn")
HORIZON = 3


def _job(path):
    import numpy as np
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    from tools.board_tensor import board_tensor, scalars
    try:
        s = json.load(open(path))
    except Exception:
        return None
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    try:
        env.run([make_script_agent(s["actions"][0]),
                 make_script_agent(s["actions"][1])])
    except Exception:
        return None

    X, S, Y, M = [], [], [], []
    for seat in (0, 1):
        # one snapshot per day at hour 0, plus the tile states that follow
        day_state = {}
        for stp in env.steps[:-1]:
            o = stp[seat]["observation"] if isinstance(stp[seat], dict) \
                else stp[seat].observation
            d = o["day"]
            if o["hour"] == 0 and d not in day_state:
                day_state[d] = o
        days = sorted(day_state)
        grids = {}
        for d in days:
            tiles = day_state[d]["farms"][seat]["tiles"]
            grids[d] = [[(t.get("kind") if isinstance(t, dict) else None)
                         for t in row] for row in tiles]
        for d in days:
            if d + HORIZON > max(days):
                continue
            o = day_state[d]
            tiles = o["farms"][seat]["tiles"]
            n = len(tiles)
            y = np.zeros((n, n), dtype="float32")
            m = np.zeros((n, n), dtype="float32")
            for yy in range(n):
                for xx in range(n):
                    t = tiles[yy][xx]
                    if not (isinstance(t, dict) and t.get("kind") == "PLANT"):
                        continue
                    m[yy, xx] = 1.0        # a crop stands here: a real question
                    for k in range(1, HORIZON + 1):
                        g = grids.get(d + k)
                        if g and g[yy][xx] == "WEED":
                            y[yy, xx] = 1.0
                            break
            if m.sum() == 0:
                continue
            X.append(board_tensor(o, seat))
            S.append(scalars(o, seat))
            Y.append(y)
            M.append(m)
    return X, S, Y, M


def build(args):
    import numpy as np
    from tools.raw_bc import cluster_episodes
    eps = cluster_episodes(3029.3)
    files = [f for f in sorted(glob.glob(os.path.join(
        REPO, "rl", "datasets", "sessions", "session-*.json")))
        if int(f.split("-")[-1][:-5]) in eps][:args.n]
    print(f"{len(files)} cluster sessions -> board tensors", flush=True)
    X, S, Y, M = [], [], [], []
    with Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, files)):
            if r:
                X += r[0]; S += r[1]; Y += r[2]; M += r[3]
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(files)}  {len(X):,} board-days", flush=True)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "boards.npz")
    np.savez_compressed(out, X=np.array(X), S=np.array(S),
                        Y=np.array(Y), M=np.array(M))
    tot = float(np.array(M).sum())
    pos = float((np.array(Y) * np.array(M)).sum())
    print(f"\nwrote {out}")
    print(f"  {len(X):,} board-days, {tot:,.0f} crop tiles, {pos:,.0f} die "
          f"within {HORIZON} days = base rate {100*pos/max(1,tot):.1f}%")


def fit(args):
    import numpy as np, torch, torch.nn as nn
    d = np.load(os.path.join(DATA, "boards.npz"))
    X = torch.tensor(d["X"]); S = torch.tensor(d["S"])
    Y = torch.tensor(d["Y"]); M = torch.tensor(d["M"])
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    X, S, Y, M = X[perm], S[perm], Y[perm], M[perm]
    cut = int(len(X) * 0.85)
    C, n = X.shape[1], X.shape[2]
    base = float((Y[cut:] * M[cut:]).sum() / M[cut:].sum())
    print(f"{len(X):,} board-days, {C} channels, {n}x{n}, "
          f"held-out base rate {100*base:.1f}%")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            h = args.width
            # scalars are broadcast as extra planes so the convolution sees
            # day/prices at every cell
            self.body = nn.Sequential(
                nn.Conv2d(C + S.shape[1], h, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
                nn.Conv2d(h, 1, 1))

        def forward(self, x, s):
            p = s[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            return self.body(torch.cat([x, p], 1))[:, 0]

    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    pw = torch.tensor((1 - base) / max(base, 1e-6))
    lf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    def evaluate():
        net.eval()
        with torch.no_grad():
            lg = net(X[cut:], S[cut:])
            m = M[cut:] > 0.5
            p = torch.sigmoid(lg)[m].numpy()
            t = Y[cut:][m].numpy()
        order = p.argsort()[::-1]
        t_s = t[order]
        tp = t_s.cumsum(); k = np.arange(1, len(t_s) + 1)
        prec, rec = tp / k, tp / max(1.0, t_s.sum())
        ap = float((prec * t_s).sum() / max(1.0, t_s.sum()))
        # recall at the precision a rule could act on
        out = {}
        for want in (0.5, 0.7):
            ok = prec >= want
            out[want] = float(rec[ok].max()) if ok.any() else 0.0
        return ap, out, float(t.mean())

    for ep in range(args.epochs):
        net.train()
        for i in range(0, cut, 64):
            opt.zero_grad()
            lg = net(X[i:i+64], S[i:i+64])
            loss = (lf(lg, Y[i:i+64]) * M[i:i+64]).sum() / M[i:i+64].sum().clamp(min=1)
            loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            ap, rec, br = evaluate()
            print(f"  epoch {ep+1:>3}  PR-AUC {ap:.3f}  (base rate {br:.3f}, "
                  f"lift {ap/max(br,1e-9):.1f}x)   recall@prec.5 {rec[0.5]:.2f}  "
                  f"@prec.7 {rec[0.7]:.2f}", flush=True)

    os.makedirs(DATA, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(DATA, "cnn.pt"))
    print(f"\nwrote {os.path.join(DATA, 'cnn.pt')}")
    print("  PR-AUC vs base rate is the number that matters -- accuracy is "
          "meaningless\n  when ~90% of tiles do not die.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fit"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--width", type=int, default=48)
    args = ap.parse_args()
    (build if args.mode == "build" else fit)(args)


if __name__ == "__main__":
    main()
