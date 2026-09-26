"""Train execnet on the dataset from rl/execnet/dataset.py, split by GAME,
and report held-out accuracy, per-op precision / recall, the agreement of
the model and of our scheduler with the top players, and the confusion
between what the top players did and what our scheduler would have done.

    PYTHONPATH=. uv run python rl/execnet/train.py --data rl/datasets/execnet \\
        --out rl/datasets/execnet/execnet.pt --epochs 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl.execnet import features as F  # noqa: E402
from rl.execnet.model import ExecNet, save  # noqa: E402


def load_data(data_dir, held_frac=0.1, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no npz under {data_dir}")
    rng = np.random.RandomState(seed)
    order = list(range(len(files)))
    rng.shuffle(order)
    n_held = max(1, int(round(held_frac * len(files))))
    held = set(order[:n_held])
    parts = {"train": [], "held": []}
    for i, f in enumerate(files):
        z = np.load(f)
        gid = os.path.splitext(os.path.basename(f))[0]
        rec = {k: z[k] for k in ("tiles", "scal", "mask", "y", "ours", "step", "unit")}
        rec["game"] = np.full(len(rec["y"]), i, dtype=np.int32)
        rec["gid"] = gid
        parts["held" if i in held else "train"].append(rec)
    out = {}
    for k, recs in parts.items():
        out[k] = {key: np.concatenate([r[key] for r in recs]) for key in ("tiles", "scal", "mask", "y", "ours", "step", "unit", "game")}
        out[k]["gids"] = [r["gid"] for r in recs]
    return out


def batches(d, bs, rng=None):
    n = len(d["y"])
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield (torch.from_numpy(d["tiles"][j].astype(np.float32) / 255.0),
               torch.from_numpy(d["scal"][j].astype(np.float32)),
               torch.from_numpy(d["mask"][j].astype(np.bool_)),
               torch.from_numpy(d["y"][j].astype(np.int64)))


@torch.no_grad()
def predict(model, d, bs=8192):
    preds, probs = [], []
    for t, s, m, _ in batches(d, bs):
        logits = model(t, s, m)
        p = torch.softmax(logits, dim=-1)
        preds.append(p.argmax(-1).numpy())
        probs.append(p.max(-1).values.numpy())
    return np.concatenate(preds), np.concatenate(probs)


def per_op_report(y, pred):
    rows = {}
    for i, op in enumerate(F.OPS):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        sup = int((y == i).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        rows[op] = {"support": sup, "precision": prec, "recall": rec}
    return rows


def confusion(a, b, valid):
    """Counts of (a, b) pairs over valid rows, as {op_a: {op_b: n}}."""
    out = {}
    for i, oa in enumerate(F.OPS):
        row = {}
        sel = valid & (a == i)
        for j, ob in enumerate(F.OPS):
            n = int((sel & (b == j)).sum())
            if n:
                row[ob] = n
        if row:
            out[oa] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(REPO, "rl/datasets/execnet"))
    ap.add_argument("--out", default=os.path.join(REPO, "rl/datasets/execnet/execnet.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--held-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()
    d = load_data(args.data, args.held_frac, args.seed)
    tr, he = d["train"], d["held"]
    print(f"train {len(tr['y']):,} samples / {len(tr['gids'])} games; held-out {len(he['y']):,} / {len(he['gids'])} games "
          f"({time.time() - t0:.0f}s)", flush=True)
    counts = np.bincount(tr["y"], minlength=len(F.OPS))
    print("label share (train):", {op: round(float(c) / len(tr["y"]), 3) for op, c in zip(F.OPS, counts) if c})
    model = ExecNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for t, s, m, y in batches(tr, args.batch, rng):
            logits = model(t, s, m)
            loss = Fn.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        model.eval()
        pred, _ = predict(model, he)
        acc = float((pred == he["y"]).mean())
        print(f"epoch {ep + 1}: train nll {tot / n:.3f}  held-out acc {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pred, conf = predict(model, he)
    y = he["y"]
    ours = he["ours"]
    valid = ours >= 0
    report = {
        "n_train": int(len(tr["y"])), "n_held": int(len(y)),
        "games_train": len(tr["gids"]), "games_held": len(he["gids"]), "held_games": he["gids"],
        "label_share_train": {op: float(c) / len(tr["y"]) for op, c in zip(F.OPS, counts)},
        "accuracy_model": float((pred == y).mean()),
        "accuracy_ours_vs_top": float((ours[valid] == y[valid]).mean()) if valid.any() else None,
        "agreement_model_vs_ours": float((pred[valid] == ours[valid]).mean()) if valid.any() else None,
        "accuracy_model_moves_only": float((pred[y < 4] == y[y < 4]).mean()),
        "accuracy_model_chores_only": float((pred[y >= 5] == y[y >= 5]).mean()),
        "accuracy_ours_chores_only": float((ours[valid & (y >= 5)] == y[valid & (y >= 5)]).mean()) if (valid & (y >= 5)).any() else None,
        "per_op_model": per_op_report(y, pred),
        "per_op_ours": per_op_report(y[valid], ours[valid]),
        "confusion_top_vs_ours": confusion(y, ours, valid),
        "confusion_top_vs_model": confusion(y, pred, np.ones_like(valid)),
        "confidence_quantiles": {q: float(np.quantile(conf, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "accuracy_by_confidence": {
            str(th): [float((pred[conf >= th] == y[conf >= th]).mean()) if (conf >= th).any() else None,
                      float((conf >= th).mean())] for th in (0.0, 0.5, 0.7, 0.9)},
    }
    save(model, args.out, extra={"report": {k: v for k, v in report.items() if k not in ("confusion_top_vs_ours", "confusion_top_vs_model")}})
    rep_path = os.path.splitext(args.out)[0] + "_report.json"
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"held-out: model {report['accuracy_model']:.3f}  ours-vs-top {report['accuracy_ours_vs_top']}  "
          f"model-vs-ours {report['agreement_model_vs_ours']}  chores: model {report['accuracy_model_chores_only']:.3f} "
          f"ours {report['accuracy_ours_chores_only']}")
    print("-> ", args.out, rep_path)


if __name__ == "__main__":
    main()
