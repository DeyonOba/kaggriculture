"""Train execnet v2 (destination) on the curated dataset, split by GAME, and
report held-out destination accuracy, STAY / SHED precision-recall, the op
head, and the three-way agreement with OUR scheduler's destination class.

    PYTHONPATH=. uv run python rl/execnet/train2.py --data rl/datasets/execnet2 \
        --out rl/datasets/execnet2/execnet2.pt --epochs 6
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

from rl.execnet import features2 as F2  # noqa: E402
from rl.execnet.model2 import ExecNet2, save  # noqa: E402

_BANDS = {}

SHED_CH0 = 24          # the request planes start here in the stored array
# the conditioning vector (the "conditioned" variant, Maia-2 style): the
# seat's own executor profile, z-scored across the training seats and
# appended to the scalar vector. At play time it is clamped to the elite
# value (the mean profile of the three leaders' seats), which asks the model
# "act like a seat with this profile" on OUR states.
COND_KEYS = ["chores_per_visit", "moves_per_productive", "pickups_per_feed",
             "wheat_units_per_harvest"]
ELITE_SOURCES = ("mmpq", "boey", "dsm")


def _profiles(data_dirs):
    """gid -> (profile vector, band, source) from every meta.json."""
    out = {}
    for dd in data_dirs:
        mp = os.path.join(dd, "meta.json")
        if not os.path.exists(mp):
            continue
        for gid, v in json.load(open(mp)).items():
            pr = v.get("profile") or {}
            vec = [pr.get(k) for k in COND_KEYS]
            if any(x is None for x in vec):
                continue
            out[gid] = (np.asarray(vec, dtype=np.float32), v.get("band"), v.get("source"))
    return out


def load_data(data_dirs, held_frac=0.12, seed=0, max_games=None, cond=None):
    """cond: gid -> z-scored conditioning vector, appended to the scalars."""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    seen, files = set(), []
    for dd in data_dirs:
        for f in sorted(glob.glob(os.path.join(dd, "*.npz"))):
            gid = os.path.splitext(os.path.basename(f))[0]
            if gid in seen:
                continue
            seen.add(gid)
            files.append(f)
    if not files:
        raise SystemExit(f"no npz under {data_dirs}")
    if max_games:
        files = files[:max_games]
    rng = np.random.RandomState(seed)
    order = list(range(len(files)))
    rng.shuffle(order)
    held = set(order[:max(1, int(round(held_frac * len(files))))])
    parts = {"train": {"planes": [], "rows": [], "gids": []},
             "held": {"planes": [], "rows": [], "gids": []}}
    for i, f in enumerate(files):
        z = np.load(f)
        key = "held" if i in held else "train"
        base = sum(p.shape[0] for p in parts[key]["planes"])
        parts[key]["planes"].append(z["planes"])
        r = {k: z[k] for k in ("ux", "uy", "scal", "sidx", "dest", "opc", "ours", "inmask")}
        if cond is not None:
            gid = os.path.splitext(os.path.basename(f))[0]
            c = cond.get(gid)
            if c is None:
                continue
            r["scal"] = np.concatenate(
                [r["scal"].astype(np.float32),
                 np.repeat(c[None, :], len(r["scal"]), axis=0)], axis=1).astype(np.float16)
        r["band"] = np.full(len(r["dest"]), _BANDS.get(os.path.splitext(os.path.basename(f))[0], -1), dtype=np.int8)
        r["sidx"] = r["sidx"].astype(np.int32) + base
        r["game"] = np.full(len(r["dest"]), i, dtype=np.int32)
        parts[key]["rows"].append(r)
        parts[key]["gids"].append(os.path.splitext(os.path.basename(f))[0])
    out = {}
    for key, p in parts.items():
        d = {k: np.concatenate([r[k] for r in p["rows"]]) for k in
             ("ux", "uy", "scal", "sidx", "dest", "opc", "ours", "inmask", "game", "band")}
        d["planes"] = np.concatenate(p["planes"]) if p["planes"] else np.zeros((0, 32, 100), np.uint8)
        d["gids"] = p["gids"]
        keep = d["inmask"] > 0
        d["dropped_share"] = float(1.0 - keep.mean()) if len(keep) else 0.0
        for k in ("ux", "uy", "scal", "sidx", "dest", "opc", "ours", "game", "band"):
            d[k] = d[k][keep]
        out[key] = d
    return out


def make_batch(d, j):
    """-> planes [B,33,10,10] f32, scal [B,31] f32, mask [B,102] bool."""
    p = d["planes"][d["sidx"][j]].astype(np.float32) / 255.0        # [B, 32, 100]
    b = p.shape[0]
    unit = np.zeros((b, 1, F2.N_TILES), dtype=np.float32)
    ti = d["uy"][j].astype(np.int64) * F2.BOARD + d["ux"][j].astype(np.int64)
    unit[np.arange(b), 0, ti] = 1.0
    planes = np.concatenate([p, unit], axis=1).reshape(b, F2.N_PLANE, F2.BOARD, F2.BOARD)
    has_req = (p[:, SHED_CH0:, :] > 0).any(axis=1)                  # [B, 100]
    mask = np.zeros((b, F2.N_DEST), dtype=bool)
    mask[:, :F2.N_TILES] = has_req
    mask[np.arange(b), ti] = False                                  # STAY encodes the own tile
    at_shed = np.isin(ti, [F2.tile_index(x, y) for x, y in F2.SHED_SET])
    mask[:, F2.DEST_SHED] = ~at_shed
    mask[:, F2.DEST_STAY] = True
    return (torch.from_numpy(planes), torch.from_numpy(d["scal"][j].astype(np.float32)),
            torch.from_numpy(mask))


def batches(d, bs, rng=None):
    n = len(d["dest"])
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        planes, scal, mask = make_batch(d, j)
        yield (planes, scal, mask,
               torch.from_numpy(d["dest"][j].astype(np.int64)),
               torch.from_numpy(d["opc"][j].astype(np.int64)),
               torch.from_numpy(d["uy"][j].astype(np.int64) * F2.BOARD + d["ux"][j].astype(np.int64)))


@torch.no_grad()
def predict(model, d, bs=1024):
    preds, confs, opp = [], [], []
    for planes, scal, mask, y, opc, ti in batches(d, bs):
        logits, op_logits = model(planes, scal, mask)
        p = torch.softmax(logits, dim=-1)
        k = p.argmax(-1)
        preds.append(k.numpy())
        confs.append(p.max(-1).values.numpy())
        tgt = torch.where(y < F2.N_TILES, y, ti).clamp(0, F2.N_TILES - 1)
        opp.append(op_logits.gather(2, tgt.view(-1, 1, 1).expand(-1, F2.N_REQ, 1)).squeeze(-1).argmax(-1).numpy())
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(opp)


def _pr(y, pred, cls):
    tp = int(((pred == cls) & (y == cls)).sum()); fp = int(((pred == cls) & (y != cls)).sum())
    fn = int(((pred != cls) & (y == cls)).sum()); sup = int((y == cls).sum())
    return {"support": sup, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--cond", action="store_true",
                    help="conditioned variant: append the seat's z-scored executor profile to the scalars")
    ap.add_argument("--out", default=os.path.join(REPO, "rl/datasets/execnet2/execnet2.pt"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--op-weight", type=float, default=0.3)
    ap.add_argument("--held-frac", type=float, default=0.12)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample this many TRAINING rows (this box has 6 cores and the "
                         "paired eval holds 5 of them)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()
    data_dirs = args.data or [os.path.join(REPO, "rl/datasets/execnet2")]
    profs = _profiles(data_dirs)
    _BANDS.update({g: (b if b is not None else -1) for g, (_, b, _) in profs.items()})
    cond = cond_extra = None
    if args.cond:
        raw = np.stack([v for v, _, _ in profs.values()])
        mean, std = raw.mean(0), raw.std(0) + 1e-6
        elite = np.stack([v for v, _, src in profs.values() if src in ELITE_SOURCES])
        elite_raw = elite.mean(0) if len(elite) else mean
        cond = {g: (v - mean) / std for g, (v, _, _) in profs.items()}
        cond_extra = {"keys": COND_KEYS, "mean": mean.tolist(), "std": std.tolist(),
                      "elite_raw": elite_raw.tolist(),
                      "elite_z": ((elite_raw - mean) / std).tolist(), "n_elite_games": int(len(elite))}
        print(f"conditioning on {len(cond)} seats; elite (n={len(elite)}) raw "
              f"{[round(float(x), 3) for x in elite_raw]} -> z "
              f"{[round(float(x), 2) for x in cond_extra['elite_z']]}", flush=True)
    d = load_data(data_dirs, args.held_frac, args.seed, args.max_games or None, cond)
    tr, he = d["train"], d["held"]
    print(f"train {len(tr['dest']):,} / {len(tr['gids'])} games; held-out {len(he['dest']):,} / "
          f"{len(he['gids'])} games ({time.time() - t0:.0f}s); dropped "
          f"{tr['dropped_share']:.4f} / {he['dropped_share']:.4f} unexpressible", flush=True)
    if args.sample and args.sample < len(tr["dest"]):
        sel = np.random.RandomState(args.seed).choice(len(tr["dest"]), args.sample, replace=False)
        keep_all_rows = {k: tr[k] for k in ("planes", "gids", "dropped_share")}
        tr = {k: (v[sel] if k not in keep_all_rows else v) for k, v in tr.items()}
        print(f"subsampled to {len(tr['dest']):,} training rows", flush=True)
    mix = lambda a: {k: round(float(v) / len(a), 3) for k, v in zip(  # noqa: E731
        ("TILE", "SHED", "STAY"),
        ((a < F2.N_TILES).sum(), (a == F2.DEST_SHED).sum(), (a == F2.DEST_STAY).sum()))}
    print("label mix train", mix(tr["dest"]), "held", mix(he["dest"]), flush=True)
    model = ExecNet2(n_scalar=tr["scal"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for planes, scal, mask, y, opc, ti in batches(tr, args.batch, rng):
            logits, op_logits = model(planes, scal, mask)
            loss = Fn.cross_entropy(logits, y)
            sel = opc >= 0
            if sel.any():
                tgt = torch.where(y < F2.N_TILES, y, ti).clamp(0, F2.N_TILES - 1)[sel]
                ol = op_logits[sel].gather(2, tgt.view(-1, 1, 1).expand(-1, F2.N_REQ, 1)).squeeze(-1)
                loss = loss + args.op_weight * Fn.cross_entropy(ol, opc[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(y); n += len(y)
        model.eval()
        pred, _, _ = predict(model, he)
        print(f"epoch {ep + 1}: loss {tot / n:.3f}  held-out dest acc "
              f"{float((pred == he['dest']).mean()):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pred, conf, oppred = predict(model, he)
    y, ours = he["dest"], he["ours"]
    yx, yy = y % F2.BOARD, y // F2.BOARD
    px, py = pred % F2.BOARD, pred // F2.BOARD
    both_tile = (y < F2.N_TILES) & (pred < F2.N_TILES)
    within1 = np.zeros(len(y), bool)
    within1[both_tile] = (np.abs(yx - px) + np.abs(yy - py))[both_tile] <= 1
    ycls = np.where(y == F2.DEST_STAY, 0, np.where(y == F2.DEST_SHED, 1, 2))
    pcls = np.where(pred == F2.DEST_STAY, 0, np.where(pred == F2.DEST_SHED, 1, 2))
    names = ["STAY", "SHED", "MOVE"]
    opsel = he["opc"] >= 0
    report = {
        "n_train": int(len(tr["dest"])), "n_held": int(len(y)),
        "games_train": len(tr["gids"]), "games_held": len(he["gids"]), "held_games": he["gids"],
        "dropped_unexpressible": {"train": tr["dropped_share"], "held": he["dropped_share"]},
        "label_mix_train": mix(tr["dest"]), "label_mix_held": mix(he["dest"]),
        "dest_accuracy": float((pred == y).mean()),
        "dest_accuracy_within_1_tile": float(((pred == y) | within1).mean()),
        "stay": _pr(y, pred, F2.DEST_STAY), "shed": _pr(y, pred, F2.DEST_SHED),
        "tile_accuracy_on_tile_labels": float((pred[y < F2.N_TILES] == y[y < F2.N_TILES]).mean()),
        "op_head_accuracy": float((oppred[opsel] == he["opc"][opsel]).mean()) if opsel.any() else None,
        "class3_model": {a: {b: int(((ycls == i) & (pcls == j)).sum()) for j, b in enumerate(names)}
                         for i, a in enumerate(names)},
        "class3_ours": {a: {b: int(((ycls == i) & (ours == j)).sum()) for j, b in enumerate(names)}
                        for i, a in enumerate(names)},
        "class3_agreement_model": float((pcls == ycls).mean()),
        "class3_agreement_ours": float((ours == ycls).mean()),
        "accuracy_by_confidence": {str(th): [float((pred[conf >= th] == y[conf >= th]).mean())
                                             if (conf >= th).any() else None, float((conf >= th).mean())]
                                   for th in (0.0, 0.4, 0.6, 0.8)},
    }
    # per-band agreement of the RECORDED action with our scheduler's
    # destination class -- the diagnostic that says whether a broad-band
    # warm-up would mostly re-teach our own habits
    bands = np.concatenate([tr["band"], he["band"]])
    yall = np.concatenate([tr["dest"], he["dest"]])
    oall = np.concatenate([tr["ours"], he["ours"]])
    ycls_all = np.where(yall == F2.DEST_STAY, 0, np.where(yall == F2.DEST_SHED, 1, 2))
    band_rows = {}
    for b in sorted(set(int(x) for x in bands)):
        sel = bands == b
        if not sel.any():
            continue
        band_rows[str(b)] = {"n": int(sel.sum()),
                             "agreement_with_our_scheduler": float((oall[sel] == ycls_all[sel]).mean()),
                             "recorded_stay_share": float((ycls_all[sel] == 0).mean()),
                             "our_stay_share": float((oall[sel] == 0).mean())}
    report["band_agreement"] = band_rows
    report["conditioning"] = cond_extra
    print("band agreement with our scheduler:",
          {k: (v["n"], round(v["agreement_with_our_scheduler"], 3)) for k, v in band_rows.items()}, flush=True)
    save(model, args.out, extra={"cond": cond_extra,
                                 "report": {k: v for k, v in report.items() if not k.startswith("class3_c")}})
    rp = os.path.splitext(args.out)[0] + "_report.json"
    json.dump(report, open(rp, "w"), indent=1)
    print(f"held-out dest {report['dest_accuracy']:.3f} (within 1 tile {report['dest_accuracy_within_1_tile']:.3f}); "
          f"3-way stay/shed/move: model {report['class3_agreement_model']:.3f} vs our scheduler "
          f"{report['class3_agreement_ours']:.3f}; op head {report['op_head_accuracy']}")
    print("->", args.out, rp)


if __name__ == "__main__":
    main()
