"""Score cloned readouts in TILES, so head sets of different widths compare.

    PYTHONPATH=. uv run python tools/bc_tile_error.py \
        --corpus rl/runs/headfix/bc_v1 --version 1 --weights rl/runs/headfix/readout_v1/bc.pt \
        --corpus rl/runs/headfix/bc_v4 --version 4 --weights rl/runs/headfix/readout_v4/bc.pt

WHY NOT EXACT-PLAN ACCURACY. `rl/train.py` reports the share of dawns on which
the readout picks every head's index correctly, and that number does not
compare across action spaces. Measured on the same 1,188 sessions: the v1
readout scores 59.9% and the v4 readout 33.3%, and the v1 number is the
misleading one twice over. v1 picks from 5 values a head, v4 from up to 45,
so v4 is strictly harder to be exactly right on; and v1's own LABEL is off by
2.16 tiles from what the leader actually held (1,200 leader dawns), so a
PERFECT v1 readout still asks for the wrong farm.

What transfers is the DECODED PLAN -- tiles of each crop, head of each animal,
hands -- which is the unit the planner emits and the unit rlv3/trajectory
measures the leaders in. So every prediction and every label is decoded back
to an absolute count and scored against the unquantised truth the corpus keeps
in `raw`, and two errors are kept apart because they have different fixes:

    label error   |decoded label      - truth|   the CEILING: what the grid
                                                 cannot say before training
    model error   |decoded prediction - truth|   what the readout adds

If model error sits far above label error, the grid is not the binding
constraint and a finer one buys nothing. If they sit together, the grid is.

Anchors for the residual (v1/v2) and fraction (v3) decodes are rebuilt from
the val sessions' own dawn observations; the corpus does not store them.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TURNS = 24
SCORED = ("melon_tiles", "wheat_tiles", "carrot_tiles", "tomato_tiles",
          "straw_tiles", "goose_target", "cow_target", "sheep_target", "hires")


def truth_of(name, raw, features):
    if name in features.QUOTA_CROP:
        return float(raw["crops"].get(features.QUOTA_CROP[name], 0))
    if name in features.BUY_KIND:
        k = features.BUY_KIND[name]
        return float(raw["herd"].get(k, 0) + (raw.get("shed") or {}).get(k, 0))
    if name == "hires":
        return float(raw["hires"])
    return None


def decode(name, val, anchors, version, features):
    if version >= 4:
        return float(val)
    frac = set(getattr(features, "FRACTION_HEADS", ()) or ())
    if version == 3 and name in frac:
        return float(val) * float(anchors.get(name) or 0)
    if name in features.ANCHORED and name in anchors:
        return float(val) + float(anchors[name])
    return float(val)


def dawn_obs(path):
    """-> {(seat, day): dawn observation} by re-running the recorded game.

    Session files are ACTION-ONLY -- `{seed, actions: [side0, side1]}` -- so
    observations do not exist on disk; they are rebuilt by replaying both
    sides' recorded actions verbatim on the original seed, exactly as
    rl/session_bc.py does. The public state is shared; each seat's `private`
    is its own.
    """
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    with open(path) as fh:
        s = json.load(fh)
    scripts = s["actions"]

    def at(script, t):
        return script[t] if t < len(script) and script[t] else \
            {"farmer": ["PASS"], "hands": [], "market": []}

    e = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    e.reset(2)
    out = {}
    while not e.done:
        o0 = e.state[0].observation
        t = o0["step"]
        if t % TURNS == 0:
            d = t // TURNS
            for side in (0, 1):
                obs = dict(o0)
                obs["private"] = e.state[side].observation.get("private") or {}
                obs["player"] = side
                obs.setdefault("step", t)
                out[(side, d)] = obs
        e.step([at(scripts[0], t), at(scripts[1], t)])
    return out


def val_rows(corpus):
    """Reproduce rl/train.py's session-level val split exactly."""
    with open(corpus + "_sessions.json") as fh:
        used = json.load(fh)["sessions"]
    counts = np.asarray([u["n"] for u in used], int)
    sid = np.repeat(np.arange(len(used)), counts)
    rng = np.random.default_rng(0)
    gids = np.arange(len(used)); rng.shuffle(gids)
    nv = max(1, int(len(gids) * 0.05))
    is_val = np.isin(sid, gids[:nv])
    return np.flatnonzero(is_val), sid, used


def score(corpus, version, weights, anchor_cache):
    import torch
    from kagent import features
    from kagent.gamestate import View
    from rl.common import PlannerNet, masked_dists

    features.set_heads_version(version)
    heads = features.HEADS
    va, sid, used = val_rows(corpus)
    va_set = set(va.tolist())

    rows = []
    with open(corpus + ".jsonl") as fh:
        for i, line in enumerate(fh):
            if i in va_set:
                r = json.loads(line); r["_sid"] = int(sid[i]); rows.append(r)
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
    masks = torch.tensor([sum(r["masks"], []) if isinstance(r["masks"][0], list)
                          else r["masks"] for r in rows], dtype=torch.float32)

    net = PlannerNet(x.shape[1], board=False)
    net.load_state_dict(torch.load(weights, map_location="cpu")["policy"])
    net.eval()
    with torch.no_grad():
        dists = masked_dists(net.forward_z(x, None), masks)
        pred = torch.stack([d.probs.argmax(-1) for d in dists], 1).numpy()

    # Replay each val session ONCE; every version and the priced column read
    # the same dawn observations.
    obs_cache = anchor_cache.setdefault("_obs", {})
    for r in rows:
        path = used[r["_sid"]]["path"]
        if path not in obs_cache:
            obs_cache[path] = dawn_obs(os.path.join(REPO, path))

    per = collections.defaultdict(lambda: {"model": [], "label": [], "priced": [], "n": 0})
    for k, r in enumerate(rows):
        path = used[r["_sid"]]["path"]
        obs = obs_cache[path].get((r["seat"], r["day"]))
        if obs is None:
            continue
        view = View(obs)
        anc = features.anchors(view) if version < 4 else {}
        # The MECHANISTIC prediction, nothing fitted: rlv3.portfolio prices
        # every asset at this dawn's draw, prices and opponent board, and
        # water-fills the ground. Same units as the readout, same truth.
        pkey = (path, r["seat"], r["day"])
        if pkey not in anchor_cache:
            try:
                from rlv3 import portfolio as P
                anchor_cache[pkey] = P.target_stock(view)["stock"]
            except Exception as e:
                # LOUD, ONCE. A silently-empty priced column reads as "the
                # mechanistic model has no opinion" when it means "it crashed".
                if not anchor_cache.get("_priced_err"):
                    import traceback
                    print(f"  [priced column failed at v{version} on {pkey[0].split('/')[-1]} "
                          f"seat {pkey[1]} day {pkey[2]}: {type(e).__name__}: {e}]")
                    traceback.print_exc()
                    anchor_cache["_priced_err"] = True
                anchor_cache[pkey] = {}
        priced = anchor_cache[pkey]
        for h, (name, vals) in enumerate(heads):
            if name not in SCORED:
                continue
            t = truth_of(name, r["raw"], features)
            if t is None:
                continue
            lab = decode(name, vals[r["choices"][h]], anc, version, features)
            prd = decode(name, vals[int(pred[k, h])], anc, version, features)
            s = per[name]; s["n"] += 1
            s["label"].append(abs(lab - t)); s["model"].append(abs(prd - t))
            # An asset ABSENT from the priced stock is the model saying "zero
            # of it" -- scored as 0, not skipped. Skipping would score the
            # mechanistic model only on the assets it chose to hold, which
            # flatters it on exactly the heads it gets wrong.
            if name in features.QUOTA_CROP:
                pv = priced.get(features.QUOTA_CROP[name], 0) if priced is not None else None
            elif name in features.BUY_KIND:
                pv = priced.get(features.BUY_KIND[name], 0) if priced is not None else None
            else:
                pv = None
            if pv is not None:
                s["priced"].append(abs(float(pv) - t))
    return {k: {"n": v["n"],
                "label_mae": statistics.mean(v["label"]),
                "model_mae": statistics.mean(v["model"]),
                "model_exact": sum(1 for e in v["model"] if e < 0.5) / v["n"],
                "priced_mae": statistics.mean(v["priced"]) if v["priced"] else None}
            for k, v in per.items()}, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--version", action="append", type=int, required=True)
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--out", default=None, help="JSON of every number printed")
    args = ap.parse_args()
    if not (len(args.corpus) == len(args.version) == len(args.weights)):
        raise SystemExit("--corpus/--version/--weights must be given in matched sets")

    anchor_cache = {}
    results = []
    for c, v, w in zip(args.corpus, args.version, args.weights):
        res, n = score(c, v, w, anchor_cache)
        results.append((v, res, n))
        print(f"v{v}: {n:,} val samples scored from {c}")

    print(f"\n{'':<14}" + "".join(f"{'v%d' % v:^30}" for v, _r, _n in results))
    print(f"{'head':<14}" + "".join(f"{'label':>9}{'model':>9}{'exact':>8}    "
                                    for _ in results))
    print(f"{'':<14}" + "".join(f"{'mae':>9}{'mae':>9}{'':>8}    " for _ in results))
    for name in SCORED:
        line = f"{name:<14}"
        for _v, res, _n in results:
            r = res.get(name)
            line += (f"{r['label_mae']:>9.2f}{r['model_mae']:>9.2f}{r['model_exact']:>8.1%}    "
                     if r else f"{'-':>9}{'-':>9}{'-':>8}    ")
        print(line)
    print(f"{'MEAN':<14}" + "".join(
        f"{statistics.mean(r['label_mae'] for r in res.values()):>9.2f}"
        f"{statistics.mean(r['model_mae'] for r in res.values()):>9.2f}"
        f"{statistics.mean(r['model_exact'] for r in res.values()):>8.1%}    "
        for _v, res, _n in results))
    print("\nlabel mae = the grid's own ceiling before any training; "
          "model mae = the trained readout, decoded to tiles, vs the truth.")
    _v, res, _n = results[-1]
    pr = {k: v["priced_mae"] for k, v in res.items() if v.get("priced_mae") is not None}
    if pr:
        print("\nPRICED (rlv3.portfolio.target_stock, NOTHING fitted) vs the same truth,"
              " absent asset = 0:")
        for name in SCORED:
            if name in pr:
                print(f"  {name:<14}{pr[name]:>9.2f} tiles")
        print(f"  {'MEAN':<14}{statistics.mean(pr.values()):>9.2f} tiles")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"versions": [{"version": v, "n_val": n, "heads": res}
                                    for v, res, n in results]}, fh, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
