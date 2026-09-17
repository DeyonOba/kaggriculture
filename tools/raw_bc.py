"""Raw-action behaviour cloning: a policy that emits ACTIONS, not targets.

Both previous BC attempts here cloned the wrong thing.

  rl/replay_bc.py   clones standing TARGETS at day boundaries (crop counts,
                    herd, hires, land) onto our 13 planner heads. Allocation is
                    measured FLAT -- the LP's provably optimal allocation, run
                    through our own scheduler, tied champ-h8 at +$828 +/- 2,317
                    -- so it cloned the one layer that cannot move money.
  tools/exec_bc.py  clones TASK CHOICE, but only as a ranker over the ~60 tasks
                    our own `build_tasks` proposes. It hit 69.4% held-out
                    against our rule's 27.9% and gated NULL over 4,800 paired
                    games; it can never do anything our scheduler did not offer.

Neither emits an action. Our RL was likewise a residual on the hand-written
champion plan, and the verdict that closed that track was that training was not
the constraint -- the narrow economy the executor could express was.

The case for doing it at the action level is in our own numbers: a TAPE is a
perfect but blind replay of one champion's action stream, and it scores 0.796
(tape-129) to 0.841 (tape-sel) ladder win against champ-h8's 0.319. Champion
actions are worth ~1750 rating played without even looking at the board. A
policy that reproduces them CONDITIONED on state should beat replaying them
blindly, because it can adapt when the world diverges -- which is exactly what
the tapes' desync-repair code exists to paper over.

Label space, measured over 3000+ seats: 37 distinct unit actions (top 30 cover
99.95%) and 21 market-order types. Small enough for a plain classifier.

Sessions store actions and the seed but NOT observations, so `build` replays
each session through the engine to regenerate the state every action was taken
from.

    PYTHONPATH=. uv run python tools/raw_bc.py build --n 150 --procs 5
    PYTHONPATH=. uv run python tools/raw_bc.py fit
    PYTHONPATH=. uv run python tools/raw_bc.py export --name rawbc-v1
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

DATA = os.path.join(REPO, "rl", "datasets", "rawbc")
SESS = os.path.join(REPO, "rl", "datasets", "sessions")

# --- label space -------------------------------------------------------------
# Ordered so the index is stable across build/fit/export.
UNIT_ACTIONS = [
    ("PASS",), ("NORTH",), ("SOUTH",), ("EAST",), ("WEST",),
    ("WATER",), ("HARVEST",), ("COLLECT_FERTILIZER",), ("FEED",), ("CARE",),
    ("FERTILIZE",), ("DIG",), ("DROP",), ("BUILD_COOP",), ("BUILD_PASTURE",),
    ("PLANT", "WHEAT"), ("PLANT", "CARROT"), ("PLANT", "TOMATO"),
    ("PLANT", "STRAWBERRY"), ("PLANT", "MELON"),
    ("PICKUP", "WHEAT"), ("PICKUP", "FERTILIZER"),
    ("PICKUP", "GOOSE"), ("PICKUP", "COW"), ("PICKUP", "SHEEP"),
    ("PLACE", "GOOSE"), ("PLACE", "COW"), ("PLACE", "SHEEP"),
]
UA_INDEX = {a: i for i, a in enumerate(UNIT_ACTIONS)}

# Market orders are farm-level, one LIST per turn, and they carry the whole
# economy: HIRE is the labour supply, BUY_SEED is everything that gets planted,
# SELL is all revenue. Cloning unit actions alone would produce a farm that
# never hires, never plants and never earns. So a second head predicts, per
# turn, WHICH order types to emit (multi-label) and HOW MANY hands to hire;
# quantities for the rest are filled by rule at play time, since the recorded
# quantities are mostly 1-2 and the choice of WHAT to trade carries the signal.
MARKET_ACTIONS = [
    ("SELL", p) for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                          "EGG", "MILK", "WOOL", "FERTILIZER"]
] + [
    ("BUY_SEED", c) for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
] + [
    ("BUY_ANIMAL", a) for a in ["GOOSE", "COW", "SHEEP"]
] + [("BUY_PRODUCT", "WHEAT"), ("BUY_PRODUCT", "FERTILIZER"), ("BUY_LAND",)]
MA_INDEX = {a: i for i, a in enumerate(MARKET_ACTIONS)}
MAX_HIRE = 8

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_L = ["GOOSE", "COW", "SHEEP"]
SHED = [(4, 4), (4, 5), (5, 4), (5, 5)]


def key_of(a):
    """An action list -> its label tuple, or None if outside the label space."""
    if not isinstance(a, list) or not a:
        return ("PASS",)
    op = a[0]
    if op in ("PLANT", "PICKUP", "PLACE"):
        # PLACE carries the PRODUCT an animal yields in some recordings; map
        # it back to the animal so the class set stays small.
        p = a[1] if len(a) > 1 else None
        back = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
        if op == "PLACE":
            p = back.get(p, p)
        return (op, p) if (op, p) in UA_INDEX else None
    return (op,) if (op,) in UA_INDEX else None


# --- features ----------------------------------------------------------------
# Kept deliberately local: this policy decides ONE unit's action this turn, so
# it sees its own tile, its four neighbours, what it carries, and a few farm
# and market aggregates. Everything here is computable at play time from the
# observation alone -- export embeds this exact function.

def feats(obs, seat, idx, ux, uy):
    farm = obs["farms"][seat]
    tiles = farm["tiles"]
    n = len(tiles)
    priv = obs.get("private") or {}
    invs = priv.get("inventories") or []
    inv = (invs[idx] if idx < len(invs) else {}) or {}
    shed = priv.get("shed") or {}
    px = (obs.get("market") or {}).get("prices") or {}
    day, hour = obs["day"], obs["hour"]

    def tile_at(x, y):
        if 0 <= y < n and 0 <= x < len(tiles[y]):
            t = tiles[y][x]
            return t if isinstance(t, dict) else None
        return None

    def tile_vec(t, x, y):
        """11 numbers describing one tile."""
        if t is None:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        kind = t.get("kind")
        is_plant = kind == "PLANT"
        is_animal = "animal" in t
        age = (day - t.get("planted_day", day)) if is_plant else 0
        return [
            0,
            1 if is_plant else 0,
            1 if is_animal else 0,
            1 if kind == "WEED" else 0,
            1 if kind in ("COOP", "PASTURE") else 0,
            min(age, 20) / 20.0,
            min(t.get("yield_units", 0), 6) / 6.0,
            1 if t.get("watered_today") else 0,
            1 if t.get("fed_today") else 0,
            1 if t.get("cared_today") else 0,
            1 if t.get("fertilizer_available") else 0,
        ]

    here = tile_at(ux, uy)
    v = [day / 30.0, hour / 24.0, idx / 12.0, 1.0 if idx == 0 else 0.0,
         ux / max(1, n - 1), uy / max(1, n - 1),
         min(min(abs(ux - a) + abs(uy - b) for a, b in SHED), 12) / 12.0,
         min(float(farm.get("money", 0)), 60000) / 60000.0,
         min(len(farm.get("hands") or []), 14) / 14.0,
         min(len(farm.get("unlocked_quadrants") or ["NW"]), 4) / 4.0]
    v += tile_vec(here, ux, uy)
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        v += tile_vec(tile_at(ux + dx, uy + dy), ux + dx, uy + dy)
    # what this unit carries, and what the shed holds
    v += [min(float(inv.get(p, 0)), 8) / 8.0 for p in PRODUCTS]
    v += [min(float(shed.get(p, 0)), 40) / 40.0 for p in PRODUCTS]
    # Seeds are FARM-level in private["seeds"], not in a unit's inventory --
    # the engine checks private["seeds"] in PLANT (kaggriculture.py:425). A
    # first version read them from `inv`, so the seed features were always zero
    # AND the export's legality check made PLANT permanently illegal: the agent
    # could never plant, and scored $90.
    seeds = priv.get("seeds") or {}
    v += [min(float(seeds.get(c, 0)), 20) / 20.0 for c in CROPS_L]
    v += [min(float(inv.get(a, 0)), 4) / 4.0 for a in ANIMALS_L]
    # farm aggregates: how much work of each kind is outstanding
    unwatered = unfed = uncared = ripe = empty = weeds = fert_ready = 0
    for row in tiles:
        for t in row:
            if t is None:
                empty += 1
            elif isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    if not t.get("watered_today"):
                        unwatered += 1
                    if t.get("yield_units", 0) > 0:
                        ripe += 1
                elif t.get("kind") == "WEED":
                    weeds += 1
                if "animal" in t:
                    if not t.get("fed_today"):
                        unfed += 1
                    if not t.get("cared_today"):
                        uncared += 1
                    if t.get("yield_units", 0) > 0:
                        ripe += 1
                    if t.get("fertilizer_available"):
                        fert_ready += 1
    v += [min(unwatered, 30) / 30.0, min(unfed, 20) / 20.0,
          min(uncared, 20) / 20.0, min(ripe, 30) / 30.0,
          min(empty, 40) / 40.0, min(weeds, 20) / 20.0,
          min(fert_ready, 20) / 20.0]
    v += [min(float(px.get(p, 0)), 400) / 400.0 for p in PRODUCTS]
    return v


N_FEAT = None   # set on first use


# --- build -------------------------------------------------------------------

def _job(task):
    path, min_rating = task
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    try:
        s = json.load(open(path))
    except Exception:
        return None
    sc = s.get("scores") or [None, None]
    seats = [i for i in (0, 1) if sc[i] is not None and sc[i] >= min_rating]
    if not seats:
        return None
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    try:
        env.run([make_script_agent(s["actions"][0]),
                 make_script_agent(s["actions"][1])])
    except Exception:
        return None
    X, Y, MX, MY, MH = [], [], [], [], []
    for t in range(1, len(env.steps)):
        # steps[t] is the state AFTER the action at steps[t]; the state an
        # action was chosen from is steps[t-1]. Verified 96.8% vs 34.0% on
        # farmer MOVEs (see tools/action_audit.py).
        prev = env.steps[t - 1]
        for seat in seats:
            obs = prev[seat]["observation"]
            farm = obs["farms"][seat]
            units = [tuple(farm.get("farmer") or (0, 0))]
            units += [tuple(h) for h in (farm.get("hands") or [])]
            act = env.steps[t][seat].get("action") or {}
            taken = [act.get("farmer")] + list(act.get("hands") or [])
            for i, (ux, uy) in enumerate(units):
                if i >= len(taken):
                    break
                k = key_of(taken[i])
                if k is None:
                    continue
                X.append(feats(obs, seat, i, ux, uy))
                Y.append(UA_INDEX[k])
            # market: one sample per turn, from the farmer's viewpoint
            multi = [0.0] * len(MARKET_ACTIONS)
            hires = 0
            for m in (act.get("market") or []):
                if not isinstance(m, list) or not m:
                    continue
                if m[0] == "HIRE":
                    hires += 1
                    continue
                k2 = (m[0], m[1]) if len(m) > 1 else (m[0],)
                if k2 in MA_INDEX:
                    multi[MA_INDEX[k2]] = 1.0
            MX.append(feats(obs, seat, 0, *units[0]))
            MY.append(multi)
            MH.append(min(hires, MAX_HIRE))
    return X, Y, MX, MY, MH


def build(args):
    import numpy as np
    files = sorted(glob.glob(os.path.join(SESS, "session-*.json")))
    # Held-out convention used everywhere else in this repo: episode % 5 == 0.
    files = [f for f in files if int(f.split("-")[-1][:-5]) % 5 != 0] \
        if not args.heldout else \
        [f for f in files if int(f.split("-")[-1][:-5]) % 5 == 0]
    picked, seen = [], 0
    for f in files:
        try:
            sc = json.load(open(f)).get("scores") or [None, None]
        except Exception:
            continue
        if any(s is not None and s >= args.min_rating for s in sc):
            picked.append(f)
            if len(picked) >= args.n:
                break
        seen += 1
    print(f"scanned {seen + len(picked)} sessions -> {len(picked)} with a "
          f"{args.min_rating}+ seat", flush=True)
    X, Y, MX, MY, MH = [], [], [], [], []
    with Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(
                _job, [(f, args.min_rating) for f in picked])):
            if r:
                X.extend(r[0]); Y.extend(r[1])
                MX.extend(r[2]); MY.extend(r[3]); MH.extend(r[4])
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(picked)} replayed, {len(X):,} unit / "
                      f"{len(MX):,} market samples", flush=True)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "heldout.npz" if args.heldout else "train.npz")
    np.savez_compressed(out, X=np.array(X, dtype="float32"),
                        Y=np.array(Y, dtype="int64"),
                        MX=np.array(MX, dtype="float32"),
                        MY=np.array(MY, dtype="float32"),
                        MH=np.array(MH, dtype="int64"))
    print(f"wrote {out}: {len(X):,} unit samples x {len(X[0])} features, "
          f"{len(MX):,} market samples")


# --- fit ---------------------------------------------------------------------

def fit(args):
    import numpy as np, torch, torch.nn as nn
    d = np.load(os.path.join(DATA, "train.npz"))
    X, Y = torch.tensor(d["X"]), torch.tensor(d["Y"])
    n = len(X)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    X, Y = X[perm], Y[perm]
    cut = int(n * 0.9)
    Xtr, Ytr, Xva, Yva = X[:cut], Y[:cut], X[cut:], Y[cut:]
    print(f"{n:,} samples, {X.shape[1]} features, {len(UNIT_ACTIONS)} classes")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.ReLU(),
                          nn.Linear(args.hidden, args.hidden), nn.ReLU(),
                          nn.Linear(args.hidden, len(UNIT_ACTIONS)))
    opt = torch.optim.Adam(model.parameters(), args.lr)
    lf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for i in range(0, cut, 1024):
            opt.zero_grad()
            lf(model(Xtr[i:i+1024]), Ytr[i:i+1024]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == Yva).float().mean().item()
        print(f"  epoch {ep+1:>2}  held-out accuracy {acc:.4f}", flush=True)

    # Majority-class baseline: the number any classifier must beat to mean
    # anything at all.
    base = torch.bincount(Yva, minlength=len(UNIT_ACTIONS)).max().item() / len(Yva)
    print(f"\n  majority-class baseline {base:.4f}   model {acc:.4f}")
    with torch.no_grad():
        pred = model(Xva).argmax(1)
    print(f"\n  {'action':<26}{'support':>9}{'recall':>9}")
    for i, a in enumerate(UNIT_ACTIONS):
        m = Yva == i
        if m.sum() < 50:
            continue
        print(f"  {str(a):<26}{int(m.sum()):>9,}"
              f"{(pred[m] == i).float().mean().item():>9.3f}")
    # --- market head: which order types this turn, and how many hands -------
    MX, MY, MH = torch.tensor(d["MX"]), torch.tensor(d["MY"]), torch.tensor(d["MH"])
    pm = torch.randperm(len(MX), generator=torch.Generator().manual_seed(1))
    MX, MY, MH = MX[pm], MY[pm], MH[pm]
    mc = int(len(MX) * 0.9)
    mkt = nn.Sequential(nn.Linear(MX.shape[1], args.hidden), nn.ReLU(),
                        nn.Linear(args.hidden, args.hidden), nn.ReLU(),
                        nn.Linear(args.hidden, len(MARKET_ACTIONS) + MAX_HIRE + 1))
    mopt = torch.optim.Adam(mkt.parameters(), args.lr)
    # Each order type fires on a small fraction of turns, so an unweighted BCE
    # simply predicts "no order" everywhere: the first run scored F1 0.079 and
    # would have produced a farm that never hires, plants or sells. pos_weight
    # = negatives/positives per class restores the balance.
    pos = MY[:mc].sum(0).clamp(min=1.0)
    pw = ((mc - pos) / pos).clamp(max=200.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    ce = nn.CrossEntropyLoss()
    K = len(MARKET_ACTIONS)
    print(f"\nmarket head: {len(MX):,} turns, {K} order types + hire count")
    for ep in range(args.epochs):
        mkt.train()
        for i in range(0, mc, 1024):
            mopt.zero_grad()
            o = mkt(MX[i:i+1024])
            (bce(o[:, :K], MY[i:i+1024]) + ce(o[:, K:], MH[i:i+1024])).backward()
            mopt.step()
        mkt.eval()
        with torch.no_grad():
            o = mkt(MX[mc:])
            pred = (o[:, :K] > 0).float()
            tp = (pred * MY[mc:]).sum().item()
            prec = tp / max(1.0, pred.sum().item())
            rec = tp / max(1.0, MY[mc:].sum().item())
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            hacc = (o[:, K:].argmax(1) == MH[mc:]).float().mean().item()
        print(f"  epoch {ep+1:>2}  order P {prec:.3f} R {rec:.3f} F1 {f1:.3f}"
              f"  hire-count acc {hacc:.4f}", flush=True)

    # pos_weight deliberately over-predicts, so threshold 0 emits ~4x too many
    # orders (precision 0.17). Pick the threshold on held-out rather than
    # shipping the default: a spurious BUY spends cash the farm needs.
    with torch.no_grad():
        o = mkt(MX[mc:])[:, :K]
    best = (0.0, 0.0)
    for thr in [x / 2.0 for x in range(0, 17)]:
        pr = (o > thr).float()
        tp = (pr * MY[mc:]).sum().item()
        p_ = tp / max(1.0, pr.sum().item()); r_ = tp / max(1.0, MY[mc:].sum().item())
        f_ = 2 * p_ * r_ / max(1e-9, p_ + r_)
        if f_ > best[1]:
            best = (thr, f_)
        if thr in (0.0, 2.0, 4.0, 6.0, 8.0):
            print(f"    thr {thr:>4.1f}  P {p_:.3f}  R {r_:.3f}  F1 {f_:.3f}")
    print(f"  chosen market threshold {best[0]:.1f} (F1 {best[1]:.3f})")

    os.makedirs(DATA, exist_ok=True)
    w = {k: v.tolist() for k, v in model.state_dict().items()}
    mw = {k: v.tolist() for k, v in mkt.state_dict().items()}
    json.dump({"w": w, "mw": mw, "hidden": args.hidden, "n_feat": X.shape[1],
               "acc": acc, "baseline": base, "f1": float(best[1]),
               "thr": best[0], "hacc": hacc},
              open(os.path.join(DATA, "weights.json"), "w"))
    print(f"\nwrote {os.path.join(DATA, 'weights.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fit", "export"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-rating", type=float, default=3000)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default="rawbc-v1")
    args = ap.parse_args()
    if args.mode == "build":
        build(args)
    elif args.mode == "fit":
        fit(args)
    else:
        from tools.raw_bc_export import export
        export(args)


if __name__ == "__main__":
    main()
