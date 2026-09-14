"""Win-probability value head, and per-decision credit assignment.

Every experiment in this project has been a whole-season A/B against ~$20k of
per-world noise, which is why confirming a $2,376 improvement needed 50 games.
This fits V(s) = P(this seat finishes with more money | s) on the recorded
corpus, so a game can be scored decision by decision instead: the change in win
probability across a day is that day's credit, the way a chess engine reports
centipawn loss per move.

Why a win probability and not money: money is risk-neutral everywhere, but the
right play is not. Ahead $30k on day 27 you should liquidate and kill variance;
behind $30k you should take the gamble. The two banks in an episode correlate
+0.73, so variance decides a lot of games, and only an outcome-valued objective
prices it.

    PYTHONPATH=. uv run python tools/value_head.py build --n 2500 --procs 6
    PYTHONPATH=. uv run python tools/value_head.py fit
    PYTHONPATH=. uv run python tools/value_head.py attribute --a champ-h8 --b pub-ahmed-v38 --n 8

`build` replays recorded sessions and snapshots both seats at every day
boundary, labelled with who actually won. `fit` trains one L2 logistic model
per day (held out on episode %% 5 == 0) and reports accuracy, log-loss and
calibration against a margin-sign baseline. `attribute` plays live games and
prints where win probability is won and lost.
"""

import argparse
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, SHOPS  # noqa: E402

CROP_LIST = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_LIST = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ROWS = "rl/datasets/value/rows.npz"
WEIGHTS = "rl/value_weights.json"
N_DAYS = 30


def feature_names():
    n = ["money_me", "money_op", "margin", "log_margin"]
    n += [f"crop_me_{c}" for c in CROP_LIST] + [f"crop_op_{c}" for c in CROP_LIST]
    n += [f"herd_me_{a}" for a in ANIMAL_LIST] + [f"herd_op_{a}" for a in ANIMAL_LIST]
    n += ["hands_me", "hands_op", "quads_me", "quads_op"]
    n += ["crop_value_me", "crop_value_op", "herd_value_me", "herd_value_op"]
    n += [f"shed_{p}" for p in PRODUCTS] + ["shed_value"]
    n += [f"price_{p}" for p in PRODUCTS]
    n += [f"consumers_{p}" for p in PRODUCTS]
    return n


NAMES = feature_names()
NF = len(NAMES)


def _census(farm):
    crops = {c: 0 for c in CROP_LIST}
    herd = {a: 0 for a in ANIMAL_LIST}
    crop_value = 0.0
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                c = t.get("crop")
                if c in crops:
                    crops[c] += 1
                    cd = CROPS[c]
                    left = max(0, cd["max_yield"] - t.get("yield_units", 0))
                    crop_value += left * MARKET_PARAMS[c]["base"]
            elif t.get("kind") in ("PASTURE", "COOP"):
                a = t.get("animal")
                a = a.get("type") if isinstance(a, dict) else a
                if a in herd:
                    herd[a] += 1
    return crops, herd, crop_value


def features(obs, seat):
    """Feature row from ONE seat's observation. Public farm state for both
    sides plus this seat's own shed (the only meaningful private state)."""
    me, op = obs["farms"][seat], obs["farms"][1 - seat]
    market = obs["market"]
    prices, inv = market["prices"], market["inventory"]
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    shed = (obs.get("private") or {}).get("shed") or {}

    c_me, h_me, cv_me = _census(me)
    c_op, h_op, cv_op = _census(op)
    margin = float(me["money"]) - float(op["money"])
    hv_me = sum(h_me[a] * MARKET_PARAMS[ANIMALS[a]["product"]]["base"] for a in ANIMAL_LIST)
    hv_op = sum(h_op[a] * MARKET_PARAMS[ANIMALS[a]["product"]]["base"] for a in ANIMAL_LIST)

    x = [float(me["money"]), float(op["money"]), margin,
         float(np.sign(margin) * np.log1p(abs(margin)))]
    x += [c_me[c] for c in CROP_LIST] + [c_op[c] for c in CROP_LIST]
    x += [h_me[a] for a in ANIMAL_LIST] + [h_op[a] for a in ANIMAL_LIST]
    x += [len(me.get("hands") or []), len(op.get("hands") or []),
          len(me.get("unlocked_quadrants") or []), len(op.get("unlocked_quadrants") or [])]
    x += [cv_me, cv_op, hv_me, hv_op]
    x += [float(shed.get(p, 0)) for p in PRODUCTS]
    x += [sum(shed.get(p, 0) * prices.get(p, 0) for p in PRODUCTS)]
    x += [prices.get(p, 0) / MARKET_PARAMS[p]["base"] for p in PRODUCTS]
    x += [float(sum(1 for s in shops if p in SHOPS.get(s, ()))) for p in PRODUCTS]
    assert len(x) == NF, (len(x), NF)
    return x


def _one(path):
    """Replay one recorded session; return (rows, days, labels, episode, seats)."""
    from kaggle_environments import make
    from rl.rollout import make_script_agent
    try:
        s = json.load(open(path))
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s["seed"]}, debug=False)
        env.run([make_script_agent(s["actions"][0]), make_script_agent(s["actions"][1])])
    except Exception:  # noqa: BLE001
        return None
    final = [env.steps[-1][i]["reward"] for i in (0, 1)]
    if final[0] is None or final[1] is None:
        return None
    out = []
    for seat in (0, 1):
        y = 1.0 if final[seat] > final[1 - seat] else (0.5 if final[seat] == final[1 - seat] else 0.0)
        for d in range(N_DAYS):
            i = d * 24
            if i >= len(env.steps):
                break
            obs = env.steps[i][seat]["observation"]
            if "farms" not in obs:
                continue
            out.append((features(obs, seat), d, y, s["episode"], seat))
    return out


def build(args):
    paths = sorted(glob.glob("rl/datasets/sessions/session-*.json"))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    X, D, Y, E, S = [], [], [], [], []
    done = 0
    with Pool(args.procs) as pool:
        for res in pool.imap_unordered(_one, paths, chunksize=4):
            done += 1
            if res:
                for x, d, y, ep, seat in res:
                    X.append(x)
                    D.append(d)
                    Y.append(y)
                    E.append(ep)
                    S.append(seat)
            if done % 200 == 0:
                print(f"  {done}/{len(paths)} games, {len(X):,} rows", flush=True)
    np.savez_compressed(ROWS, X=np.array(X, dtype=np.float32), day=np.array(D, dtype=np.int16),
                        y=np.array(Y, dtype=np.float32), episode=np.array(E, dtype=np.int64),
                        seat=np.array(S, dtype=np.int8))
    print(f"wrote {ROWS}: {len(X):,} rows from {done} games")


def _fit_logistic(X, y, lam=1.0, iters=40):
    """L2 logistic regression by Newton/IRLS. X already has an intercept column."""
    n, d = X.shape
    w = np.zeros(d)
    reg = lam * np.eye(d)
    reg[0, 0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (y - p) - reg @ w
        W = np.maximum(p * (1 - p), 1e-6)
        H = (X.T * W) @ X + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(args):
    z = np.load(ROWS)
    X, day, y, ep = z["X"].astype(np.float64), z["day"], z["y"].astype(np.float64), z["episode"]
    test = (ep % 5) == 0
    print(f"{len(X):,} rows, {len(set(ep.tolist())):,} games; holdout episode%5==0 -> {test.sum():,} rows\n")
    model = {"features": NAMES, "days": {}}
    print(f"{'day':>3} {'n_train':>8} {'acc':>6} {'logloss':>8} | {'base acc':>8} {'base ll':>8} {'gain':>6}")
    accs, lls, bacc, blls = [], [], [], []
    for d in range(N_DAYS):
        # Train on a +/-1 day window so adjacent days' models agree and the V
        # trajectory is comparable across days; evaluate on day d alone.
        w_lo, w_hi = (d - args.smooth), (d + args.smooth)
        tr = (day >= w_lo) & (day <= w_hi) & ~test
        te = (day == d) & test
        if tr.sum() < 200 or te.sum() < 50:
            continue
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-9] = 1.0
        A = np.hstack([np.ones((tr.sum(), 1)), (Xtr - mu) / sd])
        w = _fit_logistic(A, ytr, lam=args.lam)
        B = np.hstack([np.ones((te.sum(), 1)), (X[te] - mu) / sd])
        p = 1.0 / (1.0 + np.exp(-np.clip(B @ w, -30, 30)))
        yte = y[te]
        acc = float(np.mean((p > 0.5) == (yte > 0.5)))
        ll = _logloss(yte, p)
        # Baseline: sign of the bank margin alone.
        mp = 1.0 / (1.0 + np.exp(-X[te][:, 2] / 20000.0))
        b_acc = float(np.mean((mp > 0.5) == (yte > 0.5)))
        b_ll = _logloss(yte, mp)
        accs.append(acc); lls.append(ll); bacc.append(b_acc); blls.append(b_ll)
        print(f"{d:>3} {tr.sum():>8,} {acc:>6.3f} {ll:>8.4f} | {b_acc:>8.3f} {b_ll:>8.4f} {acc-b_acc:>+6.3f}")
        model["days"][str(d)] = {"mean": mu.tolist(), "std": sd.tolist(), "coef": w.tolist()}
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    json.dump(model, open(WEIGHTS, "w"))
    print(f"\nmean acc {np.mean(accs):.3f} (baseline {np.mean(bacc):.3f}), "
          f"logloss {np.mean(lls):.4f} (baseline {np.mean(blls):.4f})")
    print(f"wrote {WEIGHTS} ({os.path.getsize(WEIGHTS)/1000:.0f} KB)")

    # Calibration on the pooled holdout, and the strongest signals late/mid/early.
    print("\ncalibration (pooled holdout):")
    allp, ally = [], []
    for d, m in model["days"].items():
        te = (day == int(d)) & test
        if not te.sum():
            continue
        mu, sd, w = np.array(m["mean"]), np.array(m["std"]), np.array(m["coef"])
        B = np.hstack([np.ones((te.sum(), 1)), (X[te] - mu) / sd])
        allp.append(1.0 / (1.0 + np.exp(-np.clip(B @ w, -30, 30))))
        ally.append(y[te])
    allp, ally = np.concatenate(allp), np.concatenate(ally)
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (allp >= lo) & (allp < lo + 0.1)
        if m.sum() > 30:
            print(f"  predicted {lo:.1f}-{lo+0.1:.1f}: actual {ally[m].mean():.3f}  (n={m.sum():,})")
    for d in (6, 12, 20):
        if str(d) not in model["days"]:
            continue
        w = np.array(model["days"][str(d)]["coef"])[1:]
        top = np.argsort(-np.abs(w))[:8]
        print(f"\nday {d} strongest signals: " + ", ".join(f"{NAMES[i]} {w[i]:+.2f}" for i in top))


def _load_model():
    m = json.load(open(WEIGHTS))
    return {int(k): (np.array(v["mean"]), np.array(v["std"]), np.array(v["coef"]))
            for k, v in m["days"].items()}


def value(model, obs, seat, day):
    if day not in model:
        day = max(k for k in model if k <= day) if any(k <= day for k in model) else min(model)
    mu, sd, w = model[day]
    x = (np.array(features(obs, seat)) - mu) / sd
    return float(1.0 / (1.0 + np.exp(-np.clip(np.r_[1.0, x] @ w, -30, 30))))


def attribute(args):
    from kaggle_environments import make
    from arena.registry import resolve
    model = _load_model()
    a, b = resolve(args.a), resolve(args.b)
    per_day = np.zeros(N_DAYS)
    level = np.zeros(N_DAYS)
    counts = np.zeros(N_DAYS)
    wins = 0
    for k in range(args.n):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.base_seed + k}, debug=False)
        env.run([a, b])
        vs = []
        for d in range(N_DAYS):
            i = d * 24
            if i >= len(env.steps):
                break
            vs.append(value(model, env.steps[i][0]["observation"], 0, d))
        final = [env.steps[-1][i]["reward"] for i in (0, 1)]
        won = final[0] > final[1]
        wins += won
        vs.append(1.0 if won else 0.0)
        for d in range(len(vs) - 1):
            per_day[d] += vs[d + 1] - vs[d]
            level[d] += vs[d]
            counts[d] += 1
        print(f"  seed {args.base_seed+k}: V0={vs[0]:.2f} -> {'WIN' if won else 'LOSS'}  "
              f"${final[0]:,.0f} vs ${final[1]:,.0f}")
    print(f"\n{args.a} vs {args.b}: {wins}/{args.n} wins\n")
    print("Each day has its own fitted model, so a single day's dV mixes real state change with")
    print("disagreement between two models. The LEVEL column is the robust read.\n")
    print(f"{'day':>3} {'mean V':>7} {'mean dV':>9}  {'win probability level'}")
    order = np.argsort(np.where(counts > 0, per_day / np.maximum(counts, 1), 0))
    for d in range(N_DAYS):
        if counts[d] == 0:
            continue
        dv = per_day[d] / counts[d]
        lvl = level[d] / counts[d]
        print(f"{d:>3} {lvl:>7.3f} {dv:>+9.3f}  {'#' * int(round(lvl * 40))}")
    worst = [int(d) for d in order[:5] if counts[d] > 0]
    print(f"\nworst days (largest mean loss of win probability): {worst}")
    lv = np.where(counts > 0, level / np.maximum(counts, 1), np.nan)
    valid = ~np.isnan(lv)
    if valid.sum() > 3:
        idx = np.flatnonzero(valid)
        drops = [(int(idx[i]), float(lv[idx[i + 2]] - lv[idx[i]])) for i in range(len(idx) - 2)]
        drops.sort(key=lambda t: t[1])
        print("steepest 2-day falls in level: " +
              ", ".join(f"day {d}->{d+2} {v:+.3f}" for d, v in drops[:4]))


GROUPS = {
    "money": [("money_me", "money_op")],
    "crops (all)": [(f"crop_me_{c}", f"crop_op_{c}") for c in CROP_LIST],
    "herd (all)": [(f"herd_me_{a}", f"herd_op_{a}") for a in ANIMAL_LIST],
    "hands": [("hands_me", "hands_op")],
    "quadrants": [("quads_me", "quads_op")],
    "crop value": [("crop_value_me", "crop_value_op")],
    "herd value": [("herd_value_me", "herd_value_op")],
}
for _c in CROP_LIST:
    GROUPS[f"  crop {_c.lower()}"] = [(f"crop_me_{_c}", f"crop_op_{_c}")]
for _a in ANIMAL_LIST:
    GROUPS[f"  herd {_a.lower()}"] = [(f"herd_me_{_a}", f"herd_op_{_a}")]

IDX = {n: i for i, n in enumerate(NAMES)}


def _v_from_x(model, x, day):
    if day not in model:
        day = max(k for k in model if k <= day)
    mu, sd, w = model[day]
    z = (np.asarray(x) - mu) / sd
    return float(1.0 / (1.0 + np.exp(-np.clip(np.r_[1.0, z] @ w, -30, 30))))


def diagnose(args):
    """How much win probability would we gain on each day if one part of our
    farm matched the opponent's? Counterfactual is evaluated inside a single
    day's model, so it is immune to the day-to-day model disagreement that makes
    raw dV noisy."""
    from kaggle_environments import make
    from arena.registry import resolve
    model = _load_model()
    a, b = resolve(args.a), resolve(args.b)
    days = list(range(args.lo, args.hi + 1))
    uplift = {g: np.zeros(len(days)) for g in GROUPS}
    base_v = np.zeros(len(days))
    n = 0
    for k in range(args.n):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.base_seed + k}, debug=False)
        env.run([a, b])
        n += 1
        for j, d in enumerate(days):
            i = d * 24
            if i >= len(env.steps):
                continue
            obs = env.steps[i][0]["observation"]
            x = features(obs, 0)
            v0 = _v_from_x(model, x, d)
            base_v[j] += v0
            for g, pairs in GROUPS.items():
                xc = list(x)
                for me_n, op_n in pairs:
                    xc[IDX[me_n]] = x[IDX[op_n]]
                if g == "money":  # margin follows money
                    xc[IDX["margin"]] = 0.0
                    xc[IDX["log_margin"]] = 0.0
                uplift[g][j] += _v_from_x(model, xc, d) - v0
        print(f"  seed {args.base_seed+k} done", flush=True)

    base_v /= n
    print(f"\n{args.a} vs {args.b}, {n} games: win-probability gain if OUR side matched THEIRS\n")
    hdr = "  ".join(f"d{d:<4}" for d in days)
    print(f"{'group':<16} {hdr}  {'mean':>7}")
    rows = []
    for g, arr in uplift.items():
        arr = arr / n
        rows.append((float(arr.mean()), g, arr))
    for m, g, arr in sorted(rows, key=lambda r: -r[0]):
        cells = "  ".join(f"{v:>+5.2f}" for v in arr)
        print(f"{g:<16} {cells}  {m:>+7.3f}")
    print(f"\n{'our V level':<16} " + "  ".join(f"{v:>5.2f}" for v in base_v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fit", "attribute", "diagnose"])
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--smooth", type=int, default=1, help="+/- days pooled into each day's fit")
    ap.add_argument("--a", default="champ-h8")
    ap.add_argument("--b", default="pub-ahmed-v38")
    ap.add_argument("--base-seed", type=int, default=31000)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--hi", type=int, default=17)
    args = ap.parse_args()
    {"build": build, "fit": fit, "attribute": attribute, "diagnose": diagnose}[args.cmd](args)


if __name__ == "__main__":
    main()
