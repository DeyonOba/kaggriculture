"""Single-deviation counterfactual bandit over the champion plan.

Each WORLD (seed, seat, opponent) is played once by the plain champion plan
(the twin) and K times with exactly one (day, head, offset) deviation from
it. The paired margin gain of each deviated game attributes cleanly to that
one decision, so every game is one labelled sample:

    (state features at the deviation day, head, offset) -> gain vs champ-h3

`collect` appends samples to a JSONL; `report` prints per-(head, offset)
mean gains with standard errors and the context splits that separate wins
from losses; `fit` trains per-(head, offset) ridge regressors and writes a
rules file the BanditPolicy applies at play time.

    PYTHONPATH=. uv run python -u rl/bandit.py collect --worlds 40 --k 4 --procs 6
    PYTHONPATH=. uv run python rl/bandit.py report
    PYTHONPATH=. uv run python rl/bandit.py fit --min-gain 1500
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KAGG_RESIDUAL"] = "1"  # workers inherit; residual head set everywhere
from kagent import features  # noqa: E402
from kagent.features import ANIMAL_LAST_BUY, LAST_PLANT  # noqa: E402
from rl import rollout  # noqa: E402

features.set_mode("residual")

OUT = "rl/datasets/bandit/samples.jsonl"
RULES = "rl/datasets/bandit/rules.json"
SESS_DIR = "rl/datasets/sessions"

# Day window in which a deviation on each head can matter.
HEAD_DAYS = {
    "melon_tiles": (0, LAST_PLANT["MELON"]),
    "wheat_tiles": (1, LAST_PLANT["WHEAT"]),
    "carrot_tiles": (2, LAST_PLANT["CARROT"]),
    "tomato_tiles": (2, LAST_PLANT["TOMATO"]),
    "straw_tiles": (2, LAST_PLANT["STRAWBERRY"]),
    "goose_target": (1, ANIMAL_LAST_BUY["GOOSE"]),
    "cow_target": (0, ANIMAL_LAST_BUY["COW"]),
    "sheep_target": (0, ANIMAL_LAST_BUY["SHEEP"]),
    "hires": (0, 27),
    "land": (5, 22),
    "melon_floor": (8, 26),
    "premium_floor_scale": (6, 26),
    "staple_floor_scale": (6, 26),
}


# Season-long knob arms: (knob, value) deviations from planner.KNOB_DEFAULTS,
# applied for the whole game; context = the day-6 state (shops revealed).
KNOB_ARMS_R1 = [  # round 1, twin = champ-h3 (kept for the record)
    ("straw_target", 21), ("straw_target", 27), ("straw_target", 40),
    ("cows", 6), ("cows", 10), ("sheep", 6), ("sheep", 10), ("geese", 2), ("geese", 3),
    ("wheat_engine", 24), ("wheat_engine", 40), ("carrot_standing", 8), ("carrot_standing", 16),
    ("tomato_standing", 6), ("hires_delta", -2), ("hires_delta", 2),
    ("premium_scale", 0.7), ("premium_scale", 1.3), ("carrot_day", 18), ("carrot_day", 25),
]
# Round 2, twin = champ-h4 (premium 0.7, carrot day 25 are now the defaults).
KNOB_ARMS = [
    ("premium_scale", 0.5), ("premium_scale", 0.85),
    ("carrot_day", 24), ("carrot_day", 26),
    ("straw_target", 36), ("straw_target", 40),
    ("straw_early", 12), ("straw_early", 24),
    ("wheat_engine", 28),
    ("melon_opening", 8), ("melon_opening", 16),
    ("land2_margin", 1500), ("land2_margin", 4000),
    ("wheat_floor", 20), ("wheat_floor", 30),
    ("liquidation_day", 26), ("liquidation_day", 28),
    ("carrot_cap", 12), ("carrot_cap", 30),
    ("feed_days", 1), ("feed_days", 3),
]
KNOB_PRODUCT = {"straw_target": "STRAWBERRY", "cows": "MILK", "sheep": "WOOL", "geese": "EGG",
                "wheat_engine": "WHEAT", "carrot_standing": "CARROT", "tomato_standing": "TOMATO"}
KNOB_OUT = "rl/datasets/bandit/knob_samples.jsonl"


def _knob_world_jobs(rng, worlds, k, bundles, live=None):
    sessions = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    train_sessions = [p for p in sessions if int(p.split("-")[-1][:-5]) % 5 != 0]
    arms = [a for a in KNOB_ARMS if live is None or a in live]
    jobs = []
    for w in range(worlds):
        roll = rng.random()
        if roll < 0.6 and train_sessions:
            path = rng.choice(train_sessions); script_side = rng.randrange(2)
            opp = ("session", path, script_side); side = 1 - script_side; seed = 0
        elif roll < 0.9 and bundles:
            opp = ("bundle", rng.choice(bundles)); side = rng.randrange(2); seed = 300_000 + rng.randrange(10**6)
        else:
            opp = "heuristic"; side = rng.randrange(2); seed = 300_000 + rng.randrange(10**6)
        base = {"weights_path": None, "seed": seed, "side": side, "opponent": opp,
                "mode": "bc", "style": "spec", "world": w}
        jobs.append(base)
        for _ in range(k):
            knob, val = rng.choice(arms)
            jobs.append(dict(base, mode="knob", dev={knob: val}, arm=(knob, val)))
    return jobs


def _live_knob_arms(min_n=30):
    if not os.path.exists(KNOB_OUT):
        return None
    by = defaultdict(list)
    for line in open(KNOB_OUT):
        if line.strip():
            r = json.loads(line); by[(r["head"], r["offset"])].append(r["gain_margin"])
    live = set()
    for arm in KNOB_ARMS:
        g = by.get(arm, [])
        if len(g) >= min_n:
            m, se, n = _stats(g)
            if m + 2 * se < 0:
                continue
        live.add(arm)
    return live


def collect_knobs(args):
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    bundles = tuple(b for b in (args.bundles or "").split(",") if b)
    os.makedirs(os.path.dirname(KNOB_OUT), exist_ok=True)
    total = 0
    with Pool(args.procs) as pool, open(KNOB_OUT, "a") as f:
        for batch in range(args.batches):
            live = _live_knob_arms()
            jobs = _knob_world_jobs(rng, args.worlds, args.k, bundles, live)
            t0 = time.time()
            outs = pool.map(rollout.run_episode, jobs)
            twins = {j["world"]: o for j, o in zip(jobs, outs) if j["mode"] == "bc"}
            gains = []
            for j, o in zip(jobs, outs):
                if j["mode"] != "knob":
                    continue
                tw = twins[j["world"]]; ex = o.get("extra", {})
                if "dev_x" not in ex:
                    continue
                knob, val = j["arm"]
                margin = o["final_money"] - o["opp_money"]; base_margin = tw["final_money"] - tw["opp_money"]
                row = {"head": knob, "offset": val, "off_idx": [a for a in KNOB_ARMS if a[0] == knob].index((knob, val)),
                       "day": rollout.KNOB_DAY, "applied": True, "noop": False,
                       "gain_margin": margin - base_margin, "gain_money": o["final_money"] - tw["final_money"],
                       "final": o["final_money"], "opp": o["opp_money"], "base_final": tw["final_money"], "base_opp": tw["opp_money"],
                       "opponent": j["opponent"] if isinstance(j["opponent"], str) else list(j["opponent"]),
                       "side": j["side"], "seed": j["seed"], "ctx": ex["dev_ctx"], "x": [round(v, 5) for v in ex["dev_x"]]}
                f.write(json.dumps(row) + "\n"); gains.append(row["gain_margin"]); total += 1
            f.flush()
            live_n = len(live) if live is not None else len(KNOB_ARMS)
            print(f"batch {batch + 1}/{args.batches}: {len(jobs)} games in {time.time() - t0:.0f}s, "
                  f"{len(gains)} samples (total {total}), mean gain {sum(gains) / max(1, len(gains)):+,.0f}, live arms {live_n}", flush=True)


def _zero_index(vals):
    if 0 in vals:
        return vals.index(0)
    if 1.0 in vals:
        return vals.index(1.0)
    return None


def _live_arms(min_n=25):
    """Successive elimination: an arm (head, offset index) is retired once it
    has min_n samples and either every gain was exactly zero (inert) or its
    mean + 2 s.e. is below zero (clearly harmful). Returns the live set, or
    None if no data yet."""
    if not os.path.exists(OUT):
        return None
    by = defaultdict(list)
    for line in open(OUT):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("noop"):
            continue
        by[(r["head"], r["off_idx"])].append(r["gain_margin"])
    live = set()
    for h, (name, vals) in enumerate(features.HEADS):
        zi = _zero_index(vals)
        for off in range(len(vals)):
            if off == zi:
                continue
            g = by.get((name, off), [])
            if len(g) >= min_n:
                m, se, n = _stats(g)
                if all(v == 0 for v in g) or m + 2 * se < 0:
                    continue
            live.add((h, off))
    return live


def _world_jobs(rng, worlds, k, bundles):
    """Per world: one twin (champion plan) + k single-deviation games."""
    sessions = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    train_sessions = [p for p in sessions if int(p.split("-")[-1][:-5]) % 5 != 0]
    heads = features.HEADS
    live = _live_arms()
    if live is not None:
        arms = sorted(live)
        print(f"   live arms: {len(arms)}", flush=True)
    else:
        arms = [(h, off) for h, (name, vals) in enumerate(heads)
                for off in range(len(vals)) if off != _zero_index(vals)]
    jobs = []
    for w in range(worlds):
        roll = rng.random()
        if roll < 0.6 and train_sessions:
            path = rng.choice(train_sessions)
            script_side = rng.randrange(2)
            opp = ("session", path, script_side)
            side = 1 - script_side
            seed = 0  # overridden by the session's true seed
        elif roll < 0.9 and bundles:
            opp = ("bundle", rng.choice(bundles))
            side = rng.randrange(2)
            seed = 200_000 + rng.randrange(10**6)
        else:
            opp = "heuristic"
            side = rng.randrange(2)
            seed = 200_000 + rng.randrange(10**6)
        base = {"weights_path": None, "seed": seed, "side": side, "opponent": opp,
                "mode": "bc", "style": "spec", "world": w}
        jobs.append(base)
        for _ in range(k):
            h, off = rng.choice(arms)
            name, vals = heads[h]
            lo, hi = HEAD_DAYS[name]
            day = rng.randint(lo, hi)
            jobs.append(dict(base, mode="deviate", dev=(day, h, off)))
    return jobs


def collect(args):
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    bundles = tuple(b for b in (args.bundles or "").split(",") if b)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    heads = features.HEADS
    total = 0
    with Pool(args.procs) as pool, open(OUT, "a") as f:
        for batch in range(args.batches):
            jobs = _world_jobs(rng, args.worlds, args.k, bundles)
            t0 = time.time()
            outs = pool.map(rollout.run_episode, jobs)
            twins = {}
            for j, o in zip(jobs, outs):
                if j["mode"] == "bc":
                    twins[j["world"]] = o
            gains = []
            for j, o in zip(jobs, outs):
                if j["mode"] != "deviate":
                    continue
                tw = twins[j["world"]]
                ex = o.get("extra", {})
                if "dev_x" not in ex:
                    continue  # deviation day never reached (should not happen)
                day, h, off = j["dev"]
                name, vals = heads[h]
                margin = o["final_money"] - o["opp_money"]
                base_margin = tw["final_money"] - tw["opp_money"]
                row = {
                    "head": name, "offset": vals[off], "off_idx": off, "day": day,
                    "applied": ex.get("dev_applied", True),
                    "noop": ex.get("dev_noop", False),
                    "gain_margin": margin - base_margin,
                    "gain_money": o["final_money"] - tw["final_money"],
                    "final": o["final_money"], "opp": o["opp_money"],
                    "base_final": tw["final_money"], "base_opp": tw["opp_money"],
                    "opponent": j["opponent"] if isinstance(j["opponent"], str) else list(j["opponent"]),
                    "side": j["side"], "seed": j["seed"],
                    "ctx": ex["dev_ctx"], "x": [round(v, 5) for v in ex["dev_x"]],
                }
                f.write(json.dumps(row) + "\n")
                gains.append(row["gain_margin"])
                total += 1
            f.flush()
            mean = sum(gains) / max(1, len(gains))
            print(f"batch {batch + 1}/{args.batches}: {len(jobs)} games in {time.time() - t0:.0f}s, "
                  f"{len(gains)} samples (total {total}), mean gain {mean:+,.0f}", flush=True)


SRC = {"path": OUT}


def _load():
    return [json.loads(l) for l in open(SRC["path"]) if l.strip()]


def _stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, n - 1))
    return m, sd / math.sqrt(n), n


HEAD_PRODUCT = {
    "melon_tiles": "MELON", "wheat_tiles": "WHEAT", "carrot_tiles": "CARROT",
    "tomato_tiles": "TOMATO", "straw_tiles": "STRAWBERRY", "goose_target": "EGG",
    "cow_target": "MILK", "sheep_target": "WOOL",
}


def report(args):
    rows = _load()
    noops = sum(1 for r in rows if r.get("noop"))
    print(f"{len(rows)} samples ({noops} no-op deviations excluded from the tables)")
    rows = [r for r in rows if not r.get("noop")]
    by = defaultdict(list)
    for r in rows:
        by[(r["head"], r["offset"])].append(r)
    print(f"\n{'head':<20} {'offset':>7} {'n':>5} {'mean gain':>10} {'se':>7}   by day: early(<10) / mid(10-19) / late(20+)")
    for (head, off), rs in sorted(by.items(), key=lambda kv: -_stats([r['gain_margin'] for r in kv[1]])[0]):
        g = [r["gain_margin"] for r in rs]
        m, se, n = _stats(g)
        buckets = []
        for lo, hi in ((0, 10), (10, 20), (20, 30)):
            sub = [r["gain_margin"] for r in rs if lo <= r["day"] < hi]
            bm, bse, bn = _stats(sub)
            buckets.append(f"{bm:+6,.0f}({bn})" if bn else "     -   ")
        print(f"{head:<20} {off!s:>7} {n:>5} {m:>+10,.0f} {se:>7,.0f}   " + " / ".join(buckets))
    # context splits: demand for the head's product
    print("\nsplits by open consumers of the head's product (0 / 1-2 / 3+):")
    for (head, off), rs in sorted(by.items()):
        prod = HEAD_PRODUCT.get(head) or KNOB_PRODUCT.get(head)
        if not prod or len(rs) < 12:
            continue
        parts = []
        for lo, hi in ((0, 1), (1, 3), (3, 99)):
            sub = [r["gain_margin"] for r in rs if lo <= r["ctx"]["consumers"].get(prod, 0) < hi]
            bm, bse, bn = _stats(sub)
            parts.append(f"{bm:+6,.0f}±{bse:,.0f}({bn})" if bn >= 3 else "      -      ")
        print(f"  {head:<20} {off!s:>5}: " + " / ".join(parts))


def fit(args):
    """Per-(head, offset) ridge regression gain ~ x; writes rules with the
    coefficient vector, intercept, training mean/se, so BanditPolicy can
    predict the gain of every offset in a state."""
    import numpy as np

    rows = [r for r in _load() if r.get("applied", True) and not r.get("noop")]
    by = defaultdict(list)
    for r in rows:
        by[(r["head"], r["off_idx"])].append(r)
    rules = {"heads": [[n, v] for n, v in features.HEADS], "models": []}
    for (head, off_idx), rs in sorted(by.items()):
        if len(rs) < args.min_n:
            continue
        X = np.array([r["x"] for r in rs], dtype=float)
        y = np.array([r["gain_margin"] for r in rs], dtype=float)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        lam = args.ridge * len(rs)
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        pred = Z @ w + y.mean()
        r2 = 1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())
        m, se, n = _stats(list(y))
        rules["models"].append({
            "head": head, "off_idx": off_idx, "offset": rs[0]["offset"], "n": n, "mean": m, "se": se, "r2": round(r2, 3),
            "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(y.mean()),
        })
        print(f"  {head:<20} off_idx={off_idx} n={n:<4} mean={m:+7,.0f} se={se:5,.0f} r2={r2:.3f}")
    rules["min_gain"] = args.min_gain
    rules["kind"] = "knobs" if args.knobs else "offsets"
    with open(RULES, "w") as f:
        json.dump(rules, f)
    print(f"wrote {RULES} ({len(rules['models'])} models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "report", "fit"])
    ap.add_argument("--knobs", action="store_true", help="season-long knob arms (KNOB_ARMS) instead of single-day offsets")
    ap.add_argument("--tag", default=None, help="knob sample file tag: rl/datasets/bandit/knob_<tag>.jsonl")
    ap.add_argument("--arms", default=None,
                    help="focused collection: comma list knob:value (values parsed as numbers), replaces KNOB_ARMS")
    ap.add_argument("--worlds", type=int, default=30, help="worlds per batch")
    ap.add_argument("--k", type=int, default=4, help="deviations per world")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bundles", default="arena/agents/champ-h3/main.py,arena/agents/champ-h2/main.py")
    ap.add_argument("--min-n", type=int, default=25)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--min-gain", type=float, default=1500.0)
    args = ap.parse_args()
    global KNOB_OUT, KNOB_ARMS
    if args.arms:
        parsed = []
        for spec in args.arms.split(","):
            knob, val = spec.split(":")
            val = float(val) if "." in val else int(val)
            parsed.append((knob, val))
        KNOB_ARMS = parsed
    if args.knobs:
        if args.tag:
            KNOB_OUT = f"rl/datasets/bandit/knob_{args.tag}.jsonl"
        SRC["path"] = KNOB_OUT
        if args.cmd == "collect":
            return collect_knobs(args)
    {"collect": collect, "report": report, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    main()
