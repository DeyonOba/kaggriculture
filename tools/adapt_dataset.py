"""Is the c-chain interaction PREDICTABLE from observable state, or just noise?

`c-chain` gated +$8,072 +/- 1,843 against pub-ahmed-v38 and -$5,891 +/- 2,407
against pub-tschinkel-psr. If a policy could tell the two situations apart at
decision time it would capture roughly the spread rather than the pooled mean.
That is the whole premise of an adaptive executor.

The trap to avoid is the one the day-6 tape selector fell into: its per-world
oracle looked worth +$41,666/game and an interaction-shuffled null scored
+$41,951 +/- 745, i.e. which tape wins in which world was pure noise and the
apparent headroom was fake. So this builds the dataset and then runs the SAME
null.

Design:

  * features are read from the BASELINE (champ-h8) game at `--cut` day, which is
    exactly what a deployed agent would observe before deciding whether to
    switch -- it plays baseline behaviour up to the cut, then chooses.
  * the target is the PAIRED difference, cand reward - base reward, on the
    identical (opponent, seed, swap) world.
  * "which opponent is it" is deliberately NOT a feature. A 3-class lookup
    cannot generalise to a 9,000-team leaderboard; only a STATE feature can.
  * the null shuffles the target WITHIN opponent groups, which destroys any
    state->outcome link while preserving the opponent main effect. A rule that
    does not beat that null is fitting noise.

    PYTHONPATH=. uv run python tools/adapt_dataset.py build --cand c-chain --n 24
    PYTHONPATH=. uv run python tools/adapt_dataset.py fit --cand c-chain
"""

import argparse
import collections
import json
import os
import random
import statistics as st
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OPPS = ["pub-ahmed-v38", "pub-tschinkel-psr", "pub-boatlee-v16"]
OUT = os.path.join(REPO, "rl", "datasets", "adapt")
ITEMS = ["MELON", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG",
         "WHEAT", "FERTILIZER"]


def _features(obs, seat, cut_day):
    """Everything a deployed agent could legally see at the cut."""
    f = {}
    me, opp = obs["farms"][seat], obs["farms"][1 - seat]
    for tag, farm in (("me", me), ("op", opp)):
        plants = collections.Counter()
        animals = collections.Counter()
        unlocked = 0
        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("unlocked"):
                    unlocked += 1
                if t.get("kind") == "PLANT":
                    plants[t.get("crop")] += 1
                if t.get("animal"):
                    animals[t.get("animal")] += 1
        f[f"{tag}_money"] = float(farm.get("money", 0))
        f[f"{tag}_hands"] = 1.0 + len(farm.get("hands") or [])
        f[f"{tag}_land"] = float(unlocked)
        f[f"{tag}_plants"] = float(sum(plants.values()))
        f[f"{tag}_animals"] = float(sum(animals.values()))
        for a in ("COW", "SHEEP", "GOOSE"):
            f[f"{tag}_{a.lower()}"] = float(animals[a])
        for c in ("MELON", "WHEAT", "STRAWBERRY", "CARROT", "TOMATO"):
            f[f"{tag}_{c.lower()}"] = float(plants[c])
    mk = obs.get("market") or {}
    for it in ITEMS:
        f[f"px_{it.lower()}"] = float((mk.get("prices") or {}).get(it, 0))
    f["shops"] = float(len((obs.get("town") or {}).get("unlocked_shops", [])))
    f["day"] = float(cut_day)
    # Derived: the farms' relative posture, which is what an adaptive rule wants.
    f["d_animals"] = f["op_animals"] - f["me_animals"]
    f["d_plants"] = f["op_plants"] - f["me_plants"]
    f["d_money"] = f["op_money"] - f["me_money"]
    f["d_hands"] = f["op_hands"] - f["me_hands"]
    return f


def _play(job):
    agent, opp, seed, swap, cut = job
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    a = os.path.join(REPO, "arena", "agents", agent, "main.py")
    b = os.path.join(REPO, "arena", "agents", opp, "main.py")
    specs = [b, a] if swap else [a, b]
    seat = 1 if swap else 0
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run(specs)
    me = env.steps[-1][seat]
    if me["status"] != "DONE" or me["reward"] is None:
        return None
    feats = None
    for stp in env.steps:
        o = stp[0].observation
        if o["day"] >= cut:
            feats = _features(o, seat, o["day"])
            break
    return {"agent": agent, "opp": opp, "seed": seed, "swap": swap,
            "reward": float(me["reward"]), "features": feats}


def build(args):
    seeds = [args.base_seed + k for k in range(args.n)]
    jobs = [(a, o, s, w, args.cut)
            for a in (args.base, args.cand)
            for o in OPPS for s in seeds for w in (False, True)]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.cand}_vs_{args.base}.jsonl")
    done = 0
    with open(path, "w") as fh, Pool(args.procs) as pool:
        for rec in pool.imap_unordered(_play, jobs, chunksize=2):
            done += 1
            if rec:
                fh.write(json.dumps(rec) + "\n")
            if done % 50 == 0:
                print(f"   {done}/{len(jobs)}", flush=True)
    print(f"wrote {path}")


def fit(args):
    path = os.path.join(OUT, f"{args.cand}_vs_{args.base}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by[(r["agent"], r["opp"], r["seed"], r["swap"])] = r
    data = []
    for k, r in by.items():
        if k[0] != args.cand:
            continue
        b = by.get((args.base,) + k[1:])
        if b is None or not b["features"]:
            continue
        feat = dict(b["features"])
        feat["__seed__"] = k[2]          # carried for the held-out split only
        data.append((b["opp"], feat, r["reward"] - b["reward"]))
    if not data:
        sys.exit("no paired rows; run build first")

    diffs = [d for _, _, d in data]
    print(f"{len(data)} paired games   mean {st.mean(diffs):+,.0f} "
          f"+/- {st.stdev(diffs)/len(diffs)**0.5:,.0f}\n")
    print("by opponent:")
    g = collections.defaultdict(list)
    for o, _, d in data:
        g[o].append(d)
    for o, v in sorted(g.items()):
        print(f"   {o:<22} n={len(v):>3}  {st.mean(v):>+9,.0f} "
              f"+/- {st.stdev(v)/len(v)**0.5:>6,.0f}")

    names = sorted(n for n in data[0][1] if not n.startswith("__"))

    def best_rule(rows_):
        """Best single-feature threshold rule: switch when f >= t, else don't.

        Value = mean gain captured, counting 0 where the rule declines to switch.
        """
        best = (None, None, -1e18)
        for nm in names:
            vals = sorted({r[1][nm] for r in rows_})
            if len(vals) < 2:
                continue
            for t in vals:
                v = st.mean(d if r >= t else 0.0
                            for (_, f, d), r in ((x, x[1][nm]) for x in rows_))
                if v > best[2]:
                    best = (nm, t, v)
        return best

    nm, t, val = best_rule(data)
    always = max(0.0, st.mean(diffs))
    print(f"\nbest single-feature rule: switch when {nm} >= {t:,.1f}"
          f"   captures {val:+,.0f}/game")
    print(f"always-switch baseline:                        {always:+,.0f}/game")

    # Interaction null: shuffle the target WITHIN opponent, destroying any
    # state->outcome link while keeping the opponent main effect intact.
    rng = random.Random(args.seed)
    nulls = []
    for _ in range(args.null):
        pools = {o: [d for oo, _, d in data if oo == o] for o in g}
        for v in pools.values():
            rng.shuffle(v)
        idx = collections.Counter()
        shuffled = []
        for o, f, _ in data:
            shuffled.append((o, f, pools[o][idx[o]]))
            idx[o] += 1
        nulls.append(best_rule(shuffled)[2])
    m, s = st.mean(nulls), (st.stdev(nulls) if len(nulls) > 1 else 0.0)
    print(f"\ninteraction-null ({args.null} shuffles):        {m:+,.0f} +/- {s:,.0f}")
    z = (val - m) / s if s else 0.0
    print(f"real rule beats the null by {val - m:+,.0f}  (z = {z:.2f})")

    # OUT-OF-SAMPLE. The null controls for threshold-selection bias, but the
    # in-sample number is still fitted on the games it scores. The day-6 tape
    # selector looked like +$41,666/game in-sample and was noise. Fit the rule
    # on one half of the SEEDS and score it on the other, both directions.
    halves = [[r for r in data if r[1]["__seed__"] % 2 == p] for p in (0, 1)]
    oos = []
    for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
        if not tr or not te:
            continue
        fn, ft, _ = best_rule(tr)
        got = st.mean(d if f[fn] >= ft else 0.0 for _, f, d in te)
        oos.append(got)
        print(f"\n  fit on {len(tr):>3} games -> rule '{fn} >= {ft:,.1f}'"
              f"   scores {got:+,.0f}/game on the {len(te)} held-out games")
    if oos:
        print(f"\nout-of-sample mean: {st.mean(oos):+,.0f}/game   "
              f"(in-sample claimed {val:+,.0f})")

    # Require EVERY fold positive, not just the mean. Folds that disagree on
    # which feature to use are overfitting, however good the average looks.
    ok = z >= 2 and len(oos) >= 2 and all(v > 0 for v in oos)
    print("\n" + ("SIGNAL -- a state rule predicts when the change pays, "
                  "and every held-out fold is positive"
                  if ok else
                  "NO USABLE SIGNAL -- beats the null in-sample but the "
                  "held-out folds disagree; that is overfitting, exactly like "
                  "the day-6 tape selector"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--cand", required=True)
        p.add_argument("--base", default="champ-h8")
        p.add_argument("--n", type=int, default=24)
        p.add_argument("--base-seed", type=int, default=64000)
        p.add_argument("--cut", type=int, default=8,
                       help="day at which the adaptive decision would be made")
        p.add_argument("--procs", type=int, default=6)
        p.add_argument("--null", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (build if a.cmd == "build" else fit)(a)
