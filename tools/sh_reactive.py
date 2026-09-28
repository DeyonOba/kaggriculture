"""Successive halving over REACTIVE rules: many rules, cheaply, then fewer deeply.

    PYTHONPATH=. uv run python tools/sh_reactive.py --slate 128 --procs 6

WHY NOT CEM. The constant-vector search used CEM because the space was
continuous and had a usable notion of "nearby". A rule is a discrete tuple --
(day, field, key, op, threshold, knob, value) -- and two rules that differ in
one field are not neighbours in any sense a Gaussian can refit. What this space
needs is a way to survey MANY candidates and spend the budget on the ones that
survive, which is successive halving: every rule gets a cheap look, the
survivors get a longer one, and only a handful ever see a full gate.

TWO THINGS MAKE THIS MUCH CHEAPER THAN THE CONSTANT SEARCH.

1. THE BASELINE IS ALREADY PLAYED. `tools/context_census.py` runs the champion
   once over the whole pool and records both the per-day context and the
   outcome. Every census row is therefore the paired baseline for that game,
   so a candidate costs only its own games rather than twice that. Play is
   temperature-0 deterministic, which is what makes the stored result a valid
   replay rather than an approximation.

2. ONLY FIRING GAMES ARE PLAYED. A rule that does not fire produces a plan
   bit-identical to the champion's, so the paired delta on that game is exactly
   zero -- it contributes no signal AND no variance, and including it changes
   the z-statistic not at all. (Worth stating plainly, because the opposite is
   easy to assume: gating on the firing subset is not a power trick, the z is
   the same either way.) What it buys is that a gate can be drawn where EVERY
   game fires, so the full instrument is spent on the contrast instead of
   half of it on guaranteed ties -- about 1.5x on z for a rule firing at 44%.

Rounds NEST: round 2's games include round 1's, so a survivor's earlier games
are reused rather than replayed. Games are drawn band-stratified from the
firing set, because the 2500+ band is the hardest and a pooled draw is a
top-light draw (memory: evaluate per BAND).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl.session_eval import EDGES, _label, binom_two_sided                # noqa: E402

CENSUS = os.path.join(REPO, "rl", "datasets", "context_census.json")


def _worker_init():
    import tools.fast_env as fast_env
    fast_env.install()


def play(job):
    """One reactive game. -> (index, win, money, opp, fired)."""
    from kaggle_environments import make
    from kagent import scheduler as _sched
    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy
    from kagent.reactive import ReactivePlanProvider
    from rl import rollout

    prev = dict(_sched.EXEC)
    _sched.reset_exec()
    _sched.set_exec(job["exec"])
    provider = ReactivePlanProvider(job["knobs"], job["rules"])
    pol = HeuristicPolicy(provider)
    mem = Memory()

    def learner(obs):
        view = View(obs)
        mem.observe(view)
        return pol.act(view, mem)

    path, ss = job["path"], job["script_side"]
    seed = rollout.load_session(path)["seed"]
    opponent = rollout._make_opponent(("session", path, ss))
    side = 1 - ss
    agents = [learner, opponent] if side == 0 else [opponent, learner]
    try:
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run(agents)
    finally:
        _sched.EXEC.update(prev)
    final = env.steps[-1]
    mine, theirs = final[side].reward or 0.0, final[1 - side].reward or 0.0
    return {"i": job["i"], "money": mine, "opp": theirs,
            "win": 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0),
            "fired": len(provider.fired)}


# ------------------------------------------------------------------ drawing
def rule_key(rule):
    return hashlib.sha1(json.dumps(rule, sort_keys=True).encode()).hexdigest()[:12]


def draw_games(rule, rows, n, fire_idx):
    """`n` firing games, band-stratified, in a DETERMINISTIC order.

    Deterministic so the rounds nest: the first 24 of this list are exactly
    what round 1 played, so round 2 pays only for the 24 it adds. Stratified so
    a rule is not judged mostly on the bands that are easiest to win.
    """
    by_bin = {}
    for i in fire_idx:
        by_bin.setdefault(rows[i]["bin"], []).append(i)
    rng = random.Random(int(rule_key(rule), 16) % (2 ** 31))
    for b in by_bin:
        by_bin[b].sort()
        rng.shuffle(by_bin[b])
    out, b_order = [], sorted(by_bin)
    while len(out) < n and any(by_bin[b] for b in b_order):
        for b in b_order:                      # round robin: even bands
            if by_bin[b] and len(out) < n:
                out.append(by_bin[b].pop())
    return out


def score(rows, played):
    """Paired against the census on exactly the games played."""
    d, b, c = [], 0, 0
    for i, r in played.items():
        base = rows[i]
        d.append(r["win"] - base["win"])
        if base["win"] < 0.75 <= r["win"]:
            b += 1
        elif r["win"] < 0.75 <= base["win"]:
            c += 1
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    var = sum((v - mean) ** 2 for v in d) / (n - 1) if n > 1 else float("nan")
    money = sum(r["money"] for r in played.values()) / max(1, n)
    opp = sum(r["opp"] for r in played.values()) / max(1, n)
    bm = sum(rows[i]["money"] for i in played) / max(1, n)
    bo = sum(rows[i]["opp"] for i in played) / max(1, n)
    return {"n": n, "delta": mean,
            "se": math.sqrt(var / n) if n > 1 else float("nan"),
            "p": binom_two_sided(b, b + c), "arm_only": b, "base_only": c,
            "money": money, "opp": opp, "d_money": money - bm, "d_opp": opp - bo,
            "fired_all": all(r["fired"] > 0 for r in played.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", default=CENSUS)
    ap.add_argument("--slate", type=int, default=128)
    ap.add_argument("--rounds", default="24,48,144")
    ap.add_argument("--keep", default="32,8")
    ap.add_argument("--min-fire", type=float, default=0.20)
    ap.add_argument("--max-fire", type=float, default=0.85)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "rl/runs/sh_reactive"))
    ap.add_argument("--rules", default=None,
                    help="a finalists.json to REPLAY instead of sampling a slate. "
                         "Point --census at a different draw seed and this is the "
                         "verdict: the same rules on worlds the search never saw.")
    args = ap.parse_args()

    from kagent import reactive as R, theta as T

    with open(args.census) as fh:
        census = json.load(fh)
    rows = census["rows"]
    os.makedirs(args.out, exist_ok=True)

    # Knob value ladders, mechanical from theta's own bounds: three interior
    # points plus the floor, so a rule can switch something ON or OFF but the
    # ladder is not a guess about which value is right.
    ladders = {}
    for name, lo, hi, kind in T.KNOB_SPEC:
        if kind == "bool":
            ladders[name] = [0, 1]
        elif kind == "int":
            vals = sorted({int(round(lo + (hi - lo) * q)) for q in (0.0, 0.25, 0.5, 0.75)})
            ladders[name] = [v for v in vals if v != T.champion()[name]]
        else:
            vals = sorted({round(lo + (hi - lo) * q, 3) for q in (0.1, 0.4, 0.7)})
            ladders[name] = [v for v in vals if v != T.champion()[name]]
    ladders = {k: v for k, v in ladders.items() if v}

    t0 = time.time()
    pool_rules = R.enumerate_rules(rows, ladders, args.min_fire, args.max_fire)
    print(f"{len(rows)} census games | {len(pool_rules):,} rules whose condition "
          f"fires {args.min_fire:.0%}-{args.max_fire:.0%} of the time "
          f"({time.time()-t0:.0f}s to enumerate)")

    # Sample the slate STRATIFIED BY KNOB so one knob with a long ladder does
    # not crowd out the rest -- the point of a survey is coverage.
    if args.rules:
        with open(args.rules) as fh:
            fin = json.load(fh)
        slate = [f["rule"] if "rule" in f else f for f in fin]
        # fire_share is recomputed against THIS census -- a rule's firing rate
        # is a property of the draw, and quoting the search draw's share on a
        # fresh one would be the first thing to mislead.
        for r in slate:
            r["fire_share"] = round(len(R.firing_set(r, rows)) / len(rows), 3)
        print(f"replaying {len(slate)} finalists on this census\n")
        sizes = [int(x) for x in args.rounds.split(",")][-1:]
        keeps = []
        _run(args, rows, slate, sizes, keeps, T, R)
        return

    rng = random.Random(args.seed)
    by_knob = {}
    for r in pool_rules:
        by_knob.setdefault(r["knob"], []).append(r)
    slate, knobs = [], sorted(by_knob)
    rng.shuffle(knobs)
    while len(slate) < args.slate and any(by_knob[k] for k in knobs):
        for k in knobs:
            if by_knob[k] and len(slate) < args.slate:
                slate.append(by_knob[k].pop(rng.randrange(len(by_knob[k]))))
    print(f"slate: {len(slate)} rules over {len({r['knob'] for r in slate})} knobs, "
          f"{len({(r['field'], r['key']) for r in slate})} distinct conditions\n")

    sizes = [int(x) for x in args.rounds.split(",")]
    keeps = [int(x) for x in args.keep.split(",")]
    _run(args, rows, slate, sizes, keeps, T, R)


def _run(args, rows, slate, sizes, keeps, T, R):
    base_ex = T.spec(T.champion())
    fire_cache = {rule_key(r): R.firing_set(r, rows) for r in slate}
    played = {rule_key(r): {} for r in slate}          # key -> {game index: result}
    alive = list(slate)
    trials = os.path.join(args.out, "trials.jsonl")

    with Pool(args.procs, initializer=_worker_init) as pool:
        for rnd, n_games in enumerate(sizes):
            jobs, spans = [], []
            for r in alive:
                k = rule_key(r)
                want = draw_games(r, rows, n_games, fire_cache[k])
                need = [i for i in want if i not in played[k]]
                spans.append((k, len(jobs), len(jobs) + len(need), want))
                jobs += [{"i": i, "path": rows[i]["path"],
                          "script_side": rows[i]["script_side"],
                          "knobs": base_ex["knobs"], "exec": base_ex["exec"],
                          "rules": [r]} for i in need]
            t1 = time.time()
            out = list(pool.imap(play, jobs)) if jobs else []
            for k, lo, hi, _want in spans:
                for res in out[lo:hi]:
                    played[k][res["i"]] = res

            scored = []
            for r in alive:
                k = rule_key(r)
                want = draw_games(r, rows, n_games, fire_cache[k])
                s = score(rows, {i: played[k][i] for i in want if i in played[k]})
                scored.append((s["delta"], s["d_money"] - s["d_opp"], k, r, s))
            scored.sort(reverse=True)

            noop = sum(1 for d, _m, _k, _r, sc in scored
                       if d == 0.0 and abs(sc["d_money"]) < 1.0)
            unfired = sum(1 for _d, _m, _k, _r, sc in scored if not sc["fired_all"])
            print(f"round {rnd}  {n_games} firing games a rule, {len(alive)} rules, "
                  f"{len(jobs)} games played ({time.time()-t1:.0f}s)   "
                  f"{noop} bit-identical, {unfired} did not fire everywhere")
            for d, _m, _k, r, s in scored[:6]:
                print(f"  {d:>+7.3f} +/-{s['se']:.3f} p {s['p']:.3f}  "
                      f"our {s['d_money']:>+7,.0f} their {s['d_opp']:>+7,.0f}  "
                      f"fire {r['fire_share']:.0%}  {R.describe(r)}")
            with open(trials, "a") as fh:
                for d, _m, k, r, s in scored:
                    fh.write(json.dumps({"round": rnd, "key": k, "rule": r,
                                         "score": s}) + "\n")
            if rnd >= len(keeps):
                break
            alive = [r for _d, _m, _k, r, _s in scored[:keeps[rnd]]]
            print(f"  -> {len(alive)} survive\n", flush=True)

    best = scored[0]
    os.makedirs(args.out, exist_ok=True)
    print(f"\nBEST  {best[0]:+.3f} +/- {best[4]['se']:.3f}  p {best[4]['p']:.3f}  "
          f"n {best[4]['n']}   {R.describe(best[3])}")
    print("This is the SEARCH draw. A finalist means nothing until it is "
          "re-gated on a census of a different draw seed.")
    with open(os.path.join(args.out, "finalists.json"), "w") as fh:
        json.dump([{"rule": r, "score": s} for _d, _m, _k, r, s in scored[:8]], fh, indent=1)
    print(f"-> {args.out}/finalists.json   {trials}")


if __name__ == "__main__":
    main()
