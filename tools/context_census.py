"""What the world looks like on each day of each game, under the champion.

    PYTHONPATH=. uv run python tools/context_census.py --games 144 --procs 6
    -> rl/datasets/context_census.json

WHY THIS EXISTS. A reactive policy is a set of rules that fire on the observed
state, and a rule that fires in 4 games of 144 cannot be gated no matter how
good it is: at the instrument's standard error of ~0.045 an effect diluted over
the games where nothing happened is invisible, which is exactly how the
constant-vector search ended at +0.000 (memory: parametric tuning is closed).

So before any rule is played, its FIRING SET should be known. The context at
every decision day is a property of the world and of the champion's own play,
both fixed; recording it once makes every candidate rule's firing set a lookup
over this file instead of 144 games. The rule space can then be enumerated and
filtered for testability for free.

It also fixes the partition. Scoring a conditional rule on the games where it
fires is only honest if "it fired" is decided by something that does not depend
on the rule's own outcome. The baseline's context is that: pre-specified,
identical for every candidate, and computed before the rule changes anything.

The games are the same held-out band-stratified draw `rl/session_eval.py`
uses, in the same order, so a row here lines up with a row there by index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl.session_eval import EDGES, held_out_entries                     # noqa: E402

IDX = os.path.join(REPO, "rl", "datasets", "session_index.json")
OUT = os.path.join(REPO, "rl", "datasets", "context_census.json")

# Trimmed from features.context: what a rule may legally test. The board and
# money fields are kept because the opponent's farm is fully public and is the
# one thing in this game a fitted model cannot derive (kagent/risk.py: the
# opponent's policy is the only place estimation belongs).
KEEP = ("day", "money", "land", "hands", "consumers", "price_ratio",
        "crops", "herd", "opp_crops", "opp_herd", "opp_money", "opp_land",
        "shed_total")


def _worker_init():
    import tools.fast_env as fast_env
    fast_env.install()


def census_game(job):
    """One champion game; -> {day: context} plus the outcome."""
    from kaggle_environments import make
    from kagent import scheduler as _sched
    from kagent import theta as T
    from kagent import features
    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy, ThetaPlanProvider
    from rl import rollout

    path, script_side = job["path"], job["script_side"]
    spec = T.spec(T.champion())
    prev = dict(_sched.EXEC)
    _sched.reset_exec()
    _sched.set_exec(spec["exec"])

    mem = Memory()
    pol = HeuristicPolicy(ThetaPlanProvider(spec["knobs"]))
    ctx, seen = {}, set()

    def learner(obs):
        view = View(obs)
        mem.observe(view)
        # At HOUR 0, before the day's plan: this is the state a rule would be
        # evaluated on, and it must be read before the plan acts on it or the
        # condition is testing its own consequences.
        if view.day not in seen and view.step % 24 == 0:
            seen.add(view.day)
            c = features.context(view)
            ctx[str(view.day)] = {k: c[k] for k in KEEP if k in c}
        return pol.act(view, mem)

    seed = rollout.load_session(path)["seed"]
    opponent = rollout._make_opponent(("session", path, script_side))
    side = 1 - script_side
    agents = [learner, opponent] if side == 0 else [opponent, learner]
    try:
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run(agents)
    finally:
        _sched.EXEC.update(prev)
    final = env.steps[-1]
    mine, theirs = final[side].reward or 0.0, final[1 - side].reward or 0.0
    return {"path": path, "script_side": script_side, "bin": job["bin"],
            "rating": job["rating"], "ctx": ctx,
            "money": mine, "opp": theirs,
            "win": 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0),
            "bad": any(s.status != "DONE" for s in final)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=144)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--draw-seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    per_bin = max(1, args.games // len(EDGES))
    entries = held_out_entries(IDX, args.draw_seed, per_bin)
    jobs = [{"path": os.path.join(REPO, p), "script_side": ss, "bin": b, "rating": r}
            for p, ss, b, r in entries]
    print(f"{len(jobs)} champion games, draw seed {args.draw_seed}")

    with Pool(args.procs, initializer=_worker_init) as pool:
        rows = list(pool.imap(census_game, jobs))

    wins = sum(r["win"] for r in rows) / len(rows)
    print(f"champion win {wins:.3f}   bad {sum(r['bad'] for r in rows)}")
    with open(args.out, "w") as fh:
        json.dump({"draw_seed": args.draw_seed, "per_bin": per_bin,
                   "games": len(rows), "keep": list(KEEP),
                   "note": "context at HOUR 0 of each day under the champion, "
                           "on rl/session_eval.py's held-out draw, same order",
                   "rows": rows}, fh)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
