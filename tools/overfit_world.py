"""Overfit ONE world against the tapes, then read off where the gains are.

Every experiment in this project fights a $34,749 per-world standard deviation.
Fix the world AND the opponent -- a recorded session supplies both -- and the
environment becomes DETERMINISTIC: the same action sequence returns the same
bank, to the dollar (verified, three identical runs of one setup returned
132,321.0). So a search here has NO evaluation noise, no winner's curse, and no
paired-variance tricks. It is also the cheapest thing we run: script-vs-script
is 1.09 s/game against 4.08 s for a live agent.

This is a DIAGNOSTIC OVERFIT, not a submission. The point is not the score --
it is to find out whether beating the tape is possible at all in our action
space, and if so, WHERE the gains sit. A tape is a fixed 719-step script: in
its own world it is optimal by construction, but in any other world its actions
are conditioned on a board that is not this one. Those divergence points are
the only thing an adaptive agent can exploit, and nothing so far has shown us
where they are.

Search: greedy day-wise splice. For each day, try every cluster arm's actions
for that day in place of the current ones and keep the best. The 3029.3 cluster
has byte-identical field actions for days 0-5, so early splices are lossless by
construction; later ones are not, and the search simply measures what they are
worth.

READING THE RESULT. Every gain must be sorted into reachable (the state at that
moment already implied the better arm, so a rule could find it) and hindsight
(only knowable after the fact). Only the first kind can ship. Reporting the
total without that split would repeat the $18k day-6 oracle and the value head
that preferred losing arms.

    PYTHONPATH=. uv run python tools/overfit_world.py --worlds 3 --passes 3
"""

import argparse
import copy
import json
import os
import statistics as st
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

TURNS = 24
DAYS = 30
_ARMS = None
_SESS = None


def _load_arms(indices):
    from tools.tape_agent import _load_lib
    lib = _load_lib()
    return [(i, lib[i]["actions"]) for i in indices]


def _eval(task):
    """Exact final margin for one spliced action sequence. Deterministic."""
    actions, seed, opp_actions = task
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([make_script_agent(actions), make_script_agent(opp_actions)])
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return -1e18
    return float(f[0] - f[1])


def splice(base, arm_actions, day, mode="market"):
    """Replace one day's turns in `base` with `arm_actions`'s.

    MODE MATTERS, and the first run of this tool proved why. Two cluster arms
    share byte-identical actions through turn 287 (day 11) and diverge after.
    Splicing a foreign arm's FIELD actions past that point lands them on the
    wrong tiles, because the unit positions have diverged -- the swap is not a
    strategy change, it is desync. A full-splice search found exactly one gain,
    +$308, at day 11: the last day the trajectories still agree.

    MARKET orders are farm-level and carry no coordinates, so splicing them
    cannot desync, and they carry the whole economy -- HIRE is the labour
    supply, BUY_SEED is everything planted, SELL is all revenue. That is the
    honest search space here.
    """
    out = list(base)
    lo, hi = day * TURNS, min((day + 1) * TURNS, len(base))
    for t in range(lo, hi):
        if t >= len(arm_actions):
            break
        if mode == "full":
            out[t] = arm_actions[t]
        elif mode == "market":
            turn = dict(out[t])
            turn["market"] = arm_actions[t].get("market", [])
            out[t] = turn
        else:                                   # field only
            turn = dict(arm_actions[t])
            turn["market"] = out[t].get("market", [])
            out[t] = turn
    return out


def overfit(sess, arms, passes, pool, quiet=False, mode="market"):
    seed, opp = sess["seed"], sess["actions"][1]
    # start from whichever arm is best on THIS world, so the search measures
    # what adaptation adds beyond simply picking the right fixed script
    scores = pool.map(_eval, [(a, seed, opp) for _, a in arms])
    k = max(range(len(arms)), key=lambda i: scores[i])
    cur, cur_s = list(arms[k][1]), scores[k]
    start_arm, start_s = arms[k][0], scores[k]
    if not quiet:
        print(f"  best fixed arm on this world: cl-{start_arm} "
              f"margin {start_s:>+10,.0f}", flush=True)

    switches = []
    for p in range(passes):
        improved = False
        for d in range(DAYS):
            cands = [splice(cur, a, d, mode) for _, a in arms]
            sc = pool.map(_eval, [(c, seed, opp) for c in cands])
            j = max(range(len(sc)), key=lambda i: sc[i])
            if sc[j] > cur_s + 1e-9:
                gain = sc[j] - cur_s
                cur, cur_s = cands[j], sc[j]
                switches.append({"day": d, "arm": arms[j][0], "gain": gain,
                                 "pass": p})
                improved = True
        if not quiet:
            print(f"  pass {p+1}: margin {cur_s:>+10,.0f}  "
                  f"({cur_s - start_s:>+9,.0f} over the best fixed arm)",
                  flush=True)
        if not improved:
            break
    return {"seed": seed, "episode": sess.get("episode"),
            "start_arm": start_arm, "start": start_s, "end": cur_s,
            "gain": cur_s - start_s, "switches": switches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--mode", default="market",
                    choices=["market", "field", "full"],
                    help="market: farm-level orders only, cannot "
                         "desync. field/full: desync after day 11.")
    ap.add_argument("--arms", default="5,37,46,55,63,80,121,129,144,152,171,212,257,262")
    ap.add_argument("--out", default=os.path.join(REPO, "rl", "overfit.json"))
    args = ap.parse_args()

    import glob
    from tools.raw_bc import cluster_episodes
    eps = cluster_episodes(3029.3)
    files = [f for f in sorted(glob.glob(
        os.path.join(REPO, "rl", "datasets", "sessions", "session-*.json")))
        if int(f.split("-")[-1][:-5]) in eps][:args.worlds]
    arms = _load_arms([int(x) for x in args.arms.split(",")])
    print(f"{len(files)} worlds x {len(arms)} arms, {args.passes} passes, "
          f"mode={args.mode}, deterministic (no evaluation noise)\n")

    out = []
    t0 = time.time()
    with Pool(args.procs) as pool:
        for n, f in enumerate(files):
            sess = json.load(open(f))
            print(f"world {n+1}/{len(files)}  ep {sess.get('episode')}", flush=True)
            r = overfit(sess, arms, args.passes, pool, mode=args.mode)
            out.append(r)
            print(f"  -> gain {r['gain']:>+10,.0f} over the best fixed arm, "
                  f"{len(r['switches'])} switches\n", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    gains = [r["gain"] for r in out]
    print(f"{'='*64}")
    print(f"gain over the best FIXED arm, per world: "
          f"{', '.join(f'{g:+,.0f}' for g in gains)}")
    print(f"mean {st.mean(gains):+,.0f}   median {st.median(gains):+,.0f}")
    byday = {}
    for r in out:
        for s in r["switches"]:
            byday.setdefault(s["day"], []).append(s["gain"])
    print(f"\n  {'day':>4}{'switches':>10}{'total gain':>13}")
    for d in sorted(byday):
        print(f"  {d:>4}{len(byday[d]):>10}{sum(byday[d]):>13,.0f}")
    print(f"\n  {time.time()-t0:.0f}s wall. Wrote {args.out}")
    print("  NEXT: the reachable/hindsight split -- is the chosen arm "
          "predictable\n  from the day's observable state? Only then can it ship.")


if __name__ == "__main__":
    main()
