"""Is the tape's action already the best one at a given turn?

The day-splice overfit could only choose among 14 canned recordings, and it
found almost nothing reachable: 61% of its gain was hindsight (day-0 states are
byte-identical across worlds) and 33% was endgame liquidation the shipped agent
already does harder. But it never asked what a FREE action is worth.

This does. On a fixed world with a scripted opponent the game is DETERMINISTIC,
so changing exactly one (turn, unit) action and replaying gives the exact
counterfactual value of that one decision -- no noise, no paired tricks, no
winner's curse. That is the cleanest credit assignment available anywhere in
this project, and it is 1.09 s per evaluation.

Two modes:

  probe   sample K (turn, unit) pairs, try every LEGAL alternative at each, and
          report how often ANY single change beats the tape. This is an
          existence test: if the tape's action is already best almost
          everywhere, no adaptive agent has room, and we stop.
  climb   greedily apply the best improving mutation, re-deriving states after
          each acceptance, for N rounds.

Legality comes from tools/raw_bc.legal_set, the same guards the action audit
validated at 99.10% coverage of champion moves.

    PYTHONPATH=. uv run python tools/turn_mutate.py probe --samples 120
"""

import argparse
import json
import glob
import os
import random
import statistics as st
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

TURNS = 24


def _eval(task):
    actions, seed, opp = task
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([make_script_agent(actions), make_script_agent(opp)])
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return -1e18
    return float(f[0] - f[1])


def replay_states(actions, seed, opp):
    """-> per turn: (day, [(unit_idx, x, y)], observation) for legality."""
    from kaggle_environments import make
    from rl.session_bench import make_script_agent
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([make_script_agent(actions), make_script_agent(opp)])
    out = []
    for t in range(len(env.steps) - 1):
        o = env.steps[t][0].observation
        farm = o["farms"][0]
        units = [tuple(farm.get("farmer") or (0, 0))]
        units += [tuple(h) for h in (farm.get("hands") or [])]
        out.append((o, units))
    return out


def mutate(actions, turn, unit, op):
    """One action changed. Everything else identical."""
    out = list(actions)
    a = dict(out[turn])
    hands = list(a.get("hands") or [])
    if unit == 0:
        a["farmer"] = list(op)
    else:
        while len(hands) < unit:
            hands.append(["PASS"])
        if unit - 1 < len(hands):
            hands[unit - 1] = list(op)
        a["hands"] = hands
    out[turn] = a
    return out


def load_world(n=0):
    from tools.raw_bc import cluster_episodes
    eps = cluster_episodes(3029.3)
    files = [f for f in sorted(glob.glob(os.path.join(
        REPO, "rl", "datasets", "sessions", "session-*.json")))
        if int(f.split("-")[-1][:-5]) in eps]
    return json.load(open(files[n]))


def probe(args):
    from tools.tape_agent import _load_lib
    from tools.raw_bc import legal_set, key_of
    sess = load_world(args.world)
    seed, opp = sess["seed"], sess["actions"][1]
    lib = _load_lib()
    arms = [int(x) for x in args.arms.split(",")]

    with Pool(args.procs) as pool:
        base_scores = pool.map(_eval, [(lib[i]["actions"], seed, opp) for i in arms])
        k = max(range(len(arms)), key=lambda i: base_scores[i])
        base = list(lib[arms[k]]["actions"])
        base_s = base_scores[k]
        print(f"world ep {sess['episode']}  best fixed arm cl-{arms[k]}  "
              f"margin {base_s:>+10,.0f}\n", flush=True)

        states = replay_states(base, seed, opp)
        rng = random.Random(args.seed)
        # sample (turn, unit) pairs, biased to nothing -- uniform over the game
        pairs = []
        tries = 0
        while len(pairs) < args.samples and tries < args.samples * 40:
            tries += 1
            t = rng.randrange(min(len(states), len(base)))
            o, units = states[t]
            if not units:
                continue
            u = rng.randrange(len(units))
            cur = (base[t].get("farmer") if u == 0
                   else (base[t].get("hands") or [None] * 99)[u - 1])
            ck = key_of(cur) if cur else ("PASS",)
            try:
                L = legal_set(o, 0, u, *units[u])
            except Exception:
                continue
            alts = sorted(L - {ck})
            if alts:
                pairs.append((t, u, ck, alts))

        n_alts = sum(len(p[3]) for p in pairs)
        print(f"{len(pairs)} (turn, unit) pairs sampled, {n_alts} legal "
              f"alternatives to test  (~{n_alts*1.09/args.procs/60:.0f} min)\n",
              flush=True)

        tasks, index = [], []
        for t, u, ck, alts in pairs:
            for op in alts:
                tasks.append((mutate(base, t, u, op), seed, opp))
                index.append((t, u, ck, op))
        scores = pool.map(_eval, tasks)

    gains = [(s - base_s, *ix) for s, ix in zip(scores, index)]
    improving = [g for g in gains if g[0] > 1e-9]
    per_pair = {}
    for g, t, u, ck, op in gains:
        key = (t, u)
        per_pair[key] = max(per_pair.get(key, -1e18), g)
    pairs_with_gain = sum(1 for v in per_pair.values() if v > 1e-9)

    worse = [g for g in gains if g[0] < -1e-9]
    same = [g for g in gains if abs(g[0]) <= 1e-9]
    print(f"{'='*66}")
    print(f"  NOTE: changing ONE action in a SCRIPT desyncs every later action")
    print(f"  for that unit -- a policy would adapt, a script cannot. So the")
    print(f"  distribution matters more than the hit rate:\n")
    print(f"  {'better':<10}{len(improving):>8,}  ({100*len(improving)/max(1,len(gains)):>5.1f}%)")
    print(f"  {'NEUTRAL':<10}{len(same):>8,}  ({100*len(same)/max(1,len(gains)):>5.1f}%)"
          f"   <- the action did not matter at all")
    print(f"  {'worse':<10}{len(worse):>8,}  ({100*len(worse)/max(1,len(gains)):>5.1f}%)")
    if worse:
        ws = sorted(g[0] for g in worse)
        print(f"     cost of a wrong action: median {st.median(ws):>+9,.0f}  "
              f"worst {ws[0]:>+10,.0f}")
    print()
    print(f"  alternatives tested                 {len(gains):>8,}")
    print(f"  ...that IMPROVE the final margin    {len(improving):>8,} "
          f"({100*len(improving)/max(1,len(gains)):.1f}%)")
    print(f"  (turn, unit) pairs with ANY gain    {pairs_with_gain:>8,} / "
          f"{len(per_pair)}  ({100*pairs_with_gain/max(1,len(per_pair)):.0f}%)")
    if improving:
        gs = sorted((g[0] for g in improving), reverse=True)
        print(f"\n  gain from ONE changed action:")
        print(f"    best     {gs[0]:>+10,.0f}")
        print(f"    median   {st.median(gs):>+10,.0f}")
        print(f"    mean     {st.mean(gs):>+10,.0f}")
        print(f"\n  top single-action changes:")
        for g, t, u, ck, op in sorted(improving, key=lambda x: -x[0])[:8]:
            print(f"    day {t//TURNS:>2} turn {t:>3} unit {u:>2}  "
                  f"{str(ck):<22} -> {str(op):<22} {g:>+9,.0f}")
    print(f"\n  base margin {base_s:>+,.0f}. A high hit-rate means the tape's "
          f"action is\n  routinely beatable and an adaptive agent has room; a "
          f"low one means it does not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe"])
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="5,37,46,55,63,80,121,129,144,152,171,212,257,262")
    args = ap.parse_args()
    t0 = time.time()
    probe(args)
    print(f"\n  {time.time()-t0:.0f}s wall")


if __name__ == "__main__":
    main()
