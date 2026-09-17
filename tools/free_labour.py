"""Free-labour oracle: can the farm turn more actions into more money at all?

The LP says labour binds on 29 of 30 days at a shadow price of $13/action
(peak $27). But the chaining experiment RAISED productive actions 9% and lost
$6,385, so "more actions" and "more money" are not the same thing.

Hiring more hands normally costs wages (fib) and a market-order slot, which
confounds the question -- the recorded +2 hands -$5,229 / +4 hands -$19,344
could be the wage bill, not the labour. This hands seat 0 N extra workers every
morning for FREE: no hire cost, no order slot, no workload gate. If money does
not rise, the farm cannot use more labour and routing has nothing to harvest.
If it rises steeply, labour really is the constraint and better routing is worth
chasing.
"""
import sys, os, collections, statistics as st
sys.path.insert(0, os.getcwd())
import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

EXTRA = [0]
_eod = K._end_of_day


def eod(state, env, day):
    r = _eod(state, env, day)
    n = EXTRA[0]
    if n:
        obs0 = state[0].observation
        board = len(obs0.farms[0]["tiles"])
        spawn = list(K._default_spawn(board))
        # seat 0 only; hands spawn at the shed exactly as hired ones do
        obs0.farms[0]["hands"] = [list(spawn) for _ in range(n)]
    return r


K._end_of_day = eod


def run(agent, opp, seeds, extra):
    EXTRA[0] = extra
    fin, acts = [], collections.Counter()
    for sd in seeds:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd},
                   debug=False)
        env.run([agent, opp])
        fin.append(env.steps[-1][0]["reward"] - env.steps[-1][1]["reward"])
        for stp in env.steps[:-1]:
            a = stp[0].get("action") or {}
            for x in [a.get("farmer")] + list(a.get("hands") or []):
                op = x[0] if isinstance(x, list) and x else "PASS"
                acts["u"] += 1
                if op == "PASS":
                    acts["p"] += 1
                elif op in ("NORTH", "SOUTH", "EAST", "WEST"):
                    acts["m"] += 1
                else:
                    acts["w"] += 1
    n = len(seeds)
    return (st.mean(fin), st.stdev(fin) / n ** .5 if n > 1 else 0,
            acts["u"] / n, acts["w"] / n, acts["p"] / n)


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "arena/agents/champ-h8/main.py"
    opp = "arena/agents/pub-ahmed-v38/main.py"
    seeds = [61000 + k for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6)]
    print(f"{os.path.basename(os.path.dirname(agent))} vs ahmed, {len(seeds)} seeds\n")
    print(f"{'free hands':>11} {'margin':>12} {'se':>8} {'unit-turns':>11} "
          f"{'productive':>11} {'PASS':>8} {'eff':>6}")
    base = None
    for extra in (0, 2, 4, 8, 16):
        m, se, u, w, p = run(agent, opp, seeds, extra)
        if base is None:
            base = m
        print(f"{extra:>11} {m:>+12,.0f} {se:>8,.0f} {u:>11,.0f} {w:>11,.0f} "
              f"{p:>8,.0f} {w/u:>6.3f}   {m-base:+,.0f}")
