"""Session-replay tooling: fidelity gate and policy-vs-script benchmark.

Fidelity: replay BOTH recorded action scripts on the session's true seed and
compare local finals to the recorded finals — validates that the local engine
reproduces Kaggle's worlds before any training trusts them.

Benchmark: a policy bundle takes one seat against the other seat's script;
reports session win rate and profit uplift vs the seat's original occupant.

Usage:
    PYTHONPATH=. uv run python rl/session_bench.py fidelity --n 12
    PYTHONPATH=. uv run python rl/session_bench.py bench \
        --agent arena/agents/rl-v9/main.py --n 25 --procs 10
"""

import argparse
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESS_DIR = "rl/datasets/sessions"


def make_script_agent(actions):
    state = {"t": 0}

    def agent(obs):
        t = obs.get("step", state["t"])
        state["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def _load_bundle(path):
    env_dict = {}
    with open(path) as f:
        exec(f.read(), env_dict)
    return [v for v in env_dict.values() if callable(v)][-1]


def fidelity_job(path):
    from kaggle_environments import make

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([make_script_agent(s["actions"][0]), make_script_agent(s["actions"][1])])
    local = [st.reward or 0 for st in env.steps[-1]]
    local_shops = env.steps[-1][0].observation["town"]["unlocked_shops"]
    return {
        "episode": s["episode"],
        "recorded": s["finals"],
        "local": local,
        "shops_match": local_shops == s["shops"],
    }


def bench_job(job):
    from kaggle_environments import make

    path, agent_path, seat = job
    s = json.load(open(path))
    ours = _load_bundle(agent_path)
    script = make_script_agent(s["actions"][1 - seat])
    agents = [ours, script] if seat == 0 else [script, ours]
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run(agents)
    finals = [st.reward or 0 for st in env.steps[-1]]
    return {
        "episode": s["episode"],
        "seat": seat,
        "us": finals[seat],
        "script": finals[1 - seat],
        "orig_seat": s["finals"][seat],
        "orig_script": s["finals"][1 - seat],
        "win": finals[seat] > finals[1 - seat],
        "orig_seat_won": s["finals"][seat] > s["finals"][1 - seat],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fidelity", "bench"])
    ap.add_argument("--agent", default="arena/agents/rl-v9/main.py")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]

    if args.mode == "fidelity":
        with Pool(args.procs) as pool:
            results = pool.map(fidelity_job, paths)
        exact = close = far = 0
        for r in results:
            deltas = [abs(a - b) / max(1, abs(b)) for a, b in zip(r["local"], r["recorded"])]
            worst = max(deltas)
            tag = "EXACT" if worst < 0.01 else ("CLOSE" if worst < 0.10 else "FAR")
            exact += tag == "EXACT"
            close += tag == "CLOSE"
            far += tag == "FAR"
            print(f"ep {r['episode']}: recorded={[f'{v:,.0f}' for v in r['recorded']]} "
                  f"local={[f'{v:,.0f}' for v in r['local']]} "
                  f"worst_delta={worst:.1%} shops_match={r['shops_match']} [{tag}]")
        print(f"\nfidelity: {exact} exact / {close} close / {far} far of {len(results)}")
    else:
        jobs = [(p, args.agent, seat) for p in paths for seat in (0, 1)]
        with Pool(args.procs) as pool:
            results = pool.map(bench_job, jobs)
        wins = sum(r["win"] for r in results)
        uplift = [r["us"] - r["orig_seat"] for r in results]
        vs_winner = [r for r in results if not r["orig_seat_won"]]
        flips = sum(r["win"] for r in vs_winner)
        print(f"agent: {args.agent}")
        print(f"session win rate vs scripts: {wins}/{len(results)} ({wins/len(results):.0%})")
        print(f"profit uplift vs seat's original occupant: mean=${sum(uplift)/len(uplift):+,.0f}")
        print(f"LOSING seats flipped to wins: {flips}/{len(vs_winner)} ({flips/max(1,len(vs_winner)):.0%})")
        worst = sorted(results, key=lambda r: r["us"] - r["script"])[:5]
        print("hardest scripts:", ", ".join(
            f"ep{r['episode']}(us ${r['us']:,.0f} vs ${r['script']:,.0f})" for r in worst))


if __name__ == "__main__":
    main()
