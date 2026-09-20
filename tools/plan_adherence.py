"""Does the executor deliver what the planner asks for?

The question this answers: if a rollout produces a better per-day schedule,
will it be EXECUTED, or absorbed? A plan the executor cannot realise is a
plan the rollout cannot pay for.

The planner's day-d targets are computable without instrumenting the frozen
bundle -- they are the champion constants:

    hires     CHAMP_HIRES[day], then capped by workload//3 and by cash
    herd      _champ_herd_targets(day) -> GOOSE / COW / SHEEP
    wheat     knobs["wheat_engine"] standing tiles, from day 10

So run the shipped agent, read the board at dusk, and compare. The gap is
the executor's, not the planner's.

    PYTHONPATH=. uv run python tools/plan_adherence.py --agent r-mrg100 --n 20
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kagent.planner as P  # noqa: E402

D = 30
KINDS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "GOOSE", "COW", "SHEEP"]


def job(spec):
    seed, agent, opp = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(n):
        g = {}
        exec(open(f"{REPO}/arena/agents/{n}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load(agent), load(opp)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    got = np.zeros((D, len(KINDS)), "float32")
    hands = np.zeros(D, "float32")
    empty = np.zeros(D, "float32")
    money = np.zeros(D, "float32")
    while not env.done:
        o = env.state[0].observation
        d, hr = o["step"] // 24, o["step"] % 24
        if d < D and hr == 12:
            f = o["farms"][0]
            hands[d] = f.get("hires_today", 0)
            money[d] = f["money"]
            c = np.zeros(len(KINDS), "float32")
            e = 0
            for row in f["tiles"]:
                for t in row:
                    if t is None:
                        e += 1
                    elif isinstance(t, dict):
                        if t.get("kind") == "PLANT" and t["crop"] in KINDS:
                            c[KINDS.index(t["crop"])] += 1
                        elif t.get("animal") in KINDS:
                            c[KINDS.index(t["animal"])] += 1
            got[d] = c
            empty[d] = e
        env.step([a0(o), a1(env.state[1].observation)])
    return got, hands, empty, money


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="r-mrg100")
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()

    specs = [(300000 + i, args.agent, args.opp) for i in range(args.n)]
    with mp.Pool(args.procs, maxtasksperchild=20) as pool:
        out = pool.map(job, specs)
    got = np.mean([o[0] for o in out], 0)
    hands = np.mean([o[1] for o in out], 0)
    empty = np.mean([o[2] for o in out], 0)
    money = np.mean([o[3] for o in out], 0)

    kn = P.KNOB_DEFAULTS
    print(f"\nPLAN ADHERENCE   {args.agent}, {len(out)} seeds\n")
    print("HIRES — planner asks CHAMP_HIRES[day], then caps by workload/3 "
          "and cash\n")
    print(f"    {'day':>4}{'asked':>8}{'got':>8}{'shortfall':>11}{'cash':>10}")
    sf = []
    for d in range(0, 20):
        ask = P.CHAMP_HIRES[d] if d < len(P.CHAMP_HIRES) else 8
        sf.append(ask - hands[d])
        print(f"    {d:>4}{ask:>8}{hands[d]:>8.1f}{ask-hands[d]:>+11.1f}"
              f"{money[d]:>10,.0f}")
    print(f"\n    mean shortfall days 0-19: {np.mean(sf):+.2f} hands"
          f"   ({100*np.mean(sf)/max(np.mean([P.CHAMP_HIRES[d] for d in range(20)]),1):+.0f}%)")

    print("\n\nHERD — planner asks _champ_herd_targets(day)\n")
    print(f"    {'day':>4}" + "".join(f"{k[:5]+' ask':>11}{k[:5]+' got':>11}"
                                      for k in ("GOOSE", "COW", "SHEEP")))
    for d in (2, 4, 6, 8, 10, 12, 14, 16, 18):
        t = P._champ_herd_targets(d, kn)
        row = ""
        for k in ("GOOSE", "COW", "SHEEP"):
            row += f"{t[k]:>11}{got[d, KINDS.index(k)]:>11.1f}"
        print(f"    {d:>4}{row}")

    print("\n\nWHEAT — planner's engine target is "
          f"knobs['wheat_engine'] = {kn['wheat_engine']} standing tiles "
          "from day 10\n")
    print(f"    {'day':>4}{'asked':>8}{'got':>8}{'shortfall':>11}"
          f"{'empty tiles':>13}")
    for d in range(8, 24, 2):
        ask = kn["wheat_engine"] if d >= 10 else 8
        g = got[d, KINDS.index("WHEAT")]
        print(f"    {d:>4}{ask:>8}{g:>8.1f}{ask-g:>+11.1f}{empty[d]:>13.1f}")

    print("\n\nVERDICT")
    w = got[10:24, KINDS.index("WHEAT")].mean()
    print(f"    wheat: asked {kn['wheat_engine']}, held {w:.1f}"
          f"  -> {100*w/max(kn['wheat_engine'],1):.0f}% of target")
    c = got[12:19, KINDS.index("COW")].mean()
    ct = P._champ_herd_targets(14, kn)["COW"]
    print(f"    cows:  asked {ct}, held {c:.1f}"
          f"  -> {100*c/max(ct,1):.0f}% of target")
    print(f"    hands: mean shortfall {np.mean(sf):+.2f}")
    print(f"    empty tiles at day 12-20: {empty[12:21].mean():.1f}"
          f"   (idle ground the plan wanted planted)")


if __name__ == "__main__":
    main()
