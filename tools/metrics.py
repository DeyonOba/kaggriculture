"""Candidate success metrics for executor work, and whether any of them can
replace profit as the gate.

Profit carries ~$20k of per-world noise, so a $2k effect needs ~50 games. Every
cheap proxy tried so far has misled: actions-per-visit, realized price and raw
productive-action count each IMPROVED while money fell. They share a defect --
they count actions instead of valuing them.

A proxy is only worth using if it separates a known-good agent from a known-bad
one with FEWER games than final margin does. That is measured here the same way
the win-probability head was refuted (it scored |t| 1.41 against margin's 6.53).

    PYTHONPATH=. uv run python tools/metrics.py --n 16
"""
import argparse, os, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kaggle_environments.envs.kaggriculture.kaggriculture as K

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
VALUED = ("HARVEST", "COLLECT_FERTILIZER")


def _play(job):
    agent, opp, seed = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    try:
        env.run([agent, opp])
    except Exception:  # noqa: BLE001
        return None
    m = defaultdict(float)
    prev_shed = {}
    for i, step in enumerate(env.steps):
        obs = step[0]["observation"]
        a = step[0].get("action") or {}
        for act in [a.get("farmer") or []] + list(a.get("hands") or []):
            if not (isinstance(act, list) and act):
                continue
            op = act[0]
            m["turns"] += 1
            if op in MOVES:
                m["moves"] += 1
            elif op == "PASS":
                m["pass"] += 1
            else:
                m["actions"] += 1
        # value produced: goods that appeared in the shed, at BASE price, so the
        # figure is independent of the market crash and of the opponent
        shed = (obs.get("private") or {}).get("shed") or {}
        for it, n in shed.items():
            if it in K.MARKET_PARAMS:
                d = n - prev_shed.get(it, 0)
                if d > 0:
                    m["value"] += d * K.MARKET_PARAMS[it]["base"]
        prev_shed = dict(shed)
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return None
    m["margin"] = f[0] - f[1]
    m["final"] = f[0]
    m["win"] = 1.0 if f[0] > f[1] else 0.0
    m["value_per_turn"] = m["value"] / max(1, m["turns"])
    m["value_per_move"] = m["value"] / max(1, m["moves"])
    m["actions_per_turn"] = m["actions"] / max(1, m["turns"])
    return dict(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="arena/agents/champ-h8/main.py")
    ap.add_argument("--bad", default="arena/agents/champ-h2/main.py")
    ap.add_argument("--opps", default="arena/agents/pub-ahmed-v38/main.py,"
                                      "arena/agents/pub-boatlee-v16/main.py")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=80000)
    args = ap.parse_args()
    opps = [o.strip() for o in args.opps.split(",")]
    jobs = [(a, o, args.base_seed + k) for a in (args.good, args.bad)
            for o in opps for k in range(args.n)]
    out = defaultdict(list)
    with Pool(args.procs) as p:
        for job, r in zip(jobs, p.imap(_play, jobs)):
            if r:
                out[job[0]].append(r)
    g, b = out[args.good], out[args.bad]
    print(f"{os.path.basename(os.path.dirname(args.good))} n={len(g)}  vs  "
          f"{os.path.basename(os.path.dirname(args.bad))} n={len(b)}\n")
    print(f"{'metric':<18} {'good':>12} {'bad':>12} {'diff':>11} {'se':>9} {'|t|':>6}")
    rows = ["margin", "final", "win", "value", "value_per_turn", "value_per_move",
            "actions", "actions_per_turn", "moves"]
    res = []
    for k in rows:
        x = np.array([r[k] for r in g])
        y = np.array([r[k] for r in b])
        d = x.mean() - y.mean()
        se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
        t = abs(d) / max(se, 1e-9)
        res.append((t, k))
        f = ".3f" if k in ("win", "actions_per_turn", "value_per_turn", "value_per_move") else ",.0f"
        print(f"{k:<18} {x.mean():>12{f}} {y.mean():>12{f}} {d:>11{f}} {se:>9{f}} {t:>6.2f}")
    res.sort(reverse=True)
    print(f"\nbest separator: {res[0][1]} (|t| {res[0][0]:.2f});  margin scores "
          f"{[t for t, k in res if k == 'margin'][0]:.2f}")
    print("A proxy is only useful if its |t| beats margin's -- otherwise it costs games, not saves them.")


if __name__ == "__main__":
    main()
