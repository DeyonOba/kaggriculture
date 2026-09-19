"""Gate a candidate on the ladder, PAIRED against a base on identical worlds.

`rl/session_bench.py ladder` prints a win rate per rating bin but no baseline
and no standard error, so comparing two runs means eyeballing two tables. That
wastes most of the signal: the bake-off measured per-world sd at $34,749 and
the paired sd at $17,168 (correlation 0.95), so differencing on the same world
cuts an evaluation's se by ~3x. Every ladder comparison should be paired.

Reports, per candidate:

    money    paired mean difference per game, with se and z
    win      McNemar on the worlds where exactly one of base/candidate wins --
             the rating is win-based, and unpaired win rates hide that most
             worlds are decided the same way by both agents

The base is cached per (n, seed) so a second candidate costs only its own
games.

    PYTHONPATH=. uv run python tools/ladder_gate.py --cand g-tom9 g-tom14 --n 60
"""

import argparse
import json
import math
import os
import statistics as st
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from rl.session_bench import bench_job                      # noqa: E402
from tools.optimizer_bakeoff import ladder_jobs             # noqa: E402

CACHE = os.path.join(REPO, "rl", "bakeoff_runs")


def evaluate(agent, jobs, pool):
    path = os.path.join(REPO, "arena", "agents", agent, "main.py")
    res = [r for r in pool.map(bench_job,
           [(p, path, 1 - side) for p, side in jobs]) if r]
    return {(r["episode"], r["seat"]): (r["us"] - r["script"], bool(r["win"]))
            for r in res}


def load_or_run(agent, jobs, pool, tag):
    """Cached results are reused ONLY if they cover the current world list.

    The cache key is (agent, n, seed), which does not encode the CORPUS. The
    session index grows as we harvest, so ladder_jobs(45, seed=0) returned a
    different 270 worlds in September than it does now -- and the old code
    loaded the stale file anyway. The gate then intersected the two key sets
    and reported a number computed on whatever happened to overlap: for the
    melon arms that was 119 of 270 worlds, with the base's win rate printed
    over all 270 and the candidate's over the 119. Re-running one side and
    not the other is exactly how that happens, so validate instead.
    """
    f = os.path.join(CACHE, f"ladder_{agent}_{tag}.json")
    # evaluate() keys by (episode, 1 - side); jobs carry (path, side) and the
    # files are named session-<episode>.json, so derive the same key here
    # rather than opening 270 sessions.
    want = {(int(os.path.basename(pth).split("-")[1].split(".")[0]), 1 - side)
            for pth, side in jobs}
    if os.path.exists(f):
        got = {tuple(json.loads(k)): tuple(v)
               for k, v in json.load(open(f)).items()}
        if want.issubset(got):
            return {k: got[k] for k in want}
        print(f"    cache for {agent} covers {len(want & set(got))}/{len(want)} "
              f"of the current worlds -- recomputing", flush=True)
    out = evaluate(agent, jobs, pool)
    os.makedirs(CACHE, exist_ok=True)
    json.dump({json.dumps(list(k)): list(v) for k, v in out.items()}, open(f, "w"))
    return out


def mcnemar(b_only, c_only):
    n = b_only + c_only
    if n == 0:
        return 1.0
    lo = min(b_only, c_only)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--n", type=int, default=60, help="sessions per rating bin")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", action="store_true", default=True)
    args = ap.parse_args()

    jobs = ladder_jobs(args.n, heldout=args.heldout, seed=args.seed)
    tag = f"{args.n}_{args.seed}"
    print(f"{len(jobs)} held-out ladder worlds, base {args.base}\n")

    with Pool(args.procs) as pool:
        base = load_or_run(args.base, jobs, pool, tag)
        bw = sum(1 for _, w in base.values() if w)
        print(f"  {args.base:<14} {st.mean(v for v, _ in base.values()):>+10,.0f}/game"
              f"   win {bw}/{len(base)} = {bw/len(base):.3f}\n")
        print(f"  {'candidate':<14}{'money':>12}{'se':>8}{'z':>7}"
              f"{'win':>12}{'b-only':>8}{'c-only':>8}{'McNemar p':>11}"
              f"{'worlds':>7}{'base win':>9}")
        for cand in args.cand:
            c = load_or_run(cand, jobs, pool, tag)
            ks = sorted(set(base) & set(c))
            if len(ks) < 2:
                print(f"  {cand:<14} no overlap")
                continue
            d = [c[k][0] - base[k][0] for k in ks]
            m = st.mean(d)
            se = st.stdev(d) / len(d) ** 0.5
            bo = sum(1 for k in ks if base[k][1] and not c[k][1])
            co = sum(1 for k in ks if c[k][1] and not base[k][1])
            cw = sum(1 for k in ks if c[k][1])
            bwk = sum(1 for k in ks if base[k][1])
            print(f"  {cand:<14}{m:>+12,.0f}{se:>8,.0f}{m/se if se else 0:>7.2f}"
                  f"{cw/len(ks):>12.3f}{bo:>8}{co:>8}{mcnemar(bo, co):>11.4f}"
                  f"{len(ks):>7}{bwk/len(ks):>9.3f}")
    print("\n  money is a PAIRED per-world difference (paired sd ~$17k vs "
          "$35k unpaired).\n  win is McNemar on discordant worlds only -- most "
          "worlds are decided the\n  same way by both agents and carry no "
          "information about the difference.")


if __name__ == "__main__":
    main()
