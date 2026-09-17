"""Splice tapes at latch points and score the routes by WIN RATE.

The idea under test: instead of replaying one recorded tape end to end, assemble
a route from SEGMENTS of several tapes, switching between them at "latch points"
where a switch is cheap. That needs a decent pool of tapes and a way to pick the
switch steps.

We have one latch point on record: splicing at step 144 (day 6) is LOSSLESS
within a rating-3029 cluster -- replaying tape A then switching to B yields
exactly B's outcome. We built a day-6 SELECTOR on it and stopped. We never
searched for more latch points, and never spliced more than once.

Why this is worth doing when the selector was refuted. The selector's failure was
that WHICH tape wins in WHICH world is noise (oracle +$41,666 vs an
interaction-shuffled null +$41,951 +/- 745), so per-world choice is unlearnable.
Splicing is a different claim: that a route assembled from the best SEGMENTS of
several tapes beats any single tape ON AVERAGE. That is a main effect, not an
interaction, and the null result does not touch it.

Splicing is list surgery -- `make_tape_agent(A[:k] + B[k:])` -- and TapeRunner's
existing weed/hire repair handles the desync that follows.

Metric is WIN RATE, and this is the one place in the project where it has
resolution: tapes win 62-79% of games, against champ-h8's 3.5%. Margin is
reported alongside but the rating is win-based.

    PYTHONPATH=. uv run python tools/tape_splice.py base --n 60
    PYTHONPATH=. uv run python tools/tape_splice.py search --top 6 --n 40
"""

import argparse
import collections
import itertools
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OPPS = ["pub-ahmed-v38", "pub-tschinkel-psr", "pub-boatlee-v16"]
# Day boundaries are the natural latch candidates: a tape's day is self-contained
# (TapeRunner clears its repair delay at each dawn), so a switch there inherits no
# half-finished shift. 144 = day 6, the one already known to be lossless.
LATCHES = [72, 144, 240, 336, 432, 528]
# Candidate tapes are chosen by PANEL TRANSFER, never by recorded margin. The
# record is explicit that transfer is unpredicted by the recording ("a $155k/3125
# tape transfers to $32,799; a $152k/3028 tape holds"), and ranking the library
# by margin picked tapes that win 0-12% of games. Panel win rates put tape 46 at
# 65.8% and tape 129 -- the one we ship -- at 65.4%. Note tape 55 has the best
# mean margin (+$17,793) and only a 48.7% win rate: it wins big and loses often,
# which is worthless to a win-based rating.
DEFAULT_TAPES = [46, 129, 262, 63, 144]
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = json.load(open(os.path.join(REPO, "rl", "datasets", "tapes", "library.json")))
    return _LIB


def splice(a_idx, b_idx, k):
    """Tape a up to step k, then tape b. k=None or b=None -> tape a alone."""
    A = lib()[a_idx]["actions"]
    if b_idx is None or k is None:
        return list(A)
    B = lib()[b_idx]["actions"]
    return list(A[:k]) + list(B[k:])


def play_one(job):
    a_idx, b_idx, k, opp, seed, swap = job
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.tape_agent import make_tape_agent

    agent = make_tape_agent(splice(a_idx, b_idx, k))
    opp_path = os.path.join(REPO, "arena", "agents", opp, "main.py")
    specs = [opp_path, agent] if swap else [agent, opp_path]
    seat = 1 if swap else 0
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    try:
        env.run(specs)
    except Exception:
        return None
    last = env.steps[-1]
    me, them = last[seat], last[1 - seat]
    if me["status"] != "DONE" or me["reward"] is None or them["reward"] is None:
        return None
    return (a_idx, b_idx, k, me["reward"] > them["reward"], me["reward"] - them["reward"])


def run_jobs(jobs, procs):
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    done = 0
    with Pool(procs) as pool:
        for r in pool.imap_unordered(play_one, jobs, chunksize=2):
            done += 1
            if r:
                a, b, k, won, marg = r
                rec = agg[(a, b, k)]
                rec[0] += won
                rec[1] += 1
                rec[2] += marg
            if done % 200 == 0:
                print(f"   {done}/{len(jobs)}", flush=True)
    return agg


def label(a, b, k):
    if b is None:
        return f"tape{a}"
    return f"tape{a}->{b}@{k}"


def report(agg, title):
    rows = []
    for (a, b, k), (w, n, m) in agg.items():
        if n:
            rows.append((w / n, w, n, m / n, a, b, k))
    rows.sort(reverse=True)
    print(f"\n{title}")
    print(f"{'route':<20} {'n':>5} {'win%':>7} {'margin':>11}")
    for p, w, n, m, a, b, k in rows:
        print(f"{label(a, b, k):<20} {n:>5} {p*100:>6.1f}% {m:>+11,.0f}")
    return rows


def tapes_of(args):
    return ([int(x) for x in args.tapes.split(",")] if args.tapes else DEFAULT_TAPES)


def screen_cmd(args):
    """Screen tapes that have NEVER been evaluated.

    Every result this project has produced -- the day-6 selector, the 58-arm
    panel, every shipped tape -- comes from ONE cohort of 58 tapes at rating
    3029.3. The library holds 301, so 243 arms have never been run at all,
    including whole cohorts of 69 / 65 / 62 / 35 at ratings 3027.6 / 3125.1 /
    3017.2 / 3015.6. The 3125.1 cohort is the highest-rated we hold and is
    entirely unscreened.

    Cheap pass first: the point is to RANK, not to gate. A screen over K arms at
    n games has a max-order-statistic of pure noise around 3 se, so the winner
    must be confirmed on fresh seeds before it means anything.
    """
    order = json.load(open(args.order)) if args.order else None
    if order is None:
        panel = set(json.loads(l)["tape"] for l in
                    open(os.path.join(REPO, "rl", "datasets", "tapes", "panel.jsonl")))
        order = [i for i in range(len(lib())) if i not in panel]
    order = order[:args.limit] if args.limit else order
    seeds = [args.base_seed + i for i in range(args.n)]
    jobs = [(a, None, None, o, s, w)
            for a in order for o in OPPS for s in seeds for w in (False, True)]
    print(f"{len(jobs):,} games: {len(order)} unscreened tapes x {len(OPPS)} "
          f"opponents x {args.n} seeds x 2 swaps\n")
    agg = run_jobs(jobs, args.procs)
    rows = report(agg, "UNSCREENED TAPES (best first)")
    L = lib()
    out = [{"tape": r[4], "win": r[0], "n": r[2], "margin": r[3],
            "rating": L[r[4]]["rating"], "episode": L[r[4]]["episode"],
            "seat": L[r[4]]["seat"], "recorded": L[r[4]]["final"]} for r in rows]
    json.dump(out, open(os.path.join(REPO, "rl", "datasets", "tapes",
                                     "unscreened_winrate.json"), "w"), indent=1)
    print(f"\nbest known arm from the screened cohort: tape 46 at 65.8% "
          f"(panel, n=79); shipped tape 129 at 65.4%")


def base_cmd(args):
    """How good is each candidate tape ALONE? The bar any splice must clear."""
    seeds = [args.base_seed + i for i in range(args.n)]
    T = tapes_of(args)
    jobs = [(a, None, None, o, s, w)
            for a in T for o in OPPS for s in seeds for w in (False, True)]
    print(f"{len(jobs):,} games: {len(T)} tapes x {len(OPPS)} opponents "
          f"x {args.n} seeds x 2 swaps\n")
    rows = report(run_jobs(jobs, args.procs), "SINGLE TAPES")
    json.dump([{"a": r[4], "win": r[0], "n": r[2], "margin": r[3]} for r in rows],
              open(os.path.join(REPO, "rl", "datasets", "tapes", "base_winrate.json"), "w"),
              indent=1)


def search_cmd(args):
    """Every ordered pair of the top tapes, at every latch point."""
    seeds = [args.base_seed + i for i in range(args.n)]
    T = tapes_of(args)
    latches = [int(x) for x in args.latches.split(",")] if args.latches else LATCHES
    pairs = [(a, b) for a, b in itertools.permutations(T, 2)]
    jobs = []
    for a in T:                                     # the singles, as the control
        jobs += [(a, None, None, o, s, w)
                 for o in OPPS for s in seeds for w in (False, True)]
    for a, b in pairs:
        for k in latches:
            jobs += [(a, b, k, o, s, w)
                     for o in OPPS for s in seeds for w in (False, True)]
    print(f"{len(jobs):,} games: {len(T)} singles + {len(pairs)} pairs "
          f"x {len(latches)} latches, {len(OPPS)} opponents x {args.n} seeds x 2\n")
    agg = run_jobs(jobs, args.procs)
    rows = report(agg, "ALL ROUTES (best first)")
    singles = {r[4]: r[0] for r in rows if r[5] is None}
    best_single = max(singles.values()) if singles else 0
    better = [r for r in rows if r[5] is not None and r[0] > best_single]
    print(f"\nbest single tape: {best_single*100:.1f}% win")
    print(f"splices beating it: {len(better)} of {sum(1 for r in rows if r[5] is not None)}")
    json.dump([{"a": r[4], "b": r[5], "k": r[6], "win": r[0], "n": r[2], "margin": r[3]}
               for r in rows],
              open(os.path.join(REPO, "rl", "datasets", "tapes", "splice_search.json"), "w"),
              indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("base", "search", "screen"):
        p = sub.add_parser(nm)
        p.add_argument("--tapes", default=None,
                       help="comma-separated library indices; default = the top "
                            "panel-transfer tapes")
        p.add_argument("--latches", default=None, help="comma-separated step numbers")
        p.add_argument("--n", type=int, default=40, help="seeds (x2 swaps x3 opponents)")
        p.add_argument("--base-seed", type=int, default=610000)
        p.add_argument("--procs", type=int, default=6)
        p.add_argument("--order", default=None, help="json list of tape indices")
        p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"base": base_cmd, "search": search_cmd, "screen": screen_cmd}[a.cmd](a)
