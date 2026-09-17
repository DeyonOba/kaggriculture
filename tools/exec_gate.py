"""Gate an executor change against EXTERNAL opponents, paired on identical worlds.

Why this exists: on 2026-09-14 an executor change (`MIN_ZONE_TILES` 5 -> 8) beat a
copy of our own champion 24-6 at p=0.0014 and was $4,490 +/- 3,333 WORSE against a
real 2500+ agent on the same worlds. Two agents sharing one price ladder trade wins
that come purely from selling at different moments against each other, so the twin
arena cannot gate anything that changes when goods reach the shed. This runs the
candidate and the baseline against the same public agents over the same
seed/swap keys and reports the paired difference.

    uv run python tools/fetch_public_agents.py          # once, installs pub-*
    PYTHONPATH=. uv run python tools/exec_gate.py --cand var-sweep --base champ-h8

A change ships only if the paired mean is positive against every opponent and the
pooled mean clears ~2 standard errors.
"""

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "arena", "results")
DEFAULT_OPPS = ["pub-ahmed-v38", "pub-tschinkel-psr", "pub-boatlee-v16"]


def run(agent, opp, n, base_seed, procs):
    path = os.path.join(RESULTS, f"{agent}_vs_{opp}.jsonl")
    if os.path.exists(path):
        os.remove(path)
    cmd = [sys.executable if False else "uv", "run", "python", "arena/run_games.py",
           "--a", agent, "--b", opp, "--n", str(n), "--base-seed", str(base_seed),
           "--procs", str(procs)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": REPO})
    if not os.path.exists(path):
        print(f"    FAILED {agent} vs {opp}: {(r.stderr or r.stdout)[-300:]}")
        return None
    return path


def scores(path, agent):
    out = {}
    for line in open(path):
        row = json.loads(line)
        if agent not in row["players"]:
            continue
        i = row["players"].index(agent)
        if row.get("statuses") and any(s != "DONE" for s in row["statuses"]):
            continue
        out[(row["seed"], row["swap"])] = row["rewards"][i]
    return out


def beats_opponent(path, agent):
    """Did we BEAT the opponent? -> {(seed, swap): bool}

    This is the quantity the leaderboard actually scores. The rating is
    WIN-BASED, while the paired-margin criterion below compares us to our own
    baseline -- and the two came apart badly: champ-h8 wins 3.5% of games
    against the public agents on a mean margin of -$31,049, so a change worth
    +$1,381/game moves the margin of a loss and flips almost no outcomes.
    match-all gated +$1,381 +/- 641 over 360 games and took the head-to-head
    win count from 5/144 to 4/144.
    """
    out = {}
    for line in open(path):
        row = json.loads(line)
        if agent not in row["players"]:
            continue
        if row.get("statuses") and any(s != "DONE" for s in row["statuses"]):
            continue
        i = row["players"].index(agent)
        mine, theirs = row["rewards"][i], row["rewards"][1 - i]
        if mine is None or theirs is None:
            continue
        out[(row["seed"], row["swap"])] = mine > theirs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="candidate agent name in arena/agents")
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--opps", default=",".join(DEFAULT_OPPS))
    ap.add_argument("--n", type=int, default=12, help="seeds per opponent (x2 with swap)")
    ap.add_argument("--base-seed", type=int, default=19100)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--skip-base", action="store_true",
                    help="reuse existing baseline result files instead of replaying them")
    args = ap.parse_args()

    opps = [o.strip() for o in args.opps.split(",") if o.strip()]
    missing = [o for o in opps if not os.path.isfile(os.path.join(REPO, "arena", "agents", o, "main.py"))]
    if missing:
        sys.exit(f"missing opponents {missing}; run: uv run python tools/fetch_public_agents.py")

    pooled = []
    wins = []
    print(f"{args.cand} vs {args.base}, paired on identical worlds against {len(opps)} external agents\n")
    print(f"{'opponent':<22} {'n':>4} {'cand mean':>11} {'base mean':>11} "
          f"{'paired diff':>13} {'se':>8} {'better':>8} {'WINS vs opp':>16}")
    ok = True
    for opp in opps:
        cp = run(args.cand, opp, args.n, args.base_seed, args.procs)
        bp = (os.path.join(RESULTS, f"{args.base}_vs_{opp}.jsonl") if args.skip_base
              else run(args.base, opp, args.n, args.base_seed, args.procs))
        if not cp or not bp or not os.path.exists(bp):
            ok = False
            continue
        c, b = scores(cp, args.cand), scores(bp, args.base)
        cwin, bwin = beats_opponent(cp, args.cand), beats_opponent(bp, args.base)
        keys = sorted(set(c) & set(b))
        if len(keys) < 2:
            print(f"{opp:<22} {len(keys):>4}  no overlap")
            ok = False
            continue
        d = [c[k] - b[k] for k in keys]
        se = st.stdev(d) / math.sqrt(len(d))
        pooled += d
        cw = sum(cwin.get(k, False) for k in keys)
        bw = sum(bwin.get(k, False) for k in keys)
        wins.append((cw, bw, len(keys)))
        print(f"{opp:<22} {len(keys):>4} {st.mean(c[k] for k in keys):>11,.0f} "
              f"{st.mean(b[k] for k in keys):>11,.0f} {st.mean(d):>+13,.0f} {se:>8,.0f} "
              f"{sum(x > 0 for x in d):>4}/{len(d):<3} "
              f"{bw:>3}/{len(keys):<3} -> {cw:>3}/{len(keys):<3}")
        if st.mean(d) <= 0:
            ok = False

    if pooled:
        se = st.stdev(pooled) / math.sqrt(len(pooled))
        m = st.mean(pooled)
        print(f"\n{'POOLED':<22} {len(pooled):>4} {'':>11} {'':>11} {m:>+13,.0f} {se:>8,.0f} "
              f"{sum(x > 0 for x in pooled):>4}/{len(pooled):<3}")
        cw = sum(x[0] for x in wins); bw = sum(x[1] for x in wins)
        nn = sum(x[2] for x in wins)
        print(f"\n{'WIN RATE vs opponents':<22} {nn:>4} "
              f"base {bw}/{nn} ({bw/nn*100:.1f}%)  ->  cand {cw}/{nn} "
              f"({cw/nn*100:.1f}%)   {cw-bw:+d} games")
        verdict = "SHIP" if (ok and m > 2 * se) else ("hold" if m > 0 else "REJECT")
        print(f"\nverdict (margin): {verdict}  "
              f"(positive vs every opponent and pooled mean > 2 se)")
        if cw <= bw:
            print("WARNING: the margin verdict is NOT the leaderboard's metric. "
                  "The rating is WIN-BASED and this change wins no more games "
                  "than the base. champ-h8 wins 3.5% of games at a mean margin "
                  "of -$31,049, so margin gains move the size of a LOSS.")


if __name__ == "__main__":
    main()
