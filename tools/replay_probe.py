"""One probe for any agent's replays: tape-or-not, turn budget, layout, sales.

Consolidates the diagnostics that were written ad hoc while autopsying
pub-ahmed-v38 and M & M & P & Q, so the next agent we study takes one command
instead of six scratch scripts.

Probes:

  tape      Is it a fixed script or a reactive agent? Compares the opening
            action sequence across DIFFERENT seeds. A tape agrees ~100%;
            M & M & P & Q agrees 27.4%.
  budget    work / move / pass split and MOVES PER PRODUCTIVE ACTION, the
            single number that separates strength here: 0.71 (M&M&P&Q),
            0.84 (ahmed), 0.85 (a tape), 1.45 (champ-h8).
  layout    Quadrants bought by day and the per-worker-day bounding-box span.
            Wide span with LOW moves/action means sweeping; narrow span with
            HIGH moves/action means oscillating.
  ops       Actions per game by operation.
  sales     True units sold and realised $/unit by conservation -- order[2] is
            a REQUESTED quantity, not a fill, so counting orders is wrong
            (it made one agent look like it sold 17,045 carrots).

    PYTHONPATH=. uv run python tools/replay_probe.py --dir replays/mmpq --team "M & M & P & Q"
    PYTHONPATH=. uv run python tools/replay_probe.py --agent champ-h8 --opp pub-ahmed-v38 --n 4
    PYTHONPATH=. uv run python tools/replay_probe.py --dir replays/x --team T --probes tape,budget
"""

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
REFERENCE = {"M&M&P&Q": 0.71, "ahmed": 0.84, "tape": 0.85, "champ-h8": 1.45}


def from_replays(d, team):
    """-> [(steps, seat, info)] for every replay in `d` featuring `team`."""
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, d, "*.json"))):
        rep = json.load(open(p))
        names = (rep.get("info") or {}).get("TeamNames") or []
        if team and team not in names:
            continue
        out.append((rep["steps"], names.index(team) if team else 0, rep))
    return out


def from_engine(agent, opp, n, base_seed):
    """Play `agent` locally so the same probes work on our own agents."""
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    out = []
    for k in range(n):
        env = make("kaggriculture", configuration={"seed": base_seed + k}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        out.append((env.steps, 0, {"info": {"seed": base_seed + k}}))
    return out


def probe_tape(games):
    print("TAPE OR REACTIVE -- farmer's first 40 actions across different worlds")
    sigs = []
    for steps, seat, rep in games:
        sigs.append(tuple(tuple(steps[t][seat].get("action", {}).get("farmer") or ["PASS"])
                          for t in range(1, min(41, len(steps)))))
    if len(sigs) < 2:
        print("  need >= 2 replays"); return
    agree = [sum(x == y for x, y in zip(a, b)) / len(a)
             for i, a in enumerate(sigs) for b in sigs[i + 1:]]
    print(f"  {len(sigs)} games -> {len(set(sigs))} distinct sequences")
    print(f"  pairwise identical-action rate: mean {st.mean(agree)*100:.1f}%  "
          f"min {min(agree)*100:.1f}%  max {max(agree)*100:.1f}%")
    print("  => REACTIVE AGENT" if st.mean(agree) < 0.90 else "  => fixed script (tape)")


def probe_budget(games):
    c = collections.Counter()
    for steps, seat, rep in games:
        for t in range(1, len(steps)):
            a = steps[t][seat].get("action") or {}
            for x in [a.get("farmer")] + list(a.get("hands") or []):
                op = x[0] if isinstance(x, list) and x else "PASS"
                k = "move" if op in MOVES else ("pass" if op == "PASS" else "work")
                c[k] += 1
                if k == "work":
                    c[op] += 1
    g = len(games) or 1
    tot = c["work"] + c["move"] + c["pass"] or 1
    mpa = c["move"] / max(1, c["work"])
    print("TURN BUDGET")
    print(f"  work {c['work']/g:>7,.0f} ({c['work']/tot*100:>4.1f}%)   "
          f"move {c['move']/g:>7,.0f} ({c['move']/tot*100:>4.1f}%)   "
          f"pass {c['pass']/g:>6,.0f} ({c['pass']/tot*100:>4.1f}%)")
    print(f"  MOVES PER PRODUCTIVE ACTION: {mpa:.2f}")
    print("  reference: " + "  ".join(f"{k} {v}" for k, v in REFERENCE.items()))
    return c


def probe_layout(games):
    quads = collections.defaultdict(list)
    worked = collections.defaultdict(set)
    for steps, seat, rep in games:
        for t in range(len(steps) - 1):
            o = steps[t][seat].get("observation")
            if not o:
                continue
            f = o["farms"][seat]
            quads[o["day"]].append(len(f.get("unlocked_quadrants") or ["NW"]))
            units = [f.get("farmer")] + list(f.get("hands") or [])
            acts = [(steps[t + 1][seat].get("action") or {}).get("farmer")] + \
                   list((steps[t + 1][seat].get("action") or {}).get("hands") or [])
            for i, (pos, a) in enumerate(zip(units, acts)):
                op = a[0] if isinstance(a, list) and a else "PASS"
                if op in MOVES or op == "PASS" or not pos:
                    continue
                worked[(id(steps), o["day"], i)].add(tuple(pos[:2]))
    spans = []
    for pts in worked.values():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        spans.append(max(xs) - min(xs) + max(ys) - min(ys))
    print("LAYOUT")
    line = "  quadrants:"
    for d in (5, 10, 15, 20, 25):
        v = quads.get(d) or [0]
        line += f"  d{d} {st.mean(v):.1f}"
    print(line)
    print(f"  worker-day span (bounding box): {st.mean(spans) if spans else 0:.2f}")
    print("  wide span + low moves/action = SWEEPING; "
          "narrow span + high moves/action = OSCILLATING")


def probe_ops(games, c=None):
    if c is None:
        c = probe_budget(games)
    g = len(games) or 1
    print("OPS PER GAME")
    for op, k in sorted(((o, v) for o, v in c.items()
                         if o not in ("work", "move", "pass")), key=lambda x: -x[1])[:12]:
        print(f"    {op:<22} {k/g:>8,.0f}")


def probe_sales(games):
    from tools.opp_belief import holdings, harvest_gain, consumption
    from kagent.constants import MARKET_PARAMS
    PRODUCTS = ["MELON", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "WOOL", "EGG",
                "WHEAT", "FERTILIZER"]
    sold = collections.Counter(); rev = collections.Counter()
    for steps, seat, rep in games:
        for t in range(1, len(steps)):
            prev, cur = steps[t - 1], steps[t]
            o = prev[seat].get("observation")
            if not o:
                continue
            px = (prev[0].observation.get("market") or {}).get("prices") or {}
            act = cur[seat].get("action") or {}
            try:
                h0 = holdings(prev[seat].observation); h1 = holdings(cur[seat].observation)
                gn = harvest_gain(prev[seat].observation, act, seat=seat)
                bn = consumption(prev[seat].observation, act, seat=seat)
            except Exception:
                continue
            for it in PRODUCTS:
                n = h0.get(it, 0) + gn.get(it, 0) - bn.get(it, 0) - h1.get(it, 0)
                if n > 0.5:
                    sold[it] += n
                    rev[it] += n * px.get(it, MARKET_PARAMS[it]["base"])
    g = len(games) or 1
    print("SALES (true fills by conservation; order[2] is a REQUEST, not a fill)")
    print(f"    {'item':<12} {'units/game':>11} {'$/unit':>8} {'revenue/game':>13}")
    for it in PRODUCTS:
        if sold[it]:
            print(f"    {it:<12} {sold[it]/g:>11,.0f} {rev[it]/sold[it]:>8,.1f} "
                  f"{rev[it]/g:>13,.0f}")


def probe_byday(games):
    """Efficiency BY DAY. Every other metric here is a season average, which
    hides where the executor actually differs -- and it differs a lot.
    champ-h8 is WORST on days 2-8 (1.79-1.95 moves/action) and IMPROVES to 1.14
    by day 26, while a tape steps down from 1.23 to 0.66 at day 12."""
    work = collections.Counter(); move = collections.Counter(); pas = collections.Counter()
    for steps, seat, rep in games:
        for t in range(1, len(steps)):
            o = steps[t - 1][seat].get("observation")
            if not o:
                continue
            d = o["day"]
            a = steps[t][seat].get("action") or {}
            for x in [a.get("farmer")] + list(a.get("hands") or []):
                op = x[0] if isinstance(x, list) and x else "PASS"
                if op in MOVES:
                    move[d] += 1
                elif op == "PASS":
                    pas[d] += 1
                else:
                    work[d] += 1
    print("BY DAY")
    print(f"  {'day':>4} {'moves/action':>13} {'work share':>12}")
    for d in range(0, 30, 2):
        tot = work[d] + move[d] + pas[d]
        if not tot:
            continue
        print(f"  {d:>4} {move[d]/max(1,work[d]):>13.2f} "
              f"{work[d]/tot*100:>11.1f}%")


def probe_workers(games):
    """Per-worker load spread. Never measured before; turns out NOT to be a
    lever -- a tape rated 1750 runs a 9.1x spread and beats us at 3.0x."""
    per = collections.defaultdict(collections.Counter)
    for steps, seat, rep in games:
        for t in range(1, len(steps)):
            a = steps[t][seat].get("action") or {}
            for i, x in enumerate([a.get("farmer")] + list(a.get("hands") or [])):
                op = x[0] if isinstance(x, list) and x else "PASS"
                per[i]["move" if op in MOVES else
                       ("pass" if op == "PASS" else "work")] += 1
    g = max(1, len(games))
    loads = [per[i]["work"] / g for i in sorted(per) if per[i]["work"]]
    if not loads:
        return
    print("PER-WORKER LOAD (work actions per game; unit 0 = the farmer)")
    print(f"  units {len(loads)}   min {min(loads):,.0f}   "
          f"median {st.median(loads):,.0f}   max {max(loads):,.0f}   "
          f"spread {max(loads)/max(1, min(loads)):.1f}x")


PROBES = {"tape": probe_tape, "budget": probe_budget, "layout": probe_layout,
          "ops": probe_ops, "sales": probe_sales, "byday": probe_byday,
          "workers": probe_workers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="replay directory (downloaded episodes)")
    ap.add_argument("--team", help="TeamNames entry to analyse in those replays")
    ap.add_argument("--agent", help="instead: play one of our agents locally")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=71000)
    ap.add_argument("--probes", default="tape,budget,layout,ops",
                    help="comma-separated: " + ",".join(PROBES))
    args = ap.parse_args()

    if args.agent:
        games = from_engine(args.agent, args.opp, args.n, args.base_seed)
        label = f"{args.agent} vs {args.opp}"
    elif args.dir:
        games = from_replays(args.dir, args.team)[:args.n]
        label = f"{args.team or args.dir}"
    else:
        sys.exit("give --dir/--team (replays) or --agent (play locally)")
    if not games:
        sys.exit("no games found")

    print(f"=== {label} -- {len(games)} games ===\n")
    cached = None
    for name in [p.strip() for p in args.probes.split(",") if p.strip()]:
        fn = PROBES.get(name)
        if not fn:
            print(f"(unknown probe {name})"); continue
        if name == "ops":
            probe_ops(games, cached)
        else:
            r = fn(games)
            if name == "budget":
                cached = r
        print()


if __name__ == "__main__":
    main()
