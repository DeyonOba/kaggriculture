"""rlv3/scorecard_hook.py -- run the whole-agent scorecard on rlv3 AND on
champ-h10, on the SAME recorded worlds, and print the two side by side.

    cd <repo> && PYTHONPATH=. uv run python rlv3/scorecard_hook.py \
        --agent rlv3:full --games 4 --procs 2

WHY A HOOK AND NOT A FLAG ON THE TOOL. `tools/agent_scorecard.py` is the
judge for every part of this agent (docs/agent_scorecard.md) and it is one of
the four directories rlv3 may not edit. Its `parse_spec` knows
`champion | knobs: | bundle: | weights: | execnet:` and does not know
`rlv3:`, and its `_build_agent` hands anything else to
`rl.rollout._make_dev_policy_agent`, which does not know `kind="rlv3"`
either. This module supplies exactly those two missing pieces and then calls
the tool's own `trace_game` / `build_row` / `print_table` / `gap_closed`.
Nothing about a metric is re-implemented here; if a number disagrees with
`tools/agent_scorecard.py` that is a bug in this file.

THREE THINGS THIS FILE EXISTS TO GET RIGHT, each of which has already cost
this repo a run:

  1. THE SAME GAMES. `pick_games(n)` is deterministic and prefix-stable (it
     pops the median entry of band 0, 1, 2, ... in turn), so both agents get
     the identical (world, seat) list and `pick_games(1)` is
     `pick_games(4)[:1]`. Per-world sd is $34,749; an unpaired 4-game money
     comparison is not a comparison.

  2. THE PATCH MUST BE IN THE PLAYING PROCESS. `multiprocessing` starts
     **spawn** here, so `rlv3.arm.register()` in the parent is invisible in a
     worker -- the exact shape of the Stage B board arms, which never trained
     and were read as null for three runs. Registration happens INSIDE
     `_trace_agent`, and the liveness counters ride back so a dead arm cannot
     be mistaken for a bad one.

  3. THE SEAMS MUST COME BACK OFF. `rlv3.selling.install` rebinds
     `kagent.market.build_orders` and `rlv3.executor.install` rebinds the
     scheduler seam; `tools.agent_scorecard._build_agent`'s `restore` only
     puts `kagent.features.MODE` back. A champ-h10 trace that ran after an
     rlv3 trace in the same process would be scored through rlv3's sizer.
     Each agent gets its own process AND `_trace_agent` tears the seams down
     in a `finally`.

TRACES ARE CACHED. `--traces <dir>` (default `rlv3/eval/scorecard_traces`) is
reused when the file on disk names the same world and seat, so re-reading the
table, adding a column or fixing a format string costs zero games. `--fresh`
forces a replay.

WHAT THIS CANNOT SHOW. There is no paired evaluation for any rlv3 arm on
disk, so the OUTCOME group's `win_rate` / `win_band*` columns are empty for
rlv3 and the ones in the champ-h10 column are `ship2_hv`'s from the standing
995-game draw. Trace money over 4 games is not a result: it is the input to
the planner / market / executor metrics, which is what the scorecard is for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_AGENT = "rlv3:full"
DEFAULT_REF = "bundle:arena/agents/champ-h10/main.py"
DEFAULT_REF_EVAL = "rl/runs/draw995/batch2.json:ship2_hv"
DEFAULT_TRACES = os.path.join(REPO, "rlv3", "eval", "scorecard_traces")


# ------------------------------------------------------------------- specs
def parse_agent(text):
    """`rlv3[:<preset>[:<overrides>]]` -> an rlv3 spec; anything else goes to
    `tools.agent_scorecard.parse_spec` untouched.

    `exec` is put on the spec because `trace_game` reads it and applies it to
    `scheduler.EXEC` itself -- the same field `rlv3.agent.make_job` puts on a
    rollout job for the same reason. `config_from_spec` skips the key, so the
    spec still validates.
    """
    import tools.agent_scorecard as AS

    text = (text or "").strip()
    if text != "rlv3" and not text.startswith("rlv3:"):
        return AS.parse_spec(text)
    from rlv3 import agent as AGENT
    from rlv3 import arm as ARM

    spec = ARM.parse(text[5:] if text.startswith("rlv3:") else "")
    spec["exec"] = AGENT.config_from_spec(spec).all_exec()
    return spec


def label_of(spec):
    import tools.agent_scorecard as AS

    if spec.get("kind") == "rlv3":
        from rlv3 import agent as AGENT
        return AGENT.config_from_spec(spec).label
    return AS.spec_label(spec)


def _tag(label):
    return "".join(c if c.isalnum() else "_" for c in label)


# ------------------------------------------------------------------ worker
def _usable(path, session_path, script_side):
    """A cached trace counts only if it is THIS world in THIS seat."""
    try:
        with open(path) as fh:
            t = json.load(fh)
    except (OSError, ValueError):
        return False
    return (t.get("session") == os.path.basename(session_path)
            and t.get("tape_seat") == script_side
            and len(t.get("steps") or []) > 600)


def _trace_agent(payload):
    """Trace one agent over every game. RUNS IN ITS OWN PROCESS.

    Returns a plain dict (it has to pickle back): trace paths, the rlv3
    liveness counters, the executor seam's fired-lever counts, and the first
    traceback if anything blew up. It never raises, so one agent failing
    still prints the other agent's scorecard -- which is the difference
    between "rlv3 cannot be scored" and "nothing can be scored".
    """
    spec = payload["spec"]
    games = payload["games"]
    workdir = payload["workdir"]
    tag = payload["tag"]
    fresh = payload["fresh"]
    out = {"tag": tag, "label": payload["label"], "paths": [], "per_game": [],
           "counters": {}, "exec_stats": {}, "error": None, "wall_s": 0.0}
    t0 = time.time()
    AGENT = EXEC = None
    try:
        import tools.agent_scorecard as AS

        is_rlv3 = isinstance(spec, dict) and spec.get("kind") == "rlv3"
        if is_rlv3:
            from kagent import scheduler
            from rlv3 import agent as AGENT          # noqa: F811
            from rlv3 import arm as ARM
            from rlv3 import executor as EXEC        # noqa: F811
            # register() BEFORE the first trace_game: trace_game opens with
            # `reset_exec(); set_exec(spec["exec"])`, and set_exec raises on a
            # flag it has never heard of. reset_exec only zeroes the keys that
            # are already there, so registering first is enough and is what
            # makes trace_game's own EXEC restore put back an rlv3-aware dict.
            EXEC.register(scheduler)
            ARM.register()
            AGENT.reset_counters()
        os.makedirs(workdir, exist_ok=True)
        for i, (sess, side, band) in enumerate(games):
            path = os.path.join(workdir, f"{tag}_{i}_b{band}.json")
            if not fresh and _usable(path, sess, side):
                out["paths"].append(path)
                out["per_game"].append({"i": i, "band": band, "cached": True})
                continue
            js = dict(spec)
            if is_rlv3 and "band" not in spec:
                # objective.p_win is calibrated per rating band; arm.make_jobs
                # puts the band on the spec for the same reason.
                js["band"] = int(band)
            before = AGENT.counters().get("make_plan", 0) if is_rlv3 else 0
            g0 = time.time()
            if is_rlv3:
                EXEC.reset_stats()
            try:
                AS.trace_game(js, sess, side, path)
            finally:
                if is_rlv3:
                    # read the seam's counters BEFORE uninstall resets them
                    for k, v in EXEC.stats().items():
                        if v:
                            out["exec_stats"][k] = out["exec_stats"].get(k, 0) + v
                    a = AGENT.last_agent()
                    if a is not None and getattr(a, "restore", None):
                        a.restore()
            made = (AGENT.counters().get("make_plan", 0) - before) if is_rlv3 else None
            if is_rlv3 and made < 29:
                raise RuntimeError(
                    f"rlv3 played {made} plans on game {i} (a 30-day season "
                    f"makes 30): the provider was not the agent that played. "
                    f"DEAD, not null -- do not read this row.")
            out["paths"].append(path)
            out["per_game"].append({"i": i, "band": band, "cached": False,
                                    "plans": made, "wall_s": round(time.time() - g0, 1)})
        if is_rlv3:
            out["counters"] = AGENT.counters()
    except Exception:                                    # noqa: BLE001
        out["error"] = traceback.format_exc()
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ report
def identity(paths_a, paths_b):
    """Do two agents play the SAME game? -> (identical steps, total steps).

    The strong form of the check the package is built around. `rlv3:base` is
    every deviation at zero dose and must reproduce its baseline exactly; a
    matching final bank on one world is weak (any deterministic agent playing
    itself matches), a matching ACTION at all 719 steps of every world is not.
    It is also the only way to tell "the reference row is the same agent" from
    "the reference row happens to be close", which decides whether the delta
    column is rlv3's effect or a tree difference.
    """
    same = tot = 0
    for pa, pb in zip(paths_a, paths_b):
        try:
            A, B = json.load(open(pa)), json.load(open(pb))
        except (OSError, ValueError):
            return None
        if A.get("seed") != B.get("seed") or A.get("our_seat") != B.get("our_seat"):
            return None
        for x, y in zip(A["steps"], B["steps"]):
            tot += 1
            same += 1 if x["act"]["me"] == y["act"]["me"] else 0
    return (same, tot) if tot else None


def delta_row(agent, ref, keys):
    """agent - ref on every metric both rows carry. Printed in the same table
    so the two scorecards read as one."""
    d = {}
    for k in keys:
        a, r = agent.get(k), ref.get(k)
        if isinstance(a, (int, float)) and isinstance(r, (int, float)):
            d[k] = a - r
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=DEFAULT_AGENT,
                    help="rlv3[:preset[:overrides]] or any tools/agent_scorecard.py spec")
    ap.add_argument("--reference", default=DEFAULT_REF)
    ap.add_argument("--extra", action="append", default=[],
                    help="another agent as a third column (e.g. rlv3:base, the identity arm)")
    ap.add_argument("--games", type=int, default=4, help="worlds, shared by every agent")
    ap.add_argument("--procs", type=int, default=2, help="agents traced at once; capped at 2")
    ap.add_argument("--eval", default=None,
                    help="<paired eval json>:<arm> for the AGENT's OUTCOME row (none exists for rlv3)")
    ap.add_argument("--reference-eval", default=DEFAULT_REF_EVAL)
    ap.add_argument("--traces", default=DEFAULT_TRACES)
    ap.add_argument("--fresh", action="store_true", help="re-play cached traces")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, play nothing")
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    args = ap.parse_args(argv)

    import tools.agent_scorecard as AS

    procs = max(1, min(2, args.procs))          # the box is busy; 2 is the ceiling
    games = AS.pick_games(args.games)
    specs = [("agent", args.agent, args.eval), ("ref", args.reference, args.reference_eval)]
    specs += [(f"extra{i}", t, None) for i, t in enumerate(args.extra)]

    plan = []
    for role, text, ev in specs:
        spec = parse_agent(text)
        lab = label_of(spec)
        plan.append({"role": role, "text": text, "spec": spec, "label": lab,
                     "tag": _tag(lab), "eval": ev})

    print(f"{len(games)} games, bands {[b for _, _, b in games]}, "
          f"{len(plan)} agents, procs {procs}", file=sys.stderr)
    for p in plan:
        cached = sum(1 for i, (s, sd, b) in enumerate(games)
                     if _usable(os.path.join(args.traces, f"{p['tag']}_{i}_b{b}.json"), s, sd))
        print(f"  {p['role']:<6} {p['label']:<16} {p['text']}"
              f"   [{cached}/{len(games)} traces cached]", file=sys.stderr)
    if args.dry_run:
        for s, sd, b in games:
            print(f"  band {b}  seat {1 - sd}  {os.path.relpath(s, REPO)}", file=sys.stderr)
        return 0

    payloads = [{"spec": p["spec"], "games": games, "workdir": args.traces,
                 "tag": p["tag"], "label": p["label"], "fresh": args.fresh} for p in plan]
    t0 = time.time()
    if procs > 1 and len(payloads) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            results = pool.map(_trace_agent, payloads)
    else:
        results = [_trace_agent(p) for p in payloads]
    print(f"tracing done in {time.time() - t0:.0f}s", file=sys.stderr)

    built, dead = {}, []
    for p, r in zip(plan, results):
        p["result"] = r
        if r["error"] or len(r["paths"]) < len(games):
            dead.append((p["label"], r["error"] or "no traces"))
            continue
        built[p["label"]] = AS.build_row(r["paths"], p["eval"])

    # Column order: the agent, then each comparison agent immediately followed
    # by its own delta column. A delta against the FROZEN champ-h10 bundle and
    # a delta against `rlv3:base` are different questions -- the bundle is a
    # snapshot of an older kagent/ and finishes $11,638 from today's tree on
    # world 770000, so only the delta against the identity arm is rlv3's own
    # effect. Printing one without the other invites reading the tree
    # difference as a result.
    agent_lab, ref_lab = plan[0]["label"], plan[1]["label"]
    keys = [k for ks in AS.GROUPS.values() for k in ks]
    rows = {}
    if agent_lab in built:
        rows[agent_lab] = built[agent_lab]
    for p in plan[1:]:
        lab = p["label"]
        if lab not in built:
            continue
        rows[lab] = built[lab]
        if agent_lab in built:
            rows["d:" + lab.replace("rlv3:", "")] = delta_row(
                built[agent_lab], built[lab], keys)
    leaders = AS.leader_metrics()
    dev_tree = leaders.pop("_dev_tree", None)
    for k, m in AS.leader_planner().items():
        if k in leaders:
            leaders[k].update({kk: vv for kk, vv in m.items() if vv is not None})
    rows.update(leaders)

    AS.print_table(rows)

    print("\n== LIVENESS " + "=" * 52)
    for p in plan:
        r = p["result"]
        c = r["counters"]
        played = [g for g in r["per_game"] if not g.get("cached")]
        is_rlv3 = p["spec"].get("kind") == "rlv3"
        if c:
            bits = (f"plans {c.get('make_plan', 0)} dawns {c.get('dawn', 0)} "
                    f"observed {c.get('observe', 0)} projected {c.get('project', 0)} "
                    f"sell-marked {c.get('sell_marked', 0)}")
        elif is_rlv3:
            # Cached traces are read, not replayed, so there is nothing to
            # count. Saying "not an rlv3 arm" here would be a false negative
            # on the one check this package exists to make; --fresh re-plays.
            bits = "all traces cached -- no liveness counted this run (--fresh to replay)"
        else:
            bits = "(not an rlv3 arm)"
        print(f"  {p['label']:<16} {len(r['paths'])}/{len(games)} traces, "
              f"{len(played)} played, {r['wall_s']:.0f}s   {bits}")
        if r["exec_stats"]:
            print(f"  {'':<16} exec seam fired: {r['exec_stats']}")
        elif p["spec"].get("kind") == "rlv3" and p["spec"].get("exec_flags"):
            print(f"  {'':<16} exec seam fired NOTHING -- the levers are inert")
        if r["error"]:
            print(f"  {'':<16} FAILED:\n" + r["error"].rstrip())

    ok = [p for p in plan if p["label"] in built]
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            idn = identity(a["result"]["paths"], b["result"]["paths"])
            if idn and idn[0] == idn[1]:
                print(f"  IDENTITY: {a['label']} and {b['label']} play the SAME "
                      f"game -- {idn[1]} of {idn[1]} actions equal over "
                      f"{len(games)} worlds. Their columns are one agent, and a "
                      f"delta against either is the same delta.")
            elif idn:
                print(f"  {a['label']} vs {b['label']}: {idn[0]}/{idn[1]} "
                      f"actions equal ({idn[0] / idn[1]:.1%})")

    if agent_lab not in built or ref_lab not in built:
        print("\nno side-by-side: " + "; ".join(f"{l}: {e.splitlines()[-1] if e else e}"
                                                for l, e in dead))
        return 1

    gaps = {}
    for p in plan[1:]:
        anchor = p["label"]
        if anchor not in built:
            continue
        print(f"\n== GAP TO THE LEADERS, anchored on {anchor} " + "=" * 20)
        print(f"share of the {anchor} -> best-leader distance {agent_lab} has closed")
        print("(0.00 = level with that anchor, 1.00 = level with the best leader)")
        g = {}
        for group, gkeys in AS.GROUPS.items():
            frac, detail = AS.gap_closed(rows[agent_lab], built[anchor], leaders, gkeys)
            g[group] = frac
            if frac is None:
                print(f"  {group:<10} n/a")
                continue
            worst = sorted(detail, key=lambda kv: kv[1])[:3]
            best = sorted(detail, key=lambda kv: -kv[1])[:2]
            print(f"  {group:<10} {frac:+.2f}   behind: "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in worst)
                  + "   |  ahead: " + ", ".join(f"{k} {v:+.2f}" for k, v in best))
        live = {k: v for k, v in g.items() if v is not None}
        if live:
            lim = min(live, key=live.get)
            print(f"  limiting dimension: {lim} ({live[lim]:+.2f})")
        gaps[anchor] = g

    print("\nfurthest from the leaders' mean, regardless of anchor:")
    for group, keys in AS.GROUPS.items():
        far = AS.furthest_from_leaders(rows[agent_lab], leaders, keys, n=3)
        if far:
            print(f"  {group:<10} " + ",  ".join(
                f"{k} {a:,.2f} vs {t:,.2f} ({rel:+.0%})" for k, a, t, rel in far))

    print("\n== WHAT THIS TABLE IS AND IS NOT " + "=" * 31)
    n = len(games)
    bands = sorted({b for _, _, b in games})
    print(f"  * {n} worlds, bands {bands}. pick_games walks the bands in order, so "
          f"{n} games\n    never reaches band 4 (2000-2500) or 5 (2500+) -- and band 5 is the one\n"
          "    the repo measured as HARDEST to predict (R2 -0.019 against +0.239 at <1000).")
    print("  * MONEY AND MARGIN ARE NOT A RESULT HERE. Per-world sd is $34,749, so the\n"
          f"    standard error on a {n}-game mean margin is about ${34749 / n ** 0.5:,.0f}. Worse, any\n"
          "    board difference re-rolls the daily shop draw, so the OPPONENT's bank moves\n"
          "    too and a margin change cannot be attributed. The verdict instrument is\n"
          "    rl/session_eval.py, 995 paired games; nothing in rlv3 has been through one.")
    print("  * The structural metrics are what this table is for: a day-20 crop count, a\n"
          "    plant/fertilise/harvest op count, chores per visit, idle share and order\n"
          "    impact are per-season counts, not one draw, and they move far less.")
    print("  * OUTCOME win_rate is EMPTY for any rlv3 column: no paired evaluation of an\n"
          "    rlv3 arm exists on disk. The champ-h10 win rate is ship2_hv's from the\n"
          "    standing 995-game draw (rl/runs/draw995/batch2.json), attached the way\n"
          "    docs/results/scorecard_champ_h10.txt attaches it.")
    print("  * Leader columns are their OWN games against their OWN opponents (cached in\n"
          "    docs/results/top3_*.json), so their money is not comparable to ours and\n"
          "    they carry no win-rate column at all.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "gaps": gaps, "agent": args.agent,
                       "reference": args.reference, "games": args.games,
                       "liveness": {p["label"]: {k: p["result"][k] for k in
                                                 ("counters", "exec_stats", "per_game", "wall_s")}
                                    for p in plan},
                       "world_list": [(os.path.basename(s), sd, b) for s, sd, b in games]},
                      fh, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
