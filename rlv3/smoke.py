"""rlv3/smoke.py -- prove the wired agent RUNS, and that it is the one running.

    cd <repo> && PYTHONPATH=. uv run python rlv3/smoke.py --presets base,full

Two things are checked, and only the second is about a number:

  LIVENESS. The rlv3 plan provider must be called, 30 times, once per dawn.
  A spec that says `rlv3` and plays the shipped heuristic is DEAD, not null,
  and is indistinguishable from a null result once it is a row in a table.
  `rlv3.arm.run_episode` raises on `make_plan == 0`; this script additionally
  prints the whole counter block and fails on a short count.

  IDENTITY. `rlv3:base` is every deviation at zero dose. It must reproduce
  `ship2_hv` -- the arm rl/session_eval.py actually pairs against at 0.354 --
  to the DOLLAR on the same recorded world. Not "close", not "margin ~ 0":
  any deterministic agent playing itself satisfies a zero margin, so the test
  has to be exact final money on both sides. The reference comes out of
  rl/runs/draw995/batch2.json, the standing draw.

A NON-EMPTY `exec_stats` IS NOT LIVENESS. The ledger's kill switch records
itself as `<flag>.killed_day<d>` in the SAME dict, so a season in which
`co_chain_tile` fired zero reorders and was switched off on day 3 still
arrives here as a non-empty dict and used to pass. `lever_counters()` strips
the ledger's own bookkeeping keys, so every DECLARED flag has to show a
counter it earned by ACTING, and `days_live` off the episode row has to
cover more than the opening of the season.

ONE GAME IS NOT A RESULT. Per-world sd is $34,749; a two-game money
difference between presets is noise with a dollar sign on it and is printed
only to show the arm plays. The verdict instrument is a >= 995-game paired
rl/session_eval.py with McNemar on the discordant games.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rlv3 import agent as AGENT        # noqa: E402
from rlv3 import arm as ARM            # noqa: E402

BATCH2 = os.path.join(REPO, "rl", "runs", "draw995", "batch2.json")
REF_ARM = "ship2_hv"

# `<flag>.<suffix>` keys the LEDGER writes about a lever rather than the
# lever writing about itself. A lever that only ever appears through one of
# these did nothing all season.
BOOKKEEPING = ("killed_day", "revived_day", "failed")

# A lever live only over the opening is a day-0 intervention priced as a
# season. Half the season is the floor worth reporting on.
MIN_DAYS_LIVE_FRACTION = 0.5


def lever_counters(stats, flag):
    """The counters that mean `flag` ACTED, bookkeeping keys removed."""
    out = {}
    for k, v in (stats or {}).items():
        if not v or not k.startswith(flag + "."):
            continue
        suffix = k.split(".", 1)[1]
        if any(suffix.startswith(b) for b in BOOKKEEPING):
            continue
        out[k] = v
    return out


def lever_liveness(exec_stats, rlv3_row, flags):
    """-> [complaint] for every declared flag that did not really play."""
    bad = []
    days = (rlv3_row or {}).get("days") or 0
    live = (rlv3_row or {}).get("days_live") or {}
    for f in flags:
        acted = lever_counters(exec_stats, f)
        if not acted:
            bad.append(f"{f} fired NOTHING all season "
                       f"(only {sorted(k for k in (exec_stats or {}) if k.startswith(f + '.'))})")
            continue
        n = live.get(f, days)
        if days and n < MIN_DAYS_LIVE_FRACTION * days:
            bad.append(f"{f} was live for {n} of {days} days -- the ledger "
                       f"switched it off on day {n}, so the rest of the "
                       f"season played the SHIPPED executor")
    return bad


def reference(path, script_side, arm=REF_ARM, blob=BATCH2):
    """The standing draw's row for this world and seat, or None."""
    with open(blob) as fh:
        rows = json.load(fh)["results"].get(arm) or []
    rel = os.path.relpath(path, REPO)
    for r in rows:
        if r["path"] == rel and r["script_side"] == script_side:
            return r
    return None


def play(spec, path, script_side, band=None):
    """One episode. Executor counters are read out of the seam AFTER the
    episode and before `arm.run_episode`'s teardown resets them -- a
    zero-counter arm with a flag on is a lever that did not fire, which is the
    same DEAD-not-null failure one level down."""
    from rlv3 import executor as EXEC

    job = AGENT.make_job(spec, path, script_side, seed=0, temperature=0.0)
    if band is not None:
        job["learner_policy"]["band"] = int(band)
    EXEC.reset_stats()
    t0 = time.time()
    out = ARM.run_episode(job)
    out["wall_s"] = time.time() - t0
    out["exec_stats"] = {k: v for k, v in EXEC.stats().items() if v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="base,full",
                    help="comma-separated rlv3 arm strings, one episode each")
    ap.add_argument("--games", type=int, default=1,
                    help="recorded worlds; TOTAL episodes = games * presets")
    ap.add_argument("--same-world", action="store_true", default=True,
                    help="play every preset on the same world (paired)")
    args = ap.parse_args()

    from tools.agent_scorecard import pick_games

    games = pick_games(args.games)
    specs = [ARM.parse(s.strip()) for s in args.presets.split(",") if s.strip()]
    n = len(games) * len(specs)
    print(f"rlv3 smoke: {len(specs)} preset(s) x {len(games)} world(s) = {n} episode(s)")
    print(f"objective fingerprint {AGENT.OBJ.fingerprint()}")
    print()

    rows, failures = [], []
    for path, ss, band in games:
        ref = reference(path, ss)
        world = os.path.basename(path)
        print(f"world {world}  script_side={ss}  band={band}")
        if ref:
            print(f"  reference {REF_ARM:>10}: money {ref['money']:>9,.0f}  "
                  f"opp {ref['opp']:>9,.0f}  margin {ref['money'] - ref['opp']:>+9,.0f}"
                  f"  win {ref['win']:.0f}")
        for spec in specs:
            AGENT.reset_counters()
            label = AGENT.config_from_spec(spec).label
            try:
                out = play(spec, path, ss, band=band)
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"{label} on {world}: {type(exc).__name__}: {exc}")
                print(f"  {label:<18} CRASHED: {type(exc).__name__}: {exc}")
                continue
            c = out["extra"]["rlv3"]["counters"]
            m, o = out["final_money"], out["opp_money"]
            rows.append({"preset": label, "world": world, "side": 1 - ss,
                         "money": m, "opp": o, "margin": m - o, "win": out["win"],
                         "counters": c, "exec_stats": out["exec_stats"],
                         "wall_s": round(out["wall_s"], 1)})
            print(f"  {label:<18} money {m:>9,.0f}  opp {o:>9,.0f}  "
                  f"margin {m - o:>+9,.0f}  win {out['win']:.0f}  "
                  f"[{out['wall_s']:.0f}s]")
            print(f"  {'':<18} plans {c.get('make_plan', 0)}  dawns {c.get('dawn', 0)}"
                  f"  steps-observed {c.get('observe', 0)}"
                  f"  projections {c.get('project', 0)}"
                  f"  sell-marked {c.get('sell_marked', 0)}"
                  f"  statuses {set(out['statuses'])}")
            cfg = AGENT.config_from_spec(spec)
            if cfg.exec_flags:
                print(f"  {'':<18} exec {cfg.exec_flags} -> {out['exec_stats'] or 'NOTHING FIRED'}")
                rlv3_row = out.get("extra", {}).get("rlv3", {})
                if rlv3_row.get("days_live") is not None:
                    print(f"  {'':<18} days live {rlv3_row['days_live']} "
                          f"of {rlv3_row.get('days')}")
                for complaint in lever_liveness(out["exec_stats"], rlv3_row,
                                                cfg.exec_flags):
                    failures.append(f"{label} on {world}: {complaint}")
            if c.get("make_plan", 0) < 29:
                failures.append(f"{label} on {world}: only {c.get('make_plan')} "
                                f"plans; a 30-day season makes 30")
            if any(s != "DONE" for s in out["statuses"]):
                failures.append(f"{label} on {world}: statuses {out['statuses']}")
            if label == "rlv3:base" and ref is not None:
                if (round(m) != round(ref["money"])
                        or round(o) != round(ref["opp"])):
                    failures.append(
                        f"IDENTITY BROKEN: rlv3:base on {world} finished "
                        f"{m:,.0f}/{o:,.0f}, {REF_ARM} finished "
                        f"{ref['money']:,.0f}/{ref['opp']:,.0f}")
                else:
                    print(f"  {'':<18} IDENTITY OK: exactly {REF_ARM}")
        print()

    print(json.dumps(rows, indent=1))
    if failures:
        print("\nFAILURES")
        for f in failures:
            print("  " + f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
