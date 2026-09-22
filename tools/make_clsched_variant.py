"""Close the loop on the phase-B hiring schedule: read the board, not the calendar.

WHAT THE MEASUREMENT SAYS. `tools/phase_timing.py` over 514 replayed ladder
sessions (`docs/phi_ood_and_phase_timing.md` §2):

    landmark      ladder(1028)  2500+  M&M&P&Q   champ-h8
    build_done      2.0 [1-3]    1.0   1.0        **10.0** [10-12]
    peak_board      9.0 [9-12]   9.0   9.0        **21.0** [20-22]

We finish placing capital on day 10 where the field finishes on day 2, and we
peak the board on day 21 against a field median of 9 -- on the one landmark
that correlates with rating at all (Spearman **-0.361**; the other three are
-0.02 to -0.09). The entire gap sits inside phase B, days 7-17, which
`docs/game_phases.md` already identified as the only phase where an
intervention can still change a result for us: our P(win) falls 0.511 -> 0.084
across it.

WHAT THE PLANNER ACTUALLY DOES. Hiring is an open-loop calendar with a
ceiling:

    schedule = CHAMP_HIRES[day] + (hires_delta if day >= 6 else 0)
    workload = sum(crops.values()) + 2 * my_animals + 10
    plan.hires_target = min(schedule, max(3, workload // 3))

`CHAMP_HIRES` is 30 fixed numbers. `workload` counts work that ALREADY EXISTS
-- standing crops and animals -- and enters only as a cap. Three lines above,
the same function computes `empty_tiles` and `empty_by_kind`; **neither is
ever read by the hiring decision.** So the schedule cannot see unfilled ground
and cannot see a harvest about to land. It is the same 30 numbers whether the
board is empty or full.

THE RULE. In phase B only, add hands for work that is ARRIVING:

    arriving = ripe_soon + FILL_W * fillable
    schedule += round(BETA * arriving)

where `ripe_soon` counts my plant tiles already carrying yield or within
`HORIZON` days of `max_yield_day`, and `fillable` counts empty ground and
empty structures I could still stock. Two prior measurements pick the shape:
hands track **harvests r=0.764**, not plantings r=0.300, and the two are
independent (r=0.052) -- so harvest pressure leads, and unfilled ground enters
at a discount.

WHY THIS IS NOT THE REFUTED HIRING KNOB. `hires_delta` is a CONSTANT offset
across the whole season and is refuted, monotonically: +2 hands -$5,229, +4
-$19,344. The rollout showed the same lever pays as a per-day SCHEDULE
(+$3,287 +/- 1,528) -- "the knob space was never the plan space". This goes one
step further: the schedule becomes a function of the board instead of 30
frozen numbers. It is arithmetic on quantities the planner already computes,
which is the only kind of change that has ever gated positive here.

WHY IT DOES NOT READ Phi. It was going to. `rl/datasets/potential.json` scores
+0.381 R^2 in self-play and **-0.398 on replayed ladder games** -- worse than
predicting the mean, and below the two-line zero-fit rule it was meant to
beat. A potential fitted on self-play describes self-play. So the closed-loop
signal here is raw board arithmetic, with no fitted model anywhere in it.

    PYTHONPATH=. uv run python tools/make_clsched_variant.py --name x-clb10 --beta 1.0
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIND = '''    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)
    workload = sum(crops.values()) + 2 * my_animals + 10
    plan.hires_target = min(schedule, max(3, workload // 3))'''

REPL = '''    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)
    workload = sum(crops.values()) + 2 * my_animals + 10
    # CLOSED LOOP, PHASE B ONLY. The calendar above cannot see the board:
    # `workload` counts work that already exists and enters only as a cap,
    # and `empty_tiles` is computed three lines up and never read. We finish
    # placing capital on day 10 where the ladder finishes on day 2, and peak
    # the board on day 21 against a field median of 9 -- the one landmark
    # that correlates with rating (Spearman -0.361). So in days
    # _CLB_LO.._CLB_HI, add hands for the work that is ARRIVING: harvests
    # about to land (hands track harvests r=0.764) plus, at a discount,
    # ground still to fill (plantings r=0.300, independent at r=0.052).
    if _CLB_LO <= day <= _CLB_HI and _CLB_BETA:
        _ripe = 0
        for _x, _y, _t in me.plants():
            _c = CROPS.get(_t.get("crop"))
            if not _c:
                continue
            if _t.get("yield_units", 0) > 0:
                _ripe += 1
            elif day - _t.get("planted_day", day) >= _c["max_yield_day"] - _CLB_H:
                _ripe += 1
        _fillable = empty_tiles + empty_by_kind["COOP"] + empty_by_kind["PASTURE"]
        _arriving = _ripe + _CLB_FILL * _fillable
        schedule += int(round(_CLB_BETA * _arriving))
    plan.hires_target = min(schedule, max(3, workload // 3))'''


def build(base: str, name: str, beta: float, lo: int, hi: int,
          horizon: int, fill: float) -> None:
    head_block = (f"\n_CLB_BETA = {beta}\n"
                  f"_CLB_LO = {lo}\n"
                  f"_CLB_HI = {hi}\n"
                  f"_CLB_H = {horizon}\n"
                  f"_CLB_FILL = {fill}\n\n")

    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], False
    for mod, text in mods:
        if mod == "planner" and FIND in text:
            text = head_block + text.replace(FIND, REPL, 1)
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit(f"{base} has no CHAMP_HIRES hiring block to patch")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    rules = os.path.join(REPO, "arena", "agents", base, "rules.json")
    if os.path.isfile(rules):
        import shutil
        shutil.copyfile(rules, os.path.join(dst_dir, "rules.json"))

    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)")
    print(f"  beta={beta} days {lo}-{hi} horizon={horizon} fill_weight={fill}")
    if beta == 0.0:
        print("  beta=0 -- must be bit-identical to the base by construction")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="x-fut10")
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="hands per unit of arriving work")
    ap.add_argument("--lo", type=int, default=7, help="phase B first day")
    ap.add_argument("--hi", type=int, default=17, help="phase B last day")
    ap.add_argument("--horizon", type=int, default=2,
                    help="days ahead a crop counts as about to ripen")
    ap.add_argument("--fill", type=float, default=0.5,
                    help="weight on unfilled ground vs arriving harvest")
    args = ap.parse_args()
    build(args.base, args.name, args.beta, args.lo, args.hi,
          args.horizon, args.fill)


if __name__ == "__main__":
    main()
