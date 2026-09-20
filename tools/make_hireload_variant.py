"""Hire against the day's HARVEST load, not against the day number.

CHAMP_HIRES is a fixed 30-entry array indexed by day. Measured over 400 games
at hour 12 (the dawn frame is useless -- _end_of_day clears both `hands` and
`hires_today`), hand supply tracks one demand stream and not the other:

    hands vs HARVESTS      r = +0.764
    hands vs PLANTINGS     r = +0.300
    plantings vs harvests  r = +0.052

The two streams are INDEPENDENT across days, so no scalar indexed by day can
serve both. That makes the schedule the wrong SHAPE, not merely mistuned --
which is a structural argument rather than another correlation to tune, and
is why this is worth trying after melon and geese both failed.

The gap is visible as labour per unit of pending work:

    day          0      1     11     12     15
    hands     4.77   2.71  11.12   9.83  11.46
    work     17.49   0.77  23.66  17.57  23.76
    ratio     0.27   3.52   0.47   0.56   0.48

Day 1 hires 3.5 hands per unit of work; day 0 hires 0.27. A 13x swing.

WHERE THE HOOK HAS TO GO. The planner computes

    schedule = CHAMP_HIRES[day] + knobs["hires_delta"]
    workload = sum(crops) + 2 * my_animals + 10
    hires_target = min(schedule, max(3, workload // 3))

`workload` already omits harvest and pickup, so the obvious fix is to add
them there -- and it would be INERT, because at day 11 that expression is
min(11, 15) and the SCHEDULE binds. Adding to the losing side of a min()
changes nothing. So the harvest term is added to `schedule` instead, and can
only ever raise it.

    extra = alpha * max(0, harvest_load - REF) / 3
    schedule = CHAMP_HIRES[day] + extra

alpha=0 is bit-identical to the base by construction.

    PYTHONPATH=. uv run python tools/make_hireload_variant.py --name w-load100 --alpha 1.0
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)'''

REPL = '''    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)
    if _HIREA > 0:
        schedule += _hire_extra(view, day)'''

HEAD = '''
# --- hire against harvest load ---------------------------------------------
# Hand supply tracks HARVESTS (r=+0.764) and not PLANTINGS (+0.300), and the
# two demand streams are independent across days (r=+0.052), so a schedule
# indexed by day cannot serve both. This raises the schedule on days that
# actually carry harvest work; it can never lower it.
_HIREA = __ALPHA__
_HIRE_REF = __REF__          # harvest load already covered by the flat schedule
_HIRE_MAX = __MAX__          # ceiling on the extra hands


def _hire_extra(view, day):
    """Hands to add for today's pending HARVEST and PICKUP work.

    A ripe crop tile costs a HARVEST and later a haul; an animal holding
    yield costs a PICKUP. Neither appears in the planner's `workload`.
    """
    load = 0
    for _x, _y, t in view.me.plants():
        cd = CROPS.get(t["crop"])
        if cd is None:
            continue
        if t.get("yield_units", 0) > 0:
            load += 1
        elif day - t.get("planted_day", day) >= cd["first_yield_day"] - 1:
            load += 1                      # ripens today or tomorrow
    for _x, _y, t in view.me.animals():
        if t.get("yield_units", 0) > 0:
            load += 1
    extra = int(round(_HIREA * max(0, load - _HIRE_REF) / 3.0))
    return max(0, min(_HIRE_MAX, extra))

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="strength of the harvest term; 0 = base, bit-identical")
    ap.add_argument("--ref", type=int, default=6,
                    help="harvest load the flat schedule already covers")
    ap.add_argument("--max", type=int, default=5,
                    help="ceiling on extra hands")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    by_mod = dict(mods)
    if "CROPS =" not in by_mod.get("constants", ""):
        raise SystemExit("constants does not define CROPS")

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("hiring anchor not found in planner")
            text = (HEAD.replace("__ALPHA__", repr(args.alpha))
                        .replace("__REF__", repr(args.ref))
                        .replace("__MAX__", repr(args.max))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no planner module in bundle")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + harvest-reactive hiring "
          f"alpha={args.alpha}, ref={args.ref}, max +{args.max}")


if __name__ == "__main__":
    main()
