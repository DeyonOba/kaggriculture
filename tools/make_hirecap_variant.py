"""What is a hand actually WORTH? Cap the schedule and read the loss.

This bounds a direction before building it. The proposal on the table is that
better routing lets us run FEWER HANDS, which pays through the fibonacci wage
curve rather than through freed turns -- and freed turns are the thing already
measured at ~$0 (h-front added 1,440 actions for -$653).

The prize is exactly computable on the wage side:

    hands      6      8      9     10     11
    $/day     20     54     88    143    232

CHAMP_HIRES peaks at 11, so we sit ON the knee and never above it. Cutting
the peak to 8 saves $178/day, about $2,670 a season against a median margin
of $4,413.

So the gate here reads directly as the PRODUCTIVITY of a hand:

    loss < wage saved   ->  hands are idle, routing is already adequate,
                            and the whole direction is worth under $3k
    loss > wage saved   ->  hands are productive, so better routing could
                            substitute for them and the prize is real

Either answer is worth more than a router built on a guess. --cap 11 is
bit-identical to the base, since the schedule never exceeds 11.

    PYTHONPATH=. uv run python tools/make_hirecap_variant.py --name w-cap8 --cap 8
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    plan.hires_target = min(schedule, max(3, workload // 3))'''
REPL = '''    plan.hires_target = min(schedule, max(3, workload // 3), _HIRE_CAP)'''

HEAD = '''
# --- hand-value probe ------------------------------------------------------
# Hire cost is fibonacci: 8 hands $54/day, 11 hands $232/day. CHAMP_HIRES
# peaks at 11, so capping lower saves a known amount and the gate reads as
# the productivity of the hands given up.
_HIRE_CAP = __CAP__

'''


def season_wage(cap):
    """Total wages over the champion schedule, capped."""
    from kagent import econ
    import kagent.planner as P
    sched = list(P.CHAMP_HIRES)
    return sum(econ.hire_total_cost(min(h, cap)) for h in sched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cap", type=int, required=True,
                    help="ceiling on hires_target; 11 = base, bit-identical")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("hires_target anchor not found in planner")
            text = (HEAD.replace("__CAP__", repr(args.cap))
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
    base_w, capped_w = season_wage(99), season_wage(args.cap)
    print(f"{args.name}: cap {args.cap}   season wages "
          f"${capped_w:,.0f} vs ${base_w:,.0f}   SAVES ${base_w-capped_w:,.0f}"
          f"   -> a loss bigger than that means the hands were earning it")


if __name__ == "__main__":
    main()
