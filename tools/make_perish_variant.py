"""Price tasks by PERISHABILITY -- what is lost by not doing it today.

The router thesis, in its sharpest form. `v-tile-late` proved that raising
throughput loses money when the extra actions are low value, so the lever is
action VALUE, not action count. The earlier prize-collecting router had the
right idea and the wrong quantity: it priced a task at GROSS revenue, when what
matters is the COST OF DELAY.

Two engine facts settle the cost of delay for the two ops involved:

  * `tile["fertilizer_available"] = True` is set every end-of-day,
    unconditionally, for every animal (kaggriculture.py:831). It is a BOOLEAN,
    not an accumulator, and COLLECT_FERTILIZER clears it. So one unit per animal
    per day is strictly use-it-or-lose-it: delay by a day and it is gone
    forever.
  * `tile["yield_units"] = min(a["max_held"], ...)` (kaggriculture.py:827).
    Animal produce ACCUMULATES to a cap of 4 (goose) or 6 (cow/sheep). Below the
    cap, delaying a HARVEST costs approximately nothing; only at the cap is
    further production silently discarded.

The shipped priorities invert this:

    P_HARVEST = 1      harvest, emitted at held >= 2
    P_TEND    = 2      COLLECT_FERTILIZER

A cow at held=2 of max 6 has four units of headroom -- eight days at interval 2
-- so its harvest can wait indefinitely, while the fertilizer on that same tile
expires tonight. Both tasks sit on the tile the worker is standing on, so this
is not a travel trade-off at all; it is pure ordering, and it is backwards.

Measured consequence: we collect 218 fertilizer a game against ahmed's 324, out
of roughly 464 animal-days available.

    --fert-pri P      priority for COLLECT_FERTILIZER (shipped 2; try 0.5)
    --harvest-slack N harvest keeps P_HARVEST only within N units of the cap;
                      below that it drops to --harvest-pri
    --harvest-pri P   the demoted harvest priority

    PYTHONPATH=. uv run python tools/make_perish_variant.py --name p-fert --fert-pri 0.5
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''        if t.get("fertilizer_available"):
            tasks.append(Task(P_TEND, x, y, ["COLLECT_FERTILIZER"]))
        held = t.get("yield_units", 0)
        max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
        # Harvest every production so yields never clip at max_held.
        if held >= 2 or (held > 0 and (plan.liquidation or day >= 27)):
            tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))'''

REPL = '''        held = t.get("yield_units", 0)
        max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
        if t.get("fertilizer_available"):
            # Strictly perishable: the flag is re-set every end of day and
            # cleared by collection, so an uncollected unit is lost forever.
            tasks.append(Task(FERT_PRI, x, y, ["COLLECT_FERTILIZER"]))
        # Harvest every production so yields never clip at max_held -- but
        # produce ACCUMULATES to the cap, so below it a harvest can wait and
        # only near the cap does delay actually destroy output.
        if held >= 2 or (held > 0 and (plan.liquidation or day >= 27)):
            _urgent = (held >= max_held - HARVEST_SLACK
                       or plan.liquidation or day >= 27)
            tasks.append(Task(P_HARVEST if _urgent else HARVEST_PRI,
                              x, y, ["HARVEST"]))'''

CONSTS = '''FERT_PRI = __FERT__
HARVEST_SLACK = __SLACK__
HARVEST_PRI = __HPRI__

'''


def build(base, name, fert_pri=2.0, slack=0, harvest_pri=2.0):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, REPL, 1)
            text = (CONSTS.replace("__FERT__", repr(float(fert_pri)))
                          .replace("__SLACK__", repr(int(slack)))
                          .replace("__HPRI__", repr(float(harvest_pri)))) + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("animal harvest/fertilizer block not found")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fert-pri", type=float, default=2.0)
    ap.add_argument("--harvest-slack", type=int, default=0,
                    help="0 keeps shipped harvest priority everywhere")
    ap.add_argument("--harvest-pri", type=float, default=2.0)
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.fert_pri, a.harvest_slack, a.harvest_pri)}"
          f"  (fert={a.fert_pri}, slack={a.harvest_slack}, hpri={a.harvest_pri})")
