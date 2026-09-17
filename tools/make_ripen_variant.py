"""Win the ripening race with FERTILIZER, not with a harvest burst.

The melon gap is not throughput. The engine settles it:

    window_start = (max_yield_day + 1) // 2          # melon: (12+1)//2 = 6
    bonus = 2 if tile["fertilized_until_day"] >= day else 1
    tile["yield_units"] = min(max_yield, yield_units + bonus)    # melon cap 6

    ripe = age >= first_yield_day (10) and (yields >= max_yield (6) or age >= max_yield_day (12))

A melon accumulates from age 6 at +1 per WATER, or +2 while fertilized, and
FERTILIZE sets `fertilized_until_day = day + 2`, i.e. three watering days of
cover. So ONE fertilize at age 6 reaches the cap of 6 by age 8 and the tile is
ripe at age 10, the moment `first_yield_day` allows. Unfertilized it needs six
waters and is not ripe until age 11-12.

That is the whole race, and the price makes it expensive:

    day 10  $271   day 11  $220   day 12  $175   day 13  $97

pub-ahmed-v38 clears 60 units on day 10 at $271; we ripen late and sell across
days 11-13, realising $185/unit against their $247 on the same 72 units.

`scheduler.build_tasks` never fertilizes melon -- `want_fert` lists STRAWBERRY,
TOMATO, WHEAT and CARROT only -- because the rule prices fertilizer against the
EXTRA YIELD it buys ("fertilizer is cash, ~$100/unit"). For melon the yield is
capped either way; what fertilizer actually buys is TWO DAYS OF TIMING, worth
about ($271-$175) x 6 = $576 a tile against a fertilizer unit's ~$50.

A harvest burst was the wrong instrument: it tried to clear faster on a day the
crop was not yet ripe. Tested and rejected (realised price 179.7 vs the
baseline's 185.4, units 72 -> 63).

    PYTHONPATH=. uv run python tools/make_ripen_variant.py --name r-melon
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''                or (t["crop"] == "CARROT" and age == 2 and fert_price < 50)
            )'''

REPL = '''                or (t["crop"] == "CARROT" and age == 2 and fert_price < 50)
                # MELON: not for yield -- the cap of 6 is reachable either way
                # -- but for TIMING. Fertilizing at the window start doubles the
                # accumulation rate, so the tile hits the cap by age 8 and is
                # ripe at age 10 instead of 11-12. Melon price falls
                # $271 -> $220 -> $175 -> $97 across those days because both
                # farms dump at once, so two days of ripening is worth far more
                # than the fertilizer.
                or (t["crop"] == "MELON" and RIPEN_AGE_LO <= age <= RIPEN_AGE_HI
                    and fert_price < RIPEN_MAX_FERT)
            )'''

CONSTS = '''RIPEN_AGE_LO = __LO__
RIPEN_AGE_HI = __HI__
RIPEN_MAX_FERT = __MAXP__

'''


def build(base, name, lo=6, hi=7, max_fert=250.0):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, REPL, 1)
            text = (CONSTS.replace("__LO__", repr(int(lo)))
                          .replace("__HI__", repr(int(hi)))
                          .replace("__MAXP__", repr(float(max_fert)))) + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("want_fert block not found")
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
    ap.add_argument("--lo", type=int, default=6, help="earliest melon age to fertilize")
    ap.add_argument("--hi", type=int, default=7, help="latest melon age to fertilize")
    ap.add_argument("--max-fert", type=float, default=250.0,
                    help="skip when fertilizer is dearer than this")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.lo, a.hi, a.max_fert)}"
          f"  (melon fertilize at age {a.lo}-{a.hi}, fert_price < {a.max_fert})")
