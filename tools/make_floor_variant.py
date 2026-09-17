"""Scale the champion's sell floors -- the market-timing knobs.

`planner.sell_floors` returns a minimum price per item; produce is held until
the market clears it, with `decayed()` relaxing the floor to 1 over the last
days so nothing is stranded. Three of the policy's thirteen heads do exactly
this and nothing else:

    melon_floor           absolute MELON floor
    premium_floor_scale   multiplies STRAWBERRY / MILK / WOOL
    staple_floor_scale    multiplies CARROT / TOMATO

Why these matter more than the other ten. Every remaining head is an ALLOCATION
quantity (crop tiles, herd targets, hires, land), and allocation is measured
flat here: the LP's provably-optimal allocation, executed by our own scheduler,
tied champ-h8 at +$828 +/- 2,317 despite being a genuinely different plan. That
gate replaced planting, herd targets, hiring and the land gate -- it did NOT
touch the sell floors. So the floors are the only untested dimension of the
policy's action space, and they are the dimension a tape cannot express: a tape
is a fixed script, while a floor reacts to the live price.

    PYTHONPATH=. uv run python tools/make_floor_variant.py --name f-prem-1.4 --premium 1.4
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": decayed(melon_floor),
        "FERTILIZER": 1,
        "CARROT": decayed(CARROT_FLOOR),
        "TOMATO": decayed(TOMATO_FLOOR),
        "STRAWBERRY": decayed(180 if straw_shops >= 3 else 80),
        "MILK": decayed(130 if milk_shops >= 1 else 100),
        "WOOL": decayed(160 if yarn_shops >= 1 else 120),
    }'''

REPL = '''    # --- sell-floor scales (tools/make_floor_variant.py) ---
    _PM, _ST, _ML = __PREMIUM__, __STAPLE__, __MELON__

    def _s(v, k):
        # Floors are prices, so scale then clamp at 1; decayed() still relaxes
        # everything to 1 near the end of the season.
        return decayed(max(1, int(round(v * k))))

    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": _s(melon_floor, _ML),
        "FERTILIZER": 1,
        "CARROT": _s(CARROT_FLOOR, _ST),
        "TOMATO": _s(TOMATO_FLOOR, _ST),
        "STRAWBERRY": _s(180 if straw_shops >= 3 else 80, _PM),
        "MILK": _s(130 if milk_shops >= 1 else 100, _PM),
        "WOOL": _s(160 if yarn_shops >= 1 else 120, _PM),
    }'''


def build(base, name, premium=1.0, staple=1.0, melon=1.0):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(
                FIND,
                REPL.replace("__PREMIUM__", repr(float(premium)))
                    .replace("__STAPLE__", repr(float(staple)))
                    .replace("__MELON__", repr(float(melon))),
                1,
            )
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("sell_floors return dict not found")

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
    ap.add_argument("--premium", type=float, default=1.0)
    ap.add_argument("--staple", type=float, default=1.0)
    ap.add_argument("--melon", type=float, default=1.0)
    a = ap.parse_args()
    p = build(a.base, a.name, a.premium, a.staple, a.melon)
    print(f"wrote {p}  (premium x{a.premium}, staple x{a.staple}, melon x{a.melon})")
