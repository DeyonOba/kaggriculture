"""Let a worker CARE and FEED the same animal on one tile visit.

The engine banks the care bonus at END OF DAY:

    if tile["cared_today"] and tile["fed_today"]:
        tile["pending_care_bonus"] += 1

Order within the day does not matter. But `scheduler.build_tasks` only emits a
CARE task once FEED has already resolved:

    if feeding_day and t.get("fed_today") and not t.get("cared_today") ...

so a worker standing on an animal it is ABOUT to feed cannot also care it. The
two actions get serialised across turns, and the second one needs another
worker to walk back.

This is the measured defect. We FEED 334 times a game and CARE 199;
pub-ahmed-v38 feeds 328 and cares 412 -- it cares animals it did not feed that
turn. CARE from the tile you are standing on costs ZERO moves, and the 0-move
bucket is the whole efficiency gap (their 49.6% of actions vs our 38.2%). Our
shortfalls are exactly the cared products: MILK 136 vs 196 units, WOOL 150 vs
167, and the scheduler's own comment prices a fed+cared animal at 3-4x a bare
one.

The fix emits CARE when the animal is already fed OR when we are queueing a FEED
for it this same turn, so both land in one visit. It deliberately does NOT care
animals we have no intention of feeding -- that banks nothing.

    PYTHONPATH=. uv run python tools/make_carechain_variant.py --name c-chain
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''        if feeding_day and not t.get("fed_today"):'''
REPL = '''        _will_feed = False
        if feeding_day and not t.get("fed_today"):
            _will_feed = True'''

FIND2 = '''        if feeding_day and t.get("fed_today") and not t.get("cared_today") and day < 28:'''
REPL2 = '''        if (feeding_day and (t.get("fed_today") or _will_feed)
                and not t.get("cared_today") and day < 28):'''


def build(base, name, gate=None):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text and FIND2 in text:
            text = text.replace(FIND, REPL, 1).replace(FIND2, REPL2, 1)
            if gate is not None:
                g = ('or view.market_prices.get(product, 999) >= 0.35 '
                     '* MARKET_PARAMS[product]["base"]')
                if g not in text:
                    raise RuntimeError("CARE price gate not found")
                text = text.replace(g, g.replace("0.35", repr(float(gate))), 1)
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("animal task block not found")
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
    ap.add_argument("--gate", type=float, default=None,
                    help="also override the CARE crash-pivot price gate")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.gate)}  (care-before-feed"
          f"{f', gate={a.gate}' if a.gate is not None else ''})")
