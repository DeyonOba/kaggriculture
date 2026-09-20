"""Carry more per shed trip. The caps are 5/3/2 and the engine has none.

PICKUP is SHED-ADJACENT only, so every one is a trip to the middle of the
board -- and it costs ONE action whether it moves 1 unit or 18, because
`_inv_add` has no capacity limit at all. Our scheduler nonetheless caps the
haul:

    scheduler:335   batch = 5 if WHEAT else (3 if FERTILIZER else 2)
    scheduler:377   n = min(need, shed_left, 5 if WHEAT else 3)

Measured against the corpus over 80 games (seat 0 only), the cap shows up
exactly where you would expect:

                        corpus    ours
    PICKUP trips/game    179.6   347.2     1.9x the trips
    units per PICKUP      2.94    2.27
    spread              p90 6    p90 4
                        max 18    max 5    <- our hard ceiling
    animal-days/game     356.2   373.5     comparable herds
    HARVEST/game         436.3   292.7     we harvest 33% LESS

So we make nearly double the shed trips for the same herd, and perform 144
fewer harvests. Freeing ~167 trips is worth something only if the turns land
on harvests rather than filler -- `h-front` added 1,440 actions for -$653,
so that is the risk and the reason this needs a gate rather than an argument.

What makes this different from h-front: that arm ADDED actions to units that
were already saturated. This REMOVES redundant trips, and the freed unit is
standing at the shed, where the delivery and pickup tasks already are.

--mult 1 is bit-identical to the base by construction.

    PYTHONPATH=. uv run python tools/make_batch_variant.py --name x-batch3 --mult 3
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

A1 = '''        batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)'''
R1 = '''        batch = _BATCH_N("WHEAT" == item and "WHEAT" or item)'''

A2 = '''            n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)'''
R2 = '''            n = min(need, shed_left[item], _BATCH_N(item))'''

HEAD = '''
# --- haul more per shed trip ------------------------------------------------
# PICKUP is shed-adjacent and costs one action regardless of how much it
# moves; the engine's _inv_add has NO capacity limit. The corpus hauls up to
# 18 units in one trip and averages 2.94; we were capped at 5 and averaged
# 2.27, making 1.9x the shed trips for the same herd.
_BATCH_MULT = __MULT__


def _BATCH_N(item):
    base = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
    return max(base, int(round(base * _BATCH_MULT)))

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mult", type=float, required=True,
                    help="multiplier on the haul cap; 1 = base, bit-identical")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, done = [], 0
    for mod, text in mods:
        if mod == "scheduler":
            for a, r in ((A1, R1), (A2, R2)):
                if a not in text:
                    raise SystemExit(f"anchor not found in scheduler:\n{a}")
                text = text.replace(a, r, 1)
                done += 1
            text = HEAD.replace("__MULT__", repr(args.mult)) + text
        out.append((mod, text))
    if done != 2:
        raise SystemExit(f"patched {done} of 2 anchors")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    caps = {i: max(b, int(round(b * args.mult))) for i, b in
            (("WHEAT", 5), ("FERTILIZER", 3), ("other", 2))}
    print(f"{args.name}: haul caps {caps}  (base WHEAT 5 / FERT 3 / other 2, "
          f"corpus max 18)")


if __name__ == "__main__":
    main()
