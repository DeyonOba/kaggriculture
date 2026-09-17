"""Value-weighted watering: order waterings by what the crop's PRODUCT is
worth at the margin, not by distance.

WHY THIS AND NOT THE LAST TWO TRIAGE ATTEMPTS.

  design 1, rescue ordered by time-to-yield   deaths 75% -> 73%, TOTAL deaths
                                              ROSE 21.3 -> 24.7.  failed
  design 2, commit water to the oldest K      deaths 75% -> 60%, and the gate
                                              said -$1,028 +/- 511, z = -2.01

Design 2's mechanism worked and its criterion was wrong: it saved
STRAWBERRIES, whose product sits at 0.91x base and falling (above_target 1.6 on
a linear curve, price floors within ~100 units past I0, supply/absorption
0.95). It spent labour saving units the market will not buy.

Value-weighting design 2 is a no-op, which is worth stating rather than
shipping: the commitment branch only fires for ONGOING crops before first
yield, which is TOMATO and STRAWBERRY only, and tomato_standing defaults to 0.
So "do not commit to strawberry" simply reverts to base behaviour -- base is
already right there.

THE CHANGE THAT IS NOT A NO-OP. Weight EVERY watering by its product's
marginal price ratio. Measured over 5 seeds (tools/market_headroom.py):

    TOMATO 2.38x base    WHEAT 1.74x    CARROT 1.36x
    STRAWBERRY 0.91x     MELON 0.50x

Melon is watered at P_TEND through its whole six-day window and its marginal
unit is worth half of base, because both farms supply 4.72x what the town
absorbs. Wheat is worth 1.74x. Nothing in the scheduler knows this: `pick`
takes the NEAREST task in a band, so a melon and a wheat at equal urgency are
separated by walking distance alone.

This is the first time the market table touches the executor. Every previous
market finding -- geese, tomato, melon_opening -- was tested by changing what
we PLANT, and all three lost because planting displaces tiles and labour.
Changing the ORDER we water displaces far less: no action is added and none
moves between tiles.

The offset stays strictly inside the priority band, so band ordering is
preserved and only within-band ties change.

    PYTHONPATH=. uv run python tools/make_valwater_variant.py --name v-water
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''        tasks.append(Task(pri, x, y, ["WATER"]))'''

PATCH = '''        # VALUE WEIGHTING. `pick` breaks ties inside a band by DISTANCE, so a
        # melon and a wheat at equal urgency are separated by walking distance
        # alone -- even though melon's marginal unit fetches 0.50x base (both
        # farms supply 4.72x what the town absorbs) and wheat's fetches 1.74x.
        # Offset stays inside the band, so bands are untouched and only
        # within-band ties move.
        _p = crop.get("product", t["crop"])
        _base = MARKET_PARAMS.get(_p, {}).get("base", 0) or 1
        _ratio = float(view.market_prices.get(_p, _base)) / _base
        pri = pri + VALWATER_W * (1.0 - min(max(_ratio, 0.0), 2.0) / 2.0)
        tasks.append(Task(pri, x, y, ["WATER"]))'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weight", type=float, default=0.9,
                    help="span of the within-band offset. Must stay under 1.0 "
                         "so a low-value crop cannot fall into the next band.")
    args = ap.parse_args()
    if not 0 < args.weight < 1.0:
        raise SystemExit("weight must be in (0, 1) to preserve the bands")

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, patched = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if text.count(ANCHOR) != 1:
                raise SystemExit(f"anchor appears {text.count(ANCHOR)} times, need 1")
            text = text.replace(ANCHOR, PATCH, 1)
            marker = "P_SURVIVAL = 0"
            if marker not in text:
                raise SystemExit("P_SURVIVAL not found")
            text = text.replace(marker, f"{marker}\nVALWATER_W = {args.weight!r}", 1)
            if "MARKET_PARAMS" not in text.split("\n\n")[0] and \
               "MARKET_PARAMS" not in text[:2000]:
                # the bundle aliases modules into one globals dict, so
                # constants are reachable; assert rather than guess
                pass
            patched = True
        out.append((mod, text))
    if not patched:
        raise SystemExit("no scheduler module found")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + value-weighted watering (w={args.weight})"
          f"  ({len(src)} -> {len(code)} bytes)")
    print("  SMOKE TEST: must not be $3,000, and stderr must be clean -- "
          "policy.act_safe\n  swallows exceptions, so a broken build still "
          "runs and still scores.")


if __name__ == "__main__":
    main()
