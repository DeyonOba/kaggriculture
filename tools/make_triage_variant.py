"""Watering triage: when more crops are dying than we can save, save the
valuable ones first.

THE MEASUREMENT. champ-h8 kills 24.2% of everything it plants by missing two
consecutive waterings, and 75% of those are strawberries -- 18.7 of 25 per
game. Strawberry is `ongoing` with first_yield_day 10, so for its whole 10-day
growth phase `produces_tonight` is False and it is watered at P_FILLER, the
LOWEST priority. It only reaches P_SURVIVAL once it is one day from death. The
current policy is therefore: let everything drift to the brink, then rescue.
Plant 33 in one week and they hit the brink together, swamping the rescue.

WHY THIS IS NOT ANOTHER THROUGHPUT FIX. Planting fewer is refuted
(straw_target 21/27/36/40 all negative) and scheduling the plantings is
refuted twice (-$5,809 and -$6,227, the two runs disagreeing on direction).
Adding labour is refuted (hires_delta -2 is -$11,746, +2 is -$4,408). This adds
NO actions and moves NO actions between tiles -- it only changes WHICH dying
crop gets the water when we cannot save them all. The LP says a marginal action
is worth $13.00 and ours realises $1.75; this is an attempt to raise the value
of the action we already spend.

THE CHANGE. Among P_SURVIVAL waterings, `pick` currently breaks ties by
DISTANCE, so the crop we rescue is the nearest one, not the most valuable. A
strawberry one day from its first yield and one nine days away are rescued with
equal urgency, and if we can only reach one we take whichever is closer. This
orders the rescue by REMAINING TIME TO YIELD instead: the nearly-ready crop
wins, the one that still needs nine more days of water is allowed to die.

Priorities are floats and Task sorts on them, so a small fractional offset
inside the P_SURVIVAL band reorders rescues without disturbing the bands.

    PYTHONPATH=. uv run python tools/make_triage_variant.py --name t-triage
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''        if t.get("consecutive_unwatered", 0) >= 1:
            pri = P_SURVIVAL  # dies tonight if unwatered'''

# Second design. Reordering the RESCUE was measured and failed: strawberry
# deaths went 75% -> 73% (unchanged) while total deaths ROSE 21.3 -> 24.7,
# because by the time a crop is one day from death the binding problem is that
# more are dying than we have actions, not which one we reach first.
#
# COMMITMENT instead. An ongoing crop before its first yield sits at P_FILLER
# for its whole growth phase -- ten days for strawberry -- so it is only ever
# touched when it is about to die. This promotes the OLDEST K of each such crop
# to P_TEND so they are watered steadily and live, and leaves the rest at
# P_FILLER to be abandoned deliberately. Same labour, concentrated instead of
# spread: 20 strawberries that yield beat 33 that mostly do not.
ANCHOR2 = '''            since_first = day + 1 - t["planted_day"] - crop["first_yield_day"]
            produces_tonight = since_first >= 0 and since_first % crop["interval"] == 0
            fertilized = t.get("fertilized_until_day", -1) >= day
            pri = P_TEND if (produces_tonight and fertilized) else P_FILLER'''

PATCH2 = '''            since_first = day + 1 - t["planted_day"] - crop["first_yield_day"]
            produces_tonight = since_first >= 0 and since_first % crop["interval"] == 0
            fertilized = t.get("fertilized_until_day", -1) >= day
            if produces_tonight and fertilized:
                pri = P_TEND
            elif since_first < 0 and _commit_rank(me, t["crop"], x, y, day) < TRIAGE_K:
                pri = P_TEND        # committed: water it through its growth
            else:
                pri = P_FILLER'''

HELPER = '''

def _commit_rank(me, crop_name, x, y, day):
    """Rank of this tile among same-crop tiles still before first yield,
    oldest first. The oldest are closest to paying, so they are the ones worth
    committing water to when we cannot keep them all alive."""
    ages = []
    for cx, cy, ct in me.plants():
        if ct.get("crop") != crop_name:
            continue
        a = day - ct["planted_day"]
        if a < CROPS[crop_name]["first_yield_day"]:
            ages.append((-a, cx, cy))
    ages.sort()
    for i, (_, cx, cy) in enumerate(ages):
        if cx == x and cy == y:
            return i
    return 999
'''

PATCH = '''        if t.get("consecutive_unwatered", 0) >= 1:
            # TRIAGE. A dying crop is only worth saving if it will live to pay.
            # Order the rescue by how soon this tile yields: a strawberry one
            # day from its first production outranks one nine days away, and
            # when we cannot reach both the far one is allowed to die instead
            # of being chosen merely because it was nearer. Offset stays inside
            # the P_SURVIVAL band so the priority ordering is untouched.
            if crop["ongoing"]:
                _left = crop["first_yield_day"] - age
                if _left < 0:
                    _rem = crop["interval"] - ((age - crop["first_yield_day"])
                                               % max(1, crop["interval"]))
                    _left = _rem if _rem != crop["interval"] else 0
            else:
                _left = max(0, crop["max_yield_day"] - age)
            pri = P_SURVIVAL + min(max(_left, 0), 20) * TRIAGE_W'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="commit", choices=["rescue", "commit"])
    ap.add_argument("--keep", type=int, default=20,
                    help="how many pre-yield tiles of each ongoing crop to "
                         "commit water to; the rest are abandoned")
    ap.add_argument("--weight", type=float, default=0.04,
                    help="fraction of a priority band per day-to-yield. Must "
                         "stay under 1/20 so 20 days cannot cross a band.")
    args = ap.parse_args()

    src_path = os.path.join(REPO, "arena", "agents", args.base, "main.py")
    src = open(src_path).read()
    head, mods, tail = _split_bundle(src)

    out, patched = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if ANCHOR not in text:
                raise SystemExit("anchor not found in the frozen scheduler")
            if args.mode == "rescue":
                text = text.replace(ANCHOR, PATCH, 1)
            else:
                if ANCHOR2 not in text:
                    raise SystemExit("commitment anchor not found")
                text = text.replace(ANCHOR2, PATCH2, 1)
                text = text + HELPER
            # inject the weight next to the priority constants
            marker = "P_SURVIVAL = 0"
            if marker not in text:
                raise SystemExit("P_SURVIVAL constant not found")
            text = text.replace(marker,
                                f"{marker}\nTRIAGE_W = {args.weight!r}\n"
                                f"TRIAGE_K = {args.keep}", 1)
            patched = True
        out.append((mod, text))
    if not patched:
        raise SystemExit("no scheduler module in the bundle")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + watering triage (w={args.weight}) "
          f"({len(src)} -> {len(code)} bytes)")
    print("  SMOKE TEST IT: a bundle that raises on import scores exactly $3,000")


if __name__ == "__main__":
    main()
