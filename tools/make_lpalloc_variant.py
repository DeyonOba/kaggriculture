"""Move the WHOLE tile allocation toward the LP optimum, not one crop at a time.

Every single-product target has been gated and lost: `wheat_engine` 24/28/40
REFUTED, `carrot_standing` 8/16 REFUTED, `geese` 2/3 REFUTED, `straw_target`
21/27/36/40 all negative, `tomato_standing` 6 null at n=550, `cows`/`sheep`
6/10 REFUTED. That pattern is what a tile-constrained optimum looks like:
adding one crop displaces something worth more per tile-day, and removing one
frees capacity with nothing queued to use it.

Measured, champ-h8 vs the LP plan at day 13 (3 quadrants = 75 tiles):

    ours  wheat 23  carrot  0  tomato  0  straw 33  cow 8  sheep 8  goose 0 = 72
    LP    wheat 11  carrot  7  tomato 11  straw 29  cow 6  sheep 5  goose 4 = 73

Both saturated. It is purely a reallocation, and the LP funds its additions
(tomato, carrot, geese) by cutting animals and strawberry -- which no
single-knob sweep could ever express.

One dose parameter alpha moves every target together:

    target = round(base_expression + alpha * (lp_target[day] - base_expression))

Blending at the target SITE rather than substituting an array means alpha=0 is
bit-identical to the base by construction -- verify that before trusting any
other dose.

    PYTHONPATH=. uv run python tools/make_lpalloc_variant.py --name a-lp33 --alpha 0.33
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

LP_PATH = os.path.join(REPO, "rl/datasets/lp_schedule.json")

A_MELON = '''                if crop == "MELON":
                    n = knobs["melon_opening"]
                quota[crop] = max(0, n - crops.get(crop, 0))'''

R_MELON = '''                if crop == "MELON":
                    n = knobs["melon_opening"]
                if _LPA > 0:
                    _lpv = _LP_OPEN.get(crop, _ZERO30)[day] if day < 30 else n
                    n = max(0, int(round(n + _LPA * (_lpv - n))))
                quota[crop] = max(0, n - crops.get(crop, 0))'''

A_CROPS = '''            straw_target = knobs["straw_target"] if day >= 10 else min(knobs["straw_early"], knobs["straw_target"])
            quota["STRAWBERRY"] = max(0, straw_target - crops.get("STRAWBERRY", 0))
        if knobs["carrot_standing"] and 6 <= day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(0, knobs["carrot_standing"] - crops.get("CARROT", 0))
        if knobs["tomato_standing"] and 6 <= day <= TOMATO_LAST_PLANT_DAY:
            quota["TOMATO"] = max(0, knobs["tomato_standing"] - crops.get("TOMATO", 0))
        if day >= knobs["carrot_day"] and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(quota.get("CARROT", 0), min(knobs["carrot_cap"], free))
        elif day > 1 and day <= WHEAT_LAST_PLANT_DAY:
            if day < 10:
                wheat_target = 8
            else:
                wheat_target = min(knobs["wheat_engine"], crops.get("WHEAT", 0) + free)
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))'''

R_CROPS = '''            straw_target = knobs["straw_target"] if day >= 10 else min(knobs["straw_early"], knobs["straw_target"])
            straw_target = _lpmix(straw_target, _LP_STRAWBERRY, day)
            quota["STRAWBERRY"] = max(0, straw_target - crops.get("STRAWBERRY", 0))
        _ct = _lpmix(knobs["carrot_standing"] if 6 <= day <= CARROT_LAST_PLANT_DAY
                     else 0, _LP_CARROT, day)
        if _ct > 0 and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(0, _ct - crops.get("CARROT", 0))
        _tt = _lpmix(knobs["tomato_standing"] if 6 <= day <= TOMATO_LAST_PLANT_DAY
                     else 0, _LP_TOMATO, day)
        if _tt > 0 and day <= TOMATO_LAST_PLANT_DAY:
            quota["TOMATO"] = max(0, _tt - crops.get("TOMATO", 0))
        if day >= knobs["carrot_day"] and day <= CARROT_LAST_PLANT_DAY:
            quota["CARROT"] = max(quota.get("CARROT", 0), min(knobs["carrot_cap"], free))
        elif day > 1 and day <= WHEAT_LAST_PLANT_DAY:
            if day < 10:
                wheat_target = 8
            else:
                wheat_target = min(knobs["wheat_engine"], crops.get("WHEAT", 0) + free)
            wheat_target = _lpmix(wheat_target, _LP_WHEAT, day)
            if _LPA > 0:
                # The LP's early wheat is far above ours; never ask for more
                # tiles than exist, or the quota starves the other crops.
                wheat_target = min(wheat_target, crops.get("WHEAT", 0) + free)
            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))'''

A_HERD = '''    return {
        "GOOSE": geese,
        "COW": cow if day <= 18 else 0,
        "SHEEP": sheep if day <= 20 else 0,
    }'''

R_HERD = '''    out = {
        "GOOSE": geese,
        "COW": cow if day <= 18 else 0,
        "SHEEP": sheep if day <= 20 else 0,
    }
    if _LPA > 0:
        for _k, _arr in (("COW", _LP_COW), ("SHEEP", _LP_SHEEP),
                         ("GOOSE", _LP_GOOSE)):
            out[_k] = _lpmix(out[_k], _arr, day)
    return out'''


def header(alpha, lp):
    return f'''
# --- LP allocation blend ----------------------------------------------------
# Per-day standing targets from the season LP (rl/datasets/lp_schedule.json,
# margin ${lp["margin"]:,.0f}). alpha blends every target at its own site, so
# alpha=0 reproduces the base expression exactly.
_LPA = {alpha}
_ZERO30 = {[0] * 30!r}
_LP_WHEAT = {lp["WHEAT"]!r}
_LP_CARROT = {lp["CARROT"]!r}
_LP_TOMATO = {lp["TOMATO"]!r}
_LP_STRAWBERRY = {lp["STRAWBERRY"]!r}
_LP_MELON = {lp["MELON"]!r}
_LP_COW = {lp["COW"]!r}
_LP_SHEEP = {lp["SHEEP"]!r}
_LP_GOOSE = {lp["GOOSE"]!r}
_LP_HANDS = {lp["HANDS"]!r}
_LP_LAND = {lp["LAND"]!r}
_LP_OPEN = {{"MELON": _LP_MELON, "WHEAT": _LP_WHEAT}}


def _lpmix(base, arr, day):
    if _LPA <= 0 or day >= 30:
        return base
    return max(0, int(round(base + _LPA * (arr[day] - base))))

'''


A_HIRES = "    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs[\"hires_delta\"] if day >= 6 else 0)"

R_HIRES = '''    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) + (knobs["hires_delta"] if day >= 6 else 0)
    schedule = _lpmix(schedule, _LP_HANDS, day)'''


A_LAND = "    if next_idx < CHAMP_LAND_BUYS and CHAMP_LAND_DAY <= day <= 22:"

R_LAND = """    _lcap = _lpmix(CHAMP_LAND_BUYS if day >= CHAMP_LAND_DAY else 0, _LP_LAND, day)
    if next_idx < _lcap and day <= 22:"""


def load_lp():
    d = json.load(open(LP_PATH))
    out = {"margin": d["margin"], "HANDS": [int(x) for x in d["hands"]]}
    # land_day maps purchase index -> day bought; turn it into a per-day cap
    # on cumulative extra quadrants, which is what the planner gates on.
    buys = sorted(int(v) for v in d["land_day"].values())
    out["LAND"] = [sum(1 for b in buys if b <= day) for day in range(30)]
    for key, src in (("WHEAT", "standing"), ("CARROT", "standing"),
                     ("TOMATO", "standing"), ("STRAWBERRY", "standing"),
                     ("MELON", "standing"), ("COW", "herd"),
                     ("SHEEP", "herd"), ("GOOSE", "herd")):
        out[key] = [int(d[src].get(str(day), {}).get(key, 0)) for day in range(30)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="0 = base (must be bit-identical), 1 = full LP plan")
    ap.add_argument("--with-land", action="store_true",
                    help="also blend the land-purchase schedule; the LP buys "
                         "quadrant #1 on DAY 0, and without it the opening "
                         "allocation does not physically fit")
    ap.add_argument("--with-hires", action="store_true",
                    help="also blend CHAMP_HIRES toward the LP hand schedule; "
                         "the LP crop mix assumes the LP labour, so crops "
                         "alone is an incoherent hybrid")
    args = ap.parse_args()

    lp = load_lp()
    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            edits = [(A_MELON, R_MELON, "opening"),
                     (A_CROPS, R_CROPS, "crop quota"),
                     (A_HERD, R_HERD, "herd")]
            if args.with_hires:
                edits.append((A_HIRES, R_HIRES, "hires"))
            if args.with_land:
                edits.append((A_LAND, R_LAND, "land"))
            for anchor, repl, what in edits:
                if anchor not in text:
                    raise SystemExit(f"{what} anchor not found in planner")
                text = text.replace(anchor, repl, 1)
            text = header(args.alpha, lp) + text
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
    print(f"{args.name}: {args.base} blended {args.alpha:.0%} toward the LP "
          f"allocation (wheat/carrot/tomato/straw/melon/cow/sheep/goose"
          f"{'/hands' if args.with_hires else ''}"
          f"{'/land' if args.with_land else ''})")


if __name__ == "__main__":
    main()
