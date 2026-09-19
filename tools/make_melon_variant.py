"""Harvest a scarce-capacity crop at its FIRST yield day, not its last.

Measured on Snorlax (rank 24) against the six 2900+ seats that beat him, and
on our own agent in the same seven worlds:

    median melon sell day    $/unit   units   share sold on day 10
    stronger seats    10       229      75            43%
    Snorlax           17       133      96             9%
    r-mrg100 (us)     12        88      88            11%

Melon ripens on day 10 for everyone, and it is in NO shop: absorption
capacity falls 111 -> 5 between day 10 and day 17 while regeneration is ~1
unit/day. So the price is set almost entirely by WHEN you sell, and the
natural experiment is in the data -- in the single game where Snorlax sold at
median day 11 he got $222/unit against the strong seat's $224, statistically
identical. Strawberry is the control: it regenerates at ~26/day, its capacity
GROWS through the season, and both seats sell at median day 22 for $131 vs
$144, i.e. timing buys nothing there.

WHY WE ARE LATE, exactly. MELON has first_yield_day=10, max_yield_day=12,
max_yield=6, ongoing=False, and the scheduler harvests on

    age >= first_yield_day and (yields >= max_yield or age >= max_yield_day)

which for melon resolves to age 12 -- our measured median sell day. We wait
two days to collect the 6th unit and sell it into a market that has already
fallen from $271. The trade is ~1 unit against ~80% of the price.

The rule: bring the harvest trigger forward inside the
[first_yield_day, max_yield_day] window by `alpha` of its width. alpha=0
leaves the trigger at max_yield_day and is bit-identical to the base by
construction; alpha=1 harvests the moment the crop first yields.

Restricted by default to MELON, because the argument is about SCARCE
absorption capacity and does not transfer: wheat and carrot never glut (8%
and 33% season utilisation) so selling them early buys nothing and costs
units. Pass --crops to widen it and let the gate decide.

    PYTHONPATH=. uv run python tools/make_melon_variant.py --name x-melon100 --alpha 1.0
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= crop["max_yield_day"]
            )'''

REPL = '''            _cut = crop["max_yield_day"]
            if _MELA > 0 and t["crop"] in _MELCROPS and yields > 0:
                # Pull the trigger forward inside [first_yield, max_yield_day].
                # Rounds DOWN, so any alpha < one day's width is a no-op rather
                # than a silent full-strength change.
                _w = crop["max_yield_day"] - crop["first_yield_day"]
                _cut = crop["max_yield_day"] - int(_MELA * _w)
            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= _cut
            )'''

HEAD = '''
# --- harvest into the ripen-day price window ---------------------------------
# MELON is in no shop: capacity 111 -> 5 between day 10 and day 17, regen ~1/day.
# Price is therefore set by WHEN you sell, and the default trigger
# (age >= max_yield_day) puts us at day 12 for a crop that peaks on day 10.
_MELA = __ALPHA__
_MELCROPS = __CROPS__

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="fraction of the [first_yield, max_yield_day] window "
                         "to bring the harvest forward; 0 = base, bit-identical")
    ap.add_argument("--crops", nargs="+", default=["MELON"],
                    help="crops the rule applies to (default MELON only)")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if ANCHOR not in text:
                raise SystemExit("harvest anchor not found in scheduler")
            text = (HEAD.replace("__ALPHA__", repr(args.alpha))
                       .replace("__CROPS__", repr(tuple(args.crops)))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no scheduler module in bundle")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
    shifts = []
    for c in args.crops:
        cd = CROPS.get(c)
        if cd:
            w = cd["max_yield_day"] - cd["first_yield_day"]
            shifts.append(f"{c} day {cd['max_yield_day']}->"
                          f"{cd['max_yield_day'] - int(args.alpha * w)}")
    print(f"{args.name}: {args.base} + early harvest at alpha={args.alpha:.2f}"
          + ("   (" + ", ".join(shifts) + ")" if shifts else ""))


if __name__ == "__main__":
    main()
