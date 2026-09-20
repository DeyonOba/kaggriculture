"""The `land` head: buy the second quadrant sooner, and maybe the third.

The one planner head never tested. Measured over the full 21,054-game corpus
traversal, relative quadrant timing is monotone in the outcome:

    we reach 3 quadrants vs them        n      win rate   mean margin
    3+ days EARLIER                   103         0.777      +$10,883
    1-2 days earlier                2,315         0.631       +$4,118
    same day                       14,398         0.497         +$218
    1-2 days later                  2,808         0.448       -$1,741
    3+ days LATER                   1,364         0.343       -$5,364

r(earlier, margin) = -0.200, the 9th strongest relative signal in the corpus
and the strongest one attached to a head we have never moved.

WHY THIS IS NOT JUST "RICH SEATS BUY LAND". Land costs money, so the obvious
read is reverse causation. But the relative MONEY lead at day 10 correlates
with the final margin at r = -0.021 -- nothing at all -- while quadrant
timing at the same day is +0.193. A pure wealth proxy would have inherited
money's null. It does not, so quadrant timing carries information the bank
balance does not. That is a defence, not a proof; only the gate settles it.

THE TWO KNOBS, and why the defaults are where they are:

    CHAMP_LAND_DAY  = 5   the day the standing wish opens
    CHAMP_LAND_BUYS = 2   "champions never bought the $4000 quadrant"

LAND_ORDER is NE $1000, SW $2000, SE $4000, and we open with $3000 -- exactly
NE+SW -- so cash, not the day gate, is what usually binds. Lowering the day
alone may therefore do nothing, which is why --buys is exposed too: only
5.9% of corpus seats ever reach a 4th quadrant, so the third purchase is
close to unobserved behaviour rather than a copied one.

--day 5 --buys 2 is bit-identical to the base by construction.

    PYTHONPATH=. uv run python tools/make_land_variant.py --name l-d3 --day 3
    PYTHONPATH=. uv run python tools/make_land_variant.py --name l-b3 --buys 3
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    if next_idx < CHAMP_LAND_BUYS and CHAMP_LAND_DAY <= day <= 22:'''
REPL = '''    if next_idx < _LAND_BUYS and _LAND_DAY <= day <= 22:'''

HEAD = '''
# --- the `land` head -------------------------------------------------------
# Relative quadrant timing is monotone in the outcome over 21,054 games:
# reaching the 3rd quadrant 3+ days before the opponent wins 77.7% of the
# time, 3+ days later wins 34.3%. The money lead at day 10 predicts nothing
# (r = -0.021) while quadrant timing predicts +0.193, so this is not simply
# a proxy for being rich early.
_LAND_DAY = __DAY__
_LAND_BUYS = __BUYS__

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--day", type=int, default=5,
                    help="day the land wish opens; 5 = base")
    ap.add_argument("--buys", type=int, default=2,
                    help="quadrants to buy; 2 = base (champions never took SE)")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("land anchor not found in planner")
            text = (HEAD.replace("__DAY__", repr(args.day))
                        .replace("__BUYS__", repr(args.buys))
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
    print(f"{args.name}: {args.base} + land day {args.day} (base 5), "
          f"buys {args.buys} (base 2)")


if __name__ == "__main__":
    main()
