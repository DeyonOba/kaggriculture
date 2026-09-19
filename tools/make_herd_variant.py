"""Substitute geese for cows/sheep: trade a crashing product for an annuity.

The champion knobs buy ZERO geese (`"geese": 0`). That is not obviously wrong
-- per head, a cow out-earns a goose at base prices -- but it prices the herd
at BASE, and the herd's products do not sell at base.

Measured on Snorlax's six losses to 2900+ seats:

                    Snorlax                    stronger seat
    COW      8 bought, 204 milk        7 bought, 172 milk
    GOOSE    2 bought,  62 eggs        4 bought, 153 eggs

    MILK   194 sold @ $132 = $25,651   169 sold @ $152 = $25,616
    EGG     62 sold @ $46  =  $2,862   153 sold @ $46  =  $6,964

The two seats take the SAME milk revenue, and Snorlax needs 25 more units to
get there: his marginal 25 units of milk earn $35 in total, because they push
the price from $152 to $132 across all 194. Marginal milk is worth about
$1.40 a unit. Meanwhile 91 extra eggs earn $4,102 at an IDENTICAL price
($45.8 vs $45.5), because egg absorption never binds -- 3% season utilisation
against milk's 57% and wool's 68%.

So this is not "geese are good". It is: beyond a herd size the market has
already absorbed, milk and wool convert to nothing while eggs convert 1:1.

WHY THE PRIOR NULL DOES NOT COVER THIS. "buy geese" was tested and lost in
all six ladder bins, recorded under "a demand table predicts price, never
wins". That test ADDED geese on top of the existing herd, so it bought the
annuity without giving up the exhausted trap, and paid for it in cash, feed
(1 WHEAT/animal/day) and tile space. --swap subtracts the geese from the
cow/sheep targets instead, which is a different intervention.

    --geese N            raise the goose target to N
    --swap               take those N head OUT of cows/sheep (substitution)
    --geese 0            bit-identical to the base by construction

    PYTHONPATH=. uv run python tools/make_herd_variant.py --name h-swap4 \
        --geese 4 --swap
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''def _champ_herd_targets(day, knobs=None):
    knobs = knobs or KNOB_DEFAULTS
    cow = 2 if day < 3 else min(knobs["cows"], 2 + (day - 2))
    # Recorded pace: 4 sheep by day 8, 6 by day 10, 8 by day 12-14.
    sheep = 2 if day < 7 else min(knobs["sheep"], 2 + (day - 6))
    geese = knobs["geese"] if 6 <= day <= 18 else 0'''

REPL = '''def _champ_herd_targets(day, knobs=None):
    knobs = knobs or KNOB_DEFAULTS
    cow = 2 if day < 3 else min(knobs["cows"], 2 + (day - 2))
    # Recorded pace: 4 sheep by day 8, 6 by day 10, 8 by day 12-14.
    sheep = 2 if day < 7 else min(knobs["sheep"], 2 + (day - 6))
    geese = knobs["geese"] if 6 <= day <= 18 else 0
    if _HERDG > 0 and 6 <= day <= 18:
        geese = _HERDG
        if _HERDSWAP:
            # Take the head out of the TRAPS, not on top of them. Split as
            # evenly as possible and never drive either below 2, which is the
            # opening pair the champion plan relies on before day 3/7.
            _c = _HERDG // 2
            _s = _HERDG - _c
            cow = max(2, cow - _c)
            sheep = max(2, sheep - _s)'''

HEAD = '''
# --- geese for cows: swap a trap for an annuity ------------------------------
# EGG season utilisation is 3%, so egg volume converts 1:1 into money. MILK is
# 57% and WOOL 68%, and at the observed herd sizes the marginal unit of milk
# is worth about $1.40 -- 25 extra units moved total milk revenue by $35.
_HERDG = __GEESE__
_HERDSWAP = __SWAP__

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--geese", type=int, required=True,
                    help="goose target from day 6; 0 = base, bit-identical")
    ap.add_argument("--swap", action="store_true",
                    help="subtract the geese from the cow/sheep targets "
                         "instead of adding them on top")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("herd-target anchor not found in planner")
            text = (HEAD.replace("__GEESE__", repr(args.geese))
                        .replace("__SWAP__", repr(bool(args.swap)))
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
    mode = "SWAP out of cows/sheep" if args.swap else "ADD on top (the refuted shape)"
    print(f"{args.name}: {args.base} + geese={args.geese}, {mode}")


if __name__ == "__main__":
    main()
