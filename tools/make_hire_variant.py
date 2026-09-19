"""Make CHAMP_HIRES a function of projected market state, not 30 fixed numbers.

The rent rule paid (+$2,978, win z 2.17 over 1,131 worlds) because it made CROP
CHOICE depend on the price a crop will fetch at ITS harvest day. `CHAMP_HIRES`
is still `[5,4,4,5,4,5,7,9,9,10,11,...]` -- thirty constants with no state
dependence whatsoever.

What the design has to respect, all measured:

  * HIRING IS NEARLY FREE. `_hire_cost` is fib-scaled from mult=1, so eleven
    hands cost $232 for a day and a whole season of eleven costs ~$6,960
    against a ~$95,000 bank. Cost is not the constraint and a rule that trades
    cost against value is solving the wrong problem.
  * THE MARKET-ORDER CAP IS. `maxMarketOrdersPerTurn = 10` and HIRE is one
    order PER HAND, so every hire spends a slot a SELL could have used. That is
    why h-front11 never fielded eleven hands.
  * MORE HANDS DOES NOT PAY. h-front7/9/11 went +$1,281 / -$653 / -$1,370,
    negative with dose, and work is underfoot 94.9% of the time.
  * BUT SLOWER HIRING IS CATASTROPHIC. m-hire5/m-hire3 went -$7,234 / -$39,678.

So the lever is SHAPE, not level: hire into days whose output will meet a
strong market, and leave the slots for selling when it will not. The scale
factor is clipped and mean-preserving-ish so the season total barely moves --
this is a reallocation, not an increase.

alpha blends against the constant and alpha=0 is bit-identical by construction.

    PYTHONPATH=. uv run python tools/make_hire_variant.py --name h-fwd50 --alpha 0.5
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = ('    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) '
          '+ (knobs["hires_delta"] if day >= 6 else 0)')

REPL = ('    schedule = (CHAMP_HIRES[day] if day < len(CHAMP_HIRES) else 8) '
        '+ (knobs["hires_delta"] if day >= 6 else 0)\n'
        '    schedule = _hire_shape(schedule, view, day)')

CALL_ANCHOR = "    knobs = _knobs(memory)\n"
CALL_REPL = ('    knobs = _knobs(memory)\n'
             '    _HIRE_VIEW["v"] = view\n')

GLUE = '''

# --- hiring shaped by the forward market model ------------------------------
# A hand hired today does work whose output reaches the market a few days
# later, so price it at the projected price THEN, exactly as the rent rule
# prices a crop at its own harvest day.
_HIREA = __ALPHA__
_HIRE_LOOK = 4          # days from work to market for the median action
_HIRE_LO, _HIRE_HI = 0.75, 1.25
_HIRE_VIEW = {"v": None}
_HIRE_CACHE = {"day": -2, "f": 1.0}


def _hire_strength(view, day):
    """Projected value of a day's output relative to today's prices.

    >1 means what we grow will sell into a stronger market than today's, so
    the hands are worth their market slots; <1 means the slots are better
    spent selling.
    """
    inv = _rent_inventory(view, _RENT_HORIZON)
    now = val = 0.0
    for p, path in inv.items():
        if p in ("FERTILIZER",):
            continue
        k = min(_HIRE_LOOK, len(path) - 1)
        # weight by what we actually hold or are growing in that product
        w = view.shed.get(p, 0) + 1.0
        now += w * econ.price(p, int(round(path[0])))
        val += w * econ.price(p, int(round(path[k])))
    if now <= 0:
        return 1.0
    return val / now


def _hire_shape(schedule, view, day):
    if _HIREA <= 0 or view is None:
        return schedule
    try:
        if _HIRE_CACHE["day"] != day:
            _HIRE_CACHE["day"] = day
            f = _hire_strength(view, day)
            _HIRE_CACHE["f"] = max(_HIRE_LO, min(_HIRE_HI, f))
    except Exception:
        return schedule
    f = _HIRE_CACHE["f"]
    return max(1, int(round(schedule * (1.0 + _HIREA * (f - 1.0)))))

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if "_rent_inventory" not in text:
                raise SystemExit(f"{args.base} carries no forward model")
            for a in (ANCHOR, CALL_ANCHOR):
                if a not in text:
                    raise SystemExit("hire anchors not found in planner")
            text = text.replace(ANCHOR, REPL, 1).replace(CALL_ANCHOR, CALL_REPL, 1)
            text = GLUE.replace("__ALPHA__", repr(args.alpha)) + text
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
    print(f"{args.name}: {args.base} + forward-model hiring shape, alpha={args.alpha}")


if __name__ == "__main__":
    main()
