"""Size the herd to realised demand -- but only the half the archive supports.

`tools/market_clock.py` over 24 top-rated sessions shows strong players move
VOLUME with the shop draw, not just price:

    WOOL   drain  1/day (12 of 24 games)  131 units @ $83
           drain 25/day ( 4 of 24 games)  534 units @ $197
    MILK   drain  7/day  186 units @ $51   ...  drain 37/day  614 units @ $185

Our herd is a constant: cows 8, sheep 8, geese 0, regardless of the draw.
`DEMAND_BOOK` exists to fix that and is disabled (BOOK_HERD = False). The knob
archive says why -- the book is right about cows and wrong about sheep:

    cows  | MILK consumers == 0   -> 6    +$1,989 +/- 551    n=  365   KEEP
    sheep | P_WOOL >= base @ d12  -> 10   +  $947 +/- 125    n=1,035   KEEP
    sheep | WOOL consumers == 0   -> 5    -$1,703 +/- 724    n=  137   REFUTED
    geese                         -> 2/3  -$2,160 / -$3,185            REFUTED
    cows/sheep unconditional      -> 6/10  all negative                REFUTED

Enabling the book wholesale applies the refuted arms too, which nets about
zero. This applies only the two that survived, separately so the split is
attributable.

    PYTHONPATH=. uv run python tools/make_herdbook_variant.py --name h-cows --rules cows
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR_DEF = '''def _knobs(memory):
    k = dict(KNOB_DEFAULTS)
    k.update(getattr(memory, "knobs", None) or {})
    return k'''

ANCHOR_CALL = "    knobs = _knobs(memory)\n"

INJECT = '''

def _demand_herd(knobs, view):
    """Herd sized to realised demand instead of held constant.

    Only the archive arms that survived: cows cut when no shop consumes milk
    (+$1,989 +/- 551 over 365 paired samples), sheep raised when wool is still
    at or above base on day 12 (+$947 +/- 125 over 1,035). The demand book's
    sheep|WOOL<=0 and geese arms are refuted and are deliberately not applied.
    """
    k = dict(knobs)
    if {cows} and view.day >= 6:
        if sum(1 for s in view.unlocked_shops if "MILK" in SHOPS.get(s, ())) == 0:
            k["cows"] = 6
    if {sheep} and view.day >= 12:
        wool_base = MARKET_PARAMS["WOOL"]["base"]
        if view.market_prices.get("WOOL", wool_base) >= wool_base:
            k["sheep"] = 10
    return k'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--rules", required=True, choices=["cows", "sheep", "both"])
    args = ap.parse_args()

    cows = args.rules in ("cows", "both")
    sheep = args.rules in ("sheep", "both")

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    by_mod = dict(mods)
    for need, mod in (("SHOPS =", "constants"), ("MARKET_PARAMS =", "constants"),
                      ("self.market_prices", "gamestate"),
                      ("self.unlocked_shops", "gamestate")):
        if need not in by_mod.get(mod, ""):
            raise SystemExit(f"{mod} does not define {need.strip(' =')}")

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR_DEF not in text or ANCHOR_CALL not in text:
                raise SystemExit("knob anchors not found in planner")
            text = text.replace(
                ANCHOR_DEF,
                ANCHOR_DEF + INJECT.format(cows=cows, sheep=sheep), 1)
            # make_champion_plan is the only _knobs caller with `view` in scope.
            text = text.replace(
                ANCHOR_CALL,
                ANCHOR_CALL + "    knobs = _demand_herd(knobs, view)\n", 1)
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
    print(f"{args.name}: {args.base} + demand-sized herd "
          f"(cows={cows}, sheep={sheep})")


if __name__ == "__main__":
    main()
