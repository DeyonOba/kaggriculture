"""Price the HERD with the same forward model that just won on tiles.

`tools/make_rent_variant.py` gated +$3,185 +/- 768 replicated by pricing a
freed tile at its harvest-day price against the inventory its own units
create. The herd is the same decision and is still a constant: cows 8,
sheep 8, geese 0, whatever the shop draw did.

That constant is expensive in the tail. `tools/market_clock.py`:
P(no shop sells WOOL) = 34.4%, and in those games top players hold 131 wool
units against 534 when a YARN_STORE opens. An animal also differs from a crop
in a way the rent formula must respect -- it costs $300-500 up front, so its
rent is NET of purchase.

    rent(animal) = (sell_revenue(product, inv_at_first_yield, block) - cost*n)
                   / (n tiles * days held)

Compared against the best crop rent from the same forecast. If an animal
cannot beat the tile it occupies, hold fewer of them.

Earlier herd attempts failed because they conditioned on a coarse binary and
almost never fired -- `cows|MILK<=0 -> 6` returned exactly $0 on 270 worlds.
This is continuous and fires every day.

Built on r-mrg100 by default so the gate measures ADDITIVITY with the win
already banked, not a fresh comparison against champ-h8.

    PYTHONPATH=. uv run python tools/make_herdrent_variant.py --name h-rent50 --alpha 0.5
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    return {
        "GOOSE": geese,
        "COW": cow if day <= 18 else 0,
        "SHEEP": sheep if day <= 20 else 0,
    }'''

REPL = '''    out = {
        "GOOSE": geese,
        "COW": cow if day <= 18 else 0,
        "SHEEP": sheep if day <= 20 else 0,
    }
    if _HERDA > 0 and _HERD_VIEW.get("v") is not None:
        out = _herd_rent_scale(out, _HERD_VIEW["v"], day)
    return out'''

# _champ_herd_targets takes only (day, knobs), so the view has to reach it
# some other way. make_champion_plan is the only caller and has `view`.
CALL_ANCHOR = "    knobs = _knobs(memory)\n"
CALL_REPL = ('    knobs = _knobs(memory)\n'
             '    _HERD_VIEW["v"] = view\n')

INJECT = '''

# --- herd priced by the forward market model --------------------------------
_HERDA = __ALPHA__
_HERD_VIEW = {"v": None}


def _herd_rent(view, day, kind, n, inv_path):
    """Net $/tile-day for holding `n` of `kind` from today to season end."""
    a = ANIMALS[kind]
    span = max(1, 29 - day)
    if span <= a["first_yield_day"] or n <= 0:
        return 0.0
    rate = 1  # CARE doubles it, but only if we pay the care action
    cycles = (span - a["first_yield_day"]) // a["interval"] + 1
    units = int(n * cycles * rate)
    if units <= 0:
        return 0.0
    k = min(len(inv_path) - 1, a["first_yield_day"])
    lvl = int(round(inv_path[a["product"]][k]))
    rev = econ.sell_revenue(a["product"], lvl, units)
    # An animal also eats one WHEAT a day and drops one FERTILIZER a night;
    # price each at ITS OWN projected inventory, not the product's.
    kf = min(len(inv_path) - 1, 1)
    rev += n * span * (econ.price("FERTILIZER", int(round(inv_path["FERTILIZER"][kf])))
                       - econ.price("WHEAT", int(round(inv_path["WHEAT"][kf]))))
    return (rev - a["cost"] * n) / float(n * span)


def _herd_rent_scale(out, view, day):
    """Shrink or grow each herd target by its rent against the best crop."""
    try:
        inv_path = _rent_inventory(view, _RENT_HORIZON)
        best = _rent_pick(view, day, 1)
    except Exception:
        return out
    if not best:
        return out
    crop_rent = best[1]
    if crop_rent <= 0:
        return out
    scaled = {}
    for kind, target in out.items():
        if target <= 0:
            scaled[kind] = target
            continue
        r = _herd_rent(view, day, kind, target, inv_path)
        ratio = max(0.25, min(1.5, r / crop_rent))
        blended = target + _HERDA * (target * ratio - target)
        scaled[kind] = max(0, int(round(blended)))
    return scaled

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="0 = base, bit-identical; 1 = full rent scaling")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if "_rent_inventory" not in text:
                raise SystemExit(f"{args.base} carries no forward model; "
                                 "build it with make_rent_variant.py first")
            for a in (ANCHOR, CALL_ANCHOR):
                if a not in text:
                    raise SystemExit("herd anchor not found in planner")
            text = text.replace(ANCHOR, REPL, 1)
            text = text.replace(CALL_ANCHOR, CALL_REPL, 1)
            text = INJECT.replace("__ALPHA__", repr(args.alpha)) + text
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
    print(f"{args.name}: {args.base} + rent-priced herd, alpha={args.alpha}")


if __name__ == "__main__":
    main()
