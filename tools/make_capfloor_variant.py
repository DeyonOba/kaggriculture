"""Sell floors that ration ABSORPTION CAPACITY instead of scaling a constant.

Every prior floor experiment moved a constant: `premium_scale` 0.1-1.3 over
4,270 bandit samples, opponent-stock floors, the glut gate. All marginal,
one-factor-at-a-time. `tools/market_capacity.py` supplies the quantity they
were all missing -- how much the market can still absorb, how fast that
regenerates, and how much of it the opponent is about to take:

  product      cap d0  cap d20  regen/day  exhausts  season utilisation
  FERTILIZER      203        0        0.0     d14.9      209%  oversold 2x
  MELON           101        1        1.0     d16.1      116%  oversold
  WOOL             38       19       12.7     d17.7       68%  contested
  MILK             31        6       20.2     d16.6       57%  contested
  STRAWBERRY       26       71       26.1     d24.8       50%
  CARROT          158      279       17.2     never        33%  capacity GROWS
  TOMATO           93      196       12.6     never        22%  capacity GROWS
  WHEAT          5000     5000       30.7     never         8%
  EGG            5000     5000       13.2     never         3%

The rule is a ratio, not a table. At each turn, for each product:

    season_cap = units sellable above the floor NOW + regen x days remaining
    claim      = our stock (shed + on the board) + THEIR visible board stock
    pressure   = claim / season_cap

pressure > 1   the market cannot absorb what is already committed; the floor
               collapses to 1 and we sell before the other seat does
pressure < lo  capacity is ample and for carrot/tomato still GROWING, so the
               floor rises and we wait

This is joint: it moves with capacity, regeneration, our holdings and their
visible holdings together, which is what "marginal probes closed L2 too early"
means in practice.

    PYTHONPATH=. uv run python tools/make_capfloor_variant.py --name c-cap5 --alpha 0.5
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''        "STRAWBERRY": decayed(pressured("STRAWBERRY",
                                        180 if straw_shops >= 3 else 80)),
        "MILK": decayed(pressured("MILK", 130 if milk_shops >= 1 else 100)),
        "WOOL": decayed(pressured("WOOL", 160 if yarn_shops >= 1 else 120)),
    }'''

ANCHOR_PLAIN = '''        "STRAWBERRY": decayed(180 if straw_shops >= 3 else 80),
        "MILK": decayed(130 if milk_shops >= 1 else 100),
        "WOOL": decayed(160 if yarn_shops >= 1 else 120),
    }'''

REPL = '''        "STRAWBERRY": decayed(_cap_floor(view, "STRAWBERRY",
                                         180 if straw_shops >= 3 else 80)),
        "MILK": decayed(_cap_floor(view, "MILK",
                                   130 if milk_shops >= 1 else 100)),
        "WOOL": decayed(_cap_floor(view, "WOOL",
                                   160 if yarn_shops >= 1 else 120)),
    }'''

GLUE = '''

# --- capacity-rationed sell floors ------------------------------------------
_CAPA = __ALPHA__
_CAP_HI = 1.0        # claim exceeds season capacity -> dump
_CAP_LO = 0.5        # ample slack -> hold out for a better price
_CAP_SRC = {"STRAWBERRY": ("PLANT", "STRAWBERRY"), "MILK": ("ANIMAL", "COW"),
            "WOOL": ("ANIMAL", "SHEEP")}


def _cap_stock(view, item):
    """Units of `item` both seats already hold -- shed, plus on the board."""
    kind, key = _CAP_SRC[item]
    n = view.shed.get(item, 0)
    for farm in (view.me, view.opp):
        if kind == "PLANT":
            for _x, _y, _t in farm.plants():
                if _t.get("crop") == key:
                    n += _t.get("yield_units", 0)
        else:
            for _x, _y, _t in farm.animals():
                if _t.get("animal") == key:
                    n += _t.get("yield_units", 0)
    return n


def _cap_floor(view, item, floor):
    """Scale the floor by how contested the remaining absorption is."""
    if _CAPA <= 0:
        return floor
    try:
        base = MARKET_PARAMS[item]["base"]
        inv = view.market_inv.get(item)
        if inv is None:
            return floor
        cap = econ.units_sellable_above(item, inv, max(1, int(0.6 * base)),
                                        cap=5000)
        regen = econ.town_drain_per_day(view.unlocked_shops).get(item, 0.0)
        season = cap + regen * max(0, 29 - view.day)
        if season <= 0:
            return 1
        pressure = _cap_stock(view, item) / season
        if pressure >= _CAP_HI:
            adj = 0.0                      # oversold: sell before they do
        elif pressure <= _CAP_LO:
            adj = 1.35                     # slack and regenerating: wait
        else:
            span = _CAP_HI - _CAP_LO
            adj = 1.35 * (_CAP_HI - pressure) / max(1e-9, span)
        return max(1, int(round(floor * (1.0 + _CAPA * (adj - 1.0)))))
    except Exception:
        return floor

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
            anchor = ANCHOR if ANCHOR in text else ANCHOR_PLAIN
            if anchor not in text:
                raise SystemExit("sell-floor anchor not found in planner")
            text = (GLUE.replace("__ALPHA__", repr(args.alpha))
                    + text.replace(anchor, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no planner module in bundle")

    p = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(p, exist_ok=True)
    f = os.path.join(p, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(f, "w").write(code)
    compile(code, f, "exec")
    print(f"{args.name}: {args.base} + capacity-rationed floors, "
          f"alpha={args.alpha}")


if __name__ == "__main__":
    main()
