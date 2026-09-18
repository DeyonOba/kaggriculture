"""Extend the melon opponent-supply rule to strawberry, milk and wool.

`tools/opp_coupling.py` measured the opponent's supply at $52,652/game, 97% of
it in STRAWBERRY ($19,756), MILK ($19,136) and WOOL ($9,062). `_sell_floors`
conditions on the opponent's board for MELON and nothing else -- the other
three read `unlocked_shops` counts only.

The signal is an observation, not a forecast. `observation.farms` is shared
(kaggriculture.py:271), so every unit already sitting on their plants and
animals is public in `tile["yield_units"]`. Compare that stock against the
market's remaining headroom, I0 - inventory: when their board alone can push
the pool past I0, the price we can still get is about to fall and holding
loses.

Why this might beat what the knob bandit already found: `premium_scale` swept
0.1-1.3 over 4,270 samples and settled at 0.35, and the one price-conditioned
arm (premium_scale|P_STRAWBERRY>=1.0@d12) was REFUTED at -$1,691. Both read
PRICE, which is the state after they have already sold. Their board leads it:
strawberry is first_yield_day 10 / interval 2, so their day-8 board prices
day-10 supply.

    PYTHONPATH=. uv run python tools/make_oppfloor_variant.py --name f-opp10 --k 1.0
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": decayed(melon_floor),
        "FERTILIZER": 1,
        "CARROT": decayed(CARROT_FLOOR),
        "TOMATO": decayed(TOMATO_FLOOR),
        "STRAWBERRY": decayed(180 if straw_shops >= 3 else 80),
        "MILK": decayed(130 if milk_shops >= 1 else 100),
        "WOOL": decayed(160 if yarn_shops >= 1 else 120),
    }'''

TEMPLATE = '''    # Opponent supply pressure on the three products that carry 97% of the
    # measured market congestion ($19,756 / $19,136 / $9,062 per game).
    # observation.farms is shared, so tile["yield_units"] on their board is
    # an observation of stock they can dump, not an estimate. Fire when that
    # stock alone covers the market's remaining headroom.
    _OPP_SRC = {{"STRAWBERRY": (0, "STRAWBERRY"), "MILK": (1, "COW"),
                "WOOL": (1, "SHEEP")}}

    def pressured(item, floor):
        is_animal, key = _OPP_SRC[item]
        stock = 0
        if is_animal:
            for _x, _y, _t in view.opp.animals():
                if _t.get("animal") == key:
                    stock += _t.get("yield_units", 0)
        else:
            for _x, _y, _t in view.opp.plants():
                if _t.get("crop") == key:
                    stock += _t.get("yield_units", 0)
        headroom = MARKET_I0 - view.market_inv.get(item, MARKET_I0)
        if stock > headroom * {k}:
            return max(1, int(floor * {cut}))
        return floor

    return {{
        "WHEAT": 1,
        "EGG": 1,
        "MELON": decayed(melon_floor),
        "FERTILIZER": 1,
        "CARROT": decayed(CARROT_FLOOR),
        "TOMATO": decayed(TOMATO_FLOOR),
        "STRAWBERRY": decayed(pressured("STRAWBERRY",
                                        180 if straw_shops >= 3 else 80)),
        "MILK": decayed(pressured("MILK", 130 if milk_shops >= 1 else 100)),
        "WOOL": decayed(pressured("WOOL", 160 if yarn_shops >= 1 else 120)),
    }}'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=float, required=True,
                    help="fraction of the market headroom their visible stock "
                         "must cover before we cut the floor; LOWER fires more")
    ap.add_argument("--cut", type=float, default=0.5,
                    help="multiplier on the floor when pressured (melon uses "
                         "80 -> 40, i.e. 0.5)")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    by_mod = dict(mods)
    # Every module execs into ONE shared dict (see the bundle tail), so the
    # planner resolves MARKET_I0 as a global from `constants` without an
    # import line of its own. Check it is actually there.
    if "MARKET_I0 =" not in by_mod.get("constants", ""):
        raise SystemExit("constants does not define MARKET_I0")
    if "self.market_inv" not in by_mod.get("gamestate", ""):
        raise SystemExit("View does not expose market_inv")

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("sell-floor anchor not found in planner")
            text = text.replace(
                ANCHOR, TEMPLATE.format(k=args.k, cut=args.cut), 1)
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
    print(f"{args.name}: {args.base} + opponent-stock floors on "
          f"STRAWBERRY/MILK/WOOL, k={args.k} cut={args.cut}")


if __name__ == "__main__":
    main()
