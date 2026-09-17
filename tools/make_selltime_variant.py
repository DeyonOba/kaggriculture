"""Opponent-aware sell timing: generalise the melon dump rule to every product.

The champion already contains this idea, applied to exactly one item:

    # Opponent melons about to ripen -> dump ours before the crash.
    opp_ripening = sum(1 for _, _, t in view.opp.plants()
                       if t["crop"] == "MELON" and view.day - t["planted_day"] >= 8)
    if opp_ripening >= 3:
        melon_floor = 40

That is a hand-rolled, single-item, integer-threshold version of a general
principle. Selling is a congestion game: both farms sell into ONE market
inventory, `market_price` falls as inventory rises, and the town's shops drain
it daily. So the right time to sell item X is when the opponent is NOT about to
supply it, and the right time to dump X is just before they flood it.

We already have the two quantities this needs, both validated:

    supply_projection(view.opp, day, horizon)   what they will bring to market
    econ.forecast_drain(view.unlocked_shops, d) what the town will absorb

`anchors()` already uses exactly this pair to size production. This applies it
to sell timing instead, for every floored product rather than melons alone:

    pressure = opp_incoming_over_H / town_drain_over_H
    pressure >= --flood   -> they are about to flood: scale the floor DOWN,
                             sell into the price we still have
    pressure <= --scarce  -> they are absent: scale the floor UP, hold and let
                             shop drain lift the price

Why this is worth trying when eleven routing attempts failed: routing and
allocation are both measured flat, but sell timing is the one dimension neither
gate touched, and it is the one a tape structurally cannot express -- a tape is
a fixed script and cannot see what the opponent is holding. It is also the only
live use we have found for the opponent belief model.

    PYTHONPATH=. uv run python tools/make_selltime_variant.py --name sell-opp
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = '''    return {
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

REPL = '''    # --- opponent-aware sell timing (tools/make_selltime_variant.py) -------
    # Same pair of quantities anchors() uses to size production, applied to
    # timing instead: what the opponent will bring to market over the next
    # _SELL_H days, against what the town will absorb in that window.
    _SELL_H, _FLOOD, _SCARCE = __HORIZON__, __FLOOD__, __SCARCE__
    _DUMP, _HOLD = __DUMP__, __HOLD__

    _days_left = max(1, 29 - view.day)
    _fc = econ.forecast_drain(view.unlocked_shops, view.day)
    _opp_in = supply_projection(view.opp, view.day, horizon=_SELL_H)

    def _tilt(base, item):
        # Town absorption over the horizon, prorated from the remaining season.
        drain = _fc.get(item, 0.0) * min(_SELL_H, _days_left) / _days_left
        if drain <= 0:
            return base
        ratio = _opp_in.get(item, 0.0) / drain
        if ratio >= _FLOOD:
            base *= _DUMP          # they are about to flood -> sell before it
        elif ratio <= _SCARCE:
            base *= _HOLD          # they are absent -> hold for the recovery
        return decayed(max(1, int(round(base))))

    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": _tilt(melon_floor, "MELON"),
        "FERTILIZER": 1,
        "CARROT": _tilt(CARROT_FLOOR, "CARROT"),
        "TOMATO": _tilt(TOMATO_FLOOR, "TOMATO"),
        "STRAWBERRY": _tilt(180 if straw_shops >= 3 else 80, "STRAWBERRY"),
        "MILK": _tilt(130 if milk_shops >= 1 else 100, "MILK"),
        "WOOL": _tilt(160 if yarn_shops >= 1 else 120, "WOOL"),
    }'''


def build(base, name, horizon=4, flood=1.2, scarce=0.4, dump=0.5, hold=1.3):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(
                FIND,
                REPL.replace("__HORIZON__", repr(int(horizon)))
                    .replace("__FLOOD__", repr(float(flood)))
                    .replace("__SCARCE__", repr(float(scarce)))
                    .replace("__DUMP__", repr(float(dump)))
                    .replace("__HOLD__", repr(float(hold))),
                1,
            )
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("sell_floors return dict not found")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--flood", type=float, default=1.2)
    ap.add_argument("--scarce", type=float, default=0.4)
    ap.add_argument("--dump", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=1.3)
    a = ap.parse_args()
    p = build(a.base, a.name, a.horizon, a.flood, a.scarce, a.dump, a.hold)
    print(f"wrote {p}  (H={a.horizon} flood>={a.flood} scarce<={a.scarce} "
          f"dump x{a.dump} hold x{a.hold})")
