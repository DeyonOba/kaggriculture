"""Opponent-aware CARE: produce into the product the opponent is NOT flooding.

Measured motivation. `c-chain` (let a worker CARE and FEED an animal on one tile
visit) gated +$8,072 +/- 1,843 against pub-ahmed-v38 and -$5,891 +/- 2,407
against pub-tschinkel-psr -- both individually significant, so it is a real
opponent INTERACTION, not noise. The sales autopsy says why:

    milk sold by the opponent     ahmed 196 @ $80.2     tschinkel 232 @ $69.8

Tschinkel already saturates milk, so our extra cared output crashes a market
that is full; ahmed leaves room, so the same change pays. Caring is only worth
the action when there is demand left for what it produces.

The shipped gate asks the wrong question:

    price_ok = view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product]["base"]

A crashed price is the SYMPTOM and it is observed late -- by the time milk is
under 35% of base, the season's actions are already spent. The opponent's
standing herd is the CAUSE and it is public from day one. `anchors()` already
pairs the two quantities to size production:

    net_per_day = (forecast_drain(item) - supply_projection(opp)[item]) / days_left

This applies the same pair to the CARE decision. Gate on headroom rather than on
a lagging price, keeping the crash check as a floor.

Restricted to MILK and WOOL, the cared products. The plan's Phase 2 flags
wheat/fertilizer supply as measured wrong and unusable; those are not gated here.

    PYTHONPATH=. uv run python tools/make_careopp_variant.py --name c-opp --base c-chain
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

HELPER = '''

# --- opponent-aware CARE headroom (tools/make_careopp_variant.py) ------------
CARE_ROOM_MIN = __ROOM__     # units/day of unmet town demand needed to bother
_CARE_ROOM = {}


def _care_room(view, product):
    """Town drain for `product` minus the opponent's projected supply, per day.

    Cached per (seat, day): supply_projection walks the opponent's whole farm,
    and the answer cannot change within a day.
    """
    if not product:
        return 1e9
    key = (view.player, view.day, product)
    hit = _CARE_ROOM.get(key)
    if hit is not None:
        return hit
    days_left = max(1, 29 - view.day)
    horizon = 29 - view.day
    fc = econ.forecast_drain(view.unlocked_shops, view.day)
    opp = supply_projection(view.opp, view.day, horizon=horizon)
    room = (fc.get(product, 0.0) - opp.get(product, 0.0)) / days_left
    if len(_CARE_ROOM) > 4096:
        _CARE_ROOM.clear()
    _CARE_ROOM[key] = room
    return room
'''

FIND = '''            if price_ok:
                tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))'''

REPL = '''            # Headroom gate: caring raises output, which is only worth an
            # action if the town still has demand the opponent has not already
            # promised to fill. Only the cared products are gated.
            room_ok = (product not in ("MILK", "WOOL")
                       or _care_room(view, product) >= CARE_ROOM_MIN)
            if price_ok and room_ok:
                tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))'''


def build(base, name, room):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, REPL, 1)
            k = text.index("def build_tasks(")
            text = text[:k] + HELPER.replace("__ROOM__", repr(float(room))) + "\n\n" + text[k:]
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("CARE emit site not found")
    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="c-chain")
    ap.add_argument("--name", required=True)
    ap.add_argument("--room", type=float, default=0.0)
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.room)}  (base={a.base}, room>={a.room}/day)")
