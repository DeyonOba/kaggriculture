"""Trend-relative sell floors: sell into a spike, hold through a trough.

A static floor is an ABSOLUTE judgement -- "never sell milk under $130". It
cannot tell a $120 print that is the best milk has been in a week from a $120
print on the way down. That distinction is what a moving average adds, and it
is the second half of the sell-timing question (the first half, what the
OPPONENT is about to supply, is tools/make_selltime_variant.py).

Rule. Keep a per-item EWMA of the observed market price. Then

    price > ewma * (1 + band)   local spike   -> LOWER the floor, sell into it
    price < ewma * (1 - band)   local trough  -> RAISE the floor, hold

This is additive to the static floor rather than a replacement: the absolute
floor still refuses genuinely bad prices, and the trend term only decides
whether a price that is near the floor is on its way up or down.

State handling. Bundle modules share one globals dict, so a module-level dict
persists across turns -- the same trick MOVE_STATS and _ROUTES use. But arena
workers REUSE a process across games, so the EWMA must be reset when a new game
starts; `view.day` going backwards is the detector.

    PYTHONPATH=. uv run python tools/make_trendsell_variant.py --name sell-ewma
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

REPL = '''    # --- trend-relative floors (tools/make_trendsell_variant.py) ----------
    _ALPHA, _BAND, _SPIKE, _TROUGH = __ALPHA__, __BAND__, __SPIKE__, __TROUGH__
    _st = _TREND.setdefault(view.player, {"day": -1, "ewma": {}})
    if view.day < _st["day"]:
        _st = _TREND[view.player] = {"day": -1, "ewma": {}}   # process reused
    _st["day"] = view.day
    _ew = _st["ewma"]
    for _it, _p in view.market_prices.items():
        _ew[_it] = _p if _it not in _ew else (1 - _ALPHA) * _ew[_it] + _ALPHA * _p

    def _trend(base, item):
        ref = _ew.get(item)
        p = view.market_prices.get(item)
        if not ref or not p:
            return decayed(base)
        if p > ref * (1 + _BAND):
            base *= _SPIKE      # best it has been lately -> take it
        elif p < ref * (1 - _BAND):
            base *= _TROUGH     # on the way down -> wait
        return decayed(max(1, int(round(base))))

    return {
        "WHEAT": 1,
        "EGG": 1,
        "MELON": _trend(melon_floor, "MELON"),
        "FERTILIZER": 1,
        "CARROT": _trend(CARROT_FLOOR, "CARROT"),
        "TOMATO": _trend(TOMATO_FLOOR, "TOMATO"),
        "STRAWBERRY": _trend(180 if straw_shops >= 3 else 80, "STRAWBERRY"),
        "MILK": _trend(130 if milk_shops >= 1 else 100, "MILK"),
        "WOOL": _trend(160 if yarn_shops >= 1 else 120, "WOOL"),
    }'''

STATE = '''

# --- price EWMA state (tools/make_trendsell_variant.py) ----------------------
# Keyed by seat; reset when view.day goes backwards, because arena pool workers
# reuse a process across games.
_TREND = {}
'''


def build(base, name, alpha=0.3, band=0.08, spike=0.7, trough=1.25):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(
                FIND,
                REPL.replace("__ALPHA__", repr(float(alpha)))
                    .replace("__BAND__", repr(float(band)))
                    .replace("__SPIKE__", repr(float(spike)))
                    .replace("__TROUGH__", repr(float(trough))),
                1,
            )
            k = text.index("def _sell_floors(")
            text = text[:k] + STATE + "\n\n" + text[k:]
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
    ap.add_argument("--alpha", type=float, default=0.3, help="EWMA weight on today")
    ap.add_argument("--band", type=float, default=0.08, help="dead zone around the EWMA")
    ap.add_argument("--spike", type=float, default=0.7, help="floor scale on a spike")
    ap.add_argument("--trough", type=float, default=1.25, help="floor scale in a trough")
    a = ap.parse_args()
    p = build(a.base, a.name, a.alpha, a.band, a.spike, a.trough)
    print(f"wrote {p}  (alpha={a.alpha} band={a.band} "
          f"spike x{a.spike} trough x{a.trough})")
