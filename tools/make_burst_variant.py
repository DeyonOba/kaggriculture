"""Harvest BURST: win the ripening-day race instead of optimising the steady state.

Measured. Melons ripen ~day 10 on BOTH farms, both players dump, and the price
collapses 64% in three days. Whoever clears first takes the top:

    day   melon price   us held / on vine   them held / on vine
     10          $271        0 /  54.3            0 /  60.0
     11          $220       20 /  37.0           12 /   0.0
     12          $175       34 /  10.3            0 /   0.0
     13           $97        8 /   0.0            0 /   0.0

pub-ahmed-v38 clears its whole 60-unit crop on day 10 at $271. We are still
selling on day 13 at $97, realising $185/unit against their $247 on the same 72
units -- the 61st vs 71st percentile of the season's price, ~$4.5k a game, and
up to $174/unit on the late units.

Diagnosis: HARVEST-limited, not sell-limited. On day 10 both sides hold ZERO and
we have 54 units still on the vine, so no sell-floor or market rule touches it.
The crew cannot clear the crop in one day while running normal zone rounds.

This is why eleven routing attempts missed it: they all optimised the STEADY
STATE. This is a single critical day, and it wants the opposite of steady-state
behaviour -- drop everything, clear the crop, deliver, dump.

Three things fire together while a burst is active:

  * the crop's HARVEST tasks jump to the top priority band, so every worker goes
    for them instead of its zone round;
  * the shed-shuttle threshold drops to 1. There is a trap here: assign() uses
    EARLY_DROP_THRESHOLD=2 on days 0-9 and DROP_THRESHOLD=10 from day 10, so on
    melon day itself the threshold QUADRUPLES and workers start hoarding
    exactly when speed matters most;
  * the crop's sell floor goes to 1, because the price is FALLING -- holding out
    for a floor is precisely backwards in a race.

    PYTHONPATH=. uv run python tools/make_burst_variant.py --name b-melon
    PYTHONPATH=. uv run python tools/make_burst_variant.py --name b-early --early 1
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

HELPER = '''

# --- harvest burst (tools/make_burst_variant.py) -----------------------------
BURST_CROP = __CROP__
BURST_MIN = __MIN__          # standing units of the crop that trigger a burst
BURST_EARLY = __EARLY__      # days before max_yield_day we accept a tile as ripe
_BURST = {}


def _burst_active(view):
    """True while our standing crop is large enough to be worth racing for.

    Cached per (seat, day): it walks every plant tile and cannot change within a
    day in a way that matters.
    """
    key = (view.player, view.day)
    hit = _BURST.get(key)
    if hit is not None:
        return hit
    units = 0.0
    for _x, _y, _t in view.me.plants():
        if _t.get("crop") == BURST_CROP:
            units += float(_t.get("yield_units", 0) or 0)
    on = units >= BURST_MIN
    if len(_BURST) > 512:
        _BURST.clear()
    _BURST[key] = on
    return on
'''

# 1. harvest priority + optional early ripeness
FIND_H = '''            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= crop["max_yield_day"]
            )
            if ripe or (yields > 0 and decaying and age >= crop["first_yield_day"]):
                tasks.append(Task(P_HARVEST if not decaying else P_SURVIVAL, x, y, ["HARVEST"]))
                continue'''

REPL_H = '''            _burst = (t["crop"] == BURST_CROP and _burst_active(view))
            _slack = BURST_EARLY if _burst else 0
            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= crop["max_yield_day"] - _slack
            )
            if ripe or (yields > 0 and decaying and age >= crop["first_yield_day"]):
                # In a burst the whole crew races the opponent to the top of a
                # falling ladder, so the crop outranks the zone round.
                _pri = P_SURVIVAL if (decaying or (_burst and ripe)) else P_HARVEST
                tasks.append(Task(_pri, x, y, ["HARVEST"]))
                continue'''

# 2. shed-shuttle threshold
FIND_D = '''        threshold = 1 if (aggressive and view.hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if view.day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)'''

REPL_D = '''        threshold = 1 if (aggressive and view.hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if view.day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)
        if _burst_active(view) and inv.get(BURST_CROP, 0) > 0:
            # Day 10 is exactly where DROP_THRESHOLD jumps 2 -> 10, so without
            # this the crew starts hoarding on the one day speed decides price.
            threshold = 1'''

# 3. sell floor
FIND_S = '''        "MELON": decayed(melon_floor),'''
REPL_S = '''        "MELON": (1 if (BURST_CROP == "MELON" and _burst_active(view))
                  else decayed(melon_floor)),'''


def build(base, name, crop="MELON", minimum=20.0, early=0, floor=True):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], set()
    for mod, text in mods:
        if FIND_H in text:
            text = text.replace(FIND_H, REPL_H, 1)
            k = text.index("def build_tasks(")
            text = text[:k] + (HELPER.replace("__CROP__", repr(crop))
                                     .replace("__MIN__", repr(float(minimum)))
                                     .replace("__EARLY__", repr(int(early)))) + "\n\n" + text[k:]
            done.add("harvest")
        if FIND_D in text:
            text = text.replace(FIND_D, REPL_D, 1)
            done.add("drop")
        if floor and FIND_S in text:
            text = text.replace(FIND_S, REPL_S, 1)
            done.add("floor")
        out.append((mod, text))
    need = {"harvest", "drop"} | ({"floor"} if floor else set())
    if done != need:
        raise RuntimeError(f"patch sites missing: {sorted(need - done)}")

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
    ap.add_argument("--crop", default="MELON")
    ap.add_argument("--min", dest="minimum", type=float, default=20.0)
    ap.add_argument("--early", type=int, default=0,
                    help="accept a tile as ripe this many days before max_yield_day")
    ap.add_argument("--no-floor", action="store_true")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.crop, a.minimum, a.early, not a.no_floor)}"
          f"  (crop={a.crop} min={a.minimum} early={a.early} floor={not a.no_floor})")
