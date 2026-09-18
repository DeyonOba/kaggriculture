"""Block a planting whose harvest lands after that product's market gluts.

Measured motivation, not a hunch. Instrumenting `_rent_pick` over six games:
it picks MELON on days 11, 12 and 13 and nowhere else -- 7 of 117 freed-tile
decisions. Melon's `max_yield_day` is 12, so those plantings complete on days
23-25, and the clock (`tools/market_clock.py`) puts melon's glut at day 11.
Every one of them matures a dozen days into a crashed market.

The constraint is a BOOLEAN GATE on the candidate list, not a price. That
matters: S4's forecast head failed through `econ.sell_revenue`, which is
convex, so a smoothed prediction was biased exactly at the threshold. A gate
never passes through that function.

Per product, from `tools/glut_target.py`:
    MELON               ridge, R2 0.986, decision error 0.13% at m=12, and
                        ALL of it in the safe direction (0.00% "too late")
    CARROT, TOMATO      glut in 8% / 3% of 15-day windows -> hard-coded never
    FERTILIZER          gluts day 1 always, but is never planted
    WHEAT, STRAWBERRY   no head yet, so no constraint

    PYTHONPATH=. uv run python tools/make_glut_variant.py --name r-glut --glut /tmp/glut.py
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''        else:
            if day + cd["max_yield_day"] > 29:
                continue
            k = min(_RENT_HORIZON, cd["max_yield_day"])
            units = cd["max_yield"]
            occupies = cd["max_yield_day"] + 1'''

REPL = '''        else:
            if day + cd["max_yield_day"] > 29:
                continue
            if not _glut_ok(crop, cd["max_yield_day"]):
                continue
            k = min(_RENT_HORIZON, cd["max_yield_day"])
            units = cd["max_yield"]
            occupies = cd["max_yield_day"] + 1'''

CALL_ANCHOR = "    knobs = _knobs(memory)\n"
CALL_REPL = ('    knobs = _knobs(memory)\n'
             '    _GLUT_OBS["o"] = getattr(memory, "last_raw_obs", None)\n')

GLUE = '''

# --- glut gate --------------------------------------------------------------
_GLUT_OBS = {"o": None}
_GLUT_CACHE = {"day": -2, "melon": None}


def _glut_ok(crop, matures_in):
    """False when this planting would ripen after its market has glutted."""
    if crop in _GLUT_NEVER:
        return True
    if crop != "MELON":
        return True
    o = _GLUT_OBS.get("o")
    if o is None:
        return True
    try:
        d = o.get("day", -1)
        if _GLUT_CACHE["day"] != d:
            _GLUT_CACHE["day"] = d
            _GLUT_CACHE["melon"] = _glut_melon_days(o)
        return _GLUT_CACHE["melon"] >= matures_in
    except Exception:
        return True          # any failure falls back to r-mrg100 behaviour

'''


def encoder_source():
    enc = open(os.path.join(REPO, "tools", "board_tensor.py")).read()
    enc = enc.replace("from kagent.constants import MARKET_PARAMS",
                      "pass  # MARKET_PARAMS is a bundle global")
    enc = enc.replace("FIRST_YIELD", "_BT_FIRST_YIELD")
    i = enc.find("def counts(obs, seat=0):")
    j = enc.find("\nN_COUNT = ", i)
    if i > 0 and j > i:
        enc = enc[:i] + enc[enc.find("\n", j + 1) + 1:]
    return "import numpy as _np\n" + enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--glut", required=True)
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    mod = encoder_source() + "\n\n" + open(args.glut).read()

    out, done = [], False
    for name, text in mods:
        if name == "planner":
            if ANCHOR not in text or CALL_ANCHOR not in text:
                raise SystemExit("rent anchors not found; is the base r-mrg100?")
            text = text.replace(ANCHOR, REPL, 1)
            text = text.replace(CALL_ANCHOR, CALL_REPL, 1)
            text = GLUE + text
            done = True
        out.append((name, text))
    if not done:
        raise SystemExit("no planner module in bundle")
    out.insert(1, ("glutmod", mod))

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + glut gate ({os.path.getsize(p)/1000:.0f} kB)")


if __name__ == "__main__":
    main()
