"""Screen the CARE crash-pivot gate -- the measured CARE shortfall.

`scheduler.build_tasks` only emits a CARE task when the product's price clears
35% of base:

    price_ok = view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product]["base"]
    if price_ok:
        tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))

The comment justifies it as releasing actions we "cannot profitably sell". The
autopsy says that trade is backwards. We FEED 334 times a game and CARE only
199; pub-ahmed-v38 feeds 328 and cares 412. CARE costs ZERO moves -- the worker
is already standing on the animal, and it lands in the 0-move bucket that is
the whole efficiency gap (their 49.6% vs our 38.2%). Meanwhile our shortfalls
are exactly the cared products: MILK 136 vs 196 units, WOOL 150 vs 167.

And the price discipline the gate buys is not worth it: they sell 196 milk at
$80.2 where we sell 136 at $83.4, and finish $4,392 ahead on that line alone.
The scheduler's own comment values a fed+cared animal at 3-4x a bare one.

The price trace says the gate bites often: MILK is under 30% of base on 32% of
days, WOOL on 23%.

    --gate 0.0   never skip CARE
    --gate 0.2   skip only in a deep collapse
    --gate 0.35  shipped behaviour

    PYTHONPATH=. uv run python tools/make_care_variant.py --name c-gate0 --gate 0.0
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = "or view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product][\"base\"]"


def build(base, name, gate):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(
                FIND,
                FIND.replace("0.35", repr(float(gate))), 1)
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("CARE price gate not found")
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
    ap.add_argument("--gate", type=float, required=True)
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.gate)}  (CARE price gate = {a.gate})")
