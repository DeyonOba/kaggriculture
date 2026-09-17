"""The shed-trip threshold expires exactly when the most valuable crop ripens.

Measured. The balance sheet puts us within $2,223 of pub-ahmed-v38 through day 9;
day 10 then costs $12,078 in one day, ~30% of the final gap. Decomposed, the
whole difference is CASH, not harvest:

                cash    inventory   deployed      book
    us  d9->10   +228      +4,945     +1,782    +6,954
    them        +14,228    +4,528       +277   +19,033

We gain MORE inventory than they do and hold MORE melon at end of day 10
($4,151 vs $2,594). We harvest fine. They convert $14,228 to cash; we convert
$228.

Where the melons sit at end of day (units in workers' hands):

    day        9      10      11      12
    us       0.0    19.0    19.0    15.0
    them     0.0    12.0     0.0     0.0

They clear the crop WITHIN day 10. We carry ~19 melons to midnight, where the
engine's end-of-day `_drop_inventories_to_shed` returns them to the shed, and we
sell a day late into a falling price ($271 -> $220 -> $175 -> $97).

The cause is one constant. `assign` step 1:

    threshold = ... else DROP_THRESHOLD          # 10
    if view.day < 10:
        threshold = min(threshold, EARLY_DROP_THRESHOLD)   # 2

Melons have first_yield_day 10. So on the exact day the crop ripens the trip
threshold quadruples, and with ~19 melons across ~11 workers each holds ~2 --
far under 10 -- so nobody goes to the shed.

Two fixes, both one line:

    --mode days N   keep the early threshold through day N (melon clears by ~13)
    --mode value V  trigger on carried VALUE, not unit count. A melon is ~$250,
                    so 2 melons are worth a trip while 10 wheat ($25 each) are
                    the same $250. Unit count is the wrong unit.

    PYTHONPATH=. uv run python tools/make_drop_variant.py --name d-days14 --mode days --days 14
    PYTHONPATH=. uv run python tools/make_drop_variant.py --name d-val400 --mode value --value 400
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

FIND = """        if view.day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)"""

DAYS_REPL = """        if view.day < DROP_EARLY_UNTIL:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)"""

VALUE_REPL = """        if view.day < 10:
            threshold = min(threshold, EARLY_DROP_THRESHOLD)
        # Value-aware trip: a unit count is the wrong unit. Ten wheat and two
        # melons are both ~$250, but only the wheat clears a count of 10, so on
        # melon day nobody goes to the shed and the crop sells a day late into a
        # falling price.
        _val = 0.0
        for _it, _n in inv.items():
            if _it in ("GOOSE", "COW", "SHEEP"):
                continue
            _val += _n * view.market_prices.get(_it, 0)
        if _val >= DROP_VALUE:
            threshold = 1"""


def build(base, name, mode="days", days=14, value=400):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], False
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, DAYS_REPL if mode == "days" else VALUE_REPL, 1)
            text = (f"DROP_EARLY_UNTIL = {int(days)}\n"
                    f"DROP_VALUE = {float(value)}\n\n") + text
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("drop-threshold anchor not found")
    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    g = {"__name__": "preflight"}
    exec(compile(open(p).read(), p, "exec"), g)      # a dead bundle scores $3,000
    if not callable(g.get("agent")):
        raise RuntimeError("bundle exports no agent()")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("days", "value"), default="days")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--value", type=float, default=400)
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.mode, a.days, a.value)}  "
          f"(mode={a.mode}, until day {a.days}, value ${a.value:g})")
