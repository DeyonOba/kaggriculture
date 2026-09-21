"""Put the shops that have not opened yet into the rent forecast.

`_rent_inventory` projects the market inventory a crop's block will be sold
into, and prices demand as:

    drain = econ.town_drain_per_day(view.unlocked_shops)   # TODAY's shops
    for k in 1..horizon: cur[p] += supply - drain[p]        # held CONSTANT

README Town Buildings says demand does not hold constant. A shop unlocks
every `townShopUnlockInterval` (3) days, drawn **uniformly with replacement**
from the eight types, capped at eight instances, and each instance consumes
one of every product it demands every `townShopSellInterval` (4) turns —
6 a day, doubled for a single-product shop. With `_RENT_HORIZON = 14` the
forecast misses roughly four future draws.

It misses them **unevenly**, and that is the point. Expected added demand per
future instance, straight off the shop table:

    WHEAT       5 of 8 types   3.75 units/day
    STRAWBERRY  4              3.00
    CARROT      2 (x2 pet cafe) 2.25
    MILK        3              2.25
    TOMATO      2              1.50
    EGG         2              1.50
    WOOL        1 (x2 yarn)    1.50
    MELON       0              0.00   <- can never gain demand
    FERTILIZER  0              0.00   <- no shop consumes it

So melon is the one crop whose price can only ever be pushed down, and the
forecast currently treats its demand exactly like wheat's.

WHY THIS IS NOT THE GLUT PENALTY AGAIN. `tools/make_glut_variant.py` failed
(−$1,230, monotone in λ) because `_rent_pick` already prices self-impact
through `econ.sell_revenue` — the penalty charged the same risk twice. This
is the opposite case: the **demand** side of the same forecast, which
`_rent_inventory` genuinely does not model. Nothing else in the planner
reads future shop draws.

`--lam` scales the expectation: 1.0 trusts it fully, 0.5 halves it. λ = 0 is
bit-identical to the base by construction.

    PYTHONPATH=. uv run python tools/make_futdemand_variant.py --name x-fut10 --lam 1.0
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def future_table():
    """Expected units/day of NEW demand per future shop instance, per product."""
    from kagent.constants import SHOPS, PRODUCTS, TURNS_PER_DAY, TOWN_SHOP_SELL_INTERVAL
    per_day = TURNS_PER_DAY / float(TOWN_SHOP_SELL_INTERVAL)      # 6 ticks/day
    n = len(SHOPS)
    out = {}
    for p in PRODUCTS:
        weight = sum((2 if len(v) == 1 else 1) for v in SHOPS.values() if p in v)
        out[p] = round(weight / float(n) * per_day, 4)
    return out


FIND = '''    cur = dict(view.market_inv)
    drain = econ.town_drain_per_day(view.unlocked_shops)
    day = view.day
    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)
    path = {p: [cur[p]] for p in cur}
    for k in range(1, horizon + 1):
        for p in cur:
            cur[p] += (mine[k].get(p, 0) + opp[k].get(p, 0)
                       - drain.get(p, 0.0))
            path[p].append(cur[p])
    return path'''

REPL = '''    cur = dict(view.market_inv)
    drain = econ.town_drain_per_day(view.unlocked_shops)
    day = view.day
    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)
    # README Town Buildings: a shop unlocks every _FUT_EVERY days, drawn
    # uniformly WITH REPLACEMENT from the eight types and capped at
    # _FUT_CAP instances. Holding `drain` at today's shops understates
    # demand over a 14-day horizon, and understates it unevenly -- melon is
    # in no shop and can never gain any, wheat is in five of eight.
    _open_now = len(view.unlocked_shops)
    path = {p: [cur[p]] for p in cur}
    for k in range(1, horizon + 1):
        _d = day + k
        _new = max(0, _d // _FUT_EVERY - day // _FUT_EVERY)
        _new = min(_FUT_CAP, _open_now + _new) - _open_now
        for p in cur:
            cur[p] += (mine[k].get(p, 0) + opp[k].get(p, 0)
                       - drain.get(p, 0.0)
                       - _FUT_LAM * _new * _FUT.get(p, 0.0))
            path[p].append(cur[p])
    return path'''


def build(base, name, lam):
    from kagent.constants import MAX_SHOP_INSTANCES, TOWN_SHOP_UNLOCK_INTERVAL
    table = future_table()
    head_block = (f"\n_FUT = {table!r}\n"
                  f"_FUT_LAM = {lam}\n"
                  f"_FUT_EVERY = {TOWN_SHOP_UNLOCK_INTERVAL}\n"
                  f"_FUT_CAP = {MAX_SHOP_INSTANCES}\n\n")

    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], False
    for mod, text in mods:
        if mod == "planner" and FIND in text:
            text = head_block + text.replace(FIND, REPL, 1)
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit(f"{base} has no _rent_inventory body to patch "
                         f"-- build it from a rent bundle")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    rules = os.path.join(REPO, "arena", "agents", base, "rules.json")
    if os.path.isfile(rules):
        import shutil
        shutil.copyfile(rules, os.path.join(dst_dir, "rules.json"))

    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  lam={lam}")
    print("  E[units/day] per future shop instance: "
          + "  ".join(f"{k} {v}" for k, v in sorted(table.items()) if v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="x-rentfx")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    build(args.base, args.name, args.lam)


if __name__ == "__main__":
    main()
