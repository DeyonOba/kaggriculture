"""Reserve market-order slots for SELL on the turns that saturate.

`build_orders` emits in a fixed sequence -- HIRE, feed, land, animals, seeds,
then SELL last -- and self-caps at MAX_MARKET_ORDERS=10 via `add()`, which
silently returns False. Measured over 6 games:

    21 turns/game hit exactly 10 orders
    46.8 SELL orders/game are suppressed by that cap, worth $29,009 at the
      prices prevailing on those turns
    order mix: HIRE 268, SELL 171, BUY_PRODUCT 125, BUY_SEED 43
    HIRE outnumbers SELL 1.6:1 and, being emitted first, wins every contest

The engine's own cap never fires (0 truncations in 30 games) because the agent
caps first, so this is entirely self-inflicted.

The $29,009 bounds a DELAY, not a loss -- end-of-bell stranding is $55/seat, so
the suppressed orders are reissued later and the real cost is price drift over
the delay. That is why the expected effect is small and why this reserves only
a couple of slots rather than restructuring the sequence.

Why this is not m-hire5/m-hire3 (which gated -$7,234 / -$39,678): those capped
hiring GLOBALLY, on every turn. This caps it only when SELLs are actually
waiting and only enough to let them through, so on the ~700 unsaturated turns
a game nothing changes at all.

    PYTHONPATH=. uv run python tools/make_order_variant.py --name o-res2 --reserve 2
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    # --- HIRE (first hours of the day only) ---------------------------------
    if view.hour <= 1:
        need = plan.hires_target - view.me.hires_today
        cost_guard = 0
        hired = 0
        while need > 0 and hired < 8:'''

REPL = '''    # --- HIRE (first hours of the day only) ---------------------------------
    # Hold back a few slots when SELLs are actually queued behind us. Sells are
    # emitted LAST, so on the 21 turns a game that reach the 10-order cap they
    # are the ones silently dropped -- 46.8 of them, worth $29,009 at the
    # prices of the moment.
    # The hire loop already stops at 8, and on a saturated turn it USES all
    # eight -- 166.5 of 210 saturated slots are HIRE. So the reservation has
    # to come off the EIGHT, not off the ten, or it is inert. Calling
    # _sell_orders a second time to count pending sells is also wrong (it
    # needs `memory`), so gate on sellable stock instead.
    _hire_cap = 8
    if _ORDER_RESERVE > 0 and view.shed_total > 0:
        _hire_cap = max(1, 8 - _ORDER_RESERVE)
    if view.hour <= 1:
        need = plan.hires_target - view.me.hires_today
        cost_guard = 0
        hired = 0
        while need > 0 and hired < _hire_cap:'''

GLUE = '''

# --- SELL slot reservation --------------------------------------------------
# Only bites on turns that would otherwise saturate; the other ~700 turns a
# game are untouched, which is what separates this from the global hire caps
# m-hire5/m-hire3 that gated -$7,234 and -$39,678.
_ORDER_RESERVE = __RESERVE__

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--reserve", type=int, required=True,
                    help="slots held for SELL when sells are queued; 0 = base")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "market":
            if ANCHOR not in text:
                raise SystemExit("HIRE anchor not found in market")
            text = (GLUE.replace("__RESERVE__", str(args.reserve))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no market module in bundle")

    p = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(p, exist_ok=True)
    f = os.path.join(p, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(f, "w").write(code)
    compile(code, f, "exec")
    print(f"{args.name}: {args.base} + {args.reserve} SELL slots reserved "
          f"when sells are queued")


if __name__ == "__main__":
    main()
