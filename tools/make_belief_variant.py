"""Patch a frozen bundle so its day-12 knob rules can condition on the
OPPONENT'S MEASURED SUPPLY rather than on price.

The shipped rule is `sheep 10 when P_WOOL >= 1.0 at day 12`. Price is a lagging
proxy: it says the wool ladder has not been flooded YET. What actually decides
whether a tenth sheep is worth buying is whether the opponent is about to flood
it, and that is measurable -- market inventory is public and moves by

    inv[t+1] = inv[t] + our_sales + opponent_sales - town_drain

Wool is the ideal product for this: it is never bought back and never consumed,
so our own side of the balance is just our holdings falling, which needs no
action tracking at all. Validated in tools/opp_belief.py to within 4-5% of the
engine's true opponent fills at days 10-14, which is exactly when this fires.

Adds a `max_opp_rate` / `min_opp_rate` condition (units per day, measured over
the preceding window) usable in rules.json alongside min_price and friends.

    PYTHONPATH=. uv run python tools/make_belief_variant.py \
        --base champ-h8 --name sheep-belief \
        --rule '{"knob":"sheep","value":10,"day":12,"max_opp_rate":{"WOOL":4.0}}'
"""

import argparse
import ast
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

# Lives in the bundle's shared namespace: every module is exec'd into one dict,
# so a tracker defined here is reachable from banditpolicy at call time.
TRACKER = '''

# --- opponent supply belief (added by tools/make_belief_variant.py) ----------
_OPP_TRACK = {"prev_inv": None, "prev_hold": None, "hist": []}


def _opp_holdings(obs, seat):
    priv = obs.get("private") or {}
    tot = {}
    for p, n in (priv.get("shed") or {}).items():
        tot[p] = tot.get(p, 0.0) + float(n)
    for inv in (priv.get("inventories") or []):
        for p, n in (inv or {}).items():
            tot[p] = tot.get(p, 0.0) + float(n)
    return tot


def _opp_drain(step, shops):
    d = {}
    if step % 4 == 0:
        for shop in shops or ():
            items = SHOPS.get(shop, ())
            mult = 2 if len(items) == 1 else 1
            for it in items:
                d[it] = d.get(it, 0.0) + mult
    if step % 24 == 0:
        for it in TOWN_CENTER_PRODUCTS:
            d[it] = d.get(it, 0.0) + 1
    return d


def _opp_observe(obs):
    """Track the opponent's sales of products we never buy or consume.

    Restricted on purpose: for WOOL, MILK, STRAWBERRY, MELON and EGG our own
    side of the balance is just holdings falling, so no action tracking is
    needed and the estimate is clean. WHEAT and FERTILIZER are excluded -- we
    buy, feed and fertilize with them, and the balance there is wrong (measured
    NEGATIVE from day 6).
    """
    seat = int(obs.get("player", 0))
    inv = dict((obs.get("market") or {}).get("inventory") or {})
    step = obs.get("step")
    if step is None:
        step = obs.get("day", 0) * 24 + obs.get("hour", 0)
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    hold = _opp_holdings(obs, seat)
    st = _OPP_TRACK
    if st["prev_inv"] is not None:
        drain = st["prev_drain"]
        turn = {}
        for p in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "CARROT", "TOMATO"):
            mine = max(0.0, st["prev_hold"].get(p, 0.0) - hold.get(p, 0.0))
            opp = (inv.get(p, 0) - st["prev_inv"].get(p, 0)) - mine + drain.get(p, 0.0)
            if opp:
                turn[p] = opp
        st["hist"].append((step - 1, turn))
    st["prev_inv"] = inv
    st["prev_hold"] = hold
    st["prev_drain"] = _opp_drain(step, shops)


def opp_rate(product, days=4, now_step=None):
    """Opponent units/day of `product` over the trailing window."""
    st = _OPP_TRACK
    if not st["hist"]:
        return 0.0
    now = now_step if now_step is not None else st["hist"][-1][0]
    lo = now - days * 24
    tot = 0.0
    for s, t in st["hist"]:
        if s >= lo:
            tot += t.get(product, 0.0)
    return tot / float(max(1, days))
'''


def patch(base, name, rules):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out = []
    for mod, text in mods:
        if mod == "econ":
            text = text + TRACKER
        elif mod == "policy":
            # Feed the tracker the real, pre-action observation every turn.
            anchor = ("def agent(observation, configuration=None):\n"
                      "    return _AGENT.act_safe(observation, configuration)")
            if anchor not in text:
                raise RuntimeError("policy agent entry not found")
            text = text.replace(anchor,
                                "def agent(observation, configuration=None):\n"
                                "    try:\n"
                                "        _opp_observe(observation)\n"
                                "    except Exception:\n"
                                "        pass\n"
                                "    return _AGENT.act_safe(observation, configuration)", 1)
        elif mod == "banditpolicy":
            anchor = ('ok = ok and all(ctx["price_ratio"].get(p, 1.0) <= r '
                      'for p, r in rule.get("max_price", {}).items())')
            if anchor not in text:
                raise RuntimeError("bandit condition block not found")
            text = text.replace(anchor, anchor +
                                '\n                ok = ok and all(opp_rate(p) <= r '
                                'for p, r in rule.get("max_opp_rate", {}).items())'
                                '\n                ok = ok and all(opp_rate(p) >= r '
                                'for p, r in rule.get("min_opp_rate", {}).items())', 1)
        elif mod == "weights":
            text = text.replace("BANDIT_RULES = ", "BANDIT_RULES = " + repr(rules)
                                + "  # noqa\n_OLD_RULES = ", 1)
        out.append((mod, text))

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    json.dump(rules, open(os.path.join(d, "rules.json"), "w"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--rule", action="append", required=True,
                    help="JSON condition to ADD, repeatable")
    ap.add_argument("--drop-knob", action="append", default=[],
                    help="remove existing rules for this knob first")
    args = ap.parse_args()

    rules = json.load(open(os.path.join(AGENTS, args.base, "rules.json")))
    conds = [c for c in rules.get("conditions", [])
             if c.get("knob") not in set(args.drop_knob)]
    conds += [json.loads(r) for r in args.rule]
    rules["conditions"] = conds
    p = patch(args.base, args.name, rules)
    print(f"wrote {p}")
    print(f"rules: {json.dumps(conds)}")


if __name__ == "__main__":
    main()
