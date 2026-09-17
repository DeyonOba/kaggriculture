"""Recover the OPPONENT's per-product supply rate from public information.

The opponent's shed is hidden, but market inventory is public and moves by
exactly three terms:

    inv[t+1] = inv[t] + our_sales[t] + opponent_sales[t] - town_drain[t]

We know our own sales, and the drain is deterministic given the shop list
(shops consume at step %% 4 == 0, 2 units if single-product else 1 of each
listed; the town centre takes 1 of every non-fertilizer product at
step %% 24 == 0). So the opponent's sales are not estimated, they are SOLVED:

    opponent_sales[t] = inv[t+1] - inv[t] - our_sales[t] + drain[t]

Why this matters. The LP bound showed the opponent term dominates the whole
game (solo $255k against $147k mirrored), and the crop-quota rollout failed
precisely because the right crop mix is world-dependent -- it turns on whether
the opponent is about to flood a ladder. Wool dies after ~59 units and milk
after ~76, so "is anyone else selling this" is the single most valuable hidden
variable in the game, and it is recoverable for free.

MEASURED ACCURACY (3 games, tape-sel3 vs pub-ahmed-v38, against the engine's
own record of opponent fills). Cumulative opponent sales, error vs truth:

             day 10   day 14   day 20
    MILK       +8%      +2%     -19%
    WOOL       +5%      +4%     -31%
    MELON        -      +10%    +10%
    STRAWBERRY   -        -      +8%
    EGG          -        -      +8%

That is the shape the decision needs: crop-mix and herd commitments are made on
days 6-13, where the estimate is within ~5%. It degrades only after day 20, and
for exactly the predicted reason -- a unit sold at the $1 floor is paid for but
is NOT added to market inventory, so once a ladder has collapsed both sides'
sales go invisible and the estimate saturates. It goes blind on products that
are already worthless.

FERTILIZER is now correct through the decision window (true/estimated: 23/23 at
day 6, 65/59 at day 10, 121/97 at day 14) once FERTILIZE consumption is counted.

*** WHEAT IS STILL WRONG AND SHOULD NOT BE USED: 54/-148 at day 6, 97/-331 at
day 14 -- negative, so the error is structural from the start, not a late
accumulation. Wheat is the only product with FOUR simultaneous flows (we buy it,
feed it to animals, harvest it and sell it), and the conservation balance has
the most room for error there; counting FEED fixed fertilizer but not this. It
matters least, though: wheat is a shallow staple whose price sits at $40-46 all
game and never collapses, so "is the opponent flooding it" is not a decision we
act on. The steep ladders that DO drive decisions -- wool, milk, strawberry,
melon -- are accurate to 1-5% through day 14. ***

    PYTHONPATH=. uv run python tools/opp_belief.py --games 3
"""

import argparse
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent.constants import (  # noqa: E402
    PRODUCTS, SHOPS, TOWN_CENTER_PRODUCTS,
    TOWN_SHOP_SELL_INTERVAL, TOWN_CENTER_SELL_INTERVAL,
)


def drain_at(step, shops):
    """Units the town removes at `step`, exactly as _town_consume does."""
    d = defaultdict(float)
    if step % TOWN_SHOP_SELL_INTERVAL == 0:
        for shop in shops or ():
            items = SHOPS.get(shop, ())
            mult = 2 if len(items) == 1 else 1
            for item in items:
                d[item] += mult
    if step % TOWN_CENTER_SELL_INTERVAL == 0:
        for item in TOWN_CENTER_PRODUCTS:
            d[item] += 1
    return d


def holdings(obs):
    """Everything we own of each product: shed plus every unit's inventory.

    Reading the shed alone is WRONG and badly so: harvested produce sits in a
    unit's inventory and is DROPped into the shed during the turn, before the
    market phase resolves, so a start-of-turn shed snapshot showed 4 carrots
    against 278 actually sold in a measured game.
    """
    priv = obs.get("private") or {}
    tot = defaultdict(float)
    for p, n in (priv.get("shed") or {}).items():
        tot[p] += float(n)
    for inv in (priv.get("inventories") or []):
        for p, n in (inv or {}).items():
            tot[p] += float(n)
    return tot


def harvest_gain(obs, action, seat=0):
    """Units our HARVEST and COLLECT_FERTILIZER actions collect this turn.

    The engine hands over exactly tile["yield_units"] on a harvest, so that part
    is exact. COLLECT_FERTILIZER must be counted too -- animals make one unit a
    day each and omitting it understates our own fertilizer sales.
    """
    from kagent.constants import ANIMALS
    me = obs["farms"][seat]
    tiles = me.get("tiles") or []
    units = [me.get("farmer")] + list(me.get("hands") or [])
    acts = [(action or {}).get("farmer")] + list((action or {}).get("hands") or [])
    got = defaultdict(float)
    for i, u in enumerate(units):
        if i >= len(acts):
            break
        a = acts[i]
        if not (isinstance(a, list) and a and a[0] in ("HARVEST", "COLLECT_FERTILIZER")):
            continue
        if not (isinstance(u, (list, tuple)) and len(u) >= 2):
            continue
        try:
            t = tiles[u[1]][u[0]]
        except Exception:
            continue
        if not isinstance(t, dict):
            continue
        if a[0] == "COLLECT_FERTILIZER":
            if t.get("animal") and t.get("fertilizer_available"):
                got["FERTILIZER"] += 1.0
            continue
        n = float(t.get("yield_units", 0) or 0)
        if n <= 0:
            continue
        if t.get("kind") == "PLANT":
            got[t.get("crop")] += n
        elif t.get("animal"):
            a_ = t["animal"]
            a_ = a_.get("type") if isinstance(a_, dict) else a_
            prod = (ANIMALS.get(a_) or {}).get("product")
            if prod:
                got[prod] += n
    return got


def consumption(obs, action, seat=0):
    """Products our own actions BURN this turn, which never reach the market.

    FEED takes one wheat per animal fed and FERTILIZE takes one fertilizer.
    Both are outflows from holdings that are not sales, so omitting them made
    our own sales look larger than they were and pushed the opponent's estimate
    down -- wheat and fertilizer were the only two rows that were wrong.
    """
    me = obs["farms"][seat]
    tiles = me.get("tiles") or []
    units = [me.get("farmer")] + list(me.get("hands") or [])
    acts = [(action or {}).get("farmer")] + list((action or {}).get("hands") or [])
    used = defaultdict(float)
    for i, u in enumerate(units):
        if i >= len(acts):
            break
        a = acts[i]
        if not (isinstance(a, list) and a):
            continue
        if a[0] not in ("FEED", "FERTILIZE"):
            continue
        if not (isinstance(u, (list, tuple)) and len(u) >= 2):
            continue
        try:
            t = tiles[u[1]][u[0]]
        except Exception:
            continue
        if not isinstance(t, dict):
            continue
        # Only count the action when its precondition holds: a FEED on an
        # already-fed animal, or on no animal at all, burns nothing.
        if a[0] == "FEED":
            if t.get("animal") and not t.get("fed_today"):
                used["WHEAT"] += 1.0
        elif t.get("kind") == "PLANT":
            used["FERTILIZER"] += 1.0
    return used


class OpponentBelief:
    """Running estimate of the opponent's cumulative sales, per product.

    Fed one observation per turn plus our own market orders for that turn.
    `rate(product, days)` gives their recent units/day, which is the quantity
    a crop-mix or sell-timing decision actually wants.
    """

    def __init__(self):
        self.cum = defaultdict(float)
        self.history = []           # (step, {product: opp units that turn})
        self._prev_inv = None
        self._pending = None        # (my_fills, drain) for the turn just acted
        self._prev_holdings = None
        self._prev_harvest = {}
        self._prev_buys = {}
        self._prev_used = {}
        self._prev_drain = {}

    def observe(self, obs, my_action=None):
        inv = dict((obs.get("market") or {}).get("inventory") or {})
        step = obs.get("step")
        if step is None:
            step = obs.get("day", 0) * 24 + obs.get("hour", 0)
        shops = (obs.get("town") or {}).get("unlocked_shops") or []

        if self._prev_inv is not None and self._pending is not None:
            my_fills, drain = self._pending
            turn = {}
            for p in PRODUCTS:
                delta = inv.get(p, 0) - self._prev_inv.get(p, 0)
                opp = delta - my_fills.get(p, 0.0) + drain.get(p, 0.0)
                if opp:
                    self.cum[p] += opp
                    turn[p] = opp
            self.history.append((step - 1, turn))

        # Our own sales are recovered by conservation on OUR side:
        #   holdings[t+1] = holdings[t] + harvest[t] + buys[t] - sells[t]
        # Everything on the right but sells is known to us.
        h = holdings(obs)
        if self._prev_holdings is not None:
            sold = {}
            for p in PRODUCTS:
                s_ = (self._prev_holdings.get(p, 0.0) + self._prev_harvest.get(p, 0.0)
                      + self._prev_buys.get(p, 0.0) - self._prev_used.get(p, 0.0)
                      - h.get(p, 0.0))
                if s_ > 0:
                    sold[p] = s_
            self._pending = (sold, self._prev_drain)
        self._prev_holdings = h
        self._prev_harvest = harvest_gain(obs, my_action or {})
        # BUY_PRODUCT adds to holdings, so omitting it understates our sells by
        # the amount bought (~350 wheat a game). The engine stops a buy when
        # money runs out; requested is the best we can do from our own side.
        # ...and capped by what we could afford: an oversized BUY_PRODUCT is
        # stopped when money runs out, so taking `requested` at face value
        # inflated our own sells enough to drive the wheat estimate NEGATIVE.
        buys = defaultdict(float)
        money = float(obs["farms"][int(obs.get("player", 0))].get("money", 0))
        prices = (obs.get("market") or {}).get("prices") or {}
        for o in ((my_action or {}).get("market") or ()):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_PRODUCT":
                try:
                    want = float(o[2])
                except (TypeError, ValueError):
                    continue
                px = max(1.0, float(prices.get(o[1], 1) or 1))
                afford = money // px
                n = min(want, afford)
                buys[o[1]] += n
                money -= n * px
        self._prev_buys = buys
        self._prev_used = consumption(obs, my_action or {})
        self._prev_drain = drain_at(step, shops)
        self._prev_inv = inv

    def rate(self, product, days=3, now_step=None):
        """Opponent units/day of `product` over the last `days` in-game days."""
        if not self.history:
            return 0.0
        now = now_step if now_step is not None else self.history[-1][0]
        lo = now - days * 24
        tot = sum(t.get(product, 0.0) for s, t in self.history if s >= lo)
        return tot / max(1.0, days)


# --------------------------------------------------------------------------
# Validation: compare against the engine's own record of opponent fills
# --------------------------------------------------------------------------

def _load_bundle(path):
    """Exec a frozen bundle and hand back its `agent` callable."""
    g = {"__name__": "belief_probe"}
    exec(compile(open(path).read(), path, "exec"), g)
    return g["agent"]


def validate(agent, opp, seeds, verbose=True):
    """Run the belief INSIDE an agent, which is the only correct way.

    env.steps[t] stores the state AFTER turn t's actions -- a recorded FEED
    lands on a tile whose fed_today is already True -- so replaying the step
    list feeds the belief post-action tiles and its consumption and harvest
    terms silently read as zero. A live agent sees the true pre-action
    observation, so the wrapper below is both the honest test and the shape the
    thing ships in.
    """
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments import make

    truth = defaultdict(float)
    pend = defaultdict(lambda: defaultdict(float))
    _commit, _market = K._commit_unit, K._process_market

    def commit(op, item, price, farm, private, market, cap=100):
        ok = _commit(op, item, price, farm, private, market, cap)
        if ok and op == "SELL":
            pend[id(farm)][item] += 1
        return ok

    def mhook(state, env):
        r = _market(state, env)
        for i, f in enumerate(state[0].observation.farms):
            got = pend.pop(id(f), {})
            if i == 1:                      # seat 1 is the opponent
                for k, v in got.items():
                    truth[k] += v
        pend.clear()
        return r

    K._commit_unit, K._process_market = commit, mhook
    inner = _load_bundle(agent)
    rows = []
    try:
        for sd in seeds:
            truth.clear()
            belief = OpponentBelief()

            def tracked(obs, configuration=None, _b=belief):
                act = inner(obs, configuration)
                _b.observe(obs, act)
                return act

            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd},
                       debug=False)
            env.run([tracked, opp])
            rows.append((sd, dict(truth), dict(belief.cum)))
    finally:
        K._commit_unit, K._process_market = _commit, _market

    if verbose:
        print(f"\n{len(rows)} games -- opponent cumulative sales, ESTIMATED from public "
              f"state vs the engine's TRUE fills\n")
        print(f"{'product':<12} {'true':>9} {'estimated':>10} {'error':>9} {'err %':>7}")
        agg_t, agg_e = defaultdict(float), defaultdict(float)
        for _, t, e in rows:
            for p in PRODUCTS:
                agg_t[p] += t.get(p, 0.0)
                agg_e[p] += e.get(p, 0.0)
        n = len(rows)
        for p in PRODUCTS:
            tv, ev = agg_t[p] / n, agg_e[p] / n
            err = ev - tv
            print(f"{p:<12} {tv:>9,.1f} {ev:>10,.1f} {err:>+9,.1f} "
                  f"{(err/tv*100 if tv else 0):>6.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="arena/agents/tape-sel3/main.py")
    ap.add_argument("--opp", default="arena/agents/pub-ahmed-v38/main.py")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=61000)
    args = ap.parse_args()
    validate(args.agent, args.opp, [args.base_seed + k for k in range(args.games)])


if __name__ == "__main__":
    main()
