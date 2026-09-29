"""Measurement for rlv3/selling.py, read off tools/agent_scorecard.py traces.

No tools/ edits, no games played. The scorecard trace already records, per
step, the PRE-step market inventory and prices, the open shops, our shed, and
both players' raw actions (`s["act"]["me"]`, `s["act"]["opp"]`) -- which is
everything the exact price identity needs to re-fill a different sell schedule
on the SAME recorded inventory path.

THE PRE-GATE
------------
`counterfactual_revenue` is the cheapest possible falsification of this whole
component. Per-world sd is $34,749, so a schedule worth less than ~$1.5k a
game cannot move a 995-game paired gate and the arm must not be run.
`sell_pace` read +/-0.5% here and was correctly never given an eval arm.

It also reports the OPPONENT's realised revenue under both schedules. The
market is ONE shared inventory: a schedule that lifts our realised price by
withdrawing supply lifts theirs by the same identity, and that is exactly how
`evesell` lost (-1.4 pts, our bank -$1.1k, theirs +$1.8k).

ONE MEASUREMENT CAVEAT, and it matters for how the opponent column is read.
A recorded market order is an OFFER: the engine fills a SELL unit by unit and
stops when that player's shed runs out. Our own offers are shed-clipped before
they are sent (kagent.market._sell_orders sizes on view.shed), and the trace
records our shed, so our side is close to a fill. The opponent's shed is NOT
in the observation and NOT in the trace, so their offers cannot be clipped and
their absolute revenue here is an UPPER BOUND -- inflated by however much they
over-offer. tools/agent_scorecard.market_metrics has the same caveat and says
the comparison still holds because both sides are measured the same way; here
it does not, because only one side can be clipped.

So the kill rule is read in RELATIVE form: `opp_delta_pct` vs `delta_pct`. A
percentage lift is invariant to the over-offer factor to first order (both
replays overstate the same orders identically), which the absolute dollar
column is not.
"""

from __future__ import annotations

import argparse
import json
import statistics as st_

from kagent import econ
from kagent.constants import MARKET_PARAMS, PRICE_FLOOR, SHED_CAPACITY

from rlv3 import market_model as mm
from rlv3.selling import SellConfig

PRODUCTS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]


# ------------------------------------------------------------------ loading
def load_steps(trace_path):
    """[(t, day, hour, prices, inv, shops, shed, my_sells, opp_sells)]."""
    T = json.load(open(trace_path))
    prods = T.get("products", PRODUCTS_L)
    out = []
    for s in T["steps"]:
        px = dict(zip(prods, s["market"]["prices"]))
        inv = dict(zip(prods, s["market"]["inv"]))
        out.append({
            "t": s["t"], "day": s["day"], "hour": s["hour"],
            "prices": px, "inv": inv,
            "shops": list(s["market"].get("shops") or []),
            "shed": dict(s.get("shed") or {}),
            "me": _sells(s["act"].get("me")),
            "opp": _sells(s["act"].get("opp")),
            "my_buys": _buys(s["act"].get("me")),
            "opp_buys": _buys(s["act"].get("opp")),
        })
    return attach_opponent_fills(out), T


def _buys(act):
    """BUY_PRODUCT offers: the only orders that REMOVE market inventory."""
    orders = (act or {}).get("market") or []
    out = {}
    for o in orders:
        if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_PRODUCT":
            out[o[1]] = out.get(o[1], 0) + int(o[2])
    return out


def attach_opponent_fills(steps):
    """Turn the opponent's recorded OFFERS into their exact FILLS.

    Market inventory is an identity:

        inv[t+1] = inv[t] + my_fills + their_fills - drain(step t) - buys

    Every term but `their_fills` is in the trace or is exact: inv on both
    sides, the drain from the open shops (the tick fires at the END of step t,
    after _process_market, when t % 4 == 0), my own fills (my offer clipped by
    my recorded shed, which is how the engine fills unit by unit), and buys --
    which exist only for WHEAT and FERTILIZER, so for the seven products that
    matter the identity is closed and their fill is EXACT, not a bound.

    This is what makes the opponent column of the pre-gate trustworthy: a
    recorded order is an offer, their shed is not observable, and without this
    their revenue is inflated by however much they over-offer.
    """
    for k, s in enumerate(steps):
        offers = {}
        for item, n, _ in s["opp"]:
            offers[item] = offers.get(item, 0) + n
        if k + 1 >= len(steps):
            s["opp_fill"] = dict(offers)            # last step: no identity left
            continue
        nxt = steps[k + 1]
        my_fill = {}
        for item, n, _ in s["me"]:
            my_fill[item] = my_fill.get(item, 0) + min(n, int(s["shed"].get(item, 0)))
        fills = {}
        for item, off in offers.items():
            d_inv = int(nxt["inv"].get(item, 0)) - int(s["inv"].get(item, 0))
            drain = mm.drain_between(item, s["t"], s["t"] + 1, s["shops"])
            buys = s["my_buys"].get(item, 0) + s["opp_buys"].get(item, 0)
            exact = d_inv + drain + buys - my_fill.get(item, 0)
            fills[item] = int(max(0, min(off, round(exact))))
        s["opp_fill"] = fills
    return steps


def _sells(act):
    orders = (act or {}).get("market") or []
    out = []
    for i, o in enumerate(orders):
        if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
            out.append((o[1], int(o[2]), i))
    return out


# ------------------------------------------------------------------- sizers
def shipped_sizer(ctx):
    """The baseline: exactly the orders the traced agent actually sent, filled
    only as far as the shed allowed (the engine fills a SELL unit by unit and
    stops when the shed runs out, so a recorded order is an OFFER)."""
    return [(item, min(n, int(ctx["stock"].get(item, 0)))) for item, n, _ in ctx["me"]]


def make_rlv3_sizer(cfg=None):
    """A drop-in sizer that applies the rlv3 schedule to the recorded offers.

    It is deliberately a REPLAY of the schedule's two mechanical caps --
    accrued-drain participation and the within-order impact budget -- against
    the recorded inventory path, not a re-run of plan_slices: plan_slices
    needs a live View (board, opponent tiles, money) that the trace format
    does not fully carry, and mixing a partial View into the pre-gate would
    make the number un-attributable.
    """
    cfg = cfg or SellConfig()

    def sizer(ctx):
        out = []
        for item, offered, _slot in ctx["me"]:
            inv = ctx["inv_adj"].get(item)
            if inv is None:
                continue
            have = int(ctx["stock"].get(item, 0))
            n = min(offered, have)
            if n <= 0:
                continue
            if not mm.is_drained(item, ctx["shops"]) or ctx["final_push"] or ctx["pressure"]:
                out.append((item, n))                      # ratchet / bypass
                continue
            if cfg.paced_only and item not in ("STRAWBERRY", "MILK", "WOOL", "MELON"):
                out.append((item, n))
                continue
            if ctx["prices"].get(item, 0) <= 0.35 * MARKET_PARAMS[item]["base"]:
                out.append((item, n))                      # crashed release
                continue
            last = ctx["last_slice"].get(item, ctx["day"] * 24 - 1)
            accrued = mm.drain_between(item, last, ctx["t"], ctx["shops"])
            pcap = int(cfg.participation * accrued + ctx["carry"].get(item, 0.0))
            first = econ.price(item, inv)
            icap = econ.units_sellable_above(
                item, inv, first * (1.0 - cfg.impact_cap_pct / 100.0))
            n = min(n, pcap, icap)
            if n > 0:
                out.append((item, n))
        return out

    return sizer


# --------------------------------------------------------- the counterfactual
def lockstep_revenue(item, inv, mine_n, their_n):
    """Both sides' revenue when the two SELLs resolve in the SAME order slot.

    ENGINE (`_process_market`): inside one slot the per-unit loop builds
    `quoted[player_id]` from the same `market["inventory"][item]` before either
    side commits, so for unit k BOTH players are paid the same price; the
    inventory then advances by the units that committed above the $1 floor
    (`_commit_unit` skips the `+= 1` at the floor). There is NO within-slot
    first-seller premium -- README, "Selling inventory to the market": "when
    both players issue SELL CARROT 10 first, we take the current carrot price,
    give both players that price for their first carrot, then add 2 carrots to
    the market".

    Once the shorter order is exhausted the longer one runs on alone, which is
    an ordinary solo walk. Returns (ours, theirs).
    """
    mine = theirs = 0.0
    a, b = int(mine_n), int(their_n)
    while a > 0 and b > 0:
        p = econ.price(item, inv)
        mine += p
        theirs += p
        a -= 1
        b -= 1
        if p > PRICE_FLOOR:
            inv += 2
    if a > 0:
        mine += econ.sell_revenue(item, inv, a)
    elif b > 0:
        theirs += econ.sell_revenue(item, inv, b)
    return mine, theirs


def _replay(steps, sizer, our_slot):
    """Walk the recorded inventory path re-filling our SELL orders under
    `sizer`, and the opponent's recorded orders alongside.

    The shared-inventory correction: wherever our cumulative supply differs
    from what the trace recorded, every later quote (ours AND theirs) moves by
    that difference. Drain and their arrivals stay at their recorded values,
    which is exact to first order and is the only honest thing to do without
    replaying the whole game.

    `our_slot` models engine fact 2 -- slot i resolves for both players before
    slot i+1. "recorded" quotes each side at the slot index the trace actually
    carries (champ-h10 lets sells fill the REMAINING slots, so ours are last
    all season); "lead" puts ours at slot 0, which is what rlv3 build_orders
    does and is free -- it changes no walking and no allocation. Slot 0 is
    AHEAD of an opponent at slot 1+ and LEVEL with one already at slot 0: a
    tie pays no premium to either side (`lockstep_revenue`).
    """
    cum_ours = {p: 0 for p in PRODUCTS_L}       # our units under `sizer`
    cum_shipped = {p: 0 for p in PRODUCTS_L}    # our units as recorded
    stock = {}
    rev = {p: 0.0 for p in PRODUCTS_L}
    opp_rev = {p: 0.0 for p in PRODUCTS_L}
    slices = {p: [] for p in PRODUCTS_L}        # steps at which we sliced
    last_slice, carry = {}, {}
    units = {p: 0 for p in PRODUCTS_L}

    for s in steps:
        # Our shed under the arm: recorded shed, plus whatever the arm held
        # back on earlier steps (capped at the shed, whose midnight overflow
        # the engine destroys).
        held = {p: cum_shipped[p] - cum_ours[p] for p in PRODUCTS_L}
        stock = {p: max(0, int(s["shed"].get(p, 0)) + held[p]) for p in PRODUCTS_L}
        room = max(0, SHED_CAPACITY - sum(int(v) for v in s["shed"].values()))
        for p in PRODUCTS_L:
            if held[p] > room:
                stock[p] = int(s["shed"].get(p, 0)) + room

        inv_adj = {p: int(s["inv"].get(p, 0)) + cum_ours[p] - cum_shipped[p]
                   for p in PRODUCTS_L}
        ctx = dict(s)
        ctx.update({"inv_adj": inv_adj, "stock": stock, "last_slice": last_slice,
                    "carry": carry,
                    "final_push": s["t"] >= 718 - 8,
                    "pressure": sum(int(v) for v in s["shed"].values()) > 80})
        ours = [(i, n) for i, n in sizer(ctx) if n > 0]
        our_units = {}
        for item, n in ours:
            our_units[item] = our_units.get(item, 0) + n

        # Their EXACT fills (attach_opponent_fills), quoted against the same
        # adjusted inventory. Using their raw offers here would inflate their
        # revenue by however much they over-offer, and the kill rule reads off
        # exactly this column.
        theirs = {item: n for item, n in s["opp_fill"].items() if n > 0}
        our_lead = {item: min((sl for i, _, sl in s["me"] if i == item), default=99)
                    for item in our_units}
        their_lead = {item: min((sl for i, _, sl in s["opp"] if i == item), default=99)
                      for item in theirs}

        for item in set(list(our_units) + list(theirs)):
            base_inv = inv_adj.get(item, 0)
            mine_n, their_n = our_units.get(item, 0), theirs.get(item, 0)
            # Engine fact 2: the premium is ACROSS slots only. `lead` puts our
            # SELL at slot 0, so it lands ahead of an opponent at slot 1+ and
            # LEVEL with one already at slot 0 -- it cannot buy a premium off a
            # seller it merely ties. Reading `lead` as "always first" credited
            # us a within-slot premium the engine never pays.
            my_slot = 0 if our_slot == "lead" else our_lead.get(item, 99)
            op_slot = their_lead.get(item, 99)
            if mine_n and their_n:
                if my_slot == op_slot:
                    mine_r, their_r = lockstep_revenue(item, base_inv, mine_n, their_n)
                    rev[item] += mine_r
                    opp_rev[item] += their_r
                elif my_slot < op_slot:
                    rev[item] += econ.sell_revenue(item, base_inv, mine_n)
                    opp_rev[item] += econ.sell_revenue(item, base_inv + mine_n, their_n)
                else:
                    opp_rev[item] += econ.sell_revenue(item, base_inv, their_n)
                    rev[item] += econ.sell_revenue(item, base_inv + their_n, mine_n)
            elif mine_n:
                rev[item] += econ.sell_revenue(item, base_inv, mine_n)
            elif their_n:
                opp_rev[item] += econ.sell_revenue(item, base_inv, their_n)
            units[item] += mine_n

        for item, n in our_units.items():
            cum_ours[item] += n
            slices[item].append(s["t"])
            last_slice[item] = s["t"]
            carry[item] = max(0.0, carry.get(item, 0.0) - n)
        for item, n, _ in s["me"]:
            cum_shipped[item] += min(n, max(0, int(s["shed"].get(item, 0))))

        if s["hour"] == 0:
            carry = {}

    # Units the arm still holds when the trace ends are NOT free: the shipped
    # agent turned them into cash and the season ends at zero. Liquidate them
    # at the last step's adjusted inventory. Without this, "hold everything"
    # would win the pre-gate by inflating the price on a path it never paid
    # for -- the same shape of error as scoring a plan at the marginal price.
    tail = {}
    if steps:
        last = steps[-1]
        final_inv = {p: int(last["inv"].get(p, 0)) + cum_ours[p] - cum_shipped[p]
                     for p in PRODUCTS_L}
        for p in PRODUCTS_L:
            held = cum_shipped[p] - cum_ours[p]
            if held > 0:
                tail[p] = held
                rev[p] += econ.sell_revenue(p, final_inv[p], held)
                units[p] += held
    return {"revenue": rev, "opp_revenue": opp_rev, "units": units,
            "slices": slices, "tail_liquidated": tail}


def counterfactual_revenue(trace_path, sizer=None, cfg=None, slot_order=True):
    """THE PRE-GATE. Per-product and total realised revenue, shipped vs rlv3,
    on the same recorded inventory path. Costs no games and takes seconds."""
    cfg = cfg or SellConfig()
    sizer = sizer or make_rlv3_sizer(cfg)
    steps, T = load_steps(trace_path)
    base = _replay(steps, shipped_sizer, our_slot="recorded")
    # The SLOT re-ordering on its own: same sizes, same volume, same walking --
    # only our order index moves. It withdraws no supply, so it is the one part
    # of this component that cannot lift the opponent's price.
    base_lead = _replay(steps, shipped_sizer, our_slot="lead")
    arm = _replay(steps, sizer, our_slot="lead" if slot_order else "recorded")
    arm_no_slot = _replay(steps, sizer, our_slot="recorded") if slot_order else arm
    out = {
        "trace": trace_path,
        "final": T.get("final", {}),
        "shipped_revenue": sum(base["revenue"].values()),
        "rlv3_revenue": sum(arm["revenue"].values()),
        "delta": sum(arm["revenue"].values()) - sum(base["revenue"].values()),
        "delta_no_slot_order": (sum(arm_no_slot["revenue"].values())
                                - sum(base["revenue"].values())),
        "shipped_opp_revenue": sum(base["opp_revenue"].values()),
        "rlv3_opp_revenue": sum(arm["opp_revenue"].values()),
        "opp_delta": sum(arm["opp_revenue"].values()) - sum(base["opp_revenue"].values()),
        "shipped_units": sum(base["units"].values()),
        "rlv3_units": sum(arm["units"].values()),
        "rlv3_tail_liquidated": sum(arm["tail_liquidated"].values()),
        "per_product": {},
        "slice_gap_steps": slice_gap_steps(arm["slices"]),
        "shipped_slice_gap_steps": slice_gap_steps(base["slices"]),
    }
    out["margin_delta"] = out["delta"] - out["opp_delta"]
    out["delta_pct"] = out["delta"] / max(1.0, out["shipped_revenue"])
    out["opp_delta_pct"] = out["opp_delta"] / max(1.0, out["shipped_opp_revenue"])
    out["slots_only_delta"] = sum(base_lead["revenue"].values()) - out["shipped_revenue"]
    out["slots_only_opp_delta"] = (sum(base_lead["opp_revenue"].values())
                                   - out["shipped_opp_revenue"])
    out["slots_only_margin_delta"] = out["slots_only_delta"] - out["slots_only_opp_delta"]
    for p in PRODUCTS_L:
        if base["units"].get(p) or arm["units"].get(p):
            out["per_product"][p] = {
                "shipped_revenue": base["revenue"][p], "rlv3_revenue": arm["revenue"][p],
                "delta": arm["revenue"][p] - base["revenue"][p],
                "shipped_units": base["units"][p], "rlv3_units": arm["units"][p],
                "opp_delta": arm["opp_revenue"][p] - base["opp_revenue"][p],
            }
    return out


def slice_gap_steps(slices):
    """MECHANISM / anti-self-deception: the median steps between two slices of
    one item. Below cfg.min_slice_gap (4) the schedule has collapsed back into
    sell_pace -- re-offering the remainder on the next step, where the drain is
    zero three times in four -- and the arm is the already-refuted one."""
    gaps = []
    for item, ts in slices.items():
        for a, b in zip(ts, ts[1:]):
            gaps.append(b - a)
    return st_.median(gaps) if gaps else None


# -------------------------------------------------------- relative metrics
def relative_sell_metrics(trace_paths):
    """The metric the existing instrument lacks: revenue_share, relative by
    construction, so it cannot be gamed by banking well while they bank more.

    Also the opponent's realised price over the product's game-mean price, and
    the slice gap we actually shipped.
    """
    rows = []
    for path in trace_paths:
        steps, T = load_steps(path)
        gm = {}
        for p in PRODUCTS_L:
            vals = [s["prices"].get(p, 0.0) for s in steps]
            gm[p] = st_.mean(vals) if vals else 0.0
        mine = theirs = 0.0
        mine_u = theirs_u = 0
        their_realised = 0.0
        my_realised = 0.0
        slices = {}
        for s in steps:
            for item, n, _ in s["me"]:
                inv = int(s["inv"].get(item, 0))
                mine += econ.sell_revenue(item, inv, n)
                mine_u += n
                if gm.get(item):
                    my_realised += n * s["prices"][item] / gm[item]
                slices.setdefault(item, []).append(s["t"])
            for item, n in s["opp_fill"].items():
                if n <= 0:
                    continue
                inv = int(s["inv"].get(item, 0))
                theirs += econ.sell_revenue(item, inv, n)
                theirs_u += n
                if gm.get(item):
                    their_realised += n * s["prices"][item] / gm[item]
        rows.append({
            "trace": path,
            "revenue_share": mine / max(1.0, mine + theirs),
            "our_revenue": mine, "opp_revenue": theirs,
            "our_units": mine_u, "opp_units": theirs_u,
            "our_realised_over_mean": my_realised / max(1, mine_u),
            "opp_realised_over_mean": their_realised / max(1, theirs_u),
            "slice_gap_steps": slice_gap_steps(slices),
            "win": T.get("final", {}).get("win"),
            "margin": T.get("final", {}).get("margin"),
        })
    keys = ("revenue_share", "our_revenue", "opp_revenue", "our_units",
            "opp_units", "our_realised_over_mean", "opp_realised_over_mean")
    mean = {k: st_.mean([r[k] for r in rows]) for k in keys} if rows else {}
    gaps = [r["slice_gap_steps"] for r in rows if r["slice_gap_steps"] is not None]
    mean["slice_gap_steps"] = st_.median(gaps) if gaps else None
    return {"rows": rows, "mean": mean}


# ------------------------------------------------------------------ report
def report(traces, out=None, cfg=None, slot_order=True):
    """Prints beside tools/sell_metrics.py's columns."""
    cfg = cfg or SellConfig()
    cfs = [counterfactual_revenue(t, cfg=cfg, slot_order=slot_order) for t in traces]
    rel = relative_sell_metrics(traces)
    w = 16
    print(f"{'PRE-GATE (counterfactual, no games)':<38}{'value':>{w}}")
    rows = [
        ("games", f"{len(cfs)}"),
        ("shipped sell revenue / game", f"{st_.mean(c['shipped_revenue'] for c in cfs):,.0f}"),
        ("rlv3 sell revenue / game", f"{st_.mean(c['rlv3_revenue'] for c in cfs):,.0f}"),
        ("delta / game", f"{st_.mean(c['delta'] for c in cfs):,.0f}"),
        ("delta without slot order", f"{st_.mean(c['delta_no_slot_order'] for c in cfs):,.0f}"),
        ("opponent delta / game  (exact fills)", f"{st_.mean(c['opp_delta'] for c in cfs):,.0f}"),
        ("our lift %", f"{100 * st_.mean(c['delta_pct'] for c in cfs):,.2f}%"),
        ("opponent lift %  (KILL if >= ours)",
         f"{100 * st_.mean(c['opp_delta_pct'] for c in cfs):,.2f}%"),
        ("margin delta / game", f"{st_.mean(c['margin_delta'] for c in cfs):,.0f}"),
        ("-- slot re-order alone: our delta",
         f"{st_.mean(c['slots_only_delta'] for c in cfs):,.0f}"),
        ("-- slot re-order alone: opp delta",
         f"{st_.mean(c['slots_only_opp_delta'] for c in cfs):,.0f}"),
        ("-- slot re-order alone: margin",
         f"{st_.mean(c['slots_only_margin_delta'] for c in cfs):,.0f}"),
        ("units shipped / rlv3",
         f"{st_.mean(c['shipped_units'] for c in cfs):,.0f} / {st_.mean(c['rlv3_units'] for c in cfs):,.0f}"),
        ("of which liquidated at the tail",
         f"{st_.mean(c['rlv3_tail_liquidated'] for c in cfs):,.0f}"),
        ("slice_gap_steps (KILL < 4)",
         f"{st_.median([c['slice_gap_steps'] for c in cfs if c['slice_gap_steps'] is not None] or [0]):.1f}"),
        ("shipped slice_gap_steps",
         f"{st_.median([c['shipped_slice_gap_steps'] for c in cfs if c['shipped_slice_gap_steps'] is not None] or [0]):.1f}"),
    ]
    for k, v in rows:
        print(f"{k:<38}{v:>{w}}")
    print()
    print(f"{'RELATIVE (recorded schedule)':<38}{'value':>{w}}")
    for k in ("revenue_share", "our_realised_over_mean", "opp_realised_over_mean"):
        print(f"{k:<38}{rel['mean'].get(k, 0):>{w}.3f}")
    print(f"{'our_revenue':<38}{rel['mean'].get('our_revenue', 0):>{w},.0f}")
    print(f"{'opp_revenue':<38}{rel['mean'].get('opp_revenue', 0):>{w},.0f}")
    print()
    print(f"{'per product':<14}{'ship $':>12}{'rlv3 $':>12}{'delta':>12}"
          f"{'ship u':>9}{'rlv3 u':>9}{'opp d$':>12}")
    keys = sorted({p for c in cfs for p in c["per_product"]})
    for p in keys:
        got = [c["per_product"][p] for c in cfs if p in c["per_product"]]
        print(f"{p.lower():<14}"
              f"{st_.mean(g['shipped_revenue'] for g in got):>12,.0f}"
              f"{st_.mean(g['rlv3_revenue'] for g in got):>12,.0f}"
              f"{st_.mean(g['delta'] for g in got):>12,.0f}"
              f"{st_.mean(g['shipped_units'] for g in got):>9,.0f}"
              f"{st_.mean(g['rlv3_units'] for g in got):>9,.0f}"
              f"{st_.mean(g['opp_delta'] for g in got):>12,.0f}")
    verdict = _verdict(cfs)
    print()
    print(verdict)
    if out:
        with open(out, "w") as fh:
            json.dump({"counterfactual": cfs, "relative": rel, "verdict": verdict},
                      fh, indent=1)
    return {"counterfactual": cfs, "relative": rel, "verdict": verdict}


def _verdict(cfs, threshold=1500.0):
    if not cfs:
        return "VERDICT: no traces."
    d = st_.mean(c["delta"] for c in cfs)
    od = st_.mean(c["opp_delta"] for c in cfs)
    dp = st_.mean(c.get("delta_pct", 0.0) for c in cfs)
    odp = st_.mean(c.get("opp_delta_pct", 0.0) for c in cfs)
    gaps = [c["slice_gap_steps"] for c in cfs if c["slice_gap_steps"] is not None]
    gap = st_.median(gaps) if gaps else 0.0
    md = st_.mean(c.get("margin_delta", d - od) for c in cfs)
    sm = st_.mean(c.get("slots_only_margin_delta", 0.0) for c in cfs)
    bad = []
    # THE decision quantity is MARGIN, not our bank. The score is relative and
    # every losing arm in the record banked well while the opponent banked
    # more.
    if md < threshold:
        bad.append(f"margin delta ${md:,.0f}/game < ${threshold:,.0f} (per-world sd "
                   f"$34,749): cannot move a 995-game gate")
    if odp >= dp:
        bad.append(f"the opponent's realised price rises {100 * odp:.2f}% against our "
                   f"{100 * dp:.2f}%: withdrawing supply lifts ONE shared curve and they "
                   "sell into it too (the evesell failure)")
    if gap < 4:
        bad.append(f"slice_gap_steps {gap:.1f} < 4: the schedule collapsed back into sell_pace")
    if bad:
        return ("VERDICT: DO NOT RUN AN EVAL ARM -- " + "; ".join(bad)
                + f". [our bank {d:+,.0f} = {100 * dp:.2f}%, theirs {od:+,.0f} = "
                  f"{100 * odp:.2f}%; slot re-order alone is margin {sm:+,.0f}]")
    return (f"VERDICT: pre-gate PASSED (margin ${md:,.0f}/game; our bank {100 * dp:+.2f}%, "
            f"theirs {100 * odp:+.2f}%, gap {gap:.1f}). Trace one game and confirm SELL is "
            "at slot 0-2 before launching an arm.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+", help="agent_scorecard trace JSON files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--participation", type=float, default=1.0)
    ap.add_argument("--impact-cap-pct", type=int, default=10)
    ap.add_argument("--paced-only", action="store_true")
    ap.add_argument("--no-slot-order", action="store_true")
    a = ap.parse_args()
    cfg = SellConfig(participation=a.participation, impact_cap_pct=a.impact_cap_pct,
                     paced_only=a.paced_only)
    report(a.traces, a.out, cfg, slot_order=not a.no_slot_order)


if __name__ == "__main__":
    main()
