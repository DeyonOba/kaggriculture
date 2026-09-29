"""Regression: two SELLs in the SAME market-order slot pay the SAME price.

ENGINE (`kaggle_environments/envs/kaggriculture/kaggriculture.py`,
`_process_market`): order slots resolve one at a time for both players, and
INSIDE one slot the per-unit loop builds `quoted[player_id]` from the same
`market["inventory"][item]` before either side commits. Unit k is quoted at one
inventory for both players; the inventory then advances by the units that
committed above the $1 floor. README, "Selling inventory to the market", says
the same thing in words: "when both players issue SELL CARROT 10 first, we take
the current carrot price, give both players that price for their first carrot,
then add 2 carrots to the market".

THE DEFECT. `sell_eval._replay` split every contested block sequentially --
`sell_revenue(inv, mine)` for the first seller and `sell_revenue(inv + mine,
theirs)` for the second -- and in `our_slot="lead"` mode it set
`mine_first = True` without ever looking at the opponent's slot. So it paid us
a within-slot premium the engine does not pay. Measured against the engine:

    both players SELL STRAWBERRY 20 at slot 0, market inventory 10,000
        engine            player0 1670   player1 1670   (difference 0)
        _replay as shipped        2034           1268   (difference 766)

That premium is the whole of the one change this package recommends
(`selling.RECOMMENDED = SellConfig(slots_only=True)`), and on the two traces
the opponent is already at slot 0 on 46 of 71 contested product-steps, so most
of the credited premium was never on offer.

WHY THE EXISTING TESTS MISSED IT. `test_selling._synthetic_trace` puts our SELL
at slot 1 (behind a HIRE) and the opponent's at slot 0 -- deliberately, because
that is champ-h10's real order shape. Every sell_eval test is built on that one
fixture, so `our_lead != their_lead` in all of them and the same-slot branch was
never reached: `test_slot_reorder_is_measured_on_its_own` only ever exercised a
genuine leapfrog (slot 1 -> slot 0 ahead of a slot-0 seller was scored as a
leapfrog too, which is the bug, but with unequal blocks the test's sign
assertions still passed). Nothing pinned the tie, and nothing pinned any
revenue number to an engine measurement.
"""

from __future__ import annotations

import json

import pytest

from kagent import econ

from rlv3 import market_model as mm
from rlv3 import sell_eval

I0 = 10_000
PRODUCTS_L = sell_eval.PRODUCTS_L
SHOPS = ["BRUNCH_SPOT"]

# Measured by running the engine (kaggle_environments.make("kaggriculture"),
# both agents SELL STRAWBERRY 20 on step 0 with 20 in each shed):
#   player0 revenue 1670, player1 revenue 1670.
ENGINE_STRAWBERRY_20v20 = 1670.0


def _trace(tmp_path, my_slot, opp_slot, units=20, item="STRAWBERRY"):
    """Three steps; both players sell `units` of `item` on step 0 only.

    Step 0's inventory is exactly I0, so the block is quoted where the engine
    measurement above was taken. The later inventories honour the identity
    `inv[t+1] = inv[t] + my_fill + their_fill - drain` that
    `attach_opponent_fills` inverts, so the opponent's fill comes back as
    `units` rather than a clipped guess.
    """
    def orders(slot, n):
        q = [["BUY_SEED", "WHEAT", 1]] * slot
        return q + [["SELL", item, n]]

    inv = I0
    steps = []
    for t in range(3):
        inv_map = {p: I0 for p in PRODUCTS_L}
        inv_map[item] = inv
        me = {"market": orders(my_slot, units)} if t == 0 else {"market": []}
        opp = {"market": orders(opp_slot, units)} if t == 0 else {"market": []}
        steps.append({
            "t": t, "day": 0, "hour": t,
            "market": {"prices": [econ.price(p, inv_map[p]) for p in PRODUCTS_L],
                       "inv": [inv_map[p] for p in PRODUCTS_L],
                       "shops": list(SHOPS)},
            "shed": {item: units},
            "act": {"me": me, "opp": opp},
        })
        fills = 2 * units if t == 0 else 0
        inv += fills - int(mm.drain_between(item, t - 1, t, SHOPS))

    path = tmp_path / f"slot{my_slot}v{opp_slot}.json"
    path.write_text(json.dumps({"products": PRODUCTS_L,
                                "final": {"win": True, "margin": 1_000},
                                "steps": steps}))
    return str(path)


def _replay_lead(path, our_slot="lead"):
    steps, _ = sell_eval.load_steps(path)
    return sell_eval._replay(steps, sell_eval.shipped_sizer, our_slot=our_slot)


# ------------------------------------------------------------ the helper
def test_lockstep_matches_the_engine_on_the_measured_block():
    """20 v 20 STRAWBERRY at I0: the engine pays both players 1670."""
    mine, theirs = sell_eval.lockstep_revenue("STRAWBERRY", I0, 20, 20)
    assert mine == pytest.approx(ENGINE_STRAWBERRY_20v20)
    assert theirs == pytest.approx(ENGINE_STRAWBERRY_20v20)
    # And the shipped sequential split is what it is NOT.
    assert econ.sell_revenue("STRAWBERRY", I0, 20) == pytest.approx(2034)
    assert econ.sell_revenue("STRAWBERRY", I0 + 20, 20) == pytest.approx(1268)


@pytest.mark.parametrize("mine,theirs", [(30, 7), (5, 25), (1, 1), (40, 40)])
def test_lockstep_tail_runs_on_alone(mine, theirs):
    """Once the shorter order is exhausted the longer one is an ordinary solo
    walk, so the two sides' unit counts are unchanged and the shorter side is
    never paid more per unit than the longer one."""
    a, b = sell_eval.lockstep_revenue("MILK", I0, mine, theirs)
    n = min(mine, theirs)
    short_rate = (a if mine <= theirs else b) / n
    long_rate = (b if mine <= theirs else a) / max(mine, theirs)
    assert short_rate >= long_rate - 1e-9
    if mine == theirs:
        assert a == pytest.approx(b)


# ------------------------------------------------- the replay, `lead` mode
def test_lead_mode_ties_a_slot_zero_opponent_instead_of_leapfrogging(tmp_path):
    """THE REGRESSION. `lead` puts our SELL at slot 0. The opponent is ALREADY
    at slot 0, so the two resolve in the same slot and the engine quotes both
    at the same inventory. Pre-fix `_replay` set `mine_first = True` regardless
    and handed us the whole block at the pre-sell inventory."""
    path = _trace(tmp_path, my_slot=1, opp_slot=0)
    got = _replay_lead(path)
    assert got["revenue"]["STRAWBERRY"] == pytest.approx(ENGINE_STRAWBERRY_20v20)
    assert got["opp_revenue"]["STRAWBERRY"] == pytest.approx(ENGINE_STRAWBERRY_20v20)


def test_lead_mode_still_leapfrogs_an_opponent_behind_slot_zero(tmp_path):
    """The fix must not throw the real effect away: against an opponent at
    slot 3 our slot-0 order genuinely is quoted first."""
    path = _trace(tmp_path, my_slot=1, opp_slot=3)
    got = _replay_lead(path)
    assert got["revenue"]["STRAWBERRY"] == pytest.approx(
        econ.sell_revenue("STRAWBERRY", I0, 20))
    assert got["opp_revenue"]["STRAWBERRY"] == pytest.approx(
        econ.sell_revenue("STRAWBERRY", I0 + 20, 20))


def test_recorded_mode_tie_pays_no_premium_either(tmp_path):
    """`recorded` mode had the same bug in `<=`: a tie on the slot index gave
    us the whole block first."""
    path = _trace(tmp_path, my_slot=2, opp_slot=2)
    got = _replay_lead(path, our_slot="recorded")
    assert got["revenue"]["STRAWBERRY"] == pytest.approx(ENGINE_STRAWBERRY_20v20)
    assert got["opp_revenue"]["STRAWBERRY"] == pytest.approx(ENGINE_STRAWBERRY_20v20)


# ------------------------------------------------------ the headline number
def test_slot_reorder_credit_is_capped_by_the_engine_not_by_a_leapfrog(tmp_path):
    """`slots_only_*` is what `RECOMMENDED` is sold on. Against an opponent
    already at slot 0 it must be the tie -> tie delta, which is zero, not the
    leapfrog delta the pre-fix code reported."""
    path = _trace(tmp_path, my_slot=1, opp_slot=0)
    got = sell_eval.counterfactual_revenue(path)

    solo = econ.sell_revenue("STRAWBERRY", I0, 20)
    behind = econ.sell_revenue("STRAWBERRY", I0 + 20, 20)
    prefix_delta = solo - behind          # what `mine_first = True` claimed
    prefix_opp_delta = behind - solo

    # Recorded: they are at slot 0, we are at slot 1, so we were quoted second.
    # Lead: we move to slot 0 and TIE them -- both sides land on 1670.
    assert got["slots_only_delta"] == pytest.approx(
        ENGINE_STRAWBERRY_20v20 - behind)
    assert got["slots_only_opp_delta"] == pytest.approx(
        ENGINE_STRAWBERRY_20v20 - solo)
    assert 0 < got["slots_only_delta"] < prefix_delta
    assert prefix_opp_delta < got["slots_only_opp_delta"] < 0
    assert got["slots_only_margin_delta"] == pytest.approx(
        got["slots_only_delta"] - got["slots_only_opp_delta"])
    # The margin the shipped code booked was more than double the real one.
    assert got["slots_only_margin_delta"] < prefix_delta - prefix_opp_delta
