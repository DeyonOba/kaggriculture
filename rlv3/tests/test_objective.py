"""Unit tests for rlv3/objective.py. No game is played.

The tests are grouped by the failure each one is there to prevent, and the
named ones are prior refutations in this repo, not hypotheticals:

  R1  an asymmetric margin term that PAYS TO LOSE -- one algebra slip between
      the two tanh branches and there is a region where a larger loss scores
      higher. Swept densely across the join.
  R2  a margin that outvotes a WIN. The objective is lexicographic; if
      2 * MARGIN_CAP ever reaches the smallest win step the ordering is a
      weighted sum wearing a lexicographic hat.
  R3  Stage B4: a provably policy-invariant potential HALVED held-out wins
      (0.037 -> 0.018, z -3.67, p 0.0002, 785 paired games). Shaping is off
      and the configuration is refused outside {none} / {margin, <= 3 iters}.
  R4  an objective that DRIFTS between resumed training segments. The user's
      plan is to resume from checkpoints across compute-limited days, so the
      fingerprint has to move when a constant moves and not otherwise.
  R5  LinearPotential: an unclamped lookup answered -$850,541 against a
      $5,563 season sd. Every table read is clamped in both arguments.
  R6  the opponent-supply solve. The identity is the engine's, and its two
      documented failure modes -- a one-time HARVEST that deletes the tile,
      and a drain tick counted one step early -- each cost hundreds of units
      per game when they were live.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_objective.py -q
"""
from __future__ import annotations

import json
import math
import os
import re

import pytest

from kagent.constants import BOARD_SIZE, PRODUCTS
from kagent.gamestate import View
from kagent.planner import DayPlan
from tools import shaping as _shaping
from rlv3 import objective as O

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------- fixtures

def make_obs(*, day=10, hour=0, step=None, my_money=5000.0, opp_money=5000.0,
             my_tiles=(), opp_tiles=(), shed=None, unit_inv=None,
             inv=None, shops=("BAKERY",)):
    """A minimal but engine-shaped observation. Hermetic: no file, no game."""
    def board(spec):
        tiles = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for x, y, t in spec:
            tiles[y][x] = t
        return tiles

    def farm(money, spec):
        return {"money": money, "tiles": board(spec), "farmer": [4, 4],
                "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0}

    market_inv = {p: 10_000 for p in PRODUCTS}
    if inv:
        market_inv.update(inv)
    return {
        "player": 0,
        "step": day * 24 + hour if step is None else step,
        "day": day, "hour": hour,
        "farms": [farm(my_money, my_tiles), farm(opp_money, opp_tiles)],
        "private": {"shed": dict(shed or {}), "seeds": {},
                    "inventories": [dict(unit_inv or {})]},
        "market": {"inventory": market_inv,
                   "prices": {p: 100 for p in PRODUCTS}},
        "town": {"unlocked_shops": list(shops)},
    }


def plant(crop, planted_day, yield_units=0, fert=-1):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "watered_today": False, "consecutive_unwatered": 0,
            "yield_units": yield_units, "max_lifespan_step": -1,
            "fertilized_until_day": fert}


def animal(kind, placed_day, yield_units=0, fed=False, fert_avail=False):
    struct = {"GOOSE": "COOP"}.get(kind, "PASTURE")
    return {"kind": struct, "animal": kind, "placed_day": placed_day,
            "yield_units": yield_units, "fed_today": fed,
            "consecutive_unfed": 0, "cared_today": False,
            "fertilizer_available": fert_avail, "pending_care_bonus": 0}


@pytest.fixture
def view():
    return View(make_obs(
        my_tiles=[(0, 0, plant("WHEAT", 6, 3)), (1, 0, plant("STRAWBERRY", 0, 2)),
                  (2, 0, animal("COW", 0, 2))],
        opp_tiles=[(0, 1, plant("MELON", 0, 4)), (1, 1, animal("SHEEP", 0, 3))],
        shed={"WHEAT": 20}))


# ========================================================= (a) the reward

def test_constants_still_mirror_rl_train():
    """Mirrored BY VALUE because rl.train imports torch at module scope.

    A mirror nobody checks is a fork. This reads the source rather than
    importing it, so the test costs nothing and still fails loudly if
    rl/train.py moves underneath us.
    """
    src = open(os.path.join(REPO, "rl", "train.py")).read()
    win = float(re.search(r"^SESSION_WIN_COEF\s*=\s*([\d.]+)", src, re.M).group(1))
    tie = float(re.search(r"^MARGIN_TIE\s*=\s*([\d.]+)", src, re.M).group(1))
    assert O.WIN_COEF == win
    assert O.MARGIN_CAP == tie


def test_win_of_matches_the_rollout_convention():
    src = open(os.path.join(REPO, "rl", "rollout.py")).read()
    assert '"win": 1.0 if my_r > opp_r else (0.5 if my_r == opp_r else 0.0)' in src
    assert (O.win_of(10, 5), O.win_of(5, 5), O.win_of(5, 10)) == (1.0, 0.5, 0.0)


def test_margin_term_is_continuous_and_bounded():
    assert O.margin_term(0.0) == 0.0
    assert abs(O.margin_term(-1e-9)) < 1e-9
    for m in (-1e9, -1e6, 1e6, 1e9):
        assert abs(O.margin_term(m)) <= O.MARGIN_CAP
    assert O.margin_term(1e9) == pytest.approx(O.MARGIN_CAP, abs=1e-9)
    assert O.margin_term(-1e9) == pytest.approx(-O.MARGIN_CAP, abs=1e-9)


def test_margin_term_never_pays_to_lose():
    """R1. Dense sweep across the join.

    STRICTLY increasing out to +/- $150k, which is past any margin this game
    can produce -- both banks together rarely clear $250k and the largest
    |margin| in the corpus is a fifth of that. Beyond it `math.tanh`
    saturates to exactly 1.0 in double precision, so the requirement there is
    only that it never DECREASES: a float ceiling is not an algebra slip, but
    a decrease anywhere would be one.
    """
    xs = sorted(set([i * 0.5 for i in range(-40, 41)]
                    + [i * 137.0 for i in range(-730, 731)]))
    ys = [O.margin_term(x) for x in xs]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        assert y1 > y0, f"margin_term not increasing between {x0} and {x1}"
    assert max(abs(x) for x in xs) > 100_000
    far = [O.margin_term(x) for x in
           (-1e12, -1e9, -1e7, -2e5, 2e5, 1e7, 1e9, 1e12)]
    assert all(b >= a for a, b in zip(far, far[1:]))


def test_margin_term_asymmetry_is_the_objectives_clause_3():
    """dR/dm just below 0 is WIN_SCALE / LOSS_SCALE = 3.1x the slope above."""
    h = 1e-4
    down = (O.margin_term(0.0) - O.margin_term(-h)) / h
    up = (O.margin_term(h) - O.margin_term(0.0)) / h
    assert down / up == pytest.approx(O.WIN_SCALE / O.LOSS_SCALE, rel=1e-4)
    assert down / up == pytest.approx(3.125, rel=1e-3)
    # the measured decision band: champ-h10's -$2,350 is worth more to close
    # than an equal lead is to grow.
    assert abs(O.margin_term(-2350.0)) > abs(O.margin_term(2350.0))


def test_terminal_reward_is_exactly_lexicographic():
    """R2. Every win above +WIN_COEF, every loss below -WIN_COEF. No exceptions."""
    for ours, theirs in [(1, 0), (100_000, 1), (50_001, 50_000),
                         (1_000_000, 999_999)]:
        assert O.terminal_reward(ours, theirs) > O.WIN_COEF
    for ours, theirs in [(0, 1), (1, 100_000), (50_000, 50_001),
                         (999_999, 1_000_000)]:
        assert O.terminal_reward(ours, theirs) < -O.WIN_COEF
    assert O.terminal_reward(7_777, 7_777) == 0.0
    # the narrowest win beats the narrowest loss by more than 2 * WIN_COEF
    assert (O.terminal_reward(50_001, 50_000)
            - O.terminal_reward(50_000, 50_001)) > 2 * O.WIN_COEF


def test_margin_can_never_outvote_a_win_in_the_paired_form():
    """R2, paired. Win differences come in steps of 0.5; the whole margin
    term spans 2 * MARGIN_CAP. The step has to be bigger."""
    assert 2 * O.MARGIN_CAP < 0.5 * O.WIN_COEF
    lost_big = O.terminal_reward(1, 1_000_000, base_final=500, base_opp=400)
    won_small = O.terminal_reward(501, 500, base_final=500, base_opp=400)
    assert won_small > lost_big


def test_money_level_enters_nowhere():
    """reward_design Rule 1: an absolute level measures the world, not the
    player. Lift both banks by $1M and nothing may move."""
    a = O.terminal_reward(60_000, 58_000)
    b = O.terminal_reward(1_060_000, 1_058_000)
    assert a == pytest.approx(b)


def test_episode_rewards_are_sparse_and_terminal():
    ep = {"records": [{}] * 7, "final_money": 90_000.0, "opp_money": 95_000.0}
    r = O.episode_rewards(ep, 0.997)
    assert len(r) == 7
    assert r[:-1] == [0.0] * 6
    assert r[-1] == O.terminal_reward(90_000.0, 95_000.0)
    assert O.episode_rewards({"records": []}, 0.997) == []


def test_episode_rewards_paired_uses_the_twin():
    ep = {"records": [{}] * 3, "final_money": 90_000.0, "opp_money": 95_000.0,
          "base_final": 80_000.0, "base_opp": 95_000.0}
    r = O.episode_rewards(ep, 0.997)
    # both lost the same world; we lost by less, so the reward is positive
    assert r[-1] > 0
    assert r[-1] == O.terminal_reward(90_000.0, 95_000.0,
                                      base_final=80_000.0, base_opp=95_000.0)


# ----------------------------------------------------------- R3: shaping

def test_shaping_is_off_by_default():
    assert O.SHAPING == "none" and O.SHAPING_ANNEAL_ITERS == 0
    outs = [O.Outlook(day=d, our_bank=10_000.0 * d, opp_bank=0.0)
            for d in range(5)]
    assert O.shaping_terms(outs, 0.997) == [0.0] * 5


def test_shaping_refuses_a_stage_b4_configuration():
    """R3. Invariance is asymptotic; 20 iterations is not the asymptote."""
    outs = [O.Outlook(day=d, our_bank=0.0, opp_bank=0.0) for d in range(3)]
    with pytest.raises(ValueError):
        O.shaping_terms(outs, 0.997, mode="margin", anneal=1.0, anneal_iters=4)
    with pytest.raises(ValueError):
        O.shaping_terms(outs, 0.997, mode="portfolio")
    with pytest.raises(ValueError):
        O.shaping_terms(outs, 0.997, mode="margin", anneal=-0.5)
    # inside the cap it is allowed, and it is still a POTENTIAL
    ok = O.shaping_terms(outs, 0.997, mode="margin", anneal=1.0, anneal_iters=3)
    assert len(ok) == 3


def test_enabled_shaping_is_potential_based():
    """It telescopes to gamma^n * 0 - Phi(s_0): a constant per start state,
    which is the whole guarantee."""
    gamma = 0.99
    outs = [O.Outlook(day=d, our_bank=float(3_000 * d), opp_bank=0.0)
            for d in range(6)]
    terms = O.shaping_terms(outs, gamma, mode="margin", anneal=1.0, anneal_iters=1)
    phis = [O.MARGIN_CAP * math.tanh(o.margin / O.WIN_SCALE) for o in outs]
    assert _shaping.discounted_sum(terms, gamma) == pytest.approx(
        _shaping.telescope_sum(phis, gamma), abs=1e-12)


# --------------------------------------------------------- R4: fingerprint

def test_fingerprint_is_stable_and_moves_with_the_constants():
    a = O.fingerprint()
    assert a == O.fingerprint()
    old = O.LOSS_SCALE
    try:
        O.LOSS_SCALE = old + 1.0
        assert O.fingerprint() != a
    finally:
        O.LOSS_SCALE = old
    assert O.fingerprint() == a


# ===================================================== (b) the comparator

def test_calibration_file_is_present_and_well_formed():
    tab = O.load_calibration()
    assert tab["kind"] == "rlv3.p_win"
    assert "pooled" in tab["bands"]
    assert len(tab["margin_knots"]) == len(
        tab["bands"]["pooled"]["rows"][str(tab["day_knots"][0])]["p"])
    assert tab["provenance"]


def test_load_calibration_falls_back_rather_than_raising():
    tab = O.load_calibration("/nonexistent/calibration.json")
    assert tab["kind"] == "rlv3.p_win"
    assert "FALLBACK" in tab["provenance"]


def test_p_win_is_monotone_in_margin_everywhere():
    """A p_win that fell as the margin rose would be clause 1 inverted."""
    ms = [-60_000, -30_000, -12_000, -6_000, -2_000, 0, 2_000, 6_000,
          12_000, 30_000, 60_000]
    for band in [None] + list(range(len(O.EDGES))):
        for day in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29):
            ps = [O.p_win(m, day, band) for m in ms]
            assert all(0.0 <= p <= 1.0 for p in ps)
            assert all(b >= a - 1e-9 for a, b in zip(ps, ps[1:])), (band, day, ps)


def test_p_win_is_clamped_in_both_arguments():
    """R5. Nothing extrapolates: a $10M margin reads as the table's edge."""
    assert O.p_win(1e7, 20) == O.p_win(1e9, 20)
    assert O.p_win(-1e7, 20) == O.p_win(-1e9, 20)
    assert O.p_win(5_000, -50) == O.p_win(5_000, 0)
    assert O.p_win(5_000, 999) == O.p_win(5_000, O.SEASON_END_DAY)


def test_p_win_is_a_coin_flip_early_because_the_corpus_says_so():
    """reward_design Rule 2: through day 12 the lead is sign-correct 50.7%.
    A table that is confident on day 8 is lying about a measured coin flip."""
    for m in (-5_000, -1_000, 0, 1_000, 5_000):
        assert abs(O.p_win(m, 6) - 0.5) < 0.12
    assert O.p_win(20_000, 28) > O.p_win(20_000, 12)


def test_p_win_is_tabulated_per_band_not_pooled():
    tab = O.load_calibration()
    assert set(str(b) for b in range(len(O.EDGES))) <= set(tab["bands"])
    assert O.band_of(300) == 0 and O.band_of(1200) == 2 and O.band_of(9999) == 5
    assert O.band_of(None) is None


def test_outlook_derives_margin_and_p_win_and_is_frozen():
    o = O.Outlook(day=20, our_bank=90_000.0, opp_bank=85_000.0)
    assert o.margin == 5_000.0
    assert o.p_win == O.p_win(5_000.0, 20, None)
    assert o.projects_win
    with pytest.raises(Exception):
        o.our_bank = 1.0


def test_better_is_irreflexive_and_prefers_the_higher_p_win():
    a = O.Outlook(day=20, our_bank=100.0, opp_bank=0.0, p_win=0.60)
    b = O.Outlook(day=20, our_bank=1e6, opp_bank=0.0, p_win=0.40)
    assert not O.better(a, a)
    assert O.better(a, b)          # clause 1 beats a $1M margin
    assert not O.better(b, a)


def test_better_clause_2_maximises_profit_when_winning():
    a = O.Outlook(day=25, our_bank=110_000.0, opp_bank=100_000.0, p_win=0.80)
    b = O.Outlook(day=25, our_bank=103_000.0, opp_bank=100_000.0, p_win=0.80)
    assert O.better(a, b) and not O.better(b, a)


def test_better_clause_3_minimises_the_loss_when_losing():
    a = O.Outlook(day=25, our_bank=98_000.0, opp_bank=100_000.0, p_win=0.20)
    b = O.Outlook(day=25, our_bank=90_000.0, opp_bank=100_000.0, p_win=0.20)
    assert O.better(a, b) and not O.better(b, a)


def test_the_loss_dead_band_is_three_times_tighter():
    """Closing a loss is worth 3.1x growing a lead, so the margin difference
    that counts as a decision is 3.1x smaller on the losing branch."""
    db = O.DEAD_BAND
    d_loss = db * O.LOSS_SCALE
    d_win = db * O.WIN_SCALE
    assert d_win / d_loss == pytest.approx(3.125, rel=1e-9)
    step = 0.5 * (d_loss + d_win)        # decisive when losing, not when winning
    lose_a = O.Outlook(day=25, our_bank=100_000.0 - 1_000, opp_bank=100_000.0, p_win=0.2)
    lose_b = O.Outlook(day=25, our_bank=100_000.0 - 1_000 - step,
                       opp_bank=100_000.0, p_win=0.2)
    win_a = O.Outlook(day=25, our_bank=100_000.0 + 1_000 + step,
                      opp_bank=100_000.0, p_win=0.8)
    win_b = O.Outlook(day=25, our_bank=100_000.0 + 1_000, opp_bank=100_000.0, p_win=0.8)
    assert O.better(lose_a, lose_b)
    assert not O.better(win_a, win_b)


def test_better_prefers_a_projected_win_to_a_projected_loss_inside_the_band():
    a = O.Outlook(day=18, our_bank=100.0, opp_bank=0.0, p_win=0.50)
    b = O.Outlook(day=18, our_bank=0.0, opp_bank=100.0, p_win=0.50)
    assert O.better(a, b) and not O.better(b, a)


def test_rank_agrees_with_better_on_well_separated_outlooks():
    outs = [O.Outlook(day=22, our_bank=1_000.0, opp_bank=0.0, p_win=0.30),
            O.Outlook(day=22, our_bank=9_000.0, opp_bank=0.0, p_win=0.90),
            O.Outlook(day=22, our_bank=500.0, opp_bank=0.0, p_win=0.60)]
    order = O.rank(outs)
    assert order == [1, 2, 0]
    for i, j in zip(order, order[1:]):
        assert O.better(outs[i], outs[j])


def test_score_orders_outlooks_that_project_produced():
    """score() is order-preserving with better() on CONSISTENT outlooks --
    those whose p_win came from their own margin. That is the qualifier."""
    prev = None
    for m in range(-30_000, 30_001, 2_500):
        o = O.Outlook(day=20, our_bank=float(m), opp_bank=0.0)
        if prev is not None:
            assert O.score(o) > O.score(prev)
            assert O.better(o, prev) or o.p_win == prev.p_win
        prev = o


# --------------------------------------------------- projection arithmetic

def test_project_is_self_consistent(view):
    o = O.project(view)
    assert o.day == view.day
    assert o.margin == pytest.approx(o.our_bank - o.opp_bank)
    assert o.our_bank > view.me.money        # standing production is worth something
    assert o.opp_bank > view.opp.money
    assert set(o.our_parts) <= set(PRODUCTS)


def test_project_is_deterministic(view):
    a, b = O.project(view), O.project(view)
    assert (a.our_bank, a.opp_bank, a.p_win) == (b.our_bank, b.opp_bank, b.p_win)


def test_opponent_bank_ignores_their_shed():
    """An oracle on their shed is worth 1.5% of forecast error, and guessing
    it is how a projection starts inventing the opponent."""
    base = View(make_obs(opp_tiles=[(0, 1, plant("MELON", 0, 4))]))
    o1, _ = O.opponent_bank(base)
    loaded = make_obs(opp_tiles=[(0, 1, plant("MELON", 0, 4))])
    loaded["farms"][1]["shed"] = {"MELON": 90}       # invisible to us anyway
    o2, _ = O.opponent_bank(View(loaded))
    assert o1 == o2


def test_opponent_bank_sees_their_board(view):
    """Their BOARD is 36% of forecast error and is fully visible."""
    bare = View(make_obs(opp_tiles=[]))
    rich = View(make_obs(opp_tiles=[(x, 1, plant("STRAWBERRY", 0, 3))
                                    for x in range(6)]))
    assert O.opponent_bank(rich)[0] > O.opponent_bank(bare)[0] + 1_000


def test_project_and_opponent_bank_agree(view):
    o = O.project(view)
    bank, parts = O.opponent_bank(view)
    assert o.opp_bank == pytest.approx(bank)
    assert o.opp_parts == parts


def test_plan_overlay_adds_supply_and_charges_seeds(view):
    plain = O.project(view)
    plan = DayPlan()
    plan.plant_quota = {"WHEAT": 6}
    planted = O.project(view, plan=plan)
    assert planted.our_parts.get("WHEAT", 0) > plain.our_parts.get("WHEAT", 0)
    assert planted.our_bank != plain.our_bank
    free = O.project(view, plan=plan, horizon_end=view.day + 1)
    # nothing this plan grows lands inside one day, so all that is left is
    # the bill: 6 WHEAT seeds at $10.
    assert free.our_bank == pytest.approx(
        O.project(view, horizon_end=view.day + 1).our_bank - 60.0)


def test_our_own_plan_moves_the_OPPONENTS_bank_too(view):
    """The market is ONE shared inventory. Six wheat tiles of ours push the
    wheat price down, and the opponent feeds his sheep out of that same
    market -- so his bank moves. Every losing arm on 2026-09-26 handed him
    $2-7k through this channel and nobody saw it until a 995-game eval came
    back; `opponent_bank` exists to make it visible at projection time."""
    plan = DayPlan()
    plan.plant_quota = {"WHEAT": 6}
    plain, planted = O.project(view), O.project(view, plan=plan)
    assert planted.opp_bank != plain.opp_bank
    assert abs(planted.opp_bank - plain.opp_bank) < abs(
        planted.our_bank - plain.our_bank)


def test_plan_overlay_charges_land():
    v = View(make_obs(my_money=20_000.0))
    plan = DayPlan()
    plan.land_buys = ["NE"]
    assert O.project(v, plan=plan).our_bank == pytest.approx(
        O.project(v).our_bank - 1_000.0)


def test_a_sell_floor_above_every_price_sells_nothing(view):
    plan = DayPlan()
    plan.sell_floors = {"WHEAT": 10_000}     # no wheat unit ever clears this
    floored = O.project(view, plan=plan)
    assert "WHEAT" not in floored.our_parts
    assert floored.our_parts.get("STRAWBERRY") == O.project(
        view).our_parts.get("STRAWBERRY")


def test_feed_is_netted_against_the_harvest_not_bought_back(view):
    """`_rent_board_supply` charges one WHEAT per animal per day as a
    NEGATIVE entry. Pricing each day on its own sells the harvest and buys
    the feed back on the next line, which is not what a farm does."""
    feed = [{"WHEAT": -1}] * 3
    path = {"WHEAT": [10_000] * 4}
    parts_h, cost_h = O._price_side([{"WHEAT": 10}] + feed, path, 3)
    parts_0, cost_0 = O._price_side([{}] + feed, path, 3)
    assert cost_h < cost_0          # the harvest covers the feed it can
    assert cost_0 > 0.0             # with nothing standing, feed is bought
    assert parts_h["WHEAT"] < 10 * 25    # and fewer units reach the market
    assert O.FEED_RESERVE_DAYS == 2 == DayPlan().feed_days


# -------------------------------------------------- R6: the opponent solve

def test_drain_ticks_land_one_step_after_the_engine_step():
    """`_town_consume` fires on step % interval == 0 and its effect appears
    in the NEXT observation. Counting the tick on the observation itself was
    worth 264 units of WHEAT a game."""
    assert O.ProjState._ticks(4, 0, 1) == 1      # step 0 ticks, seen at obs 1
    assert O.ProjState._ticks(4, 1, 2) == 0
    assert O.ProjState._ticks(4, 4, 5) == 1
    assert O.ProjState._ticks(4, 0, 24) == 6     # six shop ticks a day
    assert O.ProjState._ticks(24, 0, 24) == 1    # one town-centre tick a day


def test_drain_is_exact_against_the_shipped_econ_model():
    from kagent import econ
    per_day = econ.town_drain_per_day(["BAKERY", "YARN_STORE"])
    mine = O.ProjState._drain_between(["BAKERY", "YARN_STORE"], 0, 24)
    for item, v in per_day.items():
        assert mine.get(item, 0.0) == pytest.approx(v)


def test_a_one_time_harvest_that_deletes_the_tile_is_still_production():
    """R6. The engine sets the tile to None on a non-ongoing HARVEST. Drop
    that and every WHEAT, CARROT and MELON harvest leaves the ledger -- the
    solve inverted, r -0.79, while that was live."""
    before = View(make_obs(my_tiles=[(0, 0, plant("WHEAT", 5, 6))]))
    after = View(make_obs(my_tiles=[]))          # harvested: tile is None
    made = O.ProjState._produced(O.ProjState._board(before.me),
                                 O.ProjState._board(after.me))
    assert made.get("WHEAT") == 6


def test_a_plant_that_dies_is_not_production():
    before = View(make_obs(my_tiles=[(0, 0, plant("WHEAT", 5, 6))]))
    dead = make_obs(my_tiles=[])
    dead["farms"][0]["tiles"][0][0] = {"kind": "WEED"}
    made = O.ProjState._produced(O.ProjState._board(before.me),
                                 O.ProjState._board(View(dead).me))
    assert made.get("WHEAT", 0) == 0


def test_collected_fertilizer_is_production_but_an_escape_is_not():
    on = View(make_obs(my_tiles=[(2, 2, animal("COW", 0, fert_avail=True))]))
    off = View(make_obs(my_tiles=[(2, 2, animal("COW", 0, fert_avail=False))]))
    assert O.ProjState._produced(O.ProjState._board(on.me),
                                 O.ProjState._board(off.me)).get("FERTILIZER") == 1
    gone = make_obs(my_tiles=[])
    gone["farms"][0]["tiles"][2][2] = {"kind": "PASTURE"}
    assert O.ProjState._produced(O.ProjState._board(on.me),
                                 O.ProjState._board(View(gone).me)
                                 ).get("FERTILIZER", 0) == 0


def test_feed_and_fertilize_are_counted_as_consumption():
    a = O.ProjState._chores(View(make_obs(
        my_tiles=[(2, 2, animal("COW", 0)), (3, 3, animal("COW", 0)),
                  (0, 0, plant("WHEAT", 5))])).me)
    b = O.ProjState._chores(View(make_obs(
        my_tiles=[(2, 2, animal("COW", 0, fed=True)),
                  (3, 3, animal("COW", 0, fed=True)),
                  (0, 0, plant("WHEAT", 5, fert=12))])).me)
    burned = O.ProjState._consumed(a, b)
    assert burned["WHEAT"] == 2           # one per FEED
    assert burned["FERTILIZER"] == 1      # one per FERTILIZE
    # the dawn reset of fed_today is not a negative feed
    assert O.ProjState._consumed(b, a).get("WHEAT", 0) == 0


def test_observe_solves_a_known_opponent_sale():
    """The identity, end to end, on a board where WE do nothing: every unit
    that appears in the market is theirs."""
    st = O.ProjState()
    st.observe(View(make_obs(step=100, day=4, hour=4)))
    after = make_obs(step=101, day=4, hour=5)
    after["market"]["inventory"]["WOOL"] = 10_000 + 7
    st.observe(View(after))
    # one town-centre tick does not fall here; the shop tick at step 100 does
    assert st._day_opp["WOOL"] == pytest.approx(7.0 + 0.0)
    assert st._day_our.get("WOOL", 0.0) == 0.0


def test_observe_does_not_charge_us_for_our_own_sale():
    st = O.ProjState()
    st.observe(View(make_obs(step=101, day=4, hour=5, shed={"WOOL": 7})))
    after = make_obs(step=102, day=4, hour=6, shed={})
    after["market"]["inventory"]["WOOL"] = 10_000 + 7
    st.observe(View(after))
    assert st._day_our["WOOL"] == pytest.approx(7.0)
    assert st._day_opp["WOOL"] == pytest.approx(0.0)


def test_multipliers_start_at_one_so_no_signal_is_the_shipped_model(view):
    st = O.ProjState()
    assert st.conversion == {} and st.fill == {}
    assert O.project(view, state=st).our_bank == pytest.approx(
        O.project(view).our_bank)


def test_the_ewma_is_bounded_and_half_lifed():
    st = O.ProjState()
    assert st.alpha == pytest.approx(1.0 - 0.5 ** (1 / 3.0))
    for _ in range(50):                                  # a pathological outlier
        st._ewma(st.conversion, {"WOOL": 1e9}, {"WOOL": 1.0}, st.CONV_CLAMP)
    assert st.conversion["WOOL"] <= st.CONV_CLAMP[1] + 1e-9
    st2 = O.ProjState()
    st2._ewma(st2.conversion, {"WOOL": 1e9}, {"WOOL": 1.0}, st2.CONV_CLAMP)
    lo, hi = st2.CONV_CLAMP
    assert st2.conversion["WOOL"] <= 1.0 + st2.alpha * (hi - 1.0) + 1e-9


def test_an_undefined_ratio_is_skipped_not_taken_as_zero():
    st = O.ProjState()
    st._ewma(st.conversion, {"WOOL": 0.0}, {"WOOL": 0.0}, st.CONV_CLAMP)
    assert "WOOL" not in st.conversion


def test_nothing_corrects_the_drain():
    """The drain is an exact engine identity. A correction there would be
    fitting noise onto arithmetic we already own."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "objective.py")).read()
    body = src[src.index("def _update("):src.index("def _ewma(")]
    assert "drain" not in body
    assert body.count("self._ewma") == 2


def test_dawn_returns_residuals_and_survives_a_checkpoint(view):
    st = O.ProjState(band=3)
    assert st.dawn(view) == {}                 # nothing to compare on day one
    st.observe(view)
    nxt = View(make_obs(day=view.day + 1, my_money=6_000.0, opp_money=4_000.0,
                        my_tiles=[(0, 0, plant("WHEAT", 6, 3))]))
    res = st.dawn(nxt)
    assert "our_bank" in res and "opp_bank" in res
    assert any(k.startswith("inv:") for k in res)
    blob = st.to_json()
    assert blob["objective"] == O.fingerprint()
    back = O.ProjState.from_json(json.loads(json.dumps(blob)))
    assert back.band == 3 and back.conversion == st.conversion
    assert back.days_seen == st.days_seen


def test_an_ongoing_crops_vanished_tile_is_a_dig_not_a_harvest():
    """Engine HARVEST clears the tile only `if not crop_data["ongoing"]`, so a
    strawberry that disappears was DUG or died. Counting it as production
    reads a dig as a sale, and at 20-40 strawberry tiles a side that was the
    difference between r 0.99 and r 0.5 on the top-board replays."""
    before = View(make_obs(my_tiles=[(0, 0, plant("STRAWBERRY", 0, 4))]))
    after = View(make_obs(my_tiles=[]))
    assert O.ProjState._produced(O.ProjState._board(before.me),
                                 O.ProjState._board(after.me)).get(
                                     "STRAWBERRY", 0) == 0
    # ... while the same tile harvested in place IS production
    picked = View(make_obs(my_tiles=[(0, 0, plant("STRAWBERRY", 0, 0))]))
    assert O.ProjState._produced(O.ProjState._board(before.me),
                                 O.ProjState._board(picked.me))["STRAWBERRY"] == 4


def test_end_of_day_shed_overflow_is_discarded_not_sold():
    """`_drop_inventories_to_shed`: "up to capacity; overflow is discarded."
    A seat holding 154 units loses 54 of them outright at dusk."""
    from kagent.constants import SHED_CAPACITY
    held = {"MILK": 100, "WOOL": 54}
    prev = O.ProjState.seat_snapshot(View(make_obs(day=9, hour=23, unit_inv=held)))
    cur = O.ProjState.seat_snapshot(View(make_obs(day=10, hour=0, unit_inv=held)))
    net = O.ProjState.seat_net(prev, cur)
    over = sum(held.values()) - SHED_CAPACITY
    assert sum(net.values()) == pytest.approx(-over)   # destroyed, not sold
    assert net["MILK"] < 0 and net["WOOL"] < 0
    # under the cap nothing is lost
    small = {"MILK": 10}
    a = O.ProjState.seat_snapshot(View(make_obs(day=9, hour=23, unit_inv=small)))
    b = O.ProjState.seat_snapshot(View(make_obs(day=10, hour=0, unit_inv=small)))
    assert O.ProjState.seat_net(a, b).get("MILK", 0.0) == 0.0


def test_a_floored_product_is_blind_and_is_not_attributed():
    """Engine `_commit_unit`: "Sales at $1 do not increase market supply."
    Once a product floors, units keep leaving both sheds and the market stops
    recording them. Attributing that gap to the opponent invents supply that
    is not there -- measured at -392 MILK units in one top-board game."""
    from kagent.constants import MARKET_I0
    glut = MARKET_I0 + 4_000
    assert O.ProjState._blind("MILK", {"MILK": glut})
    assert not O.ProjState._blind("MILK", {"MILK": MARKET_I0})
    st = O.ProjState()
    st.observe(View(make_obs(step=200, day=8, hour=8, inv={"MILK": glut})))
    st.observe(View(make_obs(step=201, day=8, hour=9, inv={"MILK": glut + 30})))
    assert "MILK" in st.blind
    assert "MILK" not in st._day_opp          # nothing invented
    assert "WHEAT" in st._day_opp             # the live channels still solve


def test_blind_channels_travel_in_the_checkpoint():
    st = O.ProjState()
    st.blind.add("WOOL")
    back = O.ProjState.from_json(json.loads(json.dumps(st.to_json())))
    assert back.blind == {"WOOL"}
