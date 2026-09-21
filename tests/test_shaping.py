"""Every claim in `tools/shaping.py`, as a test rather than an assertion.

The hypotheses, in the order the spec states them (`docs/shaping_spec.md`):

  H1  correct PBRS telescopes to a constant per start state
  H2  the un-discounted difference does NOT, and the residue has closed form
      `(1 - gamma) * sum gamma^t Phi(s_{t+1})` -- a stall bonus
  H3  correct PBRS preserves the greedy policy on every MDP
  H4  the three defects in `rl/train.py` each FLIP a constructed policy
  H5  Phi(terminal) must be 0 or the boundary term stops being constant
  H6  the potentials obey the two measured reward-design rules

H3/H4 are decided by exact value iteration on tiny deterministic MDPs -- no
sampling, no seeds, no tolerance beyond float epsilon. A property this
project has been burned by is "measured on held-out data and wrong under
intervention"; these are the interventions.

    PYTHONPATH=. uv run pytest tests/test_shaping.py -q
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional, Sequence, Tuple

import pytest

from tools import shaping as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-9


# ------------------------------------------------------------------ a tiny MDP

@dataclass(frozen=True)
class MDP:
    """Deterministic finite MDP. Enough to decide policy invariance exactly."""

    actions: Dict[str, Tuple[str, ...]]
    step: Callable[[str, str], Tuple[str, float]]
    terminal: FrozenSet[str]

    @property
    def states(self) -> Tuple[str, ...]:
        return tuple(self.actions) + tuple(sorted(self.terminal))


Shaping = Optional[Callable[[str, str], float]]


def solve(mdp: MDP, gamma: float, shaping: Shaping = None,
          iters: int = 20_000) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Value iteration to numerical fixpoint, then the greedy policy."""
    def r_of(s: str, a: str) -> Tuple[str, float]:
        s2, r = mdp.step(s, a)
        return s2, r + (shaping(s, s2) if shaping else 0.0)

    V = {s: 0.0 for s in mdp.states}
    for _ in range(iters):
        nxt = dict(V)
        for s in mdp.actions:
            nxt[s] = max(r + gamma * V[s2] for s2, r in
                         (r_of(s, a) for a in mdp.actions[s]))
        if max(abs(nxt[s] - V[s]) for s in V) < 1e-13:
            V = nxt
            break
        V = nxt
    pol = {}
    for s in mdp.actions:
        pol[s] = max(mdp.actions[s],
                     key=lambda a: (lambda p: p[1] + gamma * V[p[0]])(r_of(s, a)))
    return V, pol


def pbrs_of(phi: Dict[str, float], gamma: float) -> Callable[[str, str], float]:
    return lambda s, s2: gamma * phi.get(s2, 0.0) - phi.get(s, 0.0)


def legacy_of(phi: Dict[str, float]) -> Callable[[str, str], float]:
    """What `rl/train.py` computes today: no gamma on the successor."""
    return lambda s, s2: phi.get(s2, 0.0) - phi.get(s, 0.0)


# ------------------------------------------------------- H1: it telescopes

@pytest.mark.parametrize("gamma", [1.0, 0.997, 0.99, 0.9, 0.5])
def test_h1_pbrs_telescopes_to_a_constant(gamma):
    """sum_t gamma^t F_t == gamma^n Phi_T - Phi(s_0), for ANY trajectory."""
    rng = random.Random(0)
    for _ in range(200):
        phis = [rng.uniform(-50, 50) for _ in range(rng.randint(1, 40))]
        got = S.discounted_sum(S.shaping_terms(phis, gamma), gamma)
        assert got == pytest.approx(S.telescope_sum(phis, gamma), abs=1e-9)


def test_h1_the_constant_depends_only_on_the_start_state():
    """Two policies from one start state receive the SAME shaping total.

    This is the whole guarantee: whatever the agent does in between cannot
    change its shaping income, so it cannot be farmed.
    """
    gamma = 0.997
    start = 12.5
    a = [start, 40.0, 5.0, -3.0, 80.0, 0.1]
    b = [start, -20.0, 99.0, 1.0, 2.0, 3.0]
    ta = S.discounted_sum(S.shaping_terms(a, gamma), gamma)
    tb = S.discounted_sum(S.shaping_terms(b, gamma), gamma)
    assert ta == pytest.approx(tb, abs=1e-9)
    assert ta == pytest.approx(-start, abs=1e-9)


# --------------------------------------- H2: the missing gamma is a stall bonus

@pytest.mark.parametrize("gamma", [0.997, 0.99, 0.95])
def test_h2_legacy_bias_matches_its_closed_form(gamma):
    rng = random.Random(1)
    for _ in range(200):
        phis = [rng.uniform(0, 60) for _ in range(rng.randint(1, 40))]
        legacy = S.discounted_sum(S.legacy_shaping_terms(phis), gamma)
        correct = S.discounted_sum(S.shaping_terms(phis, gamma), gamma)
        assert legacy - correct == pytest.approx(S.legacy_bias(phis, gamma), abs=1e-9)


def test_h2_bias_is_positive_and_grows_with_holding_time():
    """Hold the same potential longer and the un-discounted form pays more.

    A constant offset would be harmless. This one is proportional to the
    discounted INTEGRAL of Phi over time, so it prices patience -- the agent
    is paid for how long it sits on the portfolio, in a game that scores only
    what it sold.
    """
    gamma = 0.997
    short = S.legacy_bias([40.0] * 10, gamma)
    long_ = S.legacy_bias([40.0] * 30, gamma)
    assert short > 0.0
    assert long_ > short * 2.5


def test_h2_gamma_one_is_the_only_case_where_legacy_is_correct():
    phis = [3.0, 9.0, -4.0, 20.0]
    assert S.legacy_bias(phis, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert S.legacy_bias(phis, 0.997) > 0.0


# ------------------------------------------- H3: invariance on every MDP

def _random_mdp(rng: random.Random, n: int = 6, k: int = 3) -> MDP:
    names = [f"s{i}" for i in range(n)]
    term = frozenset({names[-1]})
    live = [s for s in names if s not in term]
    table = {}
    for s in live:
        for a in range(k):
            table[(s, f"a{a}")] = (rng.choice(names), rng.uniform(-10, 10))
    return MDP(actions={s: tuple(f"a{a}" for a in range(k)) for s in live},
               step=lambda s, a: table[(s, a)],
               terminal=term)


@pytest.mark.parametrize("gamma", [0.997, 0.9, 0.5])
def test_h3_correct_pbrs_never_changes_the_greedy_policy(gamma):
    """The theorem, checked against exact value iteration on 300 MDPs.

    Phi is drawn adversarially large relative to the rewards precisely to
    show that magnitude cannot matter.
    """
    rng = random.Random(7)
    checked = 0
    for _ in range(300):
        mdp = _random_mdp(rng)
        phi = {s: rng.uniform(-500, 500) for s in mdp.states}
        phi[next(iter(mdp.terminal))] = S.TERMINAL_POTENTIAL
        v_base, base = solve(mdp, gamma)
        _, shaped = solve(mdp, gamma, pbrs_of(phi, gamma))
        for s in base:
            if base[s] != shaped[s]:
                # A differing argmax is only a violation if the unshaped
                # optimum was STRICT; exact ties are broken arbitrarily and
                # both choices are optimal.
                qa = _q(mdp, gamma, v_base, s, base[s])
                qb = _q(mdp, gamma, v_base, s, shaped[s])
                assert qa == pytest.approx(qb, abs=1e-7), (
                    f"PBRS changed a STRICT optimum at {s}: "
                    f"{base[s]} -> {shaped[s]} ({qa:.6f} vs {qb:.6f})")
            checked += 1
    assert checked > 1000


def _q(mdp: MDP, gamma: float, V: Dict[str, float], s: str, a: str) -> float:
    s2, r = mdp.step(s, a)
    return r + gamma * V[s2]


# ------------------------------------- H4: each shipped defect flips a policy

def liquidation_mdp(sale_price: float) -> MDP:
    """The Kaggriculture endgame in two states.

    HOLD a portfolio worth P, or SELL it for `sale_price`. The engine scores
    SOLD product only, so waiting forever scores zero and selling anything
    positive is optimal. Every defect below turns that around.
    """
    return MDP(
        actions={"HOLD": ("WAIT", "SELL")},
        step=lambda s, a: ("HOLD", 0.0) if a == "WAIT" else ("SOLD", sale_price),
        terminal=frozenset({"SOLD"}),
    )


def test_h4a_missing_gamma_makes_never_selling_optimal():
    """Defect 1. `Phi(s') - Phi(s)` pays (1-gamma)*Phi per step to stall."""
    gamma, P, price = 0.997, 400.0, 50.0
    mdp = liquidation_mdp(price)
    phi = {"HOLD": P, "SOLD": 0.0}

    assert solve(mdp, gamma)[1]["HOLD"] == "SELL"
    assert solve(mdp, gamma, pbrs_of(phi, gamma))[1]["HOLD"] == "SELL"
    # and the defect:
    assert solve(mdp, gamma, legacy_of(phi))[1]["HOLD"] == "WAIT"


def test_h4a_the_flip_threshold_is_exactly_the_potential():
    """WAIT wins iff P > sale_price -- i.e. the bribe is the whole portfolio.

    Under `rl/train.py`'s portfolio_value, P is tens of thousands of dollars
    and any single sale is smaller, so the flip is not a corner case: it is
    the generic case for the whole back half of the season.
    """
    gamma, P = 0.997, 400.0
    phi = {"HOLD": P, "SOLD": 0.0}
    assert solve(liquidation_mdp(P - 1.0), gamma, legacy_of(phi))[1]["HOLD"] == "WAIT"
    assert solve(liquidation_mdp(P + 1.0), gamma, legacy_of(phi))[1]["HOLD"] == "SELL"


def melon_mdp(prices: Sequence[float], carry: float = 0.0) -> MDP:
    """Waiting RAISES the price, because the town drains inventory daily.

    `docs/reward_design.md` Rule 5: melon's median sell day correlates
    POSITIVELY with winning (r +0.136) and the intervention agreed --
    `x-mel100`, which harvests at first yield, gated -$1,482 (z -3.52).
    `carry` is `rl/train.py`'s CARRY_COEF * stock, charged per decision.
    """
    n = len(prices)
    states = {f"d{i}": ("WAIT", "SELL") for i in range(n - 1)}
    states[f"d{n - 1}"] = ("SELL",)

    def step(s: str, a: str) -> Tuple[str, float]:
        i = int(s[1:])
        if a == "SELL":
            return "SOLD", prices[i]
        return f"d{i + 1}", -carry

    return MDP(actions=states, step=step, terminal=frozenset({"SOLD"}))


def _first_sell_day(prices: Sequence[float], gamma: float, carry: float) -> int:
    pol = solve(melon_mdp(prices, carry=carry), gamma)[1]
    for i in range(len(prices)):
        if pol[f"d{i}"] == "SELL":
            return i
    return len(prices) - 1


def test_h4b_per_step_carry_cost_flips_melon_to_selling_early():
    """Defect 2. A per-step cost on a state variable is not a difference.

    So it is not potential-based, so it moves the optimum -- and it moves it
    monotonically onto the one intervention the ladder has already refuted:
    the bigger the carry charge, the earlier the agent dumps.
    """
    gamma, prices = 0.99, (100.0, 140.0, 190.0)
    assert _first_sell_day(prices, gamma, 0.0) == 2          # hold to the top
    days = [_first_sell_day(prices, gamma, c) for c in (0, 20, 40, 60, 80)]
    assert days == sorted(days, reverse=True), days          # monotone earlier
    assert days[-1] == 0, days                               # and reaches day 0


def test_h4b_pbrs_cannot_produce_that_flip_for_any_potential():
    """The contrast that makes the point: 500 random potentials, none flips."""
    gamma, prices = 0.99, (100.0, 140.0, 190.0)
    mdp = melon_mdp(prices)
    rng = random.Random(11)
    for _ in range(500):
        phi = {s: rng.uniform(-1e4, 1e4) for s in mdp.states}
        phi["SOLD"] = S.TERMINAL_POTENTIAL
        assert solve(mdp, gamma, pbrs_of(phi, gamma))[1]["d0"] == "WAIT"


def unsold_mdp(penalty_per_unit: float) -> MDP:
    """Sell high and leave stock, or sell low and clear the shed.

    The engine already scores unsold product at ZERO -- it is simply absent
    from `final_money`. `rl/train.py` then subtracts UNSOLD_COEF * its market
    value on top, which prices it NEGATIVE.
    """
    def step(s: str, a: str) -> Tuple[str, float]:
        if a == "KEEP_PREMIUM":       # 50 banked, 60 of stock left behind
            return "END", 50.0 - penalty_per_unit * 60.0
        return "END", 40.0            # dump everything, 40 banked, nothing left

    return MDP(actions={"LAST_DAY": ("KEEP_PREMIUM", "DUMP")},
               step=step, terminal=frozenset({"END"}))


def test_h4c_double_charging_unsold_stock_flips_the_last_day():
    """Defect 3. Charging a cost the objective already prices at zero."""
    gamma = 0.997
    assert solve(unsold_mdp(0.0), gamma)[1]["LAST_DAY"] == "KEEP_PREMIUM"
    assert solve(unsold_mdp(0.2), gamma)[1]["LAST_DAY"] == "DUMP"


# ------------------------------------------- H5: the terminal potential is 0

def test_h5_nonzero_terminal_potential_leaks_the_horizon():
    """With Phi(terminal) != 0 the boundary term depends on WHEN you stop.

    Two policies that reach the terminal at different times then collect
    different shaping totals, and the guarantee is gone.
    """
    gamma = 0.9
    short = S.telescope_sum([10.0] * 3, gamma, terminal_potential=25.0)
    long_ = S.telescope_sum([10.0] * 9, gamma, terminal_potential=25.0)
    assert short != pytest.approx(long_, abs=1e-6)

    z_short = S.telescope_sum([10.0] * 3, gamma, terminal_potential=0.0)
    z_long = S.telescope_sum([10.0] * 9, gamma, terminal_potential=0.0)
    assert z_short == pytest.approx(z_long, abs=1e-12)


# ----------------------------------------- H6: the measured usefulness rules

def test_h6_lead_potential_is_odd_in_the_leads():
    """Phi(-lead) == -Phi(lead): seat-symmetric, so it measures the PLAYER.

    `docs/reward_design.md` Rule 1 -- differencing within the game is worth
    2-6x, because both seats share the seed, the shop draw and one ladder.
    """
    phi = S.LeadPotential()
    for day in (0, 5, 11, 14, 29):
        a = phi({"E:invested_lead": 900.0, "E:money_lead": -1400.0}, day)
        b = phi({"E:invested_lead": -900.0, "E:money_lead": 1400.0}, day)
        assert a == pytest.approx(-b, abs=1e-12)


def test_h6_lead_potential_crosses_over_where_the_corpus_says():
    """Investment early, money late, and nothing but money by the endgame."""
    phi = S.LeadPotential()
    # the recorded features are tanh-squashed; 1.0 is the saturated lead
    inv_only = {"E:invested_lead": 1.0, "E:money_lead": 0.0}
    mon_only = {"E:invested_lead": 0.0, "E:money_lead": 1.0}
    assert phi(inv_only, 2) == pytest.approx(phi.invest_scale)
    assert phi(mon_only, 2) == pytest.approx(0.0)
    assert phi(inv_only, 28) == pytest.approx(0.0)
    assert phi(mon_only, 28) == pytest.approx(phi.money_scale)
    # monotone handover, no reversal
    mix = [phi(mon_only, d) for d in range(30)]
    assert all(b >= a - 1e-12 for a, b in zip(mix, mix[1:]))


def test_h6_validate_catches_early_cash_and_absolute_levels():
    bad = S.LinearPotential(
        coefs={2: {"E:money_lead": 1.0}, 20: {"E:invested": 1.0}},
        intercepts={2: 0.0, 20: 0.0},
    )
    problems = bad.validate()
    assert any("day 2" in p and "cash" in p for p in problems)
    assert any("day 20" in p and "level" in p for p in problems)

    good = S.LinearPotential(
        coefs={2: {"E:invested_lead": 1.0}, 20: {"E:money_lead": 1.0}},
        intercepts={2: 0.0, 20: 0.0},
    )
    assert good.validate() == []


def test_h6_linear_potential_round_trips_and_falls_back_by_day(tmp_path):
    p = S.LinearPotential(coefs={0: {"a": 2.0}, 10: {"a": -1.0}},
                          intercepts={0: 5.0, 10: 1.0}, note="unit")
    path = str(tmp_path / "phi.json")
    p.to_json(path)
    q = S.LinearPotential.from_json(path)
    for day in (0, 3, 4, 5, 6, 10, 29):
        assert q({"a": 3.0}, day) == pytest.approx(p({"a": 3.0}, day))
    assert q({"a": 1.0}, 29) == pytest.approx(0.0)      # nearest row is day 10
    assert q.note == "unit"


def test_gamma_is_validated():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            S.shaping_terms([1.0, 2.0], bad)


# ------------------ H9: what PBRS CANNOT move, which is most of our toolchain

def test_h9_pbrs_is_a_no_op_for_a_paired_episode_return_gate():
    """`tools/ladder_gate.py` pairs two agents on one world and differences
    their episode returns. PBRS adds `-Phi(s_0)` to each, and `s_0` is the
    same state for both (same seed, same board, same opening stock), so the
    paired difference is unchanged to the last bit.

    So no amount of work on Phi can move a gate number, a bandit arm, a CEM
    elite set, or an ES gradient. Phi is only worth anything to a learner
    that bootstraps per step. This is the single most important practical
    consequence of the theorem and it is easy to forget.
    """
    gamma = 0.997
    phi_0 = 31.4                       # identical start state for both arms
    a = [phi_0, 40.0, 12.0, 90.0, 3.0]
    b = [phi_0, -5.0, 77.0, 1.0, 50.0, 2.0, 9.0]
    ret_a, ret_b = 101_300.0, 96_100.0

    plain = ret_a - ret_b
    shaped = ((ret_a + S.discounted_sum(S.shaping_terms(a, gamma), gamma))
              - (ret_b + S.discounted_sum(S.shaping_terms(b, gamma), gamma)))
    assert shaped == pytest.approx(plain, abs=1e-6)


def test_h9_it_is_not_a_no_op_for_a_bootstrapped_per_step_target():
    """The flip side: per-step targets DO change, which is the whole point.

    The episode total is fixed, but where inside the episode the signal sits
    is not -- that is the redistribution a TD critic learns from.
    """
    gamma = 0.997
    phis = [10.0, 55.0, 12.0, 80.0]
    terms = S.shaping_terms(phis, gamma)
    assert max(terms) - min(terms) > 50.0
    assert S.discounted_sum(terms, gamma) == pytest.approx(-phis[0], abs=1e-9)


# ------------------------------- regression guard: the trainer stays correct

def test_trainer_uses_the_discounted_form():
    """`rl/train.py` must not drift back to `Phi(s') - Phi(s)`.

    The defect is one missing multiplication and it survived fourteen PPO
    generations unnoticed, so it gets a guard rather than a comment.
    """
    src = open(os.path.join(REPO, "rl", "train.py")).read()
    assert "shaping_terms" in src or "GAMMA * pvs[i + 1]" in src, (
        "rl/train.py no longer routes its shaping through the discounted "
        "PBRS form -- see tools/shaping.py and docs/shaping_spec.md")
    assert "pvs[i + 1] - pvs[i]" not in src, (
        "rl/train.py is back on the un-discounted difference (H2/H4a)")
