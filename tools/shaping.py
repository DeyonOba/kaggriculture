"""Potential-based reward shaping: the only shaping form that cannot lie to us.

WHY THIS FILE EXISTS. Every intermediate signal this project has added moved
the policy the wrong way while looking right on held-out data -- BC at 69.4%
accuracy gated null, the value head got the sign wrong on 3 of 4 arms,
capacity channels bought +0.0187 R^2 and -44% out-of-distribution, the melon
glut penalty was arithmetically correct and lost $1,230. The pattern has a
name in this repo: observationally correct != interventionally correct.

Potential-based reward shaping (Ng, Harada & Russell 1999) is the one
construction that escapes it, because its correctness is a *theorem* rather
than a gate result. For any real-valued Phi over states,

    F(s, a, s') = gamma * Phi(s') - Phi(s)                            (PBRS)

leaves the set of optimal policies of the underlying MDP EXACTLY unchanged,
and that form is necessary as well as sufficient. Phi may be arbitrarily
wrong, fitted on the wrong distribution, or actively adversarial: it can
change how fast a learner converges and it cannot change what it converges
to. So a fitted model entering here carries none of the risk the same model
carries as a decision rule.

    Ng, Harada & Russell, "Policy invariance under reward transformations"
    (ICML 1999); Wiewiora, "Potential-based shaping and Q-value
    initialization are equivalent" (JAIR 2003), arXiv:1106.5267.

THREE THINGS THE FORM WILL NOT SURVIVE, all of which are in `rl/train.py`
today and all of which are checked against in `tests/test_shaping.py`:

  1. dropping the gamma. `Phi(s') - Phi(s)` does not telescope under a
     discount. The discounted sum carries a residue

         bias = (1 - gamma) * sum_t gamma^t * Phi(s_{t+1})            (B1)

     which is strictly positive wherever Phi is, i.e. the agent is paid, per
     step, simply for *remaining* in a high-potential state. With
     Phi = portfolio value that is a standing bribe to accumulate assets and
     never liquidate -- against an engine whose scoring rule is that unsold
     inventory is worth zero.

  2. a per-step cost on a state variable (`- CARRY_COEF * stock_t`). Not a
     difference of anything, so not potential-based, so it moves the optimum.
     It moves it toward selling held product earlier, which is the exact
     intervention `x-mel100` tested and lost $1,482 on (z -3.52).

  3. a terminal penalty on a quantity the environment already prices
     (`- UNSOLD_COEF * unsold_value`). The engine scores SOLD product only,
     so unsold stock is already worth 0; charging it again prices it at
     -0.2x. Same failure as the glut penalty: before adding a term, find
     where in the objective the quantity already enters.

TIME-DEPENDENT POTENTIALS ARE FINE. Phi(s, t) keeps the guarantee (Devlin &
Kudenko, "Dynamic potential-based reward shaping", AAMAS 2012). That matters
here because the corpus says the right potential is not stationary: capital
on the board is an asset at day 4 (r +0.366) and a liability at day 20
(r -0.200), and the money lead is worth literally nothing before day 14
(r ~ 0.00, sign-correct 50%) and everything after (0.552 -> 0.892). A
stationary Phi cannot express that; a Phi(s, t) can, and still cannot bias
the optimum.

THE LIMIT, STATED UP FRONT. Along any trajectory the shaping telescopes to
`gamma^n * Phi(s_n) - Phi(s_0)`, i.e. to a constant per start state. So for
any search that consumes only the EPISODE RETURN -- evolution strategies,
CEM, the knob bandit, `tools/ladder_gate.py` -- PBRS is identically a no-op.
It is worth exactly nothing outside a learner that bootstraps per step. See
`docs/shaping_spec.md` S5.

    PYTHONPATH=. uv run pytest tests/test_shaping.py -q
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "pbrs_increment",
    "shaping_terms",
    "legacy_shaping_terms",
    "discounted_sum",
    "telescope_sum",
    "legacy_bias",
    "Potential",
    "LinearPotential",
    "LeadPotential",
    "ZeroPotential",
    "TERMINAL_POTENTIAL",
    "CASH_FEATURES",
    "LEVEL_FEATURES",
    "CASH_EMBARGO_DAY",
]

# The episodic guarantee needs Phi(terminal) = 0. Ng et al. carve the
# absorbing state out explicitly; if it is given any other value the
# telescoped boundary term stops being policy-independent as soon as two
# policies can terminate at different times.
TERMINAL_POTENTIAL = 0.0

# `docs/reward_design.md` Rule 2: the money lead correlates r ~ 0.00 with the
# final margin through day 12 and is sign-correct 50% of the time. A
# potential that reads cash before this day is not wrong -- nothing can make
# PBRS wrong -- it is USELESS, and worse, it makes early cash look like
# progress to the value head. Enforced by `LinearPotential.validate`.
CASH_EMBARGO_DAY = 14
CASH_FEATURES = ("E:money", "E:money_lead", "E:money_over_next_land")

# `docs/reward_design.md` Rule 1: both seats share the seed, the shop draw
# and one price ladder, so an absolute metric is attenuated by everything
# they share -- differencing within the game is worth 2-6x (WOOL revenue
# r +0.097 absolute -> +0.445 relative). A potential built on levels is
# measuring the world, not the player.
LEVEL_FEATURES = ("E:invested", "E:money", "E:occ_all", "E:hands")


# --------------------------------------------------------------- the theorem

def pbrs_increment(phi_s: float, phi_next: float, gamma: float) -> float:
    """F(s, a, s') = gamma * Phi(s') - Phi(s).

    The gamma multiplies the SUCCESSOR potential and only the successor
    potential. This is not a convention -- it is what makes the discounted
    sum telescope, and Ng et al. prove the form is necessary for policy
    invariance, not merely sufficient.
    """
    return gamma * phi_next - phi_s


def shaping_terms(
    potentials: Sequence[float],
    gamma: float,
    terminal_potential: float = TERMINAL_POTENTIAL,
) -> List[float]:
    """Per-step shaping F_t for a trajectory that ends after the last state.

    `potentials[t]` is Phi(s_t) for the t-th DECISION state. The successor of
    the final decision state is the absorbing terminal, whose potential must
    be `TERMINAL_POTENTIAL`.
    """
    _check_gamma(gamma)
    if not potentials:
        return []
    nxt = list(potentials[1:]) + [terminal_potential]
    return [pbrs_increment(a, b, gamma) for a, b in zip(potentials, nxt)]


def legacy_shaping_terms(
    potentials: Sequence[float],
    terminal_potential: float = TERMINAL_POTENTIAL,
) -> List[float]:
    """`Phi(s') - Phi(s)` -- the un-discounted difference in `rl/train.py`.

    Kept so the defect can be measured rather than asserted. Correct only at
    gamma == 1.
    """
    if not potentials:
        return []
    nxt = list(potentials[1:]) + [terminal_potential]
    return [b - a for a, b in zip(potentials, nxt)]


def discounted_sum(terms: Sequence[float], gamma: float) -> float:
    """sum_t gamma^t * terms[t]."""
    _check_gamma(gamma)
    return math.fsum(gamma ** t * v for t, v in enumerate(terms))


def telescope_sum(
    potentials: Sequence[float],
    gamma: float,
    terminal_potential: float = TERMINAL_POTENTIAL,
) -> float:
    """Closed form of the discounted PBRS sum: gamma^n * Phi_T - Phi(s_0).

    Depends on the start state and the horizon, and on NOTHING the policy
    chose in between. That is the whole guarantee, in one line.
    """
    _check_gamma(gamma)
    if not potentials:
        return 0.0
    n = len(potentials)
    return gamma ** n * terminal_potential - potentials[0]


def legacy_bias(
    potentials: Sequence[float],
    gamma: float,
    terminal_potential: float = TERMINAL_POTENTIAL,
) -> float:
    """How much free return the missing gamma hands out, in reward units.

    Closed form (B1): `(1 - gamma) * sum_t gamma^t * Phi(s_{t+1})`. Positive
    whenever Phi is, and proportional to how LONG the potential is held --
    which is what makes it a stall bonus rather than a constant.
    """
    _check_gamma(gamma)
    if not potentials:
        return 0.0
    nxt = list(potentials[1:]) + [terminal_potential]
    return (1.0 - gamma) * math.fsum(
        gamma ** t * v for t, v in enumerate(nxt)
    )


def _check_gamma(gamma: float) -> None:
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")


# ------------------------------------------------------------------ potentials

class Potential:
    """Phi(features, day) -> float, in dollars of expected final margin.

    Subclasses must be PURE state functions. A potential that reads anything
    outside the state it is handed -- the action taken, the step counter of
    the learner, a running average -- silently leaves the theorem behind.
    """

    def __call__(self, feats: Mapping[str, float], day: int) -> float:
        raise NotImplementedError

    def trajectory(
        self, rows: Iterable[Tuple[Mapping[str, float], int]]
    ) -> List[float]:
        return [self(f, d) for f, d in rows]


class ZeroPotential(Potential):
    """Phi == 0. The control arm: PBRS with this is exactly no shaping."""

    def __call__(self, feats: Mapping[str, float], day: int) -> float:
        return 0.0


@dataclasses.dataclass(frozen=True)
class LinearPotential(Potential):
    """Per-day linear map from metrics to expected final margin, in dollars.

    Day-dependent by construction (Devlin & Kudenko 2012), because the sign
    of the strongest early feature FLIPS between day 8 and day 10 and no
    stationary Phi can hold both ends.

    `coefs[day]` is a sparse {feature: weight}; `intercepts[day]` completes
    it. A day with no row falls back to the nearest day present, so a fit
    that skips days still yields a total function.

    A linear model extrapolates without limit, and a metric that barely moved
    across the training games can arrive far outside that range at inference
    -- measured: `E:money_over_next_land` at day 17 is 3,677 training sigmas
    out on one held-out game, and the un-clamped potential answered -$850,541
    against a season margin whose whole standard deviation is $5,563.
    `clip[day]` is the range the fit actually SAW, so the potential stays a
    bounded function of state. Clamping cannot affect policy invariance --
    a clamped state function is still a state function.
    """

    coefs: Mapping[int, Mapping[str, float]]
    intercepts: Mapping[int, float]
    clip: Mapping[int, Tuple[float, float]] = dataclasses.field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.coefs:
            raise ValueError("LinearPotential needs at least one day of coefficients")
        object.__setattr__(self, "_days", tuple(sorted(self.coefs)))

    def _row(self, day: int) -> Tuple[Mapping[str, float], float, int]:
        days: Tuple[int, ...] = getattr(self, "_days")
        d = day if day in self.coefs else min(days, key=lambda k: abs(k - day))
        return self.coefs[d], self.intercepts.get(d, 0.0), d

    def __call__(self, feats: Mapping[str, float], day: int) -> float:
        row, b, d = self._row(day)
        v = b + math.fsum(w * float(feats.get(k, 0.0)) for k, w in row.items())
        lo, hi = self.clip.get(d, (-math.inf, math.inf))
        return min(hi, max(lo, v))

    # -- hygiene ------------------------------------------------------------
    def validate(self, embargo_day: int = CASH_EMBARGO_DAY) -> List[str]:
        """Return the reward-design rules this potential breaks.

        These are USEFULNESS rules, not correctness rules. Breaking them
        cannot bias the optimum -- it can only spend the shaping budget on a
        signal measured to be worthless.
        """
        problems: List[str] = []
        for day, row in sorted(self.coefs.items()):
            if day < embargo_day:
                hit = [k for k in row if k in CASH_FEATURES and row[k] != 0.0]
                if hit:
                    problems.append(
                        f"day {day} < {embargo_day} reads cash {hit} "
                        f"(reward_design Rule 2: r ~ 0.00 before day 14)"
                    )
            lvl = [k for k in row if k in LEVEL_FEATURES and row[k] != 0.0]
            if lvl:
                problems.append(
                    f"day {day} reads absolute level {lvl} "
                    f"(reward_design Rule 1: use the lead, worth 2-6x)"
                )
        return problems

    # -- io -----------------------------------------------------------------
    def to_json(self, path: str) -> None:
        blob = {
            "kind": "LinearPotential",
            "note": self.note,
            "days": {
                str(d): {"intercept": self.intercepts.get(d, 0.0),
                         "coefs": dict(self.coefs[d]),
                         "clip": list(self.clip[d]) if d in self.clip else None}
                for d in sorted(self.coefs)
            },
        }
        with open(path, "w") as fh:
            json.dump(blob, fh, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "LinearPotential":
        with open(path) as fh:
            blob = json.load(fh)
        if blob.get("kind") != "LinearPotential":
            raise ValueError(f"{path} is not a LinearPotential ({blob.get('kind')!r})")
        coefs, inter, clip = {}, {}, {}
        for k, row in blob["days"].items():
            d = int(k)
            coefs[d] = dict(row["coefs"])
            inter[d] = float(row["intercept"])
            if row.get("clip"):
                clip[d] = (float(row["clip"][0]), float(row["clip"][1]))
        return cls(coefs=coefs, intercepts=inter, clip=clip,
                   note=blob.get("note", ""))


@dataclasses.dataclass(frozen=True)
class LeadPotential(Potential):
    """The zero-fit potential: declared from measurement, not regressed.

    Every gain this project has shipped came from arithmetic on a rule the
    README or the corpus already states, and every fitted model has failed
    the gate. So the fitted `LinearPotential` gets an unfitted control, built
    from three published numbers and nothing else:

      before `crossover`  invested_lead carries the signal
                          (`docs/metric_propagation.md`: r +0.366 at day 2,
                          +0.211 at day 8, flipping to -0.032 by day 10)
      after  `crossover`  money_lead dominates and nothing is close
                          (r 0.552 at day 12 -> 0.892 at day 28)

    The ramp between them is linear in the day, which is an assertion about
    shape and the one thing here that is not measured. It is also the only
    free parameter, which is the point.

    The scales are not fitted either: the two lead features are recorded as
    `tanh(lead / s)` (`tools/early_metrics.py:124,126`), so multiplying by the
    same `s` inverts the squash to first order and puts Phi back in DOLLARS,
    which is the unit the terminal reward is in.
    """

    crossover: float = 11.0
    ramp_days: float = 4.0
    invest_scale: float = 2000.0    # early_metrics: tanh(invested_lead / 2000)
    money_scale: float = 5000.0     # early_metrics: tanh(money_lead / 5000)

    def _mix(self, day: int) -> float:
        """0 -> pure investment potential, 1 -> pure money potential."""
        if self.ramp_days <= 0:
            return 1.0 if day >= self.crossover else 0.0
        t = (day - (self.crossover - 0.5 * self.ramp_days)) / self.ramp_days
        return min(1.0, max(0.0, t))

    def __call__(self, feats: Mapping[str, float], day: int) -> float:
        w = self._mix(day)
        invested = float(feats.get("E:invested_lead", 0.0))
        money = float(feats.get("E:money_lead", 0.0))
        return ((1.0 - w) * self.invest_scale * invested
                + w * self.money_scale * money)


def load_potential(path: Optional[str]) -> Potential:
    """`None` -> ZeroPotential; `lead[:crossover]` -> LeadPotential; else JSON."""
    if not path:
        return ZeroPotential()
    if path.startswith("lead"):
        _, _, arg = path.partition(":")
        return LeadPotential(crossover=float(arg)) if arg else LeadPotential()
    return LinearPotential.from_json(path)
