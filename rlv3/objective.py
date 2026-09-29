"""The lexicographic objective: WIN -> profit-when-winning -> loss-when-losing.

WHY THIS FILE EXISTS. Ten of the eleven single-lever rules measured on
2026-09-26 lost, and every losing arm handed the opponent $2-7k. The market
is ONE shared inventory and the score is relative, so banking well while the
opponent banks more is a loss. Our learned components have optimised our own
bank; the ladder scores wins. This module is the one place the actual
objective is written down:

    1. WIN RATE   -- final money > opponent final money. Dominates.
    2. Then, when winning, maximise profit.
    3. Then, when losing, minimise the loss margin.

Lexicographic, not a weighted sum. It appears here in two forms, because the
trainer and the planner consume different things:

    (a) TERMINAL REWARD -- one scalar per episode, for PPO.
    (b) PLAN COMPARATOR -- `better(Outlook, Outlook)` over candidate day
        plans, built on a deterministic projection of BOTH banks.

WHAT IS AND IS NOT FITTED. The projection is arithmetic: the forward market
model that gated +$3,185 +/- 768 (`kagent.planner._rent_inventory` /
`_rent_board_supply`), the exact price identity (`kagent.econ`) and the exact
remaining-shop-draw convolution (`econ.forecast_drain`). The ONE fitted
object is `rlv3/calibration.json`, the empirical P(win | projected margin,
day, band) table -- and it is deliberately coarse, monotone and clamped.

THE REFUTATION THIS FILE IS AIMED AT, stated up front because it is aimed at
this file's centre. V-RANKING IS CLOSED: four value heads (two encoders x two
label schemes, plus an auxiliary) all ordered actions at CHANCE out of
sample; the search-label heads scored 0.503 and 0.500, and the fitted head
was 4.8x tighter on held-out error while getting the SIGN of the effect wrong
on 3 of 4 arms. `better()` and `p_win()` ARE a value-ranking device over
candidate plans. Behind that sits the general law: BC at 69.4% accuracy, the
market demand table and the value head all fit the data and all failed to
predict the effect of a change.

Three consequences, all load-bearing:

  * `calibration.json` must NEVER be judged by its Brier score or its MAE.
    Those are exactly the numbers that have lied before. It is judged by
    RETRODICTION on arms already paid for (`rlv3/objective_check.py`), and
    then by a >= 995-game paired `rl/session_eval.py`.
  * The comparator consumes STATE VALUES, never returns. Potential-based
    shaping is provably a no-op for every episode-return search we gate on
    (ladder_gate, the bandit, CEM, ES -- `tools.shaping.telescope_sum`), so
    expressing the comparator as a potential would make it worth exactly
    nothing where we actually measure.
  * Shaping defaults to OFF and is capped. Stage B4 put a provably
    policy-invariant phase potential into the trainer and HALVED held-out
    wins (0.037 -> 0.018, z -3.67, p 0.0002 on 785 paired games, money
    -$22k), because invariance is asymptotic and at 20 iterations the
    per-step increments of a bounded-15 potential dwarf a +/-15 terminal
    that arrives 1-5 times in 64 episodes.

CONSTANTS ARE MIRRORED BY VALUE from `rl/train.py`, not imported: `rl.train`
imports torch at module scope (line 26) and this module must stay importable
in a bare process. `tests/test_objective.py` asserts they still agree.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_objective.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from kagent import econ
from kagent import planner as kplanner
from kagent.constants import (
    ANIMALS,
    BUYABLE_PRODUCTS,
    CROPS,
    SHED_CAPACITY,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    PRICE_FLOOR,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
)
from tools import shaping as _shaping

__all__ = [
    "WIN_COEF", "MARGIN_CAP", "LOSS_SCALE", "WIN_SCALE",
    "SHAPING", "SHAPING_ANNEAL_ITERS", "DEAD_BAND",
    "margin_term", "win_of", "terminal_reward", "episode_rewards",
    "shaping_terms", "fingerprint",
    "Outlook", "project", "opponent_bank", "p_win", "score", "better", "rank",
    "band_of", "ProjState", "CALIBRATION", "load_calibration",
]

# ---------------------------------------------------------------- constants
# Mirrored BY VALUE from rl/train.py. tests/test_objective.py re-reads that
# file's source and asserts the numbers still match, so a change there is a
# test failure here rather than a silent divergence.
WIN_COEF = 15.0          # == rl.train.SESSION_WIN_COEF
MARGIN_CAP = 2.0         # == rl.train.MARGIN_TIE
LOSS_SCALE = 8_000.0     # dollars at 76% tanh saturation, LOSING branch
WIN_SCALE = 25_000.0     # dollars at 76% saturation, WINNING branch (flatter)

# Shaping is OFF. Stage B4 is the reason; `shaping_terms` refuses any config
# outside {mode "none"} or {mode "margin", anneal_iters <= 3}.
SHAPING = "none"
SHAPING_ANNEAL_ITERS = 0
_MAX_ANNEAL_ITERS = 3

# The comparator's p_win dead band. A 0.001 difference in a projected win
# probability is not a decision; the projection is not that sharp.
DEAD_BAND = 0.01

# Longest forward projection, in days. The season is 30 days, so 29 covers
# any dawn. `kagent.planner._RENT_HORIZON` is 14 and is what the SHIPPED rent
# model uses; the comparator projects to the season end because clause 1 is
# about the FINAL bank, not a 14-day rent.
MAX_HORIZON = 29
SEASON_END_DAY = 29

# Include our own shed on our side of the projection.
#
# THE SPEC ASKED FOR TRUE AND THE MEASUREMENT SAYS FALSE, so it is a flag and
# the default is the measurement. Their shed is never guessed -- an oracle on
# it is worth 1.5% of forecast error, and rl.phase_potential.board_assets
# excludes both sheds for exactly that reason -- but reading OURS and not
# theirs makes the projection asymmetric, and an asymmetric projection is a
# constant bias on the margin, which is a bias on its SIGN. Measured on 80
# seat-games of the leaders' replays (objective_check.py --check sign), sign
# accuracy of the projected final margin:
#
#     day            6      12      16      20      28
#     shed ON    0.613   0.713   0.637   0.650   0.725
#     shed OFF   0.600   0.625   0.775   0.725   0.900
#
# Both sheds out is also what makes the two sides one function of the board,
# which is the half the price-forecast ablation showed was worth 36%.
INCLUDE_OUR_SHED = False

# Sign convention for the opponent-supply solve in ProjState.observe.
# "engine" is the identity the engine actually implements (selling RAISES
# market inventory -- econ.sell_marginal_prices increments `inv` per unit
# sold, econ.buy_cost decrements it), so
#       d_inv = our_sold + opp_sold - drain - our_bought - opp_bought
#   =>  opp_net = d_inv - our_net + drain
# The component spec writes the negated form; "spec" reproduces it so the
# two can be measured against a replay rather than argued about.
# `rlv3/objective_check.py --check identity` is that measurement.
INV_SIGN = "engine"

# Days of FEED held back before wheat is sold, mirroring the shipped
# planner's `DayPlan.feed_days` default. Without a reserve the projection
# sells the harvest on the day it lands and buys the feed back from the
# market on the next line, which is not what a farm does.
FEED_RESERVE_DAYS = 2

# Rating bins, mirrored from rl/session_eval.py:38 and rl/train.py:EDGES.
EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]


def band_of(rating: Optional[float]) -> Optional[int]:
    """Rating -> band index, the exact binning rl/session_eval.py draws on."""
    if rating is None:
        return None
    for i, (lo, hi) in enumerate(EDGES):
        if lo <= rating < hi:
            return i
    return len(EDGES) - 1


# =====================================================================
# (a) TERMINAL REWARD
# =====================================================================

def margin_term(m: float) -> float:
    """Bounded, asymmetric, strictly increasing margin term, in reward units.

        MARGIN_CAP * tanh(m / LOSS_SCALE)   for m < 0
        MARGIN_CAP * tanh(m / WIN_SCALE)    for m >= 0

    Continuous at 0 (both branches are 0 there), strictly increasing on all
    of R, bounded by +/- MARGIN_CAP -- so no blowout can outvote a win.

    THE ASYMMETRY IS CLAUSE 3 AGAINST CLAUSE 2. dR/dm at 0- is
    WIN_SCALE / LOSS_SCALE = 3.1x dR/dm at 0+, i.e. closing a $2,400 loss is
    worth about three times growing an equal lead. The scales come from
    measured spread, not taste: the median |margin| over 21,054 corpus games
    is $4,413 (docs/reward_design.md) and champ-h10's eval margin is -$2,350,
    so the decision-relevant band is +/- $5k, which LOSS_SCALE = 8k keeps on
    the steep part of tanh.

    STRICT MONOTONICITY IS A SAFETY PROPERTY, not a nicety: one algebra slip
    between the branches and there is a region where a LARGER loss scores
    higher, i.e. the objective pays to lose. tests/test_objective.py sweeps
    it densely across the join.
    """
    m = float(m)
    if m < 0.0:
        return MARGIN_CAP * math.tanh(m / LOSS_SCALE)
    return MARGIN_CAP * math.tanh(m / WIN_SCALE)


def win_of(final_money: float, opp_money: float) -> float:
    """1.0 / 0.5 / 0.0 -- the exact convention of rl.rollout.run_episode's
    "win" key (rl/rollout.py:580)."""
    if final_money > opp_money:
        return 1.0
    return 0.5 if final_money == opp_money else 0.0


def terminal_reward(final_money: float, opp_money: float, *,
                    base_final: Optional[float] = None,
                    base_opp: Optional[float] = None,
                    base_win: Optional[float] = None,
                    win_coef: float = WIN_COEF) -> float:
    """The whole objective, as one scalar.

    Unpaired:   win_coef * (2*win - 1)          + margin_term(margin)
    Paired:     win_coef * (win - base_win)     + margin_term(margin)
                                                - margin_term(base_margin)

    LEXICOGRAPHY IS EXACT, not approximate. In the unpaired form every win
    scores strictly above +win_coef and every loss strictly below -win_coef,
    because a win means margin > 0 means margin_term > 0 (and conversely).
    No margin can buy a win, and no loss can be dressed up as one.

    MONEY ENTERS NOWHERE. An absolute level measures the world, not the
    player: both seats share the seed, the shop draw and one price ladder, so
    differencing within the game is worth 2-6x (docs/reward_design.md Rule 1).

    The paired form is the one that matters for training: the zero-offset
    champion plays the SAME world (seed, seat, opponent), and rewarding the
    difference removes the +/-$30k shared-world variance that drowned the
    per-decision advantage. It is shaped to drop straight into the
    REWARD == "win" branch of rl/train.py:419-436.
    """
    win = win_of(final_money, opp_money)
    margin = float(final_money) - float(opp_money)
    if base_final is None:
        return win_coef * (2.0 * win - 1.0) + margin_term(margin)
    b_opp = base_final if base_opp is None else base_opp
    b_win = win_of(base_final, b_opp) if base_win is None else float(base_win)
    b_margin = float(base_final) - float(b_opp)
    return (win_coef * (win - b_win)
            + margin_term(margin) - margin_term(b_margin))


def episode_rewards(ep: Mapping, gamma: float) -> List[float]:
    """Per-decision reward vector for one `rl.rollout.run_episode` result.

    Zeros everywhere except the last entry, which carries `terminal_reward`;
    plus `shaping_terms` when shaping is enabled, which by default it is not.
    This replaces the reward loop of `rl.train._episode_advantages` wholesale
    -- rlv3's own trainer calls it, because rl/ is read-only here.

    `ep` is consumed exactly as the rollout writes it: "records",
    "final_money", "opp_money", and optionally "base_final" / "base_opp" /
    "base_win" for a paired twin. `ep["win"]` is not read: it is by
    definition `win_of(final_money, opp_money)` and recomputing it keeps this
    function total on hand-built episodes.
    """
    n = len(ep.get("records") or ())
    if n == 0:
        return []
    rewards = [0.0] * n
    rewards[-1] = terminal_reward(
        ep["final_money"], ep.get("opp_money", ep["final_money"]),
        base_final=ep.get("base_final"), base_opp=ep.get("base_opp"),
        base_win=ep.get("base_win"))
    if SHAPING != "none":
        outlooks = ep.get("outlooks")
        if outlooks:
            for i, f in enumerate(shaping_terms(outlooks, gamma)):
                if i < n:
                    rewards[i] += f
    return rewards


def shaping_terms(outlooks: Sequence["Outlook"], gamma: float, *,
                  mode: str = SHAPING, anneal: float = 0.0,
                  anneal_iters: int = SHAPING_ANNEAL_ITERS) -> List[float]:
    """Potential-based shaping over projected margins. Zero by default.

    Delegates to `tools.shaping.shaping_terms`, which already enforces
    F = gamma * Phi(s') - Phi(s) and Phi(terminal) = TERMINAL_POTENTIAL = 0.
    Phi = anneal * MARGIN_CAP * tanh(Outlook.margin / WIN_SCALE), so at
    anneal = 0 (the default) the whole vector is identically zero and the
    trainer sees the sparse terminal reward alone.

    THE CAP IS THE POINT. Stage B4's potential was provably policy-invariant
    and still halved held-out wins, because invariance is asymptotic and 20
    iterations is not the asymptote. So: `anneal > 0` is refused beyond
    `_MAX_ANNEAL_ITERS = 3` iterations, and an unknown mode is refused
    outright.
    """
    if mode not in ("none", "margin"):
        raise ValueError(f"unknown shaping mode {mode!r}; use 'none' or 'margin'")
    if mode == "none" or anneal == 0.0:
        return [0.0] * len(outlooks)
    if anneal < 0.0:
        raise ValueError(f"anneal must be >= 0, got {anneal!r}")
    if anneal_iters > _MAX_ANNEAL_ITERS:
        raise ValueError(
            f"shaping anneal over {anneal_iters} iterations is refused "
            f"(max {_MAX_ANNEAL_ITERS}): Stage B4 halved held-out wins with a "
            f"provably policy-invariant potential, 0.037 -> 0.018, z -3.67")
    phis = [anneal * MARGIN_CAP * math.tanh(float(o.margin) / WIN_SCALE)
            for o in outlooks]
    return _shaping.shaping_terms(phis, gamma,
                                  terminal_potential=_shaping.TERMINAL_POTENTIAL)


def fingerprint() -> str:
    """Stable hash of the constants plus the calibration table.

    Written into every checkpoint. The user's plan is to resume training from
    checkpoints across compute-limited days, and an objective that drifts
    between resumed segments is not an objective -- a `--start-iter` resume
    that silently changed LOSS_SCALE would make every iteration before and
    after it incomparable. The trainer compares this string and refuses the
    resume on a mismatch; the check belongs here, with the constants, not in
    the trainer.
    """
    blob = json.dumps({
        "win_coef": WIN_COEF, "margin_cap": MARGIN_CAP,
        "loss_scale": LOSS_SCALE, "win_scale": WIN_SCALE,
        "shaping": SHAPING, "shaping_anneal_iters": SHAPING_ANNEAL_ITERS,
        "dead_band": DEAD_BAND, "inv_sign": INV_SIGN,
        "include_our_shed": INCLUDE_OUR_SHED,
        "calibration": _calibration_digest(),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# =====================================================================
# p_win: the calibration table
# =====================================================================
# docs/reward_design.md Rule 2, measured on 21,054 corpus games: r between
# the MONEY LEAD at day k and the final margin, with the sign-correctness of
# the lead as a predictor of the winner.
#
#   day     2      6     10     12     14     16     18     20     24     28
#   r    0.009 -0.009 -0.021 -0.017  0.049  0.190  0.404  0.575  0.777  0.928
#   sign  48.8%  50.9%  49.4%  50.7%  55.1%  55.3%  62.6%  70.0%  79.9%  88.2%
#
# THROUGH DAY 12 THE MONEY LEAD IS A COIN FLIP. That is the baseline this
# module's projection has to beat, and it is the reason the fallback table
# below is nearly flat early: a p_win that is confident on day 10 is lying.
#
# Under a bivariate normal the two rows are one number: sign accuracy is
# 1/2 + arcsin(r)/pi, which reproduces the measured row to within a point at
# every day (r 0.928 -> 87.9% vs 88.2%; 0.575 -> 69.5% vs 70.0%; 0.190 ->
# 56.1% vs 55.3%). So the fallback is that normal model, and nothing more.
_R_BY_DAY = {0: 0.0, 2: 0.009, 6: -0.009, 10: -0.021, 12: -0.017, 14: 0.049,
             16: 0.190, 18: 0.404, 20: 0.575, 24: 0.777, 28: 0.928, 29: 0.945}

# Median |margin| over the corpus is $4,413; /0.6745 puts a normal sd at
# ~$6,500. Used only to give the fallback table a scale.
MARGIN_SD = 6_500.0

DEFAULT_MARGIN_KNOTS = [-40000.0, -20000.0, -10000.0, -5000.0, -2500.0, -1000.0,
                        0.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 40000.0]
DEFAULT_DAY_KNOTS = [0, 4, 8, 12, 16, 20, 24, 28, 29]

_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "calibration.json")


def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear, CLAMPED at both ends.

    The clamp is not cosmetic. An unclamped linear potential in this repo
    answered -$850,541 against a season margin whose whole sd is $5,563,
    because one feature arrived 3,677 training sigmas out of range
    (tools.shaping.LinearPotential). A lookup that extrapolates is the same
    bug with a different shape.
    """
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            lo, hi = xs[i - 1], xs[i]
            t = 0.0 if hi == lo else (x - lo) / (hi - lo)
            return float(ys[i - 1] + t * (ys[i] - ys[i - 1]))
    return float(ys[-1])


def _r_at(day: float) -> float:
    ks = sorted(_R_BY_DAY)
    return _interp(ks, [_R_BY_DAY[k] for k in ks], float(day))


# Spread of the money lead per day, MEASURED on 240 seat-games of the
# leaders' replays (`objective_check.py --check sign`, the "lead sd $"
# column). It is nothing like smooth: the lead is worth $363 of spread at day
# 6 because everybody is broke until day 9, and then jumps 23x by day 12 as
# the first melons and wool clear. A declared ramp had day 6 at $2,893 and
# day 20 at $5,090 -- wrong by 8x in one direction and 2x in the other.
_LEAD_SD_BY_DAY = {0: 60.0, 6: 363.0, 12: 8451.0, 16: 9815.0,
                   20: 10377.0, 28: 9976.0, 29: 9976.0}


def _lead_sd(day: float) -> float:
    """Spread of the money lead at day d, in dollars. Fallback table only.

    The shipped `rlv3/calibration.json` replaces the whole table with the
    empirical one and this function is then unused; it exists so the module
    is still usable with no data file present.
    """
    ks = sorted(_LEAD_SD_BY_DAY)
    return max(50.0, _interp(ks, [_LEAD_SD_BY_DAY[k] for k in ks], float(day)))


def _analytic_row(day: float, knots: Sequence[float]) -> List[float]:
    """P(final margin > 0 | lead = m) under the bivariate normal at r(day)."""
    r = _r_at(day)
    sl = _lead_sd(day)
    denom = sl * math.sqrt(max(1e-6, 1.0 - r * r))
    return [_ncdf(r * m / denom) for m in knots]


def _default_table() -> Dict:
    rows = {str(d): {"p": _analytic_row(d, DEFAULT_MARGIN_KNOTS),
                     "n": [0] * len(DEFAULT_MARGIN_KNOTS),
                     "clamp": [DEFAULT_MARGIN_KNOTS[0], DEFAULT_MARGIN_KNOTS[-1]]}
            for d in DEFAULT_DAY_KNOTS}
    return {
        "kind": "rlv3.p_win",
        "version": 1,
        "provenance": ("FALLBACK, not measured: bivariate-normal probit at the "
                       "corpus r(day) of docs/reward_design.md Rule 2, with a "
                       "declared lead-sd schedule. Replaced by the empirical "
                       "table when rlv3/calibration.json is present."),
        "margin_knots": list(DEFAULT_MARGIN_KNOTS),
        "day_knots": list(DEFAULT_DAY_KNOTS),
        "bands": {"pooled": {"source": "analytic", "rows": rows}},
    }


def load_calibration(path: str = _CALIBRATION_PATH) -> Dict:
    """The table, or the analytic fallback when the file is absent/unreadable.

    Never raises on a missing file: a comparator that cannot be imported
    without a data file is a comparator nothing can unit-test.
    """
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return _default_table()
    if blob.get("kind") != "rlv3.p_win":
        return _default_table()
    return blob


CALIBRATION: Dict = load_calibration()


def _calibration_digest() -> str:
    blob = json.dumps(CALIBRATION, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _band_rows(table: Mapping, band: Optional[int]) -> Mapping:
    bands = table.get("bands") or {}
    if band is not None and str(band) in bands:
        return bands[str(band)]["rows"]
    return bands["pooled"]["rows"]


def p_win(margin: float, day: int, band: Optional[int] = None, *,
          table: Mapping = None) -> float:
    """Empirical P(final margin > 0 | projected margin at this day).

    Piecewise-linear in margin, linear between day knots, CLAMPED to the
    range the table actually saw in both arguments.

    TABULATED PER BAND, NEVER POOLED, because the 2500+ band is the hardest
    to predict -- R^2 -0.019 against +0.239 below 1000 -- and a pooled number
    would hide failure exactly where the win rate is worst (band 3 at 0.200).
    A band with no measured row falls back to the pooled row; which is which
    is recorded in the table's own `source` field, so a caller can tell a
    measurement from an interpolation.

    THIS IS THE FITTED PART OF THE OBJECTIVE AND THE ONE WITH A PRIOR
    REFUTATION AIMED AT IT. Do not read its Brier score as evidence: the
    value head was 4.8x tighter on held-out error than the alternative and
    still got the sign of 3 of 4 interventions wrong. It is judged by
    retrodiction, then by a paired 995-game eval.
    """
    tab = CALIBRATION if table is None else table
    rows = _band_rows(tab, band)
    knots = tab["margin_knots"]
    days = sorted(int(k) for k in rows)
    d = min(max(float(day), days[0]), days[-1])
    lo = max([k for k in days if k <= d])
    hi = min([k for k in days if k >= d])
    t = 0.0 if hi == lo else (d - lo) / (hi - lo)
    out = []
    for k in (lo, hi):
        row = rows[str(k)]
        clo, chi = row.get("clamp", [knots[0], knots[-1]])
        out.append(_interp(knots, row["p"], min(max(float(margin), clo), chi)))
    return min(1.0, max(0.0, out[0] + t * (out[1] - out[0])))


# =====================================================================
# (b) PER-DAY PLAN COMPARATOR
# =====================================================================

@dataclasses.dataclass(frozen=True)
class Outlook:
    """A deterministic forward projection of BOTH banks to the season end.

    `margin` and `p_win` are derived if not supplied, so an Outlook is always
    internally consistent: there is no way to hand the comparator a margin
    that disagrees with the two banks it came from.
    """
    day: int
    our_bank: float
    opp_bank: float
    margin: float = None          # derived: our_bank - opp_bank
    p_win: float = None           # derived: p_win(margin, day, band)
    our_parts: Mapping[str, float] = dataclasses.field(default_factory=dict)
    opp_parts: Mapping[str, float] = dataclasses.field(default_factory=dict)
    residual: Mapping[str, float] = dataclasses.field(default_factory=dict)
    band: Optional[int] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.margin is None:
            object.__setattr__(self, "margin",
                               float(self.our_bank) - float(self.opp_bank))
        if self.p_win is None:
            object.__setattr__(self, "p_win",
                               p_win(self.margin, self.day, self.band))

    @property
    def projects_win(self) -> bool:
        return self.margin > 0.0


def _supply_slot(n: int) -> List[Dict[str, float]]:
    return [{} for _ in range(n + 1)]


def _merge_supply(a: Sequence[Mapping], b: Sequence[Mapping]) -> List[Dict[str, float]]:
    out = []
    for k in range(max(len(a), len(b))):
        row = dict(a[k]) if k < len(a) else {}
        if k < len(b):
            for item, n in b[k].items():
                row[item] = row.get(item, 0) + n
        out.append(row)
    return out


def _overlay_path(path: Mapping[str, List[float]], extra: Sequence[Mapping],
                  horizon: int) -> Dict[str, List[float]]:
    """Fold a plan's extra supply into the projected inventory path.

    The plan's own units press on the same shared inventory as everybody
    else's -- that is the whole point of pricing a block at its harvest-day
    inventory rather than at today's marginal price. Pricing at the marginal
    price is what made the first rent version pick tomato every single day.
    """
    out = {p: list(v) for p, v in path.items()}
    for item in out:
        acc = 0.0
        for k in range(1, min(horizon, len(out[item]) - 1) + 1):
            if k < len(extra):
                acc += float(extra[k].get(item, 0) or 0)
            out[item][k] += acc
    return out


def _plan_supply(plan, view, horizon: int) -> Tuple[List[Dict[str, float]], float]:
    """The supply a DayPlan ADDS on our side, and what it costs in cash.

    Mirrors `kagent.planner._rent_board_supply`'s yield arithmetic exactly --
    same first_yield_day, same interval stepping, same per-animal FERTILIZER
    and WHEAT-feed entries -- so a planted tile and a quota tile are priced
    by identical code. Structures are free in this engine (BUILD costs a turn,
    not money: README "Farm Actions"), so only seeds, animals and land are
    charged.
    """
    out = _supply_slot(horizon)
    cost = 0.0
    for crop, n in dict(getattr(plan, "plant_quota", None) or {}).items():
        cd = CROPS.get(crop)
        n = int(n or 0)
        if not cd or n <= 0:
            continue
        cost += cd["seed"] * n
        if cd["ongoing"]:
            k, made = cd["first_yield_day"], 0
            while k <= horizon and made < cd["max_yield"]:
                out[k][crop] = out[k].get(crop, 0) + n
                made += 1
                k += max(1, cd["interval"])
        else:
            k = cd["max_yield_day"]
            if k <= horizon:
                # WATERED-ONLY TOTAL, NOT `cd["max_yield"]`. Engine
                # `_apply_unit_action` WATER adds `bonus = 2 if
                # fertilized_until_day >= day else 1` per watered day inside
                # the window [(max_yield_day+1)//2, max_yield_day], on top of
                # the single unit the seed is planted with, so `max_yield` is
                # the FERTILISED cap -- README "Object Types": "Wheat and
                # Carrot only reach their listed Max Yield of 6 and 4 with
                # fertilizer; watering alone peaks at 4 and 3" -- and nothing
                # in this plan ever buys the fertilizer. This is the identical
                # arithmetic `_rent_board_supply` runs on a freshly planted
                # tile (held 1, age 0, `rem = max_yield_day - (ws - 1)`) and
                # the identical total `rent.crop_block` returns: 4 / 3 / 6.
                # Melon is unchanged at 6 -- its window is long enough to
                # reach the cap on water alone.
                ws = (cd["max_yield_day"] + 1) // 2
                units = min(cd["max_yield"], 1 + cd["max_yield_day"] - (ws - 1))
                out[k][crop] = out[k].get(crop, 0) + n * units
    n_animals = 0
    for kind, n in dict(getattr(plan, "animal_buys", None) or {}).items():
        a = ANIMALS.get(kind)
        n = int(n or 0)
        if not a or n <= 0:
            continue
        cost += a["cost"] * n
        n_animals += n
        k = a["first_yield_day"]
        while k <= horizon:
            out[k][a["product"]] = out[k].get(a["product"], 0) + n
            k += a["interval"]
    if n_animals:
        for k in range(1, horizon + 1):
            out[k]["FERTILIZER"] = out[k].get("FERTILIZER", 0) + n_animals
            out[k]["WHEAT"] = out[k].get("WHEAT", 0) - n_animals
    owned = set(getattr(view.me, "unlocked_quadrants", ()) or ())
    idx = sum(1 for q in LAND_ORDER if q in owned)
    for _ in list(getattr(plan, "land_buys", None) or []):
        if idx < len(LAND_PRICES):
            cost += LAND_PRICES[idx]
            idx += 1
    return out, cost


def _price_side(supply: Sequence[Mapping], path: Mapping[str, List[float]],
                horizon: int, *, conv: Optional[Mapping[str, float]] = None,
                floors: Optional[Mapping[str, float]] = None
                ) -> Tuple[Dict[str, float], float]:
    """Dollars this side's standing production is worth, product by product.

    Each day's block is priced at ITS OWN harvest-day inventory with
    `econ.sell_revenue`, i.e. block-priced WITH SELF-IMPACT. That is the
    forward market model that gated +$3,185 +/- 768 and replicated on two
    blocks; the market is an exact identity and needs no learning.

    A RUNNING BALANCE, NOT A DAY AT A TIME. `_rent_board_supply` charges one
    WHEAT per animal per day for FEED, as a NEGATIVE entry. Pricing each day
    independently sells the harvest and then buys the feed back from the
    market on the next line, which is not what a farm does and cost real
    dollars in the ledger. Carrying a balance nets the feed against the
    harvest first and only charges `econ.buy_cost` for a genuine deficit --
    and WHEAT and FERTILIZER, the two products with a deficit, are exactly
    the two BUYABLE_PRODUCTS, so the deficit is priced the way the engine
    would charge it.

    A sell FLOOR holds units back rather than dropping them: they stay in the
    balance and are re-priced against the next day's inventory. Anything
    still held at the horizon is worth zero, which is the engine's own rule
    -- only SOLD product counts.
    """
    parts: Dict[str, float] = {}
    cost = 0.0
    bal: Dict[str, float] = {}
    for k in range(horizon + 1):
        units = supply[k] if k < len(supply) else {}
        for item, n in units.items():
            if item not in MARKET_PARAMS or item not in path:
                continue
            n = float(n)
            if n > 0 and conv is not None:
                n *= float(conv.get(item, 1.0))
            bal[item] = bal.get(item, 0.0) + n
        for item, n in list(bal.items()):
            series = path[item]
            inv = int(round(series[min(k, len(series) - 1)]))
            if n <= -1.0:
                if item in BUYABLE_PRODUCTS:
                    cost += float(econ.buy_cost(item, inv, int(round(-n))))
                bal[item] = 0.0
                continue
            reserve = 0.0
            for j in range(k + 1, min(horizon, k + FEED_RESERVE_DAYS) + 1):
                if j < len(supply):
                    reserve += max(0.0, -float(supply[j].get(item, 0) or 0))
            u = int(n - reserve)
            if u <= 0:
                continue
            floor = None if not floors else floors.get(item)
            if floor:
                u = min(u, econ.units_sellable_above(item, inv, int(floor)))
            if u <= 0:
                continue
            parts[item] = parts.get(item, 0.0) + float(econ.sell_revenue(item, inv, u))
            bal[item] -= u
    return parts, cost


def _shed_revenue(view, path: Mapping[str, List[float]]) -> float:
    """Our shed, priced at today's projected inventory. Ours only."""
    if not INCLUDE_OUR_SHED:
        return 0.0
    total = 0.0
    for item, n in dict(getattr(view, "shed", {}) or {}).items():
        if n <= 0 or item in ANIMALS or item not in path:
            continue
        total += float(econ.sell_revenue(item, int(round(path[item][0])), int(n)))
    return total


def _horizon(day: int, horizon_end: int) -> int:
    return max(1, min(int(horizon_end) - int(day), MAX_HORIZON))


def opponent_bank(view, *, state: Optional["ProjState"] = None,
                  horizon_end: int = SEASON_END_DAY) -> Tuple[float, Dict[str, float]]:
    """Their projected final bank, and where it comes from, product by product.

    THE CLAUSE THE OBJECTIVE DEMANDS EXPOSED. Every losing arm on 2026-09-26
    moved this number the wrong way -- geese handed them +$7,046 against our
    own -$6,182, more than our own loss -- and nobody saw it until a
    995-game eval came back. This function makes it visible at projection
    time.

    = their money + every unit their standing board will bring to market,
    priced at the inventory it will meet, scaled per product by the
    conversion rate ProjState has solved from public market flow.

    THEIR SHED IS EXCLUDED DELIBERATELY. An oracle on it is worth 1.5% of
    forecast error, and `rl.phase_potential.board_assets` excludes both sheds
    for the same reason. Their BOARD is worth 36%, and their board is fully
    visible.
    """
    day = int(view.day)
    h = _horizon(day, horizon_end)
    path = kplanner._rent_inventory(view, h)
    supply = kplanner._rent_board_supply(view.opp.tiles, day, h)
    conv = None if state is None else state.conversion
    parts, cost = _price_side(supply, path, h, conv=conv)
    return float(view.opp.money) + sum(parts.values()) - cost, parts


def project(view, memory=None, *, plan=None, horizon_end: int = SEASON_END_DAY,
            state: Optional["ProjState"] = None, band: Optional[int] = None,
            label: str = "") -> Outlook:
    """Deterministic forward projection of both banks. NO FITTED MODEL.

    The inventory PATH comes from `kagent.planner._rent_inventory` -- the
    forward market model that gated +$3,185 +/- 768, replicated -- and each
    side's standing production is priced from `_rent_board_supply` at its own
    harvest-day inventory via `econ.sell_revenue`, block-priced with
    self-impact.

    `plan` is an optional `kagent.planner.DayPlan`. Its plant_quota,
    animal_buys, land_buys and sell_floors are overlaid on OUR side -- both
    in the supply we price and in the inventory path everybody's units meet
    -- which is what makes this a COMPARATOR over candidate plans rather than
    a report on the current board.

    EVERYTHING READ IS PUBLIC AND CURRENT: view.day, both farms' money, both
    farms' tiles, market inventory and prices, unlocked shops. Our own shed
    is read for our side only; theirs is never guessed.
    """
    day = int(view.day)
    h = _horizon(day, horizon_end)
    path = kplanner._rent_inventory(view, h)
    mine = kplanner._rent_board_supply(view.me.tiles, day, h)
    theirs = kplanner._rent_board_supply(view.opp.tiles, day, h)
    plan_cost = 0.0
    if plan is not None:
        extra, plan_cost = _plan_supply(plan, view, h)
        path = _overlay_path(path, extra, h)
        mine = _merge_supply(mine, extra)
    fill = None if state is None else state.fill
    conv = None if state is None else state.conversion
    floors = None if plan is None else dict(getattr(plan, "sell_floors", None) or {})
    our_parts, our_cost = _price_side(mine, path, h, conv=fill, floors=floors)
    opp_parts, opp_cost = _price_side(theirs, path, h, conv=conv)
    our_bank = (float(view.me.money) + _shed_revenue(view, path)
                + sum(our_parts.values()) - our_cost - plan_cost)
    opp_bank = float(view.opp.money) + sum(opp_parts.values()) - opp_cost
    if band is None and state is not None:
        band = state.band
    return Outlook(day=day, our_bank=our_bank, opp_bank=opp_bank,
                   our_parts=our_parts, opp_parts=opp_parts,
                   residual=dict(state.residual) if state is not None else {},
                   band=band, label=label)


def score(o: Outlook) -> float:
    """One float, for anything that needs a scalar: a bandit arm, a value target.

    = WIN_COEF * (2 * p_win - 1) + margin_term(margin)

    ORDER-PRESERVING WITH better() ON OUTLOOKS THAT project() PRODUCED, and
    that qualifier is load-bearing. p_win is a monotone function of margin at
    a fixed day and band, so within one decision the two agree. On outlooks
    hand-built with p_win and margin pulled apart they need not: the p_win
    term moves by 2 * WIN_COEF * DEAD_BAND = 0.3 at the dead band while
    margin_term spans +/- MARGIN_CAP. Prefer better() for a decision and
    score() only where a scalar is unavoidable.
    """
    return WIN_COEF * (2.0 * float(o.p_win) - 1.0) + margin_term(o.margin)


def better(a: Outlook, b: Outlook, *, dead_band: float = DEAD_BAND) -> bool:
    """The comparator. Lexicographic: win first, then the right margin clause.

    1. `a.p_win > b.p_win + dead_band`                     -- clause 1.
    2. inside the dead band, BOTH projecting a win: higher margin
       -- clause 2, maximise profit, at a $250 dead band (dead_band * WIN_SCALE).
    3. inside the dead band, BOTH projecting a loss: higher margin, i.e. a
       SMALLER loss -- clause 3, at an $80 dead band (dead_band * LOSS_SCALE),
       3.1x tighter, because closing a loss is worth 3.1x growing a lead and
       the dead bands fall out of the objective's own two scales rather than
       being a new free number.
    4. inside the dead band, one projecting a win and one a loss: the win.

    THE DEAD BAND EXISTS BECAUSE THE PROJECTION IS NOISE-LIMITED. Through day
    12 the money lead -- the baseline this projection must beat -- is
    sign-correct 50.7% of the time. A 0.001 difference in p_win is not a
    decision, and a comparator that pretends otherwise is fitting noise.

    Not a total order: the dead band makes it intransitive at the margins, by
    design. `rank()` is its transitive completion.
    """
    dp = float(a.p_win) - float(b.p_win)
    if dp > dead_band:
        return True
    if dp < -dead_band:
        return False
    aw, bw = a.projects_win, b.projects_win
    if aw != bw:
        return aw
    tol = dead_band * (WIN_SCALE if aw else LOSS_SCALE)
    return float(a.margin) > float(b.margin) + tol


def rank(outlooks: Sequence[Outlook], *, dead_band: float = DEAD_BAND) -> List[int]:
    """Indices best-first. The transitive completion of `better()`.

    Buckets p_win at the dead band, then orders by margin inside a bucket --
    so whenever `better(a, b)` is decided by clause 1 the bucket is strictly
    higher, and whenever it is decided by clause 2/3 the buckets are equal
    and the margin decides. The two agree on any set whose pairwise p_win
    gaps are either zero or wider than the dead band; inside a single bucket
    boundary they can differ, which is what "dead band" means.
    """
    def key(i: int):
        o = outlooks[i]
        bucket = math.floor(float(o.p_win) / dead_band) if dead_band > 0 else o.p_win
        return (-bucket, -float(o.margin))
    return sorted(range(len(outlooks)), key=key)


# =====================================================================
# ProjState: the self-correcting loop
# =====================================================================

class ProjState:
    """Per-episode projection state. Lives at `memory.rlv3_proj`; kagent/Memory
    is read-only from here.

    NOT `memory.rlv3`, which this docstring used to claim and which is already
    taken: `rlv3.land.observe_dawn` owns that attribute as a plain DICT
    (`{"day", "occupied", "chore"}`) and re-creates it every dawn whenever it
    is not a dict, and `rlv3.allocation.chore_cost` reads
    `memory.rlv3["chore"]` off it. A ProjState parked there is destroyed on
    day 1 and takes the allocator down with it. `rlv3.agent.Rlv3PlanProvider`
    owns the attachment; see its module docstring for the whole map.

    WHAT IT FIXES. The opponent's supply is SOLVABLE from public market
    inventory -- the market is ONE shared pool -- and today nothing solves
    it. `kagent/gamestate.py:103-105` DECLARES `prev_market_inv`,
    `my_sells_last_turn` and `opp_sell_ema`, and nothing in kagent/, rl/ or
    tools/ ever writes them (grep). The whole shipped planner models
    opponent supply for MELON alone (`planner.py:277 _projected_melons`).
    Their supply costs us $52,652 a game through that one shared variable,
    97% of it strawberry + milk + wool, and their board is fully visible.

    THE SOLVE, every step (`observe`). Market inventory RISES when a unit is
    sold into it and falls when the town drains it or someone buys it back
    (`econ.sell_marginal_prices` increments `inv` per unit sold;
    `econ.buy_cost` decrements it), so

        d_inv = our_net + opp_net - drain
        opp_net = d_inv - our_net + drain                       (INV_SIGN "engine")

    `drain` is EXACT, not estimated: `town_drain_per_day` against the shops
    actually open, ticked on the engine's own cadence -- shops every
    TOWN_SHOP_SELL_INTERVAL turns, the town centre every
    TOWN_CENTER_SELL_INTERVAL. `our_net` is read as the REALISED holdings
    delta (shed + every unit inventory) with the standing-yield drop added
    back, never from the order we submitted: a SELL order is a REQUEST, not a
    fill, and the engine fills only as far as the shed allows. What survives
    is consumption -- FEED eats one WHEAT per animal per day, FERTILIZE
    consumes FERTILIZER -- and those are exactly the two BUYABLE_PRODUCTS;
    the residual loop below is what absorbs them.

    THE CORRECTION, at dawn. Yesterday's one-day-ahead forecast is compared
    against today's actuals on three channels -- our bank, their bank, and
    market inventory per product -- and the residual drives a bounded EWMA
    (half-life 3 days) on exactly TWO multipliers:

      (a) per product, the opponent's board-to-market CONVERSION rate: their
          board says what they CAN bring, the solved flow says what they DO.
      (b) our own offered-to-filled ratio, which is shed-limited and is why
          `units_offered` is an upper bound in the scorecard.

    NOTHING CORRECTS THE DRAIN. The drain is an exact engine identity; a
    correction there would be fitting noise onto arithmetic we already own.
    Every multiplier is clamped before it enters the EWMA, so a day-1 outlier
    cannot move day 20, and every multiplier starts at 1.0 -- with no
    residual the projection is the pure forward market model, so a bad
    correction degrades to the shipped model rather than to nonsense.
    """

    HALF_LIFE_DAYS = 3.0
    CONV_CLAMP = (0.25, 3.0)
    FILL_CLAMP = (0.25, 1.5)
    MIN_IMPLIED = 1.0      # units below which a ratio is undefined, not zero

    def __init__(self, band: Optional[int] = None):
        self.band = band
        self.conversion: Dict[str, float] = {}
        self.fill: Dict[str, float] = {}
        self.residual: Dict[str, float] = {}
        self.opp_flow: Dict[str, float] = {}        # season total, solved
        self.our_flow: Dict[str, float] = {}        # season total, solved
        # Products whose inventory channel has gone BLIND at least once.
        self.blind: set = set()
        self._day_opp: Dict[str, float] = {}
        self._day_our: Dict[str, float] = {}
        self._prev_step: Optional[int] = None
        self._prev_inv: Optional[Dict[str, float]] = None
        self._prev_snap: Optional[Dict] = None
        self._fc: Optional[Dict] = None
        self.days_seen = 0

    # -- alpha of an EWMA with a 3-day half-life ---------------------------
    @property
    def alpha(self) -> float:
        return 1.0 - 0.5 ** (1.0 / max(1e-6, self.HALF_LIFE_DAYS))

    # -- observables -------------------------------------------------------
    @staticmethod
    def _holdings(view) -> Dict[str, float]:
        """Every unit we hold: shed plus every unit inventory. Animals are
        not market products and are dropped."""
        out: Dict[str, float] = {}
        for item, n in dict(getattr(view, "shed", {}) or {}).items():
            if item in ANIMALS:
                continue
            out[item] = out.get(item, 0.0) + float(n)
        for idx, _, _ in view.me.units:
            for item, n in (view.unit_inv(idx) or {}).items():
                if item in ANIMALS:
                    continue
                out[item] = out.get(item, 0.0) + float(n)
        return out

    @staticmethod
    def _board(farm) -> Dict[str, Dict]:
        """One snapshot of a farm's collectable units and its tile kinds.

        `y` is {(x, y, product): units} -- collectable units keyed BY TILE,
        because summing per product first loses the event: at the day
        boundary every growing plant gains yield while a harvest takes it off
        one tile, and the aggregate drop is smaller than what entered our
        hands. `fertilizer_available` is the engine's own flag (README,
        animal structure dict): set at end of day for every surviving animal
        and cleared by COLLECT_FERTILIZER, so it is one collectable unit.

        `k` is {(x, y): kind} for every tile, which is what makes a
        DISAPPEARED tile readable. Harvesting a one-time crop sets the tile
        to None (engine HARVEST, `farm["tiles"][fy][fx] = None`), so without
        the kind map every WHEAT, CARROT and MELON harvest vanishes from the
        ledger -- measured: the solve inverted, r -0.79, when those were
        dropped.
        """
        yld: Dict[Tuple[int, int, str], float] = {}
        kind: Dict[Tuple[int, int], str] = {}
        for x, y, t in farm.iter_tiles():
            if t is None:
                kind[(x, y)] = ""
            elif not isinstance(t, dict):
                kind[(x, y)] = "LOCKED"
            elif t.get("kind") == "PLANT":
                kind[(x, y)] = "PLANT:" + str(t.get("crop"))
                yld[(x, y, t["crop"])] = float(t.get("yield_units", 0) or 0)
            elif t.get("kind") == "WEED":
                kind[(x, y)] = "WEED"
            elif t.get("animal"):
                a = ANIMALS.get(t["animal"])
                kind[(x, y)] = "ANIMAL:" + str(t["animal"])
                if a:
                    yld[(x, y, a["product"])] = float(t.get("yield_units", 0) or 0)
                if t.get("fertilizer_available"):
                    yld[(x, y, "FERTILIZER")] = 1.0
            else:
                kind[(x, y)] = "STRUCT"
        return {"y": yld, "k": kind}

    @staticmethod
    def _produced(prev: Optional[Mapping], cur: Mapping) -> Dict[str, float]:
        """Units that left the tiles into someone's hands since `prev`.

        A tile that lost yield produced it. A tile that VANISHED produced it
        only if it vanished the way a harvest makes a tile vanish -- emptied,
        or emptied and immediately replanted. A tile that became a WEED, or a
        structure whose animal escaped, LOST its units; counting those as
        production would read a dead plant as a sale.

        Two known over-counts, both small and both documented rather than
        patched: `_decay_plants` sheds one unit every two steps off an
        expired one-time crop, and DIG on a standing plant discards it. Both
        look like production here. The bounded EWMA in `dawn` is what absorbs
        them; nothing here is allowed to correct the drain, which is exact.
        """
        if not prev:
            return {}
        out: Dict[str, float] = {}
        cy, ck = cur["y"], cur["k"]
        for key, was in prev["y"].items():
            now = cy.get(key)
            if now is not None:
                if now < was:
                    out[key[2]] = out.get(key[2], 0.0) + (was - now)
                continue
            kind = ck.get((key[0], key[1]), "")
            if key[2] == "FERTILIZER":
                if kind.startswith("ANIMAL"):          # collected, not escaped
                    out[key[2]] = out.get(key[2], 0.0) + was
                continue
            crop = CROPS.get(key[2])
            if crop is not None and crop["ongoing"]:
                # An ONGOING crop's tile survives its own harvest (engine
                # HARVEST clears the tile only `if not crop_data["ongoing"]`),
                # so a vanished strawberry or tomato was DUG or died. Counting
                # it as production reads a dig as a sale, and at 20-40
                # strawberry tiles a side that is the dominant error: it cost
                # r 0.99 -> 0.5-0.8 on the top-board replays.
                continue
            if kind == "" or kind.startswith("PLANT") or kind.startswith("ANIMAL"):
                out[key[2]] = out.get(key[2], 0.0) + was
        return out

    @staticmethod
    def _chores(farm) -> Dict:
        """The two chores that CONSUME a market product, as engine flags.

        FEED eats one WHEAT per animal per day (`fed_today`, reset each end
        of day) and FERTILIZE eats one FERTILIZER per application
        (`fertilized_until_day` on the plant). Both take units out of our
        holdings WITHOUT a sale, so without them the solve reads a feed as a
        wheat sale -- measured at -270 units over 400 turns, the single
        largest error in the identity.
        """
        fed = sum(1 for _, _, t in farm.animals() if t.get("fed_today"))
        fert = {(x, y): int(t.get("fertilized_until_day", -1) or -1)
                for x, y, t in farm.plants()}
        return {"fed": fed, "fert": fert}

    @staticmethod
    def _consumed(prev: Optional[Dict], cur: Dict) -> Dict[str, float]:
        """Units of WHEAT and FERTILIZER burned by chores since `prev`."""
        if not prev:
            return {}
        out: Dict[str, float] = {}
        feeds = cur["fed"] - prev["fed"]
        if feeds > 0:                      # fed_today resets at dawn; a DROP is
            out["WHEAT"] = float(feeds)    # the reset, never a negative feed
        pf = prev["fert"]
        n = sum(1 for k, v in cur["fert"].items() if v > pf.get(k, -1))
        if n:
            out["FERTILIZER"] = out.get("FERTILIZER", 0.0) + float(n)
        return out

    @staticmethod
    def seat_snapshot(view) -> Dict:
        """Everything one seat's ledger needs, from that seat's own View.

        Used for OUR side inside `observe`, and by
        `objective_check.py --check identity` for the opponent's side, where
        the replay happens to carry their private state. One function, so the
        solve and the thing that validates the solve cannot drift apart.
        """
        return {"hold": ProjState._holdings(view),
                "board": ProjState._board(view.me),
                "chores": ProjState._chores(view.me),
                "day": int(view.day)}

    @staticmethod
    def seat_net(prev: Optional[Dict], cur: Dict) -> Dict[str, float]:
        """Units this seat pushed INTO the market (negative = bought back).

            sold - bought = produced - consumed - discarded - d(holdings)

        DISCARDED is the one that is easy to miss and is worth hundreds of
        units a game on a hoarding agent. At end of day the engine dumps
        every per-farmer inventory into the shed "up to capacity; OVERFLOW IS
        DISCARDED" (`_drop_inventories_to_shed`), so a seat holding more than
        SHED_CAPACITY loses the excess outright. Without this term the
        destroyed units read as a sale, and on the top-board replays that
        showed up as a -400 unit MILK residual and a -355 STRAWBERRY one.
        Attribution across products is pro rata, because the engine's own
        allocation depends on farmer iteration order, which the observation
        does not expose.
        """
        if not prev:
            return {}
        made = ProjState._produced(prev["board"], cur["board"])
        burned = ProjState._consumed(prev["chores"], cur["chores"])
        lost: Dict[str, float] = {}
        if cur["day"] != prev["day"]:
            total = sum(prev["hold"].values())
            over = max(0.0, total - float(SHED_CAPACITY))
            if over > 0.0 and total > 0.0:
                for item, n in prev["hold"].items():
                    if n > 0:
                        lost[item] = over * n / total
        out: Dict[str, float] = {}
        for item in set(made) | set(burned) | set(lost) | set(cur["hold"]) | set(prev["hold"]):
            d_hold = cur["hold"].get(item, 0.0) - prev["hold"].get(item, 0.0)
            out[item] = (made.get(item, 0.0) - burned.get(item, 0.0)
                         - lost.get(item, 0.0) - d_hold)
        return out

    @staticmethod
    def _blind(item: str, inv: Mapping[str, float]) -> bool:
        """Is this product's market-inventory channel carrying information?

        No, once its price is on the $1 floor: the engine still lets the sale
        happen and still pays the seller, but it does NOT increment market
        inventory. The channel is then blind and stays blind until the town
        drains the glut back off the floor.
        """
        if item not in MARKET_PARAMS:
            return False
        return econ.price(item, int(round(float(inv.get(item, 0.0))))) <= PRICE_FLOOR

    @staticmethod
    def _ticks(interval: int, s0: int, s1: int) -> int:
        """Engine ticks visible between the observations at s0 and s1.

        `_town_consume(env, state, step)` fires when `step % interval == 0`
        DURING the processing of that step, so its effect appears in the
        observation at step + 1. The ticks between two observations are
        therefore the steps in [s0, s1), not (s0, s1] -- an off-by-one that
        cost 264 units of WHEAT over one game and showed up as a clean
        alternating residual, +107 at hour 0 against -126 at hour 1.
        """
        return (s1 - 1) // interval - (s0 - 1) // interval

    @staticmethod
    def _drain_between(unlocked_shops, s0: int, s1: int) -> Dict[str, float]:
        """EXACT town consumption between two observations. No estimate."""
        out: Dict[str, float] = {}
        shop_ticks = ProjState._ticks(TOWN_SHOP_SELL_INTERVAL, s0, s1)
        cent_ticks = ProjState._ticks(TOWN_CENTER_SELL_INTERVAL, s0, s1)
        if cent_ticks:
            for item in TOWN_CENTER_PRODUCTS:
                out[item] = out.get(item, 0.0) + float(cent_ticks)
        if shop_ticks:
            for shop in unlocked_shops:
                items = SHOPS.get(shop, ())
                mult = 2 if len(items) == 1 else 1
                for item in items:
                    out[item] = out.get(item, 0.0) + float(mult * shop_ticks)
        return out

    # -- every step --------------------------------------------------------
    def observe(self, view) -> None:
        """Solve the opponent's realised net market flow since the last step."""
        step = int(getattr(view, "step", 0) or 0)
        inv = {p: float(v) for p, v in view.market_inv.items()}
        snap = self.seat_snapshot(view)
        if self._prev_step is None or step <= self._prev_step:
            self._prev_step, self._prev_inv, self._prev_snap = step, inv, snap
            return
        drain = self._drain_between(view.unlocked_shops, self._prev_step, step)
        ours = self.seat_net(self._prev_snap, snap)
        for item in set(inv) | set(self._prev_inv or {}) | set(ours):
            if self._blind(item, inv) or self._blind(item, self._prev_inv or {}):
                # THE CHANNEL IS DEAD FOR THIS PRODUCT. Engine `_commit_unit`:
                # "Sales at $1 do not increase market supply". Once a product
                # floors, units keep leaving both sheds and the market
                # inventory stops recording them, so the identity has nothing
                # to solve with -- attributing the gap to the opponent
                # invents supply that is not there. Measured on the top-board
                # replays, this is the whole remaining residual and it lands
                # exactly on the three glut products: STRAWBERRY -347,
                # MILK -392, WOOL -89 units in one game.
                self.blind.add(item)
                continue
            d_inv = inv.get(item, 0.0) - (self._prev_inv or {}).get(item, 0.0)
            our_net = ours.get(item, 0.0)
            flow = d_inv - our_net + drain.get(item, 0.0)
            if INV_SIGN != "engine":
                flow = -flow
            self._day_opp[item] = self._day_opp.get(item, 0.0) + flow
            self._day_our[item] = self._day_our.get(item, 0.0) + our_net
        self._prev_step, self._prev_inv, self._prev_snap = step, inv, snap

    # -- hour 0 ------------------------------------------------------------
    def dawn(self, view) -> Dict[str, float]:
        """Residuals, then one bounded EWMA step. Returns the residuals."""
        res: Dict[str, float] = {}
        fc = self._fc
        if fc is not None and int(view.day) > int(fc["day"]) - 1:
            res["our_bank"] = float(view.me.money) - float(fc["our_money"])
            res["opp_bank"] = float(view.opp.money) - float(fc["opp_money"])
            for item, pred in fc["inv"].items():
                res[f"inv:{item}"] = float(view.market_inv.get(item, pred)) - float(pred)
            self._update(fc)
        self.residual = res
        for item, v in self._day_opp.items():
            self.opp_flow[item] = self.opp_flow.get(item, 0.0) + v
        for item, v in self._day_our.items():
            self.our_flow[item] = self.our_flow.get(item, 0.0) + v
        self._day_opp, self._day_our = {}, {}
        self._fc = self._forecast(view)
        self.days_seen += 1
        return res

    def _forecast(self, view) -> Dict:
        """One day ahead, from the same arithmetic project() uses at k=1.

        Deliberately a ONE-DAY forecast and not the season-end projection:
        a season-end number cannot be checked against tomorrow, and a
        residual you cannot check is not a residual.
        """
        day = int(view.day)
        h = _horizon(day, SEASON_END_DAY)
        path = kplanner._rent_inventory(view, h)
        mine = kplanner._rent_board_supply(view.me.tiles, day, h)
        theirs = kplanner._rent_board_supply(view.opp.tiles, day, h)
        k = min(1, h)
        our_imp = {i: float(n) for i, n in mine[k].items() if n > 0}
        opp_imp = {i: float(n) for i, n in theirs[k].items() if n > 0}
        # the FULL supply list, truncated by the horizon argument, so the
        # feed reserve can see the days just past the forecast window
        our_rev, _ = _price_side(mine, path, k, conv=self.fill)
        opp_rev, _ = _price_side(theirs, path, k, conv=self.conversion)
        return {
            "day": day + 1,
            "our_money": float(view.me.money) + sum(our_rev.values()),
            "opp_money": float(view.opp.money) + sum(opp_rev.values()),
            "inv": {i: float(path[i][min(k, len(path[i]) - 1)]) for i in path},
            "our_implied": our_imp,
            "opp_implied": opp_imp,
        }

    def _update(self, fc: Dict) -> None:
        """The whole learned part of rlv3: two multipliers, bounded, online."""
        self._ewma(self.conversion, self._day_opp, fc["opp_implied"], self.CONV_CLAMP)
        self._ewma(self.fill, self._day_our, fc["our_implied"], self.FILL_CLAMP)

    def _ewma(self, store: Dict[str, float], realised: Mapping[str, float],
              implied: Mapping[str, float], clamp: Tuple[float, float]) -> None:
        lo, hi = clamp
        a = self.alpha
        for item, imp in implied.items():
            if imp < self.MIN_IMPLIED:
                continue                       # undefined, not zero
            ratio = min(hi, max(lo, float(realised.get(item, 0.0)) / float(imp)))
            store[item] = (1.0 - a) * store.get(item, 1.0) + a * ratio

    # -- checkpoint travel -------------------------------------------------
    def to_json(self) -> Dict:
        return {"kind": "rlv3.ProjState", "band": self.band,
                "conversion": dict(self.conversion), "fill": dict(self.fill),
                "opp_flow": dict(self.opp_flow), "our_flow": dict(self.our_flow),
                "blind": sorted(self.blind),
                "days_seen": self.days_seen,
                "objective": fingerprint()}

    @classmethod
    def from_json(cls, blob: Mapping) -> "ProjState":
        s = cls(band=blob.get("band"))
        s.conversion = dict(blob.get("conversion") or {})
        s.fill = dict(blob.get("fill") or {})
        s.opp_flow = dict(blob.get("opp_flow") or {})
        s.our_flow = dict(blob.get("our_flow") or {})
        s.blind = set(blob.get("blind") or ())
        s.days_seen = int(blob.get("days_seen") or 0)
        return s
