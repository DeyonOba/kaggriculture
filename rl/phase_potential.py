"""The phase reward: a day-dependent potential over LEADS, and why that is the
only admissible form.

THE AIM is a reward per phase of the game, judged on win rate at the end.
Under potential-based shaping (Ng, Harada & Russell 1999; Devlin & Kudenko
2012 for the time-dependent case) a potential Phi(s, day) can carry any
phase-specific signal -- investment early, cash late -- and cannot change
which policy is optimal, only how fast the critic finds it. That is the ONLY
place a phase reward is admissible here. A per-step bonus or cost on a state
variable (the old CARRY_COEF and UNSOLD_COEF terms) moves the optimum, and
the ladder has already refuted the behaviours those moved it toward
(tests/test_shaping.py H4b/H4c, docs/shaping_spec.md S3).

WHAT THE ROLLOUT SEES at each dawn (kagent.gamestate.View): both boards, both
farms' money, its own shed. So the potential is built from LEADS, not levels
-- docs/reward_design.md Rule 1: both seats share the seed, the shop draw and
one price ladder, so differencing within the game is worth 2-6x.

    asset_lead(s) = board_assets(me) - board_assets(opp)
                    standing crops (yield on the plant + growth still to come,
                    at 0.6 x today's price) and the herd (remaining production
                    over at most 8 days + 0.3 x purchase cost) -- the same
                    arithmetic rl.rollout.portfolio_value has always used, for
                    BOTH farms. Sheds are excluded on both sides: ours is
                    visible and theirs is not, and counting one would bias the
                    lead by their hidden stock. Their shed was 1.5% of forecast
                    error even as an oracle (docs/forward market model).
    money_lead(s) = money(me) - money(opp)

MIXED BY DAY, from measurement, not fitting (docs/metric_propagation.md):
investment lead correlates +0.366 with the final margin at day 2, +0.211 at
day 8, and flips to -0.032 by day 10; the money lead is worth ~nothing before
day 12 and dominates after (0.552 -> 0.892 by day 28).

    w(d)      = clamp((d - (crossover - ramp/2)) / ramp, 0, 1)    crossover 11, ramp 4
    lead(s,d) = (1 - w(d)) * asset_lead + w(d) * money_lead
    Phi(s,d)  = W * tanh(lead / S)                                W 15, S 20,000

W is the session win coefficient, so Phi lives on the terminal reward's scale
(Phi should approximate V, and V under --reward win is +/-15). S is the
margin tiebreaker's dollar scale. tanh keeps Phi bounded whatever the world
does. The linear ramp is the one assertion about SHAPE here that is not
measured; it is also the only free choice. Nothing is fitted: a potential
fitted on self-play scored -0.398 R2 out of distribution, while this zero-fit
shape lost -0.054 AUC (docs/phi_ood_and_phase_timing.md).

TERMINAL. Phi(terminal) = 0 (tools/shaping.TERMINAL_POTENTIAL), so the final
shaping term is -Phi(s_last): ending the season on a lead the engine does not
pay for is charged at the potential's own scale, with no coefficient to tune.

WHAT IT IS NOT. It is not the win reward, and it cannot become one. By the
telescoping identity its discounted sum along any trajectory is
gamma^n * 0 - Phi(s_0), a constant per start state. So a gain from this file
can only ever show up as FASTER convergence under the same terminal reward,
and it must be judged as a convergence arm, never as a source of win rate.

WIRING (after Stage B2 finishes; nothing the running queue imports is touched):
    rollout   record ingredients(view) per decision  -> asset_lead, money_lead
    train     phis = [phase_potential(r["asset_lead"], r["money_lead"], r["day"]) ...]
    flag      --potential {pv, lead}   pv = the current 1e-4 x own portfolio value

    PYTHONPATH=. uv run pytest tests/test_phase_potential.py -q
"""
from __future__ import annotations

import math
from typing import Dict

from kagent.constants import ANIMALS, CROPS

# Rough per-animal value of remaining daily production, as rl.rollout uses.
ANIMAL_DAILY = {"GOOSE": 60, "COW": 100, "SHEEP": 70}

CROSSOVER = 11.0        # day at which the mix is half asset, half money
RAMP = 4.0              # days over which it moves from one to the other
W = 15.0                # bound on |Phi|; equals SESSION_WIN_COEF
S = 20_000.0            # dollars of lead at 76% saturation; equals MARGIN_SCALE


def board_assets(farm, view) -> float:
    """Dollar value of what stands on ONE farm's tiles: crops and herd.

    Mirrors rl.rollout.portfolio_value's crop and animal terms exactly, so the
    two potentials price the same thing; only the seat is a parameter.
    """
    v = 0.0
    days_left = max(0, 28 - view.day)
    for _, _, t in farm.animals():
        v += ANIMAL_DAILY.get(t.get("animal"), 50) * min(days_left, 8)
        v += 0.3 * ANIMALS.get(t.get("animal"), {}).get("cost", 300)
    for _, _, t in farm.plants():
        crop = CROPS.get(t.get("crop"))
        if not crop:
            continue
        base = 0.6 * view.market_prices.get(t["crop"], 20)
        v += base * t.get("yield_units", 0)
        age = view.day - t.get("planted_day", view.day)
        if age <= crop["max_yield_day"]:
            v += 0.3 * base * max(0, crop["max_yield"] - t.get("yield_units", 0))
    return v


def ingredients(view) -> Dict[str, float]:
    """The two leads Phi is built from, plus the day. Recorded per decision
    so the potential's SHAPE can change without re-rolling a single game."""
    return {
        "asset_lead": board_assets(view.me, view) - board_assets(view.opp, view),
        "money_lead": float(view.me.money) - float(view.opp.money),
        "day": int(view.day),
    }


def phase_mix(day: float, crossover: float = CROSSOVER, ramp: float = RAMP) -> float:
    """0 -> the potential is all asset lead, 1 -> all money lead."""
    if ramp <= 0:
        return 1.0 if day >= crossover else 0.0
    return min(1.0, max(0.0, (day - (crossover - 0.5 * ramp)) / ramp))


def phase_potential(asset_lead: float, money_lead: float, day: float,
                    w: float = W, s: float = S,
                    crossover: float = CROSSOVER, ramp: float = RAMP) -> float:
    """Phi(s, day) in reward units, bounded in (-w, w)."""
    m = phase_mix(day, crossover, ramp)
    lead = (1.0 - m) * asset_lead + m * money_lead
    return w * math.tanh(lead / s)


def potential_of(view, **kw) -> float:
    """Phi straight from a View, for tools that have one in hand."""
    i = ingredients(view)
    return phase_potential(i["asset_lead"], i["money_lead"], i["day"], **kw)
