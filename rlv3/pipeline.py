"""The leaders' late one-time-crop pipeline, as a pure addition on the
champion plan (rlv3 config `late_pipe`).

Measured 2026-09-29 on the fresh leaders' winner seats (40 games, replays/now)
against the champion on tape worlds (tools/endgame_scorecard.py):

    plantings/day      d23   d24   d25   d26   d27
    leaders WHEAT      8.8   7.8   8.7   6.8   2.8
    leaders CARROT     5.5   6.2   5.8   8.2   1.8
    champion WHEAT     9.7   7.0   0.0   0.0   0.0     (WHEAT_LAST_PLANT_DAY = 25, standing >= target)
    champion CARROT    0.0   0.0  10.3   6.7   0.0     (CARROT_LAST_PLANT_DAY = 26, liquidation from 27)
    FERTILIZE/day      leaders 13-18 through day 28; champion 0 from day 27 (price-gated before)

Their harvests rise into day 29 (33 / 39 / 37) where ours fall (30 / 23 / 9),
and their days-20-29 cash flow is $52.8k to our $45.8k. Two liquidation
switches alone (tend_end, late_plant) gated NULL: the tending had nothing
to tend. This adds the pipeline itself:

  * days 25-27: the wheat/carrot plantings the champion's cutoffs drop, at
    the leaders' measured rate (LEADER_PLANT), never below what the champion
    already asks and never more than the bare ground;
  * the market builder keeps buying seeds through day 27 (memory.last_plant);
  * days 23-28: plan.fert_budget = FERT_RATE, so the age-2 fertiliser windows
    of those plantings are not price-gated (the v5 fert head's mechanism, as
    a constant from the leaders' count).

Needs executor flag `late_plant` for the day-29 harvest of what day 27
planted (the shipped `ripe` rule waits for max yield).
"""
from __future__ import annotations

LEADER_PLANT = {25: {"WHEAT": 9}, 26: {"WHEAT": 7, "CARROT": 8}, 27: {"WHEAT": 3, "CARROT": 2}}
LAST_PLANT = {"WHEAT": 27, "CARROT": 27}    # first yield in 2 days: lands on day 29
FERT_RATE = 16                              # leaders' FERTILIZE/day, days 23-28
FERT_DAYS = range(23, 29)


def extend(plan, view, memory):
    """Mutate `plan` in place; returns it. Identity outside days 23-28."""
    day = view.day
    if day in FERT_DAYS:
        plan.fert_budget = FERT_RATE
    memory.last_plant = dict(LAST_PLANT)
    want = LEADER_PLANT.get(day)
    if want:
        quota = dict(plan.plant_quota or {})
        room = len(list(view.me.empty_tiles())) - sum(quota.values())
        for crop, n in want.items():
            extra = max(0, min(n, room) - quota.get(crop, 0))
            if extra > 0:
                quota[crop] = quota.get(crop, 0) + extra
                room -= extra
        plan.plant_quota = {c: n for c, n in quota.items() if n > 0}
    return plan
