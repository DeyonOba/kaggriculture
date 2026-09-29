"""A kill switch may not outlive its own evidence.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_ledger_lever_lifetime.py -q

NO GAMES ARE PLAYED HERE. The season this file replays was MEASURED, once,
on the world `tools.agent_scorecard.pick_games(1)` returns (preset `exec`,
seat 0) and is pasted in as `RECORDED_SEASON`.

THE DEFECT. `rlv3.ledger.correct()` is a within-episode kill switch and
`self.off` was only ever ADDED to. On that world it switched `co_chain_tile`
off at dawn on day 3 and `pl_wheat_adj` / `pl_crop_cluster` at dawn on day 4,
and all three stayed off for the remaining 26 days:

    final EXEC stats: {'pl_wheat_adj.displaced': 143, 'pl_wheat_adj.moved': 29,
                       'co_chain_tile.killed_day3': 1,
                       'pl_wheat_adj.killed_day4': 1,
                       'pl_crop_cluster.killed_day4': 1}

So every `rlv3/arms/*.json` single-lever arm and the `exec` / `full` presets
were pricing a four-day intervention as a thirty-day season, and reading it
as `null`.

WHAT THE EXISTING TESTS MISSED, and why none of them could have caught it.

  * `test_seam_identity.py::test_pl_wheat_adj_switches_off_after_two_days_below_target`
    ended on `# and it is permanent for the episode`. Permanence was the
    ASSERTED contract, so the defect had a passing test defending it. The
    line is now the opposite assertion and points here.

  * Every ledger test in this package -- `roll_days()` there, `roll()` in
    `test_ledger_kill_switch.py` -- drives at most FIVE hand-written dawns.
    A switch that fires on day 3 of 30 and a switch that is correct are the
    same thing over five days; the bug only exists in the ratio `days_live /
    days`, and no test ever looked at more than a sixth of a season. The
    replay below is all 30 dawns.

  * The kill-day tests hand-set `led.rates` with no denominator on the row,
    so `feeds_from_hand_share == 0.0` off ONE feed and off forty feeds were
    indistinguishable -- `dawn_roll` did not even publish the denominator.

  * `smoke.py` checked `if not out["exec_stats"]`, and `_kill` writes
    `<flag>.killed_day<d>` into exactly that dict. A lever that fired ZERO
    reorders all season and was then switched off therefore made the dict
    NON-empty and passed the liveness check. `co_chain_tile` did precisely
    that: there is no `co_chain_tile.reordered` key in the run above.

GROUND TRUTH for the horizon half of the rule lives in
`test_ledger_kill_switch.py` (README "Object Types": wheat is 2 days to first
yield). This file is about the other half: what happens after the kill.
"""

from __future__ import annotations

import collections

import pytest

from kagent.gamestate import View
from rlv3 import executor, ledger
from rlv3.tests.conftest import blank_obs

LEVERS = ("pl_wheat_adj", "pl_crop_cluster", "co_chain_tile")


@pytest.fixture(autouse=True)
def clean():
    ledger.LEDGER.reset_state()
    yield
    ledger.LEDGER.reset_state()


def roll(led, rates, prev=None, day=None, sink=None):
    """One dawn with hand-set realised rates, as the sibling files do."""
    led.prev_rates = dict(prev or {})
    led.rates = dict(rates)
    if day is not None:
        led.day = day
    led._corr_day, led._corr = -2, None
    if sink is None:
        return led.correct(executor.PARAMS)
    return led.correct(executor.PARAMS, stats=sink)


# --------------------------------------------------------- the kill is lifted
def test_a_kill_is_lifted_the_day_the_metric_pays_again():
    """`correct()` already reset `_handfeed_low` to 0 on a recovering day and
    then changed nothing else, so the verdict outlived its evidence."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    sink = collections.Counter()
    dead = {"feeds_from_hand_share": 0.0, "n_feeds": 10}
    roll(led, dict(dead, day=2), day=3, sink=sink)
    roll(led, dict(dead, day=3), day=4, sink=sink)
    assert "pl_wheat_adj" in led.off and sink["pl_wheat_adj.killed_day4"] == 1

    good = {"feeds_from_hand_share": 0.30, "n_feeds": 10, "day": 4}
    roll(led, good, day=5, sink=sink)
    assert "pl_wheat_adj" not in led.off, \
        "the metric came back and the lever did not"
    assert led._handfeed_low == 0
    # a revival is a liveness event and is recorded the way a kill is
    assert sink["pl_wheat_adj.revived_day5"] == 1
    assert led.killed == {}


def test_the_streak_does_not_run_on_a_lever_that_is_already_off():
    """With the wheat no longer seated beside the animals the share measures
    the SHIPPED executor, so it can neither confirm nor extend the kill. On
    the traced world `_handfeed_low` reached 24 on a lever dead since day 4."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    dead = {"feeds_from_hand_share": 0.0, "n_feeds": 10}
    for d in range(2, 20):
        roll(led, dict(dead, day=d), day=d + 1)
    assert "pl_wheat_adj" in led.off
    assert led._handfeed_low == 2, \
        "the streak must stop at the kill, not run to the end of the season"


def test_co_chain_tile_comes_back_the_day_chores_per_visit_rises():
    led = ledger.ExecLedger()
    sink = collections.Counter()
    roll(led, {"chores_per_visit": 1.5}, prev={"chores_per_visit": 1.6}, day=1)
    roll(led, {"chores_per_visit": 1.4}, prev={"chores_per_visit": 1.5}, day=2)
    assert "co_chain_tile" in led.off
    roll(led, {"chores_per_visit": 1.9}, prev={"chores_per_visit": 1.4}, day=3,
         sink=sink)
    assert "co_chain_tile" not in led.off
    assert led._chain_flat == 0
    assert sink["co_chain_tile.revived_day3"] == 1


def test_the_cluster_radius_climbs_back_when_adjacency_recovers():
    """The ladder was one-way: two down-days anywhere in the season pinned it
    at 0, and `next_chore_adjacent_share` recovering to 0.91 by day 19 could
    not move it."""
    led = ledger.ExecLedger()
    base = executor.PARAMS["cluster_radius"]
    seq = [(0.90, 1.00), (0.80, 0.90), (0.85, 0.80), (0.95, 0.85)]
    got, off = [], []
    for day, (cur, prev) in enumerate(seq, start=1):
        got.append(roll(led, {"next_chore_adjacent_share": cur},
                        prev={"next_chore_adjacent_share": prev}, day=day))
        off.append("pl_crop_cluster" in led.off)
    assert [p["cluster_radius"] for p in got] == [1.0, 0.0, 1.0, base]
    assert off == [False, True, False, False]


# ------------------------------------------------- the denominator on the row
def test_dawn_roll_states_the_denominators_the_kill_switch_needs():
    """A rate with no denominator on it cannot be told from a thin one."""
    led = ledger.ExecLedger()
    led.dawn_roll(View(blank_obs(day=0, hour=0, quadrants=("NW",), n_hands=0)))
    led.feeds, led.feeds_from_hand = 3, 0
    led.visits, led.visit_chores = 5, 6
    rates = led.dawn_roll(View(blank_obs(day=1, hour=0, quadrants=("NW",),
                                         n_hands=0)))
    assert rates["n_feeds"] == 3
    assert rates["n_visits"] == 5


def test_a_share_off_fewer_than_four_feeds_is_not_evidence():
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    thin = {"feeds_from_hand_share": 0.0, "n_feeds": 1}
    roll(led, dict(thin, day=2), day=3)
    roll(led, dict(thin, day=3), day=4)
    assert "pl_wheat_adj" not in led.off
    assert led._handfeed_low == 0
    # the same two days with a real denominator DO kill it
    fat = {"feeds_from_hand_share": 0.0, "n_feeds": ledger.MIN_FEEDS}
    roll(led, dict(fat, day=4), day=5)
    roll(led, dict(fat, day=5), day=6)
    assert "pl_wheat_adj" in led.off and led.killed["pl_wheat_adj"] == 6


# ------------------------------------------------------------- the whole season
# The realised rates of ONE measured 30-day episode: preset `exec`, seat 0,
# `pick_games(1)`'s world, read off `LEDGER.rates` at every dawn. Rows are
# (measured day, feeds_from_hand_share, next_chore_adjacent_share,
# chores_per_visit). They were recorded with the levers DEAD from day 4, so
# they are a LOWER bound on what a live lever sees -- the fix cannot be
# flattered by them.
RECORDED_SEASON = (
    (0, 0.000, 0.972, 1.818), (1, 0.000, 0.758, 1.357), (2, 0.000, 0.767, 1.207),
    (3, 0.000, 0.741, 1.179), (4, 0.200, 0.765, 1.444), (5, 0.000, 0.706, 1.379),
    (6, 0.000, 0.660, 1.447), (7, 0.000, 0.525, 1.111), (8, 0.000, 0.822, 1.456),
    (9, 0.000, 0.566, 1.185), (10, 0.000, 0.788, 1.520), (11, 0.000, 0.861, 1.693),
    (12, 0.000, 0.796, 1.615), (13, 0.000, 0.683, 1.119), (14, 0.000, 0.704, 1.258),
    (15, 0.071, 0.847, 1.732), (16, 0.000, 0.650, 1.150), (17, 0.000, 0.741, 1.183),
    (18, 0.000, 0.795, 1.173), (19, 0.143, 0.906, 1.662), (20, 0.000, 0.848, 1.400),
    (21, 0.000, 0.837, 1.398), (22, 0.000, 0.883, 1.359), (23, 0.083, 0.853, 1.606),
    (24, 0.000, 0.837, 1.634), (25, 0.000, 0.838, 1.519), (26, 0.000, 0.862, 1.458),
    (27, 0.000, 0.899, 1.500), (28, 0.000, 0.656, 1.385),
)
SEASON_DAYS = 30


def replay_season(led, rows=RECORDED_SEASON, first_wheat_day=0, sink=None):
    """Drive the ledger through a whole season the way the seam does.

    At the dawn of day D, `rates` is YESTERDAY's row -- `dawn_roll` stamps
    `day: self.day` before advancing -- and `correct()` runs once. Returns
    `{flag: days the flag was live}`.
    """
    by_day = {d: r for d, *r in rows}
    led.first_wheat_day = first_wheat_day
    led.rates = led.prev_rates = {}
    live = dict.fromkeys(LEVERS, 0)
    for day in range(SEASON_DAYS):
        row = by_day.get(day - 1)
        led.prev_rates = led.rates
        led.rates = {} if row is None else {
            "feeds_from_hand_share": row[0],
            "next_chore_adjacent_share": row[1],
            "chores_per_visit": row[2],
            "n_feeds": 10, "n_visits": 40, "day": day - 1,
        }
        led.day = day
        led._corr_day, led._corr = -2, None
        led.correct(executor.PARAMS, stats=sink)
        for f in LEVERS:
            live[f] += f not in led.off
    return live


def test_the_recorded_season_no_longer_kills_every_lever_by_day_four():
    """THE regression. Pre-fix this exact table gave days_live of 4 / 4 / 3
    out of 30 -- `co_chain_tile.killed_day3`, `pl_wheat_adj.killed_day4`,
    `pl_crop_cluster.killed_day4`, matching the measured episode exactly."""
    led = ledger.ExecLedger()
    sink = collections.Counter()
    live = replay_season(led, sink=sink)

    assert live["co_chain_tile"] >= 20, live
    assert live["pl_crop_cluster"] >= 20, live
    # pl_wheat_adj's own metric only clears the 0.15 target once all season,
    # so it does NOT survive -- but the switch is two-way, so the day it pays
    # it comes back, and the arm no longer reports a four-day dose as thirty.
    assert live["pl_wheat_adj"] > 4, live
    assert sink["pl_wheat_adj.revived_day5"] == 1, dict(sink)
    assert all(v < SEASON_DAYS for v in live.values()), \
        "the switch must still be able to fire; this is not a disable"
    # and the ledger's own count is the same number, because `days_live` is
    # what `rlv3.arm.ledger_report` puts on every gate row
    assert led.days_live == live, (led.days_live, live)


def test_days_live_is_counted_not_inferred_from_the_kill_day():
    """`ledger_report` used to read `killed[flag]` as "days live", which is
    only true while `off` is one-way. With revivals a lever killed on day 4,
    revived on day 5 and killed again on day 13 played 8 days, not 13."""
    led = ledger.ExecLedger()
    live = replay_season(led)
    assert led.days_live["pl_wheat_adj"] == live["pl_wheat_adj"]
    assert led.killed["pl_wheat_adj"] > led.days_live["pl_wheat_adj"], \
        "the kill day overstates the dose once the switch is two-way"


def test_the_season_replay_reproduces_the_measured_kill_days_before_recovery():
    """The table is the real world, not a convenient one: the FIRST kill of
    each lever lands on exactly the day the traced episode recorded."""
    led = ledger.ExecLedger()
    sink = collections.Counter()
    replay_season(led, sink=sink)
    first_kill = {}
    for key in sink:
        flag, _, day = key.partition(".killed_day")
        if day:
            first_kill[flag] = min(first_kill.get(flag, 99), int(day))
    assert first_kill["co_chain_tile"] == 3
    assert first_kill["pl_wheat_adj"] == 4
    assert first_kill["pl_crop_cluster"] == 7   # was 4 one-way; the ladder
    #                                             now climbs back on days 2
    #                                             and 4 before it bottoms out


# ------------------------------------------- the liveness check can see it now
def test_the_liveness_check_does_not_count_the_ledgers_own_bookkeeping():
    """`smoke.py` failed only on an EMPTY `exec_stats`, and `_kill` writes
    into that same dict -- so a lever that did nothing but die looked alive."""
    from rlv3 import smoke

    stats = {"pl_wheat_adj.displaced": 143, "pl_wheat_adj.moved": 29,
             "co_chain_tile.killed_day3": 1}
    assert smoke.lever_counters(stats, "co_chain_tile") == {}, \
        "a kill record is the LEDGER writing, not the lever acting"
    assert smoke.lever_counters(stats, "pl_wheat_adj") == \
        {"pl_wheat_adj.displaced": 143, "pl_wheat_adj.moved": 29}

    row = {"days": 30, "days_live": {"pl_wheat_adj": 4, "co_chain_tile": 3}}
    bad = smoke.lever_liveness(stats, row, ("pl_wheat_adj", "co_chain_tile"))
    assert len(bad) == 2, bad
    assert any("fired NOTHING" in b for b in bad)
    assert any("4 of 30" in b for b in bad)

    ok = smoke.lever_liveness(
        {"pl_wheat_adj.moved": 29, "co_chain_tile.reordered": 12},
        {"days": 30, "days_live": {"pl_wheat_adj": 24, "co_chain_tile": 30}},
        ("pl_wheat_adj", "co_chain_tile"))
    assert ok == []
