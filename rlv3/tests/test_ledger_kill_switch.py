"""The ledger's kill switch must be ON THE RECORD, and no shorter than the
lever it judges.

    cd <repo> && PYTHONPATH=. uv run pytest rlv3/tests/test_ledger_kill_switch.py -q

NO GAMES ARE PLAYED HERE.

WHAT THE ORIGINAL TESTS MISSED, and why none of them could have caught it.

  * `test_seam_identity.py::test_pl_wheat_adj_switches_off_after_two_days_below_target`
    and its three siblings assert that the switch FIRES. They assert nothing
    about the switch being VISIBLE, because `correct()` had nowhere to record
    it -- no counter, no stat key, no field on the episode result. A lever
    killed on day 2 of 30 means 28 days of the shipped executor, which is the
    DEAD-not-null failure rlv3/README.md "Liveness, which is not optional"
    exists to stop; the contract was written for the plan provider only and
    nothing carried it down to the levers.

  * Those same tests drive the ledger through `roll_days()`, which hand-sets
    `led.rates` on synthetic days with no board behind them. With no wheat
    ever planted and no `day` on the rate row, the kill window's length was
    unobservable: the test could not tell "two days below target" from "two
    days before a wheat tile could possibly have been harvested". Only
    running it on a real world showed `pl_wheat_adj` dying on day 2 and
    `co_chain_tile` on day 3, every game.

  * `smoke.py:127` checks `if not out["exec_stats"]` and
    `scorecard_hook.py:349` prints the same dict. On the traced world that
    dict was `{'pl_wheat_adj.displaced': 143, 'pl_wheat_adj.moved': 29}` --
    all of it earned on days 0 and 1, before the kill. A non-empty stats dict
    is not evidence the lever is still live, so every liveness check reported
    the arm healthy.

  * `test_placement.py::test_ledger_kill_switch_removes_the_flag_from_active_flags`
    asserts a killed lever goes inert downstream. It treats the silence as
    the correct behaviour.

GROUND TRUTH for the horizon half: README "Object Types" gives wheat a Time
to First Yield of 2 days, and the engine agrees
(`CROPS["WHEAT"]["first_yield_day"] == 2`). `pl_wheat_adj` exists to make
FEED come from wheat in a hand, so its metric cannot move until a wheat tile
it seated has been harvested -- no earlier than `plant day + 2`.
"""

from __future__ import annotations

import collections

import pytest

from kagent import scheduler
from kagent.gamestate import View
from rlv3 import arm, executor, ledger
from rlv3 import agent as AGENT
from rlv3.tests.conftest import blank_obs


@pytest.fixture(autouse=True)
def clean():
    scheduler.reset_exec()
    executor.register(scheduler)
    executor.reset_stats()
    ledger.LEDGER.reset_state()
    yield
    executor.uninstall()
    arm.unregister()
    scheduler.reset_exec()
    executor.reset_stats()
    ledger.LEDGER.reset_state()


def roll(led, rates, prev=None, day=None, sink=None):
    """One dawn with hand-set realised rates, as `roll_days` does, plus the
    `day` the rates were MEASURED on -- which is what the horizon reads.

    `sink` is passed ONLY by the recording tests, so the horizon tests below
    fail on their own assertion rather than on `correct()`'s signature.
    """
    led.prev_rates = dict(prev or {})
    led.rates = dict(rates)
    if day is not None:
        led.day = day
    led._corr_day, led._corr = -2, None
    if sink is None:
        return led.correct(executor.PARAMS)
    return led.correct(executor.PARAMS, stats=sink)


# ------------------------------------------------------- the kill is recorded
def test_a_kill_bumps_a_counter_the_liveness_checks_already_print():
    """`<flag>.killed_day<d>` has to land in the SAME dict smoke.py and
    scorecard_hook.py already read, or the fix is a field nobody looks at."""
    led = ledger.ExecLedger()
    sink = collections.Counter()
    roll(led, {"chores_per_visit": 1.5}, day=1, sink=sink)
    roll(led, {"chores_per_visit": 1.5}, prev={"chores_per_visit": 1.5}, day=2,
         sink=sink)
    assert "co_chain_tile" not in led.off, "one flat day must not be enough"
    roll(led, {"chores_per_visit": 1.4}, prev={"chores_per_visit": 1.5}, day=3,
         sink=sink)
    assert "co_chain_tile" in led.off
    assert sink["co_chain_tile.killed_day3"] == 1


def test_the_kill_records_the_day_and_records_it_once():
    """`killed[flag]` is the day the lever was switched off, which IS the
    number of days it was live (killed on day 2 -> live on days 0 and 1)."""
    led = ledger.ExecLedger()
    sink = collections.Counter()
    for day in (1, 2, 3, 4, 5):
        roll(led, {"next_chore_adjacent_share": 0.9 - 0.1 * day},
             prev={"next_chore_adjacent_share": 1.0 - 0.1 * day}, day=day,
             sink=sink)
    assert led.killed["pl_crop_cluster"] == 2       # radius 2 -> 1 -> 0
    # the cluster rule re-fires every day once the radius is at zero; the
    # record must not drift to the last day it fired.
    assert sum(v for k, v in sink.items() if k.startswith("pl_crop_cluster")) == 1
    assert sink["pl_crop_cluster.killed_day2"] == 1


def test_correct_takes_no_stats_sink_and_still_records_on_the_ledger():
    """Every existing caller passes one positional argument. The sink is
    optional so the ledger stays importable without the executor."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    led.day = 4
    led.rates = {"feeds_from_hand_share": 0.0, "day": 3}
    led._handfeed_low = 1
    led.correct(executor.PARAMS)
    assert led.killed == {"pl_wheat_adj": 4}


# ------------------------------------------------- the seam hands the sink in
def test_the_seam_call_site_gives_the_ledger_the_executor_stats_sink(cases):
    """A kill recorded on a ledger nobody reads is still silent. This is the
    WIRING: `rlv3_build_tasks` must pass `executor._STATS` into `correct()`."""
    _, view, plan = cases[0]
    executor.install(scheduler)
    led = ledger.LEDGER
    led.reset_state()
    led.last_step = view.step - 1          # not an episode regression
    led.day = view.day                     # so dawn_roll is a no-op today
    led.rates = {"chores_per_visit": 1.0}
    led.prev_rates = {"chores_per_visit": 1.5}
    led._chain_flat = 1                    # today is the second flat day

    executor.rlv3_build_tasks(view, plan)

    assert "co_chain_tile" in led.off
    assert executor.stats().get(f"co_chain_tile.killed_day{view.day}") == 1


# ------------------------------------------------------ the causal horizon
def test_pl_wheat_adj_is_not_killed_before_its_wheat_could_be_harvested():
    """README Object Types: wheat is 2 days to first yield. A wheat tile
    seated on day 0 cannot put a unit in a hand before day 2, so days 0 and 1
    carry no information about the lever and must not count toward the
    streak. They did, which killed the lever on day 2 of every game."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    dead = {"feeds_from_hand_share": 0.0}
    roll(led, dict(dead, day=0), day=1)
    roll(led, dict(dead, day=1), day=2)
    assert "pl_wheat_adj" not in led.off, \
        "killed inside its own causal horizon: wheat needs 2 days to yield"
    # past the horizon the switch works exactly as before
    roll(led, dict(dead, day=2), day=3)
    assert "pl_wheat_adj" not in led.off, "one bad day must not be enough"
    roll(led, dict(dead, day=3), day=4)
    assert "pl_wheat_adj" in led.off and led.killed["pl_wheat_adj"] == 4


def test_the_horizon_moves_with_the_day_the_wheat_was_actually_seated():
    """It is not a constant: the clock starts at the first wheat the lever
    planted, which on a real board is not day 0."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 9
    assert led.wheat_effect_day() == 9 + ledger.WHEAT_FIRST_YIELD_DAYS
    dead = {"feeds_from_hand_share": 0.0}
    for d in range(0, 11):
        roll(led, dict(dead, day=d), day=d + 1)
    assert "pl_wheat_adj" not in led.off
    roll(led, dict(dead, day=11), day=12)
    roll(led, dict(dead, day=12), day=13)
    assert "pl_wheat_adj" in led.off


def test_a_lever_that_never_acted_is_never_killed():
    """No wheat seated means the lever did nothing, so the share is a
    measurement of something else entirely. Refuting an unreachable
    behaviour is `kaggriculture-unreachable-by-construction`."""
    led = ledger.ExecLedger()
    dead = {"feeds_from_hand_share": 0.0}
    for d in range(0, 20):
        roll(led, dict(dead, day=d), day=d + 1)
    assert led.wheat_effect_day() is None
    assert "pl_wheat_adj" not in led.off


def test_observe_stamps_the_first_wheat_planting_and_only_the_first():
    obs = blank_obs(day=5, quadrants=("NW",), n_hands=0)
    view = View(obs)
    led = ledger.ExecLedger()
    led.observe(view, {0: ["PLANT", "STRAWBERRY"]})
    assert led.first_wheat_day is None, "only WHEAT starts the wheat clock"
    led.observe(view, {0: ["PLANT", "WHEAT"]})
    assert led.first_wheat_day == 5

    obs2 = blank_obs(day=9, quadrants=("NW",), n_hands=0)
    led.observe(View(obs2), {0: ["PLANT", "WHEAT"]})
    assert led.first_wheat_day == 5, "the clock starts once, at the FIRST wheat"


# ----------------------------------------------- the row carries the liveness
def _fake_episode(monkeypatch, kills, last_day):
    from rl import rollout

    led = ledger.LEDGER

    def fake_run_episode(job):
        AGENT.COUNTERS["make_plan"] += 30
        for flag, day in kills.items():
            led.day = day
            led._kill(flag)
        led.day = last_day
        return {"final_money": 1.0, "opp_money": 0.0, "win": 1.0}

    monkeypatch.setattr(rollout, "run_episode", fake_run_episode)
    monkeypatch.setattr(AGENT, "last_agent", lambda: None)


def test_run_episode_puts_days_live_per_lever_on_the_result(monkeypatch):
    """The gate reads rows, not stderr. A row that says `rlv3:exec` while the
    levers were live for 2 and 3 of 30 days has to say so on the row."""
    _fake_episode(monkeypatch, {"pl_wheat_adj": 6, "co_chain_tile": 3}, 29)
    # `exec` carries no flags since maia2 was turned off, so the lever
    # bookkeeping is exercised through an EXPLICIT dose instead of a preset.
    spec = {**arm.parse("exec"), "exec_flags": list(AGENT.MAIA2_EXEC)}
    job = AGENT.make_job(spec, "unused-session.json", 0)
    out = arm.run_episode(job)
    r = out["extra"]["rlv3"]
    assert r["days"] == 30
    assert r["killed_day"] == {"pl_wheat_adj": 6, "co_chain_tile": 3}
    assert r["days_live"] == {"pl_wheat_adj": 6, "co_chain_tile": 3}
    assert r["levers_off"] == ["co_chain_tile", "pl_wheat_adj"]


def test_a_lever_that_survived_the_season_reports_the_whole_season(monkeypatch):
    _fake_episode(monkeypatch, {"co_chain_tile": 3}, 29)
    spec = {**arm.parse("exec"), "exec_flags": list(AGENT.MAIA2_EXEC)}
    out = arm.run_episode(AGENT.make_job(spec, "unused-session.json", 0))
    assert out["extra"]["rlv3"]["days_live"] == {"pl_wheat_adj": 30,
                                                 "co_chain_tile": 3}


def test_a_flagless_arm_reports_no_lever_row_at_all(monkeypatch):
    """`rlv3:base` never installs the seam, so LEDGER still holds the PREVIOUS
    episode's kills in a reused worker. A stale row is worse than none."""
    _fake_episode(monkeypatch, {"pl_wheat_adj": 2}, 29)
    spec = arm.parse("base")
    out = arm.run_episode(AGENT.make_job(spec, "unused-session.json", 0))
    r = out["extra"]["rlv3"]
    assert r["make_plan"] == 30
    assert "days_live" not in r and "levers_off" not in r
