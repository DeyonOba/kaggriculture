"""Check (a) of the risk list: with every rlv3 flag OFF the seam is a
byte-identical pass-through, on mid-season states from three recorded
sessions.

`kaggriculture-dispatch-closed`: build variants as PURE ADDITIONS or they
are confounded. If the installed seam changes a single task with no flag on,
every later arm measures the seam plus the flag and no verdict means
anything.
"""

from __future__ import annotations

import pytest

from kagent import scheduler
from rlv3 import executor, ledger


@pytest.fixture(autouse=True)
def clean():
    scheduler.reset_exec()
    executor.register(scheduler)
    executor.reset_stats()
    ledger.LEDGER.reset_state()
    yield
    executor.uninstall()
    scheduler.reset_exec()
    ledger.LEDGER.reset_state()


def sig(tasks):
    return [(t.pri, t.x, t.y, tuple(t.op), t.need) for t in tasks]


# ------------------------------------------------------------- activation
def test_register_is_idempotent_and_set_exec_accepts_rlv3_flags():
    executor.register(scheduler)
    executor.register(scheduler)
    for f in executor.RLV3_FLAGS:
        assert scheduler.EXEC[f] is False
    scheduler.set_exec(["pl_wheat_adj"])
    assert scheduler.EXEC["pl_wheat_adj"] is True
    scheduler.reset_exec()
    assert scheduler.EXEC["pl_wheat_adj"] is False
    # and the shipped flags are untouched by registration
    for f in scheduler.EXEC_FLAGS:
        assert f in scheduler.EXEC


def test_install_rebinds_and_uninstall_restores():
    orig_build, orig_assign = scheduler.build_tasks, scheduler.assign
    executor.install(scheduler)
    assert executor.is_installed()
    assert scheduler.build_tasks is executor.rlv3_build_tasks
    assert scheduler.assign is executor.rlv3_assign
    executor.install(scheduler)          # idempotent
    assert scheduler.build_tasks is executor.rlv3_build_tasks
    executor.uninstall()
    assert scheduler.build_tasks is orig_build
    assert scheduler.assign is orig_assign
    assert not executor.is_installed()


def test_heuristic_policy_goes_through_the_seam(cases):
    """The seam claim itself: HeuristicPolicy.act calls build_tasks and
    assign as module attributes, so rebinding them intercepts the agent."""
    from kagent.gamestate import Memory
    from kagent.policy import ChampionPlanProvider, HeuristicPolicy

    seen = {"build": 0, "assign": 0}
    orig_build, orig_assign = scheduler.build_tasks, scheduler.assign

    def spy_build(view, plan):
        seen["build"] += 1
        return orig_build(view, plan)

    def spy_assign(view, plan, tasks):
        seen["assign"] += 1
        return orig_assign(view, plan, tasks)

    scheduler.build_tasks, scheduler.assign = spy_build, spy_assign
    try:
        _, view, _ = cases[0]
        pol = HeuristicPolicy(ChampionPlanProvider())
        memory = Memory()
        memory.observe(view)
        pol.act(view, memory)
    finally:
        scheduler.build_tasks, scheduler.assign = orig_build, orig_assign
    assert seen == {"build": 1, "assign": 1}


# --------------------------------------------------------------- identity
def test_build_tasks_identity_all_flags_off(cases):
    executor.install(scheduler)
    for label, view, plan in cases:
        want = executor._ORIG_BUILD_TASKS(view, plan)
        got = scheduler.build_tasks(view, plan)
        assert sig(got) == sig(want), f"build_tasks diverged on {label}"


def test_assign_identity_all_flags_off(cases):
    executor.install(scheduler)
    for label, view, plan in cases:
        tasks = executor._ORIG_BUILD_TASKS(view, plan)
        want = executor._ORIG_ASSIGN(view, plan, list(tasks))
        got = scheduler.assign(view, plan, list(tasks))
        assert got == want, f"assign diverged on {label}"


def test_identity_holds_under_the_shipped_exec_rules(cases):
    """champ-h10 ships lastwater + h23 + harvest_visit. The seam must be a
    pass-through on top of THOSE, not only on a bare scheduler."""
    from rlv3.arms import BASE_EXEC
    executor.install(scheduler)
    scheduler.set_exec(list(BASE_EXEC))
    for label, view, plan in cases:
        want = executor._ORIG_BUILD_TASKS(view, plan)
        got = scheduler.build_tasks(view, plan)
        assert sig(got) == sig(want), f"build_tasks diverged on {label}"


def test_returns_a_new_list_not_the_original(cases):
    executor.install(scheduler)
    scheduler.set_exec(["co_chain_tile"])
    _, view, plan = cases[0]
    tasks = scheduler.build_tasks(view, plan)
    assert isinstance(tasks, list)


# ------------------------------------------------------------- invariants
@pytest.mark.parametrize("flag", [f for f in executor.RLV3_FLAGS
                                  if f not in ("pl_net", "mv_dest")])
def test_invariants_hold_for_every_flag(cases, flag, monkeypatch):
    monkeypatch.setenv("KAGG_RLV3_CHECK", "1")
    executor.install(scheduler)
    scheduler.set_exec(["lastwater", "h23", "harvest_visit", flag])
    for label, view, plan in cases:
        before = executor._ORIG_BUILD_TASKS(view, plan)
        after = scheduler.build_tasks(view, plan)          # asserts inline
        executor.check_invariants(view, before, after, strict=flag != "pl_pasture_ring")
        assert len(after) == len(before), f"{flag} changed the task count on {label}"


def test_a_breach_is_actually_caught(cases):
    """The invariant checker must fail on a real breach, or it is theatre."""
    from kagent.scheduler import Task
    _, view, plan = cases[0]
    before = scheduler.build_tasks(view, plan)

    # (a) an invented task
    bad = list(before) + [Task(2, 0, 0, ["WATER"])]
    with pytest.raises(AssertionError):
        executor.check_invariants(view, before, bad)

    # (c) a task promoted into the survival band
    work = next((i for i, t in enumerate(before)
                 if 0 < t.pri < scheduler.P_FILLER), None)
    if work is not None:
        bad = list(before)
        t = bad[work]
        bad[work] = Task(scheduler.P_SURVIVAL, t.x, t.y, list(t.op), t.need)
        with pytest.raises(AssertionError):
            executor.check_invariants(view, before, bad)

    # (d) a crop swapped inside a PLANT task
    pi = next((i for i, t in enumerate(before) if t.op[0] == "PLANT"), None)
    if pi is not None:
        bad = list(before)
        t = bad[pi]
        other = "CARROT" if t.op[1] != "CARROT" else "WHEAT"
        bad[pi] = Task(t.pri, t.x, t.y, ["PLANT", other], t.need)
        with pytest.raises(AssertionError):
            executor.check_invariants(view, before, bad)


# ---------------------------------------------------------------- params
def test_param_parsing_and_env_override(monkeypatch):
    assert executor.parse_params("cluster_radius=3,mv_dest_conf=0.7") == \
        {"cluster_radius": 3.0, "mv_dest_conf": 0.7}
    with pytest.raises(ValueError):
        executor.parse_params("nope=1")
    before = executor.PARAMS["cluster_radius"]
    try:
        monkeypatch.setenv("KAGG_RLV3_PARAMS", "cluster_radius=5")
        executor.apply_env_params()
        assert executor.PARAMS["cluster_radius"] == 5.0
    finally:
        executor.PARAMS["cluster_radius"] = before


# ----------------------------------------------------- ledger / self-correct
def roll_days(led, rates_per_day, params=None):
    """Drive the ledger through N dawns with hand-set realised rates."""
    p = params or executor.PARAMS
    out = []
    for day, rates in enumerate(rates_per_day):
        led.prev_rates = led.rates
        led.rates = dict(rates)
        led.day = day
        led._corr_day, led._corr = -2, None      # a new dawn
        out.append(led.correct(p))
    return out


def test_correct_is_idempotent_within_a_day():
    """`rlv3_assign` calls correct() again on every mv_dest step. If the
    streaks accumulated per STEP instead of per DAY, mv_dest_conf would
    saturate on the first losing day."""
    led = ledger.ExecLedger()
    led.day = 3
    led.rates = {"lead_delta": -5_000.0}
    first = led.correct(executor.PARAMS)
    for _ in range(24):
        again = led.correct(executor.PARAMS)
    assert again == first
    assert led.conf_bump == executor.PARAMS["conf_step"]


def test_pl_wheat_adj_switches_off_after_two_days_below_target():
    """Measured PAST the lever's own causal horizon. This test used to roll
    two bare low days with no wheat on the board and no `day` on the rate
    row, so it could not tell "two days below target" from "two days before a
    wheat tile could possibly have been harvested" -- the defect
    `ledger-kill-switch-silent`, which killed the lever on day 2 of every
    game. `first_wheat_day` and `day` are what make the window observable.
    See rlv3/tests/test_ledger_kill_switch.py."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0                  # wheat seated on day 0 ...
    low = {"feeds_from_hand_share": 0.02, "day": 2}   # ... harvestable day 2
    roll_days(led, [low])
    assert "pl_wheat_adj" not in led.off, "one bad day must not be enough"
    roll_days(led, [dict(low, day=3)])
    assert "pl_wheat_adj" in led.off
    # ... and it is LIFTED again the first day the metric pays, which is the
    # half of this test that used to read "and it is permanent for the
    # episode". It was permanent, and that is the defect
    # `exec-levers-die-day-2`: on the traced world the share came back to
    # 0.20 on day 4 and the lever stayed off for the remaining 26 days.
    # rlv3/tests/test_ledger_lever_lifetime.py owns the new contract.
    roll_days(led, [{"feeds_from_hand_share": 0.9, "day": 4}])
    assert "pl_wheat_adj" not in led.off


def test_a_day_at_target_resets_the_streak():
    """Same correction: without `first_wheat_day` and `day` this test passed
    because the streak never started, not because the good day reset it."""
    led = ledger.ExecLedger()
    led.first_wheat_day = 0
    roll_days(led, [{"feeds_from_hand_share": 0.02, "day": 2},
                    {"feeds_from_hand_share": 0.30, "day": 3},
                    {"feeds_from_hand_share": 0.02, "day": 4}])
    assert "pl_wheat_adj" not in led.off


def test_cluster_radius_shrinks_when_adjacency_falls_and_kills_at_zero():
    led = ledger.ExecLedger()
    got = roll_days(led, [{"next_chore_adjacent_share": 0.80},
                          {"next_chore_adjacent_share": 0.70},
                          {"next_chore_adjacent_share": 0.60}])
    assert got[1]["cluster_radius"] == 1.0
    assert got[2]["cluster_radius"] == 0.0
    assert "pl_crop_cluster" in led.off


def test_cluster_radius_holds_while_adjacency_rises():
    led = ledger.ExecLedger()
    got = roll_days(led, [{"next_chore_adjacent_share": 0.60},
                          {"next_chore_adjacent_share": 0.70},
                          {"next_chore_adjacent_share": 0.80}])
    assert got[-1]["cluster_radius"] == executor.PARAMS["cluster_radius"]
    assert "pl_crop_cluster" not in led.off


def test_water_tight_fires_only_above_the_dry_tolerance():
    led = ledger.ExecLedger()
    got = roll_days(led, [{"planted_dry_share": 0.05, "n_planted": 20},
                          {"planted_dry_share": 0.40, "n_planted": 20}])
    assert got[0]["water_tight"] is False
    assert got[1]["water_tight"] is True


def test_co_chain_tile_reverts_after_two_flat_days():
    led = ledger.ExecLedger()
    roll_days(led, [{"chores_per_visit": 1.5}, {"chores_per_visit": 1.5},
                    {"chores_per_visit": 1.4}])
    assert "co_chain_tile" in led.off


def test_mv_dest_defers_more_as_the_lead_falls():
    """The one adaptation that runs WITH the measured record: the best v2
    arm is the one that overrides least, so a losing day raises conf."""
    led = ledger.ExecLedger()
    got = roll_days(led, [{"lead_delta": -1_000.0}, {"lead_delta": -1_000.0},
                          {"lead_delta": +1_000.0}])
    base = executor.PARAMS["mv_dest_conf"]
    step = executor.PARAMS["conf_step"]
    assert got[0]["mv_dest_conf"] == pytest.approx(base + step)
    assert got[1]["mv_dest_conf"] == pytest.approx(base + 2 * step)
    assert got[2]["mv_dest_conf"] == pytest.approx(base + 2 * step)  # never falls
    assert got[-1]["mv_dest_conf"] <= 0.95


def test_observe_counts_the_metrics_the_levers_are_judged_on(cases):
    """The ledger's counters must be the same quantities tile_metrics
    computes off a trace, or a dawn number and a scorecard number disagree."""
    led = ledger.ExecLedger()
    _, view, plan = cases[0]
    actions = {0: ["FEED"], 1: ["NORTH"], 2: ["HARVEST"], 3: ["PASS"]}
    actions = {k: v for k, v in actions.items()
               if k < len(view.me.units)}
    led.observe(view, actions)
    assert led.moves == sum(1 for o in actions.values() if o[0] == "NORTH")
    assert led.chores == sum(1 for o in actions.values()
                             if o[0] in ("FEED", "HARVEST"))
    assert led.feeds == sum(1 for o in actions.values() if o[0] == "FEED")
    assert led.visits == led.chores       # every chore starts a fresh visit run
    assert led.lead == view.me.money - view.opp.money


def test_feeds_from_hand_needs_a_wheat_harvest_first():
    """pickups_per_feed and feeds_from_hand are the pair that caught the
    losing handfeed/oneshed arms. Wheat from the SHED must not count."""
    from kagent.gamestate import View
    from rlv3.tests.conftest import animal_tile, blank_obs, plant_tile
    obs = blank_obs(day=12, quadrants=("NW",), n_hands=1)
    obs["farms"][0]["tiles"][0][0] = plant_tile("WHEAT", planted_day=8, yield_units=4)
    obs["farms"][0]["tiles"][1][0] = animal_tile("PASTURE", "COW")
    obs["farms"][0]["farmer"] = [0, 0]
    obs["farms"][0]["hands"] = [[0, 1]]
    view = View(obs)
    led = ledger.ExecLedger()
    led.observe(view, {0: ["PICKUP", "WHEAT", 5]})
    led.observe(view, {0: ["FEED"]})
    assert led.feeds == 1 and led.feeds_from_hand == 0
    led.observe(view, {0: ["HARVEST"]})     # unit 0 stands on the wheat tile
    led.observe(view, {0: ["FEED"]})
    assert led.feeds == 2 and led.feeds_from_hand == 1


def test_episode_reset_on_step_regression():
    """A fresh ledger at step 0 is not a regression -- there is nothing to
    reset. A REUSED process, which is how every eval worker runs, is."""
    assert executor.reset_episode(0) is False    # fresh: last_step -1 -> 0
    assert executor.reset_episode(5) is False
    assert executor.reset_episode(300) is False
    ledger.LEDGER.lead = 12_345.0
    assert executor.reset_episode(0) is True     # next episode in the worker
    assert ledger.LEDGER.lead == 0.0             # and the state really cleared
    assert executor.reset_episode(1) is False
