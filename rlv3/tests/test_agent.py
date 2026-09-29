"""Tests for the COMPOSITION: rlv3/agent.py, rlv3/arm.py, and the three
places the four components disagreed with each other before they were wired.

No games are played here. The behavioural check is rlv3/smoke.py, which plays
two recorded games and asserts the zero-dose identity against the standing
draw; these tests pin the wiring that makes that run mean something.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kagent import market, scheduler                      # noqa: E402
from kagent.gamestate import Memory, View                 # noqa: E402
from kagent.constants import (                            # noqa: E402
    SHOPS, TOWN_CENTER_PRODUCTS, TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
)

from rlv3 import agent as AGENT                           # noqa: E402
from rlv3 import allocation as ALLOC                      # noqa: E402
from rlv3 import arm as ARM                               # noqa: E402
from rlv3 import land as LAND                             # noqa: E402
from rlv3 import market_model as MM                       # noqa: E402
from rlv3 import objective as OBJ                         # noqa: E402
from rlv3 import rent as RENT                             # noqa: E402
from rlv3 import selling as SELL                          # noqa: E402

from rlv3.tests.conftest import blank_obs, plant_tile     # noqa: E402


# ============================================================ arm strings
def test_every_preset_builds():
    for name in AGENT.PRESETS:
        cfg = AGENT.config_from_spec({"kind": "rlv3", "preset": name})
        assert cfg.label == f"rlv3:{name}"
        cfg.sell_cfg()                      # raises on an unknown sell mode
        assert cfg.all_exec()[:len(AGENT.BASE_EXEC)] == list(AGENT.BASE_EXEC)


def test_an_unknown_preset_is_refused_at_parse_time():
    with pytest.raises(ValueError, match="unknown rlv3 preset"):
        ARM.parse("nosuch")


def test_an_unknown_spec_field_is_refused_rather_than_ignored():
    """A typo must not silently produce the default arm: that is the shape of
    a result nobody can tell from a null one."""
    with pytest.raises(ValueError, match="unknown rlv3 spec field"):
        ARM.parse("full:lamda=0.5")


def test_an_unknown_sell_mode_is_refused():
    with pytest.raises(ValueError, match="unknown sell mode"):
        AGENT.config_from_spec({"kind": "rlv3", "sell": "hold_everything"}).sell_cfg()


def test_overrides_parse_as_json_and_survive_brackets():
    spec = ARM.parse("select:lam=0.25,band=3,lam_grid=[0,1],project=false")
    cfg = AGENT.config_from_spec(spec)
    assert cfg.lam == 0.25 and cfg.band == 3
    assert cfg.lam_grid == (0, 1)
    assert cfg.project is False and cfg.select is True


def test_every_preset_now_runs_the_base_scheduler():
    """MAIA2 IS OFF (2026-09-29): `agent.FULL_EXEC` is empty, so no preset
    carries an rlv3 executor flag and `install_seams` never rebinds
    build_tasks/assign. If this ever fails, someone re-enabled the executor
    seam without saying so, and every alloc/sell number after that point is
    confounded by a movement lever worth ~$0."""
    assert AGENT.FULL_EXEC == ()
    for name in AGENT.PRESETS:
        cfg = AGENT.config_from_spec(ARM.parse(name))
        assert cfg.exec_flags == (), f"{name} carries rlv3 exec flags"
        assert cfg.all_exec() == list(AGENT.BASE_EXEC), name


def test_the_shipped_exec_rules_are_always_first_and_never_dropped():
    """champ-h10's three rules are the BASELINE, not part of rlv3. An arm that
    drops them measures their loss, not this package. Tested against an
    EXPLICIT flag rather than a preset, so it keeps working whether maia2 is
    on or off."""
    cfg = AGENT.config_from_spec(
        {"kind": "rlv3", "exec_flags": list(AGENT.MAIA2_EXEC)})
    flags = cfg.all_exec()
    assert flags[:len(AGENT.BASE_EXEC)] == list(AGENT.BASE_EXEC)
    assert flags.count("pl_wheat_adj") == 1
    assert len(flags) == len(set(flags))


def test_exec_flags_that_duplicate_a_shipped_rule_are_not_doubled():
    cfg = AGENT.config_from_spec(
        {"kind": "rlv3", "exec_flags": ["lastwater", "pl_wheat_adj"]})
    assert cfg.all_exec() == list(AGENT.BASE_EXEC) + ["pl_wheat_adj"]


def test_make_job_keeps_the_callers_spec_verbatim():
    """Round-tripping through to_json() silently dropped params/rules/land_theta
    once; the job spec must carry what config_from_spec validated."""
    spec = ARM.parse("full:lam=0.75")
    spec["params"] = {"lam": 0.75, "fill_bare": False}
    job = AGENT.make_job(spec, "rl/datasets/sessions/x.json", 0)
    js = job["learner_policy"]
    assert js["kind"] == "rlv3"
    assert js["params"] == {"lam": 0.75, "fill_bare": False}
    assert js["exec"] == AGENT.config_from_spec(spec).all_exec()
    # and it is still a valid spec after the round trip
    assert AGENT.config_from_spec(js).lam == 0.75


def test_make_job_seats_us_opposite_the_recorded_side():
    for ss in (0, 1):
        job = AGENT.make_job(ARM.parse("base"), "p.json", ss)
        assert job["side"] == 1 - ss
        assert job["opponent"] == ("session", "p.json", ss)
        assert job["temperature"] == 0.0


# ======================================================= seams and isolation
def _attrs():
    return market.build_orders, scheduler.build_tasks, scheduler.assign


def test_the_zero_dose_preset_patches_nothing():
    """`base` must be reachable without touching a single process global, or
    the identity arm is not an identity arm."""
    before = _attrs()
    restore = AGENT.install_seams(AGENT.config_from_spec(ARM.parse("base")))
    try:
        assert _attrs() == before
    finally:
        restore()
    assert _attrs() == before


@pytest.mark.parametrize("preset", ["recommended", "sell", "exec", "full", "select"])
def test_every_seam_installs_and_restores_exactly(preset):
    before = _attrs()
    cfg = AGENT.config_from_spec(ARM.parse(preset))
    restore = AGENT.install_seams(cfg)
    try:
        if cfg.sell_cfg() is not None:
            assert market.build_orders is not before[0]
        if cfg.exec_flags:
            assert scheduler.build_tasks is not before[1]
            assert scheduler.assign is not before[2]
    finally:
        restore()
    assert _attrs() == before


def test_the_rlv3_exec_flags_are_registered_before_rollout_sets_them():
    """rl.rollout.run_episode calls scheduler.set_exec(spec['exec']) AFTER the
    agent is built, and set_exec raises on a flag it has never heard of. The
    registration therefore has to happen during the build."""
    cfg = AGENT.config_from_spec(
        {"kind": "rlv3", "exec_flags": list(AGENT.MAIA2_EXEC)})
    restore = AGENT.install_seams(cfg)
    try:
        prev = dict(scheduler.EXEC)
        scheduler.set_exec(cfg.all_exec())          # must not raise
        assert scheduler.EXEC["pl_wheat_adj"] is True
        scheduler.EXEC.update(prev)
    finally:
        restore()


def test_registering_the_exec_seam_does_not_install_it():
    """`base` and `recommended` carry no rlv3 flag, so the scheduler must keep
    the shipped build_tasks even though the flag KEYS exist."""
    before = _attrs()
    restore = AGENT.install_seams(AGENT.config_from_spec(ARM.parse("recommended")))
    try:
        assert scheduler.build_tasks is before[1]
        assert "pl_wheat_adj" in scheduler.EXEC
    finally:
        restore()


def test_an_unmarked_memory_gets_the_shipped_sell_sizer():
    """ARM ISOLATION. rl.rollout.run_episode has no teardown hook for a market
    patch, so a worker that plays an rlv3 arm and then a baseline arm would
    run the baseline through rlv3's sizer. `install(None)` routes only the
    memories an rlv3 provider has marked."""
    calls = []
    orig = market.build_orders

    def spy(view, plan, memory):
        calls.append("shipped")
        return []

    market.build_orders = spy
    try:
        restore = SELL.install(None)
        try:
            market.build_orders(None, None, Memory())      # unmarked
            assert calls == ["shipped"]
        finally:
            restore()
    finally:
        market.build_orders = orig


def test_install_is_idempotent_and_never_wraps_its_own_wrapper():
    """`selling.build_orders` calls the SHIPPED builder; reading
    `market.build_orders` for it made the installed path recurse until the
    stack blew. Two installs must not produce two layers either."""
    before = market.build_orders
    r1 = SELL.install(SELL.RECOMMENDED)
    r2 = SELL.install(SELL.RECOMMENDED)
    try:
        assert SELL._SHIPPED is before
    finally:
        r2()
        r1()
    assert market.build_orders is before
    assert SELL._SHIPPED is None


# ====================================================== the rollout adapter
def test_register_delegates_every_other_kind_untouched():
    from rl import rollout
    shipped = rollout._make_dev_policy_agent
    ARM.register()
    try:
        assert rollout._make_dev_policy_agent is not shipped
        with pytest.raises(ValueError, match="unknown learner_policy kind"):
            rollout._make_dev_policy_agent({"kind": "not_a_kind"})
    finally:
        ARM.unregister()
    assert rollout._make_dev_policy_agent is shipped


def test_register_is_idempotent():
    from rl import rollout
    shipped = rollout._make_dev_policy_agent
    ARM.register()
    ARM.register()
    ARM.unregister()
    assert rollout._make_dev_policy_agent is shipped
    assert not ARM.is_registered()


def test_a_dead_arm_raises_instead_of_returning_a_number(monkeypatch):
    """The whole point of the counter. An episode that reports itself as rlv3
    having never reached the plan provider ran the SHIPPED agent, and its row
    in a table is indistinguishable from a null result."""
    from rl import rollout

    monkeypatch.setattr(rollout, "run_episode",
                        lambda job: {"final_money": 1.0, "opp_money": 0.0})
    job = AGENT.make_job(ARM.parse("base"), "p.json", 0)
    with pytest.raises(RuntimeError, match="DEAD, not null"):
        ARM.run_episode(job)
    ARM.unregister()


def test_a_non_rlv3_job_is_passed_straight_through(monkeypatch):
    from rl import rollout

    seen = {}

    def stub(job):
        seen["job"] = job
        return {"ok": True}

    monkeypatch.setattr(rollout, "run_episode", stub)
    out = ARM.run_episode({"learner_policy": {"kind": "champion"}})
    assert out == {"ok": True} and seen["job"]["learner_policy"]["kind"] == "champion"
    ARM.unregister()


def test_make_jobs_puts_the_band_on_every_spec():
    games = [("a.json", 0, 0), ("b.json", 1, 5)]
    jobs = ARM.make_jobs(ARM.parse("base"), games)
    assert [j["learner_policy"]["band"] for j in jobs] == [0, 5]


def test_an_explicit_band_wins_over_the_games_band():
    jobs = ARM.make_jobs(ARM.parse("base:band=2"), [("a.json", 0, 5)])
    assert jobs[0]["learner_policy"]["band"] == 2


# ============================================ the Memory attribute collision
def _view(day=12, **kw):
    obs = blank_obs(day=day, **kw)
    for i in range(6):
        obs["farms"][0]["tiles"][0][i] = plant_tile(planted_day=day - 4)
        obs["farms"][1]["tiles"][0][i] = plant_tile(planted_day=day - 4)
    return View(obs)


def test_the_projection_state_does_not_live_where_land_keeps_its_ledger():
    """`land.observe_dawn` owns `memory.rlv3` as a DICT and re-creates it every
    dawn when it is not one; `allocation.chore_cost` reads
    `memory.rlv3["chore"]`. A ProjState parked there is destroyed on day 1 and
    takes the allocator down with it."""
    mem = Memory()
    view = _view()
    prov = AGENT.Rlv3PlanProvider(ARM.parse("base"))
    st = prov.proj(mem)
    assert isinstance(st, OBJ.ProjState)
    assert mem.rlv3_proj is st

    LAND.observe_dawn(mem, view)                     # the other component's dawn
    assert isinstance(mem.rlv3, dict)                # land still owns this one
    assert mem.rlv3_proj is st                       # and ours survived it
    ALLOC.chore_cost("COW", mem)                     # would raise on a ProjState


def test_the_provider_reuses_one_projection_state_across_the_season():
    mem = Memory()
    prov = AGENT.Rlv3PlanProvider(ARM.parse("base"))
    assert prov.proj(mem) is prov.proj(mem)


def test_a_new_episode_gets_a_fresh_projection_state():
    """Workers are reused. Two agents built from the same spec must not share
    a solved opponent-conversion table from a different season."""
    a, _ = AGENT.make_dev_agent(ARM.parse("base"))
    b, _ = AGENT.make_dev_agent(ARM.parse("base"))
    try:
        m1, m2 = Memory(), Memory()
        assert a.provider.proj(m1) is not b.provider.proj(m2)
    finally:
        a.restore()
        b.restore()


# ================================================ the two duplicated helpers
@pytest.mark.parametrize("a,b", [(0, 1), (0, 4), (3, 4), (4, 5), (23, 24),
                                 (24, 25), (0, 24), (23, 25), (47, 96),
                                 (100, 100), (96, 720)])
def test_the_two_drain_clocks_agree(a, b):
    """rlv3 grew TWO implementations of the town-drain clock:
    `market_model.ticks_between` (selling) and
    `objective.ProjState._drain_between` (the opponent-flow solve). They
    disagreed by one step -- market_model counted (a, b], objective counts
    [a, b) -- and objective is the one that matches the engine:
    `_town_consume(env, state, step)` drains on `step % interval == 0` and the
    framework records the result at index step+1. market_model was corrected;
    this pins them together so the next edit cannot re-fork them."""
    shops = ("BAKERY", "YARN_STORE", "BRUNCH_SPOT")
    mm_shop, mm_cent = MM.ticks_between(a, b)
    obj_shop = OBJ.ProjState._ticks(TOWN_SHOP_SELL_INTERVAL, a, b) if b > a else 0
    obj_cent = OBJ.ProjState._ticks(TOWN_CENTER_SELL_INTERVAL, a, b) if b > a else 0
    assert (mm_shop, mm_cent) == (obj_shop, obj_cent)

    per_item = OBJ.ProjState._drain_between(shops, a, b) if b > a else {}
    for item in set(per_item) | {"WHEAT", "WOOL", "STRAWBERRY", "MILK"}:
        assert MM.drain_between(item, a, b, shops) == pytest.approx(
            per_item.get(item, 0.0)), item


def test_the_drain_clock_matches_the_engines_own_rule():
    """Brute force, straight off kaggriculture.py: `_town_consume` fires during
    step k and the result is observed at k+1, so ticks between observations at
    a and b are the ones at a .. b-1."""
    shops = ("BAKERY", "YARN_STORE")
    for a, b in [(0, 1), (3, 5), (23, 26), (0, 48)]:
        for item in ("WHEAT", "WOOL", "MILK"):
            total = 0.0
            for k in range(a, b):
                if k % TOWN_SHOP_SELL_INTERVAL == 0:
                    for shop in shops:
                        items = SHOPS[shop]
                        if item in items:
                            total += 2 if len(items) == 1 else 1
                if k % TOWN_CENTER_SELL_INTERVAL == 0 and item in TOWN_CENTER_PRODUCTS:
                    total += 1
            assert MM.drain_between(item, a, b, shops) == pytest.approx(total), (item, a, b)


def test_the_two_inventory_paths_agree():
    """`rent.inventory_path` (memoised on the View) and
    `market_model.inventory_path` (a bare delegate) both wrap
    `planner._rent_inventory`. Left as two names on purpose -- the memo is
    what makes the water-fill's few hundred evaluations a dawn cheap -- but
    they must never diverge."""
    view = _view()
    for h in (1, 7, 14):
        a = RENT.inventory_path(view, horizon=h)
        b = MM.inventory_path(view, h)
        assert set(a) == set(b)
        for item in a:
            assert a[item] == pytest.approx(b[item]), item


# ================================================================ counters
def test_counters_are_resettable_and_count_episodes():
    AGENT.reset_counters()
    assert AGENT.counters() == {}
    fn, _ = AGENT.make_dev_agent(ARM.parse("base"))
    try:
        assert AGENT.counters()["episodes"] == 1
        assert fn.provider.calls == 0          # nothing played yet
    finally:
        fn.restore()
        AGENT.reset_counters()


def test_the_agent_exposes_the_handles_a_driver_needs():
    fn, switched = AGENT.make_dev_agent(ARM.parse("full"))
    try:
        assert callable(fn)
        assert isinstance(fn.provider, AGENT.Rlv3PlanProvider)
        assert isinstance(fn.cfg, AGENT.Rlv3Config)
        assert callable(fn.restore)
        assert fn.last_view == {"v": None}     # rollout reads unsold_value here
        assert AGENT.last_agent() is fn
        assert switched in (True, False)
    finally:
        fn.restore()


def test_arm_parse_carries_the_shipped_executor_rules():
    """rl/session_eval builds rlv3 specs through arm.parse and rollout applies
    spec["exec"]; without it the arm plays with no executor rules at all."""
    from kagent.weights import EXEC_RULES
    spec = ARM.parse("readout")
    assert spec.get("exec") == list(EXEC_RULES), spec.get("exec")
    assert "shedguard" in spec["exec"]
