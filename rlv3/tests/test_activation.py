"""Check (b) of the risk list: the SPAWN hazard, which fails silently.

`multiprocessing.get_start_method()` is "spawn" here, so a monkeypatch made
in the parent process never reaches `rl/session_eval.py`'s workers. An arm
installed that way would play the SHIPPED executor and report itself as
rlv3 -- a clean, plausible, completely dead result. That is exactly what
happened to the Stage B board arms (`kaggriculture-silent-trainer-failures`):
three runs read as null verdicts before anyone checked that the thing under
test had ever run.

So these tests do not check that `install()` works in THIS process. They
check that it reaches a spawned child, that the arm JSON the two instruments
consume actually turns the flag on there, and that a counter in `stats()` is
non-zero afterwards -- the "the arm is dead, not null" check, run before any
evaluation is spent.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from rlv3 import arms, executor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def child_env(env=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{REPO}:{os.path.join(REPO, 'rlv3')}"
    for k in ("KAGG_EXEC", "KAGG_RLV3", "KAGG_RLV3_PARAMS", "KAGG_RLV3_CHECK"):
        e.pop(k, None)
    e.update(env or {})
    return e


def run_child(body, env=None, timeout=300):
    """Run `body` in a FRESH interpreter with PYTHONPATH=.:rlv3, so `site`
    imports rlv3/sitecustomize.py exactly as it does in an eval worker.

    The body goes to a real FILE, never `python -c`. Under the spawn start
    method a child re-imports the parent's `__main__` module, and with `-c`
    there is no path to re-import from: two workers came up sharing one pipe
    handle and the Pool deadlocked. A tempfile gives spawn a real
    `__main__` to load, which is also what an eval worker actually has.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=REPO, delete=False) as fh:
        fh.write(textwrap.dedent(body))
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], cwd=REPO, env=child_env(env),
                           capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)
    assert p.returncode == 0, f"child failed:\n{p.stdout}\n{p.stderr}"
    return p.stdout.strip(), p.stderr


def test_the_start_method_really_is_spawn():
    """If this ever becomes 'fork', a parent-process patch would reach the
    workers and this whole file is about a hazard that no longer exists.
    It is asserted so the assumption cannot rot silently."""
    assert mp.get_start_method() == "spawn"


def test_sitecustomize_is_inert_without_the_env_flag():
    out, _ = run_child("""
        import sitecustomize
        from kagent import scheduler
        from rlv3 import executor
        print("installed", executor.is_installed(),
              "patched", scheduler.build_tasks is executor.rlv3_build_tasks)
    """)
    assert out == "installed False patched False"


def test_sitecustomize_installs_at_interpreter_startup():
    """No import of rlv3 anywhere in the body: `site` must have done it."""
    out, _ = run_child("""
        from kagent import scheduler
        from rlv3 import executor
        print("installed", executor.is_installed(),
              "patched", scheduler.build_tasks is executor.rlv3_build_tasks)
    """, env={"KAGG_RLV3": "1"})
    assert out == "installed True patched True"


def test_it_reaches_a_SPAWNED_worker():
    """THE test. A Pool worker is a fresh interpreter; the patch must be
    there without the parent sending anything."""
    out, _ = run_child("""
        import multiprocessing as mp

        def probe(_):
            from kagent import scheduler
            from rlv3 import executor
            return (executor.is_installed(),
                    scheduler.build_tasks is executor.rlv3_build_tasks,
                    scheduler.EXEC.get("pl_wheat_adj"))

        if __name__ == "__main__":
            ctx = mp.get_context("spawn")
            with ctx.Pool(2) as pool:
                print(pool.map(probe, [0, 1]))
    """, env={"KAGG_RLV3": "1"})
    assert "(True, True, False)" in out
    assert out.count("(True, True, False)") == 2


def test_kagg_exec_with_an_rlv3_flag_does_not_break_the_scheduler_import():
    """The ordering trap: kagent.scheduler calls set_exec(KAGG_EXEC) AT
    IMPORT TIME and set_exec raises on an unknown flag, so the rlv3 half of
    KAGG_EXEC has to be held back until register() has run."""
    out, _ = run_child("""
        from kagent import scheduler
        print("lastwater", scheduler.EXEC["lastwater"],
              "rlv3", scheduler.EXEC["pl_wheat_adj"])
    """, env={"KAGG_RLV3": "1", "KAGG_EXEC": "lastwater,pl_wheat_adj"})
    assert out == "lastwater True rlv3 True"


def test_an_unknown_flag_kills_the_process_rather_than_limping():
    """`site.execsitecustomize` CATCHES every exception out of sitecustomize,
    prints one stderr line and carries on -- measured: a child with a bad
    KAGG_EXEC exited 0 having run the shipped executor. So a typo must take
    the process down, or the arm is dead and the run looks fine."""
    e = child_env({"KAGG_RLV3": "1", "KAGG_EXEC": "pl_wheat_adjj"})
    p = subprocess.run([sys.executable, "-c", "from kagent import scheduler"],
                       cwd=REPO, env=e, capture_output=True, text=True, timeout=180)
    assert p.returncode != 0, "a bad flag must not exit 0"
    assert "unknown executor flag" in p.stderr
    assert "DEAD, not null" in p.stderr


def test_soft_mode_limps_on_but_says_so():
    e = child_env({"KAGG_RLV3": "1", "KAGG_EXEC": "pl_wheat_adjj",
                   "KAGG_RLV3_SOFT": "1"})
    p = subprocess.run([sys.executable, "-c", "from kagent import scheduler"],
                       cwd=REPO, env=e, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0
    assert "continuing WITHOUT rlv3" in p.stderr


def test_params_reach_the_child_through_the_environment():
    """`rlv3_params` in an arm JSON is provenance only -- neither instrument
    reads an unknown key. KAGG_RLV3_PARAMS is what actually takes effect,
    and being an env var it survives spawn."""
    out, _ = run_child("""
        from rlv3 import executor
        print(executor.PARAMS["cluster_radius"], executor.PARAMS["mv_dest_conf"])
    """, env={"KAGG_RLV3": "1", "KAGG_RLV3_PARAMS": "cluster_radius=3,mv_dest_conf=0.7"})
    assert out == "3.0 0.7"


# ------------------------------------------------------------- the arms
def test_every_arm_file_parses_the_way_both_instruments_parse_it():
    written = arms.write_all()
    for name, path in written.items():
        with open(path) as fh:
            rules = json.load(fh)
        assert rules["kind"] == "knobs"
        assert rules["exec"][:3] == list(arms.BASE_EXEC)
        # tools/agent_scorecard.parse_spec("knobs:<path>") is literally this
        from tools.agent_scorecard import parse_spec
        spec = parse_spec(f"knobs:{path}")
        assert spec["kind"] == "knobs"
        # and rl/rollout's branch reads the same key
        flags = spec.get("exec") or (spec.get("rules") or {}).get("exec")
        assert flags == rules["exec"], name


def test_the_base_arm_is_exactly_champ_h10():
    with open(os.path.join(REPO, arms.BASE_RULES_PATH)) as fh:
        base = json.load(fh)
    with open(os.path.join(REPO, "rlv3/arms/base.json")) as fh:
        mine = json.load(fh)
    assert mine["exec"] == base["exec"] == list(arms.BASE_EXEC)
    assert mine["conditions"] == base["conditions"]


def test_every_arm_turns_its_flag_on_in_a_spawned_worker():
    """The counter check at the flag level, for EVERY arm in one child: run
    each arm's exec list through the same `reset_exec(); set_exec(flags)`
    pair both instruments use, inside a spawned worker, and confirm the flag
    is live there. One child rather than one per arm because the machine is
    busy and each child is a fresh torch import."""
    arms.write_all()
    out, _ = run_child("""
        import glob, json, multiprocessing as mp

        def probe(path):
            from kagent import scheduler
            with open(path) as fh:
                rules = json.load(fh)
            scheduler.reset_exec()
            scheduler.set_exec(rules["exec"])
            return (path, {f: scheduler.EXEC.get(f) for f in rules["exec"]})

        if __name__ == "__main__":
            paths = sorted(glob.glob("rlv3/arms/*.json"))
            with mp.get_context("spawn").Pool(2) as pool:
                print(json.dumps(dict(pool.map(probe, paths))))
    """, env={"KAGG_RLV3": "1"})
    got = json.loads(out)
    assert got, "no arm files found"
    for name, flags in arms.ARMS.items():
        state = got[f"rlv3/arms/{name}.json"]
        for f in tuple(arms.BASE_EXEC) + tuple(flags):
            assert state.get(f) is True, f"{name}: {f} not live in the worker"


def test_stats_counter_is_non_zero_after_the_flag_actually_fires():
    """"If the counter is zero, the arm is dead, not null." This is that
    check, run on mid-season states instead of a game so it costs nothing.
    """
    out, _ = run_child("""
        import glob, json
        from kagent import planner, scheduler
        from kagent.gamestate import Memory, View
        from rlv3 import executor

        scheduler.reset_exec()
        scheduler.set_exec(json.load(open("rlv3/arms/pl_wheat_adj.json"))["exec"])
        assert executor.is_installed()
        fired = 0
        for team in ("mmpq", "boey", "dsm"):
            hits = sorted(glob.glob(f"replays/{team}/*.json"))
            if not hits:
                continue
            steps = json.load(open(hits[0]))["steps"]
            for day in (6, 12, 20):
                k = day * 24 + 8
                obs = dict(steps[k][0]["observation"])
                obs.setdefault("step", k)
                v = View(obs)
                m = Memory(); m.observe(v)
                plan = planner.make_day_plan(v, m, speculative=True)
                scheduler.build_tasks(v, plan)
        st = executor.stats()
        print(json.dumps({k: v for k, v in st.items() if v}))
    """, env={"KAGG_RLV3": "1", "KAGG_RLV3_CHECK": "1"})
    st = json.loads(out)
    assert st.get("pl_wheat_adj.moved", 0) > 0, \
        f"pl_wheat_adj never fired on mid-season states: {st}"


def test_run_line_carries_the_env_that_makes_the_arm_live():
    line = arms.run_line("pl_wheat_adj")
    assert "KAGG_RLV3=1" in line and "PYTHONPATH=.:rlv3" in line
    assert "knobs:rlv3/arms/pl_wheat_adj.json" in line
    ev = arms.run_line("pl_wheat_adj", instrument="session_eval")
    assert "dev:knobs:rlv3/arms/pl_wheat_adj.json" in ev
    assert "--procs 2" in ev, "the machine is busy: never above 2"
    p = arms.run_line("pl_crop_cluster", params={"cluster_radius": 3})
    assert 'KAGG_RLV3_PARAMS="cluster_radius=3"' in p


def test_flags_and_params_are_registered_consistently():
    assert set(executor.RLV3_FLAGS) == {
        "pl_wheat_adj", "pl_crop_cluster", "pl_water_first", "pl_pasture_ring",
        "pl_net", "co_chain_tile", "co_animal_first", "co_value", "mv_dest"}
    for k in ("cluster_radius", "pasture_shed_dist", "pl_net_conf", "mv_dest_conf",
              "conf_step", "handfeed_target", "dry_tol", "contest_day"):
        assert k in executor.PARAMS
