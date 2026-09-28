"""The strategy vector must REACH the board, and reproduce the champion at rest.

Why this file exists. The project's expensive mistakes are all the same shape:
a change that looks applied and is not. The learned head grid could not express
a leader's goose count, `dfb06e5` edited a function the planner calls zero
times, three model-unpack sites passed the whole suite because nothing
exercised the path, and the goose line was "refuted" while three independent
gates held it at zero. A search over `kagent.theta` is worth exactly the
coordinates that move a tile, so this pins:

  1. the vector round-trips through the unit cube without drifting,
  2. the champion's point IS the champion (a search that starts somewhere else
     is measuring its own starting error),
  3. `dev:theta:` dispatches in a FRESH interpreter -- multiprocessing starts
     `spawn`, so anything set only in the parent is invisible to the Pool that
     does the actual evaluating,
  4. the coordinates that were previously unreachable now change the plan.

Only (4) touches the planner; no games are played here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------- the vector
def test_unit_cube_round_trip_is_exact():
    from kagent import theta as T

    c = T.champion()
    assert T.from_unit(T.to_unit(c)) == c, "the champion drifts through the cube"


def test_every_knob_coordinate_is_read_by_the_planner():
    """A coordinate the planner never reads is a coordinate the search wastes
    its whole budget on. KNOB_DEFAULTS is the planner's own table."""
    from kagent import planner, theta as T

    unknown = [n for n in T.KNOB_NAMES if n not in planner.KNOB_DEFAULTS]
    assert not unknown, f"not planner knobs: {unknown}"


def test_every_exec_coordinate_is_a_real_flag():
    from kagent import scheduler, theta as T

    unknown = [n for n in T.EXEC_NAMES if n not in scheduler.EXEC_FLAGS]
    assert not unknown, f"not executor flags: {unknown}"


# Coordinates whose champion value IS the floor because the floor is the
# identity -- "no geese", "no standing carrot patch", "sell all the wheat".
# There is nothing below them to search, so sitting there is correct. Every
# OTHER coordinate on a bound means the bound was drawn too tight, and the
# search would only ever be able to move it one way.
NATURAL_FLOOR = {"geese", "carrot_standing", "tomato_standing", "wheat_floor"}


def test_champion_is_interior_to_every_bound():
    """A coordinate pinned at its own bound cannot be searched in one
    direction, and the champion sitting on an edge means the bound is wrong."""
    from kagent import theta as T

    c = T.champion()
    edge = []
    for name, lo, hi, kind in T.KNOB_SPEC:
        if kind == "bool":
            continue
        v = float(c[name])
        if v >= float(hi):
            edge.append(f"{name}={v} at the CEILING of [{lo}, {hi}]")
        elif v <= float(lo) and name not in NATURAL_FLOOR:
            edge.append(f"{name}={v} at the floor of [{lo}, {hi}]")
    assert not edge, "champion sits on a bound: " + "; ".join(edge)


def test_the_shipped_exec_rules_are_all_coordinates():
    """The search must be able to turn OFF what the agent currently ships, or
    it is only exploring additions."""
    from kagent import theta as T
    from kagent.weights import EXEC_RULES

    missing = [f for f in (EXEC_RULES or []) if f not in T.EXEC_NAMES]
    assert not missing, f"shipped but not searchable: {missing}"


# ------------------------------------------------------------ the dispatch
def test_session_eval_parses_a_theta_dev_arm():
    src = open(os.path.join(REPO, "rl", "session_eval.py")).read()
    assert '"theta"' in src, "rl/session_eval.py does not parse dev:theta:<vector.json>"
    assert "dev:theta:<vector.json>" in src, "the error message must advertise it"


def test_rollout_dispatches_theta_in_a_fresh_interpreter():
    """`spawn` re-imports the module in every worker, so the branch has to live
    in rl/rollout.py itself -- a parent-process patch passes in-process tests
    and raises 'unknown learner_policy kind' inside the Pool."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from rl import rollout\n"
        "from kagent import theta as T\n"
        "fn, switched = rollout._make_dev_policy_agent(T.spec(T.champion()))\n"
        "assert callable(fn) and switched is False\n"
        "print('OK')\n" % REPO
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_theta_spec_carries_exec_flags_where_run_episode_reads_them():
    """run_episode applies spec['exec'] per episode and restores it after, so a
    worker playing several candidates does not leak one arm's rules into the
    next. The flags must be on that key and nowhere else."""
    from kagent import theta as T

    c = T.champion()
    c["animal_visit"] = True
    spec = T.spec(c)
    assert "animal_visit" in spec["exec"]
    assert not any(k in spec["knobs"] for k in T.EXEC_NAMES), \
        "executor flags must not be passed as planner knobs"


# ------------------------------------------------------- reaching the plan
def _plan_for(theta, day):
    """The day plan a vector produces on a real mid-season state."""
    from kagent.gamestate import Memory, View
    from kagent.policy import ThetaPlanProvider
    from tests.helpers import make_view          # type: ignore

    view = make_view(day=day)
    mem = Memory()
    return ThetaPlanProvider(theta["knobs"] if "knobs" in theta else theta).make_plan(view, mem)


def test_geese_reach_the_animal_buys():
    """The coordinate that was pinned at zero through every previous arm."""
    from kagent import planner, theta as T
    from kagent.gamestate import Memory

    class _Stub:
        pass

    c = T.champion()
    assert planner._champ_herd_targets(10, {**planner.KNOB_DEFAULTS, **{
        n: c[n] for n in T.KNOB_NAMES}})["GOOSE"] == 0, "champion should run no geese"

    g = {**planner.KNOB_DEFAULTS, "geese": 6, "geese_day": 6}
    assert planner._champ_herd_targets(10, g)["GOOSE"] == 6, \
        "the geese knob does not reach the herd target"
    assert planner._champ_herd_targets(3, g)["GOOSE"] == 0, \
        "geese_day must gate the target"


def test_theta_provider_sets_knobs_before_the_first_plan():
    """The knob bandit only decides on days 6/9/12/15, so the opening is
    unreachable through it; the provider must set the whole vector at day 0."""
    from kagent.policy import ThetaPlanProvider
    from kagent import planner

    knobs = {"melon_opening": 4, "geese": 5}
    seen = {}

    def fake(view, memory, speculative=False):
        seen["knobs"] = dict(getattr(memory, "knobs", None) or {})
        return "plan"

    class _Mem:
        knobs = None

    orig = planner.make_day_plan
    planner.make_day_plan = fake
    try:
        ThetaPlanProvider(knobs).make_plan(object(), _Mem())
    finally:
        planner.make_day_plan = orig
    assert seen["knobs"] == knobs, f"knobs not set before the plan: {seen}"


# --------------------------------------------------------------- shipping
def test_a_theta_bundle_runs_the_vector_and_not_the_champion():
    """A searched vector that cannot be bundled is a vector that cannot ship.

    `dfb06e5` is the standing lesson: a careful change to a function the
    planner calls ZERO times. This execs the bundle's own wiring block in a
    flat namespace -- the shape a built main.py actually has -- and asserts the
    agent that comes out is driven by the vector.
    """
    import kagent.planner as planner
    import kagent.scheduler as scheduler
    from kagent.policy import (ChampionPlanProvider, HeuristicPolicy,  # noqa: F401
                               SafeAgent, ThetaPlanProvider)

    src = open(os.path.join(REPO, "kagent", "policy.py")).read()
    start = src.index("_LAYERS = decode_weights(WEIGHTS_DATA)")
    body = src[start:src.index("\nelif BANDIT_RULES:", start)] + "\nelse:\n    _AGENT = None\n"
    ns = {"decode_weights": lambda _x: None, "WEIGHTS_DATA": None,
          "THETA_KNOBS": {"geese": 6, "melon_opening": 4},
          "SafeAgent": SafeAgent, "HeuristicPolicy": HeuristicPolicy,
          "ThetaPlanProvider": ThetaPlanProvider,
          "ChampionPlanProvider": ChampionPlanProvider}
    exec(compile(body, "bundle", "exec"), ns)
    agent = ns["_AGENT"]
    assert agent is not None, "the THETA_KNOBS branch did not build an agent"
    provider = agent.policy.plan_provider
    assert isinstance(provider, ThetaPlanProvider), type(provider)
    assert provider.knobs == {"geese": 6, "melon_opening": 4}


def test_theta_takes_precedence_over_the_knob_bandit():
    """The two are alternatives. A bundle carrying both would be unattributable
    -- the bandit overwrites four knobs at day 6 and the gate could not say
    which layer chose what -- so the exporter must refuse the combination."""
    out = subprocess.run(
        [sys.executable, "rl/export.py", "--clear", "--champion",
         "--theta", "/dev/null", "--bandit-rules", "/dev/null"],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    assert out.returncode != 0
    assert "alternatives, not layers" in out.stderr or "Expecting value" in out.stderr

    src = open(os.path.join(REPO, "kagent", "policy.py")).read()
    assert src.index("if THETA_KNOBS:") < src.index("elif BANDIT_RULES:")


def test_exporter_rejects_a_knob_the_planner_does_not_read():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"knobs": {"not_a_knob": 3}, "exec": []}, fh)
        path = fh.name
    try:
        out = subprocess.run(
            [sys.executable, "rl/export.py", "--clear", "--champion", "--theta", path],
            capture_output=True, text=True, cwd=REPO,
            env={**os.environ, "PYTHONPATH": REPO},
        )
        assert out.returncode != 0, out.stdout
        assert "does not read" in out.stderr, out.stderr[-400:]
    finally:
        os.unlink(path)


# ------------------------------------------------- the sampler must reach
def test_every_coordinate_is_in_at_least_one_mechanism_group():
    """Group sampling is how a coupled combination gets tried. A coordinate in
    no group is only ever reached by the uniform branch, at a fraction of the
    rate."""
    from kagent import theta as T

    covered = {n for ns in T.GROUPS.values() for n in ns}
    missing = [n for n in T.NAMES if n not in covered]
    assert not missing, f"not in any mechanism group: {missing}"
    unknown = [n for n in covered if n not in T.NAMES]
    assert not unknown, f"group names a coordinate that does not exist: {unknown}"


def test_the_goose_joint_is_actually_SAMPLED():
    """The measured reason this project's goose arms were meaningless.

    Under uniform subset sampling `geese` and `animal_visit` are each selected
    about 7% of the time, so both land in 0.5% of candidates -- half a
    candidate in a full run -- and accumulation cannot help because each half
    loses alone. If this rate ever falls back to noise, the search is once more
    unable to express the line of play it is supposed to be testing.
    """
    import random

    from kagent import theta as T

    rng = random.Random(3)
    c = T.champion()
    mu, sigma = T.to_unit(c), [0.25] * T.DIM
    both = sum(1 for _ in range(1000)
               if (lambda t: t["geese"] > 0 and t["animal_visit"])(
                   T.from_unit(T.sample_block(rng, mu, sigma, 4))))
    assert both >= 40, f"goose joint sampled in only {both/10:.1f}% of candidates"


def test_a_perturbation_always_changes_something():
    """A candidate costs 144 games. One that decodes back to the mean buys
    nothing -- measured: premium_scale 0.35 -> 0.3431 scored +0.000 +/- 0.000,
    bit-identical on every game."""
    import random

    from kagent import theta as T

    rng = random.Random(5)
    c = T.champion()
    mu, sigma = T.to_unit(c), [0.25] * T.DIM
    noop = sum(1 for _ in range(400)
               if T.from_unit(T.sample_block(rng, mu, sigma, 4)) == c)
    assert noop == 0, f"{noop} of 400 candidates changed nothing"
