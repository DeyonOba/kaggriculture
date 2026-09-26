"""rlv3 must be reachable as a dev arm from the PAIRED instrument.

Why this file exists. rlv3 shipped 40 files and 364 passing tests while being
unmeasurable: `rl/session_eval.py` rejected `dev:rlv3:...` outright, so no rlv3
preset had ever been scored on the 995-game paired draw and every number in its
build report came from four in-process smoke episodes. `rlv3/arm.py` offered a
monkeypatch instead, which cannot work through the eval's Pool -- multiprocessing
starts `spawn` on this machine, so a patch applied in the parent is invisible in
the worker, and the worker raises "unknown learner_policy kind 'rlv3'".

These tests pin the two dispatch points and, critically, that the dispatch lives
in the MODULE rather than in a parent-process patch. `test_dispatch_survives_a_
fresh_interpreter` is the one that would have caught the original design: a
monkeypatch passes every in-process test and still fails in a spawn worker.

No games are played here.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_session_eval_parses_an_rlv3_dev_arm():
    """The `dev:` block must know rlv3, or no preset can ever be gated."""
    src = open(os.path.join(REPO, "rl", "session_eval.py")).read()
    assert '"rlv3"' in src, "rl/session_eval.py does not parse dev:rlv3:<preset>"
    # and the error message must advertise it, so the next person finds it
    assert "dev:rlv3:<preset>" in src


def test_rollout_dispatches_rlv3_before_the_shipped_kinds():
    from rl import rollout

    agent_fn, switched = rollout._make_dev_policy_agent({"kind": "rlv3",
                                                         "preset": "base"})
    assert callable(agent_fn)
    # liveness handles rlv3 promises; a fallback to the heuristic has none
    assert getattr(agent_fn, "provider", None) is not None
    assert agent_fn.provider.calls == 0          # no dawn yet
    if getattr(agent_fn, "restore", None):
        agent_fn.restore()


@pytest.mark.parametrize("spec", [{"kind": "champion"}])
def test_the_shipped_kinds_are_untouched(spec):
    from rl import rollout

    agent_fn, switched = rollout._make_dev_policy_agent(spec)
    assert callable(agent_fn)
    assert getattr(agent_fn, "provider", None) is None, (
        "a shipped kind was routed through rlv3")


def test_dispatch_survives_a_fresh_interpreter():
    """THE spawn test.

    `rl/session_eval.py` runs episodes through `multiprocessing.Pool`, which
    spawns a brand-new interpreter that imports `rl.rollout` from disk. A
    monkeypatch applied in the parent is gone. This runs the dispatch in a
    subprocess with nothing pre-imported, which is what a worker sees.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from rl import rollout;"
        "a, _ = rollout._make_dev_policy_agent({'kind': 'rlv3', 'preset': 'base'});"
        "print('OK' if getattr(a, 'provider', None) is not None else 'FELL_BACK')"
        % REPO
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout, out.stdout + out.stderr[-2000:]


def test_an_unknown_kind_still_raises():
    from rl import rollout

    with pytest.raises(Exception):
        rollout._make_dev_policy_agent({"kind": "not_a_real_kind"})


@pytest.mark.xfail(strict=True, reason=(
    "OPEN DEFECT, finding 17 in docs/results/rlv3_build_review.md. "
    "dev:rlv3:base is documented as the zero-dose identity control but matches "
    "neither dev:champion (3 of 12 paired games differ, once by $32,952) nor "
    "dev:knobs:rules_h8_ship2_hv (-1 win, -$4,549). The fix belongs in rlv3/ "
    "and is held while the ground-truth loop is editing that tree; remove this "
    "marker when arm.parse surfaces the flags."))
def test_a_preset_spec_carries_the_executor_flags_its_rules_assume():
    """The root cause of finding 17, at millisecond cost.

    `rl/rollout.py` applies executor flags from `spec["exec"]` or
    `spec["rules"]["exec"]`. `rlv3.arm.parse("base")` returns neither, so
    champ-h10's lastwater/h23/harvest_visit never apply -- while
    `Rlv3Config.load()` hands the KNOB half of those same rules to the
    allocator. The arm is then champ-h10's day plan under the SHIPPED
    executor: a third agent, and a control that controls for nothing.

    Why rlv3/tests missed it: they compare plan objects at dawn, and the plans
    are byte-identical on 48 mid-season boards across every attribute. The
    divergence is entirely in the executor, which no plan-level test can see.
    """
    from rlv3.agent import config_from_spec
    from rlv3.arm import parse

    spec = parse("base")
    rules = config_from_spec(spec).load() or {}
    wanted = list(rules.get("exec") or [])
    assert wanted, "the base preset's rules carry no exec flags; fixture is stale"

    applied = list(spec.get("exec") or (spec.get("rules") or {}).get("exec") or [])
    assert applied == wanted, (
        f"rl/rollout would apply {applied!r} but the rules assume {wanted!r}")
