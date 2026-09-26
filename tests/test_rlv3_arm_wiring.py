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
