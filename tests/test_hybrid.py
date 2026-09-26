"""The hybrid arm: an injectable DAY PLAN under the learned executor.

Every `dev:execnet:` arm string that worked before must still parse to
exactly what it parsed to before (two queued evals depend on it), and the
new form -- a to_layers weights JSON where the rules JSON used to go --
must select a neural planner.

    PYTHONPATH=. uv run pytest tests/test_hybrid.py -q
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl.execnet.policy import PLANNERS, make_plan_provider, parse_arm_fields  # noqa: E402

EXECNET = "rl/datasets/execnet/execnet.pt"
RULES = "tools/rules/rules_h8.json"
KERNEL = "rl/runs/kaggle_topboard/out/out/top_board_s1234/eval.json"


def test_existing_arm_strings_are_unchanged():
    # bare weights -> champion plan, no planner key, defaults elsewhere
    spec = parse_arm_fields([EXECNET])
    assert spec["kind"] == "execnet" and spec["weights"] == os.path.abspath(EXECNET)
    assert "planner" not in spec and "rules" not in spec
    assert "conf" not in spec and "mode" not in spec

    # the form both queued evals use: weights:rules:conf:mode
    spec = parse_arm_fields([EXECNET, RULES, "0.4", "dest"])
    assert spec["planner"] == "knobs"
    assert spec["rules"] == json.load(open(os.path.join(REPO, RULES)))
    assert spec["conf"] == 0.4 and spec["mode"] == "dest"

    # empty fields stay skipped
    spec = parse_arm_fields([EXECNET, "", "0.5"])
    assert "rules" not in spec and spec["conf"] == 0.5


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, KERNEL)), reason="kernel weights absent")
def test_planner_weights_are_detected_by_content():
    spec = parse_arm_fields([EXECNET, KERNEL, "0.4", "dest"])
    assert spec["planner"] == "weights"
    assert spec["planner_weights"] == os.path.abspath(KERNEL)
    assert "rules" not in spec
    assert spec["conf"] == 0.4 and spec["mode"] == "dest"


def test_champion_provider_is_the_default():
    from kagent.policy import ChampionPlanProvider
    provider, switched, needs_board = make_plan_provider()
    assert isinstance(provider, ChampionPlanProvider)
    assert switched is False and needs_board is False


def test_knobs_provider_builds_the_bandit():
    from kagent import features
    from kagent.banditpolicy import KnobBanditPolicy
    prev = features.MODE
    try:
        rules = json.load(open(os.path.join(REPO, RULES)))
        provider, _, needs_board = make_plan_provider("knobs", rules=rules)
        assert isinstance(provider, KnobBanditPolicy) and needs_board is False
        assert features.MODE == "residual"
    finally:
        if features.MODE != prev:
            features.set_mode(prev)


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, KERNEL)), reason="kernel weights absent")
def test_weights_provider_reports_its_board_trunk():
    from kagent.nnpolicy import NeuralPolicy
    provider, _, needs_board = make_plan_provider("weights", weights=os.path.join(REPO, KERNEL))
    assert isinstance(provider, NeuralPolicy)
    assert provider.temperature == 0.0
    # the kernel net carries a trunk, so the caller must feed provider.board
    assert needs_board is True


def test_unknown_planner_is_refused():
    with pytest.raises(ValueError):
        make_plan_provider("nosuchplanner")
    assert "weights" in PLANNERS and "champion" in PLANNERS
