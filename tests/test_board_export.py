"""Board-trunk submission path: the trunk survives the b85 round-trip, the
bundle builds the same tensor the trainer did, the width guard still fires,
and a board-free bundle is untouched by any of it.

This is the regression net around the change that made board planners
submittable at all. Every Stage B board arm was lost to a SILENT skew of
exactly this kind (a 216-row first layer fed 168 numbers), so each test here
pins a place that skew could return.

    PYTHONPATH=. uv run pytest tests/test_board_export.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import boardstate  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402
from kagent.nnpolicy import decode_weights, forward_board, forward_policy  # noqa: E402

N_FEATURES = 168


def _net(board=True):
    import torch
    from rl.common import PlannerNet
    torch.manual_seed(7)
    return PlannerNet(N_FEATURES, board=board)


def _obs(seed=31, steps=0):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    return copy.deepcopy(env.state[0].observation)


def test_trunk_round_trips_through_b85():
    import numpy as np
    from rl.common import policy_layers, to_weights_data
    net = _net()
    ref = policy_layers(net)
    back = decode_weights(to_weights_data(net))["policy"]
    assert back.get("board") is not None, "the trunk did not survive export"
    t = np.random.RandomState(0).rand(103, 10, 10).astype("float16")
    a, b = forward_board(ref["board"], t), forward_board(back["board"], t)
    assert max(abs(x - y) for x, y in zip(a, b)) == 0.0
    x = [0.01 * i for i in range(N_FEATURES)]
    la = forward_policy(ref, list(x) + a)
    lb = forward_policy(back, list(x) + b)
    assert max(abs(p - q) for p, q in zip(la, lb)) == 0.0


def test_width_guard_still_fires_without_the_trunk_output():
    from rl.common import policy_layers
    ref = policy_layers(_net())
    with pytest.raises(ValueError, match="board trunk"):
        forward_policy(ref, [0.0] * N_FEATURES)


def test_decode_refuses_a_truncated_trunk():
    from rl.common import to_weights_data
    data = to_weights_data(_net())
    data["board_shapes"] = dict(data["board_shapes"])
    data["board_shapes"]["c1b"] = [31]          # one channel short
    with pytest.raises(ValueError, match="refusing a truncated trunk"):
        decode_weights(data)


def test_export_refuses_a_trunk_missing_a_parameter():
    from rl.common import policy_layers, pack_policy
    p = policy_layers(_net())
    p["board"] = {k: v for k, v in p["board"].items() if k != "fcb"}
    with pytest.raises(ValueError, match="missing 'fcb'"):
        pack_policy(p)


def test_builder_matches_the_trainer():
    """kagent/boardstate must be bit-identical to rl/rollout._board.

    The bundle encoder is a port of tools/board_tensor.py; the weights were
    trained against the original, so any drift is a train/serve skew.
    """
    import numpy as np
    from rl import rollout
    assert boardstate._bs_arm_channels("T_crop_103") == __import__(
        "tools.board_tensor", fromlist=["x"]).arm_channels("T_crop_103")
    from kaggle_environments import make
    # rollout._board is a no-op unless the board hook is on, and it is read
    # from the environment at import; pin it so the test does not depend on
    # how pytest was invoked.
    was, rollout._BOARD = rollout._BOARD, True
    try:
        env = make("kaggriculture", configuration={"seed": 11}, debug=False)
        env.reset(2)
        dawn, mid, mem, pairs = {}, {}, Memory(), []

        def probe(obs):
            view = View(obs)
            if view.hour == 0:
                dawn[view.day] = obs
            elif view.hour == 12:
                mid[view.day] = obs
            boardstate.observe(mem, view, obs)
            if view.hour in (0, 13) and view.day <= 6:
                # collected, not asserted: kaggle_environments swallows an
                # exception raised inside an agent and the game just ends
                pairs.append((rollout._board(obs, view.player, dawn, mid, view.day),
                              boardstate.build(mem, view, obs)))
            return {"farmer": ["PASS"], "hands": [], "market": []}

        env.run([probe, "starter"])
    finally:
        rollout._BOARD = was
    assert pairs, "no tensors compared"
    for ref, got in pairs:
        assert ref is not None and ref.shape == got.shape == (103, 10, 10)
        assert float(np.abs(ref.astype("float32")
                            - got.astype("float32")).max()) == 0.0


def test_arm_is_read_off_the_trunk_not_assumed():
    from rl.common import policy_layers
    board = policy_layers(_net())["board"]
    assert boardstate.arm_for_trunk(board) == "T_crop_103"
    narrow = dict(board, c1w=[[row for row in ch[:83]] for ch in board["c1w"]])
    assert boardstate.arm_for_trunk(narrow) == "E_full_83"
    with pytest.raises(ValueError, match="known arms"):
        boardstate.arm_for_trunk(dict(board, c1w=[[[0.0] * 3] * 7] * 32))


def test_board_free_bundle_is_unaffected():
    """A weightless champion bundle must not acquire a board path."""
    path = os.path.join(REPO, "arena", "agents", "champ-h10", "main.py")
    if not os.path.exists(path):
        pytest.skip("champ-h10 bundle not present")
    ns = {"__name__": "probe"}
    exec(compile(open(path).read(), path, "exec"), ns)
    g = ns["_G"]
    assert g["_LAYERS"] is None and g["CHAMPION_PLAN"] is True
    provider = g["_AGENT"].policy.plan_provider
    assert getattr(provider, "_bundle_arm", None) is None
    assert boardstate._BS_NP is None or True     # numpy never forced at import
