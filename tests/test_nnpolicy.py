"""Parity between the pure-python inference code and the torch models
(arch v2: GELU + LayerNorm policy, decoupled value net), plus an end-to-end
neural-policy episode smoke test."""

import math
import random

import pytest

torch = pytest.importorskip("torch")

from kagent import features, nnpolicy  # noqa: E402
from rl.common import (  # noqa: E402
    PlannerNet,
    ValueNet,
    masked_dists,
    to_layers,
    to_weights_data,
)

N_FEAT = 37  # arbitrary for parity test


def _rand_input(rng, n):
    return [rng.uniform(-1, 1) for _ in range(n)]


def test_policy_forward_parity():
    torch.manual_seed(0)
    net = PlannerNet(N_FEAT)
    layers = to_layers(net)
    rng = random.Random(1)
    for _ in range(5):
        x = _rand_input(rng, N_FEAT)
        logits_py = nnpolicy.forward_policy(layers["policy"], x)
        with torch.no_grad():
            logits_t = net(torch.tensor([x], dtype=torch.float32))
        for a, b in zip(logits_py, logits_t[0].tolist()):
            assert abs(a - b) < 1e-4


def test_value_forward_parity():
    torch.manual_seed(0)
    vnet = ValueNet(N_FEAT)
    layers = to_layers(PlannerNet(N_FEAT), vnet)
    rng = random.Random(2)
    for _ in range(5):
        x = _rand_input(rng, N_FEAT)
        v_py = nnpolicy.forward_value(layers["value"], x)
        with torch.no_grad():
            v_t = vnet(torch.tensor([x], dtype=torch.float32))[0].item()
        assert abs(v_py - v_t) < 1e-4


def test_logprob_parity_with_masked_dists():
    torch.manual_seed(0)
    net = PlannerNet(N_FEAT)
    layers = to_layers(net)
    rng = random.Random(3)
    x = _rand_input(rng, N_FEAT)
    head_masks = [[1] * n for n in features.HEAD_SIZES]
    head_masks[0][0] = 0  # exercise masking
    logits_py = nnpolicy.forward_policy(layers["policy"], x)
    choices, logprob_py = nnpolicy.choose(logits_py, head_masks, rng=random.Random(4))
    with torch.no_grad():
        logits_t = net(torch.tensor([x], dtype=torch.float32))
        m = torch.tensor([sum(head_masks, [])], dtype=torch.float32)
        dists = masked_dists(logits_t, m)
        lp = sum(
            d.log_prob(torch.tensor([c])).item() for d, c in zip(dists, choices)
        )
    assert abs(logprob_py - lp) < 1e-4


def test_weights_data_roundtrip():
    torch.manual_seed(0)
    net = PlannerNet(N_FEAT)
    data = to_weights_data(net)
    assert data["arch"] == 2
    decoded = nnpolicy.decode_weights(data)
    ref = to_layers(net)["policy"]
    rng = random.Random(5)
    for _ in range(3):
        x = _rand_input(rng, N_FEAT)
        a = nnpolicy.forward_policy(decoded["policy"], x)
        b = nnpolicy.forward_policy(ref, x)
        assert all(math.isclose(u, v, abs_tol=1e-5) for u, v in zip(a, b))


def test_neural_policy_full_episode():
    from kaggle_environments import make

    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy
    from rl import rollout

    probe = rollout.run_episode(
        {"weights_path": None, "seed": 3, "side": 0, "opponent": "starter", "mode": "bc"}
    )
    n_features = len(probe["records"][0]["x"])
    torch.manual_seed(0)
    layers = to_layers(PlannerNet(n_features), ValueNet(n_features))

    mem = Memory()
    pol = HeuristicPolicy(nnpolicy.NeuralPolicy(layers, rng=random.Random(0)))

    def agent(obs):
        view = View(obs)
        mem.observe(view)
        return pol.act(view, mem)

    env = make("kaggriculture", configuration={"seed": 3}, debug=False)
    env.run([agent, "starter"])
    final = env.steps[-1]
    assert final[0].status == "DONE"
    assert final[0].reward is not None