"""Stage 2 wiring: the RL net decodes through rlv3.readout.commit (the gate's
readout arm), executor flags reach net jobs on both seats, and the trainer
can start from a BC readout checkpoint that carries no value net."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rlv3.tests.test_readout import _leader_dawn  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_features():
    from kagent import features
    v, m = features.HEADS_VERSION, features.MODE
    yield
    if features.HEADS_VERSION != v:
        features.set_heads_version(v)
    if features.MODE != m:
        features.set_mode(m)


def test_commit_decode_is_the_gates_readout_arm():
    """NeuralPolicy(decode="commit") at temperature 0 with the readout's own
    weights must produce the plan the readout arm commits: same masks (at the
    readout's horizon), same argmax, same commit."""
    pytest.importorskip("torch")
    from kagent import features, planner
    from kagent.gamestate import Memory
    from kagent.nnpolicy import NeuralPolicy
    from rl.common import to_layers
    from rlv3 import portfolio as PF, readout as RO
    if not os.path.exists(RO.DEFAULT_WEIGHTS):
        pytest.skip("readout weights not on disk")
    ro = RO.Readout()
    layers = to_layers(ro.net)
    assert layers["policy"].get("heads_version") == RO.VERSION
    for day in (3, 10, 18):
        view = _leader_dawn(day)
        mem = Memory()
        mem.observe(view)
        pred = ro.predict(view, mem)
        order = [r["item"] for r in PF.rank(view, 1.0, False)["rows"]]
        want, _ = RO.commit(planner.make_day_plan(view, mem, speculative=True), view, pred, order=order)
        mem2 = Memory()
        mem2.observe(view)
        pol = NeuralPolicy(layers, temperature=0.0, decode="commit")
        got = pol.make_plan(view, mem2)
        _x, masks, _c, _lp, _v = pol.last_decision
        assert masks == features.masks(view, horizon=RO.HORIZON), "records must carry the horizon masks"
        assert got.plant_quota == want.plant_quota, day
        assert got.animal_buys == want.animal_buys, day
        assert got.structure_builds == want.structure_builds, day
        assert got.hires_target == want.hires_target, day
        assert got.fert_budget == want.fert_budget, day


def test_plan_decode_is_unchanged_by_default():
    """The default decode is the legacy path: nothing an existing net arm
    plays through has moved."""
    from kagent.nnpolicy import NeuralPolicy
    pol = NeuralPolicy(None)
    assert pol.decode == "plan" and pol.apply is None


def test_job_exec_reaches_the_worker_and_is_restored():
    from kagent import scheduler
    from rl import rollout
    before = dict(scheduler.EXEC)
    prev = rollout._set_exec(["h23"])
    assert prev == before
    assert scheduler.EXEC.get("h23") is True
    assert not any(v for k, v in scheduler.EXEC.items() if k != "h23")
    scheduler.EXEC.update(prev)
    assert dict(scheduler.EXEC) == before


def test_decode_switch_is_read_from_the_environment(monkeypatch):
    from rl import rollout
    monkeypatch.delenv("KAGG_DECODE", raising=False)
    assert rollout._decode() == "plan"
    monkeypatch.setenv("KAGG_DECODE", "commit")
    assert rollout._decode() == "commit"


def test_apply_checkpoints_accepts_a_policy_only_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from kagent import features
    from rl import train as T
    from rl.common import PlannerNet, ValueNet
    features.set_heads_version(5)
    net, vnet = PlannerNet(features.N_FEATURES, board=False), ValueNet(features.N_FEATURES)
    p = tmp_path / "bc.pt"
    torch.save({"policy": net.state_dict()}, p)
    T.apply_checkpoints(net, vnet, resume=str(p))       # must not KeyError on "value"
    assert "exec" not in T.JOB_EXTRA
