"""execnet: the legality mask follows the verified engine rules, the
features have the declared shapes, the label vocabulary round-trips, the
dataset builder produces aligned arrays from a replay, and the policy only
overrides with legal ops from its take-over set.

    PYTHONPATH=. uv run pytest tests/test_execnet.py -q
"""
import copy
import glob
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent.gamestate import View  # noqa: E402
from rl.execnet import features as F  # noqa: E402


def _obs(hour=6, day=12):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 9}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, hour, day * 24 + hour
    farm = obs["farms"][0]
    farm["farmer"] = [0, 0]
    farm["hands"] = [[4, 4], [2, 2]]
    obs["private"]["inventories"] = [{}, {"WHEAT": 3, "STRAWBERRY": 2}, {"FERTILIZER": 1}]
    obs["private"]["seeds"] = {"WHEAT": 5}
    obs["private"]["shed"] = {"WHEAT": 10}
    farm["tiles"][2][2] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 9, "watered_today": False,
                           "consecutive_unwatered": 0, "yield_units": 2, "max_lifespan_step": -1,
                           "fertilized_until_day": -1}
    farm["tiles"][0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": False,
                           "fertilizer_available": True, "yield_units": 1, "consecutive_unfed": 0}
    return obs


def _mask(view, ui):
    m = F.legal_mask(view, ui)
    return {op for op, ok in zip(F.OPS, m) if ok}


def test_mask_follows_the_engine_rules():
    view = View(_obs())
    corner = _mask(view, 0)                       # farmer at (0,0) on a bare tile, empty pocket
    assert "NORTH" not in corner and "WEST" not in corner and {"SOUTH", "EAST", "PASS"} <= corner
    assert "PLANT" in corner and "BUILD_COOP" in corner and "DIG" not in corner
    assert "DROP" not in corner and "PICKUP" not in corner and "WATER" not in corner
    shed = _mask(view, 1)                         # hand at (4,4): shed-adjacent, bare, carrying
    assert {"DROP", "PICKUP", "PLACE", "PLANT"} <= shed and "FEED" not in shed
    plant = _mask(view, 2)                        # hand at (2,2) on a dry wheat, age 3, 2 units, fertilizer in pocket
    assert {"WATER", "HARVEST", "FERTILIZE", "DIG"} <= plant and "PLANT" not in plant and "FEED" not in plant
    # an unfed cow with fertilizer waiting: FEED needs wheat in the pocket
    obs = _obs()
    obs["farms"][0]["farmer"] = [1, 0]
    view = View(obs)
    cow = _mask(view, 0)
    assert {"CARE", "COLLECT_FERTILIZER", "HARVEST"} <= cow and "FEED" not in cow and "DIG" not in cow
    obs["private"]["inventories"][0] = {"WHEAT": 1}
    assert "FEED" in _mask(View(obs), 0)
    # locked tile: only moves, PASS and (if shed-adjacent) shed ops
    obs["farms"][0]["farmer"] = [7, 7]
    assert obs["farms"][0]["tiles"][7][7] == "LOCKED"
    assert _mask(View(obs), 0) == {"NORTH", "SOUTH", "EAST", "WEST", "PASS"}


def test_feature_shapes_and_labels():
    view = View(_obs())
    counts = F.farm_counts(view)
    for ui in range(3):
        t, s = F.unit_features(view, ui, counts)
        assert len(t) == F.N_TILE and len(s) == F.N_SCALAR
        assert all(0.0 <= v <= 1.0 for v in t) and all(0.0 <= v <= 1.0 for v in s)
    assert F.label_of(["WATER"]) == F.OP_INDEX["WATER"] and F.label_of(["PLACE", "COW"]) == F.OP_INDEX["PLACE"]
    assert F.label_of(None) == F.OP_INDEX["PASS"] and F.label_of(["NOSUCHOP"]) == -1
    ops = F.unit_ops({"farmer": ["NORTH"], "hands": [["FEED"]]}, 3)
    assert ops == [["NORTH"], ["FEED"], None]


def test_dataset_from_a_replay(tmp_path):
    from rl.execnet import dataset
    files = sorted(glob.glob(os.path.join(REPO, "replays", "mmpq", "*.json")))
    if not files:
        pytest.skip("no MMPQ replays")
    out = str(tmp_path / "g.npz")
    path, n = dataset.build_replay((files[0], 1, out))
    z = np.load(out)
    assert n > 1000 and len(z["y"]) == n
    assert z["tiles"].shape == (n, F.N_TILE) and z["scal"].shape == (n, F.N_SCALAR)
    assert z["mask"].shape == (n, len(F.OPS)) and z["ours"].shape == (n,)
    # every recorded op is legal under our mask (the engine would not have discarded it)
    legal = z["mask"][np.arange(n), z["y"].astype(int)]
    assert legal.mean() > 0.97, legal.mean()
    assert (z["ours"] >= 0).mean() > 0.9


def test_policy_overrides_only_legal_takeover_ops():
    torch = pytest.importorskip("torch")
    from kagent.gamestate import Memory
    from kagent.policy import ChampionPlanProvider
    from rl.execnet.model import ExecNet
    from rl.execnet.policy import ExecNetPolicy, OVERRIDE

    torch.manual_seed(0)
    pol = ExecNetPolicy(ExecNet(hidden=32, tile_hidden=32, scal_hidden=16), ChampionPlanProvider(), conf=0.0)
    mem = Memory()
    obs = _obs()
    view = View(obs)
    mem.observe(view)
    action = pol.act(view, mem)
    ops = [action["farmer"]] + action["hands"]
    assert len(ops) == 3
    for ui, op in enumerate(ops):
        assert op and op[0] in set(F.OPS)
        legal = _mask(view, ui)
        assert op[0] in legal or op[0] not in OVERRIDE, (ui, op)
    assert sum(pol.stats.values()) == 3
