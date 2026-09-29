"""Every herd head must be able to express what the top players run.

Measured on 40 leader replays (2026-09-26): 4.6-5.9 geese standing at day 20,
~15-16 cows+sheep. A head that cannot represent those counts makes that line
of play unreachable for any learned planner, however good the model is.

    PYTHONPATH=. uv run pytest tests/test_head_granularity.py -q
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import features  # noqa: E402

LEADER_GEESE = (4.6, 5.9)          # min and max mean across the three leaders


def _vals(mode, name):
    prev = features.MODE
    try:
        features.set_mode(mode)
        return dict(features.HEADS)[name]
    finally:
        features.set_mode(prev)


def test_goose_head_can_express_the_leaders_herd():
    for mode in ("absolute", "residual"):
        vals = _vals(mode, "goose_target")
        assert 0 in vals, f"{mode}: must still be able to run no geese"
        # some option within one goose of what the leaders actually keep
        reachable = [v for v in vals if LEADER_GEESE[0] - 1 <= v <= LEADER_GEESE[1] + 1]
        assert reachable, f"{mode}: goose head {vals} cannot express {LEADER_GEESE} geese"


def test_index_zero_is_still_no_geese():
    # a trained net that always picked index 0 keeps meaning "no geese"
    assert _vals("absolute", "goose_target")[0] == 0
    assert _vals("residual", "goose_target")[0] == 0


def test_head_widths_unchanged_so_trained_nets_still_load():
    assert len(_vals("absolute", "goose_target")) == 5
    assert len(_vals("residual", "goose_target")) == 3


def test_n_features_follows_the_heads_version_not_the_last_encode():
    """features.N_FEATURES is a cache encode() fills from its LAST call. It
    must be re-pointed on every set_heads_version, or a switch back leaves the
    previous version's width behind: an encode at v4 (196) then
    set_heads_version(1) left it at 196, and rlv3.fingerprint.probe_n_features
    -- which reads the alias directly -- stamped "v1 / nfeat=196", a state no
    net was ever trained in, and marked a valid PPO run stale. Found by an
    order-dependent test failure; would have hit any process that plays a v4
    readout arm and then a v1 net."""
    import json
    import os
    from kagent import features as F
    from kagent.gamestate import Memory, View

    rep = sorted(__import__("glob").glob(os.path.join(REPO, "replays", "mmpq", "*.json")))
    if not rep:
        pytest.skip("no leader replay on disk")
    steps = json.load(open(rep[0]))["steps"]
    obs = dict(steps[6 * 24][0]["observation"]); obs.setdefault("step", 6 * 24)
    v = View(obs); m = Memory(); m.observe(v)
    prev = F.HEADS_VERSION
    try:
        F.set_heads_version(1); w1 = len(F.encode(v, m))
        F.set_heads_version(4); w4 = len(F.encode(v, m))
        assert w4 != w1, "v4 must widen the encoder or this test proves nothing"
        assert F.N_FEATURES == w4
        F.set_heads_version(1)
        assert F.N_FEATURES == w1, f"N_FEATURES stayed at {F.N_FEATURES} after switching back to v1"
    finally:
        F.set_heads_version(prev)
