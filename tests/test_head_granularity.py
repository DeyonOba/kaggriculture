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
