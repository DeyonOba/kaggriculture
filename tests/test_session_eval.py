"""rl/session_eval.py: the draw is seeded and reusable, the reuse check
refuses a different draw, and the paired statistics are what they claim.

    PYTHONPATH=. uv run pytest tests/test_session_eval.py -q
"""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl import session_eval as se  # noqa: E402

IDX = os.path.join(REPO, "rl", "datasets", "session_index.json")
pytestmark = pytest.mark.skipif(not os.path.exists(IDX), reason="no session index")


def _rows(entries, win=0.0):
    return [{"win": win, "money": 1.0, "opp": 2.0, "bin": b, "rating": r,
             "path": p, "script_side": ss, "bad": False} for p, ss, b, r in entries]


def test_draw_is_seeded_held_out_and_capped():
    a = se.held_out_entries(IDX, seed=0, per_bin=5)
    b = se.held_out_entries(IDX, seed=0, per_bin=5)
    assert [e[:2] for e in a] == [e[:2] for e in b], "same seed, same games"
    c = se.held_out_entries(IDX, seed=1, per_bin=5)
    assert [e[:2] for e in a] != [e[:2] for e in c], "different seed, different games"
    for p, ss, bin_, r in a:
        assert int(os.path.basename(p).split("-")[-1][:-5]) % 5 == 0, "held-out only"
        lo, hi = se.EDGES[bin_]
        assert lo <= r < hi
    assert all(sum(1 for e in a if e[2] == b_) <= 5 for b_ in range(6))


def test_reuse_accepts_the_same_draw_and_refuses_another():
    entries = se.held_out_entries(IDX, seed=0, per_bin=3)
    rows = _rows(entries)
    assert se.reuse_results(rows, entries) is rows
    other = _rows(se.held_out_entries(IDX, seed=1, per_bin=3))
    with pytest.raises(SystemExit, match="not the same draw"):
        se.reuse_results(other, entries)
    with pytest.raises(SystemExit, match="prior games"):
        se.reuse_results(rows[:-1], entries)
    flipped = list(rows)
    flipped[0] = dict(rows[0], script_side=1 - rows[0]["script_side"])
    with pytest.raises(SystemExit, match="seat"):
        se.reuse_results(flipped, entries)


def test_mcnemar_and_paired_delta():
    assert se.binom_two_sided(0, 0) == 1.0
    assert se.binom_two_sided(3, 13) == pytest.approx(0.0923, abs=1e-4)
    assert se.binom_two_sided(0, 10) == pytest.approx(2 / 1024)
    base = [{"win": w} for w in [1, 0, 0, 1, 0, 1, 0, 0]]
    arm = [{"win": w} for w in [1, 1, 0, 1, 1, 1, 0, 0]]
    p = se.paired(base, arm)
    assert p["b_arm_only"] == 2 and p["c_base_only"] == 0 and p["n"] == 8
    assert p["delta"] == pytest.approx(0.25)
    assert p["p_mcnemar"] == pytest.approx(0.5)
    assert math.isfinite(p["se"]) and p["se"] > 0
