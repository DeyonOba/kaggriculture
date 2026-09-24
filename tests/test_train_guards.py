"""Guards on rl/train.py that turn Stage B's silent failures into refusals.

Every one of these is a failure that happened, produced twenty iterations of
plausible-looking output, and was found by reading checkpoints afterwards:

  - the worker and the trainer ran DIFFERENT policies (no trunk serialised),
    so the KL early stop fired on the first minibatch and no update ran;
  - --resume loaded a BC-trained trunk over the --trunk-init one;
  - BC trained the pretrained trunk toward a board-blind teacher;
  - the terminal reward was money-and-margin with the win a few percent of it.

    PYTHONPATH=. uv run pytest tests/test_train_guards.py -q
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl import train  # noqa: E402
from rl.common import BoardTrunk, PlannerNet, ValueNet  # noqa: E402


def _ep(n=3, win=1.0, final=50_000.0, opp=20_000.0, session=True):
    recs = [{"pv": 10_000.0 * (i + 1), "value": 0.0, "stock": 0.0} for i in range(n)]
    return {"records": recs, "final_money": final, "opp_money": opp,
            "win": win, "is_session": session}


def _terminal(ep, monkeypatch, reward):
    """The terminal reward alone: V = 0 everywhere and no shaping, so the
    last return IS the last reward."""
    monkeypatch.setattr(train, "REWARD", reward)
    _, ret = train._episode_advantages(ep, shaping_w=0.0)
    return ret[-1]


# ------------------------------------------------------------- the reward

def test_win_reward_is_win_first(monkeypatch):
    wc = train.SESSION_WIN_COEF
    blowout_win = _terminal(_ep(win=1.0, final=200_000.0, opp=0.0), monkeypatch, "win")
    blowout_loss = _terminal(_ep(win=0.0, final=0.0, opp=200_000.0), monkeypatch, "win")
    narrow_win = _terminal(_ep(win=1.0, final=20_001.0, opp=20_000.0), monkeypatch, "win")
    # sign follows the win
    assert blowout_win > 0 > blowout_loss
    # a $200k margin adds at most MARGIN_TIE on top of the win coefficient
    assert abs(blowout_win - wc) <= train.MARGIN_TIE + 1e-9
    assert abs(blowout_loss + wc) <= train.MARGIN_TIE + 1e-9
    # a $1 win still beats a $200k loss by at least a full win coefficient
    assert narrow_win - blowout_loss >= wc
    # and money does not enter at all: same margin, different bank, same reward
    a = _terminal(_ep(win=1.0, final=30_000.0, opp=25_000.0), monkeypatch, "win")
    b = _terminal(_ep(win=1.0, final=130_000.0, opp=125_000.0), monkeypatch, "win")
    assert a == pytest.approx(b)


def test_legacy_reward_unchanged(monkeypatch):
    """--reward legacy must still be the pre-existing mix, byte for byte."""
    ep = _ep(win=1.0, final=50_000.0, opp=20_000.0)
    r = _terminal(ep, monkeypatch, "legacy")
    expect = (train.MONEY_COEF * 50_000.0 + train.SESSION_WIN_COEF * 1.0
              + train.MARGIN_COEF * 30_000.0)
    assert r == pytest.approx(expect)


def test_legacy_reward_is_not_win_first(monkeypatch):
    """The documented reason for --reward win: under legacy a blowout LOSS can
    out-reward a narrow WIN."""
    narrow_win = _terminal(_ep(win=1.0, final=20_001.0, opp=20_000.0), monkeypatch, "legacy")
    rich_loss = _terminal(_ep(win=0.0, final=120_000.0, opp=121_000.0), monkeypatch, "legacy")
    assert rich_loss > narrow_win


# ----------------------------------------------------- on-policy identity

def test_onpolicy_check_passes_on_float_noise():
    old = torch.randn(512)
    now = old + 1e-6 * torch.randn(512)
    assert abs(train._onpolicy_check(old, now, it=0)) < train.ONPOLICY_KL_TOL


def test_onpolicy_check_refuses_a_different_policy():
    old = torch.randn(512)
    now = old - 0.38            # the FROZEN arm's measured gap, the smallest of the five
    with pytest.raises(RuntimeError, match="disagree"):
        train._onpolicy_check(old, now, it=4)


# ----------------------------------------------------------- checkpoints

def test_trunk_init_survives_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGG_BOARD_ARM", "T_crop_103")
    torch.manual_seed(0)
    net, vnet = PlannerNet(12, board=True), ValueNet(12)
    other = PlannerNet(12, board=True)            # a "BC checkpoint" with its own trunk
    torch.save({"policy": other.state_dict(), "value": vnet.state_dict()}, tmp_path / "bc.pt")
    pre = BoardTrunk()
    torch.save({"trunk": pre.state_dict(), "arm": "T_crop_103", "val_r2": 0.4},
               tmp_path / "trunk.pt")

    info = train.apply_checkpoints(net, vnet, resume=str(tmp_path / "bc.pt"),
                                   trunk_init=str(tmp_path / "trunk.pt"),
                                   freeze_trunk=True)
    got = net.state_dict()
    for k, v in pre.state_dict().items():
        assert torch.equal(got[f"board.{k}"], v), "the trunk must be the FILE's"
    for k, v in other.state_dict().items():
        if not k.startswith("board."):
            assert torch.equal(got[k], v), "everything else must be the resume's"
    assert info["trunk_dist"] == 0.0
    assert info["resume_trunk_dist"] > 0.0, "the log must show bc.pt's trunk differed"
    assert all(not p.requires_grad for p in net.board.parameters())


def test_trunk_init_refuses_wrong_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGG_BOARD_ARM", "T_crop_103")
    net, vnet = PlannerNet(12, board=True), ValueNet(12)
    torch.save({"trunk": BoardTrunk().state_dict(), "arm": "E_full_83"}, tmp_path / "t.pt")
    with pytest.raises(SystemExit):
        train.apply_checkpoints(net, vnet, trunk_init=str(tmp_path / "t.pt"))


def test_bc_params_exclude_the_trunk():
    net = PlannerNet(12, board=True)
    trunk = {id(p) for p in net.board.parameters()}
    ps = train._bc_params(net)
    assert ps and not any(id(p) in trunk for p in ps)
    assert len(ps) + len(trunk) == len(list(net.parameters()))
    # and a boardless net is untouched
    plain = PlannerNet(12)
    assert len(train._bc_params(plain)) == len(list(plain.parameters()))


# ------------------------------------------------------------ curriculum
#
# Stage B2's frontier curriculum kept ~35% of episodes on 0-600 tapes and
# ~35-42% on the heuristic while the eval was 89% opponents rated 1000+; the
# board arms beat the training mix and lost held out. These pin the draw.

def _fake_pool():
    """Six bins of (path, script_side, bin), the 600-1000 bin small."""
    sizes = [40, 15, 300, 450, 700, 1600]
    bins = [[(f"s{b}_{i}.json", i % 2, b) for i in range(n)] for b, n in enumerate(sizes)]
    return [e for b in bins for e in b], bins


def _draw_many(n, session_mix, session_frac, band_w=None, league=(), snapshots=()):
    import random
    entries, bins = _fake_pool()
    rng = random.Random(0)
    bin_wins = [[1, 2] for _ in bins]
    bw = band_w or [1 / 6] * 6
    return [train._draw_opponent(rng, rng.random(), entries, bins, [], bin_wins, bw,
                                 session_mix, session_frac, (), list(league), list(snapshots))
            for _ in range(n)]


def _is_session(o):
    return isinstance(o, tuple) and o[0] == "session"


def test_eval_mix_matches_band_weights():
    bw = [0.082, 0.027, 0.223, 0.223, 0.223, 0.223]
    draws = _draw_many(20_000, "eval", 0.9, band_w=bw)
    sess = [o for o, s in draws if _is_session(o)]
    assert abs(len(sess) / len(draws) - 0.9) < 0.01
    got = [int(o[1][1]) for o in sess]                    # bin, from the fake path
    for b in range(6):
        assert abs(got.count(b) / len(got) - bw[b]) < 0.012, (b, got.count(b) / len(got))
    for o, s in draws:                                     # learner sits OPPOSITE the tape
        if _is_session(o):
            assert s == 1 - o[2]
    assert all(o == "heuristic" for o, s in draws if not _is_session(o))


def test_frontier_mix_reproduces_the_legacy_proportions():
    draws = _draw_many(20_000, "frontier", 0.5)
    sess = [o for o, s in draws if _is_session(o)]
    assert abs(len(sess) / len(draws) - 0.5) < 0.01
    # bin_wins starts at 1/2 < 0.55, so bin 0 is the frontier and takes 70% of
    # session draws plus its share of the 30% drawn from anywhere
    got = [int(o[1][1]) for o in sess]
    assert abs(got.count(0) / len(got) - (0.7 + 0.3 * 40 / 3105)) < 0.015
    assert all(o == "heuristic" for o, s in draws if not _is_session(o))


def test_remainder_keeps_the_legacy_split_at_any_session_frac():
    """league / heuristic / snapshots keep 56 / 28 / 16 of the non-session
    remainder (bundles' 40 falls through when none are configured) whether
    that remainder is 50% or 10% of episodes."""
    for frac in (0.5, 0.9):
        draws = _draw_many(40_000, "eval", frac, league=["L"], snapshots=["S"])
        rest = [o for o, s in draws if not _is_session(o)]
        assert abs(len(rest) / len(draws) - (1 - frac)) < 0.01
        kinds = [o if isinstance(o, str) else o[0] for o in rest]
        n = len(kinds)
        assert abs(kinds.count("league") / n - 0.56) < 0.02
        assert abs(kinds.count("heuristic") / n - 0.28) < 0.02
        assert abs(kinds.count("snapshot") / n - 0.16) < 0.02


def test_eval_band_weights_come_from_the_paired_eval_draw():
    w = train._eval_band_weights(os.path.join(REPO, "rl", "datasets", "session_index.json"))
    assert len(w) == 6 and abs(sum(w) - 1.0) < 1e-9
    upper = w[2:]
    assert max(upper) - min(upper) < 1e-9, "the four 1000+ bins are drawn at the cap"
    assert w[0] <= upper[0] and w[1] <= upper[0], "the two small bins are taken whole"
