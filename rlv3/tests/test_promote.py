"""The lexicographic objective, on the real frozen rows.

    PYTHONPATH=. uv run pytest rlv3/tests/test_promote.py -q

`rl/runs/draw995/batch2.json` is the 995-game band-stratified paired draw. It
is read, never written. No games are played.
"""

from __future__ import annotations

import json
import os

import pytest

from rlv3 import promote as PR

BATCH2 = os.path.join(PR.REPO, "rl", "runs", "draw995", "batch2.json")
pytestmark = pytest.mark.skipif(not os.path.exists(BATCH2),
                                reason="rl/runs/draw995/batch2.json absent")


def _rows(arm):
    return PR.load_rows(BATCH2, arm)


# ------------------------------------------------- the objective, as measured
def test_objective_reproduces_champ_h10():
    """champ-h10 is arm ship2_hv on this draw: 0.354 of 995, +$24,596 over the
    352 games it won, -$17,100 over the 643 it lost."""
    win, won, lost, n = PR.objective(_rows("ship2_hv"))
    assert n == 995
    assert win == pytest.approx(0.354, abs=0.001)
    assert won == pytest.approx(24596, abs=1)
    assert lost == pytest.approx(-17100, abs=1)


def test_objective_reproduces_champ_h8():
    win, won, lost, n = PR.objective(_rows("champ_h8"))
    assert n == 995
    assert win == pytest.approx(0.230, abs=0.001)
    assert won == pytest.approx(24283, abs=1)
    assert lost == pytest.approx(-23351, abs=1)


def test_per_band_reproduces_the_recorded_bands():
    """0.440 / 0.367 / 0.366 / 0.200 / 0.406 / 0.349 -- band 3 (1500-2000) is
    the hole, and a slice boundary must not move any band by more than its se."""
    bands = PR.per_band(_rows("ship2_hv"))
    got = [round(bands[b][0], 3) for b in sorted(bands)]
    assert got == [0.440, 0.367, 0.366, 0.200, 0.406, 0.349]
    assert sum(bands[b][1] for b in bands) == 995


def test_margin_is_relative_not_absolute():
    """The market is ONE shared inventory; banking well while the opponent
    banks more is a loss."""
    row = {"money": 90_000, "opp": 100_000, "win": 0.0}
    assert PR.margin(row) == -10_000


def test_win_threshold_matches_session_eval():
    from rl import session_eval
    rows = _rows("ship2_hv")
    mine = sum(1 for r in rows if float(r["win"]) >= PR.WIN)
    theirs = session_eval.paired(rows, rows)
    assert theirs["n"] == 995
    assert mine == round(PR.objective(rows)[0] * 995)


def test_the_two_comparators_agree_on_what_a_win_is():
    """rlv3.objective.better is the IN-GAME comparator over projected
    Outlooks; this module is the OFFLINE one over finished games. Same three
    clauses, different domains -- but one definition of a win, taken from the
    sibling rather than restated here."""
    PR._assert_win_convention()
    from rlv3 import objective as OBJ
    assert OBJ.win_of(10.0, 5.0) >= PR.WIN
    assert OBJ.win_of(5.0, 5.0) < PR.WIN      # a draw is not a win
    # and a row's own margin sign agrees with the win flag it was given
    for r in _rows("ship2_hv"):
        assert (float(r["win"]) >= PR.WIN) == (PR.margin(r) > 0)


# ----------------------------------------------------- rank 1: the win rate
def test_champ_h10_beats_champ_h8_on_rank_one():
    better, why = PR.lexicographic_better(_rows("ship2_hv"), _rows("champ_h8"))
    assert better and why.startswith("rank 1:")
    assert "+0.12" in why or "delta +0.1" in why


def test_champ_h8_does_not_beat_champ_h10():
    better, why = PR.lexicographic_better(_rows("champ_h8"), _rows("ship2_hv"))
    assert not better and "REFUSED" in why


def test_refuses_more_money_with_fewer_wins():
    """The exact failure mode of every learned component so far: our bank
    rises while the opponent's rises faster. ship2 banks MORE on the games it
    wins (+$26,303 vs +$24,596) and wins fewer of them (0.315 vs 0.354)."""
    cand, base = _rows("ship2"), _rows("ship2_hv")
    assert PR.objective(cand)[1] > PR.objective(base)[1], "premise: more money"
    assert PR.objective(cand)[0] < PR.objective(base)[0], "premise: fewer wins"
    better, why = PR.lexicographic_better(cand, base)
    assert not better and "rank 1 REFUSED" in why and "fewer wins" in why


def test_a_significant_win_gain_promotes_even_on_less_money():
    """ship2_hv wins more than ship2 AND banks less on its wins. Rank 1
    dominates: the extra wins promote it."""
    better, why = PR.lexicographic_better(_rows("ship2_hv"), _rows("ship2"))
    assert better and why.startswith("rank 1:")


# --------------------------------------------- ranks 2 and 3: only on a tie
def _synth(wins, won_margin, lost_margin, n=40):
    """n paired games with exactly `wins` wins at fixed margins."""
    out = []
    for i in range(n):
        w = i < wins
        out.append({"win": 1.0 if w else 0.0,
                    "money": 100_000 + (won_margin if w else lost_margin),
                    "opp": 100_000, "bin": i % 6, "rating": 1000.0,
                    "path": f"s{i}.json", "script_side": i % 2, "bad": False})
    return out


def test_rank_two_breaks_a_win_rate_tie_on_won_margin():
    base = _synth(10, 20_000, -20_000)
    cand = _synth(10, 30_000, -20_000)
    better, why = PR.lexicographic_better(cand, base)
    assert better and why.startswith("rank 2:")
    assert "tie inside noise" in why


def test_rank_two_refuses_a_worse_won_margin_on_a_tie():
    base = _synth(10, 30_000, -20_000)
    cand = _synth(10, 20_000, -20_000)
    better, why = PR.lexicographic_better(cand, base)
    assert not better and why.startswith("rank 2:")


def test_rank_three_breaks_a_tie_on_a_smaller_loss_margin():
    """When winning is tied and profit-when-winning is tied, the smaller loss
    wins: champ-h10's -$17,100 against champ-h8's -$23,351 is this clause."""
    base = _synth(10, 20_000, -23_351)
    cand = _synth(10, 20_000, -17_100)
    better, why = PR.lexicographic_better(cand, base)
    assert better and why.startswith("rank 3:")
    assert "lost-game margin" in why


def test_rank_three_refuses_a_bigger_loss_margin():
    base = _synth(10, 20_000, -17_100)
    cand = _synth(10, 20_000, -23_351)
    better, why = PR.lexicographic_better(cand, base)
    assert not better and why.startswith("rank 3:")


def test_an_insignificant_win_gain_falls_through_to_rank_two():
    """delta > 0 but McNemar cannot separate it: the win column is a tie
    inside noise, and rank 2 decides. ~+/-45 rating points of noise is this
    project's standing reason not to read a small delta as a result."""
    base = _synth(10, 20_000, -20_000)
    cand = _synth(11, 10_000, -20_000)         # one more win, much less money
    better, why = PR.lexicographic_better(cand, base, p_max=0.01)
    assert not better and why.startswith("rank 2:")
    # with a looser p, the same single win IS rank 1
    better, why = PR.lexicographic_better(cand, base, p_max=1.0)
    assert better and why.startswith("rank 1:")


def test_unpaired_rows_are_refused():
    with pytest.raises(ValueError):
        PR.lexicographic_better(_synth(5, 1, -1, n=40), _synth(5, 1, -1, n=39))


# ---------------------------------------------------------------- promotion
def _eval_file(tmp_path, cand, base, name="session_eval.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"seed": 0, "per_bin": 175, "arms": {},
                             "reused": {},
                             "results": {"cand": cand, "champ_h10": base}}))
    return str(p)


def _weights(path, heads_version=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"policy": {"arch": 2, "blocks": [], "out": [[], []],
                              "heads_version": heads_version,
                              "heads_mode": "absolute"}, "value": None}, fh)


FP_NOW = {"heads_version": 1, "heads_mode": "absolute", "board": False,
          "head_sizes_sha": "a" * 40, "n_action_logits": 57, "n_features": 168,
          "board_arm": "T_crop_103", "board_ch": 103,
          "corpus": {"draw_sha": "c" * 40}, "code": "d" * 40}


def test_promote_copies_the_winner_and_records_the_bands(tmp_path):
    run = str(tmp_path / "a1")
    ppo = os.path.join(run, "ppo")
    _weights(os.path.join(ppo, "eval.json"))
    with open(os.path.join(ppo, "last.pt"), "w") as fh:
        fh.write("w")
    ev = _eval_file(tmp_path, _synth(20, 20_000, -20_000),
                    _synth(10, 20_000, -20_000))

    path = PR.promote(run, ev, "cand", fp=FP_NOW)
    assert path and os.path.exists(path)
    assert os.path.exists(os.path.join(run, "best", "eval.json"))
    assert os.path.exists(os.path.join(run, "best", "session_eval.json"))

    from rlv3 import manifest as MF
    man = MF.load(run)
    assert man["best"]["beat"] == "champ_h10"
    assert man["gate"]["promoted"] is True
    assert len(man["gate"]["per_band"]) == 6
    assert man["gate"]["objective"][0] == pytest.approx(0.5)


def test_promote_returns_none_and_still_records_the_gate(tmp_path):
    run = str(tmp_path / "a1")
    ppo = os.path.join(run, "ppo")
    _weights(os.path.join(ppo, "eval.json"))
    ev = _eval_file(tmp_path, _synth(5, 90_000, -1_000),
                    _synth(20, 20_000, -20_000))
    assert PR.promote(run, ev, "cand", fp=FP_NOW) is None
    from rlv3 import manifest as MF
    man = MF.load(run)
    assert man["gate"]["promoted"] is False
    assert "REFUSED" in man["gate"]["verdict"]
    assert man.get("best") is None
    assert not os.path.exists(os.path.join(run, "best"))


def test_promote_refuses_a_crossed_head_version(tmp_path):
    """Those stored `choices` indices mean different land, crop and livestock
    targets (job_topboard.json run 3: goose_target [0,8,16,24,32] ->
    [0,4,6,10,16])."""
    from rlv3 import fingerprint as FP
    run = str(tmp_path / "a1")
    ppo = os.path.join(run, "ppo")
    _weights(os.path.join(ppo, "eval.json"), heads_version=2)
    ev = _eval_file(tmp_path, _synth(20, 20_000, -20_000),
                    _synth(10, 20_000, -20_000))
    with pytest.raises(FP.HeadMismatch):
        PR.promote(run, ev, "cand", fp=FP_NOW)


def test_promote_refuses_a_baseline_from_a_different_draw(tmp_path):
    """The 995-game draw already changed once (785 -> 995). Two win columns
    from two different world sets is the gate-cache-stale failure."""
    run = str(tmp_path / "a1")
    ppo = os.path.join(run, "ppo")
    _weights(os.path.join(ppo, "eval.json"))
    p = tmp_path / "session_eval.json"
    p.write_text(json.dumps({"results": {"cand": _synth(5, 1, -1, n=40)}}))
    other = tmp_path / "old.json"
    other.write_text(json.dumps({"results": {"champ_h10": _synth(5, 1, -1, n=30)}}))
    with pytest.raises(SystemExit) as e:
        PR.promote(run, str(p), "cand", baseline_arm="champ_h10",
                   baseline_json=str(other), fp=FP_NOW)
    assert "different --per-bin" in str(e.value) or "refusing" in str(e.value)


def test_report_line_names_all_six_bands():
    line = PR.report(_rows("ship2_hv"), "champ_h10")
    assert "0.354" in line and line.count("0.") >= 7
