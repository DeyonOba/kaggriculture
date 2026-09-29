"""The anti-stale layer: a stamp round-trip, three ways of going stale, the
quarantine, and the head-version guard.

    PYTHONPATH=. uv run pytest rlv3/tests/test_fingerprint.py -q

No games are played and the real 4MB session_index.json is touched once, for
the live-fingerprint smoke test only.
"""

from __future__ import annotations

import json
import os

import pytest

from rlv3 import fingerprint as FP
from rlv3 import stages as ST


def _fp(**over):
    """A synthetic fingerprint with today's shape but fixed values."""
    fp = {
        "schema": 1,
        "heads_version": 1, "heads_mode": "absolute",
        "head_sizes_sha": "a" * 40, "n_action_logits": 57,
        "n_features": 168, "board_arm": "T_crop_103", "board_ch": 103,
        "board": False,
        "corpus": {"index_sha": "b" * 40, "seed": 0, "per_bin": 175,
                   "counts": [64, 21, 175, 175, 175, 175], "n": 785,
                   "draw_sha": "c" * 40},
        "code": "d" * 40, "engine": "1.32.7",
    }
    fp.update(over)
    return fp


def _artifact(tmp_path, name="bc.pt", body="weights"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


# ------------------------------------------------------------- round trip
def test_stamp_round_trip(tmp_path):
    fp = _fp()
    art = _artifact(tmp_path)
    keys = FP.ACTION_KEYS + FP.BOARD_KEYS
    stamp = FP.write_stamp(art, fp, keys, extra={"target_iters": 24})

    assert os.path.exists(FP.stamp_path(art))
    assert stamp["keys"] == list(keys)
    assert stamp["bytes"] == len("weights")
    assert stamp["extra"]["target_iters"] == 24
    assert FP.read_stamp(art)["fp"]["heads_version"] == 1

    fresh, why = FP.check(art, fp, keys)
    assert fresh and why == "fresh"


def test_missing_stamp_is_never_fresh(tmp_path):
    art = _artifact(tmp_path)
    fresh, why = FP.check(art, _fp(), FP.ACTION_KEYS)
    assert not fresh and why == "unstamped"


def test_missing_artifact(tmp_path):
    fresh, why = FP.check(str(tmp_path / "nope.pt"), _fp(), FP.ACTION_KEYS)
    assert not fresh and why == "missing"


# ----------------------------------------------------- three ways to go stale
@pytest.mark.parametrize("field,newval,needle", [
    ("heads_version", 2, "heads_version"),
    ("head_sizes_sha", "z" * 40, "head_sizes_sha"),
])
def test_action_space_change_is_stale(tmp_path, field, newval, needle):
    """job_topboard.json's run-3: goose_target [0,8,16,24,32] ->
    [0,4,6,10,16] made every stored `choices` index mean a different number of
    geese. head_sizes_sha catches that in milliseconds."""
    fp = _fp()
    art = _artifact(tmp_path)
    FP.write_stamp(art, fp, FP.ACTION_KEYS)
    fresh, why = FP.check(art, _fp(**{field: newval}), FP.ACTION_KEYS)
    assert not fresh and needle in why


def test_corpus_change_is_stale(tmp_path):
    """The 995-game draw already changed once (785 -> 995 on 26 September)."""
    fp = _fp()
    art = _artifact(tmp_path, "session_bc_top.jsonl", "{}\n")
    keys = FP.ACTION_KEYS + ("corpus",)
    FP.write_stamp(art, fp, keys)
    grown = _fp()
    grown["corpus"] = dict(grown["corpus"], n=995, draw_sha="e" * 40)
    fresh, why = FP.check(art, grown, keys)
    assert not fresh and "corpus changed" in why


def test_content_change_is_stale(tmp_path):
    fp = _fp()
    art = _artifact(tmp_path)
    FP.write_stamp(art, fp, FP.ACTION_KEYS)
    with open(art, "w") as fh:
        fh.write("someone rewrote this between slices")
    fresh, why = FP.check(art, fp, FP.ACTION_KEYS)
    assert not fresh and "content changed" in why


def test_a_stamp_that_does_not_cover_a_key_is_not_evidence(tmp_path):
    """A stamp written under a narrower key set cannot vouch for a wider one:
    silence about `board` is not the same as agreement about it."""
    fp = _fp()
    art = _artifact(tmp_path)
    FP.write_stamp(art, fp, FP.ACTION_KEYS)
    fresh, why = FP.check(art, fp, FP.ACTION_KEYS + FP.BOARD_KEYS)
    assert not fresh and "does not cover" in why and "board" in why


def test_stage_only_declaring_action_ignores_a_code_edit(tmp_path):
    """The gate declares [action, corpus] and not [code], so an unrelated
    source edit must not throw away an hour of eval."""
    fp = _fp()
    art = _artifact(tmp_path, "session_eval.json", "{}")
    keys = FP.ACTION_KEYS + ("corpus",)
    FP.write_stamp(art, fp, keys)
    assert FP.check(art, _fp(code="f" * 40), keys)[0]


# ------------------------------------------------------------- quarantine
def test_stale_artifact_is_quarantined_and_the_stage_runs(tmp_path):
    """The whole point: a skip_if hit under a changed fingerprint turns into
    quarantine-and-run, not into a silent reuse (kaggriculture-gate-cache-stale)."""
    run = tmp_path / "run"
    out = run / "out"
    out.mkdir(parents=True)
    art = out / "session_bc_top.jsonl"
    art.write_text("{}\n")

    stage = ST.StageSpec(
        name="clone", cmd="py rl/session_bc.py --out {out}/session_bc_top.jsonl",
        depends=("action", "corpus"),
        skip_if=("{out}/session_bc_top.jsonl",),
        produces=("{out}/session_bc_top.jsonl",))
    fmt = dict(procs=2, out=str(out), repo=str(tmp_path), python="py", run=str(run))
    # stamped with the stage's OWN key set: a stamp that does not cover a key
    # the stage depends on is not evidence about that key, and check() says so.
    FP.write_stamp(str(art), _fp(), stage.depend_keys)

    # unchanged: skipped as fresh
    res = ST.resolve(stage, fmt, 40_000, _fp(), str(run))
    assert res.cmd is None and res.reason.startswith("skip: fresh")
    assert art.exists()

    # head version bumped: quarantined into <run>/stale and the stage runs
    res = ST.resolve(stage, fmt, 40_000, _fp(heads_version=2), str(run))
    assert res.cmd is not None, "a stale output must not be honoured as a skip"
    assert "stale" in res.reason and "heads_version" in res.reason
    assert not art.exists(), "the stale artifact was left in place"
    moved = [p for p in (run / "stale").rglob("session_bc_top.jsonl")]
    assert moved, "the stale artifact was not quarantined"
    assert (moved[0].parent / "reason.txt").exists()
    assert os.path.exists(FP.stamp_path(str(moved[0]))), "the stamp travels with it"


def test_unstamped_output_is_not_honoured_as_a_skip(tmp_path):
    run = tmp_path / "run"
    out = run / "out"
    out.mkdir(parents=True)
    (out / "execnet2.pt").write_text("w")
    stage = ST.StageSpec(name="maia2", cmd="py rl/execnet/train2.py",
                         depends=("code",), skip_if=("{out}/execnet2.pt",))
    fmt = dict(procs=2, out=str(out), repo=str(tmp_path), python="py", run=str(run))
    res = ST.resolve(stage, fmt, 40_000, _fp(), str(run))
    assert res.cmd is not None and "unstamped" in res.reason


# ------------------------------------------------------- head-version guard
def _weights(tmp_path, name="eval.json", heads_version=1, mode="absolute", board=None):
    pol = {"arch": 2, "blocks": [], "out": [[], []],
           "heads_version": heads_version, "heads_mode": mode}
    if board is not None:
        pol["board"] = board
    p = tmp_path / name
    p.write_text(json.dumps({"policy": pol, "value": None}))
    return str(p)


def test_verify_weights_head_accepts_a_matching_policy(tmp_path):
    assert FP.verify_weights_head(_weights(tmp_path), _fp())


def test_verify_weights_head_raises_on_a_crossed_head_version(tmp_path):
    w = _weights(tmp_path, heads_version=2)
    with pytest.raises(FP.HeadMismatch) as e:
        FP.verify_weights_head(w, _fp())
    assert "heads_version" in str(e.value)


def test_verify_weights_head_raises_on_a_crossed_mode(tmp_path):
    w = _weights(tmp_path, mode="delta")
    with pytest.raises(FP.HeadMismatch):
        FP.verify_weights_head(w, _fp())


def test_board_policy_without_board_enabled_raises(tmp_path):
    """Stage B's silent truncation: the weights carried a 216-input h1 and
    inference fed 168 features, and no error was raised."""
    w = _weights(tmp_path, board={"c1w": [[[[0.0]]]]})
    with pytest.raises(FP.HeadMismatch) as e:
        FP.verify_weights_head(w, _fp(board=False))
    assert "board" in str(e.value)


def test_board_enabled_but_policy_has_no_trunk_raises(tmp_path):
    with pytest.raises(FP.HeadMismatch):
        FP.verify_weights_head(_weights(tmp_path), _fp(board=True))


# ------------------------------------------------------------ the live read
def test_live_fingerprint_reads_the_real_modules():
    """One pass over the real action space and the real draw. No games."""
    fp = FP.fingerprint({"engine": "1.32.7"})
    assert fp["schema"] == FP.SCHEMA
    assert fp["heads_version"] >= 1 and fp["heads_mode"]
    assert fp["n_action_logits"] > 0
    assert len(fp["head_sizes_sha"]) == 40 and len(fp["code"]) == 40
    c = fp["corpus"]
    if not c.get("missing"):
        assert c["n"] == sum(c["counts"]) and c["per_bin"] == 175
        assert len(c["draw_sha"]) == 40
    # it must agree with the module it claims to be reading
    from kagent import features
    assert fp["n_action_logits"] == features.N_ACTION_LOGITS
    assert fp["heads_version"] == features.HEADS_VERSION


def test_code_digest_moves_when_a_source_file_moves(tmp_path):
    a = tmp_path / "pkg"
    a.mkdir()
    (a / "m.py").write_text("x = 1\n")
    d1 = FP.code_digest((str(a),))
    (a / "m.py").write_text("x = 2\n")
    assert FP.code_digest((str(a),)) != d1
