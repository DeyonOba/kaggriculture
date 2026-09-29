"""The manifest: atomic save, the resume point after a kill, and a poisoned
stage that survives a reload and blocks the push.

    PYTHONPATH=. uv run pytest rlv3/tests/test_manifest.py -q
"""

from __future__ import annotations

import json
import os

from rlv3 import manifest as MF


def _slice_with(man, stage="ppo", status="ran", start=0, n=6, total=24):
    MF.record_stage(man, stage, status, "t0", "t1", "py rl/train.py", [],
                    plan={"slice": {"start": start, "n": n, "total": total,
                                    "lr": 3.8e-4, "ent": 0.03,
                                    "lr_max_rel_dev": 0.12}})


def test_save_and_load_round_trip(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run, job={"name": "rlv3_train"})
    MF.begin_slice(man, budget_s=3600, procs=2, host="box", fp={"heads_version": 1})
    MF.record_stage(man, "index", "ran", "t0", "t1", "py -c x", [])
    MF.save(run, man)

    again = MF.load(run)
    assert again["schema"] == MF.SCHEMA
    assert again["job"]["name"] == "rlv3_train"
    assert again["slices"][0]["stages"][0]["name"] == "index"
    assert again["fingerprint"]["heads_version"] == 1


def test_atomic_save_survives_a_truncated_tmp(tmp_path):
    """A kill during a write must leave the PREVIOUS manifest intact: save
    never replaces the real file until the write completed."""
    run = str(tmp_path / "a1")
    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    MF.record_stage(man, "clone", "ran", "t0", "t1", "py", [])
    MF.save(run, man)
    good = open(MF.path_for(run)).read()

    # a slice killed mid-write leaves a half-written .tmp behind
    with open(MF.path_for(run) + ".tmp", "w") as fh:
        fh.write('{"schema": 1, "slices": [{"idx": 0, "sta')
    assert open(MF.path_for(run)).read() == good
    reloaded = MF.load(run)
    assert reloaded["slices"][0]["stages"][0]["name"] == "clone"

    # and the next save clears the debris rather than tripping over it
    MF.save(run, reloaded)
    assert not os.path.exists(MF.path_for(run) + ".tmp")


def test_a_corrupt_manifest_does_not_crash_the_slice(tmp_path):
    run = str(tmp_path / "a1")
    os.makedirs(run)
    with open(MF.path_for(run), "w") as fh:
        fh.write("{ truncated")
    man = MF.load(run)
    assert man["schema"] == MF.SCHEMA and man["slices"] == []


def test_resume_point_after_a_killed_slice(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    MF.record_stage(man, "clone", "ran", "t0", "t1", "py", [])
    _slice_with(man, "ppo", "ran", start=0, n=6, total=24)
    MF.save(run, man)

    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    MF.record_stage(man, "clone", "skipped", "t0", "t1", None, [])
    _slice_with(man, "ppo", "budget", start=6, n=8, total=24)   # killed by the wall
    MF.save(run, man)

    rp = MF.load(run)
    point = MF.resume_point(rp)
    assert point["stage"] == "ppo"
    # the slice did NOT complete, so nothing past iteration 6 is claimed
    assert point["iters_done"] == 6 and point["target"] == 24
    # and the completed-iteration tally agrees
    assert MF.iters_recorded(rp, "ppo") == 6


def test_iters_recorded_accumulates_completed_slices(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    _slice_with(man, "ppo", "ran", 0, 6, 24)
    MF.begin_slice(man, budget_s=3600, procs=2)
    _slice_with(man, "ppo", "ran", 6, 9, 24)
    MF.save(run, man)
    assert MF.iters_recorded(MF.load(run), "ppo") == 15


def test_poisoned_survives_a_reload_and_blocks(tmp_path):
    """The Stage B signature looks like completed work to job.resolve_step
    (telemetry rows >= target plus an eval.json) and would be skipped forever.
    POISONED is how that is made visible to the NEXT slice and to
    rlv3/kaggle/push_checkpoints.sh."""
    run = str(tmp_path / "a1")
    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    MF.poison(man, run, "ppo", "policy_updates=0 outside warmup; onpolicy_kl=+3.7")

    again = MF.load(run)
    assert MF.is_poisoned(again, "ppo")
    assert "onpolicy_kl" in again["poisoned"]["ppo"]["reason"]
    assert not MF.is_poisoned(again, "gate")
    assert "POISONED ppo" in MF.summary(again)

    MF.unpoison(again, "ppo")
    MF.save(run, again)
    assert not MF.is_poisoned(MF.load(run), "ppo")


def test_record_artifact_mirrors_the_dataset_manifest_shape(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run)
    art = tmp_path / "bc.pt"
    art.write_text("w")
    rec = MF.record_artifact(man, str(art),
                             {"sha256": "ab" * 32, "bytes": 1, "fp": {}})
    for k in ("path", "bytes", "sha256", "stamp"):
        assert k in rec
    assert man["artifacts"][os.path.abspath(str(art))]["sha256"] == "ab" * 32


def test_summary_is_one_screen_and_names_the_draw(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run)
    man["fingerprint"] = {"heads_version": 1, "heads_mode": "absolute",
                          "n_action_logits": 57, "n_features": 168,
                          "board": False, "board_arm": "T_crop_103",
                          "board_ch": 103, "code": "d" * 40, "engine": "1.32.7",
                          "corpus": {"n": 995, "per_bin": 175, "seed": 0,
                                     "counts": [64, 21, 175, 175, 175, 175],
                                     "draw_sha": "c" * 40}}
    MF.begin_slice(man, budget_s=3600, procs=2)
    _slice_with(man, "ppo", "ran", 0, 6, 24)
    man["gate"] = {"arm": "cand", "eval_json": "x.json",
                   "objective": [0.354, 24596.0, -17100.0, 995]}
    text = MF.summary(man)
    assert "995" in text and "heads v1" in text and "win 0.354" in text
    assert len(text.splitlines()) < 25


def test_record_stage_refuses_an_unknown_status(tmp_path):
    man = MF.load(str(tmp_path / "a1"))
    MF.begin_slice(man, budget_s=1, procs=1)
    try:
        MF.record_stage(man, "ppo", "finished", "t0", "t1", None, [])
    except ValueError as e:
        assert "finished" in str(e)
    else:
        raise AssertionError("an unknown status must be refused")


def test_manifest_is_valid_json_on_disk(tmp_path):
    run = str(tmp_path / "a1")
    man = MF.load(run)
    MF.begin_slice(man, budget_s=3600, procs=2)
    _slice_with(man)
    MF.save(run, man)
    with open(MF.path_for(run)) as fh:
        json.load(fh)
