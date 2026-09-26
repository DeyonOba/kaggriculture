"""tools/kaggle_job/job.py step resolution: skip on existing output, skip on a
missing input, continue a PPO run from its last.pt with the remaining
iterations, and skip a step that cannot fit the remaining budget.

    PYTHONPATH=. uv run pytest tests/test_kaggle_job.py -q
"""
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _job():
    spec = importlib.util.spec_from_file_location("kjob", os.path.join(REPO, "tools", "kaggle_job", "job.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_string_steps_and_budget(tmp_path):
    job = _job()
    fmt = dict(procs=4, out=str(tmp_path), repo=str(tmp_path), python="py")
    cmd, why = job.resolve_step("{python} -u x.py --procs {procs}", fmt, 4000)
    assert cmd == "py -u x.py --procs 4" and why == "run"
    cmd, why = job.resolve_step({"cmd": "py x", "min_left_s": 5000}, fmt, 4000)
    assert cmd is None and "left" in why


def test_skip_if_and_require(tmp_path):
    job = _job()
    fmt = dict(procs=4, out=str(tmp_path), repo=str(tmp_path), python="py")
    (tmp_path / "trunk.pt").write_text("x")
    assert job.resolve_step({"cmd": "py x", "skip_if": "{out}/trunk.pt"}, fmt, 4000)[0] is None
    assert job.resolve_step({"cmd": "py x", "skip_if": ["{out}/nope", "{out}/trunk.pt"]}, fmt, 4000)[0] is None
    assert job.resolve_step({"cmd": "py x", "require": "{out}/nope"}, fmt, 4000)[0] is None
    assert job.resolve_step({"cmd": "py x", "require": ["{out}/trunk.pt"]}, fmt, 4000)[0] == "py x"


def test_resume_run_continues_from_last_pt(tmp_path):
    job = _job()
    fmt = dict(procs=4, out=str(tmp_path), repo=str(tmp_path), python="py")
    run = tmp_path / "top_control_s1234"
    run.mkdir()
    step = {"cmd": "py rl/train.py --resume {out}/bc/bc.pt --iters 20 --run {out}/top_control_s1234",
            "resume_run": "{out}/top_control_s1234", "iters": 20}
    # nothing done yet: unchanged
    cmd, _ = job.resolve_step(step, fmt, 40000)
    assert "--resume " + str(tmp_path) + "/bc/bc.pt" in cmd and "--iters 20" in cmd and "--start-iter" not in cmd
    # five iterations done, last.pt present: continue with 15 from iteration 5
    (run / "telemetry.jsonl").write_text("\n".join(json.dumps({"iter": i}) for i in range(5)) + "\n")
    (run / "last.pt").write_text("w")
    cmd, _ = job.resolve_step(step, fmt, 40000)
    assert f"--resume {run}/last.pt" in cmd and "--iters 15" in cmd and cmd.endswith("--start-iter 5")
    # complete: skipped
    (run / "telemetry.jsonl").write_text("\n".join(json.dumps({"iter": i}) for i in range(20)) + "\n")
    (run / "eval.json").write_text("{}")
    cmd, why = job.resolve_step(step, fmt, 40000)
    assert cmd is None and "complete" in why
