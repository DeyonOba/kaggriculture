"""Stage resolution: job.py's four behaviours through rlv3, the budget-aware
PPO slice, the LR compensation, the slice invariant, and the trainer-health
read-back.

    PYTHONPATH=. uv run pytest rlv3/tests/test_stages.py -q

The job.py assertions are deliberately the SAME ones as tests/test_kaggle_job.py
-- if `resolve()` stopped reproducing them, this driver would have quietly
forked the contract instead of extending it. No games are played.
"""

from __future__ import annotations

import json
import os

import pytest

from rlv3 import fingerprint as FP
from rlv3 import stages as ST


def _fp(**over):
    fp = {"schema": 1, "heads_version": 1, "heads_mode": "absolute",
          "head_sizes_sha": "a" * 40, "n_action_logits": 57, "n_features": 168,
          "board_arm": "T_crop_103", "board_ch": 103, "board": False,
          "corpus": {"draw_sha": "c" * 40, "n": 995}, "code": "d" * 40,
          "engine": "1.32.7"}
    fp.update(over)
    return fp


def _fmt(tmp_path):
    return dict(procs=4, out=str(tmp_path), repo=str(tmp_path), python="py",
                run=str(tmp_path))


def _telemetry(run, n, start=0, updates=8, kl=1e-7, stop=None, warmup_to=3):
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "telemetry.jsonl", "a") as fh:
        for i in range(start, start + n):
            fh.write(json.dumps({
                "iter": i, "money": 1.0, "policy_updates": updates,
                "onpolicy_kl": kl, "kl_stop_at": stop,
                "warmup": i < warmup_to}) + "\n")


PPO_CMD = ("py rl/train.py --resume {out}/bc/bc.pt --run {out}/ppo --iters 24 "
           "--episodes 64 --procs {procs} --reward win --vary-worlds "
           "--session-mix eval --session-frac 0.5 --plateau-cool 1.0 "
           "--patience 99 --eval-every 6 --eval-per-bin 8 --seed 1234")


# ==================================================== job.py's four behaviours
def test_job_module_loads_and_is_the_real_one():
    job = ST.load_job_module()
    assert hasattr(job, "resolve_step") and hasattr(job, "_telemetry_iters")
    assert job.resolve_step("{python} -u x.py --procs {procs}",
                            dict(procs=4, out="/o", repo="/r", python="py"),
                            4000) == ("py -u x.py --procs 4", "run")


def test_resolve_reproduces_min_left_s(tmp_path):
    stage = ST.StageSpec(name="s", cmd="py x", min_left_s=5000)
    res = ST.resolve(stage, _fmt(tmp_path), 4000, _fp(), str(tmp_path))
    assert res.cmd is None and "left" in res.reason


def test_resolve_reproduces_require(tmp_path):
    (tmp_path / "trunk.pt").write_text("x")
    f = _fmt(tmp_path)
    assert ST.resolve(ST.StageSpec(name="s", cmd="py x", require=("{out}/nope",)),
                      f, 4000, _fp(), str(tmp_path)).cmd is None
    assert ST.resolve(ST.StageSpec(name="s", cmd="py x", require=("{out}/trunk.pt",)),
                      f, 4000, _fp(), str(tmp_path)).cmd == "py x"


def test_resolve_reproduces_skip_if_but_only_when_stamped(tmp_path):
    """job.py skips when the output exists. rlv3 skips when it exists AND was
    made under today's fingerprint -- the one-line difference this whole
    component is built around."""
    (tmp_path / "trunk.pt").write_text("x")
    stage = ST.StageSpec(name="s", cmd="py x", depends=("code",),
                         skip_if=("{out}/nope", "{out}/trunk.pt"))
    fp = _fp()
    res = ST.resolve(stage, _fmt(tmp_path), 4000, fp, str(tmp_path))
    assert res.cmd is not None and "unstamped" in res.reason
    (tmp_path / "trunk.pt").write_text("x")
    FP.write_stamp(str(tmp_path / "trunk.pt"), fp, stage.depend_keys)
    res = ST.resolve(stage, _fmt(tmp_path), 4000, fp, str(tmp_path))
    assert res.cmd is None and res.reason.startswith("skip: fresh")


def test_resolve_reproduces_resume_run_for_a_shell_stage(tmp_path):
    """A non-PPO stage keeps job.py's own --resume/--iters/--start-iter
    rewriting verbatim -- the assertions of tests/test_kaggle_job.py."""
    run = tmp_path / "top_control_s1234"
    stage = ST.StageSpec(
        name="s", kind="shell",
        cmd=("py rl/train.py --resume {out}/bc/bc.pt --iters 20 "
             "--run {out}/top_control_s1234"),
        resume_run="{out}/top_control_s1234", iters=20)
    f = _fmt(tmp_path)

    res = ST.resolve(stage, f, 40_000, _fp(), str(tmp_path))
    assert f"--resume {tmp_path}/bc/bc.pt" in res.cmd
    assert "--iters 20" in res.cmd and "--start-iter" not in res.cmd

    _telemetry(run, 5)
    (run / "last.pt").write_text("w")
    res = ST.resolve(stage, f, 40_000, _fp(), str(tmp_path))
    assert f"--resume {run}/last.pt" in res.cmd
    assert "--iters 15" in res.cmd and res.cmd.endswith("--start-iter 5")

    (run / "telemetry.jsonl").write_text(
        "\n".join(json.dumps({"iter": i}) for i in range(20)) + "\n")
    (run / "eval.json").write_text("{}")
    res = ST.resolve(stage, f, 40_000, _fp(), str(tmp_path))
    assert res.cmd is None and "complete" in res.reason


# ============================================================== the PPO slice
def test_iters_done_is_job_pys_own_counter(tmp_path):
    run = tmp_path / "ppo"
    assert ST.iters_done(str(run)) == 0
    _telemetry(run, 7)
    assert ST.iters_done(str(run)) == 7


def test_sec_per_iter_prefers_this_runs_own_record(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    assert ST.sec_per_iter(str(run), default=900.0) == 900.0
    ST.record_slice_timing(str(run), 0, 5, 1500.0)     # 300 s/iter
    ST.record_slice_timing(str(run), 5, 10, 2000.0)    # 400 s/iter
    assert ST.sec_per_iter(str(run)) == pytest.approx(350.0)


def test_sec_per_iter_falls_back_to_the_trainers_own_log(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    (run / "train.log").write_text(
        "iter   0: money mean=1 win-vs-gate=0.10 lr=2.5e-04 steps=9 bad=0 "
        "upd=8 kl0=+1e-08 ep_dur p50=2.0s max=3.0s (120s)\n"
        "iter   1: money mean=1 win-vs-gate=0.10 lr=2.4e-04 steps=9 bad=0 "
        "upd=8 kl0=+1e-08 ep_dur p50=2.0s max=3.0s (180s)\n")
    assert ST.sec_per_iter(str(run)) == pytest.approx(150.0)


def test_ppo_slice_uses_the_wall_and_never_eats_the_reserve(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    ST.record_slice_timing(str(run), 0, 2, 600.0)      # 300 s/iter
    ST.record_slice_timing(str(run), 2, 4, 600.0)
    _telemetry(run, 4)
    # 4h wall, 1h held for the gate -> 3h usable -> 36 iterations of headroom,
    # capped by the target.
    sl = ST.ppo_slice(str(run), 24, remaining_s=4 * 3600, reserve_s=3600)
    assert sl.start == 4 and sl.n == 24 - 4 and sl.total == 24
    # exactly the reserve and no more: 1h usable at 300 s/iter = 12 iterations
    sl = ST.ppo_slice(str(run), 24, remaining_s=2 * 3600, reserve_s=3600)
    assert sl.n == 12
    assert sl.deadline_s <= 2 * 3600 - 3600 + 1e-6


def test_ppo_slice_returns_none_below_min_iters(tmp_path):
    """Adam's moments are not serialised, so a 1-iteration slice pays a full
    optimiser restart for one iteration of work -- and the gate would lose its
    hour for it."""
    run = tmp_path / "ppo"
    run.mkdir()
    ST.record_slice_timing(str(run), 0, 2, 1800.0)     # 900 s/iter
    ST.record_slice_timing(str(run), 2, 4, 1800.0)
    _telemetry(run, 4)
    assert ST.ppo_slice(str(run), 24, remaining_s=5400, reserve_s=3600,
                        min_iters=3) is None           # 1800s usable = 2 iters
    assert ST.ppo_slice(str(run), 24, remaining_s=6300, reserve_s=3600,
                        min_iters=3).n == 3


def test_ppo_slice_returns_none_at_target(tmp_path):
    run = tmp_path / "ppo"
    _telemetry(run, 24)
    assert ST.ppo_slice(str(run), 24, 40_000, 0) is None


# ======================================================= the LR compensation
def test_lr_compensation_is_a_noop_on_an_unsliced_run():
    lr, ent, dev = ST.lr_compensation(0, 20, 20)
    from rl import train as T
    assert lr == pytest.approx(T.LR) and ent == pytest.approx(T.ENT_COEF)
    assert dev == pytest.approx(0.0)


def test_lr_compensation_prices_the_spec_example():
    """rl/train.py:710 divides by the SLICE length. At k=10, m=10, T=20 the
    resumed slice sits at LR_FLOOR=0.15 from its first iteration where the
    unsliced run is at 0.50 -- a 3.3x error."""
    from rl import train as T
    k, m, T_ = 10, 10, 20
    # the defect, uncompensated
    uncomp_first = T.LR * max(T.LR_FLOOR, 1 - k / m)
    intended_first = T.LR * max(T.LR_FLOOR, 1 - k / T_)
    assert uncomp_first == pytest.approx(T.LR * T.LR_FLOOR)
    assert intended_first / uncomp_first == pytest.approx(0.5 / 0.15, rel=1e-6)

    lr, ent, dev = ST.lr_compensation(k, m, T_)
    assert lr > T.LR and ent / lr == pytest.approx(T.ENT_COEF / T.LR)
    # a constant cannot track a ramp; the residual is reported, not hidden
    assert dev == pytest.approx((1.0 - 0.3) / (1.0 + 0.3), rel=1e-6)

    intended, realised = ST.lr_curves(k, m, T_, lr_eff=lr)
    worst = max(abs(r / i - 1.0) for r, i in zip(realised, intended))
    assert worst == pytest.approx(dev, rel=1e-6)
    # and it is strictly better than shipping the uncompensated command
    intended_u, realised_u = ST.lr_curves(k, m, T_, lr_eff=T.LR)
    worst_u = max(abs(r / i - 1.0) for r, i in zip(realised_u, intended_u))
    assert worst < worst_u and worst_u > 0.6


def test_lr_compensation_is_the_best_constant():
    """The returned scale minimises the WORST relative deviation: nudging it
    either way makes the worst end worse."""
    k, m, T_ = 10, 10, 20
    lr, _, dev = ST.lr_compensation(k, m, T_)
    for f in (0.9, 1.1):
        i, r = ST.lr_curves(k, m, T_, lr_eff=lr * f)
        assert max(abs(a / b - 1.0) for a, b in zip(r, i)) > dev


def test_lr_tol_shortens_the_slice_when_asked(tmp_path):
    """Switchable, and OFF by default: honouring 0.15 at T=20 caps a slice
    near three iterations however much wall is free."""
    run = tmp_path / "ppo"
    run.mkdir()
    ST.record_slice_timing(str(run), 0, 5, 500.0)      # 100 s/iter
    ST.record_slice_timing(str(run), 5, 10, 500.0)
    _telemetry(run, 10)
    wide = ST.ppo_slice(str(run), 20, remaining_s=40_000, reserve_s=0)
    assert wide.n == 10 and wide.dev > 0.15            # the default: use the wall
    tight = ST.ppo_slice(str(run), 20, remaining_s=40_000, reserve_s=0,
                         lr_tol=0.15)
    assert tight.n < wide.n and tight.dev <= 0.15
    assert "LR" in tight.reason


# ========================================================== the invariant
def _cmd(**over):
    base = {"iters": 10, "start_iter": 10, "plateau_cool": 1.0, "patience": 99,
            "session_mix": "eval", "lr": 3.8e-4, "ent_coef": 0.03,
            "resume": "/run/ppo/last.pt"}
    base.update(over)
    return (f"py rl/train.py --resume {base['resume']} --run /run/ppo "
            f"--iters {base['iters']} --session-mix {base['session_mix']} "
            f"--plateau-cool {base['plateau_cool']} --patience {base['patience']} "
            f"--lr {base['lr']} --ent-coef {base['ent_coef']} "
            f"--start-iter {base['start_iter']}")


def test_invariant_passes_a_correct_slice():
    assert ST.assert_slice_invariant(_cmd(), 10, 10, 20) == []


@pytest.mark.parametrize("over,needle", [
    ({"plateau_cool": 0.6}, "plateau-cool"),
    ({"patience": 4}, "patience"),
    ({"session_mix": "frontier"}, "session-mix"),
    ({"start_iter": 7}, "start-iter"),
    ({"iters": 24}, "iters"),
    ({"resume": "/run/bc/bc.pt"}, "resume"),
])
def test_invariant_refuses(over, needle):
    bad = ST.assert_slice_invariant(_cmd(**over), 10, 10, 20)
    assert any(needle in b for b in bad), bad


def test_invariant_refuses_a_missing_compensation():
    cmd = ("py rl/train.py --resume /run/ppo/last.pt --run /run/ppo --iters 10 "
           "--session-mix eval --plateau-cool 1.0 --patience 99 --start-iter 10")
    bad = ST.assert_slice_invariant(cmd, 10, 10, 20)
    assert any("--lr" in b for b in bad)


def test_invariant_refuses_bare_defaults():
    """Absence is not neutrality: the trainer's own defaults are
    --plateau-cool 0.6, --patience 4, --session-mix frontier."""
    bad = ST.assert_slice_invariant(
        "py rl/train.py --resume /run/ppo/last.pt --iters 10 --start-iter 10",
        10, 10, 20)
    joined = " ".join(bad)
    for needle in ("plateau-cool", "patience", "session-mix", "--lr"):
        assert needle in joined


def test_resolve_of_a_ppo_stage_writes_a_legal_slice(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    (tmp_path / "bc").mkdir()
    (tmp_path / "bc" / "bc.pt").write_text("w")
    ST.record_slice_timing(str(run), 0, 5, 500.0)
    ST.record_slice_timing(str(run), 5, 10, 500.0)
    _telemetry(run, 10)
    (run / "last.pt").write_text("w")
    stage = ST.StageSpec(name="ppo", cmd=PPO_CMD, kind="ppo",
                         depends=("action", "corpus", "code"),
                         resume_run="{out}/ppo", iters=24,
                         require=("{out}/bc/bc.pt",),
                         produces=("{out}/ppo/eval.json",))
    # the driver stamps a run dir before the FIRST slice launches, so a wall
    # kill still leaves a resumable run
    FP.write_stamp(str(run), _fp(), stage.depend_keys, extra={"target_iters": 24})
    res = ST.resolve(stage, _fmt(tmp_path), 40_000, _fp(), str(tmp_path),
                     reserve_s=3600)
    sl = res.plan["slice"]
    assert sl["start"] == 10 and sl["n"] == 14 and sl["total"] == 24
    assert f"--resume {run}/last.pt" in res.cmd
    assert "--iters 14" in res.cmd and res.cmd.endswith("--start-iter 10")
    assert "--lr " in res.cmd and "--ent-coef " in res.cmd
    assert ST.assert_slice_invariant(res.cmd, 10, 14, 24, run_dir=str(run)) == []


def test_resolve_of_a_ppo_stage_refuses_a_bad_command(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    (tmp_path / "bc").mkdir()
    (tmp_path / "bc" / "bc.pt").write_text("w")
    _telemetry(run, 4)
    (run / "last.pt").write_text("w")
    stage = ST.StageSpec(name="ppo", kind="ppo", resume_run="{out}/ppo", iters=24,
                         cmd=PPO_CMD.replace("--session-mix eval",
                                             "--session-mix frontier"))
    with pytest.raises(RuntimeError) as e:
        ST.resolve(stage, _fmt(tmp_path), 40_000, _fp(), str(tmp_path))
    assert "session-mix" in str(e.value)


# ===================================================== the health read-back
def test_verify_training_ran_passes_a_healthy_slice(tmp_path):
    run = tmp_path / "ppo"
    _telemetry(run, 10)
    out = ST.verify_training_ran(str(run), 0)
    assert out["rows"] == 10 and out["max_abs_kl"] < 1e-3
    assert ST.verify_training_ran(str(run), 5)["rows"] == 5


def test_verify_training_ran_raises_on_the_stage_b_signature(tmp_path):
    """No trunk serialised -> worker/trainer logprob gap +3.7 nats -> the KL
    stop fired on the first minibatch of every iteration, for twenty
    iterations, and nothing said so."""
    run = tmp_path / "ppo"
    _telemetry(run, 20, updates=0, kl=3.7, stop=[0, 0], warmup_to=3)
    with pytest.raises(ST.Poisoned) as e:
        ST.verify_training_ran(str(run), 0)
    msg = str(e.value)
    assert "onpolicy_kl" in msg and "policy_updates=0" in msg


def test_verify_training_ran_raises_on_zero_updates_alone(tmp_path):
    run = tmp_path / "ppo"
    _telemetry(run, 8, updates=0, kl=1e-8, stop=None, warmup_to=3)
    with pytest.raises(ST.Poisoned) as e:
        ST.verify_training_ran(str(run), 0)
    assert "policy_updates=0" in str(e.value)


def test_verify_training_ran_tolerates_warmup(tmp_path):
    """it < WARMUP_ITERS updates the value head only, so policy_updates is
    legitimately 0 there."""
    run = tmp_path / "ppo"
    _telemetry(run, 3, updates=0, warmup_to=3)
    _telemetry(run, 5, start=3, updates=8, warmup_to=3)
    assert ST.verify_training_ran(str(run), 0)["rows"] == 8


def test_verify_training_ran_raises_when_a_slice_appended_nothing(tmp_path):
    run = tmp_path / "ppo"
    _telemetry(run, 4)
    with pytest.raises(ST.Poisoned) as e:
        ST.verify_training_ran(str(run), 4)
    assert "no telemetry rows" in str(e.value)


# ============================================================= the real graph
def test_the_shipped_job_graph_loads_and_is_well_formed():
    path = os.path.join(ST.REPO, "rlv3", "jobs", "rlv3_train.json")
    cfg, specs = ST.load_stages(path)
    names = [s.name for s in specs]
    assert names == ["index", "clone", "maia2", "bc", "ppo", "gate", "card",
                     "promote"]
    ppo = next(s for s in specs if s.name == "ppo")
    assert ppo.kind == "ppo" and ppo.iters > 0 and ppo.resume_run
    for pin in ("--session-mix eval", "--plateau-cool 1.0", "--patience 99",
                "--reward win"):
        assert pin in ppo.cmd, f"the ppo stage must pin {pin}"
    gate = next(s for s in specs if s.name == "gate")
    assert gate.reserve and "--per-bin 175" in gate.cmd
    assert "champ_h10" in gate.cmd and "--baseline champ_h10" in gate.cmd
    # every non-cheap stage declares what its output depends on
    for s in specs:
        if s.produces:
            assert s.depend_keys, f"{s.name} produces without declaring depends"


# ========================================== inputs: the gate must not skip itself
def _gate_stage():
    return ST.StageSpec(
        name="gate", kind="gate", depends=("action", "corpus"),
        cmd="py rl/session_eval.py --arm cand={out}/ppo/eval.json --out {out}/session_eval.json",
        skip_if=("{out}/session_eval.json",), require=("{out}/ppo/eval.json",),
        produces=("{out}/session_eval.json",), inputs=("{out}/ppo/eval.json",))


def test_a_changed_candidate_makes_the_gate_rerun(tmp_path):
    """Without `inputs` the gate skips itself out of existence: its own
    session_eval.json is still there after PPO appends ten more iterations,
    and the action space and corpus have not moved -- so the 995-game verdict
    would never be re-run against the checkpoint it is judging."""
    out = tmp_path / "out"
    (out / "ppo").mkdir(parents=True)
    cand = out / "ppo" / "eval.json"
    cand.write_text('{"policy": {"heads_version": 1}}')
    gate_out = out / "session_eval.json"
    gate_out.write_text('{"results": {}}')

    stage = _gate_stage()
    fmt = dict(procs=2, out=str(out), repo=str(tmp_path), python="py", run=str(tmp_path))
    fp = _fp()
    fp_eff, keys = ST.effective(stage, fmt, fp)
    assert "inputs" in keys
    FP.write_stamp(str(gate_out), fp_eff, keys)

    # same candidate: the hour of eval is kept
    assert ST.resolve(stage, fmt, 40_000, fp, str(tmp_path)).cmd is None

    # PPO advanced -> the candidate's bytes changed -> the verdict is stale
    cand.write_text('{"policy": {"heads_version": 1}, "more": "training"}')
    res = ST.resolve(stage, fmt, 40_000, fp, str(tmp_path))
    assert res.cmd is not None and "inputs changed" in res.reason
    assert not gate_out.exists()


def test_input_digest_ignores_the_directory_it_lives_in(tmp_path):
    """Two runs at different paths with identical bytes must agree, or a
    restored Kaggle checkpoint would look stale purely for being unpacked
    somewhere else."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "eval.json").write_text("same")
    (b / "eval.json").write_text("same")
    assert ST.input_digest([str(a / "eval.json")]) == ST.input_digest([str(b / "eval.json")])
    (b / "eval.json").write_text("different")
    assert ST.input_digest([str(a / "eval.json")]) != ST.input_digest([str(b / "eval.json")])


def test_a_stale_ppo_run_is_quarantined_not_resumed(tmp_path):
    """`last.pt` carries the net's weights under the action space it was
    trained in. A run dir is not a skip_if hit, so it would otherwise never be
    checked -- the single most dangerous reuse in the graph."""
    out = tmp_path / "out"
    (out / "bc").mkdir(parents=True)
    (out / "bc" / "bc.pt").write_text("w")
    ppo = out / "ppo"
    ppo.mkdir()
    _telemetry(ppo, 9)
    (ppo / "last.pt").write_text("w")
    stage = ST.StageSpec(name="ppo", kind="ppo", cmd=PPO_CMD,
                         depends=("action", "corpus", "code"),
                         resume_run="{out}/ppo", iters=24,
                         require=("{out}/bc/bc.pt",), inputs=("{out}/bc/bc.pt",))
    fmt = dict(procs=2, out=str(out), repo=str(tmp_path), python="py", run=str(tmp_path))
    fp = _fp()
    fp_eff, keys = ST.effective(stage, fmt, fp)
    FP.write_stamp(str(ppo), fp_eff, keys, extra={"target_iters": 24})

    # unchanged: the run resumes from iteration 9
    res = ST.resolve(stage, fmt, 40_000, fp, str(tmp_path))
    assert res.plan["slice"]["start"] == 9

    # bc.pt rebuilt under a new corpus: resuming last.pt would be a crossed net
    (out / "bc" / "bc.pt").write_text("a different clone")
    res = ST.resolve(stage, fmt, 40_000, fp, str(tmp_path))
    assert "stale run" in res.reason
    assert res.plan["slice"]["start"] == 0, "a stale run must restart, not resume"
    assert not (ppo / "last.pt").exists()
    assert list((tmp_path / "stale").rglob("last.pt")), "the run was not quarantined"


# ================================================ the driver, without running
def test_dry_run_over_the_real_graph_executes_nothing(tmp_path):
    """The local check: prints the resolution table, plays no games, touches
    no dataset. Safe on a busy machine."""
    from rlv3 import train as DRV
    from rlv3 import manifest as MF

    run = str(tmp_path / "dry")
    man = DRV.run_job(os.path.join(ST.REPO, "rlv3", "jobs", "rlv3_train.json"),
                      run, budget_s=39_600, procs=2, dry_run=True)
    names = [st["name"] for st in man["slices"][-1]["stages"]]
    assert names == ["index", "clone", "maia2", "bc", "ppo", "gate", "card",
                     "promote"]
    assert all(st["status"] in ("skipped", "budget")
               for st in man["slices"][-1]["stages"])
    # nothing was produced, nothing was stamped, and the manifest is on disk
    assert man["artifacts"] == {} and not man["poisoned"]
    assert os.path.exists(MF.path_for(run))
    assert not os.path.exists(os.path.join(run, "ppo"))
    assert man["fingerprint"]["corpus"]["per_bin"] == 175


def test_dry_run_plans_a_real_ppo_slice(tmp_path):
    """With a half-finished run on disk the ppo stage resolves to a legal,
    compensated slice -- still without executing it."""
    from rlv3 import train as DRV

    run = tmp_path / "a1"
    (run / "bc").mkdir(parents=True)
    (run / "bc" / "bc.pt").write_text("w")
    ppo = run / "ppo"
    ppo.mkdir()
    _telemetry(ppo, 9)
    (ppo / "last.pt").write_text("w")
    ST.record_slice_timing(str(ppo), 0, 5, 1000.0)     # 200 s/iter
    ST.record_slice_timing(str(ppo), 5, 9, 800.0)

    job = os.path.join(ST.REPO, "rlv3", "jobs", "rlv3_train.json")

    # A run dir that rlv3 did not stamp is NOT adopted -- an unstamped run is
    # exactly the thing this component refuses to reuse, and a dry run says so
    # rather than promising a resume it would not perform.
    man = DRV.run_job(job, str(run), budget_s=39_600, procs=2, dry_run=True,
                      only=["ppo"])
    assert man["slices"][-1]["stages"][0]["status"] == "stale"
    assert "unstamped" in man["slices"][-1]["stages"][0]["note"]

    # stamped the way the driver stamps it before launching: now it resumes
    _, specs = ST.load_stages(job)
    spec = next(s for s in specs if s.name == "ppo")
    fmt = dict(procs=2, out=str(run), repo=ST.REPO, python="py", run=str(run))
    fp_eff, keys = ST.effective(spec, fmt, DRV.FP.fingerprint({"engine": "1.32.7"}))
    DRV.FP.write_stamp(str(ppo), fp_eff, keys, extra={"target_iters": 24})

    man = DRV.run_job(job, str(run), budget_s=39_600, procs=2, dry_run=True,
                      only=["ppo"])
    st = man["slices"][-1]["stages"][0]
    assert st["name"] == "ppo" and st["status"] == "skipped"   # dry run
    sl = st["plan"]["slice"]
    assert sl["start"] == 9 and sl["n"] == 24 - 9
    assert sl["sec_per_iter"] == pytest.approx(200.0)
    assert "--start-iter 9" in st["cmd"] and "--iters 15" in st["cmd"]
    assert ST.assert_slice_invariant(st["cmd"], 9, 15, 24, run_dir=str(ppo)) == []
    assert not (ppo / "eval.json").exists()
