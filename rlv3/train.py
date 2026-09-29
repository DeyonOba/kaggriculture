"""rlv3's checkpointed, resumable, slice-safe training driver.

    PYTHONPATH=. uv run python -m rlv3.train \\
        --job rlv3/jobs/rlv3_train.json --run rlv3/runs/a1 \\
        --budget-s 39600 --procs 2 [--dry-run] [--only ppo] [--force ppo]

WHAT THIS IS FOR. Compute arrives in slices -- an 11-hour Kaggle session, a
laptop that has to give the CPU back, a run killed by a wall. A training plan
that only works when it runs to completion in one process is not a plan. This
driver turns the plan into a STAGE GRAPH whose every output is stamped with
the fingerprint it was made under, so the next slice can tell "this exists"
apart from "this is still valid", and resumes only where resuming is sound.

THE DAWN, since this component makes no in-game decision: the start of each
slice, and its state is the run, not the board. At the top of every slice it
recomputes `rlv3.fingerprint.fingerprint()` -- the live action space
(HEADS_VERSION / MODE / HEAD_SIZES sha / N_ACTION_LOGITS), the observation
width, the board arm and rl.common.BOARD_CH, the corpus draw (the actual
(path, seat) list `rl.session_eval.held_out_entries` returns at the gate's own
seed and per-bin), the rlv3+kagent+rl code digest and the engine pin. Each
stage then asks two questions instead of one: does my output exist, and was it
made under today's fingerprint? Fresh skips. Stale is quarantined into
`<run>/stale/` and re-run. For the PPO stage it also reads the run's own
iterations completed, its measured seconds an iteration, the wall remaining
and the reserve held for the gate, and from those plans this slice's
(start, n, lr, ent) -- so the plan self-corrects to the machine it woke up on.
Afterwards it reads back the rows the slice just appended and the head version
inside the eval.json it just produced, and marks the stage POISONED rather
than done if either disagrees.

SELECTION IS NOT THE TRAINER'S. `rl/train.py` keeps best.pt by mean final
money and prints an 18-game SESSION-WR. Promotion here runs through
`rlv3.promote`, on the 995-game paired gate, under the lexicographic
objective: win rate first, then profit when winning, then loss margin when
losing.

--dry-run prints the resolution table and exits without executing anything.
That is the local check: it costs no games and is safe on a busy machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rlv3 import fingerprint as FP          # noqa: E402
from rlv3 import manifest as MF             # noqa: E402
from rlv3 import promote as PR              # noqa: E402
from rlv3 import stages as ST               # noqa: E402

DEFAULT_JOB = os.path.join(REPO, "rlv3", "jobs", "rlv3_train.json")


def _iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------- the shell
def _run_cmd(cmd, env, log_path, timeout_s):
    """Run one stage, tee-ing stdout to `<run>/<stage>.log`. -> (rc, note).

    The deadline is a watchdog thread, not a check inside the read loop: a
    trainer that wedges with no output would never reach a check that only
    fires when a line arrives, and would eat the gate's reserve in silence.

    The log is also the audit trail for the trainer's own `lr=` print, which
    the slice planner's intended curve is checked against -- a schedule that
    is not the one we asked for is a different experiment from the unsliced
    run, and this project has been burned by un-paired ablations before.
    """
    import threading
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    killed = {"by_wall": False}
    with open(log_path, "a", buffering=1) as lf:
        lf.write(f"\n=== {_iso()}  {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)

        def _wall():
            killed["by_wall"] = True
            lf.write(f"=== killed at the time budget ({timeout_s:.0f}s)\n")
            p.kill()

        watchdog = threading.Timer(max(1.0, timeout_s), _wall)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in p.stdout:
                lf.write(line)
                sys.stdout.write(line)
            rc = p.wait()
        finally:
            watchdog.cancel()
            try:
                p.stdout.close()
            except Exception:
                pass
    if killed["by_wall"]:
        return None, "hit the time budget; partial outputs kept"
    return rc, None


def _lr_audit(log_path, intended):
    """Parse the trainer's own `lr=<x>` prints and compare with the plan.

    Believing the plan is what the whole slice machinery is trying to avoid.
    """
    import re
    seen = []
    if not os.path.exists(log_path):
        return None
    pat = re.compile(r"lr=([0-9.eE+-]+)")
    with open(log_path, errors="replace") as fh:
        for line in fh:
            if line.startswith("iter "):
                m = pat.search(line)
                if m:
                    try:
                        seen.append(float(m.group(1)))
                    except ValueError:
                        pass
    if not seen or not intended:
        return None
    tail = seen[-len(intended):]
    devs = [abs(r / i - 1.0) for r, i in zip(tail, intended) if i]
    return {"realised": tail, "intended": list(intended),
            "max_rel_dev": max(devs) if devs else None}


# ------------------------------------------------------------------ the loop
def run_job(job_path, run_dir, *, budget_s, procs, dry_run=False, only=None,
            force=(), started_at=None, lr_tol=None, probe_features=False):
    """One compute slice over the whole stage graph. -> the manifest dict."""
    t0 = float(started_at) if started_at else time.time()
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cfg, specs = ST.load_stages(job_path)
    only = set(only or ())
    force = set(force or ())

    # ---- dawn: today's fingerprint
    fp = FP.fingerprint({
        "engine": cfg.get("engine"),
        "index_path": cfg.get("index_path"),
        "seed": cfg.get("seed", FP.DEFAULT_SEED),
        "per_bin": cfg.get("per_bin", FP.DEFAULT_PER_BIN),
        "n_features": FP.probe_n_features(
            os.path.join(run_dir, "n_features.json"), play=probe_features),
    })
    man = MF.load(run_dir, job=cfg)
    MF.begin_slice(man, budget_s=budget_s, procs=procs, fp=fp)
    MF.save(run_dir, man)

    fmt = dict(procs=procs, out=run_dir, repo=REPO, python=sys.executable,
               run=run_dir)
    base_env = dict(os.environ, PYTHONPATH=REPO)

    print(f"== rlv3.train  run={run_dir}  budget={budget_s / 3600:.2f}h  "
          f"procs={procs}{'  DRY RUN' if dry_run else ''}")
    print(MF.summary(man))
    print(f"{'stage':<9}{'verdict':<9}  reason")

    stopped = "completed"
    for i, spec in enumerate(specs):
        if only and spec.name not in only:
            continue
        remaining = budget_s - (time.time() - t0)
        # Budget held back for later reserve=True stages (the gate must start).
        reserve = sum(s.min_left_s for s in specs[i + 1:]
                      if s.reserve and (not only or s.name in only))
        started = _iso()

        if MF.is_poisoned(man, spec.name) and spec.name not in force:
            note = (f"POISONED: {man['poisoned'][spec.name]['reason'][:160]} "
                    f"-- re-run with --force {spec.name} to discard and restart")
            print(f"{spec.name:<9}{'poisoned':<9}  {note}")
            MF.record_stage(man, spec.name, "poisoned", started, _iso(), None,
                            [], note=note)
            MF.save(run_dir, man)
            stopped = f"poisoned at {spec.name}"
            break
        if spec.name in force and MF.is_poisoned(man, spec.name):
            ppo_dir = (spec.resume_run or "").format(**fmt)
            if ppo_dir and os.path.isdir(ppo_dir) and not dry_run:
                FP.quarantine(ppo_dir, run_dir, "forced re-run of a poisoned stage")
            MF.unpoison(man, spec.name)

        # ---- in-process stages
        if spec.kind == "promote":
            missing = [p for p in (q.format(**fmt) for q in spec.require)
                       if not os.path.exists(p)]
            if missing:
                why = f"skip: input missing {missing[0]}"
                print(f"{spec.name:<9}{'skipped':<9}  {why}")
                MF.record_stage(man, spec.name, "skipped", started, _iso(),
                                None, [], note=why)
                MF.save(run_dir, man)
                continue
            res = ST.Resolution(spec.cmd, "run (in-process)", {"stage": spec.name})
            print(f"{spec.name:<9}{'run':<9}  in-process promotion check")
            if dry_run:
                MF.record_stage(man, spec.name, "skipped", started, _iso(),
                                None, [], note="dry run", plan=res.plan)
                MF.save(run_dir, man)
                continue
            try:
                path = PR.promote(
                    run_dir,
                    eval_json=cfg["promote"]["eval_json"].format(**fmt),
                    arm=cfg["promote"]["arm"],
                    baseline_arm=cfg["promote"].get("baseline_arm", "champ_h10"),
                    ppo_run=cfg["promote"].get("ppo_run", "{out}/ppo").format(**fmt),
                    fp=fp, man=man)
                note = (f"promoted -> {path}" if path
                        else (man.get("gate") or {}).get("verdict", "not promoted"))
                MF.record_stage(man, spec.name, "ran", started, _iso(), None,
                                [path] if path else [], note=note, plan=res.plan)
            except Exception as exc:                    # noqa: BLE001
                MF.record_stage(man, spec.name, "failed", started, _iso(), None,
                                [], note=f"{type(exc).__name__}: {exc}",
                                plan=res.plan)
                print(f"           promote failed: {exc}")
            MF.save(run_dir, man)
            print("           " + (man.get("gate") or {}).get("verdict", ""))
            continue

        # ---- resolution
        try:
            res = ST.resolve(spec, fmt, remaining, fp, run_dir,
                             reserve_s=reserve, lr_tol=lr_tol,
                             quarantine_stale=not dry_run)
        except Exception as exc:                       # noqa: BLE001
            # Broad on purpose: a slice invariant violation, a malformed job
            # JSON and a bad {placeholder} must all end with a SAVED manifest
            # saying which stage refused and why. An uncaught traceback here
            # would lose the whole slice's record.
            print(f"{spec.name:<9}{'REFUSED':<9}  {type(exc).__name__}: {exc}")
            MF.record_stage(man, spec.name, "failed", started, _iso(), None, [],
                            note=f"{type(exc).__name__}: {exc}"[:600])
            MF.save(run_dir, man)
            stopped = f"refused at {spec.name}"
            break

        if res.cmd is None:
            status = ("stale" if res.reason.startswith("stale")
                      else "budget" if "budget" in res.reason or "left" in res.reason
                      else "skipped")
            print(f"{spec.name:<9}{status:<9}  {res.reason}")
            MF.record_stage(man, spec.name, status, started, _iso(), None, [],
                            note=res.reason, plan=res.plan)
            MF.save(run_dir, man)
            continue

        print(f"{spec.name:<9}{'run':<9}  {res.reason}")
        print(f"           $ {res.cmd}")
        if dry_run:
            MF.record_stage(man, spec.name, "skipped", started, _iso(), res.cmd,
                            [], note="dry run", plan=res.plan)
            MF.save(run_dir, man)
            continue

        # ---- the PPO run-dir stamp is written BEFORE the trainer starts, so a
        # slice killed by the wall still leaves a stamped run and the next
        # slice resumes instead of quarantining an hour of work.
        sl = (res.plan.get("slice") or {})
        ppo_dir = res.plan.get("run_dir")
        fp_pre, keys_pre = ST.effective(spec, fmt, fp)
        if spec.kind == "ppo" and ppo_dir:
            os.makedirs(ppo_dir, exist_ok=True)
            if FP.read_stamp(ppo_dir) is None:
                FP.write_stamp(ppo_dir, fp_pre, keys_pre,
                               extra={"target_iters": spec.iters})

        env = dict(base_env)
        env.update({k: str(v) for k, v in (spec.env or {}).items()})
        log_path = os.path.join(run_dir, f"{spec.name}.log")
        timeout = max(60.0, min(budget_s - (time.time() - t0) - reserve,
                                sl.get("deadline_s", 1e18)))
        wall0 = time.time()
        rc, note = _run_cmd(res.cmd, env, log_path, timeout)
        wall = time.time() - wall0

        status = "ran"
        if rc is None:
            status, stopped = "budget", f"wall at {spec.name}"
        elif rc != 0:
            status, note = "failed", f"exit {rc}"

        # ---- read back what the slice actually did
        if spec.kind == "ppo" and ppo_dir and status in ("ran", "budget"):
            after = ST.iters_done(ppo_dir)
            ST.record_slice_timing(ppo_dir, sl.get("start", 0), after, wall)
            try:
                health = ST.verify_training_ran(ppo_dir, sl.get("start", 0))
                note = (f"{note + '; ' if note else ''}iters "
                        f"{sl.get('start')}->{after}, max|kl| "
                        f"{health['max_abs_kl']:.1e}, "
                        f"lr dev {sl.get('lr_max_rel_dev', 0):.2f}")
            except ST.Poisoned as exc:
                MF.poison(man, run_dir, spec.name, str(exc))
                status, note = "poisoned", str(exc)[:600]
                print(f"           POISONED: {exc}")
            audit = _lr_audit(log_path, ST.lr_curves(
                sl.get("start", 0), sl.get("n", 1), sl.get("total", 1),
                lr_eff=sl.get("lr"))[0]) if sl else None
            if audit:
                res.plan["lr_audit"] = audit

        # ---- stamp every produced artifact, under the SAME fingerprint the
        # skip was judged against (inputs folded in for stages that declare
        # them, so the gate's verdict is tied to the candidate it judged).
        fp_stage, keys_stage = ST.effective(spec, fmt, fp)
        produced = []
        if status == "ran":
            for p in res.plan.get("produces", []):
                if not os.path.exists(p):
                    continue
                if p.endswith(".json") and _looks_like_weights(p):
                    try:
                        FP.verify_weights_head(p, fp)
                    except FP.HeadMismatch as exc:
                        MF.poison(man, run_dir, spec.name, str(exc))
                        status, note = "poisoned", str(exc)[:600]
                        print(f"           POISONED: {exc}")
                        break
                stamp = FP.write_stamp(p, fp_stage, keys_stage)
                MF.record_artifact(man, p, stamp)
                produced.append(p)
            if spec.kind == "ppo" and ppo_dir and status == "ran":
                FP.write_stamp(ppo_dir, fp_stage, keys_stage,
                               extra={"target_iters": spec.iters,
                                      "iters_done": ST.iters_done(ppo_dir)})

        MF.record_stage(man, spec.name, status, started, _iso(), res.cmd,
                        produced, note=note, plan=res.plan)
        MF.save(run_dir, man)
        if status in ("failed", "poisoned", "budget"):
            if status == "budget":
                break
            print(f"           {spec.name} {status}; continuing to the next stage")

    MF.end_slice(man, stopped_at=stopped,
                 remaining_s=budget_s - (time.time() - t0))
    MF.save(run_dir, man)
    print()
    print(MF.summary(man))
    return man


def _looks_like_weights(path):
    """A `rl.common.to_layers` JSON -- has policy.heads_version."""
    try:
        with open(path) as fh:
            head = fh.read(4096)
        return '"policy"' in head and '"heads_version"' in head
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", default=DEFAULT_JOB)
    ap.add_argument("--run", required=True, help="the run directory ({out})")
    ap.add_argument("--budget-s", type=float, default=39600.0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolution table and exit; executes nothing")
    ap.add_argument("--only", action="append", default=[],
                    help="run only this stage (repeatable)")
    ap.add_argument("--force", action="append", default=[],
                    help="discard a POISONED stage's run and restart it")
    ap.add_argument("--started-at", type=float, default=None,
                    help="epoch the OUTER budget started, so a nested runner's "
                         "clock is the real one")
    ap.add_argument("--lr-tol", type=float, default=None,
                    help="shorten a slice until the LR schedule is within this "
                         "relative deviation of the unsliced one (default: use "
                         "the wall and record the residual instead)")
    ap.add_argument("--probe-features", action="store_true",
                    help="play ONE BC episode to resolve n_features (off by "
                         "default: the fingerprint must not cost a game)")
    args = ap.parse_args()
    man = run_job(args.job, args.run, budget_s=args.budget_s, procs=args.procs,
                  dry_run=args.dry_run, only=args.only, force=args.force,
                  started_at=args.started_at, lr_tol=args.lr_tol,
                  probe_features=args.probe_features)
    if man.get("poisoned"):
        sys.exit(0)   # the manifest carries the verdict; the kernel keeps going


if __name__ == "__main__":
    main()
