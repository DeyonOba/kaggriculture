"""Stage resolution -- a strict superset of the `tools/kaggle_job/job.py`
contract, with the two things that contract cannot do on its own: a stamp
check on every skip, and a budget-aware PPO slice instead of a count-aware one.

WHAT IS REUSED, NOT COPIED. `job.resolve_step` already implements skip_if /
require / min_left_s / resume_run, and `tests/test_kaggle_job.py` already pins
its behaviour. This module loads that file with the same importlib pattern and
CALLS it, then layers on top:

  1. a `skip: output exists` verdict is re-opened with `rlv3.fingerprint.check`
     on that stage's `depends` keys; stale means quarantine-and-run,
  2. for a PPO stage the iteration rewrite is replaced by `ppo_slice()`, which
     reads the run's own measured seconds/iteration and the wall remaining and
     holds back the gate's reserve,
  3. the resulting command is put through `assert_slice_invariant()`, whose
     findings are a refusal rather than a warning.

THREE THINGS INSIDE rl/train.py DO NOT SURVIVE A PROCESS BOUNDARY, and this
module's job is to neutralise them, not to pretend otherwise.

  * The rating-bin curriculum's rolling frontier (`bin_wins`, initialised
    [1, 2] per bin) is in-memory, so it restarts at every resume. We pin
    `--session-mix eval`, whose band weights come from
    `_eval_band_weights(session_index.json)` and are therefore slice-invariant
    AND matched to the 995-game gate's composition.
  * `best_score` / `evals_since_best` restart too, and `cool =
    PLATEAU_COOL ** evals_since_best` feeds straight into the learning rate.
    We pin `--plateau-cool 1.0` and `--patience 99`, so neither the cool-down
    nor the early stop depends on where a slice happened to start. Selection
    is ours, on the paired gate (`rlv3.promote`), never on the 18-game
    in-training SESSION-WR.
  * Adam's moments are not serialised (`ppo_phase` builds the optimiser
    fresh), so a one-iteration slice pays a full optimiser restart for one
    iteration of work. `ppo_slice` refuses anything below `min_iters`.

WHAT IS CONTINUOUS BY CONSTRUCTION, and is asserted rather than assumed: the
world seeds (`world_base + it*episodes + e`), the eval cadence
(`(it+1) % eval_every`), the value warmup (`it < WARMUP_ITERS`) and `shaping_w`
(which divides by `total_iters`) all key off the GLOBAL iteration counter that
`--start-iter` restores.

THE ONE THAT IS NOT CONTINUOUS: the learning rate. `rl/train.py:710` is

    decay = max(LR_FLOOR, 1.0 - it / max(1, iters))

and `iters` there is the SLICE length, not `total_iters` -- unlike `shaping_w`
on line 619, which uses `total_iters` and is continuous. Resume at k=10 of
T=20 and the slice sits at LR_FLOOR from its first iteration where an unsliced
run would be at 0.50. `lr_compensation()` prices that and returns the --lr /
--ent-coef that reproduce the unsliced schedule as closely as a constant can.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field

from rlv3 import fingerprint as FP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fingerprint key GROUPS a stage may declare. "action" carries the board flag
# as well, because flipping KAGG_BOARD changes the net's input width.
GROUPS = {
    "action": FP.ACTION_KEYS + FP.BOARD_KEYS,
    "corpus": ("corpus",),
    "code": ("code",),
    "engine": ("engine",),
}

_JOB = None


def load_job_module():
    """`tools/kaggle_job/job.py`, loaded the way tests/test_kaggle_job.py does.

    Importing it by path rather than as a package keeps `tools/` untouched and
    keeps the module-level `_discover()` work (which walks /kaggle/input) to
    the one-time cost it already is.
    """
    global _JOB
    if _JOB is None:
        path = os.path.join(REPO, "tools", "kaggle_job", "job.py")
        spec = importlib.util.spec_from_file_location("kjob_rlv3", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _JOB = mod
    return _JOB


def _train_consts():
    """rl.train's real constants, imported lazily (it pulls torch)."""
    from rl import train as T
    return T


# ---------------------------------------------------------------- stage spec
@dataclass(frozen=True)
class StageSpec:
    name: str
    cmd: str
    kind: str = "shell"                       # "shell" | "ppo" | "gate"
    depends: tuple = FP.ACTION_KEYS           # fingerprint keys, or GROUPS names
    skip_if: tuple = ()
    require: tuple = ()
    produces: tuple = ()
    inputs: tuple = ()                        # files whose CONTENT this output
                                              # depends on (folded into the stamp)
    resume_run: str | None = None
    iters: int = 0
    min_iters: int = 3
    min_left_s: int = 300
    reserve: bool = False                     # budget is held back for it
    env: dict = field(default_factory=dict)   # KAGG_BOARD, KAGG_BOARD_ARM, ...
    note: str = ""

    @property
    def depend_keys(self) -> tuple:
        """`depends` with any GROUPS name expanded, de-duplicated, ordered."""
        out, seen = [], set()
        for d in self.depends:
            for k in GROUPS.get(d, (d,)):
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return tuple(out)


@dataclass
class Resolution:
    cmd: str | None
    reason: str
    plan: dict = field(default_factory=dict)

    @property
    def will_run(self) -> bool:
        return self.cmd is not None


def from_json(obj) -> StageSpec:
    """One stage of `rlv3/jobs/*.json` -> StageSpec."""
    def tup(k):
        v = obj.get(k, ())
        return (v,) if isinstance(v, str) else tuple(v)

    return StageSpec(
        name=obj["name"], cmd=obj["cmd"], kind=obj.get("kind", "shell"),
        depends=tup("depends") or FP.ACTION_KEYS,
        skip_if=tup("skip_if"), require=tup("require"), produces=tup("produces"),
        inputs=tup("inputs"),
        resume_run=obj.get("resume_run"), iters=int(obj.get("iters", 0)),
        min_iters=int(obj.get("min_iters", 3)),
        min_left_s=int(obj.get("min_left_s", 300)),
        reserve=bool(obj.get("reserve", False)),
        env=dict(obj.get("env", {})), note=obj.get("note", ""),
    )


def _job_spec(stage: StageSpec) -> dict:
    d = {"cmd": stage.cmd, "min_left_s": stage.min_left_s}
    if stage.skip_if:
        d["skip_if"] = list(stage.skip_if)
    if stage.require:
        d["require"] = list(stage.require)
    if stage.resume_run:
        d["resume_run"] = stage.resume_run
        d["iters"] = stage.iters
    return d


# ------------------------------------------------------------ PPO slicing
def iters_done(run_dir) -> int:
    """Iterations already appended to this run's telemetry. job.py's own test."""
    return load_job_module()._telemetry_iters(run_dir)


_ITER_LOG = re.compile(r"^iter\s+(\d+):.*\((\d+(?:\.\d+)?)s\)\s*$")


def record_slice_timing(run_dir, iters_before, iters_after, wall_s):
    """Append this slice's measured cost to `<run>/slices.jsonl`.

    The trainer's telemetry rows carry no wall clock, so the run would have no
    memory of how fast this machine is. This is that memory, and it is what
    makes the next slice's plan self-correct to the machine it woke up on
    rather than to a number written in a config yesterday.
    """
    if iters_after <= iters_before:
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "slices.jsonl"), "a") as fh:
        fh.write(json.dumps({
            "iters_before": int(iters_before), "iters_after": int(iters_after),
            "wall_s": float(wall_s),
            "sec_per_iter": float(wall_s) / (iters_after - iters_before),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")


def sec_per_iter(run_dir, default=900.0) -> float:
    """Measured seconds an iteration, from this run's own record.

    Layered, cheapest first: our own `slices.jsonl`; any wall field the
    trainer's telemetry rows happen to carry; the trainer's stdout, whose
    per-iteration line ends `(123s)`. `default` on fewer than 2 observations,
    because one sample of a machine that was also running a 5-process eval is
    not a rate.
    """
    vals = []
    p = os.path.join(run_dir, "slices.jsonl")
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    try:
                        vals.append(float(json.loads(line)["sec_per_iter"]))
                    except (ValueError, KeyError):
                        pass
    if len(vals) >= 2:
        return statistics.median(vals)

    t = os.path.join(run_dir, "telemetry.jsonl")
    stamps = []
    if os.path.exists(t):
        with open(t) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for k in ("wall", "t", "elapsed", "sec", "iter_s"):
                    if isinstance(row.get(k), (int, float)):
                        stamps.append(float(row[k]))
                        break
    if len(stamps) >= 2:
        d = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if len(d) >= 2:
            return statistics.median(d)
        if stamps and all(v > 0 for v in stamps) and len(stamps) >= 2:
            return statistics.median(stamps)

    logs = sorted(glob.glob(os.path.join(run_dir, "*.log")))
    parent = os.path.dirname(run_dir.rstrip("/"))
    logs += sorted(glob.glob(os.path.join(parent, os.path.basename(run_dir.rstrip("/")) + "*.log")))
    seen = []
    for lg in logs:
        try:
            with open(lg, errors="replace") as fh:
                for line in fh:
                    m = _ITER_LOG.match(line.strip())
                    if m:
                        seen.append(float(m.group(2)))
        except OSError:
            continue
    if len(seen) >= 2:
        return statistics.median(seen)
    if vals:
        return vals[0]
    return float(default)


@dataclass
class Slice:
    start: int          # k: global iteration this slice begins at
    n: int              # m: iterations this slice will run
    total: int          # T: the run's target
    lr: float
    ent: float
    dev: float          # worst relative deviation of the realised LR schedule
    deadline_s: float   # wall this slice may use
    sec_per_iter: float = 0.0
    reason: str = ""


def lr_compensation(k, m, T, lr=None, ent=None, floor=None):
    """-> (lr_eff, ent_eff, max_rel_dev) for a slice [k, k+m) of a run of T.

    The trainer computes, per iteration,

        decay = max(LR_FLOOR, 1.0 - it / max(1, iters))     # iters = SLICE length
        lr_now  = LR       * decay * PLATEAU_COOL**evals_since_best
        ent_now = ENT_COEF * decay

    and we pin `--plateau-cool 1.0`, so `cool` is 1 and the only defect is
    that `decay` divides by the slice length. Let

        g(it) = decay_slice(it) / decay_unsliced(it)

    and pass `--lr s*LR`. The realised-over-intended ratio is then s*g(it), and
    the constant s that minimises the WORST relative deviation over the slice
    equalises the two ends: s = 2 / (g_min + g_max), giving

        max_rel_dev = (g_max - g_min) / (g_max + g_min).

    k=0, m=T is the unsliced case: g == 1, s == 1, dev == 0. k=10, m=10, T=20
    is the spec's worked example: decay_slice is LR_FLOOR for the whole slice
    while decay_unsliced runs 0.50 -> 0.15, so g runs 0.30 -> 1.00 and no
    constant can do better than 0.54. That residual is recorded, per slice, in
    the manifest -- so nobody reads a sliced PPO number as an unsliced one.
    """
    T_ = _train_consts()
    lr = T_.LR if lr is None else lr
    ent = T_.ENT_COEF if ent is None else ent
    floor = T_.LR_FLOOR if floor is None else floor
    m = max(1, int(m))
    T = max(1, int(T))
    gs = []
    for it in range(int(k), int(k) + m):
        d_slice = max(floor, 1.0 - it / float(m))
        d_true = max(floor, 1.0 - it / float(T))
        gs.append(d_slice / d_true)
    g_lo, g_hi = min(gs), max(gs)
    s = 2.0 / (g_lo + g_hi)
    dev = (g_hi - g_lo) / (g_hi + g_lo)
    return lr * s, ent * s, dev


def lr_curves(k, m, T, lr=None, floor=None, lr_eff=None):
    """-> (intended[], realised[]) per iteration. Used by the tests and by the
    driver's log audit, which parses the trainer's own `lr=` print."""
    T_ = _train_consts()
    lr = T_.LR if lr is None else lr
    floor = T_.LR_FLOOR if floor is None else floor
    lr_eff = lr if lr_eff is None else lr_eff
    intended, realised = [], []
    for it in range(int(k), int(k) + max(1, int(m))):
        intended.append(lr * max(floor, 1.0 - it / float(max(1, T))))
        realised.append(lr_eff * max(floor, 1.0 - it / float(max(1, m))))
    return intended, realised


def ppo_slice(run_dir, target_iters, remaining_s, reserve_s, min_iters=3,
              lr_tol=None, default_sec_per_iter=900.0):
    """-> Slice or None. The budget-aware replacement for job.py's `--iters`
    rewrite, which only ever subtracts the count already done.

        k = iters_done(run_dir)
        n = clamp(floor((remaining_s - reserve_s) / sec_per_iter), 0, T - k)

    `n < min_iters` returns None: the stage is skipped WHOLE, so the gate
    still gets its hour and Adam is not re-initialised to buy one iteration.

    `lr_tol` is OFF by default and this is a deliberate, documented trade. The
    spec asks for two things that fight: maximise slice length subject to the
    reserve, and keep the LR schedule within 0.15 of the unsliced one. They
    cannot both hold -- a constant cannot track a ramp, and honouring 0.15 at
    T=20 caps a slice near THREE iterations however much wall is free. Set
    `lr_tol` to shorten the slice until the deviation fits (never below
    `min_iters`); leave it None to use the wall and carry the residual into
    the manifest instead.
    """
    k = iters_done(run_dir)
    target = int(target_iters)
    if k >= target:
        return None
    spi = sec_per_iter(run_dir, default=default_sec_per_iter)
    usable = float(remaining_s) - float(reserve_s)
    n = int(math.floor(usable / spi)) if spi > 0 else 0
    n = max(0, min(n, target - k))
    if n < min_iters:
        return None
    reason = "budget"
    if lr_tol is not None:
        while n > min_iters and lr_compensation(k, n, target)[2] > lr_tol:
            n -= 1
        if lr_compensation(k, n, target)[2] > lr_tol:
            reason = "budget; LR tolerance NOT met at min_iters"
        else:
            reason = "budget, shortened to hold the LR tolerance"
    lr_eff, ent_eff, dev = lr_compensation(k, n, target)
    return Slice(start=k, n=n, total=target, lr=lr_eff, ent=ent_eff, dev=dev,
                 deadline_s=min(usable, n * spi * 1.6), sec_per_iter=spi,
                 reason=reason)


# ------------------------------------------------------------- the invariant
_FLAG = {
    "iters": r"--iters\s+(\S+)",
    "start_iter": r"--start-iter\s+(\S+)",
    "plateau_cool": r"--plateau-cool\s+(\S+)",
    "patience": r"--patience\s+(\S+)",
    "session_mix": r"--session-mix\s+(\S+)",
    "lr": r"--lr\s+(\S+)",
    "ent_coef": r"--ent-coef\s+(\S+)",
    "resume": r"--resume\s+(\S+)",
    "run": r"--run\s+(\S+)",
}


def _flags(cmd):
    return {k: (m.group(1) if (m := re.search(p, cmd)) else None)
            for k, p in _FLAG.items()}


def assert_slice_invariant(cmd, k, m, T, run_dir=None) -> list:
    """-> the list of problems with a PPO command. Empty means safe to run.

    A problem here is a REFUSAL, not a warning: every entry is something that
    would make the sliced run a different experiment from the unsliced one,
    and this project has been burned twice by a locally-better arm rating
    worse (`kaggriculture-selector-objective`) and once by ablations that were
    silently un-paired (`kaggriculture-fit-determinism`).
    """
    T_ = _train_consts()
    f = _flags(cmd)
    bad = []

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if num(f["iters"]) != float(m):
        bad.append(f"--iters is {f['iters']!r}, this slice runs {m}")
    si = 0.0 if f["start_iter"] is None else num(f["start_iter"])
    if si != float(k):
        bad.append(f"--start-iter is {f['start_iter']!r}, the run is at {k} "
                   f"(world seeds, eval cadence, warmup and shaping_w all key "
                   f"off the global counter)")
    pc = num(f["plateau_cool"])
    if pc is None:
        bad.append(f"--plateau-cool missing: the trainer would use "
                   f"PLATEAU_COOL={T_.PLATEAU_COOL}, and evals_since_best "
                   f"resets at every resume, so the LR cool-down would depend "
                   f"on where the slice started")
    elif abs(pc - 1.0) > 1e-9:
        bad.append(f"--plateau-cool {pc} != 1.0 (cool = PLATEAU_COOL ** "
                   f"evals_since_best, and evals_since_best resets at resume)")
    pat = num(f["patience"])
    if pat is None or pat < 99:
        bad.append(f"--patience is {f['patience']!r}; it is slice-local and "
                   f"selection is ours on the paired gate -- pin >= 99")
    if f["session_mix"] != "eval":
        bad.append(f"--session-mix is {f['session_mix']!r}; bin_wins is "
                   f"in-memory so the rolling frontier restarts every slice. "
                   f"Pin 'eval' (band weights from session_index.json)")
    if f["lr"] is None or f["ent_coef"] is None:
        bad.append("--lr / --ent-coef compensation missing: rl/train.py:710 "
                   "divides the LR decay by the SLICE length, so an "
                   "uncompensated resume runs at the wrong learning rate")
    if k > 0:
        r = f["resume"]
        if not r or not r.endswith("last.pt"):
            bad.append(f"--resume is {r!r}; a slice at k={k} must resume from "
                       f"<run>/last.pt")
        elif run_dir and os.path.abspath(r) != os.path.abspath(
                os.path.join(run_dir, "last.pt")):
            bad.append(f"--resume {r} is not {run_dir}/last.pt")
    return bad


# ------------------------------------------------------------ trainer health
class Poisoned(RuntimeError):
    """A slice appended rows that prove no training happened."""


def verify_training_ran(run_dir, since_iter) -> dict:
    """Read back the telemetry rows THIS slice appended. Raise if they lie.

    `kaggriculture-silent-trainer-failures`: Stage B's board arms ran twenty
    iterations at `policy_updates=0` with `onpolicy_kl` +3.7 nats -- no trunk
    was serialised, so the worker and the trainer were different policies and
    the KL stop fired on the first minibatch of every iteration. Money means
    printed happily the whole time.

    A resumable runner makes that strictly WORSE: the next slice would see
    telemetry rows >= target plus an eval.json, which is exactly
    `job.resolve_step`'s completion test, and skip the stage forever. So the
    rows are read back, and a failure marks the stage POISONED -- which blocks
    both the skip and the checkpoint push.
    """
    T_ = _train_consts()
    p = os.path.join(run_dir, "telemetry.jsonl")
    rows = []
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    new = [r for r in rows if int(r.get("iter", -1)) >= int(since_iter)]
    if not new:
        raise Poisoned(f"{run_dir}: the slice appended no telemetry rows at or "
                       f"after iter {since_iter}")
    # Grouped by KIND, not by row: the Stage B signature is three separate
    # symptoms of one cause, and a message truncated to the first few rows
    # would show `onpolicy_kl` twice and never mention that no update ran.
    bad = {}

    def flag(kind, msg):
        bad.setdefault(kind, [0, msg])[0] += 1

    for r in new:
        it = int(r.get("iter", -1))
        warm = bool(r.get("warmup", it < T_.WARMUP_ITERS))
        kl = r.get("onpolicy_kl")
        if kl is None or not math.isfinite(float(kl)) or abs(float(kl)) > T_.ONPOLICY_KL_TOL:
            flag("onpolicy_kl",
                 f"iter {it}: onpolicy_kl={kl} (tolerance "
                 f"{T_.ONPOLICY_KL_TOL}) -- the rollouts were played by a "
                 f"different policy than the one being trained")
        if r.get("kl_stop_at") is not None:
            flag("kl_stop", f"iter {it}: kl_stop_at={r['kl_stop_at']} -- the "
                            f"early stop fired inside the update")
        if not warm and int(r.get("policy_updates", 0)) <= 0:
            flag("policy_updates",
                 f"iter {it}: policy_updates=0 outside warmup "
                 f"(WARMUP_ITERS={T_.WARMUP_ITERS})")
    if bad:
        n = sum(c for c, _ in bad.values())
        raise Poisoned(
            f"{run_dir}: {n} unhealthy telemetry readings over {len(new)} rows "
            f"-- " + "; ".join(f"[x{c}] {m}" for c, m in bad.values()))
    return {
        "rows": len(new), "first_iter": int(new[0].get("iter", -1)),
        "last_iter": int(new[-1].get("iter", -1)),
        "max_abs_kl": max(abs(float(r.get("onpolicy_kl") or 0.0)) for r in new),
        "min_updates": min(int(r.get("policy_updates", 0)) for r in new
                           if not bool(r.get("warmup", False))) if any(
                               not bool(r.get("warmup", False)) for r in new) else None,
    }


# ------------------------------------------------------------------- resolve
def _fmt_paths(vals, fmt):
    return [v.format(**fmt) for v in vals]


def input_digest(paths):
    """sha1 over the CONTENT of a stage's declared inputs.

    Without this the gate skips itself out of existence. Its own output,
    `session_eval.json`, is still there after PPO appends another ten
    iterations, and its declared dependencies (action space, corpus draw) have
    not moved -- so the stamp says fresh and the 995-game eval never re-runs
    against the checkpoint it is supposed to be judging. Folding the
    candidate's bytes into the gate's stamp is what ties the verdict to the
    thing it is a verdict about.
    """
    rows = []
    for p in sorted(paths):
        rows.append((os.path.basename(p),
                     FP._sha256_file(p) if os.path.isfile(p) else None))
    return FP._sha1(json.dumps(rows).encode())


def effective(stage: StageSpec, fmt, fp):
    """-> (fp_eff, keys). The fingerprint this stage's artifacts are stamped
    under, with its input digest folded in when it declares one."""
    keys = stage.depend_keys
    if not stage.inputs:
        return fp, keys
    fp_eff = dict(fp)
    fp_eff["inputs"] = input_digest(_fmt_paths(stage.inputs, fmt))
    return fp_eff, keys + ("inputs",)


def _apply_slice(cmd, sl: Slice):
    """Rewrite --iters / --lr / --ent-coef and append --start-iter."""
    cmd = re.sub(r"--iters\s+\S+", f"--iters {sl.n}", cmd)
    for flag, val in (("--lr", f"{sl.lr:.6g}"), ("--ent-coef", f"{sl.ent:.6g}")):
        if re.search(rf"{flag}\s+\S+", cmd):
            cmd = re.sub(rf"{flag}\s+\S+", f"{flag} {val}", cmd)
        else:
            cmd += f" {flag} {val}"
    cmd = re.sub(r"\s--start-iter\s+\S+", "", cmd)
    if sl.start > 0:
        cmd += f" --start-iter {sl.start}"
    return cmd


def resolve(stage: StageSpec, fmt, remaining_s, fp, run_dir,
            reserve_s=0.0, lr_tol=None, quarantine_stale=True) -> Resolution:
    """The one entry point the driver calls. -> Resolution(cmd, reason, plan).

    Order matters: job.py first (so its four behaviours stay THE definition of
    skip/require/budget/resume), then the stamp check on any skip it granted,
    then the slice plan, then the invariant.
    """
    job = load_job_module()
    fp, keys = effective(stage, fmt, fp)
    plan = {"stage": stage.name, "kind": stage.kind, "depends": list(keys)}
    if stage.inputs:
        # carried so the driver stamps the OUTPUT under exactly the fingerprint
        # the skip was judged against
        plan["fp_eff"] = {k: fp.get(k) for k in keys}
        plan["inputs"] = _fmt_paths(stage.inputs, fmt)

    cmd, why = job.resolve_step(_job_spec(stage), fmt, remaining_s)

    # (2) a granted skip is re-opened and checked against today's fingerprint.
    if cmd is None and why.startswith("skip: output exists"):
        hits = [p for p in _fmt_paths(stage.skip_if, fmt) if os.path.exists(p)]
        stale = []
        for p in hits:
            fresh, reason = FP.check(p, fp, keys)
            if not fresh:
                stale.append((p, reason))
        if not stale:
            plan["checked"] = hits
            return Resolution(None, f"skip: fresh {os.path.basename(hits[0])}", plan)
        for p, reason in stale:
            if quarantine_stale:
                dest = FP.quarantine(p, run_dir, reason)
                plan.setdefault("quarantined", []).append(
                    {"path": p, "reason": reason, "moved_to": dest})
            else:
                plan.setdefault("stale", []).append({"path": p, "reason": reason})
        if not quarantine_stale:
            return Resolution(None, f"stale: {stale[0][1]}", plan)
        # re-resolve now that the stale outputs are gone
        cmd, why = job.resolve_step(_job_spec(stage), fmt, remaining_s)
        why = f"run (stale: {stale[0][1]})"

    if cmd is None:
        return Resolution(None, why, plan)

    # (3) a PPO stage plans its own slice; job.py's count-only rewrite is undone.
    if stage.kind == "ppo":
        ppo_dir = (stage.resume_run or "").format(**fmt) or run_dir
        # A PPO run dir is not a skip_if hit, so it would otherwise never be
        # checked -- and resuming it is the single most dangerous reuse in the
        # graph: `last.pt` carries the net's weights under the action space it
        # was trained in. The run is stamped before the FIRST slice launches
        # (so a wall kill still leaves a valid stamp) and re-checked here.
        stale_note = ""
        if os.path.isdir(ppo_dir) and iters_done(ppo_dir) > 0:
            fresh, reason = FP.check(ppo_dir, fp, keys, content=False)
            if not fresh:
                if not quarantine_stale:
                    return Resolution(None, f"stale run: {reason}", plan)
                dest = FP.quarantine(ppo_dir, run_dir, f"ppo run: {reason}")
                plan.setdefault("quarantined", []).append(
                    {"path": ppo_dir, "reason": reason, "moved_to": dest})
                stale_note = f"stale run: {reason}; "
        # job.resolve_step may already have rewritten --resume/--iters; rebuild
        # from the stage's own command so the slice is the only author.
        cmd = stage.cmd.format(**fmt)
        sl = ppo_slice(ppo_dir, stage.iters, remaining_s, reserve_s,
                       min_iters=stage.min_iters, lr_tol=lr_tol)
        if sl is None:
            k = iters_done(ppo_dir)
            if k >= stage.iters:
                return Resolution(None, f"skip: {ppo_dir} at target "
                                        f"({k}/{stage.iters} iterations)", plan)
            spi = sec_per_iter(ppo_dir)
            return Resolution(None, (
                f"skip: not enough budget for a useful slice "
                f"({(remaining_s - reserve_s) / 3600:.2f}h usable at "
                f"{spi / 60:.1f} min/iter, min_iters={stage.min_iters})"), plan)
        if sl.start > 0:
            cmd = re.sub(r"--resume\s+\S+",
                         f"--resume {os.path.join(ppo_dir, 'last.pt')}", cmd)
        cmd = _apply_slice(cmd, sl)
        plan["slice"] = {"start": sl.start, "n": sl.n, "total": sl.total,
                         "lr": sl.lr, "ent": sl.ent, "lr_max_rel_dev": sl.dev,
                         "sec_per_iter": sl.sec_per_iter,
                         "deadline_s": sl.deadline_s, "reason": sl.reason}
        plan["run_dir"] = ppo_dir
        # (4) the invariant. A problem is a refusal, not a warning.
        bad = assert_slice_invariant(cmd, sl.start, sl.n, sl.total, run_dir=ppo_dir)
        if bad:
            raise RuntimeError(f"{stage.name}: slice invariant violated:\n  - "
                               + "\n  - ".join(bad) + f"\ncommand: {cmd}")
        why = (f"{stale_note}run slice {sl.start}..{sl.start + sl.n} of "
               f"{sl.total} (lr {sl.lr:.3g}, dev {sl.dev:.2f})")

    plan["produces"] = _fmt_paths(stage.produces, fmt)
    return Resolution(cmd, why, plan)


def load_stages(job_path):
    """-> (cfg, [StageSpec]) from `rlv3/jobs/*.json`."""
    with open(job_path) as fh:
        cfg = json.load(fh)
    return cfg, [from_json(s) for s in cfg["stages"]]


__all__ = [
    "GROUPS", "StageSpec", "Resolution", "Slice", "Poisoned",
    "load_job_module", "from_json", "load_stages", "resolve",
    "iters_done", "sec_per_iter", "record_slice_timing", "ppo_slice",
    "lr_compensation", "lr_curves", "assert_slice_invariant",
    "verify_training_ran",
]
