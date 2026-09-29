"""What was trained on, with which head version, and where it stopped.

`<run>/manifest.json` is the only thing that survives a wall kill, a machine
change or a week, so it is written atomically after EVERY stage -- not at the
end of a slice, and certainly not at the end of a job. A Kaggle session that
dies at 11h59m leaves a manifest that says what ran, what was skipped, what
was quarantined, and which stage is poisoned.

Artifact records mirror `rl/datasets/MANIFEST.json` (source / count / bytes /
sha256) so the two can be read by the same eye.

Body:

    {"schema": 1, "run": ..., "fingerprint": {...}, "job": <job json inlined>,
     "slices": [{"idx", "started", "budget_s", "procs", "host", "fp",
                 "stages": [...], "stopped_at", "remaining_s"}],
     "artifacts": {path: {"path", "bytes", "sha256", "stamp"}},
     "poisoned": {stage: {"reason", "at"}},
     "gate": {"eval_json", "arm", "objective": [win, won_margin, lost_margin, n]},
     "best": {"path", "from_slice", "objective", "beat"}}
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket

SCHEMA = 1
NAME = "manifest.json"

STATUSES = ("ran", "skipped", "stale", "poisoned", "failed", "budget")


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def path_for(run_dir):
    return os.path.join(run_dir, NAME)


def blank(run_dir, job=None):
    return {"schema": SCHEMA, "run": os.path.abspath(run_dir),
            "created": _now(), "fingerprint": None, "job": job,
            "slices": [], "artifacts": {}, "poisoned": {},
            "gate": None, "best": None}


def load(run_dir, job=None):
    """The manifest, or a blank one. A truncated `.tmp` is ignored by
    construction: `save` never replaces the real file until the write
    completed, so a kill during a write leaves the PREVIOUS manifest intact."""
    p = path_for(run_dir)
    if not os.path.exists(p):
        return blank(run_dir, job)
    try:
        with open(p) as fh:
            man = json.load(fh)
    except (ValueError, OSError):
        return blank(run_dir, job)
    if man.get("schema") != SCHEMA:
        man["schema_was"] = man.get("schema")
        man["schema"] = SCHEMA
    man.setdefault("slices", [])
    man.setdefault("artifacts", {})
    man.setdefault("poisoned", {})
    if job is not None:
        man["job"] = job
    return man


def save(run_dir, man):
    """Atomic: write `.tmp`, fsync, os.replace."""
    os.makedirs(run_dir, exist_ok=True)
    p = path_for(run_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


# ------------------------------------------------------------------- slices
def begin_slice(man, *, budget_s, procs, host=None, fp=None):
    sl = {"idx": len(man["slices"]), "started": _now(),
          "budget_s": float(budget_s), "procs": int(procs),
          "host": host or socket.gethostname(), "fp": fp, "stages": [],
          "stopped_at": None, "remaining_s": None}
    man["slices"].append(sl)
    if fp is not None:
        man["fingerprint"] = fp
    return sl


def end_slice(man, *, stopped_at, remaining_s):
    if not man["slices"]:
        return
    man["slices"][-1]["stopped_at"] = stopped_at
    man["slices"][-1]["remaining_s"] = float(remaining_s)
    man["slices"][-1]["ended"] = _now()


def record_stage(man, name, status, started, ended, cmd, artifacts, note=None,
                 plan=None):
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    if not man["slices"]:
        begin_slice(man, budget_s=0, procs=0)
    row = {"name": name, "status": status, "started": started, "ended": ended,
           "cmd": cmd, "artifacts": list(artifacts or []), "note": note}
    if plan:
        row["plan"] = plan
    man["slices"][-1]["stages"].append(row)
    return row


def record_artifact(man, path, stamp):
    """Mirror of `rl/datasets/MANIFEST.json`'s artifact row."""
    rec = {"path": os.path.abspath(path),
           "bytes": (stamp or {}).get("bytes"),
           "sha256": (stamp or {}).get("sha256"),
           "stamp": stamp}
    man["artifacts"][os.path.abspath(path)] = rec
    return rec


def poison(man, run_dir, stage, reason):
    """Mark a stage poisoned. Survives a reload, blocks the skip AND the push.

    The Stage B signature (`policy_updates=0`, `onpolicy_kl` +3.7) looks like
    completed work to `job.resolve_step`, which checks only that the telemetry
    rows and eval.json exist. A poisoned stage is how that is made visible to
    the NEXT slice and to `rlv3/kaggle/push_checkpoints.sh`.
    """
    man.setdefault("poisoned", {})[stage] = {"reason": str(reason), "at": _now()}
    save(run_dir, man)
    return man["poisoned"][stage]


def unpoison(man, stage):
    man.get("poisoned", {}).pop(stage, None)


def is_poisoned(man, stage):
    return stage in (man.get("poisoned") or {})


# ------------------------------------------------------------------- resume
def resume_point(man):
    """-> {"stage", "iters_done", "target", "poisoned"}.

    The stage to start at after a kill: the first stage of the last slice that
    did not finish, else the last one recorded.
    """
    out = {"stage": None, "iters_done": 0, "target": 0,
           "poisoned": sorted((man.get("poisoned") or {}).keys())}
    for sl in reversed(man.get("slices") or []):
        for st in reversed(sl.get("stages") or []):
            plan = st.get("plan") or {}
            sli = plan.get("slice") or {}
            if sli:
                out["iters_done"] = int(sli.get("start", 0)) + (
                    int(sli.get("n", 0)) if st.get("status") == "ran" else 0)
                out["target"] = int(sli.get("total", 0))
            if out["stage"] is None:
                out["stage"] = st.get("name")
            if sli:
                return out
        if out["stage"] is not None:
            return out
    return out


def iters_recorded(man, stage="ppo"):
    """Iterations this manifest believes the PPO stage has completed."""
    k = 0
    for sl in man.get("slices") or []:
        for st in sl.get("stages") or []:
            if st.get("name") != stage or st.get("status") != "ran":
                continue
            sli = (st.get("plan") or {}).get("slice") or {}
            if sli:
                k = max(k, int(sli.get("start", 0)) + int(sli.get("n", 0)))
    return k


# ------------------------------------------------------------------ summary
def summary(man) -> str:
    """The one-screen table `--dry-run` and the kernel log print."""
    lines = []
    fp = man.get("fingerprint") or {}
    corpus = fp.get("corpus") or {}
    lines.append(f"run {man.get('run')}")
    lines.append(
        f"  heads v{fp.get('heads_version')}/{fp.get('heads_mode')} "
        f"logits={fp.get('n_action_logits')} nfeat={fp.get('n_features')} "
        f"board={fp.get('board')}/{fp.get('board_arm')}:{fp.get('board_ch')} "
        f"code={str(fp.get('code'))[:8]} engine={fp.get('engine')}")
    lines.append(
        f"  corpus draw n={corpus.get('n')} per_bin={corpus.get('per_bin')} "
        f"seed={corpus.get('seed')} counts={corpus.get('counts')} "
        f"sha={str(corpus.get('draw_sha'))[:8]}")
    for sl in man.get("slices") or []:
        lines.append(f"  slice {sl['idx']} {sl.get('started')} "
                     f"budget {float(sl.get('budget_s') or 0) / 3600:.2f}h "
                     f"procs {sl.get('procs')} -> {sl.get('stopped_at')}")
        for st in sl.get("stages") or []:
            sli = (st.get("plan") or {}).get("slice") or {}
            extra = ""
            if sli:
                extra = (f" [{sli['start']}..{sli['start'] + sli['n']}/"
                         f"{sli['total']} lr={sli['lr']:.3g} "
                         f"dev={sli['lr_max_rel_dev']:.2f}]")
            q = (st.get("plan") or {}).get("quarantined") or []
            if q:
                extra += f" [quarantined {len(q)}]"
            lines.append(f"    {st['name']:<9} {st['status']:<8}"
                         f"{extra} {st.get('note') or ''}")
    if man.get("poisoned"):
        for k, v in man["poisoned"].items():
            lines.append(f"  POISONED {k}: {v.get('reason')}")
    g = man.get("gate")
    if g:
        o = g.get("objective") or []
        lines.append(f"  gate {g.get('arm')} on {g.get('eval_json')}: "
                     + (f"win {o[0]:.3f} won {o[1]:+,.0f} lost {o[2]:+,.0f} "
                        f"n={o[3]}" if len(o) == 4 else str(o)))
    b = man.get("best")
    if b:
        o = b.get("objective") or []
        lines.append(f"  best {b.get('path')} (slice {b.get('from_slice')}) "
                     + (f"win {o[0]:.3f} beat {b.get('beat')}" if o else ""))
    n_art = len(man.get("artifacts") or {})
    unst = sum(1 for r in (man.get("artifacts") or {}).values()
               if not r.get("stamp"))
    lines.append(f"  artifacts {n_art} ({unst} unstamped)")
    return "\n".join(lines)


__all__ = ["SCHEMA", "load", "save", "blank", "path_for", "begin_slice",
           "end_slice", "record_stage", "record_artifact", "poison",
           "unpoison", "is_poisoned", "resume_point", "iters_recorded",
           "summary"]
