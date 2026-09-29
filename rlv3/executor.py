"""The rlv3 executor seam: placement and tile-local chore order as pure
task-list rewrites over kagent's shipped scheduler.

SEAM MECHANISM, verified in this repo, with no edit to kagent/, rl/ or
tools/. `kagent.policy.HeuristicPolicy.act` calls `scheduler.build_tasks`
and `scheduler.assign` as MODULE ATTRIBUTES, so rebinding those two names on
the live `kagent.scheduler` module intercepts every agent path -- champion,
knob bandit, execnet. `scheduler.EXEC` is a plain dict that accepts new
keys, so once `register()` has run, `set_exec` / `reset_exec` / `KAGG_EXEC`
and a rules JSON's `"exec"` list all take rlv3 flags with no change to the
instruments (checked: `EXEC['rlv3_probe']=False; set_exec(['rlv3_probe'])`
-> True; `reset_exec()` -> False).

Cross-turn state lives in `rlv3.ledger.LEDGER`, a module global reset on
step regression, because `DayPlan` has `__slots__` and `build_tasks` /
`assign` never receive `Memory`.

WHAT THE REWRITE MAY AND MAY NOT DO. `rlv3_build_tasks` returns a NEW list
whose tasks differ from the shipped ones only in `(x, y)` for growth ops and
in `pri` between tasks on the same tile. With `KAGG_RLV3_CHECK=1` these are
asserted every turn:

  (a) identical multiset of (op[0], op[1] if any, need) -- so the planner's
      allocation, the plant quota and the day's volume are untouched;
  (b) every (x, y) unlocked, and every growth tile one the ORIGINAL
      build_tasks offered for that op's tile pool;
  (c) every pri in [P_SURVIVAL, P_FILLER) or >= P_FILLER as it began, and
      the SET of tiles carrying P_SURVIVAL unchanged -- no survival task is
      created or destroyed;
  (d) the number of PLANT tasks per crop is unchanged.

`rlv3_assign` is a straight pass-through to the shipped `assign` unless
`mv_dest` is on, which is the measured-negative control arm.

ACTIVATION. `rl/session_eval.py` runs its workers with the SPAWN start
method, so a monkeypatch applied in the parent process is INVISIBLE to
them and the arm would play the SHIPPED executor while reporting as rlv3 --
the exact shape of the Stage B board arms that never trained. Install
through `rlv3/sitecustomize.py` and check a non-zero counter in `stats()`
from inside a worker before reading any arm as a result.
"""

from __future__ import annotations

import collections
import os

from kagent import scheduler as _sched_mod
from kagent.scheduler import P_FILLER, P_SURVIVAL
from rlv3 import choreorder, ledger, placement

RLV3_FLAGS = (
    "pl_wheat_adj", "pl_crop_cluster", "pl_water_first", "pl_pasture_ring", "pl_net",
    "co_chain_tile", "co_animal_first", "co_value", "mv_dest",
)
PL_FLAGS = tuple(f for f in RLV3_FLAGS if f.startswith("pl_"))
CO_FLAGS = choreorder.CO_FLAGS

PARAMS = {
    "cluster_radius": 2.0,
    "pasture_shed_dist": 4.0,
    "pl_net_conf": 0.5,
    "mv_dest_conf": 0.6,
    "conf_step": 0.05,
    "handfeed_target": 0.15,
    "dry_tol": 0.10,
    "contest_day": 18.0,
    # placement score weights; all-zero reproduces build_tasks exactly
    "w_wheat_adj": 3.0,
    "w_cluster": 2.0,
    "w_water_first": 1.0,
    "w_net": 1.0,
}

_ORIG_BUILD_TASKS = None
_ORIG_ASSIGN = None
_SCHED = None
_STATS = collections.Counter()
_NET = {"model": None, "cond": None, "failed": False}
_MV_SEEN = {}   # last value read out of ExecNetPolicy's cumulative counters


# ------------------------------------------------------------------ params
def parse_params(text):
    """'cluster_radius=3,mv_dest_conf=0.7' -> {name: float}. Unknown raise."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in PARAMS:
            raise ValueError(f"unknown rlv3 param {k!r}; known: {sorted(PARAMS)}")
        out[k] = float(v)
    return out


def set_params(values):
    PARAMS.update({k: float(v) for k, v in dict(values).items() if k in PARAMS})


def apply_env_params():
    if os.environ.get("KAGG_RLV3_PARAMS"):
        set_params(parse_params(os.environ["KAGG_RLV3_PARAMS"]))


# -------------------------------------------------------------- activation
def register(sched=None):
    """Add RLV3_FLAGS to scheduler.EXEC as False. Idempotent.

    MUST run before any `set_exec` naming an rlv3 flag, or `set_exec` raises
    ValueError on an unknown flag -- including the `KAGG_EXEC` read that
    `kagent.scheduler` performs at IMPORT time.
    """
    sched = sched or _sched_mod
    for f in RLV3_FLAGS:
        sched.EXEC.setdefault(f, False)


def install(sched=None):
    """Rebind build_tasks / assign on the live scheduler module."""
    global _ORIG_BUILD_TASKS, _ORIG_ASSIGN, _SCHED
    sched = sched or _sched_mod
    if _SCHED is not None:
        return
    register(sched)
    _ORIG_BUILD_TASKS = sched.build_tasks
    _ORIG_ASSIGN = sched.assign
    _SCHED = sched
    sched.build_tasks = rlv3_build_tasks
    sched.assign = rlv3_assign


def uninstall():
    global _ORIG_BUILD_TASKS, _ORIG_ASSIGN, _SCHED
    if _SCHED is None:
        return
    _SCHED.build_tasks = _ORIG_BUILD_TASKS
    _SCHED.assign = _ORIG_ASSIGN
    _SCHED = None
    _ORIG_BUILD_TASKS = _ORIG_ASSIGN = None


def is_installed():
    return _SCHED is not None


def stats():
    return dict(_STATS)


def reset_stats():
    _STATS.clear()
    _MV_SEEN.clear()


def reset_episode(step):
    """True when `step` starts a new episode; also rolls the ledger over."""
    return ledger.reset(step)


def active_flags(sched=None):
    """The rlv3 flags that are on RIGHT NOW, after the ledger's kill
    switches. A lever the ledger has switched off reads False here, so every
    downstream module sees one consistent view."""
    sched = sched or _SCHED or _sched_mod
    off = ledger.LEDGER.off
    return {f: bool(sched.EXEC.get(f)) and f not in off for f in RLV3_FLAGS}


def any_on(sched=None, names=RLV3_FLAGS):
    sched = sched or _SCHED or _sched_mod
    return any(sched.EXEC.get(f) for f in names)


# ------------------------------------------------------------- build_tasks
def rlv3_build_tasks(view, plan):
    tasks = _ORIG_BUILD_TASKS(view, plan)
    reset_episode(view.step)
    ledger.LEDGER.dawn_roll(view)
    params = ledger.LEDGER.correct(PARAMS, stats=_STATS)
    flags = active_flags()
    if not any(flags.values()):
        return tasks

    out = tasks
    if any(flags.get(f) for f in PL_FLAGS):
        prior = _prior(view, plan, tasks, flags, params)
        ctx = placement.make_ctx(view, plan, tasks, flags, params,
                                 regime=ledger.LEDGER.regime(view, params), stats=_STATS)
        out = placement.replace_growth_tasks(view, plan, out, prior, ctx)
    if any(flags.get(f) for f in CO_FLAGS):
        ctx = {
            "flags": flags,
            "params": params,
            "lead": ledger.LEDGER.lead,
            "regime": ledger.LEDGER.regime(view, params),
            "opp_standing": choreorder._opp_standing(view),
            "stats": _STATS,
        }
        out = choreorder.reprioritise(view, plan, out, ctx)

    if os.environ.get("KAGG_RLV3_CHECK"):
        check_invariants(view, tasks, out,
                         strict=not flags.get("pl_pasture_ring")
                         and os.environ["KAGG_RLV3_CHECK"] != "loose")
    return out


def _prior(view, plan, tasks, flags, params):
    if not flags.get("pl_net") or _NET["failed"]:
        return None
    try:
        if _NET["model"] is None:
            _NET["model"], _NET["cond"] = placenet_load()
        return placenet_prior(view, plan, tasks, _NET["model"], _NET["cond"],
                              params.get("pl_net_conf", 0.5))
    except Exception as exc:                      # a missing checkpoint must
        _NET["failed"] = True                     # not take the agent down;
        _STATS["pl_net.failed"] += 1              # it must be LOUD and inert
        import sys
        print(f"rlv3: pl_net disabled ({exc})", file=sys.stderr, flush=True)
        return None


def placenet_load():
    from rlv3 import placenet
    return placenet.load()


def placenet_prior(view, plan, tasks, model, cond, conf):
    from rlv3 import placenet
    return placenet.placement_prior(view, plan, tasks, model, cond, conf)


# ------------------------------------------------------------------ assign
def rlv3_assign(view, plan, tasks):
    flags = active_flags()
    if not flags.get("mv_dest"):
        actions = _ORIG_ASSIGN(view, plan, tasks)
        ledger.LEDGER.observe(view, actions)
        return actions

    sched = _SCHED or _sched_mod
    prev = sched.MOVE_STATS_ENABLED
    sched.MOVE_STATS_ENABLED = True
    sched.ASSIGN_LABEL.clear()
    try:
        actions = _ORIG_ASSIGN(view, plan, tasks)
        labels = dict(sched.ASSIGN_LABEL)
    finally:
        sched.MOVE_STATS_ENABLED = prev
    params = ledger.LEDGER.correct(PARAMS, stats=_STATS)
    try:
        if _NET["model"] is None and not _NET["failed"]:
            _NET["model"], _NET["cond"] = placenet_load()
        if _NET["model"] is not None:
            from rlv3 import placenet
            pol_stats = placenet.override_destinations(
                view, plan, tasks, actions, labels,
                _NET["model"], _NET["cond"], params.get("mv_dest_conf", 0.6))
            # ExecNetPolicy keeps its own cumulative counters with the right
            # labels (override:*, kept_low_conf, agree:*); mirror the deltas
            # rather than re-derive them, so the control arm reports exactly
            # what the measured v2 arm reported.
            for k, v in pol_stats.items():
                key = "mv_dest.override" if k.startswith("override:") else \
                      "mv_dest.agree" if k.startswith("agree:") else "mv_dest." + k
                _STATS[key] = _STATS.get(key, 0) + v - _MV_SEEN.get((id(pol_stats), k), 0)
                _MV_SEEN[(id(pol_stats), k)] = v
    except Exception as exc:
        _NET["failed"] = True
        _STATS["mv_dest.failed"] += 1
        import sys
        print(f"rlv3: mv_dest disabled ({exc})", file=sys.stderr, flush=True)
    ledger.LEDGER.observe(view, actions)
    return actions


# -------------------------------------------------------------- invariants
def _sig(t):
    return (t.op[0], t.op[1] if len(t.op) > 1 else None, t.need)


def check_invariants(view, before, after, strict=True):
    """Assert the four seam invariants. Raises AssertionError on a breach."""
    assert collections.Counter(_sig(t) for t in before) == \
        collections.Counter(_sig(t) for t in after), "rlv3 (a): op multiset changed"

    pools = {}
    for t in before:
        cls = placement.tile_class(t.op[0])
        if cls:
            pools.setdefault(cls, set()).add((t.x, t.y))
    wide = set().union(*[pools.get(c, set()) for c in ("PLANT", "BUILD")]) if pools else set()
    n = view.me.board_size
    for t in after:
        assert 0 <= t.x < n and 0 <= t.y < n, f"rlv3 (b): off board {(t.x, t.y)}"
        assert view.me.tile(t.x, t.y) != "LOCKED", f"rlv3 (b): locked tile {(t.x, t.y)}"
        cls = placement.tile_class(t.op[0])
        if cls is None:
            continue
        ok = pools.get(cls, set()) if (strict or cls == "PLACE") else wide
        assert (t.x, t.y) in ok, f"rlv3 (b): {t.op[0]} on {(t.x, t.y)} outside its pool"

    surv_before = {(t.x, t.y) for t in before if t.pri == P_SURVIVAL}
    surv_after = {(t.x, t.y) for t in after if t.pri == P_SURVIVAL}
    assert surv_before == surv_after, "rlv3 (c): survival tile set changed"
    n_surv_before = sum(1 for t in before if t.pri == P_SURVIVAL)
    n_surv_after = sum(1 for t in after if t.pri == P_SURVIVAL)
    assert n_surv_before == n_surv_after, "rlv3 (c): survival task count changed"
    band_before = sorted((0 if t.pri == P_SURVIVAL else 1 if t.pri < P_FILLER else 2)
                         for t in before)
    band_after = sorted((0 if t.pri == P_SURVIVAL else 1 if t.pri < P_FILLER else 2)
                        for t in after)
    assert band_before == band_after, "rlv3 (c): a task crossed a priority band edge"

    plants_before = collections.Counter(t.op[1] for t in before if t.op[0] == "PLANT")
    plants_after = collections.Counter(t.op[1] for t in after if t.op[0] == "PLANT")
    assert plants_before == plants_after, "rlv3 (d): PLANT count per crop changed"
    return True
