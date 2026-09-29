"""rlv3.arm -- make `rlv3` a dev arm kind WITHOUT editing rl/rollout.py.

`rl.rollout._make_dev_policy_agent` knows three kinds: `champion`, `knobs`
and `execnet`. This module adds a fourth by wrapping that function in place,
so every existing kind still goes to the shipped implementation and only
`{"kind": "rlv3", ...}` comes here. Nothing under rl/ is modified on disk.

IT HAS GRADUATED (074ba51), AND THAT CHANGED WHAT THIS MODULE IS FOR.
Both hooks now exist on disk:

    rl/rollout.py:360       if spec.get("kind") == "rlv3": ... make_dev_agent(spec)
    rl/session_eval.py:193  elif parts[1] == "rlv3": spec = rlv3.arm.parse(...)

So `rl/session_eval.py` reaches rlv3 through `rollout.run_episode`, and only
`parse()` below is on that path. `register`, `unregister`, `is_registered`,
`run_episode`, `make_jobs` and `ledger_report` are the pre-graduation
monkeypatch route and are now used only by in-process callers (smoke,
scorecard_hook) -- the paired gate does not go through any of them.

WHICH MEANS THE LIVENESS ASSERTION BELOW IS NOT ARMED ON THE GATE. See the
LIVENESS note further down: `run_episode` raises when a job declares itself
rlv3 and `make_plan == 0`, and `ledger_report` is what puts `days_live` on the
result row. Neither runs when `rl/rollout.py` dispatches directly. Until that
check is moved into the rollout path, a dead rlv3 arm on the 995-game gate is
indistinguishable from a null one -- the exact failure this package was built
to make impossible.

TWO WAYS IN, and picking the wrong one is how an arm comes back DEAD rather
than null:

  (1) SAME PROCESS -- a trace, a smoke test, `tools/agent_scorecard.py` run
      in-process. `register()` once, then call `rl.rollout.run_episode`
      normally.

  (2) A POOL -- `rl/session_eval.py`, `--procs N`. `multiprocessing` starts
      spawn on this machine, so a patch applied in the PARENT is invisible in
      the worker: the worker would import a clean `rl.rollout`, hit the
      `else` branch and raise `unknown learner_policy kind 'rlv3'`. That is
      loud, which is the good case. Use `rlv3.arm.run_episode` as the pool
      function instead of `rollout.run_episode` -- it registers inside the
      worker on first call and then delegates. It is a module-level function,
      so it pickles.

LIVENESS. `run_episode` asserts the rlv3 plan provider was actually called
before returning a result, and puts `rlv3` counters on the returned dict under
`extra`. An arm that reports itself as rlv3 with `make_plan == 0` did not play
rlv3, and a zero there raises rather than returning a number nobody can tell
apart from a null result.

LIVENESS, ONE LEVEL DOWN. The plan provider is not the only thing that can
quietly stop being rlv3 mid-episode: `rlv3.ledger.correct()` switches an
executor lever off for the rest of the season the moment its own metric
stops paying, and a lever killed on day 2 means 28 days of the SHIPPED
executor. That is the same DEAD-not-null failure, so `ledger_report` puts
`days_live` per declared lever on `result["extra"]["rlv3"]` and the ledger
bumps `<flag>.killed_day<d>` into the executor's `_STATS`, which already
reaches `exec_stats` in smoke.py and scorecard_hook.py.
"""

from __future__ import annotations

import os

from rlv3 import agent as AGENT

_PATCHED = None          # the shipped _make_dev_policy_agent, while wrapped


# --------------------------------------------------------------- arm strings
def _split_top(text):
    """Split on commas OUTSIDE brackets, so `lam_grid=[0,0.5,1]` survives."""
    out, buf, depth = [], [], 0
    for ch in text or "":
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [s for s in out if s.strip()]


def parse(text=""):
    """'', 'full', 'full:lam=0.5,band=3' -> a spec dict for make_dev_agent.

    The grammar is deliberately the same shape as `dev:execnet:<a>:<b>`: a
    preset name, then comma-separated `key=value` overrides. Values are read
    as JSON when they parse as JSON (so `lam=0.5`, `select=true`,
    `exec_flags=["pl_wheat_adj"]` all work) and as a string otherwise.
    """
    import json

    text = (text or "").strip()
    preset, _, rest = text.partition(":")
    spec = {"kind": "rlv3", "preset": preset or "recommended"}
    for item in _split_top(rest):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k = k.strip()
        try:
            spec[k] = json.loads(v)
        except ValueError:
            spec[k] = v
    AGENT.config_from_spec(spec)          # validate now, not inside a worker
    return spec


def label(spec):
    return AGENT.config_from_spec(spec).label


# ------------------------------------------------------------ lever liveness
def ledger_report(cfg):
    """How many days each DECLARED executor lever was actually live.

    `rlv3.ledger.correct()` is a within-episode kill switch: a lever whose
    metric is not paying is switched off for the rest of the season, and from
    that day the arm plays the shipped executor. Measured on the recorded
    world `pick_games(1)` returns, `pl_wheat_adj` was killed on day 2 and
    `co_chain_tile` on day 3 of 30 -- so a gate row reading "rlv3:full is
    level with recommended" was two days of perturbation and 28 days of the
    baseline. Nothing recorded it. This does, on the row.

    `days_live` is COUNTED at dawn by the ledger, not inferred: `off` is
    revocable (see `ledger._revive`), so a lever can be killed on day 4,
    revived on day 5 and killed again on day 13, and "the day it was killed"
    is then not the number of days it played. `killed[flag]` is still on the
    row as the day the kill that is STILL IN FORCE landed. The fallback chain
    is there for a ledger that never ran a dawn -- a faked episode in a test.
    """
    from rlv3 import ledger

    led = ledger.LEDGER
    days = led.day + 1
    return {
        "days": days,
        "levers_off": sorted(led.off),
        "killed_day": dict(led.killed),
        "days_live": {f: led.days_live.get(f, led.killed.get(f, days))
                      for f in cfg.exec_flags},
    }


# ------------------------------------------------------------- registration
def register():
    """Wrap `rl.rollout._make_dev_policy_agent`. Idempotent; -> restore."""
    global _PATCHED
    from rl import rollout

    if _PATCHED is not None:
        return unregister
    shipped = rollout._make_dev_policy_agent
    _PATCHED = shipped

    def _make_dev_policy_agent(spec):
        if isinstance(spec, dict) and spec.get("kind") == "rlv3":
            return AGENT.make_dev_agent(spec)
        return shipped(spec)

    rollout._make_dev_policy_agent = _make_dev_policy_agent
    return unregister


def unregister():
    global _PATCHED
    if _PATCHED is None:
        return
    from rl import rollout
    rollout._make_dev_policy_agent = _PATCHED
    _PATCHED = None


def is_registered():
    return _PATCHED is not None


# ---------------------------------------------------------------- the driver
def run_episode(job):
    """Pool-safe `rl.rollout.run_episode` for an rlv3 job. Module level so it
    pickles; registers inside the worker because spawn does not inherit the
    parent's patches.

    The episode's rlv3 counters ride back on `result["extra"]["rlv3"]`, and a
    job that declared itself rlv3 and never reached the plan provider raises
    here rather than returning a number.
    """
    from rl import rollout

    register()
    spec = job.get("learner_policy")
    is_rlv3 = isinstance(spec, dict) and spec.get("kind") == "rlv3"
    before = AGENT.counters().get("make_plan", 0)
    try:
        out = rollout.run_episode(job)
    finally:
        # `install_seams` patches kagent.scheduler and kagent.market, which are
        # PROCESS globals, and rl.rollout.run_episode has no teardown hook for
        # them (it restores only features.MODE and scheduler.EXEC). A worker
        # that plays an rlv3 arm and then a baseline arm would otherwise run
        # the baseline through rlv3's seams -- the contamination that makes a
        # paired eval unreadable in BOTH directions.
        a = AGENT.last_agent()
        if is_rlv3 and a is not None and getattr(a, "restore", None):
            a.restore()
    made = AGENT.counters().get("make_plan", 0) - before
    if is_rlv3:
        if made <= 0:
            raise RuntimeError(
                "rlv3 arm played 0 plans: the provider was never called, so this "
                "episode ran the SHIPPED agent while reporting as rlv3. DEAD, "
                "not null. Check that rlv3.arm.register() ran in THIS process "
                f"(pid {os.getpid()}, registered={is_registered()}).")
        rlv3 = {"make_plan": made, "preset": spec.get("preset"),
                "counters": AGENT.counters()}
        cfg = AGENT.config_from_spec(spec)
        if cfg.exec_flags:
            # Only an arm with flags installs the seam, and only the seam
            # rolls the ledger -- so on a flagless arm LEDGER still holds the
            # PREVIOUS episode's state in a reused worker. Reporting it there
            # would be a fabricated row, which is worse than a missing one.
            rlv3.update(ledger_report(cfg))
        out.setdefault("extra", {})["rlv3"] = rlv3
    return out


def make_jobs(spec, games, seed=0, temperature=0.0):
    """games: [(session_path, script_side, band), ...] from
    `tools.agent_scorecard.pick_games` or `rl.session_eval.held_out_entries`.

    The band goes onto the spec per game: `objective.p_win` is calibrated per
    rating band, and band 5 is the only row on disk that was MEASURED rather
    than derived. A job with no band gets the pooled row.
    """
    jobs = []
    for path, script_side, band in games:
        js = dict(spec)
        if band is not None and "band" not in spec:
            js["band"] = int(band)
        jobs.append(AGENT.make_job(js, path, script_side, seed=seed,
                                   temperature=temperature))
    return jobs


__all__ = ["parse", "label", "ledger_report", "register", "unregister",
           "is_registered", "run_episode", "make_jobs"]
