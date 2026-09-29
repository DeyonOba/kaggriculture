"""ACTIVATION. The part that is easy to get silently wrong.

Python's `site` imports a top-level module named `sitecustomize` at
interpreter startup in EVERY process, including multiprocessing workers. On
this machine `multiprocessing.get_start_method()` is "spawn" (verified), so
a monkeypatch applied in the parent process is INVISIBLE to
`rl/session_eval.py`'s workers: an arm patched in the parent would play the
SHIPPED executor while reporting itself as rlv3. That is the exact shape of
the Stage B board arms, which never trained and were read as null results
for three runs. Installing from here is what makes the patch reach a worker.

    cd <repo> && KAGG_RLV3=1 PYTHONPATH=.:rlv3 uv run python \
        tools/agent_scorecard.py --agent knobs:rlv3/arms/pl_wheat_adj.json \
        --eval rl/runs/draw995/batch2.json:ship2_hv --games 6

The body is inert unless KAGG_RLV3=1.

ORDERING TRAP, handled below. `kagent.scheduler` reads `KAGG_EXEC` and calls
`set_exec` AT IMPORT TIME, and `set_exec` raises ValueError on a flag it
does not know. So `KAGG_EXEC=pl_wheat_adj` would blow up the import before
`register()` could ever add the flag. This module therefore splits
`KAGG_EXEC`, lets the shipped flags through the import, and applies the rlv3
ones after `register()`.

FAILURE IS LOUD ON PURPOSE. Every failure path here prints to stderr and,
under KAGG_RLV3_STRICT=1, re-raises. A silent failure in this file is
indistinguishable from a null result.
"""

import os
import sys


def _install():
    """-> (executor, scheduler, held_back_KAGG_EXEC).

    KAGG_EXEC is held back across the WHOLE import, not filtered: importing
    `rlv3.executor` pulls in `kagent.scheduler` transitively, so any split
    done after that import is already too late (measured -- the child died
    on `unknown executor flag 'pl_wheat_adj'` raised from scheduler's own
    import). With the variable unset, scheduler's import-time `set_exec` is
    a no-op and the caller applies the FULL list once `register()` has run.
    """
    raw = os.environ.pop("KAGG_EXEC", "")
    try:
        from rlv3 import executor
        from kagent import scheduler

        executor.register(scheduler)
        executor.install(scheduler)
        executor.apply_env_params()
    finally:
        if raw:
            os.environ["KAGG_EXEC"] = raw
    return executor, scheduler, raw


def _die(msg):
    """Kill the process. RAISING IS NOT ENOUGH, and this is measured:
    `site.execsitecustomize` catches every exception out of this module,
    prints one line to stderr and CONTINUES. A child with a bad KAGG_EXEC
    exited 0, having quietly run the shipped executor. That is the precise
    failure the arm-is-dead-not-null rule exists to catch, so a failed
    install exits non-zero instead of limping on. KAGG_RLV3_SOFT=1 opts back
    into limping, and says so on stderr.
    """
    import traceback
    traceback.print_exc(file=sys.stderr)
    print(f"rlv3: INSTALL FAILED ({msg}) -- this process would run the SHIPPED "
          "executor while reporting as rlv3. Its result would be DEAD, not null.",
          file=sys.stderr, flush=True)
    if os.environ.get("KAGG_RLV3_SOFT") == "1":
        print("rlv3: KAGG_RLV3_SOFT=1, continuing WITHOUT rlv3.",
              file=sys.stderr, flush=True)
        return
    sys.stderr.flush()
    os._exit(70)          # EX_SOFTWARE; sys.exit would be swallowed too


if os.environ.get("KAGG_RLV3") == "1":
    _ok = None
    try:
        _ok = _install()
    except Exception:
        _die("install")
    if _ok is not None:
        _ex, _sched, _raw = _ok
        if _raw:
            # A typo in KAGG_EXEC must stay fatal, exactly as
            # `kagent.scheduler`'s own import-time `set_exec` makes it.
            try:
                _sched.set_exec(_raw)
            except Exception:
                _die(f"KAGG_EXEC={_raw!r}")
        if os.environ.get("KAGG_RLV3_VERBOSE"):
            on = [f for f in _ex.RLV3_FLAGS if _ex.active_flags().get(f)]
            print(f"rlv3: installed in pid {os.getpid()}; flags on: {on or 'none'}",
                  file=sys.stderr, flush=True)
