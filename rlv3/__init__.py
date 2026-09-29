"""rlv3 -- the win-first agent package.

Built alongside `kagent/` and `rl/`, never inside them: every module here
IMPORTS the shipped code and adds a layer. Nothing in kagent/, rl/, tools/ or
tests/ is modified by anything under rlv3/.

The package exists for one reason, stated lexicographically:

    1. WIN RATE.  A win is `final money > opponent final money`.
    2. When winning, maximise profit.
    3. When losing, minimise the loss margin.

`rlv3.objective` is the only place those three clauses are written down for an
IN-GAME decision: it ranks projected Outlooks mid-season, with dead bands sized
to a projection that is barely sign-correct before day 12.

`rlv3.promote` applies the same three clauses in the other domain -- ranking
FINISHED agents over the 995-game paired gate, where the uncertainty is
sampling error and the instrument is McNemar rather than a dead band. It takes
the definition of a win from `rlv3.objective.win_of` instead of restating it,
and `rlv3/tests/test_promote.py` asserts the two stay in step.

The four components are wired into ONE agent by `rlv3.agent`:

    rlv3.agent        Rlv3PlanProvider + make_dev_agent -- objective,
                      allocation, selling and the executor seam composed,
                      in the shape rl.rollout._make_dev_policy_agent expects
    rlv3.arm          makes "rlv3" a dev arm kind WITHOUT editing rl/rollout.py
    rlv3.smoke        the liveness + zero-dose identity check

The training driver lives alongside them and is documented in rlv3/README.md:

    rlv3.train        the slice driver (stage graph, budget, manifest)
    rlv3.stages       stage resolution over tools/kaggle_job/job.py, PPO slicing
    rlv3.fingerprint  what an artifact was made of -- the anti-stale layer
    rlv3.manifest     what was trained on, with which heads, and where it stopped

Submodules are NOT re-exported here: importing this package must stay cheap
(`rlv3.stages` reaches rl.train, and that reaches torch, only when asked).
"""

__all__ = ["agent", "arm", "objective"]
