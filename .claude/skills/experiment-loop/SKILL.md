---
name: experiment-loop
description: Run one Kaggriculture idea to exhaustion on its own branch — Spec, Research, Implement, Evaluate, Critic, Submit. Use when starting a new line of investigation rather than a one-off tweak.
---

# One idea, one branch, run to exhaustion

The rule: **do not move on until the idea is proven low-value.** Sixteen
executor mechanisms were each tried once and abandoned; the ones that were
followed through (tape cohorts) produced the only real findings.

## Branch per idea

```bash
git checkout -b exec-<idea>      # existing: arch-flow, executor-v2, lp-bound,
                                 # exec-visit, exec-burst, exec-adaptive
```
Change code freely on the branch. Commit every result, **including negatives**,
with the numbers in the message — the commit log is the experiment record and
has repeatedly stopped repeat work.

## The loop

**1. SPEC** — write the claim as something that can be false, with the number
that would falsify it. "X raises win rate by ≥10pp" not "X should help".

**2. RESEARCH** — a Workflow fan-out is worth it here. Give every agent the
MEASURED refutations (see `docs/executor_problems.md`,
`docs/research_reading.md`, the memory files) and add an adversarial **critic
stage that defaults to DEAD**. On 2026-09-16 that killed 14 of 23 proposals
before any compute, including a search already running.

**3. IMPLEMENT** — see the `build-variant` skill. Patch a frozen bundle, raise on
missing anchors, pre-flight against the $3,000 signature.

**4. EVALUATE** — see the `gate-candidate` skill. Screen the MECHANISM first
with low-variance structural metrics (moves/action, work share, actions/tile),
then gate MONEY/WINS at the right size on the right instrument. Mechanism firing
does not imply money.

**5. CRITIC** — before believing a positive: is it inside the ±45 rating noise?
Did it replicate on a second seed block? Is the instrument the one that
correlates? Would a shuffled-label null produce the same headroom? (The day-6
selector's +$41,666 "oracle" matched a null of +$41,951 ± 745.)

**6. SUBMIT** — only on a replicated result. 5/day, **only the most recent 2
active**; a submit evicts the next-oldest active entry.

**7. LOOP or CLOSE** — close only when the idea is refuted *as a class*, not when
one variant failed. Record which variant, which number, and what would reopen it.

## Do not declare a line dead while a named variant is untried

`docs/executor_problems.md` lists two: **cross-turn worker commitment**, and
**changing planner and executor together**. Neither has been attempted, so the
executor is not closed.

## Tooling

- `tools/fast_env.py` — 1.70× local games, bit-identical. Wired into
  `arena/run_games.py`; `KAG_FAST_ENV=0` disables.
- `tools/replay_probe.py` + `replay-autopsy` skill — characterise any agent.
- `tools/decision_diff.py` / `tools/taskset_diff.py` — our agent on THEIR states.
- `tools/winrate_archive.py` — re-score the whole archive by win rate.
- macOS: no GNU `timeout`; multiprocessing spawn cannot bootstrap from a
  `python -` heredoc — write the script to a file.
- `exec_gate --skip-base` is only valid when the previous base run used the SAME
  `--base-seed` and `--n`; result files are overwritten per run.
