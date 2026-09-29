# rlv3 — a win-first agent

Built beside `kagent/` and `rl/`, never inside them. Every module here IMPORTS
the shipped code and adds a layer; nothing under `kagent/`, `rl/`, `tools/` or
`tests/` is modified by anything in this package.

The objective is lexicographic, and every component is scored through it:

1. **Win rate.** A win is `final money > opponent final money`. This dominates.
2. Then, when winning, maximise profit.
3. Then, when losing, minimise the loss margin.

That is the opposite of what the learned components have done so far. Every
losing arm of 2026-09-26 handed the opponent $2–7k: the market is ONE shared
inventory and the score is relative, so banking well while the opponent banks
more is a loss. `rlv3.objective.opponent_bank` exists to put that number on the
screen at decision time instead of 995 games later.

## The modules

| module | what it decides | what it reuses |
| --- | --- | --- |
| `objective.py` | what "better" means — `terminal_reward`, `project`, `better`, `rank`, and `ProjState`, the self-correcting solve of the opponent's market flow | `planner._rent_inventory`, `planner._rent_board_supply`, `econ.sell_revenue/buy_cost`; `calibration.json` is the only fitted object |
| `rent.py` | what one more tile of a crop or one more animal is WORTH over its whole remaining stream, net of the opponent's lost revenue | `econ.sell_revenue` (asserted equal), `planner._rent_inventory` |
| `allocation.py` | crops, livestock and land per day — re-derived every dawn from the board, not scheduled. **Hiring is READ, never written** | `planner.make_champion_plan` is the base expression; `features.ANCHOR_CAPS` are the hard ceilings |
| `land.py` | whether to buy the next quadrant: hard gates first, the learned score last, so a bad `theta` can only buy LESS | `features.land_allowed` |
| `market_model.py` | the exact market identities — drain clock, inventory path, `dP/dI`, season capacity. Nothing fitted | `econ.town_drain_per_day`, `econ.price` |
| `selling.py` | when to sell, how much, and in which market-order slot | `market.build_orders` (called exactly once), `market._sell_orders` |
| `executor.py` + `placement.py`, `choreorder.py`, `placenet.py`, `ledger.py` | Maia2: movement, chore order and placement, at the `scheduler.build_tasks / assign` seam | the shipped scheduler, wrapped, never replaced |
| **`agent.py`** | **the composition** — a plan provider + policy in the shape `rl.rollout._make_dev_policy_agent` already returns | all of the above |
| **`arm.py`** | makes `rlv3` a dev arm kind WITHOUT editing `rl/rollout.py` | `rl.rollout.run_episode` |
| **`smoke.py`** | liveness + the zero-dose identity check | `tools.agent_scorecard.pick_games` |
| `train.py`, `stages.py`, `fingerprint.py`, `manifest.py`, `promote.py` | the slice driver (Part 2 below) | `tools/kaggle_job/job.py`, `rl/train.py`, `rl/session_eval.py` |

## How the pieces compose

`Rlv3PlanProvider` duck-types `kagent.policy.ChampionPlanProvider`, so the
whole wiring is `HeuristicPolicy(Rlv3PlanProvider(cfg))` — the same shape
`rl.rollout` already builds for `dev:champion`.

```
every step   ProjState.observe(view)          solve their net market flow from
                                              the shared inventory delta
hour 0       ProjState.dawn(view)             yesterday's residual -> two
                                              bounded EWMAs (their board->market
                                              conversion, our offered->filled)
             allocation.make_plan             land gates, then the water-fill
                                              on marginal RELATIVE rent
             objective.project(plan=...)      record (or rank) the outlook
             memory.rlv3_sell_cfg = ...       mark the memory for the sell seam
every step   scheduler.build_tasks / assign   <- executor seam
every step   market.build_orders              <- selling seam
```

`make_plan` runs at hour 0 and only at hour 0, because that is the only hour
`HeuristicPolicy.act` ever builds a day plan. Every planner decision here is a
DAWN decision; the two seams are what act in between.

### State on `Memory`

`kagent.gamestate.Memory` has no `__slots__`, so each component attaches its
own. They collided once and it is worth knowing where:

| attribute | owner | type |
| --- | --- | --- |
| `memory.rlv3` | `land.observe_dawn` | **dict** — `allocation.chore_cost` reads `memory.rlv3["chore"]` |
| `memory.rlv3_sell` | `selling.state_for` | `SellState` |
| `memory.rlv3_sell_cfg` | `agent.Rlv3PlanProvider` | `SellConfig` or `None` |
| `memory.rlv3_proj` | `agent.Rlv3PlanProvider` | `objective.ProjState` |

### Presets

Every deviation is a **dose**, not a code path, and `lam = 0` is bit-identical
to the baseline by construction.

| preset | lam | sell | exec | what it is for |
| --- | --- | --- | --- | --- |
| `base` | 0 | off | shipped 3 | the IDENTITY arm — must reproduce `ship2_hv` to the dollar |
| `recommended` | 0 | `slots` | shipped 3 | the only thing any component's own pre-gate supported |
| `alloc` | 1 | off | shipped 3 | the allocation dose, alone |
| `sell` | 0 | `full` | shipped 3 | the whole withholding schedule, alone |
| `exec` | 0 | off | **none (2026-09-29)** | was Maia2 alone; now the base scheduler |
| `full` | 1 | `slots` | **none (2026-09-29)** | everything the components recommend |
| `select` | 1 | `slots` | **none (2026-09-29)** | `full`, plus `objective.better()` ranking the dose per dawn |

**MAIA2 IS OFF (2026-09-29).** `agent.FULL_EXEC` is now `()`, so
`install_seams` never calls `EXEC.install` and `build_tasks`/`assign` are the
shipped functions in every preset. Two reasons on the record: the whole
movement budget optimised perfectly is worth about $0
(`kaggriculture-executor-bounded`), and the kill switches were measured
disabling every lever within 2-4 days of a 30-day season, so these presets
were dosing a day-0-only intervention. `agent.MAIA2_EXEC` keeps the old tuple:
the plan is to reintroduce it LATER as a randomness source over board
movement, to test whether the planner self-corrects once the executor stops
being deterministic.

`rlv3/arms/*.json` are the executor's own single-lever arms and are read
through `KAGG_EXEC` / `KAGG_RLV3_PARAMS`, not through these presets.

## Running it

```bash
# the wiring, no games
PYTHONPATH=. uv run pytest rlv3/tests/ -q

# liveness + zero-dose identity, 1 episode per preset on one recorded world
PYTHONPATH=. uv run python rlv3/smoke.py --presets base,full --games 1
```

### The scorecard

`tools/agent_scorecard.py` is the whole-agent metric harness — outcome,
planner, market/demand, executor, per-group gap against the three leaders,
head-to-head — and **it is how each part is judged**.

It takes `champion | knobs: | bundle: | weights: | execnet:` and does **not**
know `rlv3:`, and its `_build_agent` hands anything else to
`rl.rollout._make_dev_policy_agent`, which does not know `kind="rlv3"` either.
`rlv3/scorecard_hook.py` supplies exactly those two missing pieces and then
calls the tool's own `trace_game` / `build_row` / `print_table` /
`gap_closed`. No metric is re-implemented; a number here that disagrees with
`tools/agent_scorecard.py` is a bug in the hook.

```bash
# rlv3 and champ-h10 on the SAME worlds, plus the identity arm as a third column
cd <repo> && PYTHONPATH=. uv run python rlv3/scorecard_hook.py \
    --agent rlv3:full --extra rlv3:base --games 4 --procs 2 \
    --out rlv3/eval/scorecard_rlv3_full.json

# one component at a time: `full` is alloc (+) recommended (+) exec
PYTHONPATH=. uv run python rlv3/scorecard_hook.py --agent rlv3:alloc --games 4
```

Three things the hook exists to get right, each of which has already cost
this repo a run:

* **The same games.** `pick_games(n)` is deterministic and prefix-stable, so
  every agent gets the identical (world, seat) list and `pick_games(1)` is
  `pick_games(4)[:1]`. Per-world sd is $34,749; an unpaired money comparison
  is not a comparison.
* **The patch must be in the playing process.** `multiprocessing` starts
  **spawn** here, so `arm.register()` in the parent is invisible in a worker.
  Registration happens inside the trace worker and the liveness counters ride
  back, so a dead arm cannot be read as a bad one.
* **The seams must come back off.** `agent_scorecard._build_agent`'s
  `restore` only puts `features.MODE` back, so a champ-h10 trace played after
  an rlv3 trace in the same process would be scored through rlv3's sizer.
  Each agent gets its own process **and** the seams are torn down in a
  `finally`.

It also prints an **identity** line: whether two columns played the same game
at every one of the 719 steps of every world. Measured 2026-09-26,
`rlv3:base` and `bundle:arena/agents/champ-h10/main.py` are 2876/2876 — the
frozen bundle and today's tree with h10's knobs are one agent on these
worlds, so `d:champ-h10` is rlv3's own effect and not a tree difference.
(The $11,638 divergence `rlv3/eval/alloc_ladder.sh` warns about was on
self-play seed 770000, not on a recorded world.)

Traces are cached under `--traces` (default `rlv3/eval/scorecard_traces`) and
reused when the file on disk names the same world and seat, so re-reading the
table costs zero games; `--fresh` replays.

**What it cannot show.** `OUTCOME`'s `win_rate` / `win_band*` are empty for
every rlv3 column: no paired evaluation of an rlv3 arm exists on disk. Trace
money over 4 games is not a result — it is the input to the planner, market
and executor metrics, which is what the scorecard is for.

### The paired gate

The verdict instrument is `rl/session_eval.py`, 995 band-stratified held-out
games, McNemar on the discordant games, replicated on a second seed block.
Sub-0.02 differences are noise, and local gates have disagreed with the ladder
**twice**, both times favouring the locally-better agent. Nothing in this
package has been through one.

`rl/session_eval.py` does not know `rlv3:` either. Two ways in:

* **Now.** `rlv3.arm.run_episode` is a module-level, picklable drop-in for
  `rollout.run_episode`; hand it to the `Pool` instead. It registers inside the
  worker, because `multiprocessing` starts **spawn** here and a patch applied in
  the parent is invisible to a worker — the exact shape of the Stage B board
  arms, which were read as null results for three runs while never training.
* **DONE (074ba51), and it disarmed the liveness check.** Both lines exist on
  disk now, so `rl/session_eval.py` reaches rlv3 through `rollout.run_episode`
  and **never through `rlv3.arm.run_episode`** -- which is the function that
  raises on `make_plan == 0` and the one that calls `ledger_report()` to put
  `days_live` on the row. On the 995-game gate neither is armed. A dead rlv3
  arm is currently indistinguishable from a null one there, which is the exact
  failure the rest of this section exists to prevent. The lines, for the
  record. In `rl/rollout.py:_make_dev_policy_agent`, first line of the body:

  ```python
  if spec.get("kind") == "rlv3": from rlv3.agent import make_dev_agent; return make_dev_agent(spec)
  ```

  and in `rl/session_eval.py`'s `dev:` block, beside the `execnet` branch:

  ```python
  elif parts[1] == "rlv3": spec = __import__("rlv3.arm", fromlist=["x"]).parse(parts[2] if len(parts) > 2 else "")
  ```

  which makes `--arm rlv3_full=dev:rlv3:full` work.

### Liveness, which is not optional

`rlv3.arm.run_episode` **raises** when a job declared itself `rlv3` and the plan
provider was never called. An arm that reports as rlv3 while running the shipped
agent is DEAD, not null, and is indistinguishable from a null result once it is
a row in a table. Counters ride back on `result["extra"]["rlv3"]`; a clean
season is `make_plan 30, dawn 30, observe 719`.

**The levers have the same contract, one level down.** `rlv3.ledger.correct()`
is a within-episode kill switch, and a lever it switches off on day 2 is the
shipped executor for the other 28 days — the same DEAD-not-null failure, from
inside an arm whose plan counters are perfect. So every kill is recorded where
the existing checks already look:

* `exec_stats` gets `<flag>.killed_day<d>` (smoke.py prints it beside the flag
  list, scorecard_hook.py prints it under LIVENESS);
* `result["extra"]["rlv3"]` gets `days`, `levers_off`, `killed_day` and
  `days_live` — **read `days_live` before reading any margin**. A flagless arm
  such as `rlv3:base` never installs the seam, so those four keys are absent
  rather than stale.

A kill switch also may not be shorter than the causal horizon of the lever it
judges. `pl_wheat_adj` is gated on `feeds_from_hand_share`, which cannot move
until a wheat tile it seated has been harvested — 2 days (README "Object
Types", engine `CROPS["WHEAT"]["first_yield_day"]`) — so days before
`first wheat planted + 2` do not count toward its streak. Measured on
`session-112391485`, that moved `pl_wheat_adj` from 2 days live to 4. It does
not rescue the arm: `co_chain_tile` still dies on day 3, because its rule asks
a noisy ratio to RISE every day rather than testing it against a level, and on
that world `rlv3:exec` perturbs 4 of 30 days. That is now on the row.

## What was measured, and what was not

Smoke, 2026-09-26, `session-112391485` script side 0, 1 episode per preset:

| arm | our money | opponent | margin | plans |
| --- | --- | --- | --- | --- |
| `ship2_hv` (recorded reference) | 121,817 | 118,562 | +3,255 | — |
| `rlv3:base` | **121,817** | **118,562** | +3,255 | 30 |
| `rlv3:full` | 102,651 | 68,496 | +34,155 | 30 |

`rlv3:base` is exact, twice, including a re-run after HEAD moved. **The
`rlv3:full` row is not evidence of anything.** Per-world sd is $34,749; the
opponent is a recorded tape and a board difference RE-ROLLS the daily shop draw
(`kaggriculture-shop-draws-reroll`), so a $50k move on the tape's side is a flag
that the worlds diverged, not an attribution. Only the paired 995-game mean
attributes.

What each component expects on that gate, in its own words:

* **allocation** — "more likely to lose than win." The constant version of the
  same idea (`mix_mild`) is already −1.5 points on the live draw, and this has
  the LP refutation's exact shape.
* **selling** — the sizing half is **refuted**: margin +$656 with the whole
  schedule against +$772 for the order-slot move alone (re-measured 2026-09-27
  with the engine's same-slot lockstep; the earlier +$891 / +$1,281 credited us
  a within-slot first-seller premium the engine never pays, and the opponent is
  already at slot 0 on 46 of 71 contested product-steps). `RECOMMENDED` is
  `slots_only`, and at +$772 against a per-world sd of $34,749 it is far inside
  the noise floor and should ride along with a component that moves allocation,
  never be gated alone.
* **executor** — `kaggriculture-executor-bounded`: the whole movement budget
  optimised perfectly is worth about $0. `mv_dest` is measured refuted and
  `pl_net` measured inert; both ship off. `pl_wheat_adj` must be gated
  head-to-head against the shipped `layout` rule, not against nothing.
* **objective** — `better()` and `p_win()` ARE a value-ranking device, and
  V-ranking is CLOSED here (four heads at chance out-of-sample). The 39/39
  retrodiction is at the TERMINAL state and does not show that candidate DAY
  PLANS are ranked correctly. `select` ships off.

## Resuming training tomorrow

Part 2 is the driver. The short version:

```bash
# local, nothing executed
PYTHONPATH=. uv run python -m rlv3.train --job rlv3/jobs/rlv3_train.json \
    --run rlv3/runs/a1 --budget-s 39600 --procs 2 --dry-run

# a slice
PYTHONPATH=. uv run python -m rlv3.train --job rlv3/jobs/rlv3_train.json \
    --run rlv3/runs/a1 --budget-s 39600 --procs 2
```

Re-running the same command continues: a stage whose output exists AND whose
stamp still matches the current fingerprint is skipped, a stale one is
quarantined into `<run>/stale/<ts>/` and re-run, and PPO resumes from
`<run>/last.pt` with the remaining iterations. Read `<run>/manifest.json`, not
the console — it is written atomically after every stage, so a wall kill at
11h59m still leaves the record. Check `lr_max_rel_dev` before reading a sliced
PPO number as an unsliced one, and never read a PPO run at all before
`policy_updates > 0` and `onpolicy_kl ~ 0`.

Checkpoints travel as a Kaggle dataset:

```bash
DRY=1 rlv3/kaggle/push_code.sh "msg"     # verify the tarball, upload nothing
rlv3/kaggle/push_code.sh "msg"
rlv3/kaggle/push_checkpoints.sh rlv3/runs/a1 "msg" [--only ppo]
```

Expect the first candidates to be REFUSED and that to be correct: the best RL
net on this corpus is 0.127–0.180 against champ-h10's 0.354, and `promote()`
refuses a candidate that banks more and wins fewer.

---

# Part 2 — checkpointed, resumable, slice-safe training

Training arrives in slices: an 11-hour Kaggle session, a laptop that has to
give the CPU back, a run killed by a wall. A plan that only works when it runs
to completion in one process is not a plan. This is the driver that turns the
plan into a **stage graph** whose every output carries a stamp of the
fingerprint it was made under, so the next slice can tell *"this exists"* apart
from *"this is still valid"*.

It makes no in-game decision. Its dawn is the start of a compute slice and its
state is the run, not the board.

```
PYTHONPATH=. uv run python -m rlv3.train \
    --job rlv3/jobs/rlv3_train.json --run rlv3/runs/a1 \
    --budget-s 39600 --procs 2 [--dry-run] [--only ppo] [--force ppo]
```

`--dry-run` prints the resolution table and exits without executing anything.
That is the local check: no games, safe on a busy machine.

```
PYTHONPATH=. uv run pytest rlv3/tests/test_fingerprint.py \
    rlv3/tests/test_stages.py rlv3/tests/test_manifest.py \
    rlv3/tests/test_promote.py -q
```

## What it reuses, and never modifies

`kagent/`, `rl/`, `tools/` and `tests/` are read-only to this package.

| Reused | How |
| --- | --- |
| `tools/kaggle_job/job.py` | `resolve_step`, `_telemetry_iters` **called**, not copied — loaded with the importlib pattern from `tests/test_kaggle_job.py`. `rlv3/tests/test_stages.py` re-asserts all four of its behaviours through `rlv3.stages.resolve`. |
| `tools/kaggle_job/push.sh` | the kernel push, unmodified, via `SKIP_CODE=1 KERNEL_ID=… CONFIG=… META=…`. |
| `rl/train.py` | its CLI and its constants (`LR`, `LR_FLOOR`, `ENT_COEF`, `PLATEAU_COOL`, `WARMUP_ITERS`, `ONPOLICY_KL_TOL`). |
| `rl/session_eval.py` | `held_out_entries` (the fingerprint hashes the gate's *own* draw), `paired`, `EDGES`. |
| `rl/common.py`, `kagent/features.py` | `BOARD_CH`, `HEADS_VERSION`, `MODE`, `HEAD_SIZES`, `N_ACTION_LOGITS`. |
| `rl/runs/draw995/batch2.json` | the frozen baseline rows for `promote()` and its tests. |

## The objective

Lexicographic, and every component is scored through it:

1. **Win rate.** A win is `final money > opponent final money`.
2. Then, when winning, maximise profit.
3. Then, when losing, minimise the loss margin.

`rlv3.promote` is the only place selection happens. Not the trainer's `best.pt`
argmax on mean final money, and not the in-training SESSION-WR — 3 games a bin,
18 in all, where 0.06 is one win and 0.11 is two. Six Stage B arms were ranked
on that.

Rank 1 requires `delta > 0` with McNemar `p <= 0.10` on the 995-game paired
draw. **A candidate that banks more and wins fewer games is refused outright**,
at any margin — that is the exact failure mode of every learned component so
far: our bank rises while the opponent's rises faster, and the market is one
shared inventory.

Frozen reference on `rl/runs/draw995/batch2.json`:

| arm | win | margin when winning | margin when losing |
| --- | --- | --- | --- |
| `ship2_hv` (champ-h10) | 0.354 | +$24,596 | −$17,100 |
| `ship2` | 0.315 | +$26,303 | −$18,833 |
| `champ_h8` | 0.230 | +$24,283 | −$23,351 |

`ship2` vs `ship2_hv` is the refusal clause in real data: more money on its
wins, fewer wins, not promoted. Per-band, champ-h10 is
0.440 / 0.367 / 0.366 / **0.200** / 0.406 / 0.349 — band 3 (1500–2000) is the
hole, and every promotion records all six.

## The four things that make a slice safe

### 1. The stamp (`rlv3/fingerprint.py`)

`job.resolve_step` skips a step when its output file exists. That is the same
mechanism as **`kaggriculture-gate-cache-stale`**, where ladder_gate reused a
cached base without checking it covered the current worlds and two win columns
of one table came from different world sets. Adopting it for every stage makes
it more dangerous, not less, so every skip is re-opened:

```
exists?  -> job.resolve_step        (unchanged)
valid?   -> fingerprint.check       (new)
stale    -> quarantine to <run>/stale/<ts>/ and re-run
```

The fingerprint is the action space (`heads_version`, `heads_mode`,
sha1 of `HEAD_SIZES`, `n_action_logits`, `n_features`, board arm and
`BOARD_CH`, plus the board on/off flag), the corpus draw (sha1 of
`session_index.json` **and** of the ordered `(path, seat)` list
`held_out_entries` returns at the gate's seed and per-bin), the
rlv3+kagent+rl code digest, and the engine pin.

This is aimed at one recorded failure: `job_topboard.json`'s run-3 note, where
`goose_target` went `[0,8,16,24,32]` → `[0,4,6,10,16]`, every stored `choices`
index silently came to mean a different number of geese, and the operator had
to hand-curate the checkpoint dataset so the steps would not skip themselves
back into the old action space. `head_sizes_sha` catches it in milliseconds.
The 995-game draw itself already changed once (785 → 995 on 26 September),
which would make a resumed gate incomparable to the run it continues.

Verified end to end: bumping `HEADS_VERSION` mid-run quarantines both stamped
artifacts into `<run>/stale/<ts>/` (stamp and `reason.txt` travel with them)
and re-runs the stages.

**A stage also depends on the bytes of its inputs**, declared as `inputs` in
the job JSON and folded into its stamp. Without that the gate *skips itself out
of existence*: after PPO appends another ten iterations, `session_eval.json` is
still on disk and neither the action space nor the corpus has moved, so the
995-game verdict would never re-run against the checkpoint it is judging. The
same clause makes a `bc.pt` rebuilt from a new corpus invalidate the PPO run
that was resuming from it.

**A PPO run directory is not a `skip_if` hit**, so it would otherwise never be
checked — and resuming it is the most dangerous reuse in the graph, because
`last.pt` carries the net's weights under the action space it was trained in.
The run is stamped *before the first slice launches* (a wall kill therefore
still leaves a resumable run) and re-checked on every resume; a stale one is
quarantined and restarts from iteration 0. A run created outside rlv3 — by
`rl/train.py` directly — is **unstamped, and is not adopted**. That is the same
rule, not an exception to it.

### 2. The health read-back (`stages.verify_training_ran`)

**`kaggriculture-silent-trainer-failures`**: Stage B's board arms ran twenty
iterations at `policy_updates=0` and `onpolicy_kl` +3.7 nats, with money means
printing happily. A resumable runner makes this strictly worse — telemetry rows
≥ target plus an `eval.json` is exactly `job.resolve_step`'s completion test, so
the stage would be skipped *forever*.

After every slice the rows it just appended are read back and must satisfy
`policy_updates > 0` outside warmup, `|onpolicy_kl| <= 1e-3`, and
`kl_stop_at is None`. A failure marks the stage **POISONED**, which blocks the
skip, blocks the next slice, and blocks the checkpoint push. `--force <stage>`
is the only way past it, and it quarantines the run first.

### 3. `rlv3/kaggle/push_code.sh`

`rlv3/` is untracked by instruction and `tools/kaggle_job/push.sh` builds its
tarball from `git ls-files`. **Measured: 0 of 58 rlv3 files are tracked**, so
the stock push would upload a code dataset with no rlv3 package in it and a
kernel that dies on `import rlv3` after the mount — the same class of failure
push.sh's own comment records costing a 1.9 h kernel run on 2026-09-26.

This script tars `git ls-files` (919) **plus** rlv3's sources (58 = 977),
refuses on `rlv3/runs`, `__pycache__` or any `*.pt`, refuses when a listed file
is not on disk, then repeats push.sh's extract-and-import check with
`rlv3.train, rlv3.stages, rlv3.fingerprint, rlv3.manifest, rlv3.promote` and a
`load_stages` parse of the job graph — exiting non-zero *before* anything is
uploaded. `DRY=1` builds and verifies without uploading.

### 4. The LR schedule — and where it cannot be fixed

`rl/train.py:710` is

```python
decay = max(LR_FLOOR, 1.0 - it / max(1, iters))   # iters = the SLICE length
```

It divides by the slice length, unlike `shaping_w` on line 619, which uses
`total_iters` and *is* continuous. Resume at k=10 of T=20 and the slice sits at
`LR_FLOOR` (0.15 × LR) from its first iteration where an unsliced run is at
0.50 × LR — a 3.3× error, and `ent_now = ENT_COEF * decay` breaks identically.

`lr_compensation(k, m, T)` returns the `--lr` / `--ent-coef` that reproduce the
unsliced schedule as closely as a **constant** can (the scale that equalises
the two ends, which provably minimises the worst relative deviation), plus that
residual.

> **The spec's two metrics fight here, and the default resolves it toward the
> budget.** "Maximise slice length subject to the reserve" and "keep the LR
> within 0.15 of the unsliced schedule" cannot both hold: a constant cannot
> track a ramp, and at T=20 honouring 0.15 caps a slice near **three**
> iterations however much wall is free. Default is to use the wall and record
> the residual (`lr_max_rel_dev`, per slice, in the manifest). `--lr-tol 0.15`
> switches on shortening, never below `min_iters`. Nobody should read a sliced
> PPO number as an unsliced one without looking at that residual.

## What it must not break about the agent's in-game adaptivity

The learned parts read board, market and opponent state through
`kagent.features.encode/masks`; *which* numbers those are and *which* head
indices the choices mean is `HEADS_VERSION`. `rl.common.policy_layers` writes it
into the weights and `kagent.nnpolicy.NeuralPolicy.make_plan` calls
`features.set_heads_version(want)` before encoding. rlv3 preserves that end to
end: the fingerprint records it, the stamp carries it, `promote()` refuses a
checkpoint whose `eval.json` head version differs, and `push_checkpoints.sh`
refuses to ship an unstamped `.pt`.

Three adaptive things inside `rl/train.py` do **not** survive a process
boundary. This component neutralises them rather than pretending otherwise:

| Not slice-invariant | Neutralised by |
| --- | --- |
| the rating-bin curriculum's rolling frontier (`bin_wins`, in-memory) | pin `--session-mix eval` — band weights from `_eval_band_weights(session_index.json)`, matched to the 995-game gate |
| `best_score` / `evals_since_best`, which feed `cool = PLATEAU_COOL ** evals_since_best` straight into the LR | pin `--plateau-cool 1.0` and `--patience 99`; selection is ours, on the paired gate |
| Adam's moments (`ppo_phase` builds the optimiser fresh) | maximise slice length; refuse anything below `min_iters` |

`assert_slice_invariant` turns each of those into a **refusal**, not a warning,
and also refuses a wrong `--start-iter`, a wrong `--iters`, a `--resume` that
is not `<run>/last.pt`, and a missing `--lr`/`--ent-coef` compensation.
Absence is not neutrality: the trainer's own defaults are `--plateau-cool 0.6`,
`--patience 4`, `--session-mix frontier`.

Continuous **by construction**, and asserted: world seeds
(`world_base + it*episodes + e`), the eval cadence (`(it+1) % eval_every`), the
value warmup (`it < WARMUP_ITERS`) and `shaping_w` all key off the global
iteration counter that `--start-iter` restores.

## The stage graph

`rlv3/jobs/rlv3_train.json`. Each sibling rlv3 component owns its stage
*command*; this driver owns only the contract.

| stage | depends | kind |
| --- | --- | --- |
| `index` | corpus | cheap, never skipped — records the draw |
| `clone` | action, corpus, code | `rl/session_bc.py --min-rating 2500` |
| `maia2` | code | `rl/execnet/train2.py` |
| `bc` | action, code | `rl/train.py --bc-session … --iters 0` |
| `ppo` | action, corpus, code | sliced; `--reward win --session-mix eval` |
| `gate` | action, corpus | 995-game paired eval — **reserve** |
| `card` | code | `tools/agent_scorecard.py` — **reserve** |
| `promote` | action, corpus | in-process, lexicographic |

`reserve: true` stages have their `min_left_s` held back from every earlier
stage's budget, so the gate starts. The first topboard kernel run spent 5.3 h of
11 h on extraction and never reached its eval.

## Kaggle

```bash
DRY=1 rlv3/kaggle/push_code.sh "msg"          # verify the tarball, upload nothing
rlv3/kaggle/push_code.sh "msg"                 # code dataset + the rlv3 kernel
rlv3/kaggle/push_checkpoints.sh rlv3/runs/a1 "msg" [--only ppo]
```

`rlv3/kaggle/job_rlv3.json` is **one** step for the existing
`tools/kaggle_job/job.py`: it keeps doing repo restore, dataset restore,
`restore_checkpoints()` into `OUT_DIR` and the outer budget, and `rlv3.train`
does the stamped, per-stage, per-slice resumption inside that one step.

`push_checkpoints.sh` adds three refusals the existing script lacks — any
artifact without a `.stamp.json`, any POISONED stage, any stamp whose
fingerprint differs from the manifest's current one — all verified to exit
non-zero *before* the tar.

## The manifest

`<run>/manifest.json`, written atomically after **every** stage, so a wall kill
at 11h59m still leaves a record: what ran, what was skipped, what was
quarantined, which stage is poisoned, the slice's `(start, n, lr, dev)`, the
gate's objective and per-band rows, and the promoted checkpoint. Artifact rows
mirror `rl/datasets/MANIFEST.json`.
