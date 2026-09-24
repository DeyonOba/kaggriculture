# Stage B2 — frozen vs trainable board trunk, pre-registered

## 0 Why the first Stage B is void

Two independent defects, both silent, both found by reading checkpoints after
the fact (2026-09-24):

1. **No board arm performed a policy update.** `policy_layers()` never
   serialised the trunk, so the rollout worker played a 216-input policy fed
   168 numbers (rows 168–215 silently skipped) while the trainer scored the
   full policy. The mean of `log π_old − log π` on the first minibatch — the
   KL from the truncated policy to the full one — was **+3.68 to +3.96 nats**
   on board/trainable/slow01 and **+0.38** on frozen, against `KL_STOP = 0.03`.
   The early stop fired before the first gradient step of every iteration.
   Policy tensors are byte-identical between `bc.pt` and `last.pt` in all five
   board arms; the control's moved.
2. **The pretrained trunk never reached PPO in four of five board arms.** BC
   trained it for 40 epochs toward a board-blind teacher (0.23 relative-L2 from
   its file), then `--resume` loaded that trunk over the `--trunk-init` one.

So "PPO damages the encoder", "the gap widens", "frozen recovers 45%" were all
readings of BC checkpoints played through a broken forward pass.

## 1 What changed in the pipeline (all refusals, none silent)

| guard | where | what it catches |
|---|---|---|
| on-policy identity, every iteration | `train._onpolicy_check` | worker and trainer running different policies (tolerance 1e-3 nats) |
| `policy_updates`, `kl_stop_at` in telemetry | `train.ppo_phase` | an iteration that trained nothing |
| resume **then** trunk-init, distance printed and asserted 0 | `train.apply_checkpoints` | a checkpoint overwriting the trunk |
| trunk excluded from BC | `train._bc_params` | BC rewriting the pretrained trunk toward a board-blind teacher |
| first-layer width check | `nnpolicy.forward_policy` | a shorter input silently truncating the policy |
| b85 export refuses board nets | `common.to_weights_data` | shipping the truncation inside a bundle |
| torch threads pinned | `train.main` | cross-process un-pairing of seeded runs |
| held-out `% 5 == 0` excluded from trunk pretraining | `session_bc`, `kaggle_trunk` | the honest eval scoring memorised games |

Tests: `tests/test_nnpolicy.py` (board round-trip, width refusal, export
refusal), `tests/test_train_guards.py` (reward, identity check, load order,
BC exclusion).

## 2 Design

Three arms, `tools/stage_b2.sh`, identical except the trunk treatment:

```
control     KAGG_BOARD=0
frozen      KAGG_BOARD=1 --board --trunk-init trunk_margin_kaggle.pt --freeze-trunk
trainable   KAGG_BOARD=1 --board --trunk-init trunk_margin_kaggle.pt
shared      --seed 1234 --iters 20 --episodes 64 --patience 99 --plateau-cool 1.0
            --reward win --eval-per-bin 8 ; BC 60 games, one checkpoint per arm
```

`frozen` and `trainable` start from the **same** trunk weights (asserted by
the distance print) and the same BC procedure (trunk excluded from BC in both),
so they differ only in whether PPO's gradient reaches the trunk.

**Reward** (`--reward win`): terminal `r = w·(2·win−1) + 2·tanh(margin/20000)`,
`w = 15` on session games, `6` otherwise; money absent. Φ = 1e-4 × portfolio
value, PBRS, annealed to zero by iteration 10 as before.

**Known limitation, stated up front.** `trunk_margin_kaggle.pt` was fitted
before the held-out exclusion existed, so it saw the outcomes of the eval
sessions. It is a 48-dimensional bottleneck fitted on 21k games, so
memorisation of 750 specific ones is bounded, but the clean version of this
experiment re-pretrains the trunk with `tools/kaggle_trunk.py` (now excludes
them by default) and reruns the two board arms.

## 3 Hypotheses and stop rule

Evaluated by `rl/session_eval.py` on the **final** nets (`eval.json`), 175
held-out games per rating bin where the bin has them — the 0–600 and 600–1000
bins hold only 64 and 21 rated held-out seats and are taken whole — for 785
games per arm, the same games for every arm, temperature 0. McNemar on
discordant games.

- **H-board**: `frozen` or `trainable` beats `control` on win rate,
  `p < 0.05`, and the paired delta is positive in ≥ 4 of 6 bands.
- **H-freeze**: `trainable` vs `frozen` on win rate, same test. Either
  direction is a result; a null closes the freeze question at this budget.
- **Guard**: any arm whose telemetry shows `policy_updates == 0` after warm-up
  in ≥ 2 iterations, or a non-zero `onpolicy_kl` beyond tolerance, is not
  read at all — the run is repeated after the cause is found.

Power: at a 0.35 win rate and ~40% discordance, 785 paired games give the
paired delta a standard error near 0.023, so a 0.05 delta lands at roughly
z = 2. Smaller effects are reported as null at this budget, not as absent.
The two small bands carry their own n and are read as such.

## 4 What is deliberately NOT in this run

- A board-aware critic (the trunk is a value-shaped representation and the
  value net is board-blind). Next arm, applied identically to both board arms.
- The phase-aware lead potential Φ(s, t) (investment lead early, money lead
  late). By the PBRS theorem it cannot change which policy wins, only how
  fast; it is a convergence arm and is judged as one.
- `--paired` twin episodes (2× rollout cost).

## 5 Results (2026-09-24, eval landed 19:45)

**Guards, all arms:** on-policy KL ≤ 3e-7 on all 60 iterations; every
post-warm-up iteration performed updates (10–12 per iteration, zero iterations
with none); frozen trunk at distance 0.0 from its file after PPO, trainable
trunk moved 7e-3. The run is read.

`rl/session_eval.py`, 785 held-out games per arm on identical sessions and
seats, temperature 0 (`rl/runs/stage_b2/session_eval.{log,json}`):

| arm | wins / 785 | win rate ± se | money | 0–600 | 600–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500+ |
|---|---|---|---|---|---|---|---|---|---|
| control | 47 | 0.060 ± 0.008 | 61,072 | 0.156 | 0.190 | 0.051 | 0.029 | 0.046 | 0.063 |
| frozen | 21 | 0.027 ± 0.006 | 55,150 | 0.141 | 0.143 | 0.017 | 0.006 | 0.011 | 0.017 |
| trainable | 41 | 0.052 ± 0.008 | 62,254 | 0.172 | 0.190 | 0.023 | 0.029 | 0.040 | 0.057 |

Paired deltas, McNemar on discordant games (arm-only / other-only):

| comparison | Δ win rate | se | z | discordant | p | 95% CI | bands positive |
|---|---|---|---|---|---|---|---|
| frozen − control | −0.033 | 0.008 | −4.05 | 8 / 34 | 0.0001 | [−0.049, −0.017] | 0 of 6 |
| trainable − control | −0.008 | 0.008 | −0.97 | 16 / 22 | 0.42 | [−0.023, +0.008] | 1 of 6 (two at 0) |
| trainable − frozen | +0.026 | 0.008 | +3.26 | 29 / 9 | 0.0017 | [+0.010, +0.041] | 6 of 6 |

**H-board: FAILS.** Neither board arm beats the control. Frozen is
significantly *worse* than no board in every band; trainable is parity, and
its interval excludes an improvement over control larger than ~0.8 points at
this budget.

**H-freeze: RESOLVED, training wins.** The trainable trunk beats the frozen
one by +2.6 points of win rate, p = 0.002, positive in all six bands. A
margin-pretrained representation frozen into the actor hurts; letting PPO move
it recovers to parity with no board. This matches the earlier finding that a
frozen self-play potential did not transfer out of distribution, and it
closes the freeze arm.

**The in-training telemetry pointed the other way.** On training opponents
both board arms ended *above* control (money 53.7k / 59.9k vs 40.4k, win-vs-
gate 0.56 vs 0.32) and lost or tied on held-out sessions. The training mix
explains the gap: the curriculum's frontier bin is the 0–600 bin for the whole
run (its rolling win rate never reaches 0.55), so ~35% of episodes are against
0–600 opponents and ~35–42% against the heuristic, while only ~12% are against
randomly drawn rated sessions. The eval is 700 of 785 games against 1000+
opponents. Training exposure to that population was on the order of 130
episodes per arm. This affects all arms equally, so it does not touch the
freeze comparison, but it is the leading reason no arm improved and the reason
in-loop numbers must never be read as verdicts.

**The trunk leak cuts against the board arms, not for them:** any memorised
held-out outcome would have helped them, and they still did not win.

**Closed:** freezing the trunk. **Open:** whether a board helps at all — the
trainable arm was judged after 1,280 PPO episodes, from a BC readout fitted
toward a board-blind teacher, on a training mix dominated by opponents the
eval barely contains. The next arms, in the order the evidence suggests:
(1) train on the deployment distribution — draw session opponents by the
eval's band weights instead of the frontier bin, for control and trainable
alike; (2) a BC readout from the session corpus, whose implied choices *do*
depend on the board; (3) the board-aware critic; (4) the phase potential, as a
convergence arm.
