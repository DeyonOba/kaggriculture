# Stage B3 — trainable trunk vs control, on the deployment distribution

## 0 What B2 established and what it left

`docs/stage_b2_preregistration.md` §5. Frozen trunk: −3.3 points of win rate
against no board, z −4.05, worse in all six bands — closed. Trainable trunk:
parity with control, −0.8 ± 0.8, with +2.6 points over frozen, p 0.002.

Both board arms beat the **training** opponents by $13–19k and 0.56 vs 0.32
win-vs-gate, then lost or tied **held out**. The frontier curriculum is the
identified cause: the frontier bin never left 0–600 (its rolling win rate
peaked at 27% against a 0.55 threshold), so ~35% of episodes were against
0–600 tapes and ~35–42% against the heuristic, ~12% against random rated
sessions. The paired eval is 700 of 785 games against opponents rated 1000+.
Each arm saw that population for roughly 130 episodes.

## 1 The one change

`--session-mix eval --session-frac 0.9`. Session opponents are drawn by bin
with the paired eval's band weights — `min(held-out rated seats, 175)` per
bin, normalised: about 8 / 3 / 22 / 22 / 22 / 22 percent across the six bins —
then an entry from that bin among the **training** sessions. Ninety percent
of episodes are sessions; the remaining ten keep the legacy league /
heuristic / snapshot split. The held-out set fixes the weights and never
supplies an opponent. `--session-mix frontier --session-frac 0.5` reproduces
B2's draw exactly and stays the default.

Everything else is B2: `--seed 1234`, 20 iterations of 64 episodes,
`--reward win`, the same LR schedule, `--eval-per-bin 8`, and the **same BC
checkpoints** (`rl/runs/stage_b2/bc_control/bc.pt`, `bc_trainable/bc.pt`), so
both arms start from the exact policies B2's arms started from. Frozen is not
rerun.

Tests: `tests/test_train_guards.py` pins the draw — session share, per-bin
shares within 1.2 points of the weights over 20k draws, the learner seated
opposite the tape, the legacy proportions reproduced under `frontier`, the
remainder split invariant to the session fraction, and the band weights read
from the same draw the eval uses.

## 2 Hypotheses and stop rule

`rl/session_eval.py`, 785 held-out games per arm on identical sessions and
seats (seed 0, the same 785 games as B2), temperature 0, McNemar on
discordant games. Three arms: **B3 control** (baseline), **B3 trainable**,
and **B2 control replayed**.

- **H-curriculum**: B3 control beats B2 control on win rate, p < 0.05,
  positive in ≥ 4 of 6 bands. This is the direct paired test of the fix,
  same BC start, same seeds, same games, only the opponent mix changed.
- **H-board-3**: B3 trainable beats B3 control, p < 0.05, positive in ≥ 4 of 6
  bands. A null here, with the curriculum fixed, closes the trainable trunk at
  this budget (1,280 PPO episodes from a heuristic-BC readout).
- **H-transfer**: the in-loop telemetry direction (money, win-vs-gate on
  training opponents, `eval-per-bin 8` session win rate) agrees in sign with
  the held-out paired result for both arms. Reported, not gated.
- **Replay check**: B2 control replayed must reproduce 47/785 exactly. A
  mismatch voids the eval and is investigated before anything is read.
- **Guards**: any arm with `policy_updates == 0` after warm-up in ≥ 2
  iterations, or `|onpolicy_kl|` above tolerance, is not read.

Power as B2: paired delta se ≈ 0.008, so 0.025 resolves at z ≈ 3 and 0.016 at
z ≈ 2.

## 3 Known limitations

The trunk file still predates the held-out exclusion in pretraining; in B2
that could only have helped the board arm, and it did not. The BC readout is
still fitted toward a board-blind teacher; a session-corpus BC is the next
arm if H-board-3 is null. The in-loop eval's `train-bins` win-rate report is
now telemetry over a different mix than B2's and is not comparable to B2's
lines.

## 4 Results (2026-09-24, eval landed 22:29)

**Replay check: PASSED.** B2's control replayed per-game identical to its
original, 47/785, money $61,072. **Guards:** on-policy KL ≤ 2.4e-7 on all 40
iterations, every post-warm-up iteration updated, trunk loaded at 0.0 and
moved under PPO. The run is read.

| arm | wins / 785 | win rate ± se | money | 0–600 | 600–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500+ |
|---|---|---|---|---|---|---|---|---|---|
| B3 control | 29 | 0.037 ± 0.007 | 57,129 | 0.109 | 0.190 | 0.040 | 0.006 | 0.034 | 0.023 |
| B3 trainable | 33 | 0.042 ± 0.007 | 61,619 | 0.203 | 0.095 | 0.017 | 0.000 | 0.034 | 0.051 |
| B2 control (replay) | 47 | 0.060 ± 0.008 | 61,072 | 0.156 | 0.190 | 0.051 | 0.029 | 0.046 | 0.063 |

| comparison (paired, McNemar) | Δ | se | z | discordant | p | 95% CI | bands ≥ 0 |
|---|---|---|---|---|---|---|---|
| B3 control − B2 control | −0.023 | 0.009 | −2.66 | 14 / 32 | 0.011 | [−0.040, −0.006] | 1 of 6 (at 0) |
| B3 trainable − B3 control | +0.005 | 0.007 | +0.69 | 19 / 15 | 0.61 | [−0.009, +0.020] | 3 of 6 |
| B3 trainable − B2 trainable | −0.010 | 0.008 | −1.30 | 15 / 23 | 0.26 | [−0.026, +0.005] | 1 of 6 |
| B3 trainable − B2 control | −0.018 | 0.009 | −2.07 | 16 / 30 | 0.054 | [−0.035, −0.001] | 1 of 6 |

**H-curriculum: FAILS, in reverse.** Training on the deployment distribution
made the control *worse* by 2.3 points of win rate, p = 0.011, worse in five
bands and level in the sixth. **H-board-3: NULL.** With the curriculum
changed, the trainable trunk is +0.5 ± 0.7 over its control; the interval
excludes a gain above 2 points. Its money is again above control's
(+$4.5k held out, as +$1.2k in B2 and +$13–19k on training opponents): the
board makes bigger banks and does not turn losses into wins, which is the
project's standing pattern (memory: gate on win rate, not margin).

**Mechanism, from the training logs.** Wins per 64-episode iteration against
the gate opponents:

```
B2 control   (frontier mix)  38 34 34 29 26 14 15  9 17 19 14 14 13 16 10 23 14 22 19 24
B3 control   (eval mix)      13  5 12  7  6  8  2  2  6  2  4  5  2  2  2  5  3  1  2  3
B3 trainable (eval mix)      11  6 13  8  3  9  6  5 11  5  8 12  8  4  5  9  7  5  5  5
```

Under `--reward win` the terminal reward is ±15 with a ±2 tiebreaker, so an
iteration with two wins in 64 is 62 near-identical rewards and two informative
ones. The frontier mix, for all that it was stuck on the 0–600 bin, delivered
wins to learn from; the deployment mix starved the signal, and the policy
drifted on the tiebreaker and the entropy bonus. "Train on the deployment
distribution" is right for what to evaluate on and be robust to; it is not,
by itself, a training signal.

**H-transfer:** the in-loop 48-game session win rate read 0.17 at every
control eval while the held-out rate fell from 0.060 to 0.037; still no
information in that number.

**What this points to.** The bottleneck is the density of the win signal
against opponents the policy rarely beats. The instrument built for exactly
this regime is the phase potential (`rl/phase_potential.py`): a dense,
opponent-relative, day-mixed signal that by theorem cannot move the optimum
and can only speed convergence to the same win-first objective. Wired as
`--potential lead` (default `pv`), with the rollout now recording the lead
ingredients per decision. Next arm: `control` vs `lead` on the eval mix, one
change, paired against B3's control by reusing its per-game results
(`rl/session_eval.py --reuse`).
