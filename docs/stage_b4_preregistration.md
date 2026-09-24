# Stage B4 — the phase potential as the dense signal, pre-registered

## 0 Where B3 left it

`docs/stage_b3_preregistration.md` §4. On the eval-band opponent mix the
win-first terminal reward arrived 1–5 times per 64-episode iteration, and the
control trained on that mix lost 2.3 points of held-out win rate to B2's
control on identical games (z −2.66, p 0.011). The board trunk stayed null on
wins. The bottleneck is the density of the win signal against opponents the
policy rarely beats.

## 1 The one change

`--potential lead`: the PBRS potential becomes `rl/phase_potential.py` —
`Φ(s, d) = 15 · tanh(((1 − w(d)) · asset_lead + w(d) · money_lead) / 20,000)`
with `w` ramping from 0 at day 9 to 1 at day 13 — instead of one
ten-thousandth of our own portfolio value. Asset lead is crops plus herd on
our board minus theirs, sheds excluded on both sides; money lead is public.
Nothing is fitted. The shaping weight anneals to zero by iteration 10 as
before, and Φ(terminal) = 0, so the final term charges any lead the engine
does not pay for.

Everything else is B3's control: `rl/runs/stage_b2/bc_control/bc.pt` as the
start, `--seed 1234`, 20 iterations of 64 episodes, `--reward win`,
`--session-mix eval --session-frac 0.9`, the same LR schedule, no board.

**What the theorem allows this to show.** Along any trajectory the shaping
telescopes to a start-state constant, so the optimal policy under `lead` is
the optimal policy under `pv`, which is the optimal policy under the bare
win reward. A gain here is faster convergence to the same objective within
20 iterations. It is not a different objective, and it is judged as a
convergence arm.

## 2 Hypotheses and stop rule

`rl/session_eval.py`, the same 785 held-out games (seed 0, 175 per bin),
temperature 0, McNemar. The `lead` arm is played; **B3 control is the
baseline, reused** from `rl/runs/stage_b3/session_eval.json` after a
game-by-game check that the draw is identical; B2 control is reused for
reference.

- **H-lead**: `lead` beats B3 control on win rate, p < 0.05, positive in
  ≥ 4 of 6 bands.
- **H-recover**: `lead` is not worse than B2 control (the frontier-mix
  control): the paired interval includes zero or lies above it. Reported;
  this asks whether the dense signal recovers what the deployment mix cost.
- **Guards**: `policy_updates > 0` after warm-up, `|onpolicy_kl|` within
  tolerance, `bad == 0`; a violation means the arm is not read.

Power as before: paired se ≈ 0.008; 0.025 resolves at z ≈ 3. A null closes
the dense-potential lever at this budget and points to the curriculum that
tracks the win rate, or to more episodes.

## 3 Results (2026-09-24, eval landed 23:52)

**Guards:** on-policy KL ≤ 8.9e-8 on all 20 iterations, 9.9 updates per
iteration after warm-up, no zero-update iteration, `bad == 0`. Baselines
reused after a game-by-game match. The run is read.

| arm | wins / 785 | win rate ± se | money | 0–600 | 600–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500+ |
|---|---|---|---|---|---|---|---|---|---|
| lead | 14 | 0.018 ± 0.005 | 35,187 | 0.094 | 0.190 | 0.000 | 0.000 | 0.011 | 0.011 |
| B3 control (baseline) | 29 | 0.037 ± 0.007 | 57,129 | 0.109 | 0.190 | 0.040 | 0.006 | 0.034 | 0.023 |
| B2 control | 47 | 0.060 ± 0.008 | 61,072 | 0.156 | 0.190 | 0.051 | 0.029 | 0.046 | 0.063 |

| comparison (paired, McNemar) | Δ | se | z | discordant | p | bands ≥ 0 |
|---|---|---|---|---|---|---|
| lead − B3 control | −0.019 | 0.005 | −3.67 | 1 / 16 | 0.0002 | 1 of 6 (at 0) |
| lead − B2 control | −0.042 | — | — | — | — | 0 of 6 |

**H-lead: FAILS, in reverse.** The lead potential halved the control's
held-out win rate and cut its money by $22k. **H-recover: FAILS.** Training
wins per iteration fell through the run (13 → 1), and training money fell
from $29k to $24k: the policy got worse at both while it trained.

**Reading.** The theorem is asymptotic. With 20 iterations the shaping is not
a value initialisation the learner grows out of; it *is* the transient
objective, because per-step increments of a bounded-15 potential dwarf a
±15 terminal that arrives once in 64 episodes. And the shape rewarded the
wrong thing stepwise: asset lead until day 9, money lead from day 13, so the
crossover pays for converting assets to cash around day 11 and for holding
cash afterwards — a liquidate-and-hoard incentive in a season where mid-game
reinvestment is where the money is. "Invest early, cash late" was measured
as what *correlates* with winning, which is not the same as what to *pay for*
step by step. Both the frozen trunk and this potential are the same lesson:
an offline-measured relation put into the actor's signal does not become an
online improvement.

**Closed at this budget:** the dense-potential lever on the deployment mix.
Together with B2 and B3, none of four levers (board trunk, freezing,
deployment-mix curriculum, phase potential) raised held-out wins; two cut
them. The best configuration on held-out wins remains B2's control
(frontier mix, win-first reward, portfolio potential): 47 of 785.
