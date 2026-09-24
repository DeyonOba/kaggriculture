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
