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
