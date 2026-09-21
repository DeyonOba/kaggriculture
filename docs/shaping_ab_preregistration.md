# Shaping A/B: predictions, written before the run

2026-09-21. `docs/shaping_spec.md` §8(i) named the next experiment:

> re-run PPO with the corrected shaping against the `--legacy-shaping` control
> on identical seeds and see whether the liquidation pathology disappears.

This file states what each outcome would mean, **before** the run, so the
result cannot be read after the fact into whichever story it happens to fit.
That matters here more than usual: this project has twice recorded a
conclusion from a measurement that was later found to be an artifact.

---

## 0 What changed between the spec and this file

`docs/shaping_spec.md` sized the defects from a *proxy* Φ reconstructed out of
tanh-squashed propagation metrics. Running 24 real episodes through
`rl/rollout.run_episode` and reading `pv`, `stock` and `unsold_value` directly
off the records changes three numbers, two of them materially.

| legacy reward term | spec said | measured, 24 real episodes | status |
| --- | --- | --- | --- |
| stall bonus (missing γ) | 1.60 | **+4.790 ± 0.135** | **3× larger** |
| `final_money` injected twice | not quantified | **+113.910 ± 5.271** | **the biggest term** |
| carry cost on held stock | assumed active | **+0.000 ± 0.000** | **inert** |
| terminal unsold penalty | assumed active | **+0.000 ± 0.000** | **inert** |

Mean Φ over a season is **52.58** reward units, not the 18.87 the proxy gave;
that is the whole factor-of-three. Corrections:

**The stall bonus is 40% of the win signal, not 13.3%** (4.790 against a win
swing of 2 × WIN_COEF = 12.0). `docs/shaping_spec.md` and the changelog
understated it.

**There is a fourth defect, and it is the largest.** The legacy code builds
its potential sequence as `pvs = [r["pv"] for r in recs] + [ep["final_money"]]`,
i.e. it sets **Φ(terminal) = final_money instead of 0**. The final shaping
term therefore carries the whole season's bank *again*, at ~114 reward units,
on top of `MONEY_COEF × final_money = 124.28`. So while the curriculum weight
is on, the legacy reward pays roughly **238 for money and 12 for the win** — a
20:1 ratio in favour of the quantity this project has repeatedly recorded as
the wrong objective (*"gate on win rate, not margin"*; the rating is
win-based). The spec listed this under H5 as a convention violation and did
not price it. It should have.

**Two of the three defects the fix removes are inert on the current
trajectory distribution.** `stock` and `unsold_value` are **exactly zero in
all 24 episodes** — the heuristic liquidates completely and holds nothing at
the hour-0 decision point, so `CARRY_COEF` and `UNSOLD_COEF` never fire. The
H4b/H4c MDP proofs stand: those terms *would* move the optimum if the policy
ever reached states where they bite. They are latent, not active, and the
honest reading is **one large active defect (stall), one larger unpriced one
(money double-counted), two latent ones**.

## 1 Design

Both arms differ in exactly one argument.

```
shared    rl/train.py --bc-games 60 --iters 0 --seed 1234   -> bc.pt
arm A     --resume bc.pt --seed 1234 --iters 20 --episodes 64 --patience 99
arm B     --resume bc.pt --seed 1234 --iters 20 --episodes 64 --patience 99 \
          --legacy-shaping
```

Controlled, and why each is needed:

- **the same BC checkpoint**, so both start from an identical policy and
  critic rather than from two draws of the initialiser;
- **`--seed 1234`** pins `torch.manual_seed` and `random.seed`, which is what
  the net init and the minibatch shuffles ride on. Rollouts were already
  deterministic in (weights, world seed, side) — `rl/rollout.py:324,350`;
- **`rng = random.Random(it)`** inside `ppo_phase` already gives both arms the
  identical world seeds, opponents and sides at every iteration;
- **`--patience 99`** so neither arm early-stops, which would leave the two
  curves ragged and non-comparable;
- **20 iterations** because `shaping_w = 1 − it/(iters·0.5)` anneals the
  shaping to zero at the halfway point: iterations 0–9 are the only ones where
  the arms can differ at all, and 10–19 measure whether the difference
  persists after the shaping is switched off.

## 2 Measurement

The primary outcome is **not** final money or eval win rate. At 64 episodes an
iteration against a per-world spread of ~$30–40k, neither can resolve
anything, and this project has a standing rule that win-rate comparisons need
~1,000 games per arm.

The primary outcome is **mechanism**: the behaviour the defective term is
predicted to buy. `rl/train.py` now writes `telemetry.jsonl` per iteration
with `unsold` (season-end market value of everything not sold — what the
engine scores at **zero**), `held_stock` (mean stock across decisions),
`money`, and `win_vs_gate`.

## 3 Predictions

**P1 — primary, mechanism.** The legacy arm drifts toward holding. Its
`unsold` and/or `held_stock` rise above the corrected arm's during iterations
0–9, because the stall bonus pays +4.79 per episode for keeping Φ high and
the two terms that would counterbalance it measure exactly 0.00. The
corrected arm's `unsold` stays at or near zero throughout.

**P2 — secondary, variance.** The legacy arm's per-iteration `money` is more
volatile, because the double-counted terminal money injects world-scale
variance (±$30k → ±30 reward units) into the advantage a second time.

**P3 — the null, and it is a live possibility.** Both arms keep `unsold ≈ 0`
and `held_stock ≈ 0` for all 20 iterations. Then the defects were latent
rather than active, the fix is **correct but inert**, and the conclusion is
that the shaping was never the binding constraint on this trainer. That would
corroborate `docs/training_algorithms.md`'s recorded verdict — *"training was
not the constraint; the economy the executor could express was"* — and it
would mean the right next move is (c) from `docs/shaping_spec.md` §7: schedule
search with no learner at all, where PBRS is provably a no-op anyway.

P3 is the outcome the 24-episode measurement above makes most likely, and
saying so now is the point of writing this down first.

## 4 What this experiment cannot decide

- **Whether PPO can beat the heuristic.** 20 iterations × 64 episodes is 1,280
  games; the recorded failure took 14 generations. A null here is not evidence
  that the corrected trainer cannot work.
- **Anything about the ladder.** PBRS is provably a no-op for every paired
  episode-return instrument we gate on (`docs/shaping_spec.md` §5), so no
  ladder number can move and none will be reported.
- **Whether Φ from `rl/datasets/potential.json` helps.** This run uses the
  trainer's existing portfolio-value potential for both arms. Swapping in the
  fitted Φ is a separate experiment and is deliberately not confounded with
  this one.

---

## 5 Amendments, recorded 2026-09-21 after launch and before divergence

Both written while the arms were still producing **bit-identical** telemetry,
so neither is a post-hoc reading of a result.

**A1 — the differential window is iterations 3–9, not 0–9.** `WARMUP_ITERS = 3`
makes the first three iterations **value-head-only** ("advantages from an
untrained critic would wreck the BC policy"), and the value net is decoupled
from the policy. So the policy is untouched through iteration 2 and the
rollouts are identical; the two critics diverge immediately but cannot reach
the policy until iteration 3. Combined with `shaping_w = 1 − it/10`, the
shaping can influence behaviour for **seven iterations, 448 episodes**, after
which the curriculum weight is zero and iterations 10–19 measure persistence.
Confirmed empirically: iterations 0 and 1 match to the last digit in both
arms (money 45,987 → 41,333; unsold 277 → 394; held 3,729 → 3,696). That is
the control working, and it is also the design's main limitation — a null
result would be consistent with "no effect" *and* with "window too short",
and distinguishing them needs a longer anneal, not more seeds.

**A2 — `CARRY_COEF` and `UNSOLD_COEF` are NOT inert on the trajectories that
matter.** §0 measured both at exactly 0.000, but on 24 episodes run in
`mode: "bc"` — the **heuristic teacher**, which liquidates completely. The
BC-initialised *neural* policy does not: iteration 0 reports mean held stock
**$3,729** and terminal unsold **$277**. So on the distribution the trainer
actually trains on, the two terms are

```
carry   0.005 × 3,729/1000 × 30 decisions  ≈ 0.56 reward units/episode
unsold  0.2   ×   277/1000                 ≈ 0.06 reward units/episode
```

— small, but not zero. They sit an order of magnitude below the stall bonus
(4.79) and two below the double-counted money (113.9). The ranking of the
four defects is unchanged; the word "inert" was wrong and the right word is
**"minor on the teacher's distribution, small on the learner's"**. This is a
worked example of the measurement error the amendment exists to catch:
`mode="bc"` and `mode="rl"` are different distributions, and a quantity that
is identically zero on one can be nonzero on the other.

---

## 6 Results

Both arms ran all 20 iterations. `rl/runs/shaping_ab/{A_pbrs,B_legacy}/telemetry.jsonl`.

**The control worked exactly.** Iterations 0–3 are identical to the last digit
in both arms — 0–2 because `WARMUP_ITERS` updates only the critic (A1), and 3
because the first policy update lands *after* that iteration's rollout.
Divergence begins at iteration 4, which is the only place it could begin.

### P1 — primary, mechanism: **REFUTED**

Paired on iteration (both arms see identical world seeds, opponents and
sides at every iteration), A = corrected, B = legacy:

| window | `unsold` A−B | `held_stock` A−B | `win_vs_gate` A−B |
| --- | --- | --- | --- |
| 0–2 warmup | +0.0 ± 0.0 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| **3–9, shaping active** | **+10.0 ± 22.9** (t +0.44) | **−3.9 ± 13.5** (t −0.29) | −0.00 (t −1.74) |
| 10–19, shaping off | −26.9 ± 14.1 (t −1.91) | −12.7 ± 10.6 (t −1.20) | +0.00 (t +0.01) |

The prediction was that the legacy arm drifts toward holding, i.e. **A−B
negative** on `unsold` while the shaping is on. It is **+10.0 ± 22.9** —
wrong sign, and a twentieth of a standard error from nothing. `unsold` stays
in a $277–$468 band and `held_stock` in a $3,500–$3,900 band, in **both**
arms, for all twenty iterations. Neither drifts.

**There is no liquidation pathology to remove at this budget.** The stall
bonus is real and arithmetically large (+4.79/episode, 40% of the win swing),
and 448 episodes of differential training did not convert it into behaviour.

### P2 — variance: direction right, not resolved

Per-iteration `money` sd over iterations 3–19: **legacy 7,489, corrected
6,656** — legacy 12.5% more volatile, as predicted, on a variance ratio of
1.27 with n = 17. Nowhere near resolvable. Reported as a direction, not a
result.

### P3 — the null: **essentially confirmed**, with one correction

P3 said both arms would keep `unsold ≈ 0` and `held ≈ 0`. They keep both
**nonzero and identical** — $380 and $3,700 respectively — which is the same
conclusion by a different route: the defects did not produce differential
behaviour. **The corrected shaping is correct and, at this scale,
behaviourally inert.**

### Unpredicted: the legacy arm banks more money and wins exactly as often

| window | `money` A−B | `win_vs_gate` A−B |
| --- | --- | --- |
| 3–9 (clean) | −1,053 ± 853 (t −1.23) | −0.00 (t −1.74) |
| 10–19 (confounded) | −1,676 ± 655 (t −2.56) | +0.00 (t +0.01) |
| 3–19 | −1,420 ± 510 (t −2.78) | −0.00 (t −1.07) |

Two things have to be said about this, and the order matters.

**It is confounded after iteration 9.** `PLATEAU_COOL` cuts the learning rate
on evals that fail to improve, so the LR schedule is *endogenous to each
arm's own eval score*. By iteration 14 arm A was at lr 4.5e-05 and arm B at
7.5e-05 — B trained the back half at ~1.7× the learning rate. The 10–19
money gap is therefore not attributable to the reward change. **The clean
window is 3–9, and there the money difference is not resolved (t −1.23).**

**And in every window the wins are identical.** If the effect is real it is
exactly what the arithmetic predicts — the legacy reward pays for money twice
(≈238 for money against 12 for the win, §0) so it is the stronger money
optimiser — and it is worth nothing, because the rating is win-based. That
would make this the **fifth** independent occurrence of margin and win coming
apart in this project, after `match-all`, the hiring rollout, `h9-carry`, and
this week's null-positive planner arms.

### The result neither arm can hide

Both arms **destroyed the BC policy and never recovered it**. `win_vs_gate`
falls 0.59 → 0.13 by iteration 7 and claws back only to ~0.46 by iteration
19; the held-out session win rate sits at **0.06 at every single eval** in
arm A (0.06 / 0.11 / 0.06 / 0.06 in arm B), against the rl-v9 baseline of
0.08 recorded in the trainer's own comments. Twenty iterations of PPO from a
66.3%-accurate BC start produced nothing, under *either* reward.

That is the finding with the most weight in it, and it is not about shaping.
It corroborates `docs/training_algorithms.md` exactly as §3 of that document
states it: **"training was not the constraint; the economy the executor could
express was."** Fixing the reward was necessary — the four defects are real
and three of them are now proved to move the optimum — and it was not
sufficient, and nothing about the reward is what is stopping this trainer.

### What would have to change for a re-run to say more

1. `--plateau-cool 1.0`, to remove the endogenous LR and make the money
   comparison interpretable past iteration 9.
2. A longer anneal. Seven iterations of differential training is thin; the
   recorded PPO failure took fourteen generations.
3. Swap in `rl/datasets/potential.json` — the fitted Φ that scores +0.381 R²
   against +0.240 for the portfolio value this run used. That is the
   experiment that tests whether Φ *quality* matters, and it is the one this
   run deliberately did not confound itself with.

None of the three changes the verdict on P1, which is clean in the
unconfounded window: **no pathology, no divergence, no effect.**
