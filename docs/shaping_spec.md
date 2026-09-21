# Potential-based shaping: spec, hypotheses, verdicts

2026-09-21. Follows the project workflow: **spec → research → use-case →
experiment → evaluate → commit → train → test over baseline.** Stages 1–6 are
done and in this document. Stage 7 (train) and stage 8 (test over baseline)
are *deliberately* not started, for a reason given in §5 that is stronger
than "no time".

Code: `tools/shaping.py`, `tools/fit_potential.py`, `tests/test_shaping.py`
(28 tests), `rl/train.py` (patched), `rl/datasets/potential.json`.

---

## 1 Spec

Train a temporal model whose **major reward is win rate and margin of
profit**, using our per-phase metrics as sub-rewards, without the
sub-rewards corrupting the objective.

The last clause is the whole problem. This project's record is a list of
intermediate signals that scored well on held-out data and moved the policy
the wrong way:

| signal | how good it looked | what it did |
| --- | --- | --- |
| behaviour cloning | 69.4% action accuracy vs our 27.9% | gated **null** over 4,800 paired games |
| value head V(s)=P(win) | 4.8× tighter than the alternative | **wrong sign on 3 of 4** arms |
| capacity channels | +0.0187 flow R², p = 0.0001 | 0% forecast MAE, **−44% out-of-distribution** |
| melon glut penalty | arithmetically correct on the price curve | **−$1,230**, monotone in λ |
| melon fertilisation | copied from the top-player corpus | **−$14,897** |

So the acceptance criterion for this work is not "the sub-reward correlates
with winning". It is **"the sub-reward provably cannot change the optimum."**

## 2 Research

Searched 2026-09-21; recorded in `docs/research_reading.md` §7. One result
carries the design.

**Potential-based reward shaping** (Ng, Harada & Russell, ICML 1999). For any
state function Φ,

```
F(s, a, s') = γ·Φ(s') − Φ(s)
```

added to the reward leaves the set of optimal policies **exactly** unchanged,
and that form is *necessary* as well as sufficient. Φ may be fitted, wrong,
mis-scaled or adversarial: it changes how fast a learner converges and cannot
change what it converges to.

Three riders that turned out to matter here:

- the γ **multiplies the successor potential** and only the successor. It is
  not cosmetic — it is what makes the discounted sum telescope.
- **time-dependent Φ(s,t) keeps the guarantee** (Devlin & Kudenko, AAMAS
  2012). Needed, because our corpus says the right potential is not
  stationary: board capital correlates **+0.366 with the outcome at day 4 and
  −0.200 at day 20**, and the money lead is worth nothing before day 14
  (r ≈ 0.00, sign-correct 50%) and everything after (0.552 → 0.892).
- along any trajectory the shaping **telescopes to a constant per start
  state**, so for any search that reads only the episode return it is
  identically a no-op. §5.

Sources:
[PBRS survey](https://www.emergentmind.com/topics/potential-based-reward-shaping) ·
[Wiewiora, Q-init equivalence, arXiv:1106.5267](https://arxiv.org/pdf/1106.5267) ·
[two-player extension, arXiv:1401.3907](https://arxiv.org/pdf/1401.3907) ·
[Devlin & Kudenko, dynamic PBRS (AAMAS 2012)](https://dl.acm.org/doi/pdf/10.5555/2343576.2343638) ·
[policy-invariant shaping](https://www.emergentmind.com/topics/policy-invariant-reward-shaping) ·
[reward engineering survey, arXiv:2408.10215](https://arxiv.org/html/2408.10215v1) ·
[TCA survey, arXiv:2312.01072](https://arxiv.org/pdf/2312.01072) ·
[ES on episodic returns, arXiv:1703.03864](https://arxiv.org/pdf/1703.03864)

## 3 Use case: what the trainer was doing instead

`rl/train.py::_episode_advantages` already had shaping. It was not
potential-based, in three separate ways, and each of the three pushes the
policy toward something the ladder has **already refuted**.

```python
r  = shaping_w * (pvs[i+1] - pvs[i]) / 1000.0      # 1. no gamma
r -= CARRY_COEF * recs[i]["stock"] / 1000.0        # 2. per-step state cost
...
r -= UNSOLD_COEF * ep["unsold_value"] / 1000.0     # 3. terminal double-charge
```

**1. The missing γ.** `Φ(s') − Φ(s)` does not telescope under a discount. The
residue has closed form

```
bias = (1 − γ) · Σ_t γ^t · Φ(s_{t+1})
```

— strictly positive wherever Φ is, and **proportional to how long value is
held**. With Φ = portfolio value that is a standing payment for *not*
liquidating, against an engine whose scoring rule is that unsold inventory is
worth zero. Measured on 400 real trajectories at γ = 0.997: **1.60 reward
units per episode, 13.3% of the entire win signal** (WIN_COEF 6.0, swing 12),
at full curriculum weight.

**2. `CARRY_COEF × stock`.** A per-step cost on a state variable is not a
difference of anything, so it is not potential-based, so it moves the
optimum. It moves it toward selling held product *earlier* — and selling
earlier is precisely `x-mel100`, which gated **−$1,482 (z −3.52)**, against a
corpus that says melon's median sell day correlates **+0.136** with winning.

**3. `UNSOLD_COEF × unsold_value`.** The engine scores SOLD product only, so
unsold stock is already worth 0 in `final_money`. Subtracting 0.2× its market
value on top prices it *negative*. This is the glut-penalty failure again,
verbatim: **before adding a term, find where in the objective the quantity
already enters.**

Defects 1 and 2 point in opposite directions, which is why the whole thing
was survivable and why it needed three hand-tuned magic constants.

## 4 Experiment: nine hypotheses, all decided

H1–H6 and H9 are decided by **exact value iteration on constructed MDPs** —
no sampling, no seeds, no tolerance beyond float epsilon. H7–H8 are decided
on 400 real games.

| # | hypothesis | verdict |
| --- | --- | --- |
| **H1** | correct PBRS telescopes to a constant per start state | **held** — 1,000 random trajectories × 5 discounts, exact to 1e-9 |
| **H2** | the un-discounted form does not, and the residue is `(1−γ)Σγ^tΦ` | **held** — closed form matches; bias grows >2.5× from a 10-step to a 30-step hold |
| **H3** | correct PBRS never changes the greedy policy | **held** — 900 random MDPs × 3 discounts, Φ drawn 50× larger than the rewards; zero strict-optimum changes |
| **H4a** | the missing γ makes never-selling optimal | **CONFIRMED defect** — flips at exactly `Φ > sale price`, i.e. generically, all season |
| **H4b** | the carry cost makes selling early optimal | **CONFIRMED defect** — first-sell day is monotone earlier in the charge, and reaches day 0 |
| **H4c** | double-charging unsold stock flips the last day to dumping | **CONFIRMED defect** — at the shipped 0.2 coefficient |
| **H4x** | *no* Φ can produce the H4b flip | **held** — 500 random potentials, none moves it |
| **H5** | Φ(terminal) must be 0 | **held** — otherwise the boundary term depends on when you stop |
| **H6** | the potentials obey the two measured reward-design rules | **held** — Φ is odd in the leads, crossover is monotone, `validate()` catches violations |
| **H7** | a fitted Φ sees more of the final margin than cash does | **held, and large** — see below |
| **H8** | imposing reward-design Rules 1 and 2 costs accuracy | **REFUTED — they are free** |
| **H9** | PBRS moves an episode-return gate | **REFUTED — it is exactly a no-op**; §5 |

### H7/H8, measured

`rl/datasets/propagation_perturb.npz`: 400 games × 30 days × 48 metrics, with
the final margin (`M`) and the win (`W`) attached — *these are the "margin and
win" columns*. Split **by game**, never by row (30 rows share one label);
ridge α chosen by grouped K-fold inside train only; test games untouched by
any fit or selection. Six independent splits:

| Φ | R² vs final margin | AUC vs win |
| --- | --- | --- |
| **fit, all 48 metrics** | **+0.390 ± 0.048** | **0.775 ± 0.031** |
| fit, obeying Rules 1 & 2 | +0.381 ± 0.049 | 0.769 ± 0.032 |
| `LeadPotential` (zero-fit) | +0.232 ± 0.037 | 0.742 ± 0.028 |
| cash only (what the trainer had) | +0.240 ± 0.037 | 0.728 ± 0.022 |

Paired on the split, **fit − cash = +0.150 ± 0.019 R²**. The rule-constrained
fit gives up 0.009 ± 0.019 — nothing. So **H8 is refuted and that is the good
outcome**: the two rules are free, so we take them, and Φ is then built only
from quantities an independent 21,054-game study says carry signal.

The per-day table is the interesting part:

```
day        fit(rules)        cash only
  2     +0.153 / 0.687    -0.137 / 0.562
  5     +0.191 / 0.655    -0.136 / 0.536
  9     +0.338 / 0.714    -0.068 / 0.548
 11     +0.367 / 0.748    +0.217 / 0.742      <- cash switches on here
 20     +0.506 / 0.835    +0.329 / 0.838
 29     +0.871 / 0.996    +0.693 / 0.991
```

**Cash-only Φ has NEGATIVE R² on all eleven days 0–10** — worse than
predicting the mean — and averages **0.52 AUC** over them, dipping to 0.383
(below chance) at day 6. That independently reproduces
`docs/reward_design.md` Rule 2 on a different dataset and a different
estimator. The fitted Φ runs **0.66–0.72 AUC over days 2–10, mean 0.70**, so
there *is* early-game signal about the eventual win; it is
simply not in the money, and the trainer's portfolio-value potential could
not see it.

### Two defects found in our own fitting code, worth recording

Both were caught by out-of-sample scoring and would have been invisible to
in-sample fit quality.

1. **R² = −214 on day 17.** A near-constant column (`D:drain_cover`, train sd
   8.5e-8) survived a 1e-9 variance clamp; de-standardising gave it a
   coefficient of ~1e7 and the intercept cancellation had no float64
   precision left. Fixed by *dropping* degenerate columns, plus
   `_check_destandardised`, which re-derives predictions both ways and fails
   the fit if they disagree.
2. **R² = −214 again, for a different reason.** `E:money_over_next_land` at
   day 17 has train sd 4.5e-4 and one **test row 3,677 σ outside** the
   training range; the linear model answered −$850,541 against a margin whose
   entire sd is $5,563. Fixed by raising the variance floor and by clamping Φ
   to the range the fit actually saw. A clamped state function is still a
   state function, so invariance is untouched.

## 5 Evaluate — and the limit that decides the whole plan

**PBRS cannot move any number this project currently gates on.** The shaping
telescopes to `−Φ(s_0)`, and in a paired gate both arms start from the *same*
`s_0` (same seed, same board, same opening stock), so the paired difference
is unchanged bit for bit (H9). That covers `tools/ladder_gate.py`, the knob
bandit, CEM, evolution strategies, and the schedule rollout — **every
instrument in the shipping path.**

So Φ is worth exactly nothing unless something **bootstraps per step**. There
is precisely one such thing in this repo: `rl/train.py`'s GAE critic. That is
why the deliverable here is a corrected trainer and a scored Φ, and not a
gate result: **a gate result on this change is not merely missing, it is
mathematically guaranteed to be zero.** Anyone who reports one has measured
noise.

## 6 Commit

Shipped to `rl/train.py`:

```python
phis  = [r["pv"] / 1000.0 for r in recs]
shape = shaping_terms(phis, GAMMA, terminal_potential=0.0)   # γΦ(s') − Φ(s)
r     = shaping_w * shape[i]
```

`CARRY_COEF` and `UNSOLD_COEF` are removed from the reward path and survive
only under `--legacy-shaping`, which reproduces the old curriculum verbatim
for replaying historical runs. The unsold penalty needs no replacement:
**Φ(terminal) = 0 already charges it**, since the final shaping term is
exactly `−Φ(s_last)` — at the potential's own scale, with no free coefficient
to tune. `tests/test_shaping.py::test_trainer_uses_the_discounted_form` is a
guard so the missing γ cannot come back; it survived fourteen PPO generations
unnoticed the first time.

## 7 Expected output of the model

Asked directly, and it needs a direct answer, because three different answers
are available and the evidence only supports one.

**(a) A policy π(s) → action.** Refuted four independent ways: PPO plateaued
at rating 810 over 14 generations and never beat the hand-written heuristic;
BC at 69.4% accuracy gated null over 4,800 paired games; the full neural
planner gated **−$22,887**; a cloned executor policy went **−$1,052 → +$1,233**
across an executor swap, i.e. it did not transfer at all.

**(b) A value function V(s) = P(win) used to rank actions.** Refuted: four
heads (two encoders × two label schemes, plus auxiliary) all order actions at
**chance** out-of-sample, 0.503/0.500 on search labels.

**(c) A 30-day schedule — a vector of per-day scalars the planner reads.**
The only thing that has ever converted compute into a replicated
out-of-sample gain here: the hiring rollout, **+$3,287 ± 1,528 over 192 fresh
worlds**, where the same lever as a *constant* is refuted and monotone wrong
(+2 hands −$5,229, +4 hands −$19,344). The recorded conclusion is the design
brief: *the knob space was never the plan space.*

**So the expected output is (c), and Φ's job is not to be that output.** Φ is
a training-time device whose entire contribution is to tell a bootstrapped
critic *which day* of a 720-step episode earned the money — turning one
terminal scalar into 30 dense, policy-invariant per-day targets. The artefact
that ships is the schedule the trainer produces; Φ never appears in
`kagent/`, is never consulted at inference, and adds nothing to the
submission bundle.

And this is the first construction in the project where **a fitted model is
safe by theorem rather than by gate.** Every previous one entered as a
decision rule and could move the policy wrong. Φ enters as a reward
transform that provably cannot. That inverts the standing risk — which is
exactly why it is worth building a fitted Φ here after the fitted-model
record above.

## 8 Are we leaving something out?

Five things. Two are ordinary gaps; three are load-bearing.

**(i) There is no bootstrapped learner in the shipping path — load-bearing.**
§5 proves Φ is a no-op for every instrument we actually gate on. `rl/train.py`
is the only consumer, it is the one component whose own track record is
"plateaued at 810", and its failure was diagnosed in `docs/training_algorithms.md`
as *credit assignment arithmetic, not tuning*. That diagnosis is now
partially explained — one of the three shaping terms was paying 13.3% of the
win signal for stalling — but "we fixed the reward and PPO still may not
beat a heuristic at 1e5 episodes/day" remains the live risk. **The correct
next experiment is a cheap falsification**: re-run PPO with the corrected
shaping against the `--legacy-shaping` control on identical seeds and see
whether the liquidation pathology disappears. If it does not, Φ quality is
irrelevant and the answer is (c) by schedule search with no learner at all.

**(ii) Φ is fitted on self-play trajectories, and deployment is not self-play
— load-bearing.** `propagation_perturb.npz` is champ-h8 against champ-h8 with
randomised openings. `tools/net_ood.py` exists because a net trained on a
21%-of-corpus slice closed +71% in-distribution and −13% against champ-h8;
`ladder_gate` draws across six rating bins. Φ has **not** been scored
out-of-distribution. It cannot become *unsafe* — the theorem does not care —
but +0.39 R² in self-play may be +0.05 on the ladder, and then it buys
nothing. Scoring it on ladder trajectories is cheap and is the next item.

**(iii) Margin and win are two objectives and Φ is fitted to one.** Φ targets
the margin, because margin is continuous and lower-variance. But **the rating
is win-based**, and the two have now come apart four times (`match-all`
+$1,381 with wins 5/144 → 4/144; the hiring rollout +$1,215 with wins dead
flat; `h9-carry` −$521 losing 54–34 on wins; and again across this week's
null-positive arms). Φ is currently a *margin* potential. A win-probability
potential is one flag away (`fit(..., target="win")`) and has not been run.
Under PBRS both are safe; they are not equally useful, and we have not
measured which.

**(iv) The Friday event's own outputs are not reproducible — ordinary but
should be fixed.** `metric0`/`metric1` are `tools/kaggle_job/job_metrics0.json`
and `job_metrics1.json`, the two shards of `tools/metric_harvest.py` that
produced `docs/reward_design.md` from 21,054 games. **The `metrics_0.npz` and
`metrics_1.npz` they emitted are not in the repo and no tool reads them.**
Every number in `reward_design.md` — the ones Φ's feature rules are built on
— currently exists only as prose in a markdown file. Re-fetching the shards
and adding a reader would let those rules be re-derived instead of quoted.

**(v) The endgame is still unmeasured.** `docs/game_phases.md` defines four
phases; `tools/phase_metrics.py` instruments the opening; the mid and end
balance sheets were never built. Φ's *largest* coefficients now sit in the
endgame (`D:unsold_if_dumped` at −84,746 on day 24) and they are fitted, not
understood. Given that the endgame is a scheduling problem with a known
optimum shape, the arithmetic version is likely to beat the fitted one —
which is the pattern every time it has been tested here.

### Not on the list, deliberately

DreamerV2-style world modelling and RUDDER/TVT return decomposition both
address our problem shape on paper. Both have to beat `tools/action_value.py`,
which answers "which of the 720 turns earned the money" **exactly**, by
deterministic single-action deletion, at 1.09 s per evaluation — and which
already measured the 13× decay from $1,317/action on day 3 to $103 on day 27.
A learned decomposition that is merely approximate is a downgrade.
