# Which training algorithm, and why — from this project's measurements

Written 2026-09-17. Everything in §1–§3 is measured here; §4 is where the
literature comes in, and it is used only to *choose between* options that §1–§3
leave open.

---

## 1. The problem's actual shape, in numbers

| property | value | where measured |
| --- | --- | --- |
| horizon | 30 days × 24 turns = 720 steps | engine |
| actors | ~11 farm hands, re-hired nightly | `kaggriculture.py:880` |
| reward | **terminal only** — money at day 29 | engine |
| per-game spread | **~$30–40k** | every gate in `arena/results/` |
| effect size we chase | **~$3k** | best replicated gain, hiring rollout |
| sample cost | **4.08 s/game**, ~5,300 games/hour on 6 cores | `tools/fast_env.py` + timing |
| fitting cost | 10 epochs × 20k samples = **1.33 s** (CPU), 1.42 s (MPS) | measured |
| metric | **win rate**, not money | rating is win-based |
| opponent | adaptive, and the market couples both farms | `docs/market_structure.md` |

Two ratios decide everything else.

**Noise / effect ≈ 12.** A single game says almost nothing. Resolving a $3k
effect unpaired needs ~700 games; paired on identical worlds it needs far
fewer, which is why every instrument here is paired.

**Simulation / fitting ≈ 3,000.** Model training is free; games are not. So the
algorithm must be chosen for **sample efficiency under noise**, and model
capacity is irrelevant. This is also the whole answer to the GPU question — MPS
is *slower* than CPU at our model size, and 100% of the cost is CPU-bound game
simulation.

---

## 2. What has been refuted here, with numbers

| approach | result |
| --- | --- |
| **PPO residual**, 14 generations | never beat the hand-written heuristic; plateaued at rating 810 |
| **Behaviour cloning** from 3,093-rated seats | **69.4%** held-out action accuracy vs our rule's 27.9% — and gated **NULL** over 4,800 paired games |
| **Self-play** objective | 96.1% win rate in the twin arena → **3/144** externally, −$11,566 |
| **Knob search** (30,272 bandit samples) | a handful of small wins (`carrot_day` +$1,545, `premium_scale` +$1,949); the space is now flat |
| **Demand book** from 1,857 champion seats | crops +$231 ± 3,186; herd −$6,637; all books −$6,909 |

Two of these are diagnostic rather than merely negative.

**BC's 69.4% is the important one.** Imitation accuracy more than doubled our
rule's and converted to *nothing*. The head audit explains it: the policy
exposes thirteen heads, ten are **allocation** quantities, and allocation is
measured flat — the LP's provably optimal allocation, executed by our own
scheduler, tied champ-h8 at **+$828 ± 2,317**. There is **no execution head at
all**. So in this codebase, cloning a better player cannot learn execution; it
can only learn targets, and targets are not where the money is.

**PPO's failure is a credit-assignment arithmetic problem, not a tuning
problem.** One scalar at step 720, with per-game noise 12× the effect. The
recorded verdict was *"training was not the constraint; the economy the
executor could express was."*

## 3. What has actually worked

Exactly one thing converted compute into a replicated out-of-sample gain:
**rollout over a per-day schedule.** The same lever is refuted as a constant
(+2 hands −$5,229, +4 hands −$19,344, monotone) and pays as a schedule:
**+$3,287 ± 1,528 over 192 fresh-world games.**

The recorded conclusion is the design brief: **"the knob space was never the
plan space."**

---

## 4. Algorithm families against those constraints

| family | sample need | needs credit assignment? | fit here |
| --- | --- | --- | --- |
| Policy gradient (PPO/A2C) | 1e6–1e8 episodes | yes, per-step | **no** — we have ~1e5/day, and the reward is terminal |
| Offline RL / CQL / IQL | large offline set | yes | weak — same credit problem, plus no execution head |
| **Decision Transformer** | large offline set | no (return-conditioned) | weak *here*: it is BC with a return token, and BC's 69.4% already converted to nothing |
| **Evolution strategies / CMA-ES** | 1e3–1e4 evaluations | **no** | **good** — treats the game as a black box, scales with parameter count not horizon |
| **Cross-entropy method** | 1e3–1e4 | **no** | **good** — uses only the *rank* of the elite set, so noise that preserves ranking is harmless |
| **Successive halving / Hyperband** | fixed budget | **no** | **good** — spends evaluations where the ranking is still uncertain, which is exactly our regime |
| **Bayesian optimisation (GP-EI)** | 1e2–1e3 | **no** | **best on paper** for ≤20 expensive noisy dims; untested here |
| **Expert iteration (AlphaZero-style)** | search-bound | search supplies it | **the principled version of the one thing that works** — our rollout *is* the expert |

The literature that actually bears on the choice, rather than decorating it:

- **Salimans et al., *Evolution Strategies as a Scalable Alternative to RL***
  — ES is competitive with policy gradients precisely when the horizon is long
  and the reward sparse, because its sample complexity scales with the number
  of parameters rather than the horizon. Our reward is maximally sparse (one
  scalar at step 720). This is the single closest match to our situation.
- **Rubinstein, CEM** / **Hansen, CMA-ES** — rank-based updates are invariant to
  monotone transformations of the objective, so an objective as noisy as ours
  degrades them far less than it degrades a gradient estimate.
- **Jamieson & Talwalkar, *Non-stochastic Best Arm Identification*** — under a
  fixed budget with noisy evaluations, successive halving is near-optimal. Our
  budget is fixed by the deadline and our evaluations are very noisy.
- **Anthony et al., *Thinking Fast and Slow* (ExIt)** — search is the
  improvement operator and the network internalises it. Our rollout is a
  working improvement operator; this is how it scales.
- **Shahriari et al., *Taking the Human Out of the Loop*** — GP-based BO
  dominates when each evaluation is expensive and the dimension is small, which
  describes a 13-dimensional plan space at ~2 minutes per evaluation.
- **Cobbe et al., Procgen** — agents overfit small level sets; ~10k levels
  needed. Relevant as a *warning* about our eval, not as a training method:
  the existing pipeline trains on 5,760 fixed worlds and selects the best
  checkpoint on 18 rotating games.

**Where Decision Transformers land.** Worth stating plainly because it was
asked about specifically: DT is a good fit for offline datasets with diverse
returns, and we have 21,517 sessions with ratings. But DT is behaviour cloning
conditioned on a return token, and the BC result above is the decisive local
evidence — 69.4% accuracy, zero money. The blocker is not the conditioning, it
is that the action space our model exposes contains no execution decisions.
**DT becomes interesting only after an execution head exists**, which is E1/E2,
not a training change.

---

## 5. What is being run

`tools/optimizer_bakeoff.py` — four searches, identical space, identical game
budget (1,080 games each), scored on sampled ladder money and re-measured on a
held-out block as a **paired** difference from the base with a standard error:

- random search (the control — if it ties the rest, the space is flat)
- (1+1)-ES from the base plan (closest to the rollout that works)
- cross-entropy method
- successive halving

The space is the **plan** space, not the knob space: four per-day schedules
(hires, strawberry, wheat, land) compressed to 3–4 control points each — 13
dimensions — via `tools/sched_bundle.py`, which applies them all to one frozen
bundle and is verified bit-identical to champ-h8 on the base schedules.

**Bayesian optimisation is missing and should be added**: it is the strongest
theoretical fit in the table and the only listed family with no local evidence
either way.

### Results — all four null, and the reason is arithmetic

Base: **−$11,038/game, win 0.233**, 150 held-out worlds.

| method | search $/game | held-out $/game | held win | **paired vs base** | se | z |
| --- | --- | --- | --- | --- | --- | --- |
| cem | −8,191 | −12,340 | 0.220 | **−1,302** | 1,513 | −0.86 |
| halving | +2,722 | −12,327 | 0.227 | **−1,289** | 1,290 | −1.00 |
| bo | −13,863 | −11,038 | 0.233 | **+0** | 0 | 0.00 |
| hill (paired, σ=0.10) | +5,230 | −13,289 | 0.247 | **−2,251** | 1,582 | −1.42 |

**BO is the only method that did not move** — it returned the base point
exactly, which is the correct action under this much noise and the one method
whose model carries an explicit observation-noise term.

Two things the first round assumed turned out to be wrong, and fixing both did
not rescue it:

- **Pairing works.** Against the base on the *same* world, correlation is 0.95
  at σ=0.10, so a 36-world evaluation's se falls **$5,892 → $1,886**.
- **The step size was the bug, not the method.** A σ=0.25 perturbation costs
  **−$18,944** paired; hill used 0.25 and CEM started at 0.35, so nearly every
  draw landed outside the optimum's basin.

### The budget this search actually needs

Selecting the best of N noisy evaluations inflates the winner by
≈ `sqrt(2 ln N) · se`. For that inflation to stay under half a $3,000 effect,
with paired sd $17,168 and ~1.1 s/game:

| evals N | worlds/eval needed | total games | hours |
| --- | --- | --- | --- |
| 11 | 629 | 6,919 | **2.1** |
| 30 | 892 | 26,760 | 8.2 |
| 50 | 1,025 | 51,250 | **15.7** |
| 100 | 1,207 | 120,700 | 36.9 |

We ran **396 games per method**. Best-of-11 at se $1,886 inflates by **$4,130**
— larger than the effect being chased, which is why every search's own score
looked good and none survived.

**The two requirements do not overlap.** The 2.1-hour run is affordable but 11
hill-climbing steps in 13 dimensions barely leaves the base; the run with
enough steps to matter is ~16 hours. With 13 days left and the executor
untried, that is the wrong place to spend the largest compute block available.

**Conclusion: derivative-free search over the plan space is closed, on budget
grounds.** Note precisely what this does *not* say — it does not show the plan
space is flat. A null from an instrument that cannot resolve the effect carries
no information about the space. What it shows is that finding a $3,000 plan
improvement costs ~50,000 games, and we cannot pay that and still test the
executor.

<!-- superseded detail retained below for the record -->
| (first run, unpaired) | | | | | | |
| cem | −8,191 | −12,340 | 0.220 | −1,302 | 1,513 | −0.86 |

CEM improved monotonically on its 36 search worlds (−12,843 → −9,324 → −9,438
→ −8,191) and **none of it transferred**. The search found the noise in its own
world sample. The first smoke run showed the same shape at 1/50th the size
(hill "won" at −23,027 and returned −30,652 held-out), so this is the regime
rather than an accident.

The lesson generalises past this project: **with 36 worlds per evaluation
against a ~$40k per-game spread, an optimiser's own score is not evidence.**
Only the paired held-out number is, and reporting the two side by side is what
makes the gap visible. This is also the argument for successive halving and for
BO's explicit noise term over CEM's implicit trust in each evaluation.

**What a null would mean.** If all methods come back |z| < 2, that is a result,
not a failure: it says the plan space *around champ-h8* is locally flat, and
the remaining headroom is in the execution layer (E1/E2) rather than in any
plan any search could find. Given that the LP's optimal allocation already tied
champ-h8 at +$828 ± 2,317, a flat answer here would be consistent — and would
redirect the last days of the competition to the executor rather than to more
search.
