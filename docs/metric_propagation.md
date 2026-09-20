# Which metric propagates, and in which phase

2026-09-20. `tools/propagation.py`. 48 metrics × 30 days × 400 games.

A metric **propagates at day d** if its value at day d predicts the final
result. Measured two ways, because the project has both and they are not the
same question: `r_margin` against the final margin, `r_win` against the
binary win.

---

## The measurement that had to be thrown away first

The obvious design — play champ-h8 self-play, correlate every metric at
every day with the outcome — returns **"nothing propagates early"**: the
strongest early metric is r = 0.049 over 2,100 samples.

That is an artifact, not a finding. champ-h8 plays the *same opening every
game*, so there is no early variation for a correlation to find:

| day | mean feature sd | live features (sd > 0.01) |
| --- | --- | --- |
| 2 | 0.0019 | **4 of 48** |
| 6 | 0.0062 | 10 |
| 10 | 0.0363 | 34 |
| 25 | 0.0481 | 31 |

The fix is to **randomise our own opening** — 12 random legal farmer actions
injected into days 0–8. That creates variation to measure against, and it
also upgrades the early answer from observational to **interventional**: the
metric differences are caused by an exogenous coin flip, so the association
is a causal effect for that perturbation distribution, not a confound.

Caveat worth keeping: random actions mostly *damage* play, so an early
metric partly measures "how much did the perturbation break". The ranking is
trustworthy; a coefficient is not a lever size.

---

## The answer

### EARLY — days 2–10: INVESTMENT propagates

| metric | r margin | r win |
| --- | --- | --- |
| `melon_tiles` | **+0.379** | +0.256 |
| `invested_lead` | +0.250 | +0.154 |
| `occ_q0` | +0.222 | +0.141 |
| `board_roi` | +0.214 | +0.150 |
| `pending_yield` | +0.151 | +0.102 |
| `seeds_held` | −0.137 | −0.094 |

Melon is the single best early predictor, which fits the economics: $250
base, the highest yield/tile/day of any one-time crop at 0.55, on a 10-day
clock. `seeds_held` is *negative* — seeds in the shed are capital that has
not been put to work.

### CROSSOVER — day 11–12

### MID and END — days 12–29: MONEY LEAD dominates, and nothing else comes close

| day | top | r | runner-up | r |
| --- | --- | --- | --- | --- |
| 12 | `money_lead` | 0.552 | `invest_over_money` | −0.455 |
| 16 | `money_lead` | 0.621 | `invested_lead` | −0.290 |
| 20 | `money_lead` | 0.699 | `seeds_held` | −0.376 |
| 24 | `money_lead` | 0.804 | `seeds_held` | −0.340 |
| 28 | `money_lead` | 0.892 | `invested_lead` | −0.307 |

---

## The new fact: investment FLIPS SIGN at day ~8–10

`E:invested` against the final margin, perturbed openings:

| day | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r | **+0.366** | +0.326 | +0.366 | +0.211 | −0.032 | −0.232 | −0.200 |

Early, capital on the board is an asset. Late, it is a **liability** — money
in the ground is money not in the bank, and only the bank scores. The same
quantity is a virtue on day 4 and a vice on day 20.

This also reconciles the recorded memory (`invested r = +0.304 at day 10`).
The self-play measurement read **−0.169** at day 10 and appeared to refute
it; the perturbed measurement reads +0.366 at days 2–6, decaying through
zero at day 10. The memory was right and my first measurement was the broken
one. Two populations, two questions: in the *corpus*, stronger players
invest more, so investment correlates with skill; in *self-play* with no
opening variation, it correlates with nothing.

---

## How this enters the objective

The main objective is unchanged and stays two-phase, as specified:
**MARGIN to warm up** (dense — 73.6% of within-branch pairs carry it) then
**WIN** (sparse — 13.2% — but what rating measures).

The propagating metrics enter as an **auxiliary task**, not as a reward
shaping term. That distinction was checked rather than assumed:

| technique | applies here? |
| --- | --- |
| PBRS ([Ng, Harada & Russell 1999](https://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)) — `F(s,s') = γΦ(s') − Φ(s)` preserves the optimal policy for any Φ | **No.** It is a guarantee about transforming *RL rewards*. We fit V by supervised regression on played-out outcomes; adding a potential to a supervised target does not inherit the proof, it just biases V. |
| Auxiliary tasks ([UNREAL, Jaderberg et al. 2016](https://arxiv.org/abs/1611.05397)) | **Yes.** Extra heads on the shared trunk. The V target is untouched, so auxiliary gradients can only reshape the representation, never bias the value. |
| Terminal prediction ([Kartal et al. 2019](https://arxiv.org/abs/1907.10827)) | **No — and I had this backwards.** TP estimates *temporal closeness to a terminal state*, which needs episodes that end at DIFFERENT times. Ours are fixed: 719 steps, every game, verified. The target would be a deterministic function of the step counter, which the net is already given (`season_left` is a feature). Zero signal. I first wrote that the fixed season made it *free*; the fixed season is exactly what makes it inapplicable. Cited from a search summary rather than the paper. |

**Only ONE of the three techniques surveyed actually applied.** PBRS is a
guarantee about RL reward transformations, not supervised regression.
Terminal prediction needs variable-length episodes. That leaves auxiliary
tasks — and when it was run, the auxiliary head was a null too (arm B
scored slightly *worse* than arm A on both criteria). Three techniques
surveyed, one applicable, zero that helped.

**The auxiliary target must be a FUTURE value, not the current one.** The
trunk already sees the board, so predicting today's `melon_tiles` is
trivial and teaches nothing. The targets are therefore forecasts:

- `money_lead` **at day 20** — the day it becomes decisive
- `invested_lead` at day 20
- `board_roi` at day 20 — how much committed capital actually converts
- `shed_frac` at day 20

Days-remaining is **not** among them: see the terminal-prediction row above.
What `tools/value_search.py` actually implements is these four day-20
forecasts, z-scored.

Phase weighting follows the crossover: early states are scored mostly on the
investment forecast, late states mostly on the money forecast.

---

## What this does NOT establish

The correlations are ranked evidence about *what to predict*, not a claim
that forcing any metric up wins games. Demand-side and capacity-side
arguments have each failed twice here on exactly that step. The auxiliary
head is safe because it cannot bias V — not because these metrics are
levers. Whether the objective change helps is decided by held-out **sign
accuracy** first, and `ladder_gate` at ≥360 games on two blocks after that.
