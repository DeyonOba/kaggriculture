# Φ out of self-play, and when the field actually changes phase

2026-09-22. Two measurements on the same new instrument: `tools/ladder_panel.py`,
which replays recorded sessions exactly — recorded seed, both seats' recorded
720-step scripts, no agent of ours in the loop — and samples the same 48 per-day
metrics the self-play panel carries. 514 sessions drawn equally across the six
`ladder_gate` rating bands, ratings 197–3210, both seats, 30,840 rows.

---

## 1 Φ does not survive out of self-play

`docs/shaping_spec.md` §8(ii) named this as load-bearing and untested. It is now
tested, and it fails.

| potential | self-play R² | AUC | **ladder R²** | **AUC** | AUC drop |
| --- | --- | --- | --- | --- | --- |
| **fitted on self-play** | +0.430 | 0.796 | **−0.398** | **0.593** | **−0.202** |
| `LeadPotential` (zero-fit) | +0.233 | 0.770 | +0.101 | **0.716** | −0.054 |
| cash only | +0.242 | 0.737 | +0.094 | 0.687 | −0.050 |
| fitted on the ladder (ceiling) | — | — | +0.188 | 0.722 | — |

Three readings, in order of how much they change the plan.

**The fitted Φ is worse than useless out of distribution.** R² goes +0.430 →
**−0.398** — worse than predicting the mean — and AUC 0.796 → 0.593. It lands
**below both baselines**, including the two-line hand-written rule it was
supposed to beat. Per-day it is at chance through the whole contest phase:
AUC 0.552 at day 7, **0.531 at day 12**, 0.536 at day 17.

**The zero-fit rule transfers; the fit does not.** `LeadPotential` loses
**−0.054** AUC crossing distributions against the fit's **−0.202** — a 3.7×
difference. This is `docs/observational-vs-interventional` again, and it is the
sharpest instance in the project so far, because both models were scored on the
same rows by the same estimator.

**Fitting on the right distribution buys almost nothing.** A Φ fitted on the
ladder itself reaches AUC 0.722 against the zero-fit rule's **0.716** — +0.006,
and +0.087 R². So essentially the *entire* apparent value of the fitted Φ was
in-distribution overfitting to our own self-play. The two published numbers —
+0.381 R² and +0.150 paired over cash — describe how well Φ reads *champ-h8
playing champ-h8*, and nothing else.

**What this costs and what it does not.** It costs nothing in safety: PBRS is
policy-invariant for any Φ, so none of `tools/shaping.py`, the corrected trainer
or the 28 tests is affected. It costs the *usefulness* claim entirely. Any
closed-loop design that was going to read `rl/datasets/potential.json` must read
the zero-fit lead instead, which is within 0.006 AUC of the best thing available
and carries no distribution risk at all.

> The rule, stated once: **a potential fitted on self-play describes self-play.**
> Fit on replayed ladder sessions or do not fit.

---

## 2 When the field changes phase — and how late we are

`docs/game_phases.md` sets the boundaries by calendar (0–7, 7–17, 18–25, 26–29)
from where *our* P(win) curve bends. `tools/phase_timing.py` instead reads four
landmarks off each seat's own trajectory, so a player who runs the season faster
measures as faster:

- `build_done` — first day `E:occ_all` reaches 90% of that seat's season max
- `peak_board` — the day `M:board_value` peaks
- `peak_stock` — the day `E:shed_frac` peaks; after it the shed only drains
- `first_sell` — first stock drawdown while the board is still within 10% of peak

| landmark | ladder (all, n=1028) | 2500+ band | M & M & P & Q | **champ-h8** |
| --- | --- | --- | --- | --- |
| `build_done` | 2.0 [1–3] | 1.0 | 1.0 (29th pct) | **10.0** [10–12] |
| `peak_board` | 9.0 [9–12] | 9.0 | 9.0 (43rd pct) | **21.0** [20–22] |
| `peak_stock` | 21.0 [13–25] | 19.5 | 18.0 (45th pct) | 23.0 |
| `first_sell` | 12.0 [9–14] | 9.0 | **16.0 (90th pct)** | 22.0 |

champ-h8's figures are the **unperturbed** self-play panel; the randomised-opening
panel gives 12/21/23/20, so this is not an artifact of the perturbation.

**We finish placing capital on day 10. The field finishes on day 2, and
M & M & P & Q on day 1.** Eight to nine days of the season are spent with owned
ground still empty — and it is owned ground, `E:occ_all` is occupancy of tiles we
already have, not a land purchase. That is consistent with being labour-short
(hands/work 0.47–0.58 at days 11–15) and it is *not* an argument for buying the
4th quadrant, which gated −$6,266 twice.

**And we peak the board on day 21 against a field median of 9** — on the one
landmark that correlates with rating at all:

| landmark | Spearman vs rating | vs margin | vs win |
| --- | --- | --- | --- |
| `build_done` | −0.019 | −0.057 | −0.014 |
| **`peak_board`** | **−0.361** | +0.031 | +0.079 |
| `peak_stock` | −0.042 | +0.036 | +0.049 |
| `first_sell` | −0.085 | +0.058 | **+0.107** |

Stronger players peak their board **earlier**. We peak ours twelve days later
than the field, and the whole gap sits inside phase B — days 7–17 — which
`docs/game_phases.md` already identified as the only phase where an intervention
can still change a result for us, and where our P(win) falls 0.511 → 0.084.

### What M & M & P & Q do differently, specifically

Among 2500+ seats they are unremarkable on three landmarks and extreme on one:
they build fastest (day 1), peak the board with everyone else (day 9), and then
**hold. `first_sell` day 16 against a 2500+ median of 9 — the 90th percentile.**
Their distinguishing trait at the top of the ladder is not how they build, it is
how long they sit on stock before the first drawdown. That is the same direction
as `first_sell`'s +0.107 correlation with winning, and the same direction as
melon's median sell day correlating +0.136 with winning while `x-mel100`
(harvest at first yield) gated **−$1,482**.

They are, in other words, a third independent confirmation of the one market
timing rule this project has established: **sell later than feels right.**

---

## 3 Correction to §2: `build_done` was normalised, and overstated the gap

`build_done` is "first day occupancy reaches **90% of that seat's own season
max**", which is not comparable between seats whose maxima differ. champ-h8
tops out at 0.99 so it must pass 0.89; the ladder tops out at 0.96 so it only
needs 0.86. That normalisation alone is most of the "day 10 vs day 2".

Against a **fixed** bar (`build_abs`, occupancy ≥ 0.85) and on the raw curves:

| landmark | ladder | 2500+ | M & M & P & Q | champ-h8 | Spearman vs rating |
| --- | --- | --- | --- | --- | --- |
| `build_abs` (fixed bar) | 1.0 | 1.0 | 1.0 (40th pct) | **8.0** | −0.111 |
| `build_done` (normalised) | 2.0 | 1.0 | 1.0 | 10.0 | −0.019 |
| **`peak_board`** | 9.0 | 9.0 | 9.0 (43rd pct) | **21.0** | **−0.361** |
| `peak_stock` | 21.0 | 19.5 | 18.0 (45th pct) | 23.0 | −0.042 |
| `first_sell` | 12.0 | 9.0 | **16.0 (90th pct)** | 20.0 | −0.085 |

Median occupancy of owned ground, day by day:

```
day        0     1     2     3     5     7     9    12    20
ladder   0.48  0.92  0.92  0.96  0.88  0.78  0.96  0.93  0.96
MMPQ     0.36  0.96  1.00  1.00  1.00  0.88  0.87  0.96  0.98
champ-h8 0.28  0.68  0.80  0.80  0.76  0.72  0.92  0.97  0.99
```

So the honest statement is **not** "eight days of empty ground". It is: we are
0.12–0.24 of occupancy behind on days 1–5 and level by day 9; invested capital
tracks the ladder within 0.06 the whole way. The real outlier is `peak_board`
— day 21 against a field median of 9 — and that landmark is an argmax of the
raw curve, carries no normalisation, and is the only one that correlates with
rating (−0.361 against −0.02 to −0.11 for the rest). **That is the finding.**

## 4 The closed-loop phase-B schedule: built, gated, REFUTED

`tools/make_clsched_variant.py`. The planner's hiring is an open-loop calendar
with a ceiling — `min(CHAMP_HIRES[day] + delta, workload // 3)` — where
`workload` counts work that already exists, and `empty_tiles` is computed three
lines above and never read. The arm adds, in days 7–17 only, hands for work
that is *arriving*: crops within 2 days of `max_yield_day` plus, at half
weight, ground still to fill. Arithmetic on quantities the planner already
has, no fitted model anywhere in it, one scalar β.

β = 0 verified **bit-identical** to the base (+$0, 0 discordant worlds over
144), so the patch is a clean pure addition and every effect below is the
treatment.

| arm | β | money vs `x-fut10` | z | win | McNemar |
| --- | --- | --- | --- | --- | --- |
| base `x-fut10` | — | — | — | 0.367 | — |
| `x-clb25` | 0.25 | **−$10,665 ± 501** | −21.30 | **0.269** | 36 b-only / 2 c-only |
| `x-clb50` | 0.50 | **−$11,029 ± 497** | −22.21 | 0.263 | 38 / 2 |
| `x-clb100` | 1.00 | **−$11,227 ± 494** | −22.71 | 0.260 | 39 / 2 |

346 held-out ladder worlds. Monotone in β, losing on wins as hard as on money
(0.367 → 0.26), and catastrophic at the smallest β tested. Refuted.

**Why.** `budget = max(150, money * 0.25)` lets hiring spend a quarter of cash
*every day*, and the `min(..., workload // 3)` cap only binds when the calendar
is the smaller term — so raising the calendar entry removes the only thing
holding hiring down, on every day of phase B. The constant version of this
lever was already refuted monotonically (+2 hands −$5,229, +4 −$19,344); β=0.25
adds on the order of ten.

**What it refutes and what it does not.** It refutes the *inference* — that
hiring is the lever behind the timing gap. It does not touch the
*observation*: `peak_board` day 21 vs 9 stands, and the day-1 occupancy gap
stands. Two candidate explanations were checked and one is dead:

- **the 10-order market cap is NOT the throttle.** Instrumented over 6 games:
  the planner issues 0.46–1.14 market orders per turn on days 0–9 and touches
  the `MAX_MARKET_ORDERS = 10` ceiling in **1 turn out of 144** on days 7–9.
  There is no cap to relieve.
- **so the early fill rate is limited by the buying decision itself**, not by
  order bandwidth and not by hands. At ~0.5 orders/turn on days 0–5 we are
  simply not ordering. That is the next thing to look at, and it is a
  measurement, not a variant.
