# Model architecture, revised by measurement

2026-09-18. `docs/spatial_program.md` proposed a board encoder and a
metric predictor. Four measurements since then change it — three of them
overturn something I had asserted.

---

## 1. The output must be a VALUE, and the value must be time-indexed

The largest effect found in this project:

| phase | realised $ per action | n |
| --- | --- | --- |
| opening 0–6 | **$1,317** | 336 |
| contest 7–17 | $672 | 528 |
| consolidate 18–25 | $236 | 384 |
| endgame 26–29 | $103 | 192 |

13× decay, **z = 3.67**, monotone, replicated across 6 worlds.

**Consequence:** a model that scores actions without conditioning on the day is
wrong by up to 13× before it starts. Day is not a feature among features here —
it is the dominant term. Any head that outputs a task score must take the day
as a first-class input, not a normalised scalar buried among 24 others.

## 2. Two different values exist and the architecture must not conflate them

| quantity | measured | how |
| --- | --- | --- |
| value of a **taken** action | **$630** (season mean) | deletion counterfactual |
| value of an **added** action | **$1.75** | j-tile added 266 actions for +$465 |
| ratio | **360×** | |

Removing a needed action costs a season of compounding. Adding a twelfth hand
creates filler, because useful work is already saturated. I conflated these and
proposed front-loading hiring on the strength of the deletion number; it gated
null and *reversed* with dose (+$1,281 / −$653 / −$1,370 at 7/9/11 hands).

**Consequence:** the head should predict the **marginal** value of a candidate
task *in a schedule that is already full* — i.e. what it displaces — not the
average value of work. Training it on deletion labels alone would reproduce my
error at scale.

## 3. The LP cannot be the teacher

Validated directly: LP labour dual vs realised action value, **r = −0.34**, and
23× off in magnitude. The LP prices labour where the constraint binds (late,
$60–$101); realised value is dominated by compounding (early, $1,317). It is
**inverted**, not merely imprecise.

**Consequence:** drop the LP-dual supervision head from the design. The LP stays
useful as an offline bound and for its *coupling* structure, but it must not
supply labels. This kills the "distil the LP, it's free" branch I recommended —
free was the only thing it had going for it.

## 4. Labels must be counterfactual, and contrast must be engineered

Two findings force this.

**Fitted labels have failed five times** — BC 69.4% accuracy → null gate; the
value head 4.8× tighter → wrong sign on 3 of 4 arms; the market table → three
rejections; the legality mask → +0.0006; raw BC 75.5% → $35. Observational fit
does not transfer to interventional effect.

**And contrast has to be built into the training set.** With 12 champ-h8
variants, agent explains **6.9%** of outcome variance and no metric correlates
with the outcome above |r| = 0.21. Add tape agents and it is **66.9%**, with
metrics reaching **+0.78**. A dataset of only good players teaches nothing
about what makes a player good.

**Consequence:** train on deterministic counterfactuals (1.09 s each, exact, no
evaluation noise) drawn from agents **spanning the rating range**, not from the
3029 cluster alone.

---

## The revised architecture

```
INPUT
  board tensor      (24, 10, 10)   tools/board_tensor.py -- built, works
  k turns of history                path-dependence: consecutive_unwatered is
                                    the engine keeping memory; the market
                                    ladder is cumulative; the melon race is a
                                    rate
  scalars                           day FIRST-CLASS (finding 1), cash, shops,
                                    per-product price ratio and inventory

TRUNK
  small conv stack                  100 cells with strong locality is what
                                    convolution is for; a transformer over 100
                                    tokens buys attention we have no compute
                                    to spend

HEADS
  1. per-tile risk map              "will this crop die" -- BUILT, PR-AUC 0.528
                                    against a 1.6% base rate, 33.8x lift
  2. marginal task value            what does this task displace, given the day
                                    -- trained on DELETION counterfactuals,
                                    read as marginal not average (finding 2)

CONSUMER
  the existing scheduler            replaces six hand-set ordinal bands
                                    (P_SURVIVAL 0 ... P_FILLER 5) with prices.
                                    The bands cannot express "this task is
                                    worth 13x more today than it will be on
                                    day 27", which is the largest effect we
                                    have measured.
```

**Why a predictor feeding the scheduler, and not a policy.** Raw-action BC
reached 75.5% per-action accuracy and scored $35: 719 steps of closed-loop
control compound, and DAgger's fix needs a queryable expert while ours are
recordings. A predictor has one input and one output — no accumulation. The
scheduler stays correct by construction and only its *ordering* changes.

## What has to be true for this to pay, and how we would know early

Every component above is measurable before the whole is built:

1. **Does the marginal-value head beat the six bands at ranking?** Test on
   deletion counterfactuals held out by world. Free once the labels exist.
2. **Does re-ordering by predicted value change the outcome?** Three watering
   triage designs said no (−$1,028, −$534, and one that raised total deaths).
   That is the strongest evidence *against* this whole direction and it should
   be confronted first, not last.
3. **Gate paired on the ladder**, and draw search and gate worlds from ONE
   pool — the same agent scored −$14,183 and −$3,094 on two differently-seeded
   held-out pools (3.9 se apart), which is what invalidated the Kaggle bake-off.

## The honest prior

Point 2 is the problem. The architecture's whole thesis is that better task
*ordering* pays, and three direct tests of better task ordering came back
negative. The action-value gradient says *timing* matters 13×, which is a
different claim — and the reallocation test running now is its first real
examination. If reallocation also fails, the case for this architecture
weakens considerably, because its output would be a number nothing can act on.
