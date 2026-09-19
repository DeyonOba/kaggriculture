# The model architecture, built and tested

2026-09-18. Supersedes the plan in `docs/architecture_v2.md`, which proposed a
learned board encoder feeding a learned value head feeding the scheduler.
Three measurements made that design the wrong one, and a fourth built the
right one.

---

## 1. What changed the design

**The market does not need to be learned.** `tools/market_clock.py` verified
that market inventory is an exact accounting identity — 36/36 products, zero
error, on 24 replayed sessions:

```
inv_p(t+1) = inv_p(t) + sold - bought - drain_p(shops)
price_p    = market_price(inv_p)                        closed form
drain_p    = 6 * SUM(shop instances holding p; x2 if single-product) + 1 town
```

The only exogenous economic variable in the whole game is the shop draw:
eight uniform draws with replacement, one every three days from day 3.
Everything else is arithmetic on observables.

**The opponent's farm is public and their inventory is not worth inferring.**
`observation.farms` is shared (`kaggriculture.py:271`), so both boards are
visible every turn. Only `private = {inventories, seeds, shed}` is hidden.
The ablation in `tools/price_forecast.py` prices each:

| | k=1 | k=3 | k=5 | k=10 |
| --- | --- | --- | --- | --- |
| L0 persistence | 5.2 | 14.1 | 23.1 | 42.4 |
| L1 + known shops | 4.9 | 15.1 | 25.5 | 50.5 |
| L2 + expected future draws | 5.2 | 16.4 | 28.2 | 56.9 |
| L3 + our board | 4.4 | 12.0 | 20.8 | 44.2 |
| **L4 + THEIR board** | **4.3** | **8.0** | **13.3** | **29.4** |
| L5 + our shed | 4.3 | 7.7 | 13.0 | 29.0 |
| L6 + their shed (**ORACLE**) | 4.3 | 7.8 | **12.8** | 28.3 |

Their **board** cuts k=5 error by 36%. Their **shed**, known perfectly, cuts
it by 1.5%. **This retires the opponent-belief / latent-inference branch** —
there is no hidden state worth modelling. Model their farm, not their
inventory.

**Demand without supply is worse than nothing.** L1 and L2 — the drain model
— are *worse than assuming the price never changes*. And the shipped bundle's
`econ` module contains exactly that half: `town_drain_per_day`,
`shop_weight_per_day`, `remaining_unlocks`, `forecast_drain`, and no supply
term at all. The agent has been carrying the harmful half of a market model.

---

## 2. The architecture

```
OBSERVE            both farms' tiles (public), market inventory,
                   unlocked shops, our shed
                                |
FORWARD MARKET     inv_p(t+k) = inv + our board supply + their board supply
MODEL                           - drain(shops)
(arithmetic,       price_p(t+k) = market_price(inv_p(t+k))
no training)                    |
                                v
DECISION           rent(crop) = sell_revenue(crop, inv_at_harvest, block)
                                / (tiles * tile_days)
                                |
CONSUMER           the existing planner's freed-tile choice
```

**No neural network, but not "no learning".** Precisely what is and is not
fitted in the shipped bundle:

| component | fitted? | how |
| --- | --- | --- |
| `WEIGHTS_DATA` (neural planner) | **None** — dead path | `nnpolicy` never runs |
| `MPC_ENABLED` | `False` | engine rollout, off |
| forward market model | **no** | engine tables + exact `market_price` |
| `_rent_pick` | **no** | argmax over 5 crops |
| **`BANDIT_RULES`** | **YES — 4 fitted conditions** | paired knob bandit, 30,272 samples |

`BANDIT_RULES` carries `premium_scale 0.35`, `carrot_day 25`,
`cows 6 when no shop consumes MILK`, and `sheep 10 at day 12 when WOOL holds
base`. Those are learned parameters — scalars with conditions rather than a
network, fitted by a bandit on paired counterfactuals instead of by gradient
descent. So the accurate statement is: **no gradient-trained component and no
learned board representation in the decision path; four fitted scalars are.**

They are also easy to miss: `rl/export.py` writes `BANDIT_RULES` into the
frozen bundle's `weights` module, and it is invisible from `kagent/` source.
Re-implementing two of them as `h-cows`/`h-sheep` produced exactly $0 on 270
worlds with zero discordant worlds, because the agent already had them.

The other learned artefact — the per-tile crop-death CNN (`tools/board_cnn.py`,
PR-AUC 0.528 against a 1.6% base rate, 33.8× lift) — is built and works but is
**not in this path** and has no consumer.

**No learned board state, and none is needed.** The forward model reads both
boards fresh every turn and carries nothing across turns. The engine's own
memory — `consecutive_unwatered`, cumulative market inventory, `planted_day` —
is all *in the observation*, so the state is Markov as presented and recurrence
would have nothing to remember.

### The three engine rules that made the forecast work

Each was found from the per-product bias of the previous version, not guessed:

| rule | evidence it was missing | effect |
| --- | --- | --- |
| `CARE` doubles animal output on a fed day | milk/wool price bias **+$125 / +$88** at k=10 | supply ×2 |
| animals make one FERTILIZER each per night | fertilizer error *identical* to persistence, bias +$31 | new supply term |
| `FEED` eats one WHEAT per animal per day | wheat bias **−$7.4**, over-predicting supply | 540-unit season sink |

Together these took the model from 20% better than persistence to **43%**.
Wheat MAE alone went 7.4 → 1.9.

### The economics correction that matters most

The first decision rule priced a crop at its **marginal** price and picked
tomato every single day — tomato has the highest $/tile-day at our current
(zero) volume, $203, and collapses past T=200 at scale. Gated smoke:
**−$9,305**.

Pricing the **whole block against the inventory its own units create**
(`econ.sell_revenue(crop, inv_at_harvest, block)`) fixed it: **−$464** at the
same dose. That is the single largest correction in the build, and it is the
same non-linearity that makes per-action values non-additive
(`docs/architecture_v2.md` §5).

---

## 3. What the model says

With self-impact priced, the rule picks for a freed tile:

```
WHEAT 66    MELON 5    CARROT 2    STRAWBERRY 2      (75 decisions, 4 games)
```

88% wheat — which is what champ-h8's hand-written ladder already does. Built
from engine rules and observables with no reference to the existing
heuristic, the model **independently rediscovers it**, and it agrees with the
LP on early wheat while disagreeing everywhere the LP lost
(`docs/lp_allocation_test.md`, −$31,728 at z = −22.5).

From that pick distribution I predicted the rule could not pay: it agrees with
the incumbent 88% of the time, so the 12% of disagreements looked like the
only place left to lose money. **That was wrong** — the disagreements are
where it wins.

## 4. The gate — positive, dose-monotone, and replicated

270 held-out paired ladder worlds per block, two independent seed blocks.

**Block 1** (base champ-h8 −$3,094/game, win 0.319):

| candidate | money | se | z | win |
| --- | --- | --- | --- | --- |
| r-mrg25 | +$1,707 | 880 | 1.94 | 0.315 |
| r-mrg50 | +$2,050 | 1,028 | 1.99 | 0.307 |
| r-mrg100 | **+$3,079** | 1,049 | **2.93** | 0.337 |

**Block 2, fresh seeds** (base −$3,948/game, win 0.296):

| candidate | money | se | z | win | McNemar p |
| --- | --- | --- | --- | --- | --- |
| r-mrg50 | +$3,667 | 1,092 | 3.36 | 0.344 | 0.079 |
| r-mrg100 | +$3,291 | 1,121 | 2.93 | 0.326 | 0.23 |

Pooled, `r-mrg100` is **+$3,185 ± 768, z ≈ 4.1**, positive in 4 of 4
arm-blocks and monotone in dose on block 1. The winner's curse at N=3 arms is
~1.48 × se ≈ $1,550; the estimate survives it.

**The honest caveat, and it matters.** The win rate moves the right way but
does not clear significance: pooled discordant worlds are 31 base-only against
44 candidate-only for `r-mrg100`, z = 1.50, p ≈ 0.13. Rating is win-based
(`kaggriculture-win-rate-not-margin`), a win-rate gate needs ~1,000 games per
arm rather than 540, and this project's local gates have disagreed with the
leaderboard twice, both times against the locally-better agent. So this is a
**replicated money gain with a favourable but unproven win-rate effect**, not
a demonstrated rating gain.

It also does not change the submission: `tape-sel` ships at ~1694 and
champ-h8 rates ~810, so +$3.2k on the policy lineage does not close that. The
tape lineage replays recorded actions and never calls this planner.

## 5. Status of each component

| component | built | works as a predictor | improves the agent |
| --- | --- | --- | --- |
| forward market model | yes | **yes, 45% over persistence** | — |
| opponent board supply | yes | **yes, 36% of the gain** | — |
| opponent shed belief | oracle tested | **no, 1.5%** | retired |
| crop-death CNN | yes (v2) | yes, 33.8× lift | no consumer |
| rent-priced freed tile | yes | n/a | **+$3,185 ± 768, replicated** |

This is the first replicated positive gate in the policy lineage since the
hiring rollout (+$3,287 ± 1,528). Both came from the same place: a per-day
*schedule* or *state-dependent rule* where a constant used to sit.

## 5. Where the remaining bias points

The forecast still under-predicts supply of MILK (+$14), WOOL (+$13) and
STRAWBERRY (+$7) at k=3, because it holds the herd and crop mix fixed while
both farms build out. Closing that needs a model of the opponent's *future
purchases and plantings* — a policy model, not an accounting identity. It is
the only part of the market that is not already solved, and it is the part
that would let the forecast see a rival's ramp before it lands.

---

## 6. The Kaggle kernel run (2026-09-19)

`gideonoba/kaggriculture-train` completed. Two of my errors and one real result.

**The daily datasets never mounted.** All three
`kaggriculture-episodes-2026-09-*` were in `kernel-metadata-train.json` and are
absent from `/kaggle/input`; Kaggle dropped 64 GB of inputs silently, almost
certainly a per-kernel input-size limit. `kaggle_train.py`'s guard caught it
(`no daily episode datasets mounted; skipping harvest`) rather than training on
nothing, so the failure was loud — but no daily data was used.

**`arena/agents/` ships empty.** The log shows `arena/agents: [] []`. Agent
bundles are gitignored, so `push.sh`'s `git ls-files` tarball carries none of
them, and `net_ood.py` died on `r-mrg100/main.py`. This is already recorded in
memory (`pub-*` agents are gitignored; external-opponent work cannot run on
Kaggle) and I pushed anyway. **No kernel step that loads an agent bundle can
work until the bundles are shipped some other way.**

**The result: 2,000 games fixed the out-of-distribution failure.** Trained on
40,000 samples at `--min-rating 0`, against 500 games before. Every product the
learning curve said was still climbing improved; the saturated ones did not:

| | 500g → 2,000g | | 500g → 2,000g |
| --- | --- | --- | --- |
| WOOL | 0.730 → **0.802** | STRAWBERRY | 0.948 → 0.950 |
| CARROT | 0.632 → **0.735** | MELON | 0.868 → 0.853 |
| EGG | 0.829 → **0.907** | | |
| WHEAT | 0.828 → **0.869** | | |

And it is the first model positive in **all three** opponent regimes:

| regime | 2800+/400g | full/500g | full/2,000g |
| --- | --- | --- | --- |
| replay (2800+) | +71% | +49% | **+85%** |
| vs champ-h8 | −13% | +25% | **+27%** |
| vs r-mrg100 | −4% | −26% | **+17%** |

**Gate**, pooled over two 270-world blocks:

| arm | money | se | z | discordant b/c | win z |
| --- | --- | --- | --- | --- | --- |
| r-mrg100 | +$3,178 | 766 | 4.15 | 31/44 | 1.50 |
| **r-net2k** | +$2,808 | 743 | 3.78 | 32/53 | **2.28** |

Money remains marginally below `r-mrg100` and is block-inconsistent (+$1,883
then +$3,694). The win-rate z = 2.28 is the first thing this week to move the
metric the rating is made of — but it is the **eighth arm gated that day**, and
one z = 2.28 out of eight is what multiplicity produces by chance. A third,
larger, independent block decides it.
