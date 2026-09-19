# The metric hierarchy: what to measure at each level, and what good and bad look like

2026-09-19. The aim, stated by the user and adopted as the organising frame:

> improve every sub-section metric of the farm — **win-rate** > **profit-margin
> difference** > better (**sell timing, ordering, investment, risk management**:
> buying land / livestock / crops, demand-supply ratio) > better (**farming and
> routing**). Find a relationship that moves win-rate and minimises the profit
> margin when we lose. Each sub-section needs concrete metrics indicating both
> bad and good, so we can discover better metrics that are transferable.

The discipline that makes this work, learned the hard way this week: **a metric
is only worth optimising if a ceiling measurement says the thing it measures
carries money, and if moving it has been shown to move the level above.**
Several metrics below are already known to be decoupled, and those are marked.

---

## L0 — WIN RATE (the only thing the leaderboard scores)

Rating is win-based. Everything below is instrumental.

| | value |
| --- | --- |
| **metric** | paired win rate vs a base on identical worlds, McNemar on discordant worlds only |
| **good** | z > 2 over ≥1,000 worlds |
| **bad** | money z > 4 with win z < 1 — the signature failure of this project |
| **instrument** | `tools/ladder_gate.py`, three independent seed blocks |
| **known** | `r-net2k` z = 3.44 / 1,131 worlds is the only arm to clear it |

**Trap, measured:** money and wins decouple. `r-mrg100` was +$2,605 at z = 5.2
on money and z = 1.15 on wins across one pool, z = 2.17 across another. Win
rate needs ~1,000 worlds before it means anything.

## L1 — MARGIN AGAINST THE OPPONENT, not absolute money

A win is `our_money > their_money`. Optimising *our* money can leave the margin
untouched if it also helps them — and the rent rule plausibly does exactly
that, since planting into uncrowded markets stops us crashing prices they also
sell into.

| | value |
| --- | --- |
| **metric** | mean paired `(us − them)` per world, and its sign distribution |
| **good** | margin rises while our absolute money is flat — that is pure denial |
| **bad** | absolute money rises with margin flat — we lifted both boats |
| **instrument** | `tools/round_robin.py` (head-to-head), `tools/opp_coupling.py` |
| **known** | the opponent costs us **$52,652/game** through `market["inventory"]`, the ONLY shared state |

**Trap, measured:** denial is symmetric. Both seats reach a price trough
holding the same stock (strawberry 23.0 vs 21.5), so suppressing a recovery
costs us what it costs them: **−$46/game, 0/120 games flipped**. Denial needs
*manufactured* asymmetry — deliberately hold less of what you intend to crash.
Untested.

## L2 — SELL TIMING

| | value |
| --- | --- |
| **metric** | realised $/unit ÷ base price, per product |
| **good** | ratio > 1 on uncrashable products; selling before the glut day on crashable ones |
| **bad** | realised ratio below the recorded band for that product |
| **ceiling** | the floor binds on **2.5%** of sizing calls; milk + wool withhold **47 units a season** |
| **verdict** | **CLOSED, and not for the reason first given.** The JOINT test (capacity-rationed floors) settles it |

**The joint test, 2026-09-19.** The first closure was marginal ("the floor
binds on 2.5% of calls"), which is exactly the one-factor-at-a-time reasoning
the LP experiment was built to avoid. `tools/make_capfloor_variant.py` moves
the floor with remaining absorption capacity, its regeneration rate, our stock
and their visible stock TOGETHER:

    r-mrg100   +$3,079  z 2.93  win 0.337  discordant 18/23
    c-cap50    +$3,147  z 2.99  win 0.337  discordant 18/23
    c-cap100   +$3,103  z 2.96  win 0.337  discordant 18/23

**Identical discordant worlds across all three arms.** The floor changes on
100% of call sites with mean |delta| 57 and not one world's outcome flips.

**The floor is not a lever at all.** `units_sellable_above(item, inv, floor,
cap=stock)` is bounded by STOCK, and we never hold enough of anything for
`floor` to bind. Every floor experiment ever run here -- `premium_scale` over
4,270 samples, opponent-stock floors, the glut gate, this -- tuned a parameter
outside the binding constraint. **Sell timing is controlled by PRODUCTION
(L4/L5), not by price thresholds.**

### Absorption capacity per product (tools/market_capacity.py, 40 games)

The quantity every sell rule was missing. Capacity = units sellable above 60%
of base; season budget = capacity at day 0 plus 30 days of regeneration.

| product | cap d0 | cap d20 | regen/day | exhausts | season utilisation |
| --- | --- | --- | --- | --- | --- |
| FERTILIZER | 203 | 0 | 0.0 | d14.9 | **209% — oversold 2x** |
| MELON | 101 | 1 | 1.0 | d16.1 | **116% — oversold** |
| WOOL | 38 | 19 | 12.7 | d17.7 | 68% contested |
| MILK | 31 | 6 | 20.2 | d16.6 | 57% contested |
| STRAWBERRY | 26 | 71 | 26.1 | d24.8 | 50% |
| CARROT | 158 | **279** | 17.2 | never | 33% — capacity GROWS |
| TOMATO | 93 | **196** | 12.6 | never | 22% — capacity GROWS |
| WHEAT | 5000 | 5000 | 30.7 | never | 8% |
| EGG | 5000 | 5000 | 13.2 | never | 3% |

Glut day per product is the one live sub-metric: **fertilizer d2, wool d7,
melon/milk d11, strawberry d23, and wheat/carrot/tomato/egg never**. Predictable
(melon R² 0.986) but the boolean gate on it cost **−$2,590**.

## L3 — ORDERING (market-order slot allocation)

| | value |
| --- | --- |
| **metric** | orders suppressed per game by the 10-slot cap, and their value |
| **good** | zero suppression, or suppression only of low-value orders |
| **bad** | SELL suppressed while HIRE fills slots |
| **measured** | **21 saturated turns/game, 46.8 SELLs suppressed, $29,009 at then-current prices**; HIRE issues 268 orders vs SELL's 171 |
| **caveat** | end-of-bell stranding is only **$55/seat**, so this is a DELAY not a loss; the real cost is price drift over the delay |
| **status** | **CLOSED, 2026-09-19.** Reserving SELL slots only on saturated turns gates below baseline at every dose, monotone: o-res1 −$1,108, o-res2 −$377, o-res3 −$6,914, o-res4 −$15,822 against r-mrg100 |

The mechanism worked — `o-res2` cut suppression 87% ($31,104 → $3,936) with HIRE
unchanged at 269 — and recovering the sells made the agent WORSE. That confirms
the caveat: $29,009 bounded a DELAY, not a loss. Everything sells eventually
($55/seat stranded), so recovery buys one turn of price drift and costs the
deferred hires. **Hiring beats selling for a market slot at every measurable
dose**, reproducing m-hire5/m-hire3 (−$7,234/−$39,678) at smaller doses.

## L4 — INVESTMENT (land, livestock, crops)

| | value |
| --- | --- |
| **metric** | $/tile-day realised per crop, and herd size vs shop-draw demand |
| **good** | volume moves with the draw — top players swing wool 131→534 units as drain goes 1→25/day |
| **bad** | constant output regardless of demand — **ours is flat at 236 strawberries across a sixfold demand range** |
| **ceiling** | hiring is nearly free: 11 hands = $232/day, a season ~$6,960 against a $95k bank |
| **verdict** | single-knob moves all refuted; the whole-vector LP move gated **−$31,728, z −22.5**; forward-model hiring **null** |

## L5 — RISK: the demand/supply ratio

| | value |
| --- | --- |
| **metric** | projected inventory vs I0 at each product's harvest day |
| **good** | planting only into products whose price at *their* harvest day is above base |
| **bad** | producing into a pool already past I0 |
| **exogenous** | the shop draw: 8 uniform draws every 3 days. **P(no wool shop) = 34.4%** |
| **known** | the market is an EXACT identity — 36/36 products, zero error. Nothing to learn, only to compute |
| **verdict** | this is what the rent rule consumes, and it is the one thing that paid |

## L6 — FARMING AND ROUTING (the executor)

| | value |
| --- | --- |
| **metric** | moves per work action; pickup per feed; work actions per seat |
| **good** | moves/work → 0.91 (the 3000-band); pickup/feed → 0.52–0.63 |
| **bad** | ours: **1.43 moves/work, 1.07 pickup/feed** — worse than the LOWEST rated band |
| **ceiling** | of 3,965 moves: **2,454 irreducible, 211 placement (5.3%), 1,300 routing (33%)** |
| **conversion** | freed turns are worth **ZERO OR LESS** — h-front9 added ~1,440 actions for −$653 |
| **verdict** | **BOUNDED AT ~$0.** The haul fix gated −$12,026. The geometry is downstream of the planner |

---

## The transferable lessons, which is what the frame is for

1. **Measure the ceiling before building the model.** The opponent-shed oracle
   (1.5%) and the placement ceiling (211 moves) each saved a week of work.
2. **Check the mechanism fires before reading the gate.** Archive attempt #11
   is "mechanism does not fire" — a null that measured nothing.
   `tools/haul_probe.py` and the `alpha=0 bit-identical` convention exist for
   this.
3. **Matching a metric the corpus exhibits ≠ playing like the corpus.** The
   haul floor moved pickup/feed 1.08 → 0.88 and lost $12,026, because
   `moves/work` went the other way.
4. **Print per-target metrics before accepting a kill.** S2's first run
   reported a KILL that was a loss-imbalance artefact visible only per-kind.
5. **Train on the deployment distribution.** A net trained on the top 21% of
   the corpus was −13% out-of-distribution; retrained on the natural mix it
   was +27%.
6. **A false positive regresses with more data; a real effect strengthens.**
   `r-net2k`'s win z went 0.31 → 2.90 → 2.57, pooling to 3.44.

---

## Appendix: the three-block verdict on every surviving arm (2026-09-19)

Same three seed blocks (45/45/110, seeds 0/1/5 = 1,131 worlds), so these are
directly comparable:

| agent | money | se | z | win z | what it adds |
| --- | --- | --- | --- | --- | --- |
| `r-mrg100` | +$2,978 | 558 | 5.34 | **2.17** | nothing learned — pure arithmetic |
| `k-comp5` | **+$3,389** | 515 | **6.59** | 2.06 | closed-form risk penalty |
| `r-net2k` | +$2,731 | 538 | 5.08 | **3.44** | 327k-param BoardNet |

**None separates from the others.** `k-comp5 − r-mrg100 = +$411 ± 759`
(z = 0.54), positive in 3/3 blocks (+351, +726, +268) — suggestive of a small
real effect, not established.

**A methodological error worth keeping.** I pre-registered "anything under
+$300 over r-mrg100 is a null", which is a POINT-ESTIMATE bar. +$411 clears it
at z = 0.54, i.e. the bar would have licensed declaring a win on noise.
Pre-register a **significance** criterion, not a magnitude.

### The pattern across every learned component tried

| component | params | verdict |
| --- | --- | --- |
| BoardNet opponent supply | 327,125 | ties the arithmetic (−$276) |
| learnt risk variance | ~80 | loses to computed by ~$950 |
| exec_bc ranker + Hungarian | 25 | null (+$366) |
| pointer network | 52,289 | **−$33,414, z −25.4** |
| computed risk | **0** | best money arm |
| forward market model | **0** | the only thing that ever paid |

Every time the engine's closed form has competed with a fitted model on the
same consumer, **the closed form has won**. The engine is fully observable and
deterministic, so a fitted model can only approximate what is exactly
calculable — and pays a generalisation penalty for the privilege.
