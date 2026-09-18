# The opponent costs $52,652 a game, and we model it for one product out of nine

2026-09-18. Measured with `tools/opp_coupling.py`.

## The only coupling in the engine

The two farms never touch. Separate 10×10 grids, separate units, separate
sheds, separate cash. Grepping every interaction path in
`kaggle_environments/envs/kaggriculture/kaggriculture.py`, exactly one piece of
state is shared: `market["inventory"]`, a nine-key dict both players sell into.

Price is a deterministic closed-form function of it — `market_price(item,
inventory)`, no randomness anywhere in the call. And the curve is violent:

| product | T | above_target | 100 units above I0 |
| --- | --- | --- | --- |
| STRAWBERRY | 100 | 1.6 | $120 → **$1** |
| WOOL | 105 | 3.2 | $200 → $1 |
| MILK | 122 | 1.6 | $160 → ~$6 |
| MELON | 300 | 3.6 | $250 → $130 |
| WHEAT | 400 | 0.2 | $25 → $24 |

So the game is a congestion game wearing a farming skin, and the congestion is
concentrated in four products whose price collapses within ~100 units.

## The measurement

Deterministic counterfactual on 40 recorded sessions where **both** seats were
rated ≥2800. Replay both recorded action scripts, then replay again with one
change: the opponent's `SELL` still pays them and still drains their shed, but
does not add to `market["inventory"]`.

Muting the inventory line rather than deleting their orders keeps their cash,
their shed and therefore every one of their later scripted actions
bit-identical. Only the price channel is cut.

```
baseline final bank            $     103,373
with opponent supply MUTED     $     156,025
gain                           $      52,652  +/- 3,006   z = 17.52
as a share of the bank                 50.9%
```

80 seat-observations. **Half the bank is decided by the other player's
supply.** For scale, every lever this project has measured: the ahmed revenue
gap $39.5k, the LP's perfect-planner headroom $21k, the hiring rollout $3.3k.

## Where it lives

Muting one product at a time, 12 sessions:

| product | gain | se | z |
| --- | --- | --- | --- |
| **STRAWBERRY** | **$19,756** | 1,842 | 10.73 |
| **MILK** | **$19,136** | 2,026 | 9.44 |
| **WOOL** | **$9,062** | 2,684 | 3.38 |
| MELON | $3,427 | 465 | 7.38 |
| TOMATO | $1,875 | 808 | 2.32 |
| FERTILIZER | $1,078 | 4,287 | 0.25 |
| CARROT | $647 | 241 | 2.69 |
| EGG | $351 | 100 | 3.51 |
| WHEAT | −$5,783 | 6,867 | −0.84 |
| **sum** | **$49,550** | | |

The per-product sum ($49,550) lands within 6% of the joint mute ($52,652), so
the channels are **nearly additive** — each product's congestion can be modelled
on its own without a joint model.

Three products carry **97%** of it: strawberry, milk, wool. Exactly the three
steepest curves.

WHEAT is negative and noisy, and the per-seat minimum of the joint mute is
−$56,485, which is the same mechanism running backwards: the opponent's dumping
also *subsidises* us, because `BUY_PRODUCT` quotes off the same crashed
inventory. Fertilizer ends the season 425 units above I0 at $15 against a $100
base. Cheap inputs are the compensation for crashed outputs.

## The gap in our agent

`kagent/planner.py::_sell_floors` sets the minimum acceptable sale price per
item. It reads the opponent's board for **MELON and nothing else**:

```python
opp_ripening = sum(1 for _, _, t in view.opp.plants()
                   if t["crop"] == "MELON" and view.day - t["planted_day"] >= 8)
if opp_ripening >= 3:
    melon_floor = 40
```

STRAWBERRY, MILK and WOOL floors are set from `unlocked_shops` counts alone —
town demand, no opponent term. Those three products carry $47,954 of the
$49,550.

This is not a data problem. **The opponent's board is fully observable.**
`state[i].observation.farms = obs0.farms` (`kaggriculture.py:271`, again at
`:952`) hands every player the complete list of both farms every turn — their
tiles, crops, `planted_day`, animals, `yield_units`, money, hands, unlocked
quadrants. Their future supply of strawberry is a deterministic function of
plants we can already see; their milk and wool are a function of cows and sheep
we can already count.

Only `observation.private = {inventories, seeds, shed}` is per-player. The
hidden state is nine numbers of held goods, not a spatial field — and it is
close to an accounting identity, since harvests are visible on their tiles and
sales are visible in market inventory. `tools/opp_belief.py` already reaches
~5% through day 14 on that route.

## What this explains

- **Why the action value decays 13×** (`docs/architecture_v2.md` §1). Late
  actions are worth $103 partly because by then the market they feed is
  crashed. The decay is not purely compounding; some of it is price.
- **Why moving work late→early lost $15,915.** Late capacity liquidates into an
  already-thin market; cutting it means the season's output never converts.
- **Why the melon race was real** (`kaggriculture-harvest-race`). It was this
  effect, found for the one product we happened to instrument.
- **Why twelve routing experiments and three hiring schedules all gated null.**
  They all moved our own throughput. Throughput feeds a market whose price we
  were not modelling, so extra units arrived to meet a lower price.

## The temporal model (2026-09-18, `tools/market_clock.py`)

The mute above is a TERMINAL counterfactual and says nothing about when. Here
is the clock.

### Inventory is an exact identity, not a model

    inv_p(t) = I0_p + sold_p(t) - bought_p(t) - drained_p(t)

    drained_p per day = 6 * SUM over unlocked shop instances holding p of
                        (2 if the shop sells exactly one product else 1)
                      + 1 if p is in TOWN_CENTER_PRODUCTS

6 because `townShopSellInterval=4` against `turnsPerDay=24`. Checked against
replays of 24 top-rated sessions: **36/36 products exact, max error 0.** There
is nothing to learn here — the market side is arithmetic.

### The only exogenous economic variable is the shop draw

One uniform draw from the 8 shops every `townShopUnlockInterval=3` days, with
replacement, capped at `MAX_SHOP_INSTANCES=8`. Eight coin flips, revealed
between day 3 and day 24, fix the demand curve for the whole season.

| product | E[drain/day] | P(no shop sells it) | sd |
| --- | --- | --- | --- |
| WHEAT | 31.0 | 0.0% | 8.2 |
| STRAWBERRY | 25.0 | 0.4% | 8.5 |
| CARROT | 19.0 | 10.0% | 11.8 |
| MILK | 19.0 | 2.3% | 8.2 |
| TOMATO | 13.0 | 10.0% | 7.3 |
| EGG | 13.0 | 10.0% | 7.3 |
| **WOOL** | 13.0 | **34.4%** | 11.2 |
| MELON | 1.0 | 100% | 0 |
| FERTILIZER | 0.0 | 100% | 0 |

**Wool has a 34% chance of no shop demand at all** — the largest single
exogenous risk in the game. Melon is in no shop (town centre only, 1/day) and
fertilizer has no demand whatsoever.

### The clock, 24 sessions rated 2800+

Mean price by day, and the day inventory first crosses I0:

| product | base | d0 | d6 | d12 | d18 | d24 | d27 | glut day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FERTILIZER | 100 | 100 | 90 | 70 | 45 | 29 | 25 | **2** |
| **WOOL** | 200 | 200 | 218 | 176 | 83 | 57 | 57 | **7** |
| MELON | 250 | 250 | 267 | 153 | 134 | 92 | 94 | 11 |
| **MILK** | 160 | 160 | 191 | 152 | 62 | 54 | 71 | **11** |
| STRAWBERRY | 120 | 120 | 153 | 188 | 187 | 82 | 110 | 23 |
| WHEAT | 25 | 25 | 28 | 39 | 40 | 41 | 41 | never |
| CARROT | 35 | 35 | 36 | 39 | 45 | 51 | 54 | never |
| TOMATO | 60 | 60 | 62 | 64 | 70 | 76 | 78 | never |
| EGG | 50 | 50 | 50 | 51 | 51 | 51 | 52 | never |

Two populations. Four products crash (fertilizer d2, wool d7, melon/milk d11,
strawberry d23). **Four never crash and their prices RISE monotonically all
season.** EGG is the flattest thing in the game: 50 → 52 across thirty days,
exactly as `above_target=0.2, T=332` predicts (332 units over I0 costs $10).

### Volume tracks the draw — for them, not for us

Units sold per seat, binned by that game's realised drain:

| | ours (champ-h8) | top players |
| --- | --- | --- |
| WOOL, drain 1 → 25 | 102 → 187 (**1.8x**) | 131 → 534 (**4.1x**) |
| MILK, drain 1 → 37 | 104 → 220 (**2.1x**) | 186 → 614 (**3.3x**) |
| STRAWBERRY, drain 7 → 43 | 236 → 228 (**1.0x**) | 238 → 504 (**2.1x**) |

Our strawberry output is a **constant** — 236, 236, 235, 238, 237, 236, 228
across a sixfold range in demand. `straw_target=33` is a fixed standing count
and `DEMAND_BOOK` is disabled.

`DEMAND_BOOK` could not fix it as written anyway: it indexes on
`min(2, consumer_count)`, and milk and strawberry range over **0-6** shop
instances. Everything from 2 to 6 is one bucket.

### Our product mix is the inverse of the price gradient

Units per seat in our own games against the recorded corpus:

| product | ours | top | price d27 / base | verdict |
| --- | --- | --- | --- | --- |
| WHEAT | 178 | 496 | 2.0x | 2.8x behind |
| CARROT | 39 | 131 | 1.7x | 3.4x behind |
| TOMATO | **0** | 51 | 1.9x | absent |
| EGG | **0** | 91 | 1.2x | absent |
| STRAWBERRY | 236 | 229 | 0.8x | parity |
| MILK | 175 | 192 | 0.8x | parity |
| WOOL | 149 | 136 | 0.5x | parity |
| MELON | 72 | 76 | 0.5x | parity |

**We are at parity in the four markets that crash and 3x behind or absent in
the four that do not.** In our own games wheat ends at $50 against a $25 base
and tomato at $115 against $60, because nobody supplies them.

Caveat that keeps this honest: every single-product increase has already been
gated and lost — `wheat_engine` 24/28/40 all REFUTED, `carrot_standing` 8/16
REFUTED, `geese` 2/3 REFUTED, `tomato_standing` 6 null at n=550. Tiles and
labour are the binding constraint, so adding one product without freeing
another just displaces something worth more per tile-day. The LP's optimum
wants the whole allocation moved at once (103 wheat tile-plantings against our
~30), which is a plan-level change, not a knob.

### What this kills: the sell-floor branch

Extending the melon opponent-supply rule to strawberry/milk/wool
(`tools/make_oppfloor_variant.py`, three doses) is **null, +$74, flat across
all three doses**. Instrumenting `econ.units_sellable_above` over 6 games says
why:

    sell sizing calls        1,375
    units sold               9,231
    units WITHHELD by floor    735   (7.4% of sellable)
    calls where floor bound     35   (2.5%)

      STRAWBERRY  688 units over 24 binding calls
      WOOL         38 units over  9
      MILK          9 units over  2

**The floor binds on 2.5% of calls, and milk and wool together withhold 47
units in a whole season.** There is nothing for the rule to act on. The knob
bandit had already driven `premium_scale` to 0.35 over 4,270 samples, which is
low enough that the floor stops binding; three further escape hatches
(`shed_total > 80`, the final-push window, and the crashed-market release at
`price <= 0.3 * base`) drop it to $1 anyway.

### The gate, 270 held-out paired ladder worlds

Base champ-h8 at −$3,094/game, win 86/270 = 0.319.

| candidate | money | se | z | win | discordant | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| f-opp00 (opponent-stock floors) | +$41 | 19 | 2.19 | 0.315 | **1** | 1.0000 |
| h-cows (cows cut when no milk shop) | **+$0** | 0 | 0.00 | 0.319 | **0** | 1.0000 |
| h-sheep (sheep raised on strong wool) | −$20 | 15 | −1.40 | 0.319 | 0 | 1.0000 |
| h-hboth | −$20 | 15 | −1.40 | 0.319 | 0 | 1.0000 |

All four null. `f-opp00` clears z=2 on money but moves **one world in 270** and
$41 against ±45 points of rating noise — that is a measurement of nothing.

`h-cows` returned **exactly zero on every one of 270 worlds**: the condition
never fired. The archive's +$1,989 for `cows|MILK<=0 -> 6` came from a bandit
that evaluated the condition on its own decision day, when fewer of the eight
shop draws had been revealed; by the time the cow cap actually binds (day 10,
`min(cows, 2 + (day-2))`), four or more draws have landed and P(no milk shop)
has fallen to ~15% — and even then the cap does not bite. **A conditional arm's
archive margin does not transfer unless the condition fires at the moment the
knob binds.** Worth remembering before mining the archive for more of these.

## The honest limit

$52,652 is the value of the opponent **not existing**. It is not recoverable.
A model cannot stop them selling; it can only sell ahead of them, sell into
products they are not crowding, and avoid adding our own units to a pool that
is already past T.

What is recoverable is unmeasured. The next test is the cheap one: extend the
melon pattern in `_sell_floors` to strawberry, milk and wool, and gate it
paired on the ladder.
