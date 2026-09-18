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

## The honest limit

$52,652 is the value of the opponent **not existing**. It is not recoverable.
A model cannot stop them selling; it can only sell ahead of them, sell into
products they are not crowding, and avoid adding our own units to a pool that
is already past T.

What is recoverable is unmeasured. The next test is the cheap one: extend the
melon pattern in `_sell_floors` to strawberry, milk and wool, and gate it
paired on the ladder.
