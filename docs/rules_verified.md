# The rules, verified against the engine

2026-09-20. Every rule `README.md` states, checked against
`kaggle_environments/envs/kaggriculture/kaggriculture.py`.

    PYTHONPATH=. uv run python tools/readme_audit.py
    PYTHONPATH=. uv run python tools/readme_audit.py --section price

**Result: 286/286 verified, no discrepancies.** The README is an accurate
description of the engine.

## Why do this at all

Exploration, not usage. The question is never "does our agent use this rule"
but "is the rule we believe the rule the code implements". This project has
paid twice for skipping that:

- a **"$88/unit melon"** that never existed — a misread of a blank column
- **"FEED is the wheat-buy mechanism"** — asserted, then measured at r = −0.079

Both were beliefs about the rules, held confidently, never checked. Each
check below either *simulates* the engine or reads its constants. None
restates the README back to itself.

| section | checks |
| --- | --- |
| 1. Object Types (table) | 42 |
| 1b. Yield / tile / day | 8 |
| 1c. The three crop notes | 15 |
| 2. Season shape | 11 |
| 2a. Movement | 5 |
| 2b. Shed — PICKUP / DROP | 10 |
| 2c. Plants | 10 |
| 2d/2e. Animals + Animal Care | 18 |
| 2f. Terrain | 10 |
| 3. Market Action | 10 |
| 4. Harvest Yields | 23 |
| 5. Map Features | 14 |
| 6. The Price Function | 99 |
| 6a. Selling | 4 |
| 6b. Buying | 7 |

---

## 1. Object Types

All seed costs, base prices, first-yield days, intervals, `max_held`,
structures and products match. `FERTILIZER` base 100 confirmed.

### The "Time to Max Yield" column is NOT the engine's `max_yield_day`

This is the one place the table looks wrong and is not. Three different
quantities share similar names:

| crop | README "Time to Max Yield" | engine `max_yield_day` | what each means |
| --- | --- | --- | --- |
| Melon | 10 | **12** | README = age yield stops rising; engine = end of the bonus window |
| Tomato | 11 | **8** | README = age of the *last* scheduled yield; engine = first |
| Strawberry | 16 | **10** | same |

Reading `max_yield_day` as "time to max yield" would mis-time a melon
harvest by two days and a strawberry by six. The README's own notes state
this correctly; the trap is in the field name, not the docs.

### Yield / tile / day — the definition is exact and checkable

"Total units harvested divided by the days the tile is occupied, watering
daily and harvesting at peak yield." Occupancy runs from planting (age 0)
through the harvest day **inclusive**:

| crop | units | days | = |
| --- | --- | --- | --- |
| Wheat | 4 | 5 | 0.80 |
| Carrot | 3 | 4 | 0.75 |
| Tomato | 4 | 12 | 0.33 |
| Strawberry | 4 | 17 | 0.24 |
| Melon | 6 | 11 | 0.55 |

Melon's denominator is **11, not 13** — the peak lands at age 10, and ages
11–12 add nothing. Animals are `1 / interval` in steady state: goose 1.00,
cow 0.50, sheep 0.33.

---

## 1c. The three crop notes — all confirmed by simulation

**Where the one-time bonus actually lives.** Not in the day refresh. It is
applied inside the **WATER handler** (`_apply_unit_action`, lines 431–444):

```
window_start = (max_yield_day + 1) // 2
if window_start <= age_days <= max_yield_day:
    bonus = 2 if fertilized_until_day >= day else 1
    yield_units = min(max_yield, yield_units + bonus)
```

`_daily_refresh_plants` explicitly `continue`s for non-ongoing crops. A
model of crop growth that only reads the day refresh will conclude one-time
crops never gain yield.

- **Melon** window ages 6–12 ✓. Watering reaches the cap of 6 at **age 10**,
  so 11–12 add nothing ✓. Fertilized reaches it at **age 8** ✓.
- **Wheat / Carrot** peak at 4 and 3 on watering alone; reach the listed 6
  and 4 only with fertilizer ✓.
- **Tomato** yields at ages 8, 9, 10, 11; **Strawberry** at 10, 12, 14, 16;
  both capped at 4 scheduled productions ✓.

**The two-day death clocks are asymmetric, and the asymmetry matters:**

- a new plant starts `consecutive_unwatered = 1` — the planting day counts as
  already missed. Plant and fail to water that same day and it is a weed that
  night, **before it grows**. There is no grace period.
- a new animal starts `consecutive_unfed = 0`, so it survives its first day
  unfed.

---

## 2. Actions

Season shape: 24 turns/day, 720 turns, 30 days, `startingMoney` 3000,
`boardSize` 10, `maxMarketOrdersPerTurn` 10, `shedCapacity` 100,
`weedSpawnChance` 0.005, shop unlock 3 days, shop sell 4 turns, town centre
24 turns. All confirmed.

### Movement

Off-edge moves are no-ops ✓. **LOCKED tiles are passable** ✓ — and this is
load-bearing: tile actions (`PLANT`, `WATER`, `BUILD_*`) no-op on a locked
tile and consume nothing ✓, but the shed ops `PICKUP`, `DROP` and
`PLACE`-into-shed work from a locked shed-access tile ✓, because they use the
tile only as a standing position.

### Shed

- `PICKUP` default quantity is **1**, and clamps to what the shed holds ✓
- **any** shed item is valid, animals included — verified for GOOSE, COW,
  SHEEP, MILK, EGG, FERTILIZER ✓
- seeds live in a separate slot and are **never** picked up ✓
- `DROP` dumps the *entire* inventory; overflow past `shedCapacity` is
  discarded ✓; away from the shed it is a no-op ✓
- shed-access tiles at `boardSize` 10 are exactly (4,4), (5,4), (4,5), (5,5)
  ✓ — one per quadrant, and three start locked

### Plants

`PLANT` consumes one seed and no-ops with none ✓. A second `WATER` the same
day is a no-op ✓. `HARVEST` of a one-time crop removes the plant and the
units go to the **acting unit's bag**, not the shed ✓. `HARVEST` at 0 units
is a no-op ✓. `FERTILIZE` requires a FERTILIZER in the bag, consumes it ✓,
and its bonus applies **only on days the plant is also watered** ✓.

### Animals and Animal Care

`FEED` consumes 1 WHEAT from the acting unit's bag ✓ and no-ops without one
✓; a second FEED the same day spends no second wheat ✓. First products land
on day 4 / 8 / 6 and then every 1 / 2 / 3 days ✓. `max_held` caps
**unharvested** product on the tile ✓.

**`COLLECT_FERTILIZER` is a use-it-or-lose-it resource.** Every *surviving*
animal makes exactly 1 available at end of day — fed or not, cared or not ✓ —
and it does **not accumulate**: an animal left five days still yields 1 ✓.

`CARE` banks a bonus only on days the animal was **also fed** ✓ ("basic
needs first"). `PLACE` onto a matching structure takes exactly one and
**ignores `n`** ✓; a goose cannot be placed on a pasture ✓.

### Terrain

`BUILD_COOP` / `BUILD_PASTURE` build on unoccupied tiles and no-op on
occupied ones ✓. `DIG` removes a plant, a weed, an empty coop and an empty
pasture ✓ — but is a **no-op on an occupied structure** ✓.

---

## 3. Market Action

Six order types confirmed in `_parse_order`: `BUY_SEED`, `BUY_ANIMAL`,
`BUY_PRODUCT`, `SELL`, `HIRE`, `BUY_LAND`.

**Hire cost is fibonacci**, `farmHandCostMult * fib(n)` — measured
1, 1, 2, 3, 5, 8, 13, 21 ✓, resetting each day. This convexity is why
reactive-hiring arms die before they gate.

`LAND_PRICES` = [1000, 2000, 4000], increasing ✓, against a $3,000 opening
bank.

---

## 4. Harvest Yields

Bonus windows: wheat 2–4, carrot 2–3, melon 6–12, all equal to
`ceil(max_yield_day / 2) .. max_yield_day` ✓. One-time crops gain +1 per
watered day, +2 fertilized ✓ (measure the *first* application — a later one
can be clamped by `max_yield`, which is why carrot's second fertilized step
reads +1). Ongoing crops produce a base 1, doubled to 2 only when fertilized
**and** watered ✓. Fertilizer does not rescue an unwatered plant ✓.

**Decay.** One-time crops arm at `(planted_day + max_yield_day + 1) * 24` ✓;
ongoing crops start at −1 and arm one day after the production count reaches
`max_yield` ✓. Once armed, yield drops by 1 **every other turn** ✓ and the
tile becomes a WEED at 0 ✓.

---

## 5. Map Features

Four 5×5 quadrants, three buyable ✓. Eight shop types with exactly the
documented demand sets ✓. `YARN_STORE` and `PET_CAFE` are single-product, so
the 2× rule applies to them ✓. The town centre excludes FERTILIZER ✓. Shop
instances cap at 8 ✓.

**Hand spawn ignores locked tiles** ✓ — with the farmer on (4,4), the
least-occupied rule sends the first hire of each day to **(5,4)**, which is
locked until NE is bought. Locked tiles are passable, so the hand can walk
back out.

---

## 6. Market Mechanics

The price function, all 9 resources × 11 checks = 99, all pass — base, `I0`,
`T`, both shape functions, both targets, and the three quoted prices
P(I0−T), P(I0+T), P(I0+2T).

Structural properties confirmed: price at `I0` is exactly `base` ✓; floor is
$1 ✓; `log` uses `ln(1+x)` so `f(0) = 0` ✓; `hinge` is `u + 8·max(0,u−1)²`
with `HINGE_GAIN = 8` and `f(T) = 1` ✓; scarcity raises and glut lowers,
monotone across all nine ✓.

**The strategic split is real and sits in `above_func`:**

| shape | resources | behaviour |
| --- | --- | --- |
| `log` | WHEAT, EGG | uncrashable — P(I0+2T) is $19 / $39 |
| `sq` | MELON 3.6, WOOL 3.2 | crash to the $1 floor |
| `linear` @1.6 | STRAWBERRY, MILK | crash to $1 |

All four premium resources (base > 100) use `above_target > 1` ✓.

### Selling

A sale credits the player and adds one unit to market inventory ✓. **At the
$1 floor the unit is still paid for but is NOT added to inventory** ✓ — so
the floor stays responsive. Every product including FERTILIZER is sellable ✓.

### Buying

Only **WHEAT and FERTILIZER** are buyable via `BUY_PRODUCT` — verified
behaviourally by attempting all nine through the real agent path ✓. Running
out of money stops the order ✓.

**A full shed blocks `BUY_PRODUCT` and `BUY_ANIMAL` — but not `BUY_SEED`** ✓.
Seeds bypass the cap entirely. Buying an animal into a full shed silently
fails, which is a real failure mode for a livestock plan.

**The round-trip identity holds exactly**: the buy is quoted at post-buy
inventory and the sell at pre-sell inventory, so buy-then-sell against an
otherwise-unchanged market nets **exactly zero** ✓. There is no arbitrage in
a round trip.

---

## What the audit caught — in the audit, not the engine

Five failures surfaced on the first run. **All five were mine**, and the
pattern is worth keeping:

1. `_blank_farm` omitted `"farmer"`, so `_apply_unit_action` raised KeyError
   rather than no-opping — the check crashed instead of reporting.
2. `episodeSteps` is a bare int in `kaggriculture.json` while its neighbours
   are `{"default": ...}` schema dicts.
3. The `COLLECT_FERTILIZER` test never fed the animal, so it **escaped on day
   2** and the check was measuring its own setup error.
4. The yield/tile/day denominator: `max_yield_day + 1` overstates melon
   (peak is age 10, not 12), and `last_age` undercounts ongoing crops by the
   planting day.
5. Carrot's fertilized step measured at a **saturated** point — 1 → 3 → 4,
   clamped by `max_yield`, so the second step reads +1 through the cap.

Every one of them would have been reported as "the README is wrong about X".
A disagreement between a document and a test is evidence about *the pair*,
and the test is the younger and less-reviewed half.
