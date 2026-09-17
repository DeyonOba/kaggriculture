# Market structure: how every environment parameter connects

Written 2026-09-17, from `kagent/constants.py`, the engine
(`kaggriculture.py:431-467`, `:770-801`, `:806-833`), and 5 measured games of
`champ-h8` vs `pub-ahmed-v38`. Tools: `tools/market_table.py`,
`tools/market_headroom.py`.

This did not exist. We had per-product realised prices after the fact and we had
`econ.town_drain_per_day`, but the two were never put side by side, so the
planner has always priced a crop at `base` — the price of the *first* unit.

---

## 1. The one parameter nobody had read: MELON has no shop

`SHOPS` maps eight shop types to the products they consume. Melon appears in
**none of them**. Its only buyer is the town centre, at a flat **1.0 unit/day**
for the whole season. Every other product's demand grows with each shop unlock;
melon's never does.

Measured, 5 seeds, both farms combined:

| | melon |
| --- | --- |
| town absorbs, whole season | **30 units** |
| both farms supply | **142 units** |
| supply / demand | **4.7×** |

The price consequence, per day (start of day, before trading):

| day | 10 | 11 | 12 | 13 | 14 | … | 29 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| price | 271 | 217 | 164 | 113 | **87** | … | 123 |
| × base | 1.08 | 0.87 | 0.66 | 0.45 | **0.35** | | 0.49 |

Melons do not exist before day 10 (`first_yield_day=10`). Within four days of
existing, the market is destroyed, and it **never recovers** — the town removes
1/day from an inventory sitting ~114 units above I0.

**Melon is a one-shot, first-mover-takes-all market.** The same melon is worth
271 on day 10 and 87 on day 14. This is the mechanism behind the day-10 gap
already measured in the balance sheet: their cash **+$14,228**, ours **+$228**,
while we held *more* melon. They sold into an untouched ladder; we sold into the
crater.

It also explains, after the fact, why **seven** melon-production fixes all lost
money (`exec-burst`, `exec-ripen`, `exec-drop`, …). Melon volume is nearly
worthless at the margin. Melon *timing* is worth thousands.

---

## 2. The price curve is the real crop-value parameter

`econ.price(item, inventory)` interpolates from `base` using four parameters:
`T` (scale), `below_func`/`below_target` (undersupply), `above_func`/
`above_target` (oversupply). Evaluated at I0 + delta:

| item | base | T | above | at | −200 | −100 | **0** | +100 | +200 | +400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WHEAT | 25 | 400 | log | 0.2 | 39 | 35 | **25** | 21 | 21 | **20** |
| EGG | 50 | 332 | log | 0.2 | 62 | 56 | **50** | 42 | 41 | **40** |
| CARROT | 35 | 450 | sqrt | 0.7 | 51 | 43 | **35** | 23 | 19 | 12 |
| TOMATO | 60 | 200 | sqrt | 0.6 | 84 | 72 | **60** | 35 | 24 | 9 |
| FERTILIZER | 100 | 200 | linear | 0.4 | 140 | 120 | **100** | 80 | 60 | 20 |
| STRAWBERRY | 120 | 100 | linear | 1.6 | 239 | 204 | **120** | **1** | 1 | 1 |
| MILK | 160 | 122 | linear | 1.6 | 283 | 247 | **160** | **1** | 1 | 1 |
| WOOL | 200 | 105 | sq | 3.2 | 245 | 240 | **200** | **1** | 1 | 1 |
| MELON | 250 | 300 | sq | 3.6 | 296 | 290 | **250** | 150 | **1** | 1 |

**The game's central trade-off, stated in one line: high-base products are
traps, low-base products are annuities.**

- **Inexhaustible** — WHEAT and EGG (`log`, 0.2). Dumping 400 units past I0
  still leaves them at **80% of base**. You cannot crash these markets.
- **Fragile** — STRAWBERRY, MILK, WOOL collapse to the price floor of **1**
  within ~100 units. MELON within ~200.

`above_target` is the single most decision-relevant number in `constants.py`
and the planner reads none of it.

---

## 3. Measured absorption, whole season, both farms

| item | absorbed | supplied | sup/abs | | unclaimed $ | destroyed $ |
| --- | --- | --- | --- | --- | --- | --- |
| EGG | 330 | 74 | **0.20** | under | **18,817** | |
| TOMATO | 258 | 27 | **0.12** | under | **15,260** | |
| WHEAT | 588 | 232 | 0.38 | under | 11,094 | |
| CARROT | 300 | 129 | 0.43 | under | 3,552 | |
| STRAWBERRY | 546 | 474 | 0.95 | ~even | | 1,279 |
| WOOL | 222 | 205 | 1.04 | over | | 2,798 |
| MILK | 210 | 281 | 1.19 | over | | 3,590 |
| MELON | 30 | 142 | **4.72** | over | | 4,868 |
| FERTILIZER | **0** | 485 | **∞** | over | | **23,863** |
| | | | | | **48,723** | **36,398** |

For scale, our whole gap to `pub-ahmed-v38` is ~$44k, and this is both farms
combined. Two lines stand out:

- **FERTILIZER has zero demand.** It is in no shop and not a town-centre
  product, so `town_drain_per_day` is 0.00 forever. We put 485 units into a
  market that removes none. Its price ends at ~4. Fertilizer is a free
  by-product, so this is not a loss of principal — but it is strictly
  monotone-decreasing revenue, which makes **sell fertilizer early** a rule with
  no downside.
- **EGG and TOMATO are the two big unfilled markets**, and they are the two with
  the gentlest oversupply curves. Demand we are not serving, in the products
  that would not punish us for serving it.

---

## 4. Yield semantics (needed to price a tile correctly)

From `kaggriculture.py:770-801` (ongoing) and `:431-467` (one-shot):

**Ongoing — WHEAT, CARROT, TOMATO.** One production every `interval` days
starting at `first_yield_day`, each adding **1 unit, or 2 if watered AND
fertilized**. The tile dies after exactly `max_yield` productions
(`max_lifespan_step`). So:

- lifetime output is `max_yield` units, or up to `2 × max_yield` if fertilized
  through every production and harvested between them;
- FERTILIZE is active for `day, day+1, day+2` (3 days). On interval-1 wheat that
  is **+3 units of free output per FERTILIZE action** (~$120), from a
  by-product with no market value. We currently fertilize only tomato at ages
  8 and 11 (`kagent/scheduler.py:130`).

**One-shot — MELON, STRAWBERRY.** Units come *only* from WATER actions inside
`[(max_yield_day+1)//2, max_yield_day]`, +1 each, **+2 if fertilized**. Harvest
destroys the tile. Melon's window is days 6–12 — six waters for six melons, or
three if fertilized.

**Animals never age out.** `_daily_refresh_animals` has no lifespan; an animal
produces until the season ends or it goes unfed two days running.

---

## 5. What one tile is actually worth

`tools/market_table.py --plant` prices each unit at the market inventory
prevailing **on the day it lands**. Net of seed, 5 seeds:

| planted day | WHEAT | CARROT | TOMATO | STRAWBERRY | MELON |
| --- | --- | --- | --- | --- | --- |
| 0 | 83 | 50 | 205 | **706** | **880** |
| 1 | 83 | 50 | 207 | 727 | 569 |
| 2 | 84 | 51 | 210 | 745 | 404 |
| 10 | 116 | 56 | 248 | 619 | 526 |
| 18 | 125 | 70 | **310** | 18 | — |
| 24 | 128 | 86 | — | — | — |

A day-0 melon nets **$880**; a day-1 melon nets **$569**. One day of delay costs
35% — again the first-mover structure, now visible as a planting decision.
Tomato is the only crop that gets *better* with time (205 → 310), because the
town keeps draining a market nobody supplies.

Ranking by `$/tile-day` at base price versus at the margin we actually face
**inverts**:

| | at `base` | at the realised margin |
| --- | --- | --- |
| 1 | MELON 118.3 | WHEAT 60.5 |
| 2 | CARROT 40.0 | MELON 50.4 |
| 3 | STRAWBERRY 38.0 | CARROT 49.3 |
| 4 | WHEAT 35.0 | TOMATO 36.4 |
| 5 | TOMATO 23.8 | STRAWBERRY 8.4 |

---

## 6. Animal ROI on the ladder each product actually sells into

Bought day 2, produced through day 29, priced at the season-end inventory both
farms create:

| animal | cost | first | interval | product | units | gross @I0 | gross w/ glut | net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **GOOSE** | 300 | 4 | 1 | EGG | 24 | 1,107 | **1,641** | **+1,341** |
| COW | 400 | 8 | 2 | MILK | 10 | 1,506 | 268 | −132 |
| SHEEP | 500 | 6 | 3 | WOOL | 8 | 1,592 | 128 | −372 |

*Caveat, stated because it matters:* the glut column prices every unit at the
season-end inventory, which is the worst moment. Milk and wool sold at a mean of
118 and 138 during the season, so −132 and −372 are floors, not expectations. At
I0 prices all three animals are comparable (+1,106 cow, +1,092 sheep, +807
goose). **The entire difference is price impact** — and that is the point: the
egg market is the only animal market that survives the volume.

The goose is also cheapest (300 vs 400/500), earliest (day 4 vs 6/8) and the
most frequent (every day vs every 2–3).

**`KNOB_DEFAULTS` sets `geese: 0`.** We buy none. The mined champion medians in
the same file say 2–5 by day 18.

---

## 7. What this predicts, and where it is being tested

| prediction | status |
| --- | --- |
| Buying geese beats the `geese: 0` default | **gating now**, branch `exec-geese` |
| Selling melon on day 10 beats holding | day-10 balance-sheet gap says +$12,078; `exec-drop` tried the *planner* side and failed |
| Selling fertilizer early is free money | untested |
| Fertilizing ongoing wheat is +3 units/action | untested |
| More tomato late (day 14–18) is underpriced | untested |
| More melon volume is worth ~nothing | **confirmed** — 7 rejected experiments |

---

## 8. How to use this

- **Never price a crop or a sale at `base`.** Use `econ.price` at the inventory
  prevailing when the unit lands. `base` is the price of the first unit only.
- **Check `above_target` before adding volume.** Above 1.0 (STRAWBERRY, MILK,
  WOOL, MELON) the market will not absorb it. At 0.2 (WHEAT, EGG) it will absorb
  anything.
- **Melon and fertilizer are timing problems, not volume problems.**
- Gate on the ladder (`gate-candidate` skill); the 3-opponent panel inverts.
