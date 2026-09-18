# Moving the whole allocation: the LP plan is executable, and decisively worse

2026-09-18. `tools/make_lpalloc_variant.py`, gated with `tools/ladder_gate.py`.

## Why this test

Every single-product target has been gated and lost:

| knob | values tried | verdict |
| --- | --- | --- |
| `wheat_engine` | 24 / 28 / 40 | all REFUTED |
| `carrot_standing` | 8 / 16 | REFUTED |
| `straw_target` | 21 / 27 / 36 / 40 | all negative |
| `tomato_standing` | 6 | null, n=550 |
| `cows` | 6 / 10 | REFUTED |
| `sheep` | 6 / 10 | REFUTED |
| `geese` | 2 / 3 | REFUTED |

That is the signature of a **tile-constrained** optimum: adding one crop
displaces something worth more per tile-day, and removing one frees capacity
with nothing queued to use it. Neither direction can win alone.

Measured at day 13, three quadrants = 75 tiles:

```
ours  wheat 23  carrot  0  tomato  0  straw 33  cow 8  sheep 8  goose 0  = 72
LP    wheat 11  carrot  7  tomato 11  straw 29  cow 6  sheep 5  goose 4  = 73
```

Both saturated. It is purely a reallocation — and the LP funds its additions
(tomato, carrot, geese) by **cutting** animals and strawberry, which is
exactly what no single-knob sweep can express.

## The parameterisation

One dose α moves every target together, blended **at the target site**:

    target = round(base_expression + α * (lp_target[day] - base_expression))

Blending at the site rather than substituting an array makes **α = 0
bit-identical to champ-h8 by construction**. That was verified before any
other dose was trusted, for all three families. Three nested families:

| family | blends |
| --- | --- |
| `a-lp*` | wheat, carrot, tomato, strawberry, melon, cow, sheep, goose |
| `b-lp*` | + the LP hand schedule (`CHAMP_HIRES` → LP `hands`) |
| `c-lp*` | + the LP land schedule (quadrant #1 on **day 0**, #2 on day 9) |

The nesting matters: the LP's crop mix assumes the LP's labour and the LP's
land, so crops alone is an incoherent hybrid.

## Execution check — the plan IS realisable

The first two families could not realise the opening. `a-lp100` finishes the
season **−291 wheat tile-days** against target, because at day 0–6 it holds
25 tiles while the LP plan assumes 50:

```
a-lp100   day 2   wheat  8/31    melon 16/16    tiles used 25   (saturated)
```

Adding land fixes it. The LP's day-0 shopping list is
26 wheat $260 + 16 melon $1,280 + 1 carrot $20 + 1 goose $300 + land $1,000 =
**$2,860 against $3,000 starting cash** — it fits, and the executor delivers:

```
c-lp100   day 2   wheat 25/31   melon 16/16   tiles 42
          day 3   wheat 31/29   melon 16/16   tiles 49
          day 4   wheat 31/27   melon 16/16   tiles 49
```

**So this is not a failure to execute.** The LP allocation is reachable by our
executor once the land schedule comes with it.

A false lead worth recording: `c-lp100` runs on $4–$362 of cash for eleven
days, which looked like the mechanism. It is not — champ-h8 does the same
($4 on day 1, $14 on day 7). Cash is fully deployed in the opening either
way; only what it buys differs.

## The gate — 270 held-out paired ladder worlds

Base champ-h8 at −$3,094/game, win 86/270 = 0.319.

| candidate | money | se | z | win | McNemar p |
| --- | --- | --- | --- | --- | --- |
| a-lp33 | −$9,937 | 1,292 | −7.69 | 0.230 | 0.0005 |
| a-lp67 | −$17,950 | 1,278 | −14.05 | 0.196 | 0.0000 |
| **a-lp100** | **−$31,728** | 1,412 | **−22.47** | **0.159** | 0.0000 |

Monotone in dose, significant on money **and** win rate, and it roughly halves
the win rate. This is the largest and cleanest rejection in the project. It is
not noise and it is not a broken hybrid — the direction itself is wrong.

## What it means

**The LP is refuted as a teacher for the second time, on a second quantity.**

| use | result |
| --- | --- |
| LP labour dual → action value | r = **−0.34**, inverted, 23× off in magnitude |
| LP allocation → tile plan | **−$31,728**, z = −22.5, monotone in dose |

The reason is the same both times and it is now measurable. The LP is a
**static single-player** allocation model. It prices its output against the
whole market absorption curve, and `docs/market_congestion.md` shows the
opponent takes half of it — $52,652 a game. A plan optimised against
absorption that another player is already consuming over-produces by
construction.

`docs/lp_bound.md` should be read accordingly: the $147k ceiling is an upper
bound on a game we are not playing. It remains useful as a bound. **It is not
a direction.**

## What survives

The per-tile-day rent table, computed at realised prices from
`tools/market_clock.py --agent champ-h8`:

| crop | tile-days | units | $/unit | $/tile-day |
| --- | --- | --- | --- | --- |
| TOMATO | 20 | 52 | 78 | **203** |
| STRAWBERRY | 20 | 24 | 129 | **155** |
| MELON | 13 | 6 | 201 | 93 |
| CARROT | 4 | 4 | 62 | 62 |
| WHEAT | 5 | 6 | 45 | 54 |
| COW | 20 | 6 | 155 | 46 |
| GOOSE | 20 | 16 | 49 | 39 |
| SHEEP | 20 | 4 | 132 | 26 |

Read with care — these are **marginal** prices at our current volume, and
every one collapses at scale. Producing 11 standing tomato tiles would put
~570 units into a ~390-unit season drain, past T=200, taking the price from
$78 to roughly $26. That non-linearity is precisely what the LP does model
correctly, and it is why the LP wants only 8 tomato tiles despite tomato
having the highest marginal rent.

The honest statement: **we do not currently have a model that beats
champ-h8's allocation.** The LP is refuted, single knobs are refuted, and the
whole-vector move toward the LP is refuted at z = −22.5.
