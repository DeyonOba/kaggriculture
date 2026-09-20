# What to reward: every metric scored against 21,054 real games

2026-09-20. Computed from the full corpus traversal (`tools/metric_harvest.py`,
both shards, 21,054 games / 42,108 seat-records). Every number below is
measured against **who actually won**, not asserted.

Seat-0 win rate 0.497 (balanced). Median |margin| **$4,413** — margins are
small, which is why weak signals matter and why noise swamps most of them.

---

## Rule 1 — reward the DIFFERENCE, never the level

Margin is `us − them`, and both seats share the seed, the shop draw and one
price ladder. Any absolute metric is attenuated by everything they share.

| metric | r absolute | r relative | ratio |
| --- | --- | --- | --- |
| WOOL revenue | +0.097 | **+0.445** | 4.6× |
| MILK revenue | +0.074 | **+0.431** | 5.8× |
| STRAWBERRY revenue | +0.063 | +0.318 | 5.0× |
| units harvested | +0.155 | +0.370 | 2.4× |
| TOTAL revenue | +0.169 | +0.289 | 1.7× |

Differencing within the game is worth more than any feature choice on this
list. A reward on absolute money is throwing away 2–6× of its own signal.

## Rule 2 — the game is NOT decided before day ~16

`r` between the money lead at day *k* and the FINAL margin:

```
day     2      6     10     12     14     16     18     20     24     28
r    0.009 -0.009 -0.021 -0.017  0.049  0.190  0.404  0.575  0.777  0.928
sign  48.8%  50.9%  49.4%  50.7%  55.1%  55.3%  62.6%  70.0%  79.9%  88.2%
```

**Through day 12 the money lead is a coin flip.** Not weakly informative —
literally r ≈ 0.00 and sign-correct at 50%.

This is not in tension with "an action is worth $1,317 on day 3 and $103 on
day 27". Early actions have huge leverage; early *cash* does not, because
**cash at day 10 means you failed to invest it**. A dense reward on money
before day ~14 rewards hoarding, which is the opposite of the behaviour that
wins.

Consequence for a curriculum: shape on INVESTMENT STATE early (standing
production, quadrants, herd) and on RELATIVE REVENUE late. Do not shape on
cash before day 14 at all.

## Rule 3 — the strongest signals sit in the CONTESTED products

Top relative predictors, by |r|:

| rank | relative metric | r | R² | win gap Q4−Q1 |
| --- | --- | --- | --- | --- |
| 1 | WOOL revenue | +0.445 | 0.198 | +38.1% |
| 2 | MILK revenue | +0.431 | 0.186 | +22.7% |
| 3 | units harvested | +0.370 | 0.137 | +22.7% |
| 4 | TOMATO revenue | +0.318 | 0.101 | +4.4% |
| 5 | STRAWBERRY revenue | +0.318 | 0.101 | +24.1% |
| 6 | **TOTAL REVENUE** | +0.289 | 0.083 | **+52.2%** |
| 7 | realised $/unit | +0.282 | 0.080 | +33.5% |
| 8 | peak SHEEP | +0.255 | 0.065 | +14.4% |
| 9 | day reaching 3 quadrants | **−0.200** | 0.040 | −4.3% |
| 10 | STRAWBERRY median sell day | **−0.192** | 0.037 | −14.6% |

Wool (68% season utilisation) and milk (57%) are the two **traps** — pools
both seats crowd. Out-earning the opponent there is close to zero-sum, which
is exactly the $52,652/game congestion coupling appearing as the top
correlate in the corpus.

TOTAL REVENUE has a lower r than wool but the **largest win gap (+52.2%)**.
Use it as the terminal signal; use the per-product differences for shaping.

## Rule 4 — three popular candidates are worth nothing, measured

| candidate | result | verdict |
| --- | --- | --- |
| crop DROUGHT deaths | r = **−0.000**, win gap −0.3%, mean 3.97/game | **no signal at all** |
| moves/work | r = +0.032 abs, **+0.146 relative and POSITIVE** | more movement is mildly BETTER; routing efficiency is not a reward |
| CARE actions | r = −0.026, win gap −4.2% | mildly negative despite CARE doubling animal yield |

Drought deaths in particular look like waste and are not: 4 plants a game
against a $4,413 median margin.

## Rule 5 — two timing signals point OPPOSITE ways

```
STRAWBERRY median sell day   r = -0.192   sell EARLIER to win
MILK       median sell day   r = -0.124   sell EARLIER to win
MELON      median sell day   r = +0.136   sell LATER to win
```

Melon is the one everybody gets backwards, including me. A six-game autopsy
of a rank-24 player said strong seats clear melon on day 10; the corpus says
selling melon LATER correlates with winning, and the intervention agreed —
`x-mel100` (harvest at first yield) gated **−$1,482, z −3.52**. Corpus and
gate now concur; the small-sample autopsy was the outlier.

Melon's capacity collapses 111 → 5 by day 20 and it gluts in 100% of games.
Selling into that early appears to mean racing your own price down.

## Rule 6 — everything here is CORRELATIONAL

The largest R² in the table is 0.198. Nothing explains more than a fifth of
margin variance, and this project's record is that correlational structure
does not survive intervention: BC at 69.4% accuracy gated null, capacity
channels at p = 0.0001 moved the forecast 0%, the geese substitution had the
right marginal economics and lost $5–8k/game.

So this document ranks CANDIDATES. Every one still has to clear
`tools/ladder_gate.py` at >= 360 games with replication on a second block —
see the `gate-candidate` skill.

## What this implies for a reward function

```
terminal      relative final margin, sign-weighted      (win gap +52.2%)
shaping, late relative revenue in WOOL, MILK, STRAWBERRY (r 0.32-0.45)
shaping, early relative INVESTMENT STATE, not cash
              - day reaching 3 quadrants  (r -0.200, the one unexplored head)
              - peak SHEEP standing       (r +0.255)
              - units harvested           (r +0.370)
never         absolute cash before day 14 (r = 0.00, rewards hoarding)
never         drought deaths, moves/work, CARE count
```

Potential-based shaping keeps the optimal policy unchanged, so the early
terms should be expressed as a potential over investment state rather than
as a bonus, or they will distort the objective they are meant to accelerate.
