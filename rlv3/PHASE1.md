# Phase 1 — the opening, and what an RL model has to learn from it

## Where phase 1 ends

**The opening ends on the day the last major capital commitment lands.** Not a
fixed day: a state. Traced on the champion (`tools/theta_probe.py`, one game):

```
day  money   quads  cows sheep  strawberry     what committed
  0  3,000     1      0    0        0          melon 12 + wheat 7 seeded
  2      4     1      2    2        0          first herd
  4    360     1      2    2        0          pasture up
  6     91     1      4    2        1          herd ramping
  8      0     2      6    2       21          quadrant 2, strawberry block
 10    212     3      8    5       21          quadrant 3, cow cap  <- LAST COMMITMENT
 14 14,652     3      8    8       33          nothing new: consequence
```

Every dollar is committed by day 10 and the bank sits at $212. From day 14 the
board only *runs*. So phase 1 is **days 0-10**, ending when land is at 3
quadrants, cows are at cap, the strawberry block is seated and hands have
reached their plateau (`CHAMP_HIRES` hits 11 at day 10 and never moves again).

Two independent measurements agree: everyone is broke to day 9 and the
divergence between agents is ONE DAY, day 10
([[kaggriculture-state-capture-audit]]); and melon day 10 is a RACE -- the
leaders clear 60 units at $271 while we finish on day 13 at $97
([[kaggriculture-harvest-race]]).

## The goal, in one sentence

**Convert $3,000 and ten days of labour into the portfolio that maximises
P(win) at the moment the melon money lands, under a demand distribution that
is unknown on day 0 and resolves three times before phase 1 is over.**

Not maximise money. Phase 1 banks almost nothing by construction -- the bank
is $212 on day 10. What it produces is *standing capacity*, and the objective
on that capacity is win probability, not its book value.

## Why this phase is learnable, and the others less so

Phase 1 is the only window where the binding constraint is CAPITAL. Later the
constraint is labour (hands/work 0.47-0.58 at day 11-15) and then it is simply
the lead (money_lead explains 0.552 -> 0.892 of the outcome from day 12). A
capital-allocation problem with a known price identity and a known demand
lattice is a well-posed decision problem. "Win from here given a lead" is not.

It is also where investment PROPAGATES: melon commitment correlates +0.379
with the outcome early and **flips sign at day 8-10**
([[kaggriculture-phase-propagation]]). Phase 1 is the last point at which a
decision still compounds.

## The information structure, which is the whole design

Shops unlock on `econ._UNLOCK_DAYS = (3, 6, 9, 12, 15, 18, 21, 24)`, one drawn
uniformly from eight known types. So inside phase 1 there are **exactly three
draws** and they land on days 3, 6 and 9:

```
days 0-2   ZERO shops exist. No information. Every game is identical here,
           and the leaders' openings are near-unanimous because there is
           nothing to condition on. This is a PRIOR, and it must be believed.

day 3      draw 1 -> 1 of 8 known
day 6      draw 2
day 9      draw 3      <- the last one that can still change phase 1

day 10     melon clears. Phase 1 is priced.
```

That is why days 2-9 are the ONLY window carrying real variation between the
three leaders (0.085 mean pairwise gap overall; their days 0-1 and 10-26 are
near-unanimous). **Three decision points is the entire reactive surface of the
opening.** An RL model for phase 1 has three places to act, not thirty.

## What is already optimised here, and what an RL model inherits

The champion's opening is a decoded playbook, and every constant in it is a
phase-1 decision that has already been tuned and gated:

| constant | value | what it decides |
| --- | --- | --- |
| `CHAMP_OPENING` | melon 12, wheat 7 | the day-0 prior, spent before any draw |
| herd ramp | cows `min(8, 2+(d-2))`, sheep `min(8, 2+(d-6))` from d7 | livestock accumulation |
| `CHAMP_LAND_DAY` | 5 | earliest quadrant 2 |
| `CHAMP_STRAW_EARLY` / `_TARGET` | 18 by d9, 33 from d10 | the block staged around melon money |
| `CHAMP_HIRES[0:11]` | 5,4,4,5,4,5,7,9,9,10,11 | labour, plateaus exactly at day 10 |

A 35-coordinate CEM search over all of these and 20 more found **nothing**
(+0.000 ± 0.047 on a fresh draw). So the champion's opening is a sharp local
optimum *as a constant*. That is the transfer: an RL model does not need to
rediscover these values, and will not beat them by perturbing them. It
inherits them as the day-0 prior and spends its capacity on the one thing a
constant cannot express -- **conditioning days 3, 6 and 9 on what the draw
actually was.**

## Where variance enters

P(win) on day 0 is the base rate: both boards are identical, no draw has
landed, nothing distinguishes the game. It becomes informative only as the
draws land and the opponent's board (fully public) diverges.

So the risk posture is not a day-0 decision. It is what the three draw days
are FOR:

> When P(win) is low at day 3, 6 or 9, phase 1 should convert to higher
> variance -- and that conversion is mechanical, not a separate control.

Low P(win) buys the assets whose value is contingent on a draw that has not
happened yet: the TRAP products (`MELON`, `MILK`, `WOOL`, `STRAWBERRY` --
linear/1.6 and sq/3.2, prices that crash on volume but spike when demand
lands) and livestock bought BEFORE its shop opens, which is the lead-time
option premium `kagent/options.py` prices at $559 for a sheep.

High P(win) buys the annuities -- `WHEAT` and `EGG`, the only two whose price
survives volume (log/0.20) -- and delays livestock until its shop is confirmed.

The leaders already reveal which side they take: **sheep on day 0 in all 40
games, with P(never a yarn shop) = 0.34.** They pay a one-in-three-worthless
option premium every single game. That is a standing risk posture, expressed
as livestock, before any information exists.

## What phase 1 emits

A **commitment schedule** -- of capital available now plus expected income
across the next three days, what fraction goes where:

```
invoice[d] = {"seed": …, "livestock": …, "land": …, "hands": …}   sums to 1
```

Dollars, because dollars are what both sides of the crop/livestock choice
actually compete for; a pasture is one tile AND $400 AND feed wheat AND chore
turns, and a share of tiles prices none of that. Five knots (days 0, 6, 12,
20, 29) with monotone interpolation describe a whole season in twenty numbers,
of which phase 1 owns the first two.

## The success criterion for phase 1

**Agreement with the top boards, conditioned on the draw** -- not agreement
with champ-h11, which loses most of its games and whose ceiling we would
simply inherit. Bucket leader games by which shops had opened by day d and
compare within bucket. The noise floor is known: the three leaders agree with
each other at 0.085 mean pairwise gap and with themselves across their own
games at 0.086, so **0.085 IS the target and there is no credit below it.**

Then apply the risk posture and re-check: when a leader is behind at day 6,
does their invoice move toward traps and early livestock -- and does ours move
the same way?

## The stages, measured on the leaders (2026-09-29)

40 leader games, winner seat, from `rlv3/trajectory.py`'s invoice:

```
opening ends (last land buy)        p25/med/p75 = 8 / 8 / 9     IQR 1 day   range 5-18
contest ends (last >$150 invest)    p25/med/p75 = 26 / 26 / 29  IQR 3 days  range 23-29
quadrants OWNED at the end         3: 37 games   4: 3 games      (2 purchases = 3 owned)
```

So **OPENING 0-8 · CONTEST 9-26 · CONSOLIDATION 27-29**, and the boundaries
are tighter than the +/-0-2 days allowed: the opening ends on day 8 in half
the games and by day 9 in three quarters. One difference from our champion
worth stating plainly: the leaders end the opening on **day 8, not 10**. On
land they and the champion AGREE -- both own three quadrants -- and an
earlier draft of this paragraph said "37 of 40 stop at two quadrants", which
counted PURCHASES (2) as quadrants owned (3). Corrected 2026-09-29.

**AND THE TARGET HAS MOVED.** These 40 replays are episode ids ~109-113M.
The live leaderboard (checked 2026-09-29) shows the leaders now buying the
FOURTH quadrant and their first later than day 6. Nothing on disk is newer
than these replays (`replays/top/` is older, ~99-106M), so the readout and
every agreement number in this file describe a frontier that has since
moved. Their agents are evidently learned -- self-play against their own
strong agents is the likely mechanism -- which means the reference set has to
be refreshed, not fixed once.

Per-stage metric, as agreed:
- **Opening** -- capital deployed / capital available, and agreement of the
  standing stock with the top boards conditioned on the draw.
- **Contest** -- profit rate and risk-adjusted margin against the opponent;
  this is where the risk posture does its work.
- **Consolidation** -- dead capital at the end (unsold, destroyed at midnight)
  and realised sell price against the max achievable.
