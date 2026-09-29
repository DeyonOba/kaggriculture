# The problem, whole

## 1. The game, and the one fact that shapes everything

Kaggriculture: two players, 30 days x 24 hours, a 10x10 farm each, $3,000 to
start. Five crops, three animals, land in four quadrants at $1k/$2k/$4k.

The score is RELATIVE -- final money against the opponent's -- and the market
is ONE SHARED INVENTORY. Price is an exact closed form on that inventory. So
every unit I sell lowers the price they get, and every unit I DON'T produce is
revenue THEY collect. The opponent's supply is worth $52,652 a game to us
through that single variable. This is not a farming optimisation problem with
an opponent attached; it is a two-player game over a shared pool where the
farm is the instrument.

Demand is the only randomness that matters. The town opens one shop on each of
days 3, 6, 9, 12, 15, 18, 21, 24, drawn uniformly from eight known types. Which
products have demand is unknown on day 0 and mostly known by day 9. MELON is in
no shop and has fixed demand. The products split by market shape: WHEAT and
EGG survive volume (annuities); STRAWBERRY, MILK, WOOL, MELON crash on it
(traps -- they pay if the draw lands, little if it does not).

## 2. Where we stand

Shipped: champ-h11, a decoded heuristic playbook (opening 12 melon + 7 wheat,
a herd ramp, strawberry block, three quadrants, hands to 11 by day 10) plus
four extraction rules. It wins about 0.44 of paired games on the 144-game
band-stratified instrument and is rated below the leaders on the ladder.

Its opening is fully committed by day 10 with $212 in the bank. From day 12 on
the money lead explains 0.55 -> 0.89 of the outcome: after the opening, the
game is mostly consequence.

## 3. What the leaders do that we do not

Three top players, 40 recorded games, measured:
  - They spend 80% of day-0 capital on LIVESTOCK, 20% on seed -- sheep on day
    0 in all 40 games at P(never a yarn shop) = 0.34. That is a paid option on
    a draw that has not happened.
  - They hold ~26% of TILES as animals but ~70% of their CAPITAL in them.
    Livestock is dense capital on few tiles.
  - They run 5-9 geese (we run 0), 7-18 carrots (we run 1), 13-24 strawberry
    (we run 41), and extract ~1,000 one-time units from the same farm where we
    extract 388.
  - Days 0-2 they agree with each other at TVD 0.075 -- essentially identical,
    because no information exists yet. The first draw on day 3 triples that.
  - They finish POSITIVE on margin; we finish -$15.3k.
  - They react. Their plans change on the draw and on the opponent's board.

## 4. Why everything tried so far has lost

Three laws, each measured many times:

**A. The constant is a sharp local optimum.** A 35-coordinate CEM over the
champion's own constants -- 98 candidates, 15,000 games, coupled mechanisms
sampled together -- found nothing (+0.000 on a fresh draw). Every hand-picked
lever: 11 arms, 2 won, both EXTRACTION (harvest every visit +4.1, shedguard
+2.5). Allocation and crop mix gate flat FOUR independent ways, including the
leaders' own daily mix faithfully delivered (-4.9) and the LP optimum (ties).
Geese, with all three reachability gates open, lose MONOTONE in the dose.

**B. Withdrawn supply is sold by the opponent.** The signature of every losing
arm: our bank flat or up, THEIR bank up more. The goose ladder is the clean
case -- their bank rises $1.1k -> $5.5k as our geese rise 1 -> 5. A demand-side
argument ("EGG is a great market") has been right about the market and wrong
about the game twice.

**C. Fitted models predict the data, not the effect of a change.** Behaviour
cloning at 69.4% agreement gated null. The market demand table, the value
head, capacity channels: each fit held-out data better and each failed the
paired gate. V-ranking is at chance out of sample. PBRS is provably a no-op
for every gate we run. The RL track's best board net is 0.249 against the
shipped 0.354 and BANKS THE MOST MONEY OF ANY AGENT -- it learned to farm, in
a game scored on winning.

And one law about the instrument: shop draws RE-ROLL on any board difference,
so a single trace cannot attribute. Only the paired mean can, and at 144 games
its standard error is ~0.045.

## 5. What is being built, and why it is not (A), (B) or (C) again

rlv3, maia2 OFF, the shipped scheduler underneath. A planner that:

  1. **Emits COMMITMENTS, not shares.** A 3-day schedule of dollars into seed /
     livestock / land / hands, and deltas per crop and animal. Dollars because
     they are what both sides of the choice actually compete for; deltas
     because timing is the live variable (an action is worth $1,317 on day 3
     and $103 on day 27) and shares are time-blind, which is why they gate
     flat. These translate one-to-one into plant_quota / animal_buys /
     structure_builds / hires -- a path proven this week by an arm that
     reproduced champ-h11 at +0.000 with zero discordant games.

  2. **Believes the prior on days 0-2 and reacts on days 3, 6, 9.** The
     opening's whole reactive surface is three decision points. The leaders'
     days 0-2 are a constant (0.075) and can be taken as given.

  3. **Self-corrects from its own residual.** The 3-day plan predicts our own
     board; the next dawn observes it. The gap splits into a DELIVERY gap
     (the executor did not seat what was asked -- shrink the ask) and a
     MARKET gap (a draw landed -- re-derive, do not shrink). Two bounded EWMAs
     that already exist (rlv3.objective.ProjState) carry it.

  4. **Turns risk into a crop choice.** When P(win) is low, take variance:
     traps and livestock BEFORE its shop opens (the lead-time option). When
     high, protect: annuities, livestock after confirmation. The dial is
     (alpha, upper) from the exact drain lattice -- CVaR on a distribution we
     can compute, not fit. Built and working: COW goes from rank 1 to rank 8
     on the same board when the posture flips.

  5. **Drops tier 3 of the objective.** Win rate, then margin when winning.
     NOT "minimise the loss margin" -- a player who is losing and minimises
     variance optimises the size of the defeat. This is the one place the
     design contradicts the record on purpose, and the record's own risk
     module argues for it.

  6. **Is verified against the TOP BOARDS before it touches an executor.**
     S0 passes when the plan agrees with the leaders conditioned on the draw
     -- not with champ-h11, whose ceiling we would inherit. Then risk is
     applied and agreement is re-checked in the high-variance regime.

  7. **Prices the opponent.** Their board is public. Their policy is the one
     quantity in this game that must be estimated rather than derived, and it
     is the only place a fitted model belongs.

Why this escapes the three laws: (A) it is a function of state, which a
constant cannot be; (B) it prices the shared pool explicitly and treats
"their bank up" as the failure it is; (C) its only fitted parts are the
opponent's policy and the risk posture -- low-dimensional, bounded, and
zero-dose bit-identical to the champion, so every step is a pure addition
that can be gated.

## 6. The constraints set on it

  - Targets MUST translate to board actions.
  - Constants come from the ground truth (engine tables, the replays), never
    assumed. Seed prices and market draws are the only constants; risk is
    FITTED.
  - Tests must not hit the executor until the planner is verified.
  - Reward depends on the PHASE of the game.
  - Self-correction per risk and per shop draw.
  - The model is free to choose the land day and the hire count.
  - Maia2 comes back LATER as a source of movement randomness, to test whether
    the planner really self-corrects when the executor stops being
    deterministic.

## 7. Where it stands today

**Established this week:**
  - The strategy is a searchable 35-vector; the champion is its local optimum.
  - Geese are closed properly (monotone dose-response, all gates open).
  - One reactive rule -- day 9, 3+ wheat shops -> sheep = 0 -- is positive on
    four independent draws (+0.076/+0.090 conditional; +0.035/+0.014 as a pure
    addition to champ-h11). Promising, not proven; needs ~1,000 games.
  - The leaders' invoice, the phase-1 boundary, the 3-day window (which
    collapses days 0-2 to 0.075), and the standing-stock pricing that
    reproduces their capital split within 0.03-0.10 with NOTHING fitted.
  - Every board net ever trained here was fitted to labels pinned to v1, a
    grid that names the leaders' true count on 37.8% of dawns and misses by
    2.16 tiles. v4 reaches 98.7% / 0.04 tiles. The corpus builder could not
    select a version; it can now. Paired v1/v4 corpora and readouts exist.

**Open, in order:**
  - Whether the fixed labels change what BC learns, in TILE space (the scorer
    is one function from done -- it needs the session replayer for anchors).
  - Whether the board trunk beats its own no-board control on the 995 draw.
  - Whether any of it approaches the SHIPPED champion, not champ-h8.
  - The trajectory: how fast to approach the priced standing stock under a
    $3,000 cash constraint. This is the actual job for a fitted model -- the
    destination does not need learning.
  - Eleven daily datasets (~7,000 episodes) never ingested. External corpora
    are the only thing that has ever produced a real jump here.

## 8. The bet, said plainly

The leaders' edge is partly extraction (which we have set aside) and partly
reaction (which we are building). Allocation itself is worth ~0 -- that is
measured four ways. So the wager is that REACTING to the draw and to the
opponent, with the right risk posture, is where the points are, and that it
can be built as a pure addition to a champion that is otherwise left alone.
The one reactive rule found so far is the only evidence for it, and it is
weak. That is the honest state of the bet.
