# Critique of the current agent, and what the flow model found

Branch `arch-flow`. Written after every parameter in the hand-written planner had been measured
against real opponents and found flat (see CHANGELOG). This is the research pass: reread the
engine, attack our own assumptions, keep what survives.

## Where our approach was wrong

**1. We modelled the market as a stock. It is a flow.**

Earlier analysis computed "season sell capacity" by walking each product's price ladder down from
`I0` and concluded that wool caps near $8k a season, milk near $6.3k, strawberry near $3.9k, and
that our herd therefore sat on "the two steepest ladders in the game". That was wrong, and it
drove several rounds of work.

Market inventory only ever falls through town consumption, and that consumption is a **rate**:
shops consume at `step % 4 == 0` (a single-product shop takes 2 of its product, a multi-product
shop 1 of each it lists) and the town centre takes 1 of everything at `step % 24 == 0`. So each
product has a daily drain set by the shop draw, shared between the two farms. Sell under it and
inventory stays low and the price stays *above* base; only sustained over-supply walks the ladder
down. `tools/flow_audit.py` measures it.

**2. The market is under-supplied, not saturated.** Measured over 3 games, champ-h8 against a
2500+ agent (units per day):

| product | drain | ours | theirs | combined vs drain | realized price |
| --- | --- | --- | --- | --- | --- |
| wheat | 35.0 | 5.8 | 13.1 | **−16.1** | **1.79× base** |
| egg | 17.0 | 0.0 | 1.7 | −15.3 | — |
| tomato | 13.0 | 0.0 | 0.9 | −12.1 | — |
| carrot | 15.0 | 1.2 | 2.9 | −10.9 | 1.34× base |
| strawberry | 23.0 | 7.8 | 8.2 | −6.9 | 0.96× |
| wool | 17.0 | 4.8 | 6.0 | −6.2 | 0.54× |
| milk | 15.0 | 5.7 | 6.6 | −2.7 | 0.73× |
| melon | 1.0 | 2.4 | 2.4 | **+3.8** | 0.62× |
| fertilizer | **0.0** | 7.5 | 11.6 | **+19.1** | 0.49× |

Both farms together leave most of the town's demand unmet. Wheat and carrot trade *above* base
because they are scarce. Only two products are genuinely over-supplied, and both have almost no
drain: **fertilizer has none at all** (no shop lists it, it is not a town-centre product) and
**melon has 1/day** (town centre only — no shop lists melon either). Those two are mined, not
farmed: their price never recovers, so the only question is when to take the stock.

**3. Ranking production by revenue per worker-turn is the wrong objective.** Doing it properly,
and pricing in the scarcity each product actually realizes, says wheat is the best sustainable
line at ~$45 per worker-turn and wool the worst at ~$23 — and we run 8–10 sheep while selling 5.8
wheat a day into a 35/day drain. Three configurations were built on that thesis and **all lose**:

| config | change | pooled over 72 paired worlds |
| --- | --- | --- |
| flowA | sheep 4, cows 6, wheat 40 | −$5,619 ± 2,011 |
| flowB | + carrots 12, tomatoes 10 | −$3,612 ± 1,727 |
| flowC | sheep 2, cows 4, wheat 40, carrots 14, tomatoes 12 | −$6,356 ± 2,220 |

## What the failure taught us — work density

The prices moved exactly as the flow model predicted (wool 0.54 → **0.98** of base, strawberry
0.96 → 1.20, fertilizer 0.49 → 0.69). The model is right about the market. It is the *conversion*
that fails: freeing labour did not produce more goods.

| | champ-h8 | flowC |
| --- | --- | --- |
| crop tiles (days 10–24) | 50.9 | 66.8 |
| structures | 16.2 | 7.0 |
| empty tiles | 4.4 | 0.9 |
| MOVE share of unit-turns | 57.5% | **62.7%** |
| productive share | 41.7% | **36.6%** |

Trading pastures for crop tiles made the farm *less* productive per turn. The reason the
$/worker-turn table missed: **it ignored travel.** An animal delivers FEED, CARE,
COLLECT_FERTILIZER and HARVEST — four actions at one location. A crop tile delivers water and
harvest, two actions, and you must walk to the next tile. Animals are labour-efficient not because
of their revenue per action but because their chores **cluster in space**.

That reframes the leaders' advantage. They hold 17 animals to our 14 and place their pastures 1.76
tiles from the shed against our 2.31, so their highest-density work sits closest to the wheat they
must carry to it every day. Their 42.2% MOVE share against our 57.5% is the visible result.

## The standing open question

Every lever moves the farm along a curve where extra output costs extra travel, and the two cancel.
Fewer animals, more animals, more land, more hands, different crop mix, different assignment rule,
different delivery rule, joint combinations of all of them — all measured, all flat or negative.
The one quantity that separates us from the leaders and that no lever has moved is **actions per
tile visited** (1.65 against their 1.86).

The promising direction this branch identifies is therefore not *what* to produce but *how densely
work is laid out*: co-locating high-frequency chores so a worker extracts more actions per stop.
Reserving shed-adjacent ground for pastures was the naive version and cost $11k because it starved
planting. The version worth trying next places the herd and its feed route together **from day 0**,
when every tile is still free, rather than reserving tiles later against a farm that is already
committed.

## Tools added

- `tools/flow_audit.py` — drain rate, both farms' supply rates, realized price, over/under-supply.
- `tools/waste_audit.py` — tile lifecycle, deaths by cause, spatial layout of both seats.
- `tools/exec_gate.py` — the only gate that counts: paired against external agents, replicated.
