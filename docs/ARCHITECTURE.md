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

## Committing the herd and feed route on day 0 — tested and refuted

Implemented as a **construction rule**, not a reservation: crops colonise outward from the farm
edge, so the shed-adjacent centre is claimed last by plants and first by pastures. No tile is ever
withheld, which is what made the earlier reservation cost $11,334.

**The mechanism worked.** Tracked by day, pastures moved toward the shed and the farm carried
*more* crop tiles, not fewer:

| day | pastures, champ-h8 | pastures, outward | crop tiles, champ-h8 | crop tiles, outward |
| --- | --- | --- | --- | --- |
| 6 | 2.43 | **1.86** | 17 | 16 |
| 12 | 2.40 | **2.13** | 38 | **49** |
| 20 | 2.44 | **2.19** | 56 | **58** |

(The season-average pasture distance barely moves, 2.31 → 2.30, because most animal-days are late.
Reading only the average hides the effect entirely — worth remembering.)

**And it loses $10,020 ± 2,948** over 72 paired worlds, at every cut-off day tested (12, 20, 29 —
the last two are identical agents, since the herd stops growing well before day 20).

**Why, arithmetically.** The near ground belongs to whichever tile class absorbs the most **visits
in total**, and that is crops, despite each pasture being visited twice as often:

| class | tiles | visits/tile/day | total daily visits |
| --- | --- | --- | --- |
| crops | 58 | 2 (water, harvest) | **116** |
| pastures | 18 | 4 (feed, care, collect, harvest) | 72 |

Moving 58 crop tiles +0.20 further to bring 18 pasture tiles −0.27 closer is a net **+3.8
tile-steps of travel per day**. The trade is backwards.

**So champ-h8's existing nearest-first planting is already the right layout**, and the work-density
thesis is refuted in this form. The leaders' pastures sitting 1.76 tiles from the shed against our
2.31 is a *correlate* of their play, not a cause of their advantage — most likely a by-product of
holding 17 animals on a farm whose crop count is lower, rather than a deliberate layout policy we
failed to copy.

## Phase 1: widening the executor — the strongest mechanism yet, and it still loses

The plan was to widen the executor's decision space before building any explore/exploit machinery,
with a stated bar: **actions per tile visited must clear 1.75** (from 1.65, against the leaders'
1.86), or stop and do not build Phase 2.

**The diagnosis was the sharpest of the project.** Instrumenting every worker-turn against its own
task list: **39.7% of all moves are a worker walking away from a tile that still had a task it
could do** — 1,565 times a game. WATER 1,382, FEED 821, COLLECT_FERTILIZER 355 over two games.
The cause is structural: the assignment passes filter by priority band, so a P_FILLER task under a
worker's feet is invisible to the pass that is choosing a P_TEND task across the farm, and the
worker leaves. At distance 0 a task costs one turn and yields one action, while walking costs the
same turn and yields nothing, so this looked like free money.

**The fix worked mechanically.** A same-tile chaining pass, run before the zone rounds, takes any
unclaimed doable task underfoot:

| | champ-h8 | chained |
| --- | --- | --- |
| walked away with work available | 3,130 (39.7% of moves) | **795 (10.9%)** |
| productive actions (2 games) | 5,709 | **6,229 (+9%)** |
| moves | 7,875 | **7,314 (−561)** |
| actions per visit | 1.69 | 1.57 |

**And it loses money at every threshold**, over 72 paired worlds each:

| chain up to | pooled |
| --- | --- |
| P_TEND (valuable work only) | −$6,414 ± 2,512 |
| P_GROW | −$9,047 ± 3,808 |
| P_FILLER (everything) | −$6,385 ± 2,390 |

**The finding.** Doing 9% more work and walking 7% less makes the agent poorer. Crucially the
*restrictive* variant loses too: chaining only survival, harvest, animal and tend work — no filler
at all — is −$6,414. So this is not "filler is worthless". Finishing local work delays the
farm-wide urgent work by a turn each time, and in this game **breadth beats depth**: serving the
most urgent task everywhere is worth more than finishing the tile you are standing on. The greedy
priority-first ordering we have was right for a reason we had not articulated.

It also kills the actions-per-visit metric as a target. It *fell* to 1.57 while total work rose,
because more short visits is a perfectly good way to do more work. The leaders' 1.86 is most
likely a by-product of their tile mix — an animal tile offers four actions in one place — not a
routing policy we failed to copy, exactly as the pasture-distance correlate turned out to be.

**Phase 2 is not built.** The stated bar was not met, and the widened space produced a worse agent
rather than a richer one, so a contextual bandit over it would be searching a space we have
evidence is harmful.

## Tools added

- `tools/flow_audit.py` — drain rate, both farms' supply rates, realized price, over/under-supply.
- `tools/waste_audit.py` — tile lifecycle, deaths by cause, spatial layout of both seats.
- `tools/exec_gate.py` — the only gate that counts: paired against external agents, replicated.
