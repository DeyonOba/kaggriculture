# The whole-agent scorecard

`tools/agent_scorecard.py` characterises an entire agent on four dimensions
at once and puts it beside M & M & P & Q, Boey and DSM.

On 2026-09-26 eleven rule variants were judged on one number — paired win
rate — and all but one were refuted. A win rate says an arm lost. It does
not say whether the planner, the selling, the demand coverage or the
executor is the part holding the agent back, so every refutation cost a
full evaluation and taught nothing about where to look next. This tool
answers that question before an evaluation is spent.

    PYTHONPATH=. uv run python tools/agent_scorecard.py \
        --agent bundle:arena/agents/champ-h10/main.py \
        --eval rl/runs/draw995/batch2.json:ship2_hv \
        --games 6 --h2h 3 --out docs/results/scorecard_champ_h10.json

`--agent` takes `champion`, `knobs:<rules.json>`, `bundle:<main.py>`,
`weights:<to_layers.json>` or `execnet:<pt>[:planner[:conf[:mode]]]`.
`--reference` defaults to champ-h10. Nothing re-runs a paired evaluation:
`--eval` attaches one that already exists under `rl/runs/draw995/`.

## What it measures, and where the leaders sit

**OUTCOME** — win rate overall and per rating band from the attached paired
evaluation, plus money and margin from the traces it plays. The leaders have
no win-rate column: theirs is measured on their own games against their own
opponents, not on our instrument, so only money and margin are comparable.

**PLANNER** — crop and herd counts at days 6, 12 and 20, peak hands,
quadrants owned and the day each was bought, idle share of unlocked tiles at
dawn, weed tile-days. Leader snapshots come from four replays per team,
cached in `docs/results/top3_planner.json`.

**MARKET / DEMAND** — units our SELL orders offered, the town's realised
drain (`econ.town_drain_per_day` against the shops actually open each day,
summed), the share of that demand we covered and the share the opponent
covered, realised price over the game-mean price, the share of units offered
into a price already below 0.35 of base, and the within-step price impact of
our orders on the steep-curve products. Volume is an *offer*: the engine
fills only as far as the shed allows. The leaders are measured the same way,
so the comparison holds even though the absolute number is an upper bound.

**EXECUTOR** — moves per productive op, chores per tile visit split between
animal and crop tiles, pickups per feed, feeds from wheat already in hand,
shed ops per unit-day, harvests, plantings, fertilise ops, one-time units
harvested, units per harvest for wheat and carrot, replant-within-one-step,
hour-23 moves and wasted waters.

## Reading the gap line

Each group closes with the share of the reference-to-leader distance the
agent has covered: 0.00 is level with the reference, 1.00 is level with the
best leader on each metric. Metrics with a direction (fewer moves is better,
more chores per visit is better) use the best leader as the target. Metrics
without one — a crop mix, hands, the day a quadrant is bought — use the
leaders' *mean* as a target to approach from either side, because 41
strawberry tiles against their 20 is a difference, not an achievement.

A scorecard for the reference agent itself would score zero everywhere, so
in that case the anchor falls back to the cached dev-tree row and the tool
says so. A second block reports the metrics furthest from the leaders' mean
regardless of anchor, which always says something.

`--h2h N` plays the agent directly against the reference on N seeds, both
seats each. Run against itself it returns 3/6 and a margin of exactly zero,
which is the calibration check.

## What the champ-h10 scorecard says

Full output in `docs/results/scorecard_champ_h10.txt`.

| | champ-h10 | MMPQ | Boey | DSM |
|---|---|---|---|---|
| margin | −15,286 | −2,282 | +1,396 | +4,690 |
| idle share of tiles | 0.124 | 0.065 | 0.040 | 0.037 |
| day-20 strawberry | 41.2 | 22.5 | 23.5 | 12.8 |
| day-20 carrot | 1.0 | 6.7 | 12.3 | 17.5 |
| day-20 geese | 0 | 6.3 | 5.3 | 8.8 |
| one-time units harvested | 388 | 888 | 1,043 | 1,077 |
| plantings | 150 | 266 | 276 | 271 |
| fertilise ops | 124 | 220 | 208 | 223 |
| pickups per feed | 1.08 | 0.55 | 0.46 | 0.48 |
| evening share of sales | 0.09 | 0.31 | 0.22 | 0.29 |
| wool order impact | 0.53 | 0.90 | 0.76 | 0.70 |

**The limiting dimension is the planner, not the executor.** Against the
dev-tree anchor the executor has closed +0.16 of the distance to the leaders
— that is this morning's three rules — while the planner sits at −0.10,
slightly *further* away, because harvesting every visit leaves more spent
plants and weed tile-days rose from 36 to 45.

The planner numbers say what is wrong in one line: we put 41 tiles under
strawberry where the leaders run 13 to 24, one tile under carrot where they
run 7 to 18, and none under geese where they run 5 to 9. That concentration
is why one-time units come to 388 against their 888 to 1,077 — a third of
their volume off a farm the same size — and why 12% of our ground sits idle
at dawn against their 4%. Our plantings are 150 to their 271.

Two of those three crops have already been tested as isolated rules and
refuted: geese lost 16.5 points, and every hauling rule lost. The scorecard
says why that is not a contradiction. The gap is not a missing crop, it is
the *whole allocation*, and moving one crop at a time inside an otherwise
strawberry-heavy plan pays the cost of the change without the benefit of the
mix. The next thing to test is the allocation itself.

The market group is the second story: we realise a competitive price
(1.010 against 0.955–1.064) but sell 9% of units in the evening against
their 22–31%, and our wool orders cut the price to 0.53 of where they
started against their 0.70–0.90. We cover 34% of the town's demand while
the recorded opponent covers 70%.
