# Executor v2 — a scheduler that accepts a plan and executes it

*2026-09-30. Written after the opening-race work: h11 delivers 49% of its own asks on days 0–8, the leaders do 1.4–1.6× our productive ops, and every "do more, earlier" change on the current executor (crew, order, structures) gated −12 to −14 with the opponent's bank up. The executor is the object; this is its redesign.*

## 1. What the record says the executor must be

- **It must take a plan indexed by day, turn and shop draw**, not a set of daily quotas re-derived from ratios over the open board. Today `DayPlan` is quotas (`plant_quota`, `animal_buys`, `hires_target`) consumed greedily; nothing says *when*, nothing is approved, nothing is reserved. The ledger shows the consequence: asks of $2,652 against $269 in the bank, seeds rationed to 60% of cash, 7 tiles asked on a $1 dawn.
- **Purchases are approved, not rationed.** If the plan is approved, the seeds for it are bought — the cash is reserved at approval. The leaders buy in lots as they plant and sell wheat during the day; the plan can say so.
- **Deviation has three triggers and only three**: risk posture (P(win) / CVaR), the selling price of a crop (a crop whose price falls under its floor loses its remaining plant jobs), and capital (cash under the reserve defers purchases). Everything else is executed as planned. Shop draws land at dawn, so the daily re-plan absorbs them; the 3-day horizon is re-emitted each dawn from the realised board (the self-correction we already have).
- **Assignment must commit.** The one executor experiment that ever won self-play was commitment (+$3,233) and the one that lost most was optimal top-matching (−$3,636): a unit that keeps its job beats a unit that is re-optimised every turn. Churn is the failure mode the multi-robot literature names.
- **Routing is cheap here.** Manhattan grid, one step per turn, the shed at the centre; the cost of a job for a unit is travel + a fetch trip if it needs wheat/fertiliser + lateness against its window. That is an assignment problem with time windows, solved per turn — Hungarian over units × jobs with a commitment bonus, insertion for the next job of a unit that finishes.
- **The acceptance test is delivery, then the gate.** `tools/plan_ledger.py` delivery % (asked vs delivered per day) is the offline metric; the work ledger's productive share (leaders 35–45%, us 25–31%) the second; the 144-world paired gate the verdict — with the opponent's bank read alongside ours, because the last week says more supply from our farm is a donation unless it out-runs their share.

## 2. The plan the executor accepts

```
Plan (3 dawns, re-emitted daily)
  jobs:      Job(id, day, window=(t0, t1), kind, x, y, op, need, qty, value$, pri)
             kinds: PLANT/BUILD/PLACE (growth, from the plan) and WATER/HARVEST/FEED/CARE/FERTILIZE (derived from the board by the engine rules we keep)
  purchases: Purchase(day, turn_due, item, n, cost$)  — seeds for the plant jobs, animals, hires; approved at emission
  reserve:   cash held for approved purchases + tomorrow's hire bill
  triggers:  price floors per crop, the risk posture, the capital reserve
```
The first plan source is an **adapter** from today's `DayPlan` (so the champion planner and the readout keep working); the board model of Stage 1' then emits jobs directly.

## 3. The executor

1. **Derived jobs** come from `scheduler.build_tasks` — the engine-rule knowledge (when water adds yield, the fertiliser windows, the harvest-before-last-water rule, the four shipped flags) is kept as is.
2. **Assignment**: per turn, cost(unit, job) = travel + fetch (if `need` and the unit lacks it: shed round trip) + priority weight + lateness; a unit's *committed* job gets a bonus so it is not re-assigned unless a strictly better job appears (hysteresis); `scipy.optimize.linear_sum_assignment` over units × jobs.
3. **Execution**: at the tile → the op; otherwise one step toward it; claims prevent two units on one job; plant/build/place quotas and seed stock are enforced at execution as today.
4. **Purchases**: approved purchases are placed when due and when the reserve is met; seeds for the plan are bought in full (no 60% cap); animals as cash arrives; hires at hour 0.
5. **Logistics**: drop at the shed when the pocket is full or the day ends; fetch wheat/fertiliser for jobs that need them — the shipped shuttle rules, unchanged.
6. **Triggers**: price (drop remaining plant jobs of a crop under its floor), capital (defer purchases, never the hire bill), risk (the posture from `kagent/risk.py`, later).

## 4. Tests and order

1. `EXEC["executor2"]` switch on the champion plan: ledger delivery % on 4 worlds vs the shipped executor (must rise), work-ledger productive share (must rise), then the paired gate vs h11 with banks. The plan is h11's own, so this isolates the executor.
2. The same switch under the readout plan (the leaders' asks): does the new executor deliver *their* plan (the question that closed the readout line)?
3. Then the plan source: the board model emits jobs (Stage 1'), then Stage 2 (sell timing) on top.

## Sources
- Halite 2020 1st place (explicit planning, task scoring, non-conflicting assignment, opponent modelling — no end-to-end RL): https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio
- Kore 2022 top solutions (rule-based, forward simulation, time-indexed ledgers, route scoring in staged phases): https://www.kaggle.com/c/kore-2022-beta/discussion/313989 , https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio
- Lux AI S1 rule-based agents (multi-pass action scoring, A* with weights, clash re-runs): https://www.kaggle.com/c/lux-ai-2021/discussion/293953
- Hungarian / linear assignment for multi-robot task allocation: https://arxiv.org/pdf/1805.08712 ; auction-consensus with time windows (CBBA, ICBGA, GACA): https://cdn.aaai.org/ojs/9440/9440-13-12968-1-2-20201228.pdf , https://arxiv.org/abs/2608.15884 ; systematic review of MRTA: https://dl.acm.org/doi/10.1145/3700591
- Receding-horizon multi-agent scheduling and commitment (schedule tasks to a horizon, commit within a window, re-order to absorb delays): https://arxiv.org/html/2004.02347 , https://lacuna.tiptreesystems.com/work/receding-horizon-re-ordering-of-multi-agent-execution-schedules/wrk_b743757d43ada0c29034d5ff803d2c0d
- VRPTW heuristics (Solomon insertion, iterated local search, multi-trip with time windows): https://www.researchgate.net/publication/220371726_Heuristic_methods_for_vehicle_routing_problem_with_time_windows , https://doi.org/10.3390/math12111712

## 5. Implementation log

**Slice 1 (2026-09-30), dispatch: REFUTED.** `kagent/executor2.py` (EXEC flag `executor2`) replaced `scheduler.assign` with a min-cost assignment (scipy Hungarian, greedy fallback) over the same `build_tasks` jobs: commitment hysteresis (a unit keeps its job unless a strictly better one appears), the tile underfoot first, farm-wide morning provisioning at hours 0-1, animal-first, fetch trips priced into the cost, nearest-first weights (PRI_STEP 0.5 -- priority-first weights walked hands: 2.29 moves per productive op). Offline it reached parity with the zone sweep on the 4-world work ledger (productive share 38% vs 37%, 1.39 vs 1.38 moves per productive op) and did not raise delivery % (45/37/50/49 vs 49/35/58/55). The paired gate (`rl/runs/exec2_gate.json`, 144 worlds, h11 reused): **-0.181, z -3.94, p 0.000, 11 arm-only vs 37 base-only wins, 5/6 bands; our bank -$912, the opponent's +$6,040.** Trace over 4 worlds: feeds -16%, cares -27%, pasture harvests -21%, milk sold -37%, wool -36%. The zones of the sweep keep each hand beside its pastures; a global matcher spreads the crew and the animal line pays. Third refutation of dispatch (optimal top-matching -$3,636; commitment-only null). Kept as a measured negative; `executor2` stays off.

**Consequence for the design.** Sections 2-3 stand, but the dispatch is the shipped one. Executor v2's content is plan-side, and each piece goes on the shipped sweep as its own flag, gated alone: `approve` (seeds for the quota bought in full above the cash floor -- ledger: the herd grows 3 instead of 4 on days 2-8 when seeds take the cash first, so this is capital allocation and it gets its own gate), replant-by-turn (a one-time tile is replanted the turn it is harvested, seeds for the expected harvests bought at dawn: the champion plan's standing-count quota cannot replant on the harvest day before day 10 because the ripe tiles still count as standing), the hire cap by workload, then the three triggers.

**Slice 2 (2026-09-30), replant-by-turn: INERT.** `scheduler.replant_quota` (EXEC flag `replant`) adds today's one-time harvests back to the quota, same crop, capped by the ground the plan's own asks leave bare. On worlds 0-1 it never adds: the champion quota exceeds the bare ground at every dawn (day 7: bare 17, STRAWBERRY quota 28; day 11: bare 26, quota 34), so the room is zero and world 1 plays the identical game. The plan already over-asks the ground; what it lacks on days 5-9 is cash ($29-$269 dawns against $2,000-3,000 asks) and on day 11 labour and order flow (26 wheat asked on $4,943, 16 planted). A plan indexed by turn cannot conjure either; the plan must be SIZED to them (the approval = a capital check that prunes the ask), which is what the `approve` gate measures next.

**Slice 3 (2026-09-30), approved purchases: NULL, money up.** `approve` alone on the shipped dispatch (seeds for the whole quota bought above the cash floor, no 60%-of-cash ration; `rl/runs/approve_gate.json`, 144 worlds, h11 reused): **+0.007 (0.465 vs 0.458), 9 arm-only vs 8 base-only, p 1.000; our bank +$2,123, the opponent's +$869; bands +0.250 / +0.042 / -0.042 / -0.042 / -0.083 / -0.083.** The first executor-side flag of the week whose bank signature is ours up more than theirs, and it still buys no wins: the seeds are bought, the plants go in, and the games are decided elsewhere. Kept off (a null does not ship); it is the natural default for a plan that is sized to cash, which is the next question -- the champion plan asks $2,000-3,000 on $29-$269 dawns, so "approved" today means "as much of the ask as the cash covers", not a plan the executor can complete.

## 6. The plan sizing rule (designed with the user, 2026-09-30)

**What the leaders' games say about sizing** (`size_ledger`: the 40 winner seats of `replays/now`, h11 on tape worlds 0-3 beside them):

| days | dawn $ (them / us) | spent $ | of which feed | animals kept | plant tiles | bare tiles | income $ / day |
|---|---|---|---|---|---|---|---|
| 0 | 3,000 / 3,000 | 3,590 / 3,098 | 623 / 264 | 4.6 hands vs 3.0 | 0 / 0 | 25 / 25 | 596 / 106 |
| 1-2 | 66 / 16 | 1,301 / 349 | 642 / 274 | 5.0 / 4.0 | 19.1 / 13.4 | 0.9 / 7.4 | 1,317 / 359 |
| 3-6 | 114 / 124 | 3,189 / 1,117 | 926 / 161 | 7.6 / 4.4 | 17.2 / 18.6 | 0.2 / 1.0 | 3,203 / 1,128 |
| 7-10 | 320 / 236 | 7,168 / 2,523 | 2,109 / 635 | 16.0 / 10.1 | 42.9 / 31.2 | 2.4 / 8.1 | 8,707 / 3,960 |
| 11-15 | 15,754 / 11,486 | 2,124 / 1,349 | 1,354 / 666 | 21.7 / 15.3 | 62.5 / 52.6 | 2.4 / 6.8 | 7,250 / 4,887 |

- Nobody sizes to dawn cash: both sides wake broke through day 10 and spend 20-28x the dawn cash, funded by the day's sales, and end the day near $0 (evening cash $6-127). The budget is dawn cash plus what will sell today.
- The leaders' first claim is the herd's feed (30% of spend, 4-6x ours); they carry 7.6 animals on days 3-6 to our 4.4 and their day-0 income is collected fertiliser (~$99 a unit).
- They never leave ground bare (0.2-2.4 tiles); we hold 7-8 bare tiles from day 7 to 15 because the plan asks for $100 strawberry seed on $29 dawns and the tile waits.

**The four choices (user's):** budget = dawn cash + expected sales; a bare tile whose planned crop is unaffordable is planted with the best affordable crop now; hires by workload (work / capacity); feed at any price as the first claim.

**The rule** (`kagent/sizing.py`, `approve_plan` at dawn, `refill` every hour):
1. budget = dawn cash + forecast, where forecast = max(model, 0.8 x the best of the last three days' realised income); the model prices the dawn shed and today's one-time harvests at the exact market identity.
2. claims in order: feed for the whole herd at the market price (`plan.feed_any` lifts the $120 cap); hands = ceil((standing tiles + today's plantings + 3 per animal) / CAP 8) - 1, held to half the budget by the wage curve; the cheapest open crop (wheat, then carrot) on every bare tile; animals, and only the structures the approved animals will fill; the plan's own crops with what is left, in the plan's order; land last.
3. the approved plan buys in full (`plan.approved`): the dawn sells go first in the order list (engine fact, verified in the env: orders resolve in list order with the cash updated between them, so a $4 dawn hires on its own wheat), hires spend what cash allows, seeds come before animals and land.
4. ground freed during the day is asked for again with the fill crop each hour (`refill`), so a harvest the dawn could not see is replanted the same day.

**Ledger before the gate** (world 0, days 0-14): delivery 84% of asks vs 57%; four worlds: bare tiles days 7-10 5.8 vs 8.1, herd days 3-6 4.9 vs 4.4, dusk cash still ~$1,000 on days 3-6 (the forecast floor lags income that doubles day to day). Shipped path bit-identical (world 0 ends at $87,073 either way).

**Gates**: `sized` (the package) and each of the four flags alone vs h11 -- `rl/runs/size_*_gate.json`.

**Sizing gates (2026-09-30, 144 paired worlds each, h11 reused; `rl/runs/size_*_gate.json`):**

| arm | win | vs h11 | z | discordant | our bank | opp bank | bands (0-600 … 2500+) |
|---|---|---|---|---|---|---|---|
| h11 | 0.458 | | | | $90,532 | $89,828 | |
| sized (all four) | 0.250 | −0.208 | −4.74 | 8 vs 38 | −$1,046 | +$4,066 | −.21 −.50 −.25 +.04 −.17 −.17 |
| size_ground (fill) | 0.222 | −0.236 | −5.50 | 6 vs 40 | −$2,001 | +$10,801 | −.13 −.46 −.33 −.04 −.17 −.29 |
| size_feed | 0.458 | +0.000 | 0.00 | 1 vs 1 | +$267 | +$280 | all 0 |
| size_hires | 0.201 | −0.257 | −6.57 | 2 vs 39 | −$4,993 | +$12,183 | −.29 −.13 −.38 −.13 −.38 −.25 |
| size_budget | 0.382 | −0.076 | −1.88 | 12 vs 23 | −$484 | +$1,597 | +.04 +.04 −.13 −.04 −.33 −.04 |

- **The fill donates.** Over four worlds the fill arm sells 779 strawberry units to the shipped 1,540 and 1,933 wheat to 1,350: wheat on the bare tile is still standing when the strawberry money arrives, so the tile never becomes the strawberry tile it would have been three days later; half our strawberry supply leaves the shared market and the opponent's bank rises $10.8k. Volume knobs lose from the timing side too: the cheap crop costs the tile's next ten days.
- **Hires by workload starves the crew.** Alone, the wage cap is half of dawn cash (no forecast without size_budget), so days 1-10 run on 2 hands; in the package the forecast lifts it to 7.4-7.9 hands on days 7-10 against the shipped 8.8 and it still loses. Hands are productive at the margin (the earlier hire-cap bound said the same); CHAMP_HIRES is not over-hiring.
- **Feed at any price is a no-op**: the $120 cap almost never binds (1 discordant game).
- **The budget cut alone is a small loss** (−0.076, p 0.09, opp +$1.6k): approving against dawn cash plus a lagging forecast asks less than the shipped rationing does on the days income doubles, and the animals-before-crops order does not recover it.

**Verdict.** Plan sizing as a cut is CLOSED: on this planner every reduction of the ask loses and every substitution donates. The plan's asks are already the right size for its executor; what the ledger called "undelivered asks" is the market rationing them across the day, and that rationing beats a dawn approval because income arrives during the day. What survives from the design is the engine fact (sells first fund the dawn) and the measurement tools (size_ledger, yield_ledger). The yield ledger then moved the question: per standing tile and animal we deliver yield at or above the leaders; the farm is 30-45% smaller (the opening herd) and our CARE rate collapses after day 13 (the crash-pivot) -- `care_price` gate next.
