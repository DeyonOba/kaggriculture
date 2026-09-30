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
