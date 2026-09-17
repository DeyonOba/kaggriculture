# What is wrong with our executor

Everything below is measured, with the tool that measured it. Written 2026-09-17.

The executor is `kagent/scheduler.py` — `build_tasks` proposes work, `assign`
hands units to it. The planner (`kagent/planner.py`) sets quotas and targets.

---

## 0. The headline number

**Moves per productive action.** One number separates us from everyone good:

| agent | moves/action | source |
| --- | --- | --- |
| M & M & P & Q (reactive, rated 3,093) | **0.71** | `tools/replay_probe.py --dir replays/mmpq` |
| pub-ahmed-v38 | 0.84 | `tools/replay_probe.py --agent champ-h8` (seat 1) |
| a replay tape | 0.85 | recorded |
| **champ-h8** | **1.45** | `tools/replay_probe.py --agent champ-h8` |

It decomposes into two independent deficits (per worker-day):

| | tiles/wd | actions | moves | **actions/tile** | **moves/tile** |
| --- | --- | --- | --- | --- | --- |
| M & M & P & Q | 6.26 | 12.48 | 8.95 | **2.10** | **1.43** |
| pub-ahmed-v38 | 6.01 | 11.50 | 9.70 | 2.02 | 1.61 |
| champ-h8 | 5.59 | 9.26 | 13.39 | **1.67** | **2.39** |

We **under-chain** (1.67 actions per tile arrival vs 2.10) *and* **over-travel**
(2.39 moves between tiles vs 1.43), inside a neighbourhood of the same size
(worker-day span 6.13 vs 6.32).

---

## 1. It walks away from work under its own feet

**73.6% of our moves leave a doable task on the tile the worker is standing on**
— 2,846 moves a game, abandoning 4,110 tasks. Against M & M & P & Q's recorded
states, on 94.9% of the turns where they work and we would move, our nearest
task is at **distance 0** (mean 0.05 tiles).

Measured by `tools/decision_diff.py` (validated: replaying champ-h8 against its
own recorded actions gives 6,897/6,897 = 100.0% exact match).

### 1a. Cause: `assign` is BAND-MAJOR

`kagent/scheduler.py::pick` scores `(urgent, d, task.pri)` — priority bands are
served in whole passes and **distance is only a tie-break inside a band**. A
worker gets sent across its zone for a `P_GROW` task while a `P_FILLER` task sits
underfoot.

### 1b. Cause: the shed shuttle preempts every work pass

`assign` section 1 ("Liquidation / heavy-load shuttles") runs *before* any task
pass and calls `free.discard(idx)`. On the disputed turns our unit is carrying
goods **81.7%** of the time and mid-trip to the shed **76.9%** of the time. A
unit standing on a ripe crop with 3 items in hand is sent to the shed.

There is a trap inside it: `threshold = 1 if aggressive ... else DROP_THRESHOLD`
(10), and `if view.day < 10: threshold = min(threshold, EARLY_DROP_THRESHOLD)`
(2). **On day 10 — melon day — the threshold quadruples** and the crew starts
hoarding on the one day speed decides price.

---

## 2. It re-decides every turn, so workers oscillate

`pick` is called fresh each turn and re-chooses as tasks get claimed, so a unit's
target shifts mid-walk and it zig-zags. That is the 2.39 moves/tile: same span,
same tile count, 67% more travel. No variant we built commits a worker to a
target across turns.

---

## 3. Claims are resolved in UNIT-INDEX order

All three passes read `for idx in sorted(free)`, so unit 0 takes the task nearest
*unit 0* even when unit 5 is standing on it, and the claim is irrevocable
(`claimed` is a set). That is arbitrary-order greedy, strictly worse than
matching in theory.

Fixed by `tools/make_match_variant.py` (global-greedy matching over the identical
candidate set). Gated **+$1,381 ± 641 over 360 games** — and took head-to-head
wins from **5/144 to 4/144**. Shipped as `match-all`, rated **819.0** vs
champ-h8's 810.4, i.e. inside the ±45 rating noise.

---

## 4. It is co-adapted to our own planner's layout

Placed on M & M & P & Q's farm at their unit positions, champ-h8 would work on
only **12.4%** of unit-turns against its own 40.5%. Their farm is not task-poor —
our own `build_tasks` finds *more* tasks there than on ours (98 vs 66 at day 10).
The zone partition (`idx % n_zones`) assigns a unit only its own zone's tasks, so
foreign unit placements strand it.

Consequence: **planner and executor cannot be fixed independently**, which is the
mechanical reason every executor change in isolation gates flat.

---

## 5. Hiring is a ceiling that state can only lower

```python
schedule = CHAMP_HIRES[day] + (knobs["hires_delta"] if day >= 6 else 0)
plan.hires_target = min(schedule, max(3, workload // 3))
while plan.hires_target > 2 and econ.hire_total_cost(...) > budget: -= 1
```

`CHAMP_HIRES` is 30 hardcoded constants. The workload and cash gates only ever
*reduce* it — no state can push the crew above the schedule. The engine wipes
`farm["hands"] = []` every night (kaggriculture.py:880) and `_hire_cost(n) =
mult * fib(n)` resets daily, so the crew is a pure daily purchase with a
Fibonacci marginal cost.

`tools/rollout_hires.py` searching this per-day schedule produced
**+$3,287 ± 1,528 over 192 games** — the only time in this project that compute
converted into a replicated out-of-sample gain.

---

## 6. Specific op-level deficits

Per game, ours vs M & M & P & Q (`tools/replay_probe.py --probes ops`):

| op | ours | theirs |
| --- | --- | --- |
| WATER | 859 | 1,195 |
| HARVEST | 292 | 470 |
| CARE | 199 | **329** |
| FERTILIZE | 112 | 220 |
| COLLECT_FERTILIZER | 331 | 414 |
| **PICKUP** | **348** | **194** |

- **PICKUP is double theirs** — 154 more shed round-trips, matching the long-haul
  count (343 vs 162 moves of 4+ tiles).
- **CARE is under-used.** `build_tasks` only emits CARE once `fed_today` is true,
  but the engine banks the bonus at END of day
  (`if tile["cared_today"] and tile["fed_today"]`, kaggriculture.py:829), so
  order within the day is irrelevant and the precondition is gratuitous.
  Allowing care-before-feed gated **+$8,072 vs ahmed / −$5,891 vs tschinkel**,
  and did not replicate (+$79 on a fresh block).

---

## 7. Throughput is NOT the lever — this is the important one

Four independent attempts raised productive actions and all **lost money**:

| change | effect on work share | gated |
| --- | --- | --- |
| `v-tile` — same-tile pass before the band passes | +4.1pp | **−$4,907** |
| `w-first` — work before the shed shuttle | +0.9pp | **−$3,217** |
| `p-fert` — perishability priority | FERT 331→360 | **−$2,611** |
| chaining (historical) | +9% actions | **−$6,385** |

The extra actions are whatever is *underfoot*, mostly `P_FILLER`. Throughput
rises, average action value falls.

**But the task MIX is already right.** Against their states, the priority band we
assign to the work they choose matches what we choose almost exactly:

| band | them | us |
| --- | --- | --- |
| P_SURVIVAL | 25.9% | 26.2% |
| P_HARVEST | 9.4% | 7.4% |
| P_ANIMAL | 20.6% | 21.3% |
| P_TEND | 32.1% | 29.3% |
| P_GROW | 5.2% | 9.5% |
| P_FILLER | 6.9% | 6.2% |

And our planner proposes their work: **ABSENT only 6.6%** of the time
(`tools/taskset_diff.py`). So it is not missing task types and not mis-ranking
them. It is the *execution order and travel*, and no local rule we wrote reaches it.

---

## 8. Learning it from a teacher did not transfer either

`tools/exec_bc.py` clones task choice from M & M & P & Q (label = the unit's next
work action that day; 18,541 samples). The ranker predicts their choice at
**69.4% held-out against our hand rule's 27.9%** — it genuinely learned the
policy, and its weights confirm the mechanism (`d0` +2.38 same-tile, `filler`
−2.93, `pri` positive toward mid-bands).

Deployed: moves/action 1.45 → 1.34. Gated over **4,800 paired worlds**: 684 wins
vs 664, McNemar **z = +0.69, p = 0.49** — null. The unconstrained version
(`--mode global`) fixes travel (2.39 → 2.00) and *breaks* chaining (1.67 → 1.47).
Neither captures both.

---

## 9. What the ladder says

`rl/session_bench.py ladder --heldout`, win rate by opponent rating bin:

| bin | champ-h8 | match-all | tape-sel |
| --- | --- | --- | --- |
| <600 | 0.55 | 0.55 | 0.78 |
| 600–1000 | 0.55 | 0.57 | 0.80 |
| 1000–1500 | **0.17** | **0.23** | 0.95 |
| 1500–2000 | **0.15** | **0.10** | 0.90 |
| 2000–2500 | 0.25 | 0.28 | 0.93 |
| 2500+ | 0.20 | 0.23 | 0.70 |

The executor lineage is not uniformly weak — it **beats sub-1000 opponents and
falls off a cliff from 1000 up**. Whatever it does is sufficient against poor
play and insufficient against competent play.

---

## Summary

| # | problem | status |
| --- | --- | --- |
| 1 | walks away from underfoot work (73.6% of moves) | cause known, all fixes lose money |
| 2 | re-decides every turn → oscillation (2.39 moves/tile) | **untried**: no variant commits across turns |
| 3 | index-order claiming | fixed (`match-all`), rating-neutral |
| 4 | co-adapted to own planner (12.4% on a foreign farm) | **untried**: planner+executor never changed together |
| 5 | hiring is a ceiling, never raisable | partially explored (rollout: +$3,287) |
| 6 | PICKUP 2×, CARE under-used | CARE fix did not replicate |
| 7 | throughput ≠ money; mix already correct | settled, 4 negatives |
| 8 | learned ranker (69.4% vs 27.9%) | null over 4,800 games |

**The two genuinely untried items are #2 and #4** — cross-turn commitment, and
changing planner and executor together. Everything else is measured and closed.

---

## Tools and files

**Measurement**
- `tools/replay_probe.py` — tape-or-reactive, turn budget, moves/action, layout,
  ops, true sales. Skill: `.claude/skills/replay-autopsy/SKILL.md`
- `tools/decision_diff.py` — our agent on THEIR recorded states, turn by turn.
  Self-checks at 100% against our own actions
- `tools/taskset_diff.py` — what our planner proposes vs what they do
- `tools/sales_autopsy.py` — true fills by conservation (`order[2]` is a REQUEST,
  not a fill)
- `tools/opp_autopsy.py` — both trajectories side by side
- `tools/winrate_archive.py` — re-score every past experiment by win rate
- `rl/session_bench.py ladder --heldout` — win rate by opponent rating bin
- `tools/fast_env.py` — 1.70× faster local games, bit-identical

**The executor itself**
- `kagent/scheduler.py` — `build_tasks` (~line 91), `assign` (~line 259),
  `pick` (~line 452), shuttle (~line 272), zone_map (~line 65)
- `kagent/planner.py` — `CHAMP_HIRES` (241), hiring (366–371), `_sell_floors`
- `.venv/.../kaggriculture/kaggriculture.py` — hands wiped nightly (880),
  care bonus banked at end of day (829), `fertilizer_available` reset (831),
  yield caps (443, 827), watering window (438–443)

**Variant builders (all gated, all recorded)**
`make_match_variant.py`, `make_visit_variant.py`, `make_workfirst_variant.py`,
`make_perish_variant.py`, `make_care_variant.py`, `make_carechain_variant.py`,
`make_careopp_variant.py`, `make_layout_variant.py`, `make_burst_variant.py`,
`make_ripen_variant.py`, `make_router_variant.py`, `make_dayroute_variant.py`,
`exec_bc.py`

**Research** — see `docs/research_reading.md`. Most relevant here:
- Multi-robot task allocation, Hungarian vs greedy — arXiv:2303.04894
  (greedy ≈ 0.93× Hungarian; matched our null result)
- Maia-2 — arXiv:2409.20553 (condition on the opponent's skill; value head
  predicts the OUTCOME not the score)
- Brandfonbrener et al. — arXiv:2206.01079 (return-conditioned SL provably
  cannot stitch; read before trying a sequence model here)
- Berman et al. — arXiv:2605.18887 (the max over noisy arms is upward-biased;
  we reproduced it: tape 271 screened 83.3%, confirmed 41.9%)
