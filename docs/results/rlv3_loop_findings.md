# rlv3 ground-truth loop -- all round-1 findings (wf_90148fc7-b62), 2026-09-26

Five reviewers (rules, expressive, silent, parity, objective), each required to read
README.md in full and to DEMONSTRATE by running code, not by reading it.

Recorded here IN FULL because the loop verifies at most 8 findings per round over at
most 4 rounds, so some of these are never reached by its own Fix stage. This file is
the backlog; `demonstrated: false` means the reviewer suspected but did not execute,
so treat those as leads rather than defects.

Total: 42 findings -- 23 major, 12 minor, 7 critical

---

## [critical] `crop-block-fertilised-max-yield`

**file** `rlv3/rent.py` line 182 — **demonstrated** True
**rule** README "Object Types" note (line 28): "**Wheat** and **Carrot** only reach their listed Max Yield of 6 and 4 with fertilizer; watering alone peaks at 4 and 3." Engine `_apply_unit_action` WATER: `bonus = 2 if tile["fertilized_until_day"] >= day else 1`.

**Claim.** `rent.crop_block` prices every one-time crop at `CROPS[crop]["max_yield"]` — the FERTILISED cap — and `block_revenue`/`block_rent` never charge for the fertiliser, so the water-fill scores wheat at 6 units/5 days (1.200/tile-day) and carrot at 4/4 (1.000) where the engine delivers 4 and 3 (0.800 and 0.750, exactly README's "Yield / tile / day" column and exactly what `rent.per_tile_day` returns from `features._PER_TILE_DAY` two functions earlier).

**Failure.** `rent.marginal_rent` is the objective the crop water-fill ranks, so the dose (`lam>0`, presets `alloc`/`full`/`select`) over-prices wheat by 48% and carrot by 31% against strawberry/tomato and plants the wrong mix. On the recorded mmpq day-6 board the lam=1 plan flips from `{WHEAT:14, STRAWBERRY:7}` to `{STRAWBERRY:21}` once the engine's rates are used; on seat 1 it flips from `{WHEAT:9}` to `{STRAWBERRY:11}`. The module is ALSO internally inconsistent: `allocation.demand_caps` divides demand by 0.800 (per_tile_day) while the water-fill it feeds scores at 1.200 (crop_block), so the cap and the score disagree about the same tile by 50%.

**Fix.** In `crop_block`'s one-time branch replace `units = cd["max_yield"]` with the engine's own watered-only total, which `kagent.planner._rent_board_supply:43-46` already computes: `ws = (cd["max_yield_day"] + 1) // 2; units = min(cd["max_yield"], 1 + cd["max_yield_day"] - (ws - 1))` (wheat 4, carrot 3, melon 6). If the fertilised cap is wanted instead, `block_revenue` must subtract `econ.buy_cost("FERTILIZER", inv, n)` for the one fertiliser each tile needs.

```
$ PYTHONPATH=. uv run python /tmp/t_yield.py     # drives the engine directly
crop         occupy(d)  watered  fert  README yield/tile/day   rent.crop_block says
WHEAT                5        4     6   0.800                 units=6 occ=5 -> 1.200
CARROT               4        3     4   0.750                 units=4 occ=4 -> 1.000

$ PYTHONPATH=. uv run python /tmp/t_rent.py      # mmpq replay, day 6, real View
block_rent ($/tile-day) as coded vs with the engine's watered-only units/occupancy:
  WHEAT        as coded    40.30   engine-true    27.25   overstated x1.48
  CARROT       as coded    30.97   engine-true    23.72   overstated x1.31

$ PYTHONPATH=. uv run python /tmp/t_alloc.py     # allocation.allocate(lam=1.0)
mmpq-d6-s0     AS CODED     targets={'WHEAT': 17, 'CARROT': 2, 'STRAWBERRY': 21, ...} quota={'WHEAT': 14, 'STRAWBERRY': 7}
mmpq-d6-s0     ENGINE-TRUE  targets={'WHEAT': 7,  'CARROT': 0, 'STRAWBERRY': 33, ...} quota={'STRAWBERRY': 21}
mmpq-d6-s1     AS CODED     targets={'WHEAT': 13, 'STRAWBERRY': 2, ...} quota={'WHEAT': 9}
mmpq-d6-s1     ENGINE-TRUE  targets={'WHEAT': 0,  'STRAWBERRY': 15, ...} quota={'STRAWBERRY': 11}
```

## [critical] `exec-levers-die-day-2`

**file** `rlv3/ledger.py` line 216 — **demonstrated** True
**rule** rlv3/ledger.py docstring: 'A lever that costs turns and does not move its own metric is switched off for the rest of the game'; rlv3/executor.py docstring: 'check a non-zero counter in stats() from inside a worker before reading any arm as a result'

**Claim.** The ledger's one-way kill switches disable every rlv3 executor lever within 2-4 days of a 30-day season, so the `exec`/`full` presets and every rlv3/arms/*.json single-lever arm measure a day-0-only intervention.

**Failure.** `pl_wheat_adj` is switched off at day 2 and `co_chain_tile` at day 3-4, permanently (`self.off` is only ever added to). All 172 placement rewrites in a full season happen on day 0; `co_chain_tile` fires ZERO reorders in a whole game. `pl_wheat_adj` is structurally doomed: its metric is `feeds_from_hand_share`, `_hand_wheat` is set True only by HARVESTing a WHEAT plant, and CROPS['WHEAT']['first_yield_day']==2, so days 0 and 1 can only ever score 0.0 whenever an animal is fed -> `_handfeed_low` reaches 2 at day 2 in every such game. Any eval of an rlv3 executor arm therefore reads as 'null' while the lever was alive for one day.

**Fix.** Do not let a kill switch fire on a day whose metric is structurally undefined: skip `_handfeed_low`/`_chain_flat` increments before the first day the metric is attainable (day >= CROPS['WHEAT']['first_yield_day']+1 for feeds_from_hand, and require a minimum denominator), and make `off` revocable - `correct()` already resets `_handfeed_low` to 0 when the metric recovers, so `self.off.discard(name)` on that branch. Add a smoke assertion that each ENABLED flag has a non-zero counter after day 10, not just a non-empty stats dict.

```
PYTHONPATH=. .venv/bin/python /tmp/probe2.py  (preset `exec`, world session-112391485, seat 0)
 day 0 off [] hf_low 0 chain_flat 0
 day 1 off [] hf_low 1 ... rates {'feeds_from_hand_share': 0.0, 'chores_per_visit': 1.818, ...}
 day 2 off ['pl_wheat_adj'] hf_low 2 chain_flat 1
 day 3 off ['co_chain_tile','pl_wheat_adj'] chain_flat 2
 day 4 off ['co_chain_tile','pl_crop_cluster','pl_wheat_adj']
 day 5 rates feeds_from_hand_share 0.2 (ABOVE target) and hf_low resets to 0, but 'pl_wheat_adj' stays in `off` for the rest of the season
 final EXEC stats: {'pl_wheat_adj.displaced': 143, 'pl_wheat_adj.moved': 29}  <- identical to the totals already present at the day-1 dawn, i.e. every move was on day 0; no 'co_chain_tile.reordered' key at all.
Replicated on a second world (/tmp/probe8.py, session-98066360 seat 0 band 2, preset `full`): pl_wheat_adj killed day 2, co_chain_tile day 4, same 143/29 all on day 0.
```

## [critical] `ledger-kill-switch-silent`

**file** `rlv3/ledger.py` line 216 — **demonstrated** True
**rule** rlv3/README.md "Liveness, which is not optional": "An arm that reports as rlv3 while running the shipped agent is DEAD, not null, and is indistinguishable from a null result once it is a row in a table." The liveness contract covers the plan provider only; the levers have no equivalent.

**Claim.** `ExecLedger.correct()` switches the executor levers off inside the first three days of a 30-day season and records nothing anywhere — no counter, no stat key, no field on the episode result — so an `exec`/`full` arm is the baseline executor for 27-28 of 30 days while every liveness check reports it healthy.

**Failure.** `rlv3:exec` and `rlv3:full` are the arms the package intends to put on the 995-game gate. On the recorded world `pick_games(1)` returns, `pl_wheat_adj` was killed on day 2 and `co_chain_tile` on day 3. The gate would report `full` ≈ `recommended` and the conclusion drawn would be "the Maia2 placement levers are worth nothing", when what was actually measured is two days of perturbation followed by 28 days of the shipped scheduler. The mechanism guarantees it rather than making it unlucky: `_handfeed_low >= 2` kills `pl_wheat_adj` after two days below `handfeed_target = 0.15`, but wheat seated beside a coop needs 2 days to first yield (README Object Types) and must then be HARVESTed and used for FEED before `feeds_from_hand_share` can move at all, so the kill switch's window is strictly shorter than its own lever's causal horizon; and `_chain_flat >= 2` kills `co_chain_tile` on any two consecutive days where a noisy ratio fails to rise.

**Fix.** Count the kill in `_STATS` at the moment it happens — in `ledger.correct()` (or in `executor.active_flags`, which already knows `LEDGER.off`) emit `_STATS[f"{flag}.killed_day{self.day}"] += 1` — and make `rlv3.arm.run_episode` put `sorted(ledger.LEDGER.off)` plus the kill day on `result["extra"]["rlv3"]`, so a table row carries how many days each lever was live. Separately, the streak thresholds must be at least the lever's own effect horizon (for `pl_wheat_adj`, no earlier than the first day a wheat tile planted under the rule could have been harvested).

```
$ PYTHONPATH=. uv run python /tmp/probe14.py   (one episode, preset=exec, tools.agent_scorecard.pick_games(1), wraps ledger.ExecLedger.correct to record the day each flag enters `off`)
preset=exec, one recorded world
  day each declared lever was SWITCHED OFF by ledger.correct(): {'pl_wheat_adj': 2, 'co_chain_tile': 3, 'pl_crop_cluster': 4}
  days the lever was actually live:  pl_wheat_adj 2 of 30;  co_chain_tile 3 of 30
  stats reported to smoke/scorecard: {'pl_wheat_adj.displaced': 143, 'pl_wheat_adj.moved': 29}
  -> nothing in stats(), counters, or the result dict mentions the kill

and the unwrapped run of the same episode (/tmp/probe13.py):
  money 58029.0 opp 53533.0 win 1.0
  exec seam stats: {'pl_wheat_adj.displaced': 143, 'pl_wheat_adj.moved': 29}
  LEVERS THE LEDGER SWITCHED OFF MID-GAME: ['co_chain_tile', 'pl_crop_cluster', 'pl_wheat_adj']
  rlv3 counters: {'episodes': 1, 'observe': 719, 'make_plan': 30, 'dawn': 30, 'project': 30, 'projects_win': 27}

smoke.py:127 (`if not out["exec_stats"]: failures.append(... the levers are inert)`) and scorecard_hook.py:349-352 both pass on that stats dict.
```

## [critical] `sell-seam-drops-committed-buys`

**file** `rlv3/selling.py` line 535 — **demonstrated** True
**rule** README 'Market Action': 'any orders past that limit are silently dropped'; rlv3/selling.py's own contract -- 'The shipped builder is called EXACTLY ONCE, so its mutations of plan.land_buys / plan.animal_buys / plan.plant_quota happen exactly as shipped' and 'HIRE is never displaced'.

**Claim.** rlv3.selling.build_orders re-orders the shipped builder's order list and then re-truncates it to MAX_MARKET_ORDERS, silently discarding BUY_ANIMAL / BUY_LAND / BUY_SEED / BUY_PRODUCT orders that kagent.market.build_orders has ALREADY committed by decrementing plan.animal_buys / plan.land_buys and its internal money ledger.

**Failure.** The plan records an animal or a quadrant as bought while no order was ever submitted, and because the day plan is rebuilt only at hour 0 (HeuristicPolicy.act) the purchase is never re-issued for the rest of the day. On one recorded world with the shipped RECOMMENDED sell config (preset `recommended`, also inside `full`), 9 BUY_ANIMAL, 10 BUY_SEED and 1 BUY_PRODUCT orders were dropped this way in a single 719-step season. BUY_SEED self-heals next turn (the shipped builder mutates nothing for it); BUY_ANIMAL and BUY_LAND do not. At hour 0 the champion hires 5-11 hands (planner.CHAMP_HIRES), so `len(hires) + sell_slots_lead (3) + len(rest)` exceeds the 10-order cap on most mornings of the season.

**Fix.** Build the final list so it can never lose a non-SELL order the shipped builder emitted. In build_orders, reserve the non-SELL slots first: `keep = hires + rest` (which is already <= MAX_MARKET_ORDERS because the shipped builder's own `add()` capped it), then `room = MAX_MARKET_ORDERS - len(keep)`, place at most `min(room, sell_slots_lead)` sells in the lead and fill the remainder from the tail -- i.e. slice the SELLS, never `orders[:MAX_MARKET_ORDERS]` over a list whose head is sells. Equivalently, assert `set(map(tuple, base)) - set(map(tuple, orders))` contains only SELLs before returning.

```
Mechanism, on a synthetic mid-season state (no engine run) -- /tmp/rlv3rev/p6.py, `PYTHONPATH=. uv run python /tmp/rlv3rev/p6.py`:
  shipped build_orders (10): HIRE x5, ['BUY_PRODUCT','WHEAT',3], ['BUY_LAND'], ['BUY_ANIMAL','GOOSE',2], ['BUY_ANIMAL','COW',1], ['BUY_SEED','WHEAT',6]
  rlv3 slots_only     (10): HIRE x5, ['SELL','MILK',30], ['SELL','WOOL',20], ['SELL','STRAWBERRY',20], ['BUY_PRODUCT','WHEAT',3], ['BUY_LAND']
  NON-SELL orders the shipped builder emitted but rlv3 dropped: [['BUY_ANIMAL','GOOSE',2], ['BUY_ANIMAL','COW',1], ['BUY_SEED','WHEAT',6]]
  plan.animal_buys after shipped: {'GOOSE': 0, 'COW': 0}   after rlv3: {'GOOSE': 0, 'COW': 0}   <-- debited, order never sent
Identical result with the full schedule (SellConfig() instead of SELL.RECOMMENDED).

In real play, one recorded world, preset `recommended`, --procs 1 -- /tmp/rlv3rev/p7.py:
  money 120215.0 opp 120076.0
  build_orders steps: 719
  non-SELL orders the shipped builder emitted and rlv3 DROPPED: {'BUY_ANIMAL': 9, 'BUY_SEED': 10, 'BUY_PRODUCT': 1}
(For context only, not attribution: rlv3/README.md's smoke table has ship2_hv on this same world at 121,817/118,562, margin +3,255; this run came back at margin +139.)
```

## [critical] `sell-slot-reorder-evicts-buys`

**file** `rlv3/selling.py` line 535 — **demonstrated** True
**rule** README "Market Action": "Each turn you can submit up to `maxMarketOrdersPerTurn` (default 10) market actions; any orders past that limit are silently dropped." rlv3/params/allocation_v0.json `_note_first_deviation_day` already records this exact hazard — "The day-0 market order list already sits near MAX_MARKET_ORDERS=10, and extra day-0 orders are recorded as dropping the seed orders" — but the selling seam re-creates it on every day.

**Claim.** `build_orders` re-orders SELLs into the leading slots and then truncates the whole list to `MAX_MARKET_ORDERS`, but the SELL list comes from `market._sell_orders` unbounded by the slots left over, so the shipped builder's BUY_LAND / BUY_ANIMAL / BUY_SEED orders are pushed past the cap and silently dropped — after the shipped builder has already debited them from the DayPlan.

**Failure.** `kagent.market.build_orders` allocates its 10 slots by priority (HIRE, feed BUY_PRODUCT, BUY_LAND, BUY_ANIMAL, BUY_SEED, then sells fill whatever is left) and mutates the plan as it adds each order: `plan.land_buys = remaining_land` and `plan.animal_buys[kind] = want - n`. rlv3 then rebuilds the list as `hires + lead_sells + rest + tail_sells` and cuts it at 10, so on a day with several hires plus a land buy the evicted orders are exactly the ones that were debited. `kagent.policy.HeuristicPolicy.act:80-92` builds the DayPlan once at hour 0 and reuses it for the rest of the day, so the quadrant and the animals are never re-requested that day. This is live in the DEFAULT preset: `config_from_spec` defaults to `preset="recommended"`, whose `sell="slots"` is `RECOMMENDED = SellConfig(slots_only=True)`, and `slots_only` bypasses the sizer but NOT the slot re-order.

**Fix.** Size the SELL list to the slots that are actually free instead of truncating the assembled list: in `build_orders`, compute `room = MAX_MARKET_ORDERS - len(hires) - len(rest)` and take `slices = slices[:max(0, room)]` before splitting into `lead`/`tail`. That keeps the documented slot re-order (sells still quoted ahead of the buys) without evicting any order the shipped builder chose to send.

```
$ PYTHONPATH=. uv run python /tmp/probe12.py   (10x10 board, NW+NE, day 12 hour 1, 4 cows, plan.hires_target=6, land_buys=['SW'], animal_buys={'GOOSE':2,'SHEEP':1}, shed carries STRAWBERRY/MILK/WOOL/EGG)
shipped : [['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['BUY_PRODUCT','WHEAT',3], ['BUY_LAND'], ['BUY_ANIMAL','GOOSE',2], ['BUY_ANIMAL','SHEEP',1]]
rlv3    : [['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['SELL','WOOL',14], ['SELL','MILK',18], ['SELL','STRAWBERRY',20], ['BUY_PRODUCT','WHEAT',3]]
DROPPED by the rlv3 slot re-order: [['BUY_LAND'], ['BUY_ANIMAL','GOOSE',2], ['BUY_ANIMAL','SHEEP',1]]
after rlv3: plan.land_buys = []  plan.animal_buys = {'GOOSE': 0, 'SHEEP': 0}

A milder version fires with only 2 hires (/tmp/probe11.py): DROPPED [['BUY_SEED','CARROT',5]].
```

## [critical] `sell-slots-drop-buy-orders`

**file** `rlv3/selling.py` line 535 — **demonstrated** True
**rule** rlv3/selling.py:489-505 'The shipped builder is called EXACTLY ONCE ... Its non-SELL orders are kept in their order'; README Market Action: 'any orders past that limit are silently dropped'

**Claim.** `build_orders` re-adds the FULL `market._sell_orders` list on top of the shipped builder's already-cap-filled 10 orders, so the final `orders[:MAX_MARKET_ORDERS]` truncation silently deletes BUY_ANIMAL / BUY_SEED / BUY_PRODUCT / BUY_LAND orders the shipped builder had emitted.

**Failure.** On the `recommended` preset (slots_only=True, the configuration the package's own pre-gate supports and which `full` ships) the champion's market orders are cancelled: BUY_ANIMAL SHEEP 3 is dropped on every dawn from day 11 to day 19, BUY_SEED WHEAT 24 on day 11, BUY_SEED WHEAT 9/14 on days 16/20. 19 of 30 dawn steps overflow the 10-order cap. The module docstring claims slots_only 'changes no walking, no allocation, and no sizing'; it changes the allocation by never buying the animals and seeds the planner asked for.

**Fix.** Budget the slices against the room the shipped builder actually left: cap `lead + tail` at `MAX_MARKET_ORDERS - len(hires) - len(rest)` (plus the slots the shipped SELLs already occupied) and drop SELL slices, never `rest`. Assert in a test that `set(rest) <= set(orders)` for every step.

```
PYTHONPATH=. .venv/bin/python /tmp/probe6.py  (verbatim copy of selling.build_orders with `base` and `orders` recorded; reproduces the real preset to the dollar - smoke.py --presets recommended gives the same 120,215/120,076)
 steps where an order the SHIPPED builder emitted did not survive: 16
 lost order kinds: {'BUY_ANIMAL': 9, 'BUY_SEED': 10, 'BUY_PRODUCT': 1}
 day 11 hour 0 base 10 out 10 lost [['BUY_ANIMAL','SHEEP',2],['BUY_SEED','WHEAT',24]]
 day 12..19 hour 0 lost [['BUY_ANIMAL','SHEEP',3]] every day
/tmp/probe7.py shows the mechanism: day 11 hour 0 -> base_n 10, base_SELL 0, rlv3_slices 3, built_n 13 (truncated to 10). 19 overflow steps.
Same world, identity arm rlv3:base finishes 121,817/118,562 (+3,255); rlv3:recommended finishes 120,215/120,076 (+139) - our bank -$1,602 and the OPPONENT's +$1,514.
```

## [critical] `slots-only-evicts-buy-orders`

**file** `rlv3/selling.py` line 528 — **demonstrated** True
**rule** README 'Market Action': "you can submit up to maxMarketOrdersPerTurn (default 10) market actions; any orders past that limit are silently dropped"; rlv3/selling.py:494 docstring "Its non-SELL orders are kept in their order".

**Claim.** `selling.build_orders` re-assembles the order list as `hires + lead_sells + rest + tail_sells` and then truncates to MAX_MARKET_ORDERS, so on a saturated order book it promotes SELL orders the shipped builder had deliberately dropped and evicts the buys it had issued — including the animal-feed `BUY_PRODUCT WHEAT`.

**Failure.** On 488 of 2,592 real leader steps (18.8%) the `slots_only` config — `selling.RECOMMENDED`, used by the `recommended` and `full` presets and described in the module docstring as 'free: it changes no walking, no allocation, and no sizing' — silently deletes an order the shipped market controller issued: 319 feed BUY_PRODUCT, 266 BUY_ANIMAL, 208 BUY_SEED, 15 BUY_LAND. README says orders past `maxMarketOrdersPerTurn` are silently dropped, so nothing reports this. A dropped feed buy is one step closer to the two-consecutive-unfed-days escape; a dropped BUY_SEED is the plant quota the allocator just asked for. The docstring's claim on line 494, "Its non-SELL orders are kept in their order", is false exactly when the book is full.

**Fix.** Assemble sells into slots that are actually free rather than prepending them: build `hires + rest` first, then insert sells only up to `MAX_MARKET_ORDERS - len(hires) - len(rest)`, and take the lead slots from SELLs only by swapping with an equal number of non-HIRE orders that still fit. At minimum, cap `lead` at `max(0, MAX_MARKET_ORDERS - len(hires) - len(rest))` so the slot move is a genuine permutation and never a deletion, and assert `Counter(non_SELL(out)) == Counter(non_SELL(base))` in the unit tests.

```
PYTHONPATH=. .venv/bin/python /tmp/slots_check.py (12 leader replays, days 2-28, hours 0/1/8/16, both seats; shipped `kagent.market.build_orders` vs `rlv3.selling.build_orders(..., S.RECOMMENDED)` on the identical View/plan):
  steps checked 2592; steps where rlv3 slots_only DROPPED a non-SELL order the shipped builder issued: 488 (18.8%)
  dropped order kinds: {'BUY_ANIMAL': 266, 'BUY_SEED': 208, 'BUY_LAND': 15, 'BUY_PRODUCT': 319}
One case in full (mmpq replay, day 7 hour 0 seat 0, MAX_MARKET_ORDERS=10):
  shipped: [HIRE]x8, ['BUY_ANIMAL','COW',1], ['BUY_SEED','STRAWBERRY',1]
  rlv3   : [HIRE]x8, ['SELL','WOOL',2], ['SELL','WHEAT',3]
  LOST   : [('BUY_ANIMAL','COW',1), ('BUY_SEED','STRAWBERRY',1)]
Note the shipped list has zero SELLs: the two sells rlv3 promoted are ones `add()` had refused, so this is not a re-order of the shipped orders, it is an injection that evicts buys.
```

## [major] `arm-parse-bool-inversion`

**file** `rlv3/arm.py` line 89 — **demonstrated** True
**rule** rlv3/README.md preset table and agent.py `_select` docstring: "OFF BY DEFAULT AND IT SHOULD STAY OFF UNTIL A PAIRED GATE SAYS OTHERWISE."

**Claim.** `arm.parse` falls back to the raw string when `json.loads` fails, so a Python-cased boolean override becomes a non-empty string and `Rlv3Config` coerces it with `bool()` — `select=False` turns plan selection ON.

**Failure.** `--arm rlv3_full=dev:rlv3:full:select=False` (or any `...:project=False`, `...:select=True/False` written the Python way) produces `cfg.select = True`. That switches on `_select`, the per-dawn candidate-plan ranking, which the README and agent.py both say must ship OFF — "V-ranking is CLOSED here (four heads at chance out-of-sample) ... `select` ships off". The arm's label is unaffected (`rlv3:full` either way), so the eval table names it identically to the arm without selection, and the only place the real value appears is `to_json()["select"]`, which `make_job` does not attach when the caller passes a dict spec. The same fallback silently turns any mistyped key's value into a string rather than rejecting it.

**Fix.** In `parse`, normalise before the JSON attempt: `if v.strip() in ("True","False","None"): v = v.strip().lower().replace("none","null")`, and on a `ValueError` accept the bare string only for fields that are genuinely strings (`sell`, `label`, `rules_path`) — raise for everything else rather than silently coercing. Also fold `select`/`lam`/`exec_flags` into `Rlv3Config.label` so two differently-configured arms cannot share a name.

```
$ PYTHONPATH=. uv run python /tmp/probe2.py
full:select=false            raw=False      cfg.select=False cfg.project=True label=rlv3:full
full:select=False            raw='False'    cfg.select=True  cfg.project=True label=rlv3:full
full:project=False           raw='False'    cfg.select=False cfg.project=True label=rlv3:full
recommended:select=True      raw='True'     cfg.select=True  cfg.project=True label=rlv3:recommended
```

## [major] `fingerprint-cannot-see-heads-version-flag`

**file** `rlv3/fingerprint.py` line 146 — **demonstrated** True
**rule** rlv3/fingerprint.py module docstring: "`goose_target` went from [0, 8, 16, 24, 32] to [0, 4, 6, 10, 16], every stored `choices` index in the BC corpus silently came to mean a different number of geese ... `head_sizes_sha` catches that in milliseconds, with no games played."

**Claim.** `action_space()` reads `features.HEADS_VERSION` out of the DRIVER process, and `stages._flags` does not parse `--heads-version`, so the stamp records the action space the driver happens to have imported rather than the one each stage's subprocess will run under.

**Failure.** The one action-space knob a stage command can actually carry is the one the stamp is blind to. Add `--heads-version 2` to the `ppo` command and the fingerprint still reports `heads_version: 1`; `check()` therefore says the v1 BC corpus and the v1 `bc.pt` are FRESH and `job.resolve_step` skips them, so PPO resumes from a checkpoint whose stored `choices` indices mean different counts — precisely the `job_topboard.json` run-3 failure the module docstring says `head_sizes_sha` exists to catch in milliseconds. `assert_slice_invariant` raises no complaint either.

**Fix.** Make the head version an explicit, single-sourced field: read `job['heads_version']` (default 1) in `fingerprint.fingerprint(cfg)` and have `rlv3.stages` inject `--heads-version <that>` into every stage command it formats, plus a `_flags` entry and an `assert_slice_invariant` refusal when a command carries a `--heads-version` that differs from the fingerprint's. Then a v1 corpus under a v2 fingerprint quarantines instead of skipping.

```
PYTHONPATH=. .venv/bin/python -c "import json; from rlv3 import stages as ST, fingerprint as FP; job=json.load(open('rlv3/jobs/rlv3_train.json')); ppo=[s for s in job['stages'] if s['name']=='ppo'][0]; cmd=ppo['cmd'].replace('{python}','python').replace('{out}','/tmp/run').replace('{procs}','1')+' --heads-version 3 --lr 0.0001 --ent-coef 0.001 --start-iter 0'; print(sorted(ST._flags(cmd))); print(ST.assert_slice_invariant(cmd,0,24,24,run_dir=None)); print(FP.action_space()['heads_version'])"
  flags stages.py parses: ['ent_coef', 'iters', 'lr', 'patience', 'plateau_cool', 'resume', 'run', 'session_mix', 'start_iter']
  assert_slice_invariant complaints: []
  fingerprint heads_version regardless of the cmd: 1
```

## [major] `fingerprint-n-features-always-none`

**file** `rlv3/fingerprint.py` line 117 — **demonstrated** True
**rule** rlv3/README.md "The stamp": "The fingerprint is the action space (`heads_version`, `heads_mode`, sha1 of `HEAD_SIZES`, `n_action_logits`, `n_features`, board arm and `BOARD_CH` ...)" — listed as a guard that, as shipped, is inert.

**Claim.** `probe_n_features(play=False)` — the default, and the only mode the CLI uses without `--probe-features` — returns None in the parent process, so `n_features` is None on both sides of every stamp comparison and the one ACTION_KEY that guards the observation width never discriminates.

**Failure.** `kagent.features.N_FEATURES` is None until something encodes, and `rlv3.train.run_job` computes the fingerprint at the top of the slice in a process that has encoded nothing; the `<run>/n_features.json` cache is only ever written on the `play=True` branch, so it never exists either. Every stamp is therefore written with `"n_features": null` and every `FP.check` compares null to null and returns fresh. A change to `kagent.features.encode` that adds or removes an input — which moves `n_features` but not `heads_version`, `head_sizes_sha` or `n_action_logits` — would let a `bc.pt` and a PPO run trained at the old width be skipped as fresh and resumed at the new one. The same key also mis-fires in the other direction: an artifact stamped from a process that HAD encoded carries a real integer and is then quarantined against today's null.

**Fix.** Resolve the width without a game: encode one synthetic View (the `rlv3/tests/conftest.py:blank_obs` board is enough — `features.encode(View(blank_obs()), Memory())` sets `features.N_FEATURES`) inside `probe_n_features` before falling back, and write the result to the `n_features.json` cache on that path too. Failing that, refuse a None: make `FP.check` return `(False, "n_features unresolved")` rather than treating null == null as a match.

```
$ PYTHONPATH=. uv run python /tmp/probe5.py
features.N_FEATURES at import: None
probe_n_features(play=False) -> None
fingerprint['n_features'] = None
stamped under n_features=1024; check against today's fp -> (False, 'n_features changed: stamped 1024 != now null')
stamp n_features: None  today's: None  check -> (True, 'fresh')
```

## [major] `fitted-json-outside-the-fingerprint`

**file** `rlv3/fingerprint.py` line 206 — **demonstrated** True
**rule** rlv3/fingerprint.py:1-16 'A skip is honoured only when that stamp still matches today's fingerprint on the keys that artifact actually depends on'; rlv3/land.py: 'the JSON IS the checkpoint, so training resumes from the last file'

**Claim.** `code_digest` hashes `*.py` only, so `rlv3/calibration.json` (the p_win table - 'the ONE fitted object') and `rlv3/params/allocation_v0.json` (the land theta, the crop/herd caps, w_opp - 'the JSON IS the checkpoint') are outside the fingerprint entirely, and `objective.fingerprint()` is never written into any artifact.

**Failure.** Editing either file changes what every component decides and invalidates NO stamp: every stage reports 'fresh' and skips, a resumed run continues under a different objective and a different theta, and two arms in one table can have been judged under different calibrations. This is exactly the `kaggriculture-gate-cache-stale` failure the module was written to close, on the two files the package itself calls fitted checkpoints.

**Fix.** Add a `params` key to `fingerprint()` hashing the bytes of rlv3/calibration.json and rlv3/params/*.json (or extend code_digest to *.json under rlv3/), put it in ACTION_KEYS' sibling group, and record `objective.fingerprint()` in the manifest and in every stamp.

```
PYTHONPATH=. .venv/bin/python (probe):
 rlv3.fingerprint keys: ['board','board_arm','board_ch','code','computed','corpus','engine','head_names','head_sizes_sha','heads_mode','heads_version','n_action_logits','n_features','schema']
 keys mentioning calibration/objective: []
 files under ('rlv3','kagent','rl'): 353; hashed by code_digest (*.py only): 82
 rlv3/calibration.json hashed? False ; rlv3/params/allocation_v0.json hashed? False
 objective.fingerprint with shipped table : ef3d8334d3dae46b
 objective.fingerprint with edited  table : 28d628c33901c52c
 rlv3.fingerprint['code'] unchanged        : True
And `grep -rn 'OBJ.fingerprint'` finds it only in Rlv3Config.to_json (which agent.make_job then strips with js.pop('objective')) and in smoke.py's print - no checkpoint or manifest carries it.
```

## [major] `install-seams-uninstalls-sitecustomize`

**file** `rlv3/agent.py` line 375 — **demonstrated** True
**rule** rlv3/executor.py module docstring: "Install through `rlv3/sitecustomize.py` and check a non-zero counter in `stats()` from inside a worker before reading any arm as a result" — the teardown path removes precisely that install.

**Claim.** `install_seams` registers `EXEC.uninstall` unconditionally although `EXEC.install` is a no-op when the seam is already installed, so an rlv3 arm's teardown removes the process-wide seam that `rlv3/sitecustomize.py` installed — and every later `knobs:rlv3/arms/*.json` arm in that worker runs the SHIPPED executor while `scheduler.EXEC[flag]` still reads True.

**Failure.** `rlv3/arms.py` and the README's own scorecard command drive the single-lever gate arms as `KAGG_RLV3=1 PYTHONPATH=.:rlv3 ... --agent knobs:rlv3/arms/pl_wheat_adj.json`, which has no rlv3 plan provider and depends entirely on sitecustomize's install. Those arms have no liveness assertion at all (unlike `arm.run_episode`), so after any rlv3 arm has run and torn down in the same process they report as `pl_wheat_adj` while executing `kagent.scheduler.build_tasks`. The mirror-image trap: the README's documented graduation path ("Later, two lines" — patch `rl/rollout._make_dev_policy_agent` to call `make_dev_agent` directly) never calls `restore` at all, so both the scheduler seam and `kagent.market.build_orders` stay patched for every subsequent arm in the worker, contaminating the paired gate in both directions.

**Fix.** Make the install reference-counted: have `executor.install()` return True only when it actually rebound the module attributes, and in `install_seams` append `EXEC.uninstall` only on that True (`if EXEC.install(scheduler): undo.append(EXEC.uninstall)`). Add the same guard note to the README's "two lines" section, or have that snippet return an agent whose `restore` rollout will call.

```
$ PYTHONPATH=. uv run python /tmp/probe16.py
after sitecustomize-style install: is_installed True  EXEC['pl_wheat_adj'] True
rlv3 arm built:   is_installed True  market.build_orders patched: rlv3.selling.patched
after arm.restore(): is_installed False  EXEC['pl_wheat_adj'] STILL True  -> the next knobs:rlv3/arms/*.json arm in this worker runs the SHIPPED executor
   scheduler.build_tasks is now kagent.scheduler.build_tasks

bare make_dev_agent, no teardown (the documented one-line rollout patch):
   scheduler.build_tasks: rlv3_build_tasks  market.build_orders: patched  <- persists for every later arm in this process
```

## [major] `melon-demand-cap-deletes-the-crop`

**file** `rlv3/allocation.py` line 326 — **demonstrated** True
**rule** README Market Mechanics / The Price Function: melon is base $250 with above_func `sq` at above_target 3.60 and T=300, so its ceiling is curve depth (~150 units above I0 before the $1 floor), not town demand; README Town Buildings table lists no shop demanding melon.

**Claim.** `demand_caps` prices MELON off town drain minus opponent supply, but MELON is in no SHOPS entry, so its whole season demand is the town centre's 1 unit/day; once the opponent's projected melon supply exceeds that (i.e. essentially always), the cap is 0, MELON drops out of `crop_cands` at line 382, and the water-fill can never place a melon tile at any setting of allocation_v0.json.

**Failure.** The allocation dose (`alloc` / `full` / the lam033-lam100 bundles already built under rlv3/eval/build/) cannot express the melon line the measured leaders all run — 4-8 tiles per the brief, 4-12 on their own recorded boards — while rlv3's own `rent.marginal_rent` prices the marginal melon tile as the MOST valuable tile on the farm in 3 of 6 sampled states. Melon target is 0 in 16/18 recorded leader mid-season states at lam=1. It is not even visible as a refusal: MELON is absent from the trace's `dropped` list, so the ladder reads as "the rent model didn't want melon" when in fact the rent model was never asked.

**Fix.** The cap for a product with no SHOPS entry must not be a drain cap. Either (a) in `demand_caps`, when `not any(crop in items for items in SHOPS.values())`, set the cap to the hard ceiling and let `water_fill`'s own `m <= 0` stop bind — `rent.marginal_rent` already integrates the whole stream through `econ.sell_revenue` including self-impact and the opponent term, which is the correct ceiling for a depth-limited market; or (b) cap it by price-curve headroom, `econ.units_sellable_above(crop, inv, floor) / units_per_tile`, instead of by net drain. Either way, append every zero-capped crop to `dropped` so the trace shows a deletion rather than an absence.

```
PYTHONPATH=. .venv/bin/python /tmp/melon_rent.py (18 mid-season views from replays/{mmpq,boey,dsm} via rlv3/tests/conftest.py):
  boey-d6-s0  WHEAT 45.60  CARROT 36.00  TOMATO 12.39  STRAWBERRY 28.13  MELON 59.31 $/tile-day  -> melon demand_cap 0, in water-fill: False
  mmpq-d12-s0 WHEAT 46.70  CARROT 43.00  TOMATO 14.82  STRAWBERRY  0.24  MELON 58.92          -> melon demand_cap 0, in water-fill: False
  mmpq-d6-s1  WHEAT 40.60  CARROT 34.25  TOMATO 10.96  STRAWBERRY 35.97  MELON 47.69          -> melon demand_cap 0, in water-fill: False
PYTHONPATH=. .venv/bin/python /tmp/alloc_target.py:  melon target == 0 in 16/18 leader mid-season states (lam=1); the same boards stand 7-12 melon at day 6 and 4-8 at day 12.
PYTHONPATH=. .venv/bin/python /tmp/melon_final.py:
  MELON appears in SHOPS? False
  forecast_drain[MELON] = 24.0 at quantile 0.05, 0.35, 0.50 AND 0.95 (the stance quantile is inert on melon: no shop can ever be drawn for it)
  opponent projected MELON supply: 72.0 -> net_per_day = max(0, 24-72)/23 = 0 -> cap 0
  crop.caps.MELON=0/24/99 -> demand cap 0, crop_cands=['WHEAT','CARROT','TOMATO','STRAWBERRY'], dropped=[]
No key in rlv3/params/allocation_v0.json changes this.
```

## [major] `melon-ripe-age-12-not-10`

**file** `rlv3/rent.py` line 179 — **demonstrated** True
**rule** README "Object Types" (line 27): "**Melon**'s bonus window is ages 6–12, but base 1 plus one unit per watered day reaches the cap of 6 at age 10, so ages 11–12 add nothing." Also `kagent.planner.MELON_LAST_PLANT_DAY = 19  # d+10 harvest must land by day 29` and `features.LAST_PLANT["MELON"] = 19`.

**Claim.** `crop_block` uses `cd["max_yield_day"]` (12) as melon's ripe age everywhere — the harvest lag `k`, the occupancy divisor `occupies = 13`, and the feasibility gate `day + 12 > SEASON_END`. The engine caps a watered melon at 6 on age 10, occupying 11 tile-days. `rlv3/params/allocation_v0.json` states the intent explicitly: "rent.crop_block returns None the day a crop's harvest can no longer land by 29, straight off the engine's CROPS table" — which is the claim that fails.

**Failure.** Two effects, in opposite directions. (a) Melon's rent is understated 15% ($49.70 vs $58.74/tile-day on the mmpq day-6 board) because it is divided by 13 tile-days instead of 11, so the water-fill under-buys melon all season — the one product in NO shop, whose only competition is the opponent's own supply. (b) `allocation.demand_caps:318` reads `crop_block(...) is None` as a hard `crops["MELON"] = 0` and appends MELON to `dropped`, so melon is refused on days 18 and 19 although a melon planted on day 19 harvests its full 6 on day 29 — two days earlier than the champion base plan it is blended against, which means at lam>0 the target is pulled to zero on days the shipped agent still plants.

**Fix.** Compute the ripe age from the engine's watered-only accumulation rather than `max_yield_day`: `ws = (cd["max_yield_day"] + 1) // 2; ripe = min(cd["max_yield_day"], ws - 1 + (cd["max_yield"] - 1))` (wheat 4, carrot 3, melon 10), then use `ripe` for `k`, `occupies = ripe + 1` and the gate `day + ripe > SEASON_END`. That makes the three last-plant days 25 / 26 / 19, matching `features.LAST_PLANT`.

```
$ PYTHONPATH=. uv run python /tmp/t_melon.py
  day 10 age 10  yield_units=6
  day 11 age 11  yield_units=6      <- ages 11-12 add nothing
  day 12 age 12  yield_units=6
melon planted day 19, watered daily, HARVEST on day 29 -> {'MELON': 6}
rent.crop_block('MELON', day=17) = (12, 6.0, 13.0)   (planner.MELON_LAST_PLANT_DAY=19, features.LAST_PLANT['MELON']=19)
rent.crop_block('MELON', day=18) = None
rent.crop_block('MELON', day=19) = None

$ PYTHONPATH=. uv run python /tmp/t_rent.py
MELON       (12, 6.0, 13.0)                       0.462   features._PER_TILE_DAY 0.545   engine 0.545
  MELON        as coded    49.70   engine-true    58.74
```

## [major] `opponent-delta-count-times-price`

**file** `rlv3/rent.py` line 258 — **demonstrated** True
**rule** rlv3/rent.py:25-30 'Every function here integrates the entire remaining stream through econ.sell_revenue; none of them ever multiplies a count by a single marginal price'; :386-388 'because every score here is concave ... greedy IS the equal-marginal optimum'

**Claim.** `opponent_delta` is `their_remaining_units * (price(inv) - price(inv+n))` - a count multiplied by a single price differential - which is the arithmetic the module's own docstring forbids, and it is convex on the `sq` above-curves, so `marginal_rent` is not concave in n.

**Failure.** The denial term is up to 65% of a crop's whole block value (STRAWBERRY at n=25: our revenue $3,847 vs opponent term $7,140), so the allocation is driven mainly by a quantity priced the way `kaggriculture-marginal-is-not-average` says never to price one. And because it is convex on MELON/WOOL, the water-fill's documented precondition ('every score here is concave ... greedy IS the equal-marginal optimum') fails: greedy is then not equal-marginal and `_shadow_price(trace)` - the last accepted marginal, which is fed to `animal_rent` as `labour_price` and prices every animal's chore cost - is not a shadow price.

**Fix.** Price the opponent's loss as a stream, the way our own side is priced: sum over their per-day arrivals `k` of `sell_revenue(item, inv_k, units_k) - sell_revenue(item, inv_k + our_units_by_k, units_k)` using the same inventory path, instead of one differential times a season total. Then assert concavity of `marginal_rent` over a sweep of n in the test suite, since water_fill's correctness depends on it.

```
PYTHONPATH=. .venv/bin/python (day 8, opponent holding 20 MELON + 10 SHEEP, stance BEHIND, params/allocation_v0.json):
 MELON marginal_rent by n_already: [179.0, 181.9, 184.6, 177.6, 179.5, 181.2, 182.5, 174.2, 56.5, 0.5, ...]
 increases at n = [1, 2, 4, 5, 6]        <- not concave
 STRAWBERRY / WHEAT increases at n = []  <- concave, as documented
Share of block_value carried by w_opp*opponent_delta at stance LEVEL, day 11: STRAWBERRY 51% (n=1), 58% (n=10), 65% (n=25); WHEAT 3-4%; CARROT/TOMATO/MELON 0%.
```

## [major] `placenet-failure-latch`

**file** `rlv3/executor.py` line 220 — **demonstrated** True
**rule** The code's own comment at executor.py:219-221 — "a missing checkpoint must not take the agent down; it must be LOUD and inert" — and rlv3/README.md's "An arm that reports as rlv3 while running the shipped agent is DEAD, not null".

**Claim.** `_NET["failed"]` is a module-global latch that `reset_stats()` and `reset_episode()` never clear, and the single `pl_net.failed` counter that records the failure IS cleared at each episode boundary — so one failure in one episode disables `pl_net`/`mv_dest` for every later episode in that worker with no marker at all, while the failure counter itself reads as evidence that the seam fired.

**Failure.** Two failures in one path. (a) Episode 1 loses the checkpoint (a missing Kaggle mount, a torch import, a shape mismatch): `_STATS["pl_net.failed"] = 1`, one stderr line, and both `rlv3/smoke.py:127` and `rlv3/scorecard_hook.py:349` read the non-empty stats dict as "exec seam fired" and raise no failure — a totally dead model passes the package's own liveness check. (b) `reset_stats()` at the start of episode 2 wipes that counter but not the latch, so episodes 2..N in a `session_eval` worker (hundreds of them) run the shipped executor while reporting as the rlv3 arm, with a perfectly clean stats dict. The two levers also share one latch, so an `mv_dest` exception at executor.py:273 silently disables `pl_net` and vice versa.

**Fix.** Clear the latch with the episode: add `_NET["failed"] = False; _NET["model"] = _NET["cond"] = None` to `reset_stats()` (executor.py:153), and keep a sticky `_STATS` key that survives the reset — or re-emit `_STATS["pl_net.disabled"] += 1` on every turn the latch short-circuits `_prior`, so a dead lever shows up as a large counter rather than as silence. Separately, give `pl_net` and `mv_dest` their own latches, and make smoke.py/scorecard_hook treat any `*.failed`/`*.disabled` key as a FAILURE rather than as evidence of liveness.

```
$ PYTHONPATH=. uv run python /tmp/probe4.py   (executor.placenet_load temporarily replaced with a raiser for episode 1 only)
rlv3: pl_net disabled (no checkpoint (kaggle mount))
--- episode 1 (checkpoint unavailable) ---
  stats: {'pl_net.failed': 1}  _NET: {'model': False, 'cond': False, 'failed': True}
  smoke/scorecard liveness predicate `if exec_stats:` -> True (non-empty == 'the seam fired')
--- episode 2 in the SAME worker process ---
  _NET['failed'] still: True  prior returned: None
  stats: {} <- no failure marker at all
  smoke/scorecard liveness predicate -> False
```

## [major] `plan-quota-double-yield`

**file** `rlv3/objective.py` line 613 — **demonstrated** True
**rule** rlv3/objective.py:586-595 '_plan_supply ... Mirrors kagent.planner._rent_board_supply's yield arithmetic exactly'; README Object Types: 'Wheat and Carrot only reach their listed Max Yield of 6 and 4 with fertilizer; watering alone peaks at 4 and 3'

**Claim.** `_plan_supply` books a one-time crop quota at `cd['max_yield']` (the FERTILIZED cap) while the shipped `planner._rent_board_supply` books the identical standing tile at the watering-only yield - a 2x over-credit for WHEAT and CARROT.

**Failure.** `project(plan=...)` credits our side with twice the units a quota tile will actually deliver, so every candidate plan that plants wheat or carrot gets a fabricated revenue bump; the projected margin and hence `p_win` are inflated, and the projection jumps DOWN overnight once the same tiles are standing. This is the number `better()`/`rank()` order the `select` preset by and the number `opponent_bank` is compared against.

**Fix.** Use `_rent_board_supply`'s own one-time arithmetic for a quota tile: units = min(cd['max_yield'], max_yield_day - (max_yield_day+1)//2 + 1) for a tile planted today, i.e. call `_rent_board_supply` on a synthetic tile list rather than re-deriving it, so the docstring's claim ('a planted tile and a quota tile are priced by identical code') becomes true.

```
PYTHONPATH=. .venv/bin/python (probe): _plan_supply for a 20-tile WHEAT quota at day 6 -> {4: {'WHEAT': 120}}; planner._rent_board_supply for the SAME 20 tiles once planted -> {4: {'WHEAT': 60}}. CARROT: fresh tile booked at 4 by _plan_supply vs 2 by _rent_board_supply; MELON agrees at 6.
Outlook.our_bank at day 6: no quota 12,104; +20 WHEAT 14,424 (delta +2,320 against a $200 seed bill); +20 CARROT delta +1,418 against a $400 seed bill.
```

## [major] `plan-supply-does-not-mirror-board-supply`

**file** `rlv3/objective.py` line 613 — **demonstrated** True
**rule** Its own docstring: "Mirrors `kagent.planner._rent_board_supply`'s yield arithmetic exactly -- same first_yield_day, same interval stepping ... so a planted tile and a quota tile are priced by identical code." README "Animal Care": a fed+cared animal's scheduled production is `1 + banked bonus`, i.e. `(1 + interval)` units every `interval` days.

**Claim.** `_plan_supply` does not mirror `_rent_board_supply`. (a) one-time crops: `out[k][crop] += n * cd["max_yield"]` gives 6 units per wheat tile where `_rent_board_supply` gives 4 and the engine delivers 4 unfertilised. (b) animals (line 624): one unit per scheduled production, i.e. 1/interval per animal-day, where `_rent_board_supply:62-68` uses rate 2 for a cared animal and `features._OUT_PER_DAY` is 2.0 / 1.5 / 1.333 — the engine's fed+cared steady state, which the shipped scheduler achieves because it emits a CARE task for every fed animal every day.

**Failure.** Inside one call to `objective.project()` the SAME asset is priced two different ways depending on whether it is already on the board or is in the plan: a wheat tile the plan wants is worth 50% more than the identical tile standing, and a cow the plan wants is worth HALF the identical cow standing — while its $400 cash cost is charged in full. `better()` / `rank()` / `score()` therefore systematically prefer planting one-time crops and refuse livestock, which is exactly the ranking the `select` preset acts on and the number every recorded Outlook reports.

**Fix.** Line 613: `out[k][crop] += n * min(cd["max_yield"], 1 + cd["max_yield_day"] - ((cd["max_yield_day"] + 1)//2 - 1))`. Line 624: `out[k][a["product"]] += n * 2` to match `_rent_board_supply`'s cared rate (or read `features._OUT_PER_DAY[a["product"]] * a["interval"]`). Better: call `_rent_board_supply` on a synthetic tile list built from the quota so there is only one implementation.

```
$ PYTHONPATH=. uv run python /tmp/t_plan_supply.py
rlv3 _plan_supply, plant_quota {WHEAT: 10}:
    {4: {'WHEAT': 60}}
kagent _rent_board_supply, the SAME 10 wheat tiles standing (age 0):
    {4: {'WHEAT': 40}}

rlv3 _plan_supply, animal_buys {COW: 4}:  MILK by day-offset {8: 4, 10: 4, 12: 4, 14: 4}  => 0.29 milk/cow/day
kagent _rent_board_supply, 4 cared COWs placed the same day: MILK {8: 8, 10: 8, 12: 8, 14: 8}  => 0.57 milk/cow/day
   features._OUT_PER_DAY['MILK'] = 1.5

$ PYTHONPATH=. uv run python /tmp/t_animal.py    # engine, FEED+CARE+HARVEST every day
GOOSE   fed+cared:  44 units over 21 productive days = 2.095/animal/day | rlv3 _plan_supply rate = 1.000
COW     fed+cared:  30 units over 17 productive days = 1.765/animal/day | rlv3 _plan_supply rate = 0.500
SHEEP   fed+cared:  30 units over 19 productive days = 1.579/animal/day | rlv3 _plan_supply rate = 0.333
```

## [major] `projstate-ewma-off-by-one-day`

**file** `rlv3/objective.py` line 1259 — **demonstrated** True
**rule** rlv3/objective.py:900-918 'Yesterday's one-day-ahead forecast is compared against today's actuals ... the residual drives a bounded EWMA on exactly TWO multipliers'

**Claim.** `ProjState._update` pairs yesterday's realised market flow with TOMORROW's projected arrivals: `fc['opp_implied']`/`fc['our_implied']` are `_rent_board_supply(...)[1]` computed at dawn D-1 (units that become harvestable on day D), while `_day_opp`/`_day_our` are the flows solved over day D-1. It also records only block k=1 although `_forecast` prices blocks k=0 AND k=1.

**Failure.** The only fitted quantity in the whole package - the multiplier that scales the opponent's projected supply in `opponent_bank`/`project` - is a ratio of two different days' quantities. The correction is a $10-20k lever on the opponent's projected bank (day 21: 105,536 with the correction vs 85,763 without), so the mispairing moves exactly the number the module exists to get right.

**Fix.** Either hold `self._fc` one extra dawn so `_update` compares flow(day D) against the arrivals forecast for day D, or set `our_implied`/`opp_implied` to the supply the forecast actually priced for that window (`mine[0] + mine[1]`, not `mine[1]` alone). Add a test that a steady-state board with a known daily arrival and a known daily sale converges to conversion == 1.0.

```
PYTHONPATH=. .venv/bin/python /tmp/probe3.py then an offline replay of the EWMA over the recorded (realised, implied) pairs.
Correlation of the day's realised flow with the implied arrivals it is divided by:
 item        as coded    re-aligned one day back
 MILK          +0.066        +0.710
 WOOL          -0.266        +0.185
 STRAWBERRY    +0.746        +0.865
(strawberry+milk+wool are 97% of the shared-inventory cost)
Resulting multipliers: conversion CARROT 1.212 as coded vs 0.670 aligned (1.81x); our fill WHEAT 0.494 vs 1.194, MILK 1.133 vs 0.580.
```

## [major] `pwin-conditioned-on-money-lead`

**file** `rlv3/objective_check.py` line 613 — **demonstrated** True
**rule** rlv3/objective.py `p_win` docstring: "Empirical P(final margin > 0 | projected margin at this day)" — the builder conditions on something else. rlv3/README.md: "1. Win rate ... This dominates."

**Claim.** `rlv3/calibration.json` is fitted conditioning on the raw MONEY LEAD at day d (`m = v0.me.money - v0.opp.money`), but `objective.Outlook` feeds `p_win` the PROJECTED FINAL margin — a quantity an order of magnitude larger in the early season — so the table's clamp swallows the whole early-season range and the objective's win clause is inert exactly in the window the allocation dose acts in.

**Failure.** The shipped table's 2nd-98th-percentile clamps are ±$1,177 at day 4 and ±$884 at day 8 (money-lead scale; `_LEAD_SD_BY_DAY` puts the day-6 lead sd at $363), while `objective.project(...).margin` at day 6 on the recorded replays runs ±$10,132 to ±$13,617. Every early-season margin therefore clamps to the same row endpoint. Over the three lam-0 / lam-1 candidate-plan pairs at day 6, `|p_win(a) - p_win(b)|` is exactly 0.0000 in all six seat-games, so `better()` never reaches clause 1 and degenerates into "higher projected margin" — the money-first comparator the whole package exists to replace. `objective.py`'s own docstring for `p_win` says the conditioning variable is "projected margin at this day", which is what hid the mismatch.

**Fix.** Condition the table on the same statistic the comparator passes it: in `_build_calibration`, replace `m = v0.me.money - v0.opp.money` with `m = O.project(v0).margin` (and `-O.project(v1).margin` for seat 1, or project from each seat's own View), then rebuild `rlv3/calibration.json`. Add an assertion in `objective_check` that the day-4 and day-8 clamps cover at least the 2nd-98th percentile of projected margins actually seen at those days.

```
$ PYTHONPATH=. uv run python /tmp/probe18.py
   day  4  clamp [-1177.0, 1177.0]  source=measured on 524 seat-games
   day  8  clamp [-884.0, 884.0]    source=measured on 524 seat-games
case            money lead   PROJECTED margin         clamp at that day   p_win  outside?
mmpq-d6-s0            -250            -13,617         [-1177.0, 1177.0]   0.469  OUT
mmpq-d6-s1             250             13,617         [-1177.0, 1177.0]   0.498  OUT
dsm-d6-s0           -2,038             10,132         [-1177.0, 1177.0]   0.498  OUT
dsm-d6-s1            2,038            -10,132         [-1177.0, 1177.0]   0.469  OUT

$ PYTHONPATH=. uv run python /tmp/probe19.py   (lam0 vs lam1 plans through AllocationPlanProvider, band 5)
mmpq-d6-s0         0.4981     0.4981   0.0000       0.01  clause 2/3 (MARGIN ONLY)
mmpq-d6-s1         0.4981     0.4981   0.0000       0.01  clause 2/3 (MARGIN ONLY)
boey-d6-s0         0.4981     0.4981   0.0000       0.01  clause 2/3 (MARGIN ONLY)
boey-d6-s1         0.4981     0.4981   0.0000       0.01  clause 2/3 (MARGIN ONLY)
dsm-d6-s0          0.4981     0.4981   0.0000       0.01  clause 2/3 (MARGIN ONLY)
dsm-d6-s1          0.4688     0.4688   0.0000       0.01  clause 2/3 (MARGIN ONLY)
rankings decided by the win clause: 8/18  (0/6 at day 6)
```

## [major] `pwin-fitted-on-money-lead`

**file** `rlv3/objective_check.py` line 613 — **demonstrated** True
**rule** rlv3/objective.py p_win docstring: 'Empirical P(final margin > 0 | projected margin at this day)'; kaggriculture-observational-vs-interventional -- a fitted object must be validated on the distribution it will be evaluated on.

**Claim.** The p_win table is FITTED on the raw bank lead (`v0.me.money - v0.opp.money`) but is QUERIED at play time with Outlook.margin, the projected FINAL margin (our_bank - opp_bank including the whole season's projected board revenue). The two live on different scales, so the table's margin axis and its per-day clamp do not match the values it is evaluated on.

**Failure.** p_win saturates its clamp and stops discriminating. calibration.json's day-0 clamp is [0.0, -0.0] -- the signature of a fit on the money lead, which is identically 0 at day 0 because both seats start at $3,000 -- and days 4 and 8 clamp at +/-1177 and +/-884 (the 'everybody is broke until day 9' lead spread). Measured on real dawn states, 100% of projected margins at days 4 and 8 fall outside that clamp, and at day 8 p_win returns ONE distinct value across every state. `better()`'s clause 1 (dp > DEAD_BAND) is therefore blind for roughly the first 11 dawns of every game, and `score()` (the scalar a bandit or value target would consume) is a step function of the sign. The same mismatch is in the analytic fallback: `_analytic_row` divides by `_lead_sd(day)` (objective.py:426), a MONEY-LEAD sd table whose own comment says it was read off check_sign's 'lead sd $' column.

**Fix.** Fit the table on the quantity it is queried with: in build_calibration replace `m = v0.me.money - v0.opp.money` with `m = O.project(v0).margin` (and `-O.project(v1).margin` for seat 1, or project each seat's own view), so the margin knots and the 2nd-98th-percentile clamp are computed over the projected-final-margin distribution. Do the same for `_LEAD_SD_BY_DAY` / `_lead_sd` if the analytic fallback is to remain usable. Keeping the money-lead fit is also defensible, but then p_win must be called with the money lead, not with Outlook.margin -- one or the other, not both.

```
Fit side, rlv3/objective_check.py:613-614:
    m = (float(v0.me.money) - float(v0.opp.money))
    samples[day].append((m if seat == 0 else -m, win))
Use side, rlv3/objective.py:545 (Outlook.__post_init__): `p_win(self.margin, self.day, self.band)` where margin = our_bank - opp_bank from project().

Clamps in the shipped table (`python3 -c "json.load(open('rlv3/calibration.json'))"`):
    day  clamp_lo   clamp_hi
      0       0.0       -0.0     measured on 524 seat-games
      4   -1177.0     1177.0
      8    -884.0      884.0
     12  -15089.0    15089.0

Scale check on real dawn states from 3 recorded replays -- /tmp/rlv3rev/p3.py, `PYTHONPATH=. uv run python /tmp/rlv3rev/p3.py`:
     day   n  |proj| med  |lead| med   clamp hi  % proj outside clamp  distinct p_win
       0   6           0           0         -0                    0%               1
       4   6       4,456         954      1,177                  100%               2
       8   6       6,559         655        884                  100%               1
      12   6       6,143       5,742     15,089                    0%               6
```

## [major] `risk-posture-asymmetric`

**file** `rlv3/selling.py` line 335 — **demonstrated** True
**rule** The function's own docstring: "both sides use the same board valuation so the comparison is not biased by the missing term beyond the shed itself" — which is false as written.

**Claim.** `risk_posture` values our side with `rl.rollout.portfolio_value` and their side with `money + _board_value` — two completely different valuations — so it is not antisymmetric and reports "losing" almost always; the `posture > 0` winning branch of `plan_slices` is effectively dead code.

**Failure.** On a board where both farms are tile-for-tile identical, both banks are $50,000 and both sheds are empty, `risk_posture` returns -0.242, and it returns the SAME value from seat 1 (a lead measure must negate). On the 18 recorded mid-season states it saturates at exactly -1.000 in 14 of 18, including both seats of the same world at once; against a same-valuation lead the sign disagrees in 8 of 18. Consequence in `plan_slices` (selling.py:461-476): the `posture > 0` branch that tightens the impact cap and takes certain cash never runs, and the `posture < 0` branch that holds every PACED product above `sell_floors * behind_hold` runs every step — the "paced selling above a floor let the opponent sell first and lost $30k" behaviour the module docstring says the record is hostile to, wired permanently on. `_board_value` prices FERTILIZER at ~$100 for every animal for every remaining day while `portfolio_value` caps animal value at 8 days and books no fertilizer at all, which is where the bias comes from. Nothing would notice: `rlv3/sell_eval.py`'s `make_rlv3_sizer` deliberately replays only the participation and impact caps and never calls `risk_posture`, so the +$891 / +$1,040 pre-gate numbers quoted in selling.py and the README were measured with this branch absent.

**Fix.** Value both sides with one function. Drop the `rl.rollout` import and compute `mine = view.me.money + _board_value_of(view, view.me.tiles)` and `theirs = view.opp.money + _board_value_of(view, view.opp.tiles)`, i.e. generalise `_board_value` to take a tile array (it already only needs `view.day`, `view.market_prices` and the tiles). Add a unit test that `risk_posture` is 0 on a mirrored observation and negates when `obs["player"]` flips.

```
$ PYTHONPATH=. uv run python /tmp/probe8.py
IDENTICAL farms, identical money, empty sheds
  our side  (rollout.portfolio_value)      = 51,534
  their side(money + _board_value)         = 56,380
  lead = -4,846   risk_posture = -0.242   -> branch = LOSING (hold)
  seat 1 on the same board: risk_posture = -0.242  (a symmetric measure would be the negation of -0.242)

$ PYTHONPATH=. uv run python /tmp/probe9.py   (18 recorded mid-season views)
case           shipped posture  same-valuation   sign flip?
mmpq-d6-s0              -1.000          -0.668
mmpq-d6-s1              -1.000          +0.668   YES
mmpq-d12-s0             -1.000          -0.475
mmpq-d12-s1             -1.000          +0.475   YES
mmpq-d20-s0             -1.000          +0.199   YES
... sign flips: 8 / 18   (and -1.000 in 14 of 18)
```

## [major] `risk-posture-asymmetric-valuation`

**file** `rlv3/selling.py` line 356 — **demonstrated** True
**rule** rlv3/selling.py:335-346 'both sides use the same board valuation so the comparison is not biased by the missing term beyond the shed itself'

**Claim.** `risk_posture` compares OUR `rl.rollout.portfolio_value` (a discounted liquidation value: 0.6x price on held units, 0.3x on potential, animals capped at 8 days) against THEIR `_board_value` (the whole remaining season's arrivals at FULL market price), so the two sides are valued by different functions.

**Failure.** On an exactly symmetric position - identical boards, identical banks, empty shed - `risk_posture` returns -1.000 (fully 'losing', saturated at the clamp) on days 6, 12 and 18. In `plan_slices` that permanently selects the `posture < 0` branch, which holds every PACED product (STRAWBERRY/MILK/WOOL/MELON) back one extra tick above `behind_hold * sell_floor`, and makes the `posture > 0` 'take certain cash in small low-impact slices' branch unreachable before about day 22. The module's own docstring says the repo record is hostile to holding ('paced selling above a floor let the opponent sell first and lost $30k').

**Fix.** Value both sides with one function. Either price our board with `_board_value(view, view.me.tiles)` as well (and add our shed on top), or discount their `arrival_path` the same way `portfolio_value` discounts ours. Add a unit test that a mirrored board with equal banks gives |risk_posture| < 0.05.

```
PYTHONPATH=. .venv/bin/python (identical me/opp farms: 12 STRAWBERRY + 10 WHEAT + 6 MELON + 4 COW + 3 SHEEP, $20,000 each, empty shed)
 day  6  ours 28,737  theirs 72,135  posture -1.000
 day 12  ours 28,737  theirs 61,695  posture -1.000
 day 18  ours 28,737  theirs 49,815  posture -1.000
 day 24  ours 26,297  theirs 26,055  posture +0.012
```

## [major] `scorecard-trace-cache-tag-collision`

**file** `rlv3/scorecard_hook.py` line 160 — **demonstrated** True
**rule** rlv3/scorecard_hook.py's own module docstring, point 2: "a dead arm cannot be read as a bad one" — and rlv3/fingerprint.py's premise, "'the output file is there' is indistinguishable from 'the output file is right'", which this module does not apply to its own cache.

**Claim.** Cached traces are keyed on `_tag(label)` — the preset name alone — and `_usable()` validates only world, seat and step count, so any spec override (`lam`, `sell`, `exec_flags`, `params`, `land_theta`, `band`) silently reuses a different arm's traces, and because nothing is replayed the liveness check is skipped too.

**Failure.** `--agent rlv3:full:lam=0.5` reads the four `rlv3_full_*.json` traces that `rlv3:full` (lam=1.0) wrote, plays zero games, prints the old arm's scorecard under the new arm's name, and reports "all traces cached -- no liveness counted this run" (scorecard_hook.py:344) — so the one check the package exists to make is also skipped. Any dose ladder run through this hook (`--agent rlv3:full:lam=0.33`, `0.67`, `1.0`) produces four identical columns and reads as "the dose does nothing". The same collision laundered a DEAD arm into a cached one: a run whose `made < 29` liveness assertion fired leaves its already-written traces on disk for the next invocation to accept.

**Fix.** Put a spec digest in the filename and in the trace: make `tag = _tag(label) + "_" + sha1(json.dumps(spec, sort_keys=True))[:8]`, and have `_usable` additionally require the cached trace's recorded spec digest to match. A cheaper stopgap: refuse `--agent rlv3:<preset>:<overrides>` unless `--fresh` is given.

```
$ PYTHONPATH=. uv run python /tmp/probe3.py
rlv3:full                    label=rlv3:full  tag=rlv3_full  cached_hits=4/4  lam=None sell=None exec=[...,'pl_wheat_adj','co_chain_tile']
rlv3:full:lam=0.5            label=rlv3:full  tag=rlv3_full  cached_hits=4/4  lam=0.5  sell=None exec=[...,'pl_wheat_adj','co_chain_tile']
rlv3:full:exec_flags=[]      label=rlv3:full  tag=rlv3_full  cached_hits=4/4  lam=None sell=None exec=['lastwater','h23','harvest_visit']
rlv3:full:sell=off           label=rlv3:full  tag=rlv3_full  cached_hits=4/4  lam=None sell=off  exec=[...,'pl_wheat_adj','co_chain_tile']
```

## [major] `sell-eval-drain-window-off-by-one`

**file** `rlv3/sell_eval.py` line 118 — **demonstrated** True
**rule** Engine `interpreter`: `_town_consume(env, state, step)` runs with the PRE-increment counter and drains when `step % interval == 0`, so the tick fired during step t is first visible in the observation at t+1. `market_model.ticks_between` implements exactly that: multiples of T in `[a, b)`.

**Claim.** `attach_opponent_fills` asks for the drain with `mm.drain_between(item, s["t"] - 1, s["t"], s["shops"])`, which counts the tick at step t-1, while the identity it is solving (`inv[t+1] = inv[t] + fills - drain(step t) - buys`) needs the tick at step t — i.e. `drain_between(item, s["t"], s["t"] + 1, ...)`. `market_model.drain_between`'s own docstring is the likely cause: it advertises "(step_a, step_b]" while the `ticks_between` it calls documents and implements "[step_a, step_b)".

**Failure.** The opponent-fill column the module calls "EXACT, not a bound" is wrong by one drain tick on every step, and that column is the kill rule of the whole selling pre-gate (`opp_delta_pct` vs `delta_pct`) — the pre-gate that produced the `+$891 full schedule vs +$1,281 slots_only` numbers quoted in rlv3/README.md, selling.py and agent.py to decide which sell mode ships. On the shipped champ-h10 band-0 trace the solved opponent supply moves by 102 of 1,194 units (8.5%): STRAWBERRY +30, WHEAT +68.

**Fix.** `drain = mm.drain_between(item, s["t"], s["t"] + 1, s["shops"])`. Also correct `market_model.drain_between`'s docstring from "(step_a, step_b]" to "[step_a, step_b)" so the next caller does not repeat it.

```
$ PYTHONPATH=. uv run python /tmp/t_drain.py    # rows where NEITHER player ordered the product,
                                                # so d_inv must equal -drain exactly
rows with no order on the product: 5814
  |d_inv + drain| with sell_eval's (t-1, t) :     4419  (0 == exact)
  |d_inv + drain| with        (t, t+1)      :        0

$ PYTHONPATH=. uv run python /tmp/t_drain2.py   # solved opponent fill, both windows
champ_h10_0_b0.json
   item          as coded  window fixed    diff
   STRAWBERRY         170           200      30
   WHEAT              487           555      68
   TOTAL             1194          1296     102
```

## [major] `sell-eval-drain-window-off-by-one`

**file** `rlv3/sell_eval.py` line 118 — **demonstrated** True
**rule** rlv3/sell_eval.py:86-102 'inv[t+1] = inv[t] + my_fills + their_fills - drain(step t) - buys ... their fill is EXACT, not a bound'; rlv3/market_model.py:67-89 ticks_between counts [a, b)

**Claim.** `attach_opponent_fills` subtracts `drain_between(t-1, t)` from the inventory delta `inv[t+1] - inv[t]`, but that delta is closed by the tick fired during step t, i.e. `drain_between(t, t+1)`.

**Failure.** The opponent's solved fills - the column the pre-gate exists to make trustworthy - are wrong on exactly the steps where a drain tick fires (every 4th step, and the compound tick at hour 0, which is where the sells cluster). The error UNDER-states the opponent: 1,194 solved units against the correct 1,296 over one traced game, -68 WHEAT and -30 STRAWBERRY. A schedule that lifts the opponent's revenue therefore looks better than it is.

**Fix.** `drain = mm.drain_between(item, s['t'], s['t'] + 1, s['shops'])`. (rlv3/objective.py's `_drain_between(prev_step, step)` already uses the right convention for its own delta; the two should be checked against each other in a test.)

```
PYTHONPATH=. .venv/bin/python (probe over rlv3/eval/scorecard_traces/champ_h10_0_b0.json, restricted to (item, step) pairs with NO sell/buy orders from either side, where inv[t+1]-inv[t] must equal -drain(step t) exactly):
 quiet (item,step) pairs checked: 2412
 mismatches with drain_between(t-1, t)  [sell_eval's call]: 516
 mismatches with drain_between(t,   t+1): 0
Re-solving the fills with the correct window:
 item        as coded   correct   diff
 WHEAT            487       555    +68
 STRAWBERRY       170       200    +30
 total           1194      1296   +102
```

## [major] `sell-slots-evict-buys-past-order-cap`

**file** `rlv3/selling.py` line 535 — **demonstrated** True
**rule** README "Market Action": "Each turn you can submit up to `maxMarketOrdersPerTurn` (default 10) market actions; any orders past that limit are silently dropped." Engine `_process_market`: `queues.append(q[:max_orders])`. Its own docstring claims "Its non-SELL orders are kept in their order" and that `slots_only` is "free: it changes no walking, no allocation, and no sizing".

**Claim.** `build_orders` rebuilds the list as `hires + lead(<=3 SELLs) + rest + tail` and then truncates to `MAX_MARKET_ORDERS`, so the leading SELLs push the shipped builder's BUY_PRODUCT / BUY_LAND / BUY_ANIMAL / BUY_SEED orders off the end of the engine's 10-order window. Worse, `kagent.market.build_orders` has already mutated the plan for those orders (`plan.land_buys = remaining_land`, `plan.animal_buys[kind] = want - n`), and `HeuristicPolicy.act` reuses the SAME `memory.day_plan` object for all 24 steps of the day — so the purchase is recorded as spent and never re-requested.

**Failure.** On by default: `RECOMMENDED = SellConfig(slots_only=True)` is what presets `recommended` and `full` ship. Replayed against the three leaders' recorded boards at every hour-0 dawn, 109 of 138 dawns drop at least one non-SELL order the shipped builder issued. 55 dawns lose a BUY_ANIMAL and 3 lose a BUY_LAND that the day plan has already decremented, so the cow/quadrant is never bought that day at all; dropped feed BUY_PRODUCT risks the two-unfed-day escape rule. CHAMP_HIRES is 11 for days 10-25 and the shipped builder emits up to 8 HIREs at hour 0, so 8 HIRE + 3 SELL = 11 > 10 makes this fire on essentially every mid-season dawn.

**Fix.** Bound the lead block by the slots the shipped builder actually gave to SELLs: `n_lead = min(cfg.sell_slots_lead, MAX_MARKET_ORDERS - len(hires) - len(rest))` before slicing `lead_pool`, so the re-order only permutes slots that were already going to sells. Anything still evicted must be pushed back onto `plan.land_buys` / `plan.animal_buys` so the day plan does not record an unplaced order as spent.

```
$ PYTHONPATH=. uv run python /tmp/t_orders3.py
hour-0 dawns where rlv3 slots_only DROPS a non-SELL order the shipped builder issued: 109/138
  mmpq day 7 seat 0
   shipped: [HIRE x8, ['BUY_ANIMAL','COW',1], ['BUY_SEED','STRAWBERRY',1]]
   rlv3   : [HIRE x8, ['SELL','WOOL',2], ['SELL','WHEAT',3]]
   dropped: [['BUY_ANIMAL','COW',1], ['BUY_SEED','STRAWBERRY',1]]

$ PYTHONPATH=. uv run python /tmp/t_orders4.py
mmpq day 7 seat 0: plan.land_buys ['SW'] -> ['SW'], animal_buys {'COW': 3} -> {'COW': 2}
   orders actually submitted: [HIRE x8, ['SELL','WOOL',2], ['SELL','WHEAT',3]]
dawns where a BUY_LAND was consumed from the plan but never submitted : 3
dawns where a BUY_ANIMAL was consumed from the plan but never submitted: 55
```

## [major] `strawberry-unfertilised-rate-in-demand-cap`

**file** `rlv3/rent.py` line 138 — **demonstrated** True
**rule** `rent.per_tile_day`'s own docstring: "`features._PER_TILE_DAY` where it has an entry (strawberry's 0.5 is the fertilised+watered rate, twice the bare formula)". In fact `features._PER_TILE_DAY["STRAWBERRY"] = 0.235` (the watered-only rate, README's 0.24 column); the 0.5 lives in the separate `features._STRAW_PER_TILE_DAY`, which is what `features.anchors()` divides strawberry demand by.

**Claim.** `per_tile_day("STRAWBERRY")` returns 0.235, not the 0.5 its docstring claims. `allocation.demand_caps:326` divides the strawberry demand net of the opponent's supply by that number, so it permits 2.13x the standing strawberry tiles the shipped anchor permits for identical demand. The convention is also inconsistent within the same function: two lines later, animals are divided by `features._OUT_PER_DAY` — the FED+CARED rate — while strawberry gets the UNFERTILISED rate, and the shipped scheduler both cares every animal and fertilises strawberry at ages 9-15 as standard (`kagent/scheduler.py:348`).

**Failure.** The demand cap is the one thing stopping the water-fill from re-creating the corner solution allocation.py exists to fix — its docstring opens with "champ-h10 runs 41.2 strawberry tiles at day 20 where the three leaders run 22.5 / 23.5 / 12.75". Loosening exactly that cap by 2.13x means the anchor cannot bind on strawberry before the hard ceiling of 40 does, so the dose can only re-find the 41-tile board.

**Fix.** In `per_tile_day`, special-case strawberry the way `features.anchors` does: `if crop == "STRAWBERRY": return float(features._STRAW_PER_TILE_DAY)` before the `_PER_TILE_DAY` lookup — or state one convention and apply it to animals too (if strawberry is priced unfertilised, animals must be priced uncared at `1/interval`, not `_OUT_PER_DAY`).

```
$ PYTHONPATH=. uv run python /tmp/t_straw.py    # engine, watered daily, fertilised ages 9-15
strawberry water only: 4 units over 17 days = 0.235 /tile/day
strawberry FERT: 8 units over 17 days = 0.471 /tile/day

features._PER_TILE_DAY['STRAWBERRY'] = 0.235
features._STRAW_PER_TILE_DAY        = 0.5    <- what features.anchors() divides strawberry demand by
rent.per_tile_day('STRAWBERRY')     = 0.235  <- what allocation.demand_caps divides it by
   -> rlv3's strawberry cap is 2.13x the shipped anchor's for identical demand
```

## [major] `terminal-reward-never-wired`

**file** `rlv3/objective.py` line 251 — **demonstrated** True
**rule** rlv3/objective.py:14-19 '(a) TERMINAL REWARD -- one scalar per episode, for PPO'; :262 'This replaces the reward loop of rl.train._episode_advantages wholesale -- rlv3's own trainer calls it'

**Claim.** `objective.terminal_reward` / `episode_rewards` - half (a) of the objective, the asymmetric clause-2/clause-3 reward - is called by nothing but rlv3's own unit tests. The PPO stage runs `rl/train.py --reward win`, whose margin term is a single symmetric `MARGIN_TIE * tanh(margin / MARGIN_SCALE)`.

**Failure.** The 3.1x asymmetry that implements 'closing a loss is worth three times growing a lead' never reaches training, and `objective.fingerprint()` is never written into a checkpoint despite the docstring claiming 'The trainer compares this string and refuses the resume on a mismatch'. Worse, the only thing the stage graph ever gates and promotes is `{out}/ppo/eval.json` - a neural net - so the rlv3 AGENT (allocation, selling, executor) is never scored through the lexicographic objective by the pipeline at all.

**Fix.** Either wire it (pass `--margin-scale`/a reward hook, or add an rlv3 trainer that calls `episode_rewards`) or delete the claim: correct the `episode_rewards` docstring ('rlv3's own trainer calls it'), the `fingerprint()` docstring, and the README's 'every component is scored through it'. At minimum record `objective.fingerprint()` in the manifest so a promoted checkpoint says which objective judged it.

```
grep -rn 'episode_rewards|terminal_reward' kagent rl rlv3 tools tests arena -> only rlv3/objective.py and rlv3/tests/test_objective.py.
rlv3/jobs/rlv3_train.json ppo stage cmd: `rl/train.py --resume {out}/bc/bc.pt --run {out}/ppo --iters 24 ... --reward win ...`; gate arm `cand={out}/ppo/eval.json`; promote arm 'cand'.
rl/train.py:431 `tie = MARGIN_TIE * math.tanh(margin / MARGIN_SCALE)` - one scale for wins and losses, no LOSS_SCALE/WIN_SCALE split.
grep -rn 'OBJ.fingerprint' rlv3 -> agent.py:213 (stripped again by make_job's js.pop('objective')), objective.py:1279, smoke.py:89. No stamp, no manifest, no checkpoint.
```

## [major] `training-graph-pins-v1-action-space`

**file** `rlv3/jobs/rlv3_train.json` line 33 — **demonstrated** True
**rule** kagent/features.py:255-268 (v2 respacing note): "v1's grids are spaced for large counts and cannot say any of the low single digits where most of the mass sits: geese 0/4/6/10/16 cannot say 2, 3, 5 or 7 (48% of leader dawns), carrots 0/6/12/18/26 cannot say 1-4 (58%) ... and hires 4/8/12/14/16 cannot say 9, 10 or 11, which is where every leader sits."

**Claim.** The rlv3 stage graph passes no `--heads-version` to `clone`, `bc` or `ppo`, so all three run `features.HEADS_VERSION = 1`, whose quantity grids cannot express the counts the top-band players the `clone` stage is cloning actually run — the defect `features.ABS_HEADS_V2` was written on 2026-09-26 to fix, and which rlv3 never selects.

**Failure.** The `clone` stage is the one intervention measured to lift the RL net (0.074-0.080 -> 0.127-0.180, `kaggriculture-top-player-bc-lifts-rl`), and its labels are quantised by `rl.replay_bc._targets_to_choices` onto v1's grids before training starts. Measured over 2,320 dawns from the 40 recorded leader replays: `hires` [4,8,12,14,16] reproduces the true hire count on 19.1% of dawns — the modal leader value is 11 (887/2320 = 38%) and the grid cannot say 9, 10 or 11 at all; `goose_target` [0,4,6,10,16] 44.1%; `melon_tiles` [0,6,12,18,24] 54.5%; `carrot_tiles` [0,6,12,18,26] 59.9% (the leaders' carrot mass is 1-4). v2's respaced grids raise these to 33.8 / 61.9 / 61.0 / 63.1%. Worse, `rl/session_bc.py` has no `--heads-version` flag at all, so the corpus is pinned to v1 even if a later operator switches the trainer.

**Fix.** Pin the action space in the job graph: add `--heads-version 2` to the `bc` and `ppo` commands, and run the `clone` stage through a wrapper that calls `features.set_heads_version(2)` before `rl/session_bc.py`'s main (an rlv3-only shim, since session_bc takes no flag). Pin it in one place — a `heads_version` field on the job JSON that both `rlv3.stages` and `rlv3.fingerprint` read — so clone/bc/ppo/gate cannot disagree. Do not change one stage without the others: the stored `choices` indices mean different counts under each version.

```
PYTHONPATH=. .venv/bin/python -c "from kagent import features; print(features.HEADS_VERSION)"  -> 1   (rlv3 never calls set_heads_version; grep -rn 'heads.version' rlv3/stages.py rlv3/train.py rlv3/jobs/ -> nothing)
PYTHONPATH=. .venv/bin/python rl/session_bc.py --help  -> options are only --n --procs --seed --arm --out --include-held-out --min-rating; there is no --heads-version.
PYTHONPATH=. .venv/bin/python /tmp/quant_check.py (2,320 leader dawns, both seats, 40 replays; quantiser = rl.replay_bc._nearest_valid, the one session_bc uses):
  v1  hires        [4,8,12,14,16]   19.1% exact  mean|err| 1.04   modal true [(11,887),(10,343),(12,307),(9,216),(6,170)]
  v1  goose_target [0,4,6,10,16]    44.1% exact  mean|err| 0.79   modal true [(0,563),(2,352),(6,225),(3,208),(4,184)]
  v1  melon_tiles  [0,6,12,18,24]   54.5% exact  mean|err| 0.79
  v1  carrot_tiles [0,6,12,18,26]   59.9% exact  mean|err| 0.82
  v2  hires        [4,8,10,12,16]   33.8% | goose [0,2,3,5,8] 61.9% | melon [0,4,8,12,20] 61.0% | carrot [0,2,4,9,17] 63.1%
```

## [minor] `alloc-test-asserts-pass2-always-runs`

**file** `rlv3/tests/test_allocation.py` line 818 — **demonstrated** True
**rule** rlv3/README.md 'Running it': `PYTHONPATH=. uv run pytest rlv3/tests/ -q` is the wiring gate.

**Claim.** test_water_fill_trace_names_the_crop_the_direction_is_wrong_on asserts `alloc.trace['fill_pass2_from'] is not None`, but at the state it builds the first water-fill already consumes the entire board budget, so the bare-ground second pass correctly never runs and the key is correctly None. The assertion is false, not the code.

**Failure.** `PYTHONPATH=. uv run pytest rlv3/tests/ -q` -- the documented 'the wiring, no games' check and the only zero-cost gate this package has -- is RED (1 failed, 364 passed). A permanently failing suite cannot be used to catch a real regression in the 364 tests around it.

**Fix.** Make the assertion conditional on the anchor actually binding -- assert pass2_from is not None only when `sum(alloc.trace['share_crop'].values()) < capacity['standing'] + alloc.crop_capacity` -- or move it to a state whose demand caps bind below the board budget (e.g. tighten crop_caps or widen the farm). Do not relax the monotonicity checks above it, which pass.

```
`PYTHONPATH=. uv run pytest rlv3/tests/ -q` ->
  FAILED rlv3/tests/test_allocation.py::test_water_fill_trace_names_the_crop_the_direction_is_wrong_on - assert None is not None
  1 failed, 364 passed in 9.33s
The state, via /tmp/rlv3rev/p8.py (same _pair call as the test):
  crop_capacity 49  standing 26  unlocked 75
  board_budget = 75
  share_crop {'WHEAT': 23, 'CARROT': 7, 'TOMATO': 12, 'STRAWBERRY': 31, 'MELON': 2}  sum 75
  fill_pass2_from None   len(fill_crop) 75
allocation.allocate only sets pass2_from when `sum(share_crop.values()) < board_budget`; here 75 == 75, so pass 1 saturated and pass 2 is a no-op by design.
```

## [minor] `allocation-test-red`

**file** `rlv3/tests/test_allocation.py` line 818 — **demonstrated** True
**rule** rlv3/README.md "Running it": `PYTHONPATH=. uv run pytest rlv3/tests/ -q` is the package's own gate.

**Claim.** The rlv3 test suite ships red: `test_water_fill_trace_names_the_crop_the_direction_is_wrong_on` fails because `alloc.trace["fill_pass2_from"]` is None — the bare-ground second pass never runs on the state its own test was written for.

**Failure.** 364 pass, 1 fails, so `pytest rlv3/tests/ -q` — the command the README gives as "the wiring, no games" — exits non-zero, and anyone adding a test to this package has to decide whether their own red is new. The cause is that on that state the demand anchor does not bind at all (`sum(share_crop) == board_budget == 75`), because the synthetic opponent farm is empty so `rent.opp_supply` is zero and `demand_caps` returns near-`ANCHOR_CAPS` ceilings (STRAWBERRY 39 against a hard cap of 40). The test asserts the pass-2 branch is exercised; it cannot be on a state whose first pass already saturates.

**Fix.** Either give the test a state where the anchor binds (put crops on the opponent's board so `demand_caps` cuts below `board_budget`, e.g. add ~25 standing STRAWBERRY to `view.opp`), or split the assertion: keep the monotonicity check unconditional and assert `fill_pass2_from is not None` only when `sum(alloc.trace["share_crop"].values()) < board_budget`.

```
$ PYTHONPATH=. uv run pytest rlv3/tests/ -q
FAILED rlv3/tests/test_allocation.py::test_water_fill_trace_names_the_crop_the_direction_is_wrong_on - assert None is not None
1 failed, 364 passed in 13.83s

$ PYTHONPATH=. uv run python /tmp/probe15.py   (same state: day 11, 3 quadrants, 18 STRAWBERRY + 8 WHEAT)
board_budget parts: standing 82 capacity 49
crop_caps  : {'WHEAT': 23, 'CARROT': 7, 'TOMATO': 12, 'STRAWBERRY': 39, 'MELON': 2}
share_crop : {...} sum 75
fill_pass2_from: None  len(fill_crop) 75
```

## [minor] `allocation-test-suite-red`

**file** `rlv3/tests/test_allocation.py` line 818 — **demonstrated** True
**rule** rlv3/README.md 'the wiring, no games: PYTHONPATH=. uv run pytest rlv3/tests/ -q'

**Claim.** The rlv3 test suite does not pass: `test_water_fill_trace_names_the_crop_the_direction_is_wrong_on` asserts `alloc.trace['fill_pass2_from'] is not None`, but in the state it builds the first water-fill pass already consumes the whole board budget so the bare-ground second pass never runs.

**Failure.** The suite is red on HEAD, so a new regression cannot be told from the standing failure, and the assertion no longer tests what it names (it asserts the second pass RAN rather than that the trace names the crop). 364 pass, 1 fails.

**Fix.** Make the assertion conditional on there actually being bare ground left after pass 1 (`if sum(share_crop) < board_budget`), or build a state with a demand cap that binds below capacity so the second pass is forced; keep the monotonicity checks unconditional.

```
PYTHONPATH=. .venv/bin/python -m pytest rlv3/tests/ -q
 FAILED rlv3/tests/test_allocation.py::test_water_fill_trace_names_the_crop_the_direction_is_wrong_on - assert None is not None
 1 failed, 364 passed in 9.75s
Reproducing the state: share_crop {'WHEAT':23,'CARROT':7,'TOMATO':12,'STRAWBERRY':31,'MELON':2} sums to 75 == board_budget (26 standing + 49 capacity), so `sum(share_crop.values()) < board_budget` is False and the fill_bare pass is skipped; fill_pass2_from stays None.
```

## [minor] `best-sell-hour-wrong`

**file** `rlv3/market_model.py` line 54 — **demonstrated** True
**rule** README "Town Buildings": "Each unlocked shop *instance* consumes one of every product it demands every `townShopSellInterval` turns (default 4). So with the default interval, a shop demanding wheat removes 6 wheat from the market per day." Engine `_town_consume`: one unit per shop instance per tick, at steps where `step % 4 == 0` (hours 0, 4, 8, 12, 16, 20).

**Claim.** `BEST_SELL_HOUR = 1` and the module docstring's "ENGINE FACT 1" — "the compound tick (6 shops' worth + the town centre) lands at the END of the step whose hour is 0. **Hour 1 is therefore the lowest-inventory hour of the day**" — are wrong. A shop's 6 units a day are 6 separate one-unit ticks spread over hours 0/4/8/12/16/20, not 6 units at hour 0. Inventory falls monotonically through the day, so hour 1 carries the day's SECOND-HIGHEST inventory (one tick's worth of drain) and hours 21-23 the lowest.

**Failure.** No live code path reads `BEST_SELL_HOUR` today (it appears only at its definition and in `__all__`), so nothing mis-sells yet. The failure is latent: it is an exported constant naming the wrong hour, sitting in the file every other rlv3 module imports its market identities from, under a heading that says it was "verified against kaggle_environments". Anything that later times a slice off it would sell into the day's worst price rather than its best.

**Fix.** Set `BEST_SELL_HOUR = TURNS_PER_DAY - 3` (21) — the observation after the last shop tick of the day — and correct the docstring: each shop tick removes `mult` units per shop instance, six ticks a day, so within-day inventory is monotone decreasing and the late hours carry the best price.

```
$ PYTHONPATH=. uv run python /tmp/t_hour.py   # 4 shops open, nobody trading, day 5
hour :  cumulative drain of WHEAT visible at that hour's observation
   0 :   0.0   price=$25
   1 :   3.0   price=$27     <- BEST_SELL_HOUR: second-HIGHEST inventory
   5 :   5.0   price=$27
   9 :   7.0   price=$28
  13 :   9.0   price=$28
  17 :  11.0   price=$28
  21 :  13.0   price=$29     <- actual lowest-inventory / best-price window
  23 :  13.0   price=$29
```

## [minor] `blind-channel-fabricates-zero-supply`

**file** `rlv3/objective.py` line 1269 — **demonstrated** True
**rule** rlv3/objective.py:1186-1197 'attributing the gap to the opponent invents supply that is not there' / :915-918 'every multiplier starts at 1.0 ... a bad correction degrades to the shipped model rather than to nonsense'

**Claim.** `observe()` deliberately skips a product whose price is on the $1 floor ('the channel is blind, the identity has nothing to solve with'), but `_ewma` iterates over `implied` and reads `realised.get(item, 0.0)`, so the missing entry becomes a realised flow of ZERO and drives that product's conversion to the 0.25 clamp.

**Failure.** 'We cannot measure their supply' is converted into 'they will bring only a quarter of their board to market' - the opposite of the safe default the class documents ('every multiplier starts at 1.0 ... a bad correction degrades to the shipped model rather than to nonsense'). It bites hardest on exactly the glut products (STRAWBERRY/MILK/WOOL) that are 97% of the shared-inventory cost, and it persists after the town drains the glut and the product comes back off the floor.

**Fix.** In `_ewma`, skip any item that has no key in `realised` (rather than defaulting to 0.0), or have `observe()` record blinded items in an explicit `unsolved` set that `_update` excludes.

```
PYTHONPATH=. .venv/bin/python /tmp/probe3.py + /tmp/probe4.py (world session-112391485, preset base):
 12 dawns with opp_implied[WOOL] >= 1 and NO realised entry, every one of them with WOOL in `self.blind`:
 fc_day 14 WOOL implied 12.0 realised_key_present False blind True   (and 16,17,18,19,22,23,24,25,27,28,29)
 conversion at end: {'WOOL': 0.276, ...}  -- the 0.25 floor, from a 1.0 start
Dollar impact in this world is small (<$50/dawn) only because wool is itself floored; opponent_bank as-coded vs blind-neutral differs by at most +$46.
```

## [minor] `capacity-double-subtracts-reserved`

**file** `rlv3/allocation.py` line 190 — **demonstrated** True
**rule** The function's own contract — `crop_capacity` is documented as the plantable ground after the structure reservation, and `buildable` is meant to be the ground a structure could still be built on, not that ground minus the reservation a second time.

**Claim.** `capacity()` subtracts the planned structure builds from the free-tile count twice: `crop_capacity` already nets `reserved` off at line 169, and line 190 computes `buildable = max(0, crop_capacity - reserved)`.

**Failure.** `herd_capacity` is `min(feed_capacity, chore_budget, herd_total + structures + buildable)`, so whenever ground rather than labour or feed is the binding term, the herd target is silently capped below what the board can actually hold — by up to 4 animals in the states probed. The allocator's water-fill then never even scores those animals, and the trace reports the reduced number as if it were the real ceiling. In 61 of 270 (view, hires, structure-build) combinations drawn from the 18 recorded mid-season states the shipped `herd_capacity` is strictly below the single-subtraction value.

**Fix.** `buildable = crop_capacity` (line 190), since line 169 has already netted `reserved` off.

```
$ PYTHONPATH=. uv run python /tmp/probe6.py   (mmpq-d12-s0, free_now 5, freeing[1] 2)
  builds={'COOP':1,'PASTURE':2} reserved=3  crop_capacity=4  buildable(as written)=1  buildable(intended)=4
  builds={'COOP':0,'PASTURE':4} reserved=4  crop_capacity=3  buildable(as written)=0  buildable(intended)=3

$ PYTHONPATH=. uv run python /tmp/probe7b.py
mmpq-d6-s0  hires=8  builds={'COOP':2,'PASTURE':2} reserved=4 crop_cap=21 feed=42 chore=30 herd_capacity SHIPPED=25 vs no-double-subtract=29
mmpq-d6-s0  hires=11 builds={'COOP':1,'PASTURE':1} reserved=2 crop_cap=23 feed=42 chore=40 herd_capacity SHIPPED=29 vs no-double-subtract=31
cases where the double subtraction changes herd_capacity: 61
```

## [minor] `fill-bare-shop-only-switch-is-inert`

**file** `rlv3/allocation.py` line 414 — **demonstrated** True
**rule** rlv3/allocation.py:409-411 comment: "Switchable: fill_bare_shop_only = false lets bare ground be filled with anything up to the hard cap."

**Claim.** The bare-ground second pass builds its `hard` ceiling by iterating `crop_cands`, which already excludes every crop whose demand cap is 0, so `fill_bare_shop_only = false` — documented as letting "bare ground be filled with anything up to the hard cap" — has no effect in the only case it was written for.

**Failure.** The documented escape hatch for the melon ceiling does nothing. Setting `fill_bare_shop_only: false` in the params JSON changed the allocation on 0 of 18 recorded leader mid-season states, so an operator (or a training loop sweeping the params file) would read "the switch is off and melon still doesn't appear, so the rent model doesn't want melon" when in fact melon is not a candidate at all.

**Fix.** Build the second pass over a candidate list that is not pre-filtered by the demand anchor — e.g. `pass2_cands = [c for c in CROP_ORDER if rent.crop_block(c, day) is not None and not crashed(c)]` with `floors=share_crop` — so the hard ceiling really is the ceiling when `fill_bare_shop_only` is false. Add a unit test that asserts the two settings differ on a state whose melon demand cap is 0.

```
PYTHONPATH=. .venv/bin/python /tmp/switch_check.py  ->  fill_bare_shop_only changed the allocation on 0/18 recorded leader states  (share_crop identical for True and False on every case)
and PYTHONPATH=. .venv/bin/python /tmp/melon_fill.py:
  boey-d6-s0  default (fill_bare_shop_only=True) : share_crop={'WHEAT':24,'CARROT':8,'TOMATO':0,'STRAWBERRY':5}  plant_quota={'WHEAT':21}
  boey-d6-s0  fill_bare_shop_only=False          : share_crop={'WHEAT':24,'CARROT':8,'TOMATO':0,'STRAWBERRY':5}  plant_quota={'WHEAT':21}
```

## [minor] `inert-lever-warning-unreachable`

**file** `rlv3/scorecard_hook.py` line 351 — **demonstrated** True
**rule** rlv3/executor.py:35-40 'check a non-zero counter in stats() from inside a worker before reading any arm as a result'; rlv3/smoke.py:59-63 'a zero-counter arm with a flag on is a lever that did not fire'

**Claim.** The 'exec seam fired NOTHING -- the levers are inert' warning tests `spec.get('exec_flags')`, but a preset spec never carries that key (the flags are resolved from the preset by `config_from_spec`), so the branch is unreachable for every spec the hook is documented to take; smoke.py's equivalent check only fails when the ENTIRE stats dict is empty, so a half-dead flag set passes.

**Failure.** An arm in which one lever fires and the others are silently inert reports 'all checks passed'. That is what happens today with `rlv3:full`: `pl_wheat_adj` produces 172 moves on day 0 and `co_chain_tile` produces zero reorders for the whole season, and both the smoke test and the scorecard hook report the seam as live.

**Fix.** Use `AGENT.config_from_spec(spec).exec_flags` in both places and check EACH enabled flag has a non-zero counter whose key starts with that flag's name, not just that the stats dict is non-empty.

```
PYTHONPATH=. .venv/bin/python:
 spec from parse_agent('rlv3:full') = {'kind':'rlv3','preset':'full','exec':['lastwater','h23','harvest_visit','pl_wheat_adj','co_chain_tile']}
 spec.get('exec_flags') = None  -> the 'exec seam fired NOTHING' branch is UNREACHABLE
 resolved exec flags actually used = ('pl_wheat_adj','co_chain_tile')
And `PYTHONPATH=. .venv/bin/python rlv3/smoke.py --presets base,full --games 1` prints
 exec ('pl_wheat_adj','co_chain_tile') -> {'pl_wheat_adj.displaced':143,'pl_wheat_adj.moved':29}
 all checks passed          (no co_chain_tile counter anywhere)
```

## [minor] `mv-dest-counters-double-count`

**file** `rlv3/executor.py` line 271 — **demonstrated** True
**rule** rlv3/placement.py _move_key: 'A miscounted lever is how an arm becomes unreadable'; rlv3/ledger.py: counters must mean the same thing as the trace metric.

**Claim.** rlv3_assign mirrors ExecNetPolicy's CUMULATIVE counters as deltas against `_MV_SEEN`, but `reset_stats()` clears `_MV_SEEN` while the ExecNetPolicy object (cached in placenet._BY_MODEL) keeps its Counter. The first mv_dest step of every episode after the first therefore re-adds the whole previous history as this episode's delta.

**Failure.** mv_dest's own telemetry is inflated by the sum of every earlier episode in the process. rlv3/smoke.py reads exec_stats as its liveness check and scorecard_hook._trace_agent SUMS EXEC.stats() across games in one process, so an N-game scorecard run reports mv_dest counts growing roughly as N^2/2 -- the 'miscounted lever' this package's own placement.py warns makes an arm unreadable.

**Fix.** Reset the mirrored source too. In executor.reset_stats(), also clear the cached policies' counters -- e.g. `from rlv3 import placenet; [p.stats.clear() for p in placenet._BY_MODEL.values()]` -- or key `_MV_SEEN` off the policy object itself and seed it from `dict(pol_stats)` the first time an episode touches it, instead of defaulting the baseline to 0.

```
/tmp/rlv3rev/p9.py, `PYTHONPATH=. uv run python /tmp/rlv3rev/p9.py` -- the SAME three mid-season states replayed twice with EX.reset_stats() between:
  episode 1 (3 steps):                        {'mv_dest.override': 1, 'mv_dest.stay_no_work_kept_scheduler': 2, 'mv_dest.agree': 10, 'mv_dest.kept_low_conf': 2}
  episode 2 (SAME 3 steps, after reset_stats): {'mv_dest.override': 2, 'mv_dest.stay_no_work_kept_scheduler': 4, 'mv_dest.agree': 20, 'mv_dest.kept_low_conf': 4}
Exactly 2x, i.e. episode 1's cumulative totals re-counted.
```

## [minor] `probe-features-stamp-flip`

**file** `rlv3/fingerprint.py` line 151 — **demonstrated** True
**rule** rlv3/README.md '1. The stamp': 'The fingerprint is the action space (heads_version, heads_mode, sha1 of HEAD_SIZES, n_action_logits, n_features, ...)'; kaggriculture-gate-cache-stale.

**Claim.** The action-space stamp records `n_features: null` on any slice run without --probe-features (the default) and an integer on any slice run with it, and fingerprint.check treats null != 168 as a stale stamp. Turning the diagnostic flag on part-way through a run therefore invalidates every stamped artifact of that run.

**Failure.** stages.resolve quarantines the stale artifacts into <run>/stale/<ts>/ and, for the PPO stage, quarantines the whole run directory and restarts training from iteration 0 -- hours of compute discarded by adding an opt-in diagnostic flag whose docstring advertises it as costing only one BC episode. (The reverse direction is safe: probe_n_features reads <run>/n_features.json regardless of `play`, so once a probe has run the value sticks.) With the flag never used, the key is always null and the n_features half of the action-space guard the README advertises never fires at all.

**Fix.** Make the key slice-invariant. Either drop n_features from ACTION_KEYS (heads_version + head_sizes_sha + n_action_logits + the `code` digest already cover a width change), or normalise it: treat an unknown width as 'not covered' rather than as a distinct value -- in fingerprint.check, skip a key whose stamped value or whose current value is None, and record that the key was unchecked.

```
`PYTHONPATH=. uv run python -c "..."` against the real fingerprint/stages code:
  slice 1 stamped n_features = 168
  slice 2, SAME command as the README shows (no --probe-features):
     FP.check -> (False, 'n_features changed: stamped 168 != now null')
  slice 2 WITH --probe-features   -> (True, 'fresh')
And in a fresh driver process (`FP.action_space(FP.probe_n_features())`): n_features None, features.N_FEATURES None -- the default value for every slice.
stages.resolve:604-612 quarantines on a stale skip_if hit; :637-643 quarantines the PPO run dir on a stale run stamp.
```

## [minor] `promote-baseline-substitution`

**file** `rlv3/promote.py` line 199 — **demonstrated** False

**Claim.** When the gate's own baseline arm is missing from `session_eval.json`, `promote()` silently falls back to the frozen `rl/runs/draw995/batch2.json` rows for `ship2_hv` and records them under the requested arm name — `_assert_same_draw` checks the world list but never that it is the same AGENT.

**Failure.** `rlv3/jobs/rlv3_train.json` sets `baseline_arm: "champ_h10"`, which the gate plays as `bundle:arena/agents/champ-h10/main.py`. If that arm crashes, is renamed, or the eval is re-run without it, the promotion decision is made against `ship2_hv` instead — and `rlv3/agent.py:36-40` states these are different agents ("NOT arena/agents/champ-h10/main.py, which is a snapshot of an OLDER kagent/ and finishes $11,638 away from today's tree on world 770000"). `man["gate"]["baseline"]["arm"]` still reads `champ_h10`; only the `from` path discloses the swap. A candidate could be promoted or refused against a baseline nobody chose.

**Fix.** Make the fallback explicit rather than silent: rename the recorded arm to what was actually loaded (`man["gate"]["baseline"]["arm"] = BASELINE_ARM if base_src == BASELINE_ROWS else baseline_arm`) and add `"substituted": True` plus a printed warning; or require the caller to pass `baseline_json` before any substitution is allowed, and raise otherwise.

```
Static: promote.py:197-205 catches KeyError on `load_rows(eval_json, baseline_arm)` and rebinds `base = load_rows(BASELINE_ROWS, BASELINE_ARM)` where `BASELINE_ARM = "ship2_hv"`; `_assert_same_draw` (promote.py:253-266) compares only `len`, `path` and `script_side`. The verdict dict at promote.py:215-220 writes `"arm": baseline_arm` (the requested name) beside the substituted rows. What would settle it: build a `session_eval.json` carrying a `cand` arm on the 995 draw but no `champ_h10` arm, call `promote(..., baseline_arm="champ_h10")`, and read `man["gate"]["baseline"]` — it will name champ_h10 with ship2_hv's objective row.
```

## [minor] `rlv3-test-suite-red`

**file** `rlv3/tests/test_allocation.py` line 818 — **demonstrated** True
**rule** rlv3/README.md "Running it": `PYTHONPATH=. uv run pytest rlv3/tests/ -q` is the package's own wiring check, run before anything else.

**Claim.** `test_water_fill_trace_names_the_crop_the_direction_is_wrong_on` fails on `assert alloc.trace["fill_pass2_from"] is not None`. On the state it builds (day 11, 3 quadrants, 18 strawberry + 8 wheat standing) the first water-fill pass already consumes the whole `board_budget`, so `allocation.allocate:353` never enters the bare-ground second pass and `fill_pass2_from` stays None.

**Failure.** The package's stated pre-flight command exits non-zero (1 failed, 364 passed), so a red suite is the normal state and a genuinely new regression in the allocation trace would be indistinguishable from this standing failure. The assertion itself encodes a false invariant: pass 2 runs only when the demand anchor binds BEFORE the ground runs out, which is state-dependent.

**Fix.** Either build a state where the anchor actually binds below the board budget (raise the opponent's projected supply so `crop_caps` bite first), or relax the assertion to `assert alloc.trace["fill_pass2_from"] is None or 0 <= alloc.trace["fill_pass2_from"] <= len(fill)` and keep the monotone-marginal check, which is the part that actually tests the water-fill.

```
$ PYTHONPATH=. uv run pytest rlv3/tests/ -q
>       assert alloc.trace["fill_pass2_from"] is not None
E       assert None is not None
rlv3/tests/test_allocation.py:818: AssertionError
FAILED rlv3/tests/test_allocation.py::test_water_fill_trace_names_the_crop_the_direction_is_wrong_on
1 failed, 364 passed in 8.66s
```

