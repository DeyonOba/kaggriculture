# rlv3 adversarial review — from the build workflow (wf_fa35716a-df0), 2026-09-26

Produced by the build workflow's own Verify phase, AFTER the 40 files were written.
Independent of the ground-truth loop (wf_90148fc7-b62), which started later.
Findings here are the reviewer's claims; only those re-verified adversarially should be fixed.

Read every file under `rlv3/` and verified each claim by running read-only probes against real mid-season boards from `replays/{mmpq,boey,dsm}` (18 views, both seats, days 6/12/20). No games played, nothing modified, nothing staged.

# ADVERSARIAL REVIEW — rlv3/

## 1. CRITICAL — `slots_only` (the DEFAULT preset) silently destroys the day's BUY orders, permanently

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/selling.py:521-535`**

```python
hires = [o for o in base if o and o[0] == "HIRE"]
rest  = [o for o in base if o and o[0] not in ("HIRE", "SELL")]
...
orders = ([["SELL", s.item, s.units] for s in lead] + rest + [...tail])
orders = hires + orders
orders = orders[:MAX_MARKET_ORDERS]        # == 10
```

`kagent/market.py:24-111` never drops a buy: `add()` refuses at the cap and SELLs are appended **last**, so the cap eats sells. rlv3 inverts that — up to 3 SELLs are promoted ahead of `rest`, and `HIRE` can legally take 8 slots (`market.py:47 while need > 0 and hired < 8`). 8 + 3 = 11 > 10, so `rest` is truncated away entirely.

Worse: `market.build_orders` **mutates the plan as if the order had been placed** — `market.py:81 plan.land_buys = remaining_land` and `market.py:89 plan.animal_buys[kind] = want - n`. `selling.build_orders:509` calls it once, lets it decrement the plan, then throws the order away. The purchase is not deferred, it is lost for the day.

Verified (`mmpq` replay, day 12, hour 0, seat 0):

```
BEFORE  animal_buys: {'COW': 4}
ORDERS SUBMITTED   : 8x HIRE + SELL FERTILIZER 18 + SELL WHEAT 23
AFTER   animal_buys: {'COW': 0}      <- decremented, never submitted
```

Across 18 dawn boards (hours 0 and 1, 3 replays x days 6/12/20 x 2 seats) this fired on **9**. Casualties include `BUY_ANIMAL COW 4` ($1,600 of planned investment), `BUY_SEED STRAWBERRY 22`, `BUY_ANIMAL SHEEP 2`, and `BUY_PRODUCT WHEAT 15` — the feed buy, on a farm where an unfed animal escapes after two dry days.

This is in `RECOMMENDED` (`selling.py:623 SellConfig(slots_only=True)`), which `agent.py:112/116/117` puts in the `recommended`, `full` **and** `select` presets. The component's own pre-gate table calls the slot move "free — no supply withdrawn, no allocation changed"; it changes the allocation by deleting it. The +$1,281 measured for `slots_only` was measured on the same mechanism and is not attributable.

**Smallest fix:** rebuild with the shipped precedence and let the cap bite sells, e.g. `orders = hires + lead[:max(0, MAX_MARKET_ORDERS - len(hires) - len(rest))] + rest + tail` before truncation — or simply never let `lead` push `rest` past the cap. And do not call the shipped builder for its side effects and then discard its output: capture `plan.animal_buys`/`plan.land_buys` before the call and restore them for any order that does not survive truncation.

---

## 2. CRITICAL — the training job trains and promotes the refuted RL net; no rlv3 component is in the loop

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/jobs/rlv3_train.json`** (stages `bc`, `ppo`, `gate`, `promote`)

The graph is `rl/session_bc.py` → `rl/train.py --iters 0` → `rl/train.py --resume bc.pt --reward win` → `rl/session_eval.py --arm cand={out}/ppo/eval.json` → `rlv3.promote`. `cand` is a **weights** arm — `rl.common.to_layers` / `nnpolicy`. Verified:

* `grep -rn "rlv3" rl/ tools/ --include=*.py` → **zero hits**. Nothing in the trainer or the eval instrument knows rlv3 exists, and `rlv3/arm.py:8-18` says the two integration lines are "deliberately NOT made".
* `objective.episode_rewards` / `terminal_reward` / `shaping_terms` have **no caller anywhere outside `rlv3/tests/test_objective.py`**. `objective.py:250` claims "rlv3's own trainer calls it". There is no rlv3 trainer.
* Nothing in the graph ever writes `rlv3/params/allocation_v0.json` (the land theta, documented at `land.py:27-31` as "the JSON IS the checkpoint"), moves `lam`, or selects a sell/exec configuration.
* The `maia2` stage trains `{out}/execnet2.pt`, but `placenet.py:48 DEFAULT_PT = "rl/datasets/execnet2_cond/execnet2_cond.pt"`. The freshly trained model is never loaded.

The repo record puts the best RL net at 0.127–0.180 against champ-h10's 0.354 (`kaggriculture-rl-vs-champion-deploy-check`). `promote()` will return `None` every slice, forever, and the day of compute buys nothing for the agent that was actually built.

**Smallest fix:** the objective is not what needs wiring first — the *arms* are. Replace the `gate` stage's `cand` with `dev:rlv3:<preset>` (which requires the two one-line edits in `rl/rollout.py` and `rl/session_eval.py` that `arm.py:8-18` already specifies), and make the tunable stage a dose/theta sweep over `rlv3/params/*.json` rather than PPO over the RL head. If PPO is to stay, it must at minimum consume `objective.episode_rewards`.

---

## 3. CRITICAL — `p_win` is fitted on the MONEY LEAD and queried with the PROJECTED FINAL MARGIN

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/objective_check.py:613-614`**

```python
m = (float(v0.me.money) - float(v0.opp.money))
samples[day].append((m if seat == 0 else -m, win))
```

The table's feature is the **money lead**. `objective.Outlook.__post_init__:551` calls `p_win(self.margin, ...)` where `margin = our_bank - opp_bank` from `project()` — money **plus the whole board's projected remaining revenue**. Different quantity, different scale, by an order of magnitude early.

The clamps in the shipped `rlv3/calibration.json` prove it:

```
day  0  clamp=[0.0, -0.0]        n=[0,0,0,0,0,0,524,0,0,0,0,0,0]
day  4  clamp=[-1177, 1177]      day  8  clamp=[-884, 884]
day 12  clamp=[-15089, 15089]
```

Measured projected margins on the same replay boards at day 6 are ±$12,046, ±$7,854, ±$6,417, ±$859. Every one saturates the day-4/8 clamp. Consequence, measured:

```
day  -20000  -12000   -5000       0   +5000  +12000  +20000
  0   0.500   0.500   0.500   0.500   0.500   0.500   0.500
  4   0.439   0.439   0.439   0.498   0.498   0.498   0.498
  8   0.498   0.498   0.498   0.498   0.498   0.498   0.498
 12   0.179   0.291   0.461   0.500   0.539   0.709   0.821
```

* Day 0: `p_win` is a **constant 0.500**. Clause 1 of `better()` cannot fire at all.
* Days 4–8: exactly **two** reachable values. Clause 1 degenerates to a sign test on the projected margin; inside each bucket it falls through to the margin clauses, i.e. to "maximise our projected bank minus theirs" — the weighted-sum behaviour the module exists to forbid.
* It is not even monotone in day: at a fixed −$20,000, `p_win` runs 0.500 → 0.439 → 0.469 → 0.498 → 0.338 → 0.179 over days 0/4/6/8/10/12. The test suite asserts monotone-in-margin, not monotone-in-day.
* Day 12 saturates at 0.821 for every margin ≥ ~$18k, and the +$20k knot is `p=1.0` on **n=13** with the +$40k knot at `n=0`. `objective_check.py:638 last = 0.0` back-fills empty leading bins with 0.0 (certain loss), then PAVA locks it in — so a day-12 margin of −$15,089 reads ≈0.18 from 13 observations.

**Smallest fix:** build the table on the same statistic the comparator queries — replace line 613 with `O.project(v0, ...).margin` (both seats, as now). Failing that, refuse to return a `p_win` whose row has `n == 0` in both bracketing knots, and make the clamp saturation visible (return `None`) rather than silently collapsing clause 1.

---

## 4. HIGH — `co_animal_first` is a provable no-op and `co_chain_tile` fires on 2.9% of tiles; the tests pass because they feed the module task lists the scheduler never produces

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/choreorder.py:206-220`**

`kagent/scheduler.py:398-421` appends animal-tile tasks in a single loop in the order FEED → CARE → COLLECT_FERTILIZER → HARVEST — which **is** `CHAIN_ANIMAL` (`choreorder.py:36`). So `chain_order()` returns the identity for every animal tile, `order == band` at line 214, and the band is left untouched. FEED and CARE are mutually exclusive (`not fed_today` vs `fed_today`), so no permutation exists.

Measured over the 18 real mid-season boards with the shipped exec flags:

```
animal:tiles_ge2            189
animal:chain_differs          0      <- co_animal_first can never fire
animal:work_WOULD_REORDER     0
crop:tiles_ge2              153
crop:chain_differs          115
crop:work_band_ge2           50
crop:work_WOULD_REORDER      10      <- co_chain_tile fires on 10/342 tiles
crop:fill_band_ge2            0
```

The 115-vs-10 gap is the band split: `WATER` usually lands at `P_FILLER` (`scheduler.py:374/379`) while its `HARVEST` is at `P_HARVEST`, and invariant (c) forbids crossing the `P_FILLER` edge — so the WATER-before-HARVEST chain the module exists to enforce is unreachable in exactly the case that matters.

The docstring targets the animal half explicitly ("much the larger one: 1.617 vs 2.773"). That half is 0% expressible. Every animal test — `test_choreorder.py:63-65, 81-84, 109-111` — hand-builds `[HARVEST, COLLECT_FERTILIZER, FEED]`, the reverse of what `build_tasks` emits. The trace in the integration report (`co_chain_tile` emitted zero reorders over a season) is the predicted result, not an anomaly.

**Smallest fix:** this is an action-space problem, not a bug to patch. Either allow the reorder to move a task across the `P_FILLER` edge inside one tile (relaxing invariant (c) to "the multiset of *bands on this tile* is permuted"), or delete `co_animal_first` and the animal half of `co_chain_tile`. At minimum, replace the synthetic fixtures with `scheduler.build_tasks(view, plan)` output so a no-op lever fails its own test.

---

## 5. HIGH — `risk_posture` compares two different valuations; it is pinned at −1.000 on 14 of 18 real boards

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/selling.py:335-372`**

```python
mine   = rollout.portfolio_value(view)                 # heavily discounted
theirs = float(view.opp.money) + _board_value(view)    # full price, no discount
```

`rl/rollout.py:94-119 portfolio_value` caps shed at 40 units/item, animals at 8 days, and prices standing crops at `0.6 * price * yield_units` (+`0.3 *` remaining). `selling._board_value:361` sums **every** future arrival over the whole remaining season at today's full market price, no curve walk, no discount. The docstring claims "both sides use the same board valuation". They do not.

Measured:

```
mmpq d6 s0 −1.000   mmpq d6 s1 −1.000     <- both seats of the same world
mmpq d12 s0 −1.000  mmpq d12 s1 −1.000
boey d20 s0 −0.374  dsm d20 s0 +0.211
```

14 of 18 saturate at the clip; only one is ever positive. A "lead" that says both players are losing is not a comparator. Consequences: the `posture > 0` branch (`selling.py:458`, "winning: take certain cash") is dead code, and the `posture < 0` branch (`selling.py:467-472`, floor × `behind_hold` 1.15, withhold one extra tick) is permanently ON — the holding behaviour the module's own docstring says the record refutes ("paced selling above a floor let the opponent sell first and lost $30k").

**Smallest fix:** price both sides with the same function. `objective.project()` already does exactly this (and `objective.INCLUDE_OUR_SHED = False` at `objective.py:139` was flipped to False for this precise reason, measured). Replace the body with `o = OBJ.project(view, memory); return clip(o.margin / LEAD_SCALE, -1, 1)`.

---

## 6. HIGH — the sell schedule books an offer as a fill, so its self-correcting loop can never learn its ask was impossible

**`/Users/gideonoba/WorkSpace/OpenSource/kaggriculture/rlv3/selling.py:481-484`**

```python
for s in out:
    st.last_slice_step[s.item] = view.step
    st.sold_today[s.item] += s.units
    st.carry[s.item] = max(0.0, st.carry[s.item] - s.units)
```

Every slice is recorded as sold the moment it is *planned*. Three ways it is not: the engine fills only as far as the shed allows (`kaggriculture-ahmed-gap`: "SELL order[2] is a request, not a fill"); another player's slot may have moved the price below the floor; and — per finding 1 — `build_orders` may truncate the order off the list entirely. `dawn_update:200-204` then computes `carry = max(0, tgt - sold_today)` from that fiction, so a slice that was never submitted still clears its own carry.

Independently, `dawn_update:184-196` attributes the whole dawn inventory residual to the opponent (`ratio = 1 + err/opponent_supply`). The residual also contains our own unfilled offers and the shop draw; the docstring at line 174-178 says "the only unknown in it is the opponent's realised supply plus the actual shop draw", which is false in exactly the direction that matters.

**Smallest fix:** book `sold_today`/`carry` from the realised holdings delta on the *next* step, the way `objective.ProjState.seat_net` already solves it (`objective.py:1057-1129`) — that function exists, is unit-tested, and is not used here.

---

## 7. HIGH — `demand_caps` reads the watered-only strawberry rate where the shipped anchor reads the fertilised one; the cap is 2.13x too loose on the crop we already overplant

**`rlv3/allocation.py:326`** → **`rlv3/rent.py:138-155`**

`rent.per_tile_day("STRAWBERRY")` reads `features._PER_TILE_DAY["STRAWBERRY"] = 0.235` (watered only, set by commit `dfb06e5`). `features.anchors` (`kagent/features.py:169`) divides by `_STRAW_PER_TILE_DAY = 0.5` — the fertilised+watered rate — for the same head. A smaller divisor means *more* tiles:

```
ratio rlv3 cap / shipped anchor = 2.128
mmpq-d6   rlv3 STRAWBERRY cap 40 (the hard ceiling)  vs shipped anchor 33
mmpq-d12  rlv3 26                                     vs shipped 16
```

The component's stated purpose is to cut 41 standing strawberry to the leaders' 13–24. Its demand cap saturates the 40-tile ceiling on day 6. `rent.py:9-14`'s own docstring is also stale against the new tables ("carrot turns 1.0 units/tile-day where strawberry turns 0.5" — now 0.750 and 0.235).

Same function, two more holes:

* **`allocation.py:330-340`** has no equivalent of the shipped `prod_days` guard (`features.py:158-168`). At day 20 `features.anchors` gives `sheep_target = 0` (`28-20-6 = 2 < 6`); rlv3 gives **14**. Verified: `dsm-d20-s1` at lam=1 emits `buys={'SHEEP': 1}` plus a PASTURE build where the champion buys nothing. A $500 sheep bought on day 20 yields 2 productions (~4 units of wool into a glutted market) and eats 1 WHEAT/day plus feed+care chores for 9 days.
* **`allocation.py:326` strips the feed coupling from WHEAT.** `features.anchors["wheat_tiles"] = min(44, herd_total + 20)` is a *feed* expression; rlv3 computes wheat from demand-minus-opponent-supply. Measured at day 12: rlv3 cap 14 vs shipped 37; day 20: 6 vs 36. `capacity()` bounds the *herd* by `feed_capacity` (`allocation.py:181`), but nothing bounds the *wheat quota* by the herd it just bought.

**Smallest fix:** use `features._STRAW_PER_TILE_DAY` for strawberry in `rent.per_tile_day` (one `if`); copy the three `prod_days` guards from `features.anchors:155-181` into `demand_caps`; floor `crops["WHEAT"]` at `herd_total + 20 - standing_wheat`.

---

## 8. HIGH — structure builds and the plant quota contend for the same tiles; capacity reserves the wrong plan's structures

**`rlv3/allocation.py:169, 190, 472-475`**

```python
reserved = sum((base.structure_builds if base else {}).values() or [0])   # BASE's
crop_capacity = max(0, flow["free_now"] + flow["freeing"][1] - reserved)
...
buildable = max(0, crop_capacity - reserved)                              # again
...
alloc.plant_quota      = _plant_flow(..., cap["crop_capacity"], ...)      # uses it
alloc.structure_builds = structure_builds(view, alloc.animal_buys, pocket) # computed AFTER
```

`reserved` is subtracted twice (lines 169 and 190), and it is the **base** plan's structures — but `alloc.structure_builds` is derived from rlv3's own `animal_buys` and is computed *after* the quota has already spent the capacity. `kagent/scheduler.py:427-436` draws structures first from `empties` with a cursor and gives PLANT the remainder, so the overflow lands silently on the plant side.

Measured at `mmpq-d6-s0`, lam=1: `crop_capacity = 21`, `plant_quota = {WHEAT 14, STRAWBERRY 7}` (= 21), `structure_builds = {COOP 1, PASTURE 5}` (= 6). 27 asks for 21 tiles; 6 plantings are dropped with no counter and no trace entry.

**Smallest fix:** compute `alloc.structure_builds` before `_plant_flow` and pass `cap["crop_capacity"] - sum(builds)` as the planting budget; drop the second subtraction at line 190.

---

## 9. MEDIUM — `fill_bare` closes `idle_share` by planting everything, with no labour term — the +25-tiles refutation in a new wrapper

**`rlv3/allocation.py:399-420`**

The second water-fill pass runs "to the HARD ceiling while the marginal is still positive". The marginal is revenue per tile-day and is positive for almost any crop on almost any day; nothing in either pass prices labour. `_shadow_price` (`allocation.py:525`) is computed *after* both passes and is only used for `animal_rent`.

Measured at day 20, lam=1 (pass 2 active on all six day-20 boards):

```
mmpq-d20-s0  crop_targets WHEAT 37 (standing 20)   champion quota {}
boey-d20-s0  crop_targets WHEAT 43 (standing 23)   champion quota {CARROT 8}
dsm-d20-s0   crop_targets WHEAT 44 (standing 28)   champion quota {WHEAT 4}
```

`kaggriculture-labour-gates-land` is explicit: hands/work is 0.47–0.58 at day 11–15 and +25 tiles "buys idle ground" for −$6,266/−$6,017 on two independent blocks. The 12.4% idle share is a *labour* symptom; filling it with seeds spends seed money and chore turns on ground the crew cannot tend.

**Smallest fix:** charge pass 2 the labour shadow price — run pass 1, take `_shadow_price(trace_c)`, and require `marginal - turns_per_tile_day * shadow > 0` in pass 2. Or ship `fill_bare: false` as the default until the ladder says otherwise.

---

## 10. MEDIUM — `install_seams`' restore is never called on the documented integration path; and `uninstall` is not refcounted

**`rlv3/agent.py:365-384`, `rlv3/arm.py:152-161`**

`rl/rollout.run_episode`'s `finally` (`rl/rollout.py:546-554`) restores only `features.MODE` and `scheduler.EXEC`. Nothing calls `agent_fn.restore`. `rlv3/arm.py` compensates in its own `finally` — but `arm.py:12` documents the graduation path as a one-liner *inside* `_make_dev_policy_agent`, which reinstates the leak: a worker that plays an rlv3 arm then a baseline runs the baseline through rlv3's market patch and scheduler seam. That is the contamination the integration report says it fixed.

Second, narrower: `executor.install()` (`executor.py:121-132`) returns early if `_SCHED is not None`, but `agent.py:375` appends `EXEC.uninstall` to `undo` unconditionally. Under `KAGG_RLV3=1`, `sitecustomize._install()` has already installed the seam process-wide; the first rlv3 arm's `restore()` then **uninstalls it**, and every later arm in that worker silently runs the shipped executor while reporting as rlv3 — the exact "dead, not null" shape the package is built to prevent.

Third: `make_dev_agent:403-405` calls `install_seams(cfg)` *before* `Rlv3PlanProvider(cfg)`. If the provider raises (missing rules file), the process is patched and `_LAST["agent"]` still points at the previous agent, so `arm.run_episode`'s finally restores the wrong thing.

**Smallest fix:** refcount `executor.install/uninstall` (or have `install` return its own restore callable, `None` when it was already installed, and append only that); move `install_seams` after the provider is constructed; and make the graduation one-liner a `try/finally` in `rollout.run_episode` rather than a bare `return`.

---

## 11. MEDIUM — the ledger's kill switch for `pl_wheat_adj` reads a metric that only ratchets up

**`rlv3/ledger.py:111-141`**

```python
if name == "PICKUP" and op[1] == "WHEAT":  self._hand_wheat[idx] = False
elif name == "HARVEST" and tile crop == "WHEAT": self._hand_wheat[idx] = True
if name == "FEED" and self._hand_wheat.get(idx): self.feeds_from_hand += 1
```

`_hand_wheat[idx]` is never cleared when the FEED consumes the wheat. One wheat harvest makes **every** subsequent FEED by that unit for the rest of the day count as "from hand". `feeds_from_hand_share` is the sole kill switch for `pl_wheat_adj` (`ledger.py:209-216`, two days below 0.15 → off), so the lever's self-report is biased upward and the switch will rarely fire — a lever that costs turns and pays nothing keeps running to day 29.

Related, same file: `regime()` (`ledger.py:260-275`) selects between the `contest` and `even` branches of `value_weight`, which carry **opposite signs** on the opponent's standing count (`choreorder.py:149` vs `:155`), keyed on the raw money lead. `docs/reward_design.md` Rule 2 puts the money lead's sign accuracy at 50.7% through day 12. And `regime == "ahead"` returns `0.0` for every task, making `co_value` silently inert whenever we are nominally ahead.

**Smallest fix:** decrement/clear `_hand_wheat[idx]` on FEED. For the regime, key it on `objective.project().margin` (or on `ledger.lead` only after day 12) and make `ahead` a small positive tilt rather than a constant zero.

---

## 12. MEDIUM — `alloc_ladder.sh` never runs lam=0 as an arm, pairs bundles against a dev-tree baseline, and ships rules that differ from the file

**`rlv3/eval/alloc_ladder.sh`**

* The ladder runs `lam33/lam67/lam100` as `bundle:` arms and takes the baseline from `--reuse ship2_hv="$OUT/batch2.json:ship2_hv"` — a **dev-tree** arm. `lam000/main.py` is built and identity-checked on 2 games, then never entered in the ladder. Any bundle-vs-dev-tree difference (a knob condition that fires on some worlds, the empty `kagent/weights.py`, the head set) is attributed to `lam`. The script's own header warns that the frozen bundle and the dev tree finish $11,638 apart.
* The generated `_RULES` hardcodes `"max_knobs": 5`; `tools/rules/rules_h8_ship2_hv.json` has `"max_knobs": 4`. With exactly 4 conditions it does not bind today, but it is an undeclared divergence in the object that is supposed to make lam=0 bit-identical.
* The closing scorecard passes `--reference bundle:arena/agents/champ-h10/main.py` with `--reference-eval $OUT/batch2.json:ship2_hv` — a reference *agent* and a reference *eval* that are, by the script's own comment, two different agents.

**Smallest fix:** add `--arm lam0=bundle:$BUILD/lam000/main.py --baseline lam0` and drop the `--reuse`; load `_RULES` from `tools/rules/rules_h8_ship2_hv.json` instead of retyping it; make the scorecard reference `bundle:$BUILD/lam000/main.py`.

---

## 13. MEDIUM — the fingerprint does not cover the rlv3 checkpoints, and `n_features` can spuriously quarantine the whole run

**`rlv3/fingerprint.py:189-205, 210-227`**

`code_digest` walks `*.py` only. `rlv3/calibration.json` (the one fitted object in the objective), `rlv3/params/allocation_v0.json` (documented at `land.py:27-31` as "the JSON IS the checkpoint"), and `rlv3/arms/*.json` are invisible to every freshness check. Editing the land theta — the thing training is meant to move — invalidates nothing and every stale artifact is reused. `objective.fingerprint()` exists and is never folded in, despite `objective.py:325-332` claiming "the trainer compares this string and refuses the resume on a mismatch".

Conversely, `n_features` **is** in `ACTION_KEYS` and `check()` compares it. `probe_n_features` returns `None` unless `--probe-features` or `features.N_FEATURES` happens to be set in that process. `rlv3/README.md` says it "is simply not compared" — it is. The first slice that resolves it to an int marks every previously stamped artifact stale and quarantines the BC corpus and the PPO run, restarting from iteration 0. On a driver whose entire purpose is compute-limited resumption, that is the failure mode it was built to prevent.

**Smallest fix:** add the rlv3 JSON checkpoints to `CODE_ROOTS`' digest (hash `rlv3/**/*.json` alongside `*.py`), and fold `objective.fingerprint()` into `fingerprint()`. Either drop `n_features` from `ACTION_KEYS` or skip the comparison when either side is `None` (`fingerprint.py:224`).

---

## 14. LOW — `better()` never consults the objective for allocation, and `rank()` collapses when `p_win` saturates

`allocation.allocate()` never imports `rlv3.objective`. Its objective is `rent.block_value` = `our_revenue + w_opp * opponent_delta` (`rent.py:268-277`) — a **linear scalarisation** with `w_opp ∈ {0.25, 0.5, 1.0}`. `objective.py:14` says "Lexicographic, not a weighted sum"; the only component that acts on the allocation is a weighted sum. `objective.project()` is called at `agent.py:290` purely to record an Outlook; only the `select` preset (off by default) ever calls `better()`, and finding 3 shows clause 1 is dead before day ~10 anyway.

Also in that family:
* `rent.opponent_delta:241-258` multiplies **all** of the opponent's remaining units by a single per-unit price drop — it does not walk their curve, and it treats the post-block inventory as permanent, overstating the denial term at large `n`.
* `selling.relative_marginal:303-315` subtracts a per-unit **price spread** from a total **revenue**. Dimensionally wrong; the docstring acknowledges it ("this is the spec's quantity as written") and the function is not the decision, so the cost is only that a function named "THE win-rate term" is not one.
* `objective.p_win` bands 0–4 are a monotone logit shift of the pooled row (`objective_check.py:661-667`), so the `band` argument changes the *level* but never the *order* within a decision. "TABULATED PER BAND, NEVER POOLED" buys `better()` nothing.

## 15. LOW — `promote()` skips the head guard when `eval.json` is absent; `p_max` is looser than the standing protocol

`promote.py:209-210` runs `verify_weights_head` only `if fp is not None and os.path.exists(weights)`. A missing `ppo/eval.json` silently skips the action-space check and promotes on the gate rows alone. And `p_max=0.10` (`promote.py:183`) is looser than the repo's own bar — commit `dd6e3af` calls p=0.068 "Promising, not proven: misses p<0.05".

## 16. LOW — documentation that contradicts the code

* `market_model.drain_between:110-112` says `(step_a, step_b]`; `ticks_between:67-70` (correct, post-fix) says `[step_a, step_b)`.
* `objective.py:250` — "rlv3's own trainer calls it". No such trainer.
* `selling.py:23-24` — `min_slice_gap = 4` is "the whole mechanical difference" from `sell_pace`. The component's own measurement found champ-h10's shipped median gap is already 10.5 steps.
* `placement.py:12-14` targets "wheat tiles beside an animal", but `score_tile:170` scores `near_structure`, which includes **empty** COOPs and PASTUREs.
* `placenet.placement_prior:100-104` conditions on `view.me.units[0]` and `F.unit_features(view, 0, ...)` for every candidate tile, while the model was trained per-unit. Moot while `pl_net` is off (dispersion cv 0.018), but it would not be if the checkpoint were retrained.

---

### What I could not check
The 995-game paired gate, and therefore every "measured" number in the component reports (the `+$1,281` slot move, the `+$891` full schedule, the 39/39 retrodiction). Finding 1 means the `slots_only` figure was produced by a mechanism that deletes buy orders, so it should be re-measured before it is cited again.

---

# 17. CRITICAL — `dev:rlv3:base` is not an identity control: it matches NEITHER baseline

Found and verified 2026-09-26 21:40 while wiring rlv3 into `rl/session_eval.py`.
Reported by neither the build review nor the ground-truth loop's first round.

`rlv3/agent.py` documents `base` as "all doses zero", and the build smoke reported
"IDENTITY EXACT". It is not an identity against either candidate baseline:

| comparison | result |
|---|---|
| `dev:rlv3:base` vs `dev:champion`, 12 paired games | differs on 3, up to **$32,952** |
| `dev:rlv3:base` vs `dev:knobs:tools/rules/rules_h8_ship2_hv.json` | −1 win, −$4,549 |

Reproduces in a fresh process per game, so it is not cross-arm seam contamination.
First differing action: `session-101466835` seat 0, step 170 (day 7 hour 2), hand 7
walks WEST under the champion and NORTH under base.

ROOT CAUSE. `rlv3.arm.parse("base")` returns `{"kind": "rlv3", "preset": "base"}`.
`rl/rollout.py:530` applies executor flags from `spec.get("exec") or
(spec.get("rules") or {}).get("exec")` — the parsed spec carries neither, so
champ-h10's three executor rules (`lastwater`, `h23`, `harvest_visit`) are never
applied. Meanwhile `Rlv3Config.load()` DOES return those same rules and hands the
knob half to `AllocationPlanProvider`. So `base` is a third agent: champ-h10's day
plan run under the SHIPPED executor.

WHY THE TESTS MISSED IT. `rlv3/tests/` compares plan objects at dawn, and the plans
ARE identical on 48 mid-season boards across every attribute — the divergence is
entirely in the executor, which no plan-level test can see. The smoke test compared
against a `bundle:` arm on one session rather than against a dev arm on a paired draw.

CONSEQUENCE. Every dose measured against `base` is measured against a broken control,
and every dose measured against champ-h10 carries a constant −1 win / −$4,549
handicap that has nothing to do with the dose.

FIX. `rlv3.arm.parse` must put the resolved `exec` flags (and the rules dict) on the
spec it returns, so `rl/rollout.py` applies the same executor the knob half assumes.
Then re-assert identity as a TEST: `dev:rlv3:base` game-for-game equal to
`dev:knobs:tools/rules/rules_h8_ship2_hv.json` on at least 12 paired games.
