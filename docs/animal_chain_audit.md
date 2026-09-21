# The executor, audited against README.md

Written 2026-09-21. Pass 1 read the animal chain against the rulebook and
gated the corrections. Pass 2 answered "does the executor understand max
yields" and turned the whole rulebook into a test suite. Pass 3 separated
"the planner never asked" from "the executor would not do it", and tested
whether past results survive an executor swap at all.

- [Part 1 — the animal chain](#part-1--the-animal-chain) (the
  `BUY_ANIMAL -> PICKUP -> walk -> PLACE` path and the `n` on the carry ops)
- [Part 2 — max yields](#part-2--does-the-executor-understand-max-yields)
- [Part 3 — the rule test suite](#part-3--the-rule-test-suite-ground-truth)
- [Part 4 — commanded intents: does it OBEY?](#part-4--commanded-intents-does-the-executor-obey)
- [Part 5 — gating "nothing transfers"](#part-5--gating-nothing-transfers-every-closed-experiment-must-be-reopened)
- [Part 6 — bad actions and locked ground](#part-6--bad-actions-what-the-engine-does-and-whether-we-ever-emit-one)
- [Part 7 — fixing the 28: the real cause](#part-7--fixing-the-27-commitment-rejected-and-the-real-cause-found)
- [Part 8 — the underfoot fix, shipped](#part-8--the-underfoot-fix-shipped-to-kagentschedulerpy)
- [Part 9 — 4th executor, and BC re-opened](#part-9--h11-uf-as-a-fourth-executor-and-bc-re-opened)
- [Part 10 — CORRECTION: a harness bug](#part-10--correction-a-harness-bug-understated-every-obedience-number)
- [Part 11 — why task.DIG was stuck at 60%](#part-11--why-taskdig-was-stuck-at-60)
- [Part 12 — the drift was one line; melon fertilisation refuted](#part-12--the-dev-tree-drift-was-one-line-and-melon-fertilisation-is-refuted)
- [Part 13 — old planner weights on the fixed executor](#part-13--the-old-planner-weights-on-the-fixed-executor)
- [Part 14 — risk is settled null; the rent rule is +$4,336](#part-14--the-risk-analysis-was-already-applied-what-it-sat-on-was-never-shipped)
- [Part 15 — hiring rollout: money clears, wins do not](#part-15--the-hiring-rollout-money-clears-wins-do-not)
- [Part 16 — melon risk: right diagnosis, refuted fix](#part-16--melon-risk-the-diagnosis-was-right-the-fix-is-refuted)
- [Part 17 — future shop demand: +$676, shipped](#part-17--the-shops-that-have-not-opened-yet-676--249-shipped)

---

# Part 1 — the animal chain

Scope: the `BUY_ANIMAL -> PICKUP -> walk -> PLACE` chain (the coop/pasture ->
livestock path) and the `n` quantity on the carry ops, as champ-h8 implements
them.

Every rule quoted is from `README.md` and was already re-verified against the
engine in `docs/rules_verified.md` (286/286). Every count is measured with
`tools/` probes on the **frozen bundle**, never the dev tree.

---

## The chain the README specifies

| step | rule | source |
| --- | --- | --- |
| `BUY_ANIMAL` | the head lands in the **shed**, counts against `shedCapacity`, and the order is **refused when the shed is full** | README Market Action + Shed; engine `_commit_unit` |
| `PICKUP <item> [n]` | moves **up to** n from the shed into the acting unit's inventory, clamped to stock, default 1; any shed item including animals | README Shed |
| `BUILD_COOP` / `BUILD_PASTURE` | builds on an unoccupied tile | README Terrain |
| `PLACE <item> [n]` | on a matching **empty** structure: places **exactly one** head, **`n` is ignored**. Otherwise, adjacent to the shed, it is a shed drop of up to n | README Animals |
| `DIG` | removes an **empty** coop/pasture; **no-op once an animal is on it** | README Terrain |
| midnight | every unit inventory is dumped into the shed; **overflow is discarded** | README Shed / Farmer |

Two consequences do the work below. A structure is a **one-way commitment of a
tile** once a head lands on it, and an animal left in a pocket at midnight is
**returned to the shed for free** — or destroyed outright when the shed is at
100.

---

## What champ-h8 assumes, and what is false

### FA-1 — a head in a pocket is treated as a head we do not own (buy side)

`planner.make_champion_plan`:

```python
want = targets[kind] - herd.get(kind, 0) - view.shed.get(kind, 0)
```

`herd` counts placed animals, `view.shed` counts the shed. An animal a hand is
carrying to its pasture is in **neither**, so the plan reads the herd as one
short and orders a replacement at $300–$500. **FALSE.**

### FA-2 — a head in a pocket is treated as needing no structure (build side)

Same function, four lines later:

```python
pipeline = plan.animal_buys.get(kind, 0) + view.shed.get(kind, 0)
need = pipeline - empty_by_kind.get(structure, 0)
```

The other half of the chain disagrees with this outright —
`scheduler.build_tasks` counts shed **plus pockets**:

```python
available[kind] = view.shed.get(kind, 0) + sum(
    view.unit_inv(i).get(kind, 0) for i, _, _ in me.units)
```

so the executor hands the pocketed head a `PLACE` task while the planner never
queues it a structure. **FALSE**, and by README Terrain it does not self-heal
within the day: the head rides in the pocket until midnight returns it to the
shed, where the *next* day's plan finally sees it.

### FA-3 — a carrier can usefully hold two head

`scheduler.assign`, global provisioning:

```python
batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
```

README Animals: `PLACE` "places one animal from inventory onto the tile. The
`n` argument is ignored." So head #2 is dead weight until its carrier reaches a
**second** free structure, and README Shed's midnight drop then returns it to
the shed (or discards it at a full shed). The comment four lines above the
constant already states the correct rule — "any unit may carry **one** to its
pasture"; only the constant disagrees. **FALSE.**

The same block has a second half to it: the dispatcher picks
`min(free, key=distance-to-shed)`, which is exactly the unit that just picked
up and is still standing on the shed tile, so it stacks a second head on a
carrier that already has one even at `batch = 1`.

### FA-4 — `["PLACE", item, n]` at the shed always means "drop n into the shed"

`scheduler.assign`, shed-overflow path:

```python
total = sum(n for item, n in inv.items()
            if not (item in ("GOOSE", "COW", "SHEEP") or ...))   # animals excluded
...
item = max(inv, key=inv.get)                                     # animals INCLUDED
actions[idx] = ["PLACE", item, n]
shed_room -= n
```

The block decides whether to shuttle by counting cargo **without** animals,
then names the item over the **whole** inventory. README: `PLACE` resolves the
animal-placement branch first, so naming an animal while standing on a matching
empty structure puts one head on the tile and nothing in the shed, and
`shed_room -= n` is then wrong. **FALSE but latent** — 0 occurrences in the six
games traced, because the `DROP` branch always had room. Reachable: the build
queue sorts empties by `_dist(p,4,4) + _dist(p,5,5)`, and all four shed-access
tiles tie at 2 for the minimum, so structures land on them (measured 9 of 46).

---

## Gate: every correction is a no-op or a regression

`tools/ladder_gate.py --n 60`, paired on identical held-out ladder worlds,
base champ-h8. Two independent blocks (`--seed 0`, `--seed 1`), 346 worlds
each. `h9-*` are `champ-h8` with the named fixes patched into the frozen
bundle by `tools/make_chain_variant.py`.

| arm | fixes | money/game | z | win | discordant b/c | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| champ-h8 (block 0) | — | −$3,767 | | 0.295 | | |
| h9-build | FA-1 + FA-2 | **+$0** | 0.00 | 0.295 | 0 / 0 | 1.00 |
| h9-place | FA-4 | **+$0** | 0.00 | 0.295 | 0 / 0 | 1.00 |
| h9-carry | FA-3 | −$189 ± 785 | −0.24 | 0.272 | 24 / 16 | 0.268 |
| h9-chain | all three | −$189 ± 785 | −0.24 | 0.272 | 24 / 16 | 0.268 |
| champ-h8 (block 1) | — | −$3,257 | | 0.315 | | |
| h9-carry | FA-3 | −$853 ± 909 | −0.94 | 0.280 | 30 / 18 | 0.111 |
| h9-chain | all three | −$853 ± 909 | −0.94 | 0.280 | 30 / 18 | 0.111 |

**Pooled, 692 worlds:** money −$521 ± 601 (z −0.87, null); wins **54 base-only
vs 34 candidate-only, McNemar p = 0.042**.

### FA-1 and FA-2 are inert, and here is why

`h9-build` is **bit-identical** to champ-h8 — 0 discordant worlds out of 346,
$0 difference. The day plan is rebuilt only when `memory.day_plan_day !=
view.day`, i.e. at hour 0, and the engine empties every unit inventory at
midnight (`_drop_inventories_to_shed`, then `private["inventories"] = [{}]`,
kaggriculture.py:882). **No pocket is ever non-empty at plan time**, so the
term the planner was missing is always zero.

They are still real contradictions and the fix is shipped into `kagent/`:
`tools/rollout_plan.py` and the MPC provider re-plan off-cadence, and either
would walk straight into them. `h9-place` is likewise +$0/0 discordant, as
predicted for a latent branch, which also makes these two arms a genuine zero
control for the instrument.

### FA-3 is a correct rule reading and a wrong counterfactual

The rule is right: `PLACE` moves one head and ignores `n`. The inference —
"so a second head is dead weight" — is not. Engine-level counts over the same
3 seeds vs pub-ahmed-v38:

| | champ-h8 | h9-chain |
| --- | --- | --- |
| PICKUP actions spent on animals | 43 | **55** (+28%) |
| head actually placed | 48 | 48 |
| head bounced back to the shed at midnight | 1 | **7** |

Two head in one pocket is **one** shed trip serving **two** structures, and
the structures cluster next to the shed because the build queue sorts empties
by `_dist(p,4,4) + _dist(p,5,5)`. One head per carrier buys 12 extra shed
round-trips per 3 games for the same 48 placements — and it made the midnight
bounces *worse*, not better, because the delivery is spread over more carriers
and more of them are still in transit at nightfall. The `n` argument
constrains the PLACE action; it says nothing about the economics of the trip.

**Not shipped.** This is the same shape as the geese swap and the melon burst:
observationally correct, refuted under intervention. See
`kaggriculture-observational-vs-interventional`.

---

## Checked and found TRUE (no change)

- `["PLACE", kind]` for a delivery omits `n` → defaults to 1. Correct.
- `kinds = ("GOOSE",) if t["kind"] == "COOP" else ("COW", "SHEEP")` — a goose
  cannot go on a pasture. Correct.
- `FEED` carries `need="WHEAT"`, one unit per animal. Correct.
- `empty_structures()` uses `not t.get("animal")`; the engine writes no
  `animal` key at all on a fresh structure, and `PLACE` tests `"animal" not in
  tile`. Equivalent.
- `SHED_ACCESS` is used as a standing position only, which is what makes shed
  ops legal from a locked tile. Correct.
- `_sell_orders` skips animals. Correct.

## Tried, measured, rejected

**Gating the structure build on affordability.** README makes `BUY_ANIMAL`
refusable (no money, full shed), so `plan.animal_buys` is a request and
committing a tile to it is speculative. Capping the build pipeline at
`min(animal_buys, bank // cost, shed_room)` is the faithful reading — and it
loses. The champion plan buys animals *mid-day* on wool/milk cash by design
("the herd wins the intra-day cash race"), so hour-0 bank is near zero and the
gate suppressed almost every build: `stranded_shed_turns` doubled 107 → 216
over 3 games while `idle_pasture_turns` fell. The speculative build is load
bearing; only 4 of 24 GOOSE orders actually went unfilled. Not shipped.

---

## Regression tests

`tests/test_animal_chain.py` asserts the four rules directly against a
**bundle** (`KAGG_BUNDLE=<name>`, default `champ-h8`), not against `kagent/`,
because the dev tree and the frozen champion have diverged before.

On `champ-h8`: 4 of 5 fail (FA-1, FA-2, FA-3 twice). FA-4's guard passes only
because the synthetic scenario does not reach the branch.

On the dev tree after FA-1/FA-2/FA-4 landed: 3 pass, 2 `xfail`. The two
`xfail`s are the FA-3 pair — the assertion states the README rule correctly
and the code deliberately violates it, because the correction was measured
worse. The `reason=` string carries the number so nobody re-opens it blind.

---

## Tools

- `tools/make_chain_variant.py` — patches the frozen bundle; `--fixes
  build,carry,place` selects the corrections independently.
- probes used for the counts: engine-level instrumentation of
  `_apply_unit_action` / `_commit_unit` / `_drop_inventories_to_shed`
  (BUY/PICKUP/PLACE outcomes, midnight discards) and a per-turn latency probe
  (stranded-shed turns, pocketed turns, idle-structure turns).

---

# Part 2 — does the executor understand max yields?

Partly. It gets every *structural* rule right and the *cap* rule wrong, in the
same place three times: it never compares `yield_units` against the cap before
pricing an action.

**What the engine actually does.** Every accumulation is clamped:

| | engine | README |
| --- | --- | --- |
| one-time crop | starts at `yield_units = 1`; `WATER` does `min(max_yield, yield_units + bonus)` inside the window | "watering during the bonus window will add one unit per day ... Fertilized plants add 2 per day instead" |
| ongoing crop | nightly production does `min(max_yield, yield_units + 1 or 2)` | "production is capped at 4 scheduled yields" |
| animal | `yield_units = min(max_held, yield_units + base + bonus)`, and the banked bonus is **popped before** the clamp | "Max Yield for animals is `max_held`, the cap on *unharvested* product"; "`pending_care_bonus` is capped indirectly by the per-animal `max_held` cap" |

**Where champ-h8 gets it wrong.**

1. **Watering priority ignores the cap.** `scheduler.build_tasks` prices a
   water by window membership alone:
   ```python
   in_window = (crop["max_yield_day"] + 1) // 2 <= age <= crop["max_yield_day"]
   pri = P_TEND if in_window else P_FILLER
   ```
   There is no `yield_units < max_yield` test. README spells the consequence
   out for melon: the window is ages 6–12 but "base 1 plus one unit per
   watered day reaches the cap of 6 at age 10, so ages 11–12 add nothing.
   Fertilizing reaches the cap at age 8." So a melon gets two `P_TEND`
   waterings worth zero units — four if it was fertilised — in our
   second-most-urgent band, on the days around the melon clear.
2. **`max_held` is read and thrown away.** In the animal loop:
   ```python
   max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
   if held >= 2 or (held > 0 and (plan.liquidation or day >= 27)):
   ```
   `max_held` is never used again. The rule is a flat `held >= 2`, which does
   not know that a sheep cared for three days produces `1 + 3` at once, so
   `4 + 4 -> min(6, 8)` destroys two wool **and** the care bank that paid for
   them. Measured below as `HARVEST.animal_before_clip`, missed on both of the
   late-game cases.
3. **The two "max yield" numbers are not the same number.** README's Object
   Types column says melon Time to Max Yield 10; the engine's `max_yield_day`
   is 12, and the executor uses the engine value for *both* the window and the
   ripeness test. It survives only because the `yields >= max_yield` arm of
   `ripe` happens to fire at age 10 when watering was perfect.

The ongoing-crop side it gets *right*: `yields >= crop["max_yield"] - 1`
harvests a strawberry before the next production clips, which is the correct
reading of the same cap.

---

# Part 3 — the rule test suite (ground truth)

`tools/rule_tests.py`. TDD against the rulebook: one test rule per README
action, cases mined from recorded games rather than invented.

**Method.** Mid-game `env` resume is unfaithful, so a recorded state is only
reachable by replaying it. For each case: replay both recorded scripts from
step 0 on the session's own seed (fidelity is exact), hand OUR executor the
wheel at step T, judge it, then PASS to the end. Our agent is *called* on every
step from 0 and its output discarded before T, so its `Memory` and its hour-0
day plans evolve along the recorded trajectory — a cold call at hour 14 would
make it plan off-cadence, which no real game does.

**Ground truth.** A case exists only where a **2000+ rated seat performed the
action** and the README precondition held. Their doing it is the evidence it
was right there; the corpus defines correctness, not us. 286 cases over 28
rules, mined from 40 reference seats, spread across six day bands (0–2, 3–6,
7–10, 11–15, 16–21, 22–29).

**Two verdicts, three tiers.**

| | |
| --- | --- |
| RIGID | the op at **exactly** step T on **exactly** that tile — agreement with the reference decision |
| CAPABILITY | the op on **any** tile where the same precondition holds, any time from T until the opportunity closes — can it do the thing at all |
| T1 | emitted where the precondition held |
| T2 | the engine did not silently no-op it |
| T3 | the README rule's **value materialised** — the yield unit landed, the bonus banked, the seed survived the night, the fill was above the floor |

## Result: champ-h8

| rule | n | rigid T3 | cap T1 | cap T2 | cap T3 | days missed |
| --- | --- | --- | --- | --- | --- | --- |
| `BUILD_COOP.for_a_goose_we_hold` | 7 | 0% | 14% | 14% | **14%** | 6, 10, 10, 11, 19, 21 |
| `FERTILIZE.one_time` | 10 | 0% | 20% | 20% | **20%** | 6, 6, 7, 8, 12, 13, 18, 20 |
| `BUILD_PASTURE.for_stock_we_hold` | 10 | 10% | 50% | 50% | **50%** | 0, 6, 15, 16, 19 |
| `DIG.clears_a_weed` | 10 | 0% | 50% | 50% | **50%** | 4, 12, 15, 20, 21 |
| `DROP.nothing_is_discarded` | 12 | 42% | 50% | 50% | **50%** | 2, 3, 10, 10, 20, 21 |
| `BUY_LAND.the_quadrant_gets_used` | 6 | 17% | 50% | 50% | **50%** | 6, 11, 11 |
| `BUY_ANIMAL.reaches_a_structure` | 10 | 30% | 60% | 60% | **60%** | 0, 14, 16, 16 |
| `HARVEST.animal_before_clip` | 6 | 33% | 67% | 67% | **67%** | 25, 25 |
| `PLANT.survives_its_first_night` | 12 | 8% | 67% | 67% | **67%** | 1, 4, 10, 18 |
| `BUY_SEED.gets_planted` | 12 | 58% | 67% | 67% | **67%** | 3, 10, 15, 23 |
| `FERTILIZE.ongoing` | 7 | 14% | 71% | 71% | **71%** | 10, 23 |
| `WATER.bonus_window` | 12 | 8% | 83% | 83% | **83%** | 13, 28 |
| `WATER.fertilised_production` | 6 | 0% | 83% | 83% | **83%** | 26 |
| `CARE.banks_a_bonus` | 12 | 0% | 92% | **92%** | **83%** | 25, 26 |
| `SELL.above_the_floor` | 12 | 75% | 83% | 83% | **83%** | 18, 25 |
| `BUY_PRODUCT.wheat_is_fed_not_resold` | 12 | 67% | 83% | 83% | **83%** | 10, 12 |
| `WATER.survival` | 12 | 17% | 92% | 92% | **92%** | 27 |
| `FEED.keeps_it_alive` | 12 | 8% | 92% | 92% | **92%** | 16 |
| `HARVEST.one_time_ripe` | 10 | 10% | 100% | 100% | **100%** | — |
| `HARVEST.ongoing_before_clip` | 2 | 0% | 100% | 100% | **100%** | — |
| `COLLECT_FERTILIZER.use_it_or_lose_it` | 12 | 17% | 100% | 100% | **100%** | — |
| `PLACE.animal_on_structure` | 10 | 70% | 100% | 100% | **100%** | — |
| `PICKUP.gets_what_it_asked_for` | 12 | 92% | 100% | 100% | **100%** | — |
| `MOVE.north_is_not_off_board` | 12 | 25% | 100% | 100% | **100%** | — |
| `MOVE.south_is_not_off_board` | 12 | 42% | 100% | 100% | **100%** | — |
| `MOVE.east_is_not_off_board` | 12 | 50% | 100% | 100% | **100%** | — |
| `MOVE.west_is_not_off_board` | 12 | 25% | 100% | 100% | **100%** | — |
| `HIRE.the_hand_does_work` | 12 | 100% | 100% | 100% | **100%** | — |
| **ALL** | **286** | **32%** | **79%** | **79%** | **79%** | |

## The headline: it is omission, not malformation

**`T1 == T2 == T3` in 27 of 28 rules.** Whenever champ-h8 emits an action in a
state where the README precondition holds, the engine accepts it *and* the
rule's value materialises — 0 no-ops, 0 mistimings, across 286 cases. Every
failure is a state where it simply **never emitted the action anywhere, all
day**.

That is a much narrower diagnosis than "the executor is faulty". The ops are
formed correctly and timed correctly; the *task set* is short. It also means
the earlier `moves/action` framing was measuring a symptom: a worker that never
proposes the task cannot be routed to it.

**The one exception is `CARE`** — 92% T2 against 83% T3, the only rule where we
do the thing and the value does not land. README Animal Care: the bonus banks
only on a day the animal was **also fed** ("basic needs first"). One case
(day 26) cares an animal that then goes unfed, so the action is spent for
nothing. This is the same coupling `docs/executor_problems.md` §6 flagged from
the other direction.

**Worth separating from the rest:** `BUILD_COOP` at 14% is not a bug —
champ-h8 sets `knobs["geese"] = 0` and never builds a coop at all. The rule is
kept because it is the only test that exercises the COOP→GOOSE path at all.

## Rigid vs capability

Rigid T3 is 32% against capability's 79%. The gap is *not* a defect: the rigid
verdict asks for the same tile on the same turn as a different agent's round,
and a scheduler that serves the same tile three turns later is equally correct.
Read rigid as agreement (highest where the decision is forced — `HIRE` 100%,
`PICKUP` 92% — lowest where a farm offers many equivalent tiles), and read
capability as the pass/fail.

## Artifacts

Every case writes a watchable replay to `tests/replays/<agent>/`:

- `<rule>-<nn>.html` — the window, roughly day T−1 to day T+1 (1.4–4 MB)
- `<rule>-<nn>-full.html` — the whole season, for failing cases only (~30 MB)

347 files, 2.3 GB. The 14.7 MB visualiser bundle is pulled out into one shared
`_viewer.js` per directory and loaded as a classic `defer` script (a
`type="module"` `src` would be blocked from `file://`), so the pages open by
double-click; `tests/replays/serve.sh <agent>` serves them over HTTP if a
browser objects. Verified: the viewer loads, the payload carries the right seed
and the sliced step range, and no console errors. Not verified: the board
paints — it renders black under automated Chrome here, and so does an
untouched self-contained replay from the repo root, so that is the harness's
browser context and not these files.

## Running it

```bash
PYTHONPATH=. uv run python tools/rule_tests.py mine --sessions 40 --per-band 2
PYTHONPATH=. uv run python tools/rule_tests.py run  --agent champ-h8
PYTHONPATH=. uv run python tools/rule_tests.py run  --agent champ-h8 --no-html
PYTHONPATH=. uv run python tools/rule_tests.py report
```

`rl/rule_cases.json` is the mined case list (commit it to freeze the suite);
`rl/rule_results.json` is the last run. Mining 40 seats takes ~70 s, running
286 cases ~6 min without HTML.

## What this suite does NOT establish

It is a **capability** test, not a value test. A rule at 100% means the
executor can perform that action where a strong player did; it says nothing
about whether doing it more often earns money — and this project has four
recorded cases of raising an action count and losing money
(`docs/executor_problems.md` §7), plus the CARRY(n) refutation in Part 1. Use
it to find what the executor *cannot* do; use `tools/ladder_gate.py` to decide
whether teaching it to do so is worth shipping.

---

# Part 4 — commanded intents: does the executor OBEY?

Part 3 measured whether the executor does the right thing *unprompted*, and
21% of the time it does not. That number cannot say whose fault it is — the
planner may never have asked. The executor's actual contract is narrower: the
planner hands it intents (buy the 4th quadrant, build a coop, fertilise this
plant, fetch n of an item, dig that tile) and it must carry them out.

`tools/exec_commands.py` injects the intent into the **same mined states** and
checks it lands. Three layers, all through the real machinery — the extra task
goes into `build_tasks`' return value and the plan fields are mutated on the
live `DayPlan`, exactly as the planner would hand them over, so nothing
bypasses assignment, provisioning or the market controller.

| layer | injected | exonerates |
| --- | --- | --- |
| TASK | a `Task` at `P_SURVIVAL` on a named tile | routing + execution |
| DELIVER | a `Task` whose `need` is only in the shed | provisioning + carry |
| PLAN | a `DayPlan` field, re-asserted every turn until done | `build_tasks` + market controller |

Read against Part 3: **unprompted FAIL + commanded PASS = the planner never
asked. unprompted FAIL + commanded FAIL = the executor cannot do it.**

## Result: champ-h8, 217 commands

| command | layer | n | OBEYED | median latency | unprompted | reading |
| --- | --- | --- | --- | --- | --- | --- |
| `deliver.PLACE` | deliver | 10 | **100%** | 1 | 100% | fine both ways |
| `task.BUILD_PASTURE` | task | 10 | **100%** | 5 | 50% | planner never asked |
| `task.CARE` | task | 12 | **100%** | 6 | 83% | planner never asked |
| `plan.BUY_LAND` | plan | 6 | **100%** | 2 | 50% | planner never asked |
| `plan.COOP` | plan | 17 | **100%** | 7 | 41% | planner never asked |
| `plan.BUY_ANIMAL` | plan | 20 | **100%** | 6 | 80% | planner never asked |
| `plan.PLANT` | plan | 24 | 96% | 1 | 67% | mostly planner |
| `task.COLLECT_FERTILIZER` | task | 12 | 92% | 5 | 100% | — |
| `task.HARVEST` | task | 18 | 78% | 6 | 89% | **executor** |
| `deliver.FERTILIZE` | deliver | 17 | 76% | 3 | 41% | both |
| `plan.HIRE` | plan | 12 | 75% | 2 | 100% | **executor** |
| `task.WATER` | task | 30 | 73% | 8 | 87% | **executor** |
| `task.BUILD_COOP` | task | 7 | 71% | 4 | 14% | both |
| `task.DIG` | task | 10 | 60% | 9 | 50% | **executor** |
| `deliver.FEED` | deliver | 12 | **58%** | 6 | 92% | **executor** |
| **ALL** | | **217** | **85%** | | | |

Two clean groups fall out.

**Every PLAN-layer intent that is purely economic lands 100%** — buy the
quadrant, build the coop, buy the animal. Those are not executor problems at
all; `plan.COOP` obeys 100% against 41% unprompted, so the entire coop gap is
the planner never asking.

**Every failure is a TASK on a named tile.** And `task.WATER` is *worse*
commanded (73%) than unprompted (87%), which looks paradoxical until you see
why: the unprompted test accepts **any** qualifying tile, the command names
**one**. The executor waters *a* plant and steps off *the* one. Same shape for
`deliver.FEED`: 92% unprompted, 58% commanded.

## Why: it arrives, then walks away — 27 out of 27

For every unobeyed tile command, the minimum distance from any of our units to
the target tile across the whole window:

| command | fails | never arrived | **arrived and did something else** |
| --- | --- | --- | --- |
| `task.WATER` | 8 | 0 | **8** |
| `deliver.FEED` | 5 | 0 | **5** |
| `task.DIG` | 4 | 0 | **4** |
| `deliver.FERTILIZE` | 4 | 0 | **4** |
| `task.HARVEST` | 4 | 0 | **4** |
| `task.BUILD_COOP` | 2 | 0 | **2** |
| **total** | **27** | **0** | **27** |

**Not once did the executor fail to reach the tile.** In all 27 cases a unit
stood on it — and the op it emitted there was a move: `SOUTH`, `EAST`,
`WEST`, `NORTH`. It steps off a tile carrying an unambiguous `P_SURVIVAL`
order.

This is `docs/executor_problems.md` §1 ("73.6% of our moves leave a doable
task underfoot") demonstrated at the strongest possible priority, which rules
out the obvious explanation: it is not deprioritising the work, because there
is nothing above `P_SURVIVAL` to prefer.

## Which executor change fixes it — and which does not

The same 217 commands against three executors that share one planner and
differ only in `scheduler.py`:

| command | champ-h8 | match-all (global matching) | a-commit (cross-turn commitment) |
| --- | --- | --- | --- |
| `task.WATER` | 73% | 73% | **87%** |
| `task.HARVEST` | 78% | 78% | **89%** |
| `task.DIG` | 60% | 60% | **70%** |
| `task.BUILD_COOP` | 71% | 71% | **86%** |
| `deliver.FEED` | 58% | 58% | **75%** |
| `task.COLLECT_FERTILIZER` | 92% | 92% | **100%** |
| `deliver.FERTILIZE` | 76% | 76% | 76% |
| `deliver.PLACE` | 100% | 100% | **60%** |
| `plan.*` | 75–100% | 75–100% | 75–100% |
| **ALL** | **85%** | **85%** | **87%** |

**`match-all` is indistinguishable — identical latency on 213 of 217 cases
and the same 32 failures.** It is genuinely different code (global-greedy
bipartite matching replacing the index-order `pick` loop), so this is a
measurement, not a harness artifact.

That **refutes the claim-order explanation.** `assign` sets `claimed` when a
unit merely steps *toward* a task, and the passes run `for idx in
sorted(free)`, so the natural story was that a distant unit 0 reserves the
task and the unit standing on it is locked out — `executor_problems.md` §3.
If that were the mechanism, resolving claims cheapest-pair-first would fix it.
It changes nothing.

**`a-commit` does move it**, and in the predicted direction on exactly the
tile commands — +14pp on WATER, +17pp on FEED, +15pp on BUILD_COOP. That is
`executor_problems.md` §2, "it re-decides every turn, so workers oscillate",
which that document lists as **untried**. It is no longer untried: committing
a unit to its target across turns is the lever for the walk-away defect. It
also costs something real — `deliver.PLACE` falls 100% → 60%, because a unit
committed to a plant will not divert to deliver the animal in its pocket.

Note what this does *not* say. `a-commit` gated **null on money** (+$3,233 in
self-play, nothing on the ladder). So obedience and money are separable here:
an executor change can measurably fix the defect and still not pay. Treat the
obedience number as a diagnostic, never as a gate.

---

# Part 5 — gating "nothing transfers, every closed experiment must be reopened"

The standing claim after Part 4 was: every knob and every BC run was fitted
against champ-h8's executor, so none of it transfers and the whole corpus of
negatives is void. Reopening ~20 closed experiments is expensive enough to be
worth testing the claim first.

**Test** (`tools/knob_transfer.py`): take 6 knobs with known effects, apply
each to three executors that share one planner and differ only in
`scheduler.py`, and gate every arm against **its own** executor on the same
held-out ladder worlds. If a knob's effect is a property of the plan, its sign
survives the executor swap. If the executor was carrying the result, it flips.

`--n 30 --seed 0` → 180 worlds per arm, 18 arms + 3 bases.

| knob | E0 champ-h8 | E1 match-all | E2 a-commit |
| --- | --- | --- | --- |
| `geese 4` | **−5,618 ± 885** | **−4,952 ± 886** | **−6,178 ± 881** |
| `carrot_standing 16` | **−4,855 ± 854** | **−4,705 ± 840** | **−2,193 ± 885** |
| `hires_delta +2` | **−3,194 ± 687** | **−2,742 ± 696** | −372 ± 901 |
| `straw_target 40` | −753 ± 602 | −440 ± 543 | −1,095 ± 719 |
| `wheat_engine 40` | +61 ± 359 | +156 ± 332 | −459 ± 210 |
| `melon_opening 16` | +0 ± 0 | +0 ± 0 | +0 ± 0 |

| pair | Spearman | sign agreement | sign agreement on arms the first resolves (>2 se) |
| --- | --- | --- | --- |
| E0 vs E1 | +1.00 | 6/6 | 3/3 |
| E0 vs E2 | +0.71 | 5/6 | **3/3** |
| E1 vs E2 | +0.71 | 5/6 | 3/3 |

`melon_opening 16` returning **+$0 ± 0 on all three** is a real zero control:
16 is already the default, so the arm is bit-identical to its base and the
instrument correctly reports nothing.

## Verdict: the claim is too strong on sign, and right on magnitude

**Every knob the instrument actually resolves keeps its sign on all three
executors** — including `a-commit`, which is behaviourally very different (it
changes 144 of 217 commanded outcomes and lifts obedience 85% → 87%). Geese
do not become good on a better executor; they get *worse* (−$6,178). The one
sign disagreement, `wheat_engine`, is +$61 ± 359 on champ-h8 — noise, and a
sign flip inside noise is not evidence.

**But magnitudes move a lot.** `hires_delta +2` goes from −$3,194 ± 687 to
−$372 ± 901 — a 9× shrink, and on `a-commit` it is no longer resolvable at
all. `carrot_standing` halves. So "hires +2 is catastrophic" genuinely was
partly executor-carried; "hires +2 is bad" was not.

**The practical rule this buys:**

> Reopen a closed experiment only if its measured effect was inside ~2 se.
> Those are the results the executor could have been carrying. Anything that
> was resolved at 3σ+ on champ-h8 keeps its sign on a different executor and
> stays closed.

Applied to the recorded negatives, that keeps geese, standing carrots,
±2 hires, `straw_target 21`, `carrot_day 18` and `premium_scale 1.3` closed,
and puts the borderline set — `straw_target 40`, `wheat_engine` variants,
`land2_margin`, `carrot_cap`, `straw_early` — back in scope.

## What this does NOT gate

- **BC / the learned ranker is untested here.** This gates the *knob* half of
  the claim. A policy cloned from a teacher is a much tighter coupling to the
  executor than a scalar knob, and `exec_bc.py`'s null could still be
  executor-carried. Re-run that one on `a-commit` before reopening or closing
  it.
- Six knobs, one seed block, 180 worlds per arm. Enough to resolve $3–6k
  effects at 3–7σ; not enough to say anything about sub-$1k arms — which is
  precisely the set the rule above sends back for re-measurement anyway.
- All three executors share one planner. It says nothing about transfer across
  a *planner* change.

---

# Part 6 — bad actions: what the engine does, and whether we ever emit one

## What the rules say

README Actions closes with "Invalid actions are silent no-ops", and
`_apply_unit_action`'s docstring says the same. There is no error, no penalty
and no signal — the unit-turn is simply gone, which makes a wasted op
indistinguishable from a `PASS` in every other instrument we have.

Locked ground, precisely (kaggriculture.py:325–345, README Movement):

| op | on a LOCKED tile |
| --- | --- |
| `NORTH`/`SOUTH`/`EAST`/`WEST` | **allowed** — locked tiles are passable, and blocking them would strand a hand spawned on a locked shed-access tile |
| `PICKUP` / `DROP` / `PLACE`-into-shed | **allowed** — these resolve *before* the locked guard; they use the tile only as a standing position |
| everything else | `if tile == "LOCKED": return` — silent no-op, **the turn is spent** |

So your reading is right: a non-movement task on a locked quadrant is ignored,
and it costs the unit-turn. Two more invalid-action paths the engine hides
entirely, which no engine-level hook can see:

- **atomic PLANT** — if one turn requests more `PLANT <crop>` than seeds held,
  the interpreter rewrites **all** of that crop's plants to `PASS` before the
  engine sees them (kaggriculture.py:925–935)
- **market cap** — orders past `maxMarketOrdersPerTurn` are truncated before
  parsing, and a malformed sub-op aborts the rest of that order

## What champ-h8 actually emits

`tools/noop_audit.py`, 4 games vs pub-ahmed-v38, every emitted op classified
by whether the engine changed anything:

| | |
| --- | --- |
| unit-turns | 27,576 |
| `PASS` | 248 (0.9%) |
| ops emitted | 27,328 |
| **of those, silent no-ops** | **5 (0.02%)** |

The entire failure list is `WATER: not_a_plant` ×5 — a tile that stopped being
a plant between task generation and execution, i.e. a same-turn race.

| hidden path | count |
| --- | --- |
| `PLANT` requests voided by the atomic-seed rule | **0** |
| market orders dropped past the cap | **0** |
| tile op attempted on a LOCKED tile | **0** |

And it uses both permissive locked-tile rules rather than avoiding locked
ground: **895 moves made from a locked tile** (routing through unbought
quadrants) and **290 shed ops from a locked shed-access tile** (a hand spawns
on `(5,4)`, which is locked until NE is bought, and picks up from there
anyway).

**Verdict: the executor's action hygiene is essentially perfect.** It does not
emit invalid actions, it does not over-request seeds, it does not overflow the
market queue, and it never spends a turn on locked ground. That is worth
stating because it closes off the cheapest explanation for Part 4: the 15%
disobedience is **not** malformed orders. It chooses a different *valid*
action.

---

# Appendix — the three questions this suite answers, and the one it does not

| question | instrument | champ-h8 |
| --- | --- | --- |
| does it do the right thing unprompted? | `rule_tests.py` (takeover window) | 79% |
| does it obey a direct order? | `exec_commands.py` (injected intent) | 85% |
| does it ever emit an illegal action? | `noop_audit.py` | 0.02% |
| **does doing more of it earn money?** | `ladder_gate.py` | **answered nowhere above** |

A pure observation-substitution sweep — replay a recorded game, hand each
observation to our executor, read its action — is the cheapest way to get
breadth (thousands of states instead of 286) and is what
`tools/decision_diff.py` already does. What it cannot do on its own is
separate three different things: *choosing* the op, the engine *accepting*
it, and the README rule *paying*. And at any given state a dozen preconditions
hold at once while we only have N units, so the raw rate of "did it do X here"
measures capacity, not capability — which is why the suite uses a takeover
window for capability and an injected command for obedience instead.

## Replay layout

```
tests/replays/<agent>/<ACTION>/<case>.html              rule suite
tests/replays/<agent>-commands/<COMMAND>/<case>.html    commanded intents
tests/replays/<agent>/_viewer.js                        shared, once per agent
tests/replays/serve.sh
```

Filenames carry what, where, when and whose seat —
`DIG.clears_a_weed-04-x4y9-d12-t20-p1.html` is case 04 of that rule, tile
(4,9), day 12, turn 20, seat 1. A `-full.html` beside a case is the whole
season, written for failing cases only.

---

# Part 7 — fixing the 27: commitment REJECTED, and the real cause found

Three variants, all patched onto the frozen champ-h8 bundle, gated on 346
held-out ladder worlds paired against champ-h8 and re-run through the
217-command obedience suite.

| arm | change | money/game | z | win | McNemar p | obeyed | fixes / breaks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champ-h8 | — | — | — | 0.295 | — | 85% | — |
| `h10-cmt` | commit to a P_SURVIVAL target across turns | **−$2,162 ± 572** | **−3.78** | 0.266 | 0.064 | 83% | **0 / 32 fixed, 4 broken** |
| `h10-cmts` | + committed units exempt from the shed shuttle | **−$2,133 ± 571** | **−3.74** | 0.269 | 0.093 | 83% | **0 / 32 fixed, 4 broken** |
| `h10-pad1` | route over unbought ground only at depth ≤ 1 | +$34 ± 20 | +1.71 | 0.295 | 1.00 | 85% | 0 fixed, **0 broken** |

## Commitment is rejected, and it was the wrong theory

`h10-cmt` costs **$2,162 a game at z = −3.78** and **fixes none of the 32
failures**. `--shield` (exempting a committed unit from the shed shuttle)
changes nothing, which also rules the shuttle out.

Why it could not work, in hindsight: commitment only helps a unit that
*already had the task assigned on a previous turn*. The 27 failures are units
that were **never given the task at all** — they were standing on it. There is
nothing to commit to.

(`a-commit` does fix 16 of 32, but it also breaks 12 and replaces the entire
three-pass assignment with a value-density scorer, so its gain is not
commitment — which this arm now isolates and refutes.)

## The real cause: the zone partition

`scheduler.assign` labels every assignment by stage (`ASSIGN_LABEL`, gated on
`MOVE_STATS_ENABLED`). Reading it at the turn a unit stood on the commanded
tile:

| stage that assigned that unit instead | share |
| --- | --- |
| **`zone` (pass 1, own-zone round)** | **68%** |
| `morning` (zone wheat/fertilizer pickup) | 12% |
| `idlehelp` (pass 2) | 11% |
| `shuttle` | 5% |
| `filler` | 5% |

And the direct test of the mechanism — is the commanded tile inside the
standing unit's own zone?

| | |
| --- | --- |
| **tile is in a FOREIGN zone** | **65%** |
| tile is in the unit's own zone | 35% |

Pass 1 searches `zone_tasks[idx % n_zones]` — **only the unit's own zone**.
A unit physically standing on the tile is refused the task because the tile
belongs to a different worker's zone, and pass 1 then hands it something back
in its own zone, so it walks off. Pass 2 (`idlehelp`) could rescue it, but
pass 1 has already consumed the unit.

That is `docs/executor_problems.md` **§4** — "the zone partition
(`idx % n_zones`) assigns a unit only its own zone's tasks, so foreign unit
placements strand it" — which that document lists as one of the two genuinely
untried items. It is now the measured cause, at 65%.

**Three candidate mechanisms, three verdicts:**

| | `executor_problems.md` | verdict here |
| --- | --- | --- |
| index-order claiming | §3 | **refuted** — `match-all` is identical on 213/217 |
| re-deciding every turn | §2 | **refuted** — `h10-cmt` fixes 0/32 and costs $2,162 |
| the zone partition | §4 | **confirmed, 65%** |

**This does not make the fix free.** The nearest thing already tried,
`v-tile` ("a same-tile pass before the band passes", `executor_problems.md`
§7), gated **−$4,907**. Knowing the cause and having a profitable fix are
different things, and §4's own note is that planner and executor have to
change together. Nothing is shipped.

## The locked-ground padding constraint

Design rule: a unit may stand on unbought ground only within 1 tile of owned
land. Measured on champ-h8 **before** building it: 96.5% of locked-ground
unit-turns are already at depth 1, and only 25 moves per 4 games go deeper —
so this makes a near-invariant into an invariant rather than changing
behaviour.

`h10-pad1` enforces it in `_step_toward` (multi-source BFS depth grid, cached
on the unlocked-quadrant key; candidate order is the original's, so with the
padding satisfied it returns exactly what it always did). Result: the single
genuine deep excursion per 4 games is **eliminated**, and the residual depth-2
turns are all `(5,5)` at hour 0 — the engine's `_spawn_hand` ignores locked
status, and `(5,5)` is depth 2 while only NW is owned. The unit walks out on
its next turn.

Gate: **+$34 ± 20, zero discordant worlds, obedience unchanged, nothing
broken.** As predicted it cannot resolve at the ladder's resolution. It is
free and it makes the constraint guaranteed instead of incidental, so it is a
ship-if-you-want-the-guarantee, not a win.

Note what padding can and cannot buy: locked tiles are passable, not
obstacles, so a Manhattan path through them is never *shorter* than one
around. The detour costs no distance; what it buys is the same number of
steps taken over workable ground, which is the `actions per tile arrival`
term (ours 1.67 against the leaders' 2.10).

`tests/test_locked_ground.py` now asserts all of it: no tile op on a LOCKED
tile, shed ops only from shed-access tiles, routing within the padding
(strict for `pad` bundles, <1% for champ-h8), and no silently-discarded
PLANT or market order.

---

# Part 8 — the underfoot fix: SHIPPED to `kagent/scheduler.py`

Part 7 measured the cause: pass 1 searches `zone_tasks[idx % n_zones]` only,
so a unit standing on a task tile is refused it when the tile belongs to
another worker's zone (65% of the 28 unobeyed tile commands).

**The fix**, `tools/make_underfoot_variant.py`, is one branch inserted into
pass 1 between the delivery check and the own-zone check:

```python
task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
if task is None:
    task = _underfoot(idx)      # NEW: unclaimed task on the tile I am on,
                                #      any zone, pri < P_FILLER
if task is None:
    task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
```

Two deliberate narrowings, both because `v-tile` — "a same-tile pass before
the band passes" — already gated **−$4,907** (`executor_problems.md` §7):

- it runs **after** the delivery branch, so a carrier holding an animal still
  delivers first (`deliver.PLACE` must stay at 100%, and does)
- `pri < P_FILLER` — real work only. v-tile grabbed FILLER underfoot too, and
  filler is exactly the low-value work four throughput experiments converted
  at zero.

## Result

| | champ-h8 | `h11-uf` | `h11-uf2` (pri < 2) |
| --- | --- | --- | --- |
| **commands obeyed** | 85% | **92%** | 90% |
| median latency (turns) | 5–9 | **2** | 2 |
| unprompted capability T3 | 79% | **82%** | — |
| unprompted **rigid** T3 | 32% | **51%** | — |
| `deliver.PLACE` | 100% | **100%** | 100% |
| ladder block 0 | — | +$454 ± 874 (z 0.52), win 0.295→0.303 | +$19 ± 879, win 0.280 |
| ladder block 1 | — | +$219 ± 984 (z 0.22), win 0.315→0.301 | — |

**Pooled over 692 worlds: +$337 ± 658 (z = 0.51); wins 46 base-only vs 44
candidate-only — null.**

Per-command obedience, champ-h8 → h11-uf: WATER 73→87, HARVEST 78→94,
FERTILIZE 76→94, COLLECT_FERTILIZER 92→100, BUILD_COOP 71→86, FEED 58→75.
DIG (60%) and HIRE (75%) are unmoved.

**Read this correctly: it is not a money win.** It is money-neutral and
win-neutral, replicated across two blocks, positive in both and negative in
neither — and it fixes the measured defect while breaking nothing. Rigid
agreement nearly doubling (32% → 51%) is the sharpest signal that the right
thing changed: the executor now chooses the same tile a 2000+ seat chose far
more often.

It is the first executor change in this whole audit that is not a regression.
For contrast, on the same instrument: `v-tile` −$4,907, `h10-cmt` −$2,162,
`h9-carry` −$521, `match-all` +$1,381 on margin but −1 win.

## Landed in the executor, not left in a variant

`kagent/scheduler.py` now carries the same patch, and the suites were re-run
from a bundle built out of the dev tree (`tools/build_submission.py --freeze
dev-exec`):

| | champ-h8 | dev-exec (`kagent/`) |
| --- | --- | --- |
| commands obeyed | 85% | **90%** |
| unprompted capability T3 | 79% | **82%** |
| unprompted rigid T3 | 32% | **51%** |
| `pytest tests/` | — | 25 passed, 2 xfailed |

### But `kagent/` as a whole is NOT the gated artifact

| arm | money vs champ-h8 | win |
| --- | --- | --- |
| `h11-uf` (frozen champ-h8 + this fix) | **+$454 ± 874** | 0.295 → 0.303 |
| `dev-exec` (all of `kagent/`) | **−$777 ± 869** | 0.295 → **0.275** |

Same fix, same ladder block, opposite sign. The difference is everything
*else* the dev tree has drifted by — planner, econ, market, features — which
has never been gated and currently costs about **$1,200 a game** relative to
the frozen champion.

So: the fix is in the executor as asked and the executor tests improve there.
**`tools/submit.sh` rebuilds from `kagent/`, so submitting from it today
would ship a −$777 agent.** The gated artifact is `h11-uf`. Gating the rest
of the dev drift is a separate job and has to happen before any submission.

---

# Part 9 — h11-uf as a fourth executor, and BC re-opened

## The remaining 18 failures

After the underfoot fix, 18 of 217 commands still fail: 16 held over from
champ-h8's 32, 2 newly broken (`deliver.FEED#03`, `task.DIG#08`).

| command | remaining |
| --- | --- |
| `deliver.FEED` | #00 x4y3 d01 t21 · #03 x1y4 d03 t17 · #05 x4y2 d07 t21 |
| `task.DIG` | #00 x4y0 d04 t20 · #05 x4y8 d15 t23 · #06 x2y1 d21 t22 · #08 x8y0 d22 t11 |
| `task.WATER` | bonus#10 x1y1 d28 t22 · fert#03 x1y0 d16 t11 · fert#04 x9y1 d26 t20 · surv#10 x1y5 d27 t22 |
| `deliver.FERTILIZE` | one_time#07 x2y8 d20 t11 |
| `task.HARVEST` | ongoing_before_clip#00 x9y1 d28 t18 |
| `task.BUILD_COOP` | #02 x4y1 d10 t19 |
| `plan.HIRE` | #05 d09 · #08 d20 · #09 d19 |
| `plan.PLANT` | BUY_SEED#03 d03 |

Every one has a replay at
`tests/replays/h11-uf-commands/<command>/<case>.html`.

**Where the residue lives: the last quarter of the day.** All 14 tile cases
still *arrive* on the tile, and **10 of 14 are commanded at hours 18–23**.
That points at the end-of-day logistics — `assign` section 6 sends any loaded
unit to the shed from `hour >= TURNS_PER_DAY - 3`, and the shed shuttle in
section 1 grabs units before every work pass — rather than at zone blindness,
which the underfoot branch now covers.

Caveat on that reading: `ASSIGN_LABEL` still reports these as the `zone`
stage, but the underfoot branch now lives *inside* pass 1, so the label no
longer separates the two. The hour concentration is the reliable signal; the
stage attribution is not, and would need a new label to pin down.

## h11-uf as a fourth executor

| knob | E0 champ-h8 | E1 match-all | E2 a-commit | **E3 h11-uf** |
| --- | --- | --- | --- | --- |
| `geese 4` | −5,618 ± 885 | −4,952 ± 886 | −6,178 ± 881 | **−6,206 ± 953** |
| `carrot_standing 16` | −4,855 ± 854 | −4,705 ± 840 | −2,193 ± 885 | **−3,791 ± 904** |
| `hires_delta +2` | −3,194 ± 687 | −2,742 ± 696 | −372 ± 901 | **−3,548 ± 631** |
| `straw_target 40` | −753 ± 602 | −440 ± 543 | −1,095 ± 719 | −521 ± 524 |
| `wheat_engine 40` | +61 ± 359 | +156 ± 332 | −459 ± 210 | −163 ± 347 |
| `melon_opening 16` | +0 ± 0 | +0 ± 0 | +0 ± 0 | +0 ± 0 |

| pair | Spearman | sign | sign on arms the first resolves |
| --- | --- | --- | --- |
| E0 vs E1 | +1.00 | 6/6 | 3/3 |
| E0 vs E2 | +0.71 | 5/6 | 3/3 |
| **E0 vs E3** | **+0.94** | 5/6 | **3/3** |
| E1 vs E3 | +0.94 | 5/6 | 3/3 |
| E2 vs E3 | +0.83 | 6/6 | 3/3 |

**The better-obeying executor agrees with champ-h8 at +0.94** — higher than
any pair except the near-identical E0/E1. And `hires_delta`, the one knob
whose magnitude collapsed on `a-commit` (−$3,194 → −$372), comes back to
**−$3,548** on h11-uf. So a-commit was the outlier, not the rule: that shrink
was a property of a-commit's value-density scorer, not of "a better
executor".

This strengthens Part 5 rather than overturning it. **Knob results are a
property of the plan.** Closed negatives at >2 se stay closed.

## BC is the exception — and it flips

`tools/exec_bc.py export --mode banded` puts the learned ranker in place of
`pick`'s `(urgent, distance, priority)` score, leaving everything else alone.
The ranker weights are unchanged (`build_tasks` was never touched, so the
training candidate sets are still valid); only the executor it sits in
differs.

| arm | base | money/game | z | win |
| --- | --- | --- | --- | --- |
| `bc-h8` | champ-h8 | **−$1,052 ± 902** | −1.17 | 0.295 → 0.277 |
| `bc-h11` block 0 | h11-uf | +$888 ± 906 | 0.98 | 0.303 → 0.318 |
| `bc-h11` block 1 | h11-uf | +$1,578 ± 880 | 1.79 | 0.301 → 0.327 |
| **`bc-h11` pooled, 692 worlds** | h11-uf | **+$1,233 ± 632** | **+1.95** | wins 60 c-only vs 46 b-only (p = 0.21) |

**Same ranker, same weights, different executor: −$1,052 → +$1,233, a swing
of $2,285 a game.** Positive on both blocks, negative on neither, and the win
direction agrees.

This is the one place the "nothing transfers" claim holds, and it is the
place the Part 5 caveat predicted: a cloned *policy* couples to the executor
far more tightly than a scalar knob. `exec_bc.py`'s recorded null (z = +0.69,
p = 0.49 over 4,800 worlds) was measured inside an executor that disobeys 15%
of direct orders — the ranker was picking better targets and then being
overruled by the zone partition.

**It does not clear the bar yet** (z = 1.95 against the ~2 se rule, McNemar
p = 0.21) and it is one ranker on two blocks. But it is the only learned
component in this project that has ever read positive, and it is now the
strongest open lead. Next: a third block, then re-fit the ranker on candidate
sets generated by the *fixed* `assign` rather than reusing the champ-h8-era
weights.

## Where that leaves the premise

| | verdict |
| --- | --- |
| knob results (6 knobs, 4 executors) | **transfer** — sign 3/3 on every resolved arm, Spearman +0.94 on the fixed executor |
| BC / learned ranker | **did NOT transfer** — −$1,052 on the old executor, +$1,233 on the fixed one |

"Most of last week is invalid" is too strong for the knobs and right for BC.

---

# Part 10 — CORRECTION: a harness bug understated every obedience number

Found by reading the replays, which is what they were for.

## The bug

`exec_commands.run_case`'s wrapper computed, at step T:

```python
case["_kind"] = (case.get("op") or [None, "GOOSE"])[1] if ... else "GOOSE"
```

`op[1]` raises **IndexError for every argument-less op** — `WATER`, `HARVEST`,
`DIG`, `CARE`, `COLLECT_FERTILIZER`, `BUILD_COOP`, `BUILD_PASTURE`, `DROP`.
The wrapper replaces `build_tasks`, which is called from inside `policy.act`,
and `SafeAgent` catches **every** exception and emits all-`PASS`. So on the
commanded turn the agent did nothing at all, and the command was not injected
until T+1. A crashed harness looked exactly like an executor refusing an
order. A second instance (`plan.PLANT` resolving the crop from `op[1]` on
cases mined from `BUY_SEED`) hit 42 more.

Both are fixed; the wrapper now cannot raise, any failure is recorded in
`harness_error` and `run` prints a loud warning instead of a quiet wrong
number. A subset run (`--cmd`, `--per-cmd`, `--failures-only`) now merges
into the agent's results file rather than truncating it.

## Corrected numbers

| | reported | **corrected** |
| --- | --- | --- |
| champ-h8 | 85% | **89%** |
| `h11-uf` | 92% | **95%** |
| `dev-exec` (`kagent/`) | 90% | **95%** |
| `match-all` | 85% | **90%** |
| `a-commit` | 87% | **92%** |
| `h10-cmt` | 83% | **87%** |

| command | champ-h8 | h11-uf | match-all | a-commit | h10-cmt |
| --- | --- | --- | --- | --- | --- |
| `task.WATER` | 73% | **90%** | 73% | 90% | 73% |
| `task.HARVEST` | 83% | **100%** | 83% | 94% | 83% |
| `task.DIG` | 60% | 60% | **70%** | **80%** | 60% |
| `task.BUILD_COOP` | 71% | **100%** | 86% | 100% | 71% |
| `deliver.FERTILIZE` | 82% | **94%** | 82% | 82% | 71% |
| `deliver.FEED` | 67% | **75%** | 67% | 75% | 67% |
| `deliver.PLACE` | 100% | **100%** | 100% | 60% | 80% |
| every `plan.*` | 100% | 100% | 100% | 95–100% | 100% |

h11-uf fixes **15 of champ-h8's 24**, breaks 2, leaving **11**.

## What changes, and what does not

**Changes:**

- **`match-all` is not identical to champ-h8.** 90% vs 89%, and it lifts
  `task.DIG` 60 → 70% and `BUILD_COOP` 71 → 86%. Part 4's "index-order
  claiming is refuted — identical on 213/217" was measured on contaminated
  runs and is **too strong**. Corrected: global matching gives a small real
  improvement, far less than the underfoot fix, so claim order is a *minor*
  contributor, not a non-contributor.
- **`plan.HIRE` and `plan.PLANT` are 100%,** not 75% and 96%. Those failures
  were entirely harness artifacts, so the plan layer now obeys perfectly on
  every intent.
- The remaining-failure list shrinks from 18 to **11**.

**Does not change:** every ladder/money gate (the harness plays no part in
them) — the underfoot fix's +$337 ± 658, `h10-cmt`'s −$2,162, the knob
transfer, the BC flip; the no-op audit; and the whole unprompted rule suite
(`rule_tests.py` injects nothing and never touched this code path), so
79% → 82% capability and 32% → 51% rigid stand.

**Still true:** commitment fixes nothing (`h10-cmt` 87%, below baseline) and
the zone partition was the dominant cause — the underfoot fix is what moved
89% → 95%.

## The four replays that prompted this

| case | what is actually on the tile | verdict |
| --- | --- | --- |
| `WATER.survival#10` x1y5 d27 t22 | **not empty** — WHEAT planted d27, `consecutive_unwatered: 1`, dies that night | we DO water it, that turn. Now passes |
| `BUILD_COOP#02` x4y1 d10 t19 | **not a crop** — `null`. Reference builds at t19, places a goose at t20 | we build it at t22. Now passes |
| `HARVEST.ongoing_before_clip#00` x9y1 d28 t18 | TOMATO, 3 of max 4 units | unit 1 stood on it and went WEST; unit 4 harvested at t22. Now passes |
| `WATER.fertilised_production#03` x1y0 d16 t11 | STRAWBERRY, `fertilized_until_day 18`, produces that night | unit 6 stood on it and went SOUTH. **Still a genuine miss** |

Three of the four were the viewer's off-by-one, not bad tiles. The fourth is
real and it is the expensive kind: a fertilised strawberry watered on its
production night yields 2 instead of 1 (README Harvest Yields), so skipping
it costs a ~$165 unit.

## Indexing, and the replay window

`day` and `hour` are both **0-indexed** — README Observation Format, and the
engine computes `obs0.day = next_step // turns_per_day`,
`obs0.hour = next_step % turns_per_day` (kaggriculture.py:949-950). Hour runs
0–23, so "turn 22" is the 23rd turn of the day. Filenames use `step % 24`, so
they agree with the observation.

The off-by-one you saw is the renderer: **`env.steps[t]`'s observation is the
state AFTER step t's actions**, so a slice beginning at `t` opens on the world
as it stands at the start of `t+1`. `window_for()` now subtracts one to
compensate and renders `LEAD_DAYS = 2` in-game days before the command and
`TAIL_TURNS = 10` after the opportunity closes — a day-28 h22 command opens at
day 26 h22, 61 frames, ~2–4 MB.

---

# Part 11 — why `task.DIG` was stuck at 60%

Three causes, in descending size. One was mine.

## 1. A second scorer off-by-one (1 of 4 cases)

`task.DIG` has `settle_days = 0`, so its window is `[T, end of that in-game
day]`. The mined cases cluster late — `#05` is at **day 15 hour 23**, a window
of exactly **one turn**. And on that turn the executor **did** DIG:

```
#05 x4y8 d15 t23   window 1 turn
  ops on the tile: (383, unit 2, 'DIG')     <- the commanded turn
  weed cleared at step 384
```

`done` was only evaluated *inside* `[T, settle_end]`, but an action taken on
step S is only visible in step S+1's observation. Every command obeyed on the
last turn of its window scored as a failure. Fixed: `done` is now checked to
`settle_end + 1` while injection still stops at `settle_end`.

That alone takes `task.DIG` 60% → **70%**.

## 2. Claim-order on the tile you are standing on (1 more case)

The underfoot branch shipped in Part 8 sits *inside* pass 1's per-unit loop,
and pass 1 runs `for idx in sorted(free)`. `claimed` is set when a unit takes
the step **toward** a task, not when it arrives. So a low-index unit far away
reaches the weed first through its own-zone `pick`, reserves it, and walks —
and the unit already standing on the tile is refused and sent elsewhere:

```
#00 x4y0 d4 t20    unit 4 stood on the weed at step 116 and went SOUTH
                   weed finally dug at step 129 (+14 turns, next day)
#06 x2y1 d21 t22   units 8 and 9 stood on it at 526, went EAST / SOUTH
                   dug at 541 (+16 turns)
```

`--hoist` runs the underfoot check as its own pass **before** any zone pick,
so a unit on a task always claims it first. Carriers holding an animal are
skipped so `deliver.PLACE` keeps its priority. `task.DIG` 70% → **80%**.

## 3. Weeds are P_FILLER by design (the unprompted residue)

`build_tasks` sets `weed_pri = P_GROW if len(empties) - cursor < 6 else
P_FILLER`. With six or more empty tiles a weed is *filler*, and the underfoot
branch takes `pri < P_FILLER` only — deliberately, because `v-tile` grabbed
filler underfoot and gated −$4,907. That is why **unprompted** DIG capability
is 70% while commanded is 80%: the executor can dig on demand, and chooses
not to when there is spare ground. Whether that choice is right is an
economics question, not an execution one.

The two still-failing cases (`#00` d04 t20, `#08` d22 t11) dig the weed 14 and
37 turns later respectively — late, not never.

## Final obedience, after both scorer fixes

| command | champ-h8 | h11-uf | **h12-ufh** | match-all | a-commit |
| --- | --- | --- | --- | --- | --- |
| `task.WATER` | 73% | 90% | 90% | 73% | 90% |
| `task.HARVEST` | 83% | 100% | 100% | 83% | 100% |
| **`task.DIG`** | **60%** | **70%** | **80%** | 70% | 90% |
| `task.BUILD_COOP` | 71% | **100%** | 86% | 86% | 100% |
| `deliver.FERTILIZE` | 82% | 94% | **100%** | 82% | 88% |
| `deliver.FEED` | 67% | 75% | 75% | 67% | 75% |
| `deliver.PLACE` | 100% | 100% | **100%** | 100% | 60% |
| every `plan.*` | 100% | 100% | 100% | 100% | 95–100% |
| **ALL** | **89%** | **95%** | **96%** | 90% | 94% |

## Gate and landing

| arm | block 0 | block 1 | **pooled (692 worlds)** | wins |
| --- | --- | --- | --- | --- |
| `h11-uf` | +$454 ± 874 | +$219 ± 984 | +$336 ± 658 (z 0.51) | 44 vs 46 |
| `h12-ufh` | +$1,281 ± 903 | +$480 ± 952 | **+$880 ± 656 (z 1.34)** | 45 vs 47 |

Both positive in both blocks, negative in neither; neither clears 2 se.
`h12-ufh` is nominally better on money and strictly better on obedience
(96% vs 95%) — but it costs `task.BUILD_COOP` 100% → 86%, and the money
difference between the two arms is well inside noise. The hoist is landed in
`kagent/scheduler.py` because it is the principled fix for the mechanism the
DIG replays exposed, not because the money resolved it.

`kagent/` now reads: **obedience 96%**, `task.DIG` 80%, `deliver.FERTILIZE`
100%, `pytest tests/` 25 passed / 2 xfailed. The `dev-exec` bundle is rebuilt.

## Three scorer bugs in one instrument — what changed to stop a fourth

1. `op[1]` IndexError crashed `build_tasks` into SafeAgent's all-PASS.
2. A subset run (`--cmd`) overwrote the agent's full results file.
3. `done` was never evaluated on the step that reveals the last turn's action.

All three made the executor look worse than it is, and none of them announced
itself. The wrapper can no longer raise, every failure is recorded in
`harness_error` and printed as a loud warning, and subset runs merge. The
general lesson: a hook injected into the agent's own call path is inside
`SafeAgent`'s blast radius, so it must be incapable of raising — otherwise
your bug is silently rendered as a plausible result.

---

# Part 12 — the dev-tree drift was ONE line, and melon fertilisation is refuted

## Gating the drift: 100% of it is `weights.py`

`dev-exec` read −$777 against champ-h8 while the same executor fix on the
frozen champion read +$454. `tools/make_swap_variant.py` swaps exactly one
`kagent/` module into the gated bundle so the difference decomposes. All arms
gated against `h12-ufh`, 346 held-out worlds:

| arm | dev diff lines | money | win | McNemar |
| --- | --- | --- | --- | --- |
| `h13-weights` | 2 | **−$538 ± 433** | 0.303 → **0.277** | 14 b-only / 5 c-only, **p = 0.064** |
| `h13-planner` | 82 | **+$0**, 0 discordant | — | 1.00 |
| `h13-econ` | 14 | **+$0**, 0 discordant | — | 1.00 |
| `h13-features` | 12 | **+$0**, 0 discordant | — | 1.00 |
| `h13-market` | 2 | **+$0**, 0 discordant | — | 1.00 |
| `h13-rest` (policy, nnpolicy, mpcpolicy) | 8 | **+$0**, 0 discordant | — | 1.00 |
| `h13-all` | 120 | **−$538 ± 433** | 0.277 | identical to `h13-weights` |

`h13-all` reproducing `h13-weights` exactly is the check that the
decomposition is complete: **every other module is inert**, including the 82
diff lines in `planner.py`, which are comments and unreachable branches.

The two lines that matter:

```diff
-BANDIT_RULES = {... premium_scale 0.35, carrot_day 25,
-                    cows 6 @ MILK<=0, sheep 10 @ d12 @ WOOL>=1.0}
+BANDIT_RULES = None
```

`premium_scale` and `carrot_day` survive as `KNOB_DEFAULTS`, so what was
actually lost is the two **conditional** rules — champ-h6's cows 6 when no
milk consumer is open (+$1,989 ± 551 when found) and champ-h7's sheep 10 at
day 12 on wool ≥ base (+$1,034 ± 151). They fire in roughly 40% of worlds,
which is why the pooled cost is only −$538, and why the damage shows up on
**wins**: 9 net losses in 346.

Restored in `kagent/weights.py` from the frozen champion's own bundle.
**`dev-exec` now gates +$0 with 0 discordant worlds against `h12-ufh`** — the
dev tree and the gated artifact are behaviourally identical, and
`tools/submit.sh` is safe to build from again.

### A stale cache, caught in the act

The first re-gate after restoring the rules read −$2,058 ± 564. That was
`ladder_gate`'s cache: its key is `(agent, n, seed)` and encodes neither the
corpus nor the **agent's bytes**, so a bundle rebuilt under the same name
silently reuses the previous agent's numbers. This is the same failure as
`kaggriculture-gate-cache-stale`, in a second guise. The cache now stamps a
SHA-1 of `main.py` and recomputes when it does not match.

## Melon fertilisation: +50pp capability, −$14,897

`FERTILIZE.one_time` was the suite's worst rule at 20% / 0% on every executor.
Reading the ten mined cases: **5 are MELON at ages 6–8, and `want_fert` has
no melon clause at all**; the other 5 are wheat/carrot where the
`fert_price < 40/50` guard blocks it.

Adding `(crop == "MELON" and 6 <= age <= 8 and fert_price < 100)`:

| | capability | rigid | ladder vs h12-ufh | win |
| --- | --- | --- | --- | --- |
| `h12-ufh` | 20% | 0% | — | 0.303 |
| `p1-mel60` | 20% | 0% | **+$0**, 0 discordant | 0.303 |
| `p1-mel100` | **70%** | **30%** | **−$14,897 ± 1,939 (z −7.68)** | **0.220**, McNemar p < 0.0001 |

**The largest single regression measured in this project.** `p1-mel60` is a
no-op, so fertilizer never drops below $60 during a melon's window — the
clause only ever fires at $60–100, i.e. near base, on the opening's 12–16
melon tiles across three days. That is up to ~48 FERTILIZE actions, each
burning a ~$100 asset we otherwise sell, each a `P_TEND` action taken from
strawberry fertilisation (2 units × ~$165), and each inflating
`plan.fert_reserve` so the stock is withheld from the market as well.

And the payoff was never large: README gives melon `first_yield_day` 10, and
the engine refuses HARVEST before it (kaggriculture.py:453), so reaching the
cap at age 8 instead of 10 frees nothing. Fertiliser buys only *insurance*
against a missed watering, and in-window watering is already 92%.

**This is the fifth instance of the recurring law, and the sharpest yet.** The
2000+ seats really do fertilise melons at ages 6–8 — that is where the cases
came from — and copying it costs $15k a game. A capability score of 20% says
the executor *cannot* do a thing; it never said the thing was worth doing.
Part 3's caveat, now with a number on it.

## Where the planner work should start instead

Not from "what does the corpus do that we don't" — that is exactly what just
lost $15k. From the knobs the transfer gate put back in scope: the arms whose
champ-h8 effect was inside ~2 se, which are the only ones the old executor
could have been carrying (`straw_target 40`, the `wheat_engine` variants,
`land2_margin`, `carrot_cap`, `straw_early`). Those re-measure against
`h12-ufh` with the tooling that now exists, and the melon result says to gate
each one before believing any of it.

---

# Part 13 — the old planner weights on the fixed executor

The BC ranker flipped from −$1,052 to +$1,233 when moved onto `h12-ufh`, so
the same question is open for the trained **planner** heads: every one of
`rl/runs/g10..g14e/best_weights.json` was scored against the executor that
disobeys ~11% of direct orders, and abandoned.

`rl/export.py` cannot answer it — it writes `kagent/weights.py` and rebuilds
from the dev tree rather than the gated bundle. `tools/make_neural_variant.py`
patches the frozen bundle's own `weights` module instead.

**Flag combination matters.** `policy.py` dispatches
`BANDIT_RULES → CHAMPION_PLAN → _LAYERS`, so a neural arm only runs when
`BANDIT_RULES` is None **and** `CHAMPION_PLAN` is False. Because the rules
have to be off, the honest baseline is `h13-weights` (same bundle, rules off,
no net) — charging the arm the −$538 for losing the conditional knobs would
be measuring the wrong thing.

| arm | head | money vs `h13-weights` | z | win | discordant |
| --- | --- | --- | --- | --- | --- |
| `n1-g14e` | residual offsets over the champion plan | **+$4 ± 6** | 0.66 | 0.277 | **0 / 0** |
| `n1-g13` | full neural planner (the rl-v9 lineage) | **−$22,887 ± 1,105** | **−20.71** | **0.176** | 38 / 3, p < 0.0001 |

## The residual head is inert, not wrong

**Zero discordant worlds out of 346, and ±$4.** On a single full game it
differed from the champion plan by **$3**. That is not "it does not help on
this executor" — it is **the head emits no offsets at all**. It reproduces
`docs/VERSIONS.md`'s own account of gen-14: with the entropy bonus the policy
wandered and returns collapsed; without it the BC prior was too peaked
(P(zero offset) = 0.996) to explore; after softening it learned mostly to
*stop* deviating, and its argmax played identically to champ-h3.

So the executor was never what held it back, and fixing the executor cannot
revive it. The weights encode "do exactly what the champion plan says".

## The full planner is far worse, and worse than it used to be

`n1-g13` loses **$22,887 a game** and takes the win rate from 0.277 to 0.176.
The rl-v9/g10–g13 lineage was recorded at 0.83 / 0.08 / 0 / 0 / 0 / 0 across
the rating bins, i.e. already beaten 20–0 by champ-h1; this simply prices that
against the current field.

## What this does and does not say

It does **not** contradict Part 9. The BC ranker replaces `pick`'s ordering
*inside* the executor, so an executor that then disobeys destroys the signal —
that coupling is real and it flipped. A planner head sits *above* the
executor and emits quotas; its failure mode here is that it learned the
identity function. **Different layer, different failure, and only the
executor-coupled one transfers.**

Practical consequence: there is no trained planner artifact worth reviving.
Anything on the planner side has to be re-fitted, not re-scored — and the
melon result in Part 12 says to gate each candidate rather than trust the
corpus.

---

# Part 14 — the risk analysis was already applied; what it sat on was never shipped

## The risk half is settled, and null

`tools/fit_risk.py`, `tools/volatility.py` and `tools/make_risk_variant.py`
already exist, and the risk analysis was applied — to `_rent_pick`, the
tile-allocation rule, in three modes (computed variance from the engine,
variance fitted from 51,122 corpus samples, and opponent-holding as a
contest proxy). Against champ-h8 at 270 worlds:

| arm | vs champ-h8 | z | wins |
| --- | --- | --- | --- |
| `k-comp5` computed sd, λ=0.5 | +$4,017 ± 1,040 | +3.86 | 21 c / 12 b |
| `k-lear5` learnt sd, λ=0.5 | +$3,163 ± 1,012 | +3.13 | 22 c / 12 b |
| `k-diff5` opponent-diff | +$362 ± 1,136 | +0.32 | 16 c / 17 b |

Those look like risk wins and are not. Every `k-*` arm is built on
`r-mrg100`, which already carries `_rent_pick`; champ-h8 does not. The
numbers are **rent + risk** against a base with neither.

`docs/metric_harvest_spec.md` already separated them: **`k-comp5 − r-mrg100`
is z = 0.54 on 1,131 worlds.** Risk-adjusting the rent rule does nothing, at
a sample size four times what anything else here has used. That matches
`make_risk_variant`'s own note that the fitted variance model is weak (best
R² 0.135, negative for TOMATO and WOOL) — *risk level is knowable, risky
states are not*.

**So there is nothing to re-run on the risk side.** What the analysis sat on
is the interesting part.

## The rent rule: +$4,336 ± 633 on the fixed executor, and it wins games

`_rent_pick` prices every plantable crop at **its own harvest-day price with
self-impact** and takes the best $/tile-day — the forward market model,
recorded as this project's first fitted-model win (+$3,185 ± 768, replicated)
and as "+$2,605 ± 500 … the one thing that paid". It is in `r-mrg100`. It is
**not in champ-h8**, and was never shipped.

`r-mrg100` differs from champ-h8 in `planner` alone (127 lines), so it ports
cleanly onto the fixed executor: `x-rentfx` = `h12-ufh` + that planner.

| block | money vs `h12-ufh` | z | win | McNemar |
| --- | --- | --- | --- | --- |
| seed 0 | **+$4,208 ± 908** | **4.63** | 0.303 → **0.350** | 34 c / 18 b, p = 0.037 |
| seed 1 | **+$4,464 ± 883** | **5.05** | 0.301 → **0.344** | 32 c / 17 b, p = 0.044 |
| **pooled, 692 worlds** | **+$4,336 ± 633** | **+6.85** | **0.302 → 0.347** | **66 c / 35 b, p = 0.0027** |

**This is the only result in this whole audit that clears the bar on the
metric the rating actually uses.** Everything else was null-positive at best:
the underfoot fix +$336 (z 0.51), the hoist +$880 (z 1.34), BC +$1,233
(z 1.95). This is +$4,336 at z 6.85 with **+4.5pp of win rate**, replicated
on two independent blocks.

Landed in `kagent/planner.py` (donor source plus the intra-package import
header the bundler strips, with the Part-1 pocket fix re-applied).
**Verified: `dev-exec` now gates +$0 with 0 discordant worlds against
`x-rentfx`** — the dev tree reproduces the gated artifact exactly.

## The pattern, now three for three

`docs/board_state_plan.md` had already spotted it:

> Both replicated wins in this project — the hiring rollout (+$3,287 ± 1,528)
> and the rent rule (+$2,605 ± 500) — came from replacing a **constant** with
> a state-dependent rule computed by **arithmetic**, not from a fitted model.
> Every fitted model we have gated … has failed to transfer.

This session adds the third confirmation from the other direction: the
residual planner head is inert (+$4 ± 6, 0 discordant), the full neural
planner is −$22,887, melon fertilisation copied from the corpus is −$14,897 —
and the arithmetic rule is +$4,336.

**The remaining planner work is the other unshipped arithmetic rule**: the
per-day hiring schedule from `tools/rollout_hires.py` (+$3,287 ± 1,528
out-of-sample), which replaces `CHAMP_HIRES`' 30 hardcoded constants. Same
shape, same evidence class, same porting job — and it has never been on the
fixed executor either.

---

# Part 15 — the hiring rollout: money clears, wins do not

`tools/rollout_hires.py` replaces `CHAMP_HIRES`' 30 constants with a per-day
schedule found by one-step policy improvement. Its previous +$3,287 ± 1,528
was measured on champ-h8; the output was never saved and never shipped, and a
schedule is only meaningful for the stack it was fitted on — the rent planner
changes what a marginal hand is worth per day. So it was re-rolled on
`x-rentfx` (`--base` is now a flag, so a rollout can never again be silently
fitted to a superseded agent).

**Rollout, 8 evaluation worlds, 30 min:** margin −$26,984 → −$22,036
(**+$4,948**). It moved six days:

| day | 2 | 3 | 7 | 26 | 27 | 28 |
| --- | --- | --- | --- | --- | --- | --- |
| base | 4 | 5 | 9 | 10 | 10 | 10 |
| rollout | **2** | **7** | **8** | **11** | **11** | **11** |

Days 10–25 are untouched — the mid-season plateau at 11 hands is already
optimal, which is a check on the method rather than a disappointment.

**Gate on fresh held-out worlds, vs `x-rentfx`:**

| block | money | z | win | McNemar |
| --- | --- | --- | --- | --- |
| seed 0 | +$939 ± 797 | 1.18 | 0.350 → 0.361 | 28 c / 24 b, p 0.68 |
| seed 1 | +$930 ± 768 | 1.21 | 0.344 → 0.350 | 23 c / 21 b, p 0.88 |
| seed 2 | +$1,776 ± 718 | 2.47 | 0.335 → 0.312 | 25 c / 33 b, p 0.36 |
| **pooled, 1,038 worlds** | **+$1,215 ± 440** | **+2.76** | **0.343 → 0.341** | **76 c / 78 b, p 0.94** |

**Money clears the bar and wins are dead flat.** Positive in all three blocks,
z = 2.76 — and 76 candidate-only wins against 78 base-only, i.e. nothing.

**Verdict: hold, do not ship.** The rating is win-based. A change worth
$1,215 of margin that flips no games does not move it, and this project has
been caught by exactly that asymmetry before (`match-all` gated +$1,381 and
took head-to-head wins from 5/144 to 4/144). Compare the rent rule on the
same instrument: +$4,336 **and** +4.5pp of win rate at McNemar p = 0.0027.
That is what shipping evidence looks like here.

**The shrink is worth recording**: +$4,948 in-sample → +$1,215 out-of-sample,
a **75% loss**. `rollout_hires`' own docstring predicted it — the
cost-improvement property only guarantees the evaluation worlds. The original
+$3,287 ± 1,528 was very likely the same shape, and the honest reading now is
that the per-day hiring schedule is worth about a quarter of what the
in-sample number says.

`x-hirefx` is built and gated; `kagent/` is unchanged.

---

# Part 16 — melon risk: the diagnosis was right, the fix is refuted

## The existing risk model ranks melon exactly backwards

`k-comp5`'s `_RISK_SD` comes from `tools/volatility.py`, which computes
variance from the **shop draw**. Against the actual glut curve it is
inverted:

| crop | shops | base | P(I0+T) | glut drop | `k-comp5` sd |
| --- | --- | --- | --- | --- | --- |
| **MELON** | **0** | 250 | **$1** | **100%** | **0.00** ← ranked least risky |
| STRAWBERRY | 4 | 120 | $1 | 99% | 0.16 |
| CARROT | 3 | 35 | $10 | 71% | 1.07 |
| TOMATO | 2 | 60 | $24 | 60% | **1.60** ← ranked most risky |
| WHEAT | 5 | 25 | $20 | 20% | 0.09 |

Melon is the one plantable crop in **no shop at all** — README Town Buildings
lists bakery, pizza, brunch, yarn, ice-cream, pet cafe, smoothie and farmers
market, and melon is in none of them. Its only demand is the town centre's
flat 1 unit a day, which never ramps.

Zero shop demand means zero shop-draw variance, which is why a model fitted
on the shop draw calls melon risk-free. **Zero shop demand is exactly what
makes it dangerous**: nothing can ever add melon demand, and it carries the
steepest glut curve in the game (`sq` at `above_target` 3.60), hitting the $1
floor inside one field's production.

So the measured critique is correct and stronger than "one shop consumes it":
it is **no** shop, and the risk model had the sign of melon's risk wrong.

## Replacing shop-draw variance with the glut coefficient loses money

`tools/make_glut_variant.py` swaps `_RISK_SD` for the structural glut
coefficient — the fractional price drop after one 5×5 field's worth of
supply, computed from `MARKET_PARAMS`, no fitting — and drops the
`0.35 + 0.65 × shops_left` decay, since glut risk does not resolve as draws
land and for melon cannot.

| arm | block 0 | block 1 | **pooled, 692 worlds** | win |
| --- | --- | --- | --- | --- |
| `x-glut25` (λ 0.25) | −$393 ± 644 | −$766 ± 552 | **−$580 ± 424** (z −1.37) | 0.350 / 0.358 |
| `x-glut50` (λ 0.50) | −$1,033 ± 765 | −$1,426 ± 704 | **−$1,230 ± 520** (z −2.37) | 0.355 / 0.335 |

**Monotone in λ** — 0 → $0, 0.25 → −$580, 0.50 → −$1,230 — which is the
signature of a real effect rather than noise, and the effect is harmful.

## Why: the rent rule already charges for glut

`_rent_pick` prices the whole block against **the inventory its own units
create**:

```python
rev  = econ.sell_revenue(crop, int(round(inv[crop][k])), total)
rent = rev / float(free * occupies)
```

`sell_revenue` walks the price down unit by unit along `above_func`, so a
melon block is *already* valued at the collapsed price it will actually
realise. Melon's 100% glut coefficient is not missing information — it is
the same information, and the penalty charges it twice. Halving the rent of
the $250-base crop the champion opening is built around is simply a tax on
the best tile in the game.

That also explains the original result this came from: `k-comp5 − r-mrg100`
was z = 0.54 on 1,131 worlds. Any risk term bolted onto a rent that already
prices self-impact is near-redundant; the only question is how much damage
the redundancy does, and the answer scales with λ.

**Sixth instance of the recurring law.** The observation is correct — the sd
table really is inverted, and melon really is the highest-glut-risk crop.
The intervention is refuted. What would *not* be redundant is the half the
rent rule cannot see: the **opponent's** future supply into the same pool.
That is `k-diff`, and it gated +$362 ± 1,136 (z 0.32).

---

# Part 17 — the shops that have not opened yet: +$676 ± 249, SHIPPED

Part 16 refuted a risk term because `_rent_pick` already priced it. Re-reading
README Town Buildings against `_rent_inventory` finds the half it genuinely
does not price — and it is the demand side, not the supply side.

## The gap

```python
drain = econ.town_drain_per_day(view.unlocked_shops)   # TODAY's shops
for k in range(1, horizon + 1):
    cur[p] += mine[k] + opp[k] - drain[p]              # held CONSTANT
```

README: a shop unlocks every `townShopUnlockInterval` (3) days, drawn
**uniformly with replacement** from the eight types, capped at eight
instances; each instance consumes one of every product it demands every
`townShopSellInterval` (4) turns — 6 a day, doubled for a single-product
shop. With `_RENT_HORIZON = 14`, the forecast misses about four future draws
entirely.

It misses them **unevenly**, which is the whole point. Expected new demand
per future instance, straight off the shop table:

| product | shop types demanding it | E[units/day] per future instance |
| --- | --- | --- |
| WHEAT | 5 of 8 | **3.75** |
| STRAWBERRY | 4 | 3.00 |
| CARROT | 2 (pet cafe ×2) | 2.25 |
| MILK | 3 | 2.25 |
| TOMATO / EGG / WOOL | 2 / 2 / 1 (yarn ×2) | 1.50 |
| **MELON** | **0** | **0.00** |
| FERTILIZER | 0 | 0.00 |

So melon is the one crop whose price can only ever be pushed down, and the
forecast was treating its demand exactly like wheat's.

## Result

`tools/make_futdemand_variant.py`. λ scales the expectation; λ = 0 is
bit-identical to the base by construction, and was verified so ($100,099 on
both).

| arm | block 0 | block 1 | block 2 | **pooled** | wins |
| --- | --- | --- | --- | --- | --- |
| `x-fut05` (λ 0.5) | +$426 ± 363 | +$239 ± 273 | — | +$332 ± 227 (z 1.46) | 14 c / 6 b |
| **`x-fut10` (λ 1.0)** | +$909 ± 446 | +$227 ± 369 | +$893 ± 470 | **+$676 ± 249 (z 2.72)** | **30 c / 17 b**, p 0.079 |

Positive in all three blocks, **monotone upward in λ** — the mirror image of
the glut arms, which were monotone *downward*. Note the standard error:
$249 against ~$650 for the other planner arms, because this is a narrow
forecast correction — only 47 of 1,038 worlds are discordant, and the ones
that move, move the right way.

**Shipped** to `kagent/planner.py`. Verified: `dev-exec` gates **+$0 with 0
discordant worlds** against `x-fut10`, win rate 0.367.

## Why this one worked and the glut penalty did not

Both start from the same correct observation — melon's demand is structurally
the thinnest in the game. They differ in where they put it:

| | where it acts | already priced? |
| --- | --- | --- |
| `x-glut50` | a penalty **on the rent**, after `sell_revenue` | **yes** — `sell_revenue` already walks the price down along `above_func`. Double-counted. −$1,230 |
| `x-fut10` | the **inventory path** the rent is priced against | **no** — `_rent_inventory` never modelled future unlocks. +$676 |

That is the rule worth keeping from this pair: *before adding a term, find
where in the objective the quantity already enters.* The same fact about
melon produced −$1,230 in one place and +$676 in the other.

## Running total on the shipped stack

| change | vs its base | wins |
| --- | --- | --- |
| underfoot + hoist (executor) | +$880 ± 656 (z 1.34) | flat |
| rent planner | **+$4,336 ± 633 (z 6.85)** | **0.302 → 0.347** |
| future shop demand | **+$676 ± 249 (z 2.72)** | 0.350 → 0.367 |
| hiring rollout | +$1,215 ± 440 (z 2.76) | flat — **held** |
