# Agent Version Documentation

Reference documentation for every Kaggriculture agent version: strategy, architecture,
training configuration, benchmarks, and the lessons each one taught. Quick history:
[CHANGELOG.md](../CHANGELOG.md). Research → design mapping: [rl/RESEARCH.md](../rl/RESEARCH.md).

All money figures are final bank (the game's score: only sold product counts). Arena numbers
are fresh-seed games, both player positions. Frozen bundles live in `arena/agents/<name>/main.py`.

---

## champ-h10 — three measured executor rules (2026-09-26, submitted 12:45, ref 56579530, sha256 63a727ee)

The dev tree + champ-h8's four knob rules + the three executor rules that
cleared the paired instrument, carried in the bundle by `EXEC_RULES`
(`rl/export.py --clear --champion --bandit-rules tools/rules/rules_h8.json
--exec lastwater,h23,harvest_visit --freeze champ-h10`). The hook is inert in
dev mode, so `kagent/` behaviour and every running eval arm are unchanged.

| on 995 band-stratified held-out games | wins | win rate |
|---|---|---|
| champ-h10 (= the `ship2_hv` arm) | 352 | **0.354** |
| champ-h9 / dev tree + h8 rules | 311 | 0.313 |
| champ-h8 (frozen) | 229 | 0.230 |

+4.1 points over champ-h9 (z +3.19, p 0.0019, better in 5 of 6 bands), our
bank +$1,649 and the recorded opponent's −$3,105. The three rules:
`lastwater` (water a one-time crop on the last bonus-window day before
harvesting it — the engine adds the yield unit in the WATER op, so we were
taking wheat at 3 of 4 units and carrot at 2 of 3), `h23` (no moves on the
day's last turn, since units respawn at the shed), and `harvest_visit`
(harvest an ongoing crop whenever it holds a unit, at tend priority).
Refuted and left off: the per-quadrant priority queue (−11.6), one-shed
provisioning (−8.9), hand-feed + wheat layout (−10.2), all six stacked
(−14.5); `docs/results/draw995_batch{2,3}.log`.

Verification before submitting: the frozen bundle reproduced the `ship2_hv`
dev arm **game for game** on three spot checks across bands 1000–2500;
141 tests pass; isolation smoke from a foreign cwd OK; sha matches main.py.

**Cost of this submission, accepted deliberately:** only the two most recent
submissions stay active, so champ-h10 evicted `tape-sel` (1292.1) from the
active pair. The active pair is now champ-h10 and champ-h9 (750.2), and our
best active agent drops ~540 points until the tape is resubmitted.

## champ-h9 — the dev tree, bundled (2026-09-25, submitted 20:27, ref 56559654, sha256 e71e58cd)

Same rules.json as champ-h8. Everything else is today's `kagent/`: the planner
and executor work since h8 (rent rule on the fixed executor, future-shop
demand, the underfoot fix, global matching, the land cap plumbing which is
inert without `KAGG_MAX_QUADRANTS`, the forward_policy width guard). Built
with `rl/export.py --clear --champion --bandit-rules <h8 rules> --freeze
champ-h9`; the flags read back from the bundle are CHAMPION_PLAN, no weights,
MPC off, four knob conditions.

Why it went up: the first evaluation of the shipped bundles and the dev tree
on the same instrument, 785 band-stratified held-out recorded games, paired,
McNemar (`docs/stage_b5_preregistration.md` §11–13):

| our side | wins / 785 | win rate | money |
|---|---|---|---|
| dev tree + h8 knob rules (= champ-h9) | 190 | 0.242 | $84,125 |
| dev tree, plain champion plan | 182 | 0.232 | $83,231 |
| frozen champ-h8 | 144 | 0.183 | $80,802 |
| frozen champ-h1 | 46 | 0.059 | $72,137 |
| best RL planner (capped control, 3 seeds) | 58–63 | 0.074–0.080 | $64.7k |

dev + knobs vs frozen champ-h8: +0.059, z +3.69, p 0.0003, five of six
bands, bank +$3.3k. The knob layer adds +0.010 over the plain plan (n.s.).

Caveats: the ladder has disagreed with local gates twice (tape-sel3 < tape-sel;
champ-h8 < champ-h7), so watch the rating, not the local number. The
tape-sel lineage (rated 1750) keeps one active slot; this submission evicted
its duplicate.

Reproduction check (landed 21:41, after submission): the frozen champ-h9
bundle played the same 785 games as the dev-tree arm and matched it **game
for game** — 785/785 identical wins, bank and opponent bank, 190 wins,
$84,125. The submitted file is exactly the agent that scored 0.242 locally.
First ladder reading at 21:38, eleven minutes in: 710.3 (an opening number;
the rating moves for a day and its noise is ±45).

**Ladder, settled 26 September 00:29** (readings 704.4, 768.9, 781.5,
802.0, 784.6, then 755.9 for the last four polls, an hour inside the ±45
band): **755.9**. The champ-h8 bundles' last readings before they went
inactive were 703.2 and 787.0, so the +5.9 points measured locally became
nothing on the ladder — the third time a locally better champion-lineage
agent has failed to rate higher (tape-sel3 < tape-sel, champ-h8 < champ-h7,
now champ-h9 = champ-h8). The active tape-sel reads 1318.4, about 560
points above the whole champion lineage. Keep champ-h9 active (it is not
worse) and treat the 785-game instrument as a win-rate gate that the
ladder does not necessarily reflect.

Also this round: `tools/submit.sh`'s self-containment scan now matches
import statements and path access rather than the bare word `kagent`, which
every planner-lineage bundle carries in docstrings (champ-h8 included).

## champ-h8 — feed buffer no longer churned (2026-09-14, submitted)

Same rules.json as champ-h7. One planner change (`kagent/planner.py`, both plan paths):

```python
plan.wheat_reserve = (0 if plan.liquidation and day >= 29
                      else max(my_animals + 2, plan.feed_days * my_animals))
```

The sell path had reserved `my_animals + 2` while `kagent/market.py` bought back up to
`plan.feed_days * animals`, so the difference was sold and re-bought every day. Wheat purchases
per game fall from 459 units ($19,037) to 165 ($6,283).

Gates: 37–13 vs champ-h7 on seeds 21000+ (p = 0.0009); 22–8 on seeds 19100+; vs the public
agent pub-ahmed-v38 our mean rises $73,710 → $79,879 (still 0–24). Paired champion worlds
+$1,037 ± 3,325. Held-out ladder 0.67 / 0.58 / 0.25 / 0.17 / 0.17 / 0.17.

New instruments in this round:
- `tools/fetch_public_agents.py` — pulls competition notebooks that embed a runnable agent and
  installs them under `arena/agents/pub-*` (gitignored; other people's code). This is a much
  harsher benchmark than self-play: the strong ones are replay tapes of 2500+ players.
- The head-to-head ledger script (session scratch `h2h3.py`) wraps the engine's `_commit_unit`,
  `_do_hire` and `_do_buy_land` so every accepted unit is recorded with seat, item and realized
  price. Requested quantities are useless — agents spam oversized SELL orders that the engine
  fills only as far as the shed allows — and the ledger reconciles exactly with final money.

## champ-h7 — price-reactive sheep rule (2026-09-14, submitted)

champ-h6 plus `{"knob": "sheep", "value": 10, "day": 12, "min_price": {"WOOL": 1.0}}`
(`arena/agents/champ-h7/rules.json`). `KnobBanditPolicy` now evaluates rules at decision days
6 / 9 / 12 / 15 on `features.context(view)` (consumers and `price_ratio` = price / base) and
accumulates chosen knobs on `memory.knobs`; the sheep ramp (`2 + (day − 6)`, capped) then runs
to 10 instead of 8 from day 12 (sheep bought on days 12–14 give 4–5 productions each).

Measurement machinery (`rl/bandit.py`, `rl/rollout.py`): conditional arms
`knob:value@COND[@dN]` where COND is `PRODUCT>=k` / `PRODUCT<=k` (day-6 consumers) or
`P_PRODUCT>=r` / `<=r` (price ratio at decision day N); per batch the twins run first and their
recorded contexts determine each world's eligible arms; the deviated game applies its knobs from
day N (`dev["_day"]`). Results (gain vs champ-h5, mean ± s.e., n): sheep 10 @ P_WOOL≥1.0 @d12
**+1,034 ± 151** (714, Kaggle) / +753 ± 224 (321, local); @d15 +532 ± 140 (595, final cond3); premium 0.2 @ P_MILK≤0.6 @d15 +56 ± 12 (483); wheat 40 @
P_WHEAT≥1.1 +125 ± 64 (3,548); cows 10 @ P_MILK≥1.0 @d12 −567 ± 242 (429), @d15 −866 ± 343;
carrot_day 22 @ P_CARROT≥1.0 −986 ± 317 (40); premium 0.7 @ P_STR≥1.0 −1,438 ± 294 (69);
carrot_standing 12 @ P_MILK≤0.6 @d15 −1,099 ± 524.

Gates vs champ-h6: arena 30 games seeds 19000+ 13–13–4; 24 paired champion worlds +$721 ± 416
(fired in 5, +$3,460 there); held-out ladder (19k pool) 0.75 / 0.67 / 0.50 / 0.17 / 0.17 / 0.00.

## champ-h6 — conditional cows rule (2026-09-12, submitted)

champ-h5 plus one `KnobBanditPolicy` condition (`arena/agents/champ-h6/rules.json`):
`{"knob": "cows", "value": 6, "max_consumers": {"MILK": 0}}` — evaluated at day 6 on
`features.context(view)["consumers"]` (shops open by then), it caps the cow ramp at 6 for the
season when nothing in town buys milk. Source: knob-level imitation mining
(`tools/mine_knobs.py`; `docs/knob_mining_2026-09-12.txt`) over the 19,484-session corpus, then
the conditional counterfactual bandit (`rl/bandit.py collect --knobs --arms 'cows:6@MILK<=0,...'
--min-rating 2000`, worlds chosen so the day-6 shops satisfy the condition).

Verification table (gain vs champ-h5, mean ± s.e., n): cows 6 @ MILK≤0 **+1,989 ± 551** (365);
carrot_cap 30 @ CARROT≥1 +35 ± 11 (458); wheat_engine 24 −493 ± 246 (657); straw 27 @ STR≤0
−706 ± 655 (150); cows 10 @ MILK≥2 −1,358 ± 605 (171); straw 38 @ STR≥2 −1,529 ± 645 (106);
sheep 5 @ WOOL≤0 −1,703 ± 724 (137); sheep 11 @ WOOL≥1 −2,567 ± 549 (30); straw_early 12 @
STR≤0 −3,156 ± 1,558 (138).

Gates vs champ-h5: arena 30 games seeds 18000+ 18–10–2 ($79.3k vs $79.2k; equal means because
the rule fires in ~40% of worlds); 24 paired champion worlds +$3,309 ± 1,867 (fired in 10;
+$7,942 there); held-out ladder (new pool) 0.67 / 0.67 / 0.42 / 0.17 / 0.08 / 0.00 vs champ-h5
0.67 / 0.67 / 0.33 / 0.17 / 0.00 / 0.00.

## champ-h5 — premium floors ×0.35 (2026-09-11, submitted)

champ-h4 with one rule changed: `premium_scale = 0.35` (KnobBanditPolicy `conditions` in
`arena/agents/champ-h5/rules.json`; `planner.KNOB_DEFAULTS` carry champ-h4's values so the dev
twin equals the previous champion). Round-2 knob bandit summary (twin = champ-h4, 4,680 paired
samples; gain vs champ-h4, mean ± s.e., n):

| knob | gain | n | | knob | gain | n |
|---|---|---|---|---|---|---|
| premium_scale 0.35 | **+1,391 ± 203** | 215 | | wheat_engine 28 | −591 ± 289 | 230 |
| premium_scale 0.1 | +1,330 ± 153 | 232 | | land2_margin 4000 | −620 ± 396 | 337 |
| premium_scale 0.5 | +844 ± 121 | 541 | | premium_scale 0.85 | −792 ± 388 | 38 |
| premium_scale 0.2 | +617 ± 164 | 248 | | carrot_day 24 | −846 ± 269 | 37 |
| land2_margin 1500 | +466 ± 307 | 293 | | feed_days 3 | −907 ± 759 | 294 |
| straw_early 24 | +457 ± 536 | 281 | | liquidation_day 28 | −942 ± 173 | 30 |
| melon_opening 8 / 16 | +386 / +337 (± ~900) | 281 / 293 | | carrot_day 26 | −1,258 ± 381 | 32 |
| wheat_floor 20 / 30 | 0 / +55 | 279 / 307 | | straw_target 40 | −3,044 ± 1,151 | 56 |
| liquidation_day 26 | +13 ± 64 | 313 | | straw_early 12 | −4,230 ± 1,996 | 31 |
| straw_target 36 | −88 ± 366 (+508 ± 452 with ≥1 consumer) | 289 | | feed_days 1 | −7,911 ± 2,199 | 33 |

New knobs this round (defaults reproduce champ-h4): `melon_opening`, `land2_margin`
(`DayPlan.land_margin` consumed by the market controller), `wheat_floor`, `liquidation_day`,
`carrot_cap`, `feed_days` (`DayPlan.feed_days`). `rl/bandit.py --tag` separates rounds,
`--arms` runs focused follow-ups.

Benchmarks vs champ-h4: arena 30 games seeds 17000+: 20–10, $101,659 vs $100,904; 24 paired
champion worlds +$76 ± 347 (13/24); held-out ladder 1.00 / 0.83 / 0.17 / 0.17 / 0.00 / 0.50.

## champ-h4 — bandit knobs over champ-h3 (2026-09-11, submitted)

**Architecture.** champ-h3's plan and executor, plus `planner.KNOB_DEFAULTS` — season-long
parameters of the champion plan (strawberry target, herd caps incl. geese, wheat engine, standing
carrot/tomato patches, hire delta, premium sell-floor scale, endgame carrot day) overridden per
agent through `memory.knobs`, effective from day 6. `KnobBanditPolicy` (kagent/banditpolicy.py)
scores every (knob, value) arm at day 6 with a fitted gain model and sets the winners; champ-h4
ships two constant rules: `premium_scale = 0.7`, `carrot_day = 25`.

**How the knobs were measured** (`rl/bandit.py collect --knobs`). Per world one twin game (plain
champion plan) and four games each with one knob changed for the season; the label is the paired
margin gain. Worlds: 60% recorded champion sessions on their true seeds, 30% champ-h3/h2 on fresh
seeds, 10% the reactive heuristic. Successive elimination retires arms whose mean + 2 s.e. < 0.
Interim table at 1,200 samples (gain vs champ-h3, mean ± s.e., n):

| knob | gain | n | | knob | gain | n |
|---|---|---|---|---|---|---|
| premium_scale 0.7 | +1,836 ± 380 | 89 | | sheep 10 | −1,754 ± 742 | 55 |
| carrot_day 25 | +1,572 ± 249 | 81 | | premium_scale 1.3 | −2,360 ± 745 | 30 |
| straw_target 40 | +922 ± 879 (+2,274 ± 927 with ≥1 consumer) | 112 | | carrot_day 18 | −2,691 ± 514 | 33 |
| geese 2 | +332 ± 1,037 | 87 | | hires +2 | −4,408 ± 1,384 | 34 |
| wheat_engine 40 | −87 ± 402 | 83 | | straw_target 21 | −5,667 ± 2,565 | 30 |
| cows 10 | −1,432 ± 865 (≈0 with milk consumers) | 92 | | carrot_standing 16 | −7,702 ± 1,978 | 39 |
| cows 6 | −2,136 ± 1,442 | 80 | | hires −2 | −11,746 ± 1,829 | 31 |

**What did not work, and why** (both fully instrumented; see `rl/train.py --residual --paired`,
`rl/bandit.py collect` without `--knobs`): residual PPO over the champion plan (gen-14 a–e) — with
entropy bonus the policy wandered and returns collapsed; without it the BC prior was too peaked
(P(zero offset) 0.996) to explore; after softening it to 0.85 and adding a paired-twin reward it
learned mostly to *stop* deviating (P(zero) rose on 11 of 13 heads) and its argmax played
identically to champ-h3 on 24 paired worlds after 45 iterations. The single-day counterfactual
bandit (4,800 samples) found every one-day deviation neutral-to-harmful within ±$500: standing
quotas refill one-day changes, so they only test timing.

**Benchmarks.** Arena vs champ-h3, 30 games seeds 15000+: 29–1, $90,718 vs $86,112. 24 paired
champion worlds: +$1,478 ± 692, better in 18/24. Held-out ladder 1.00 / 0.83 / 0.17 / 0.17 /
0.00 / 0.50. p99 4.6 ms (day-6 scoring).

**Next.** Fit the context-conditional rules (strawberry target by consumers, cows by milk demand,
geese by egg demand) from the completed knob collection and gate them the same way; extend the
knob set (second-quadrant timing, sheep/cow ratio, wheat sale floor) and re-collect against
champ-h4 as the new twin.

## champ-h3 — sweep executor (2026-09-11, submitted)

Same plan and market controller as champ-h2; one scheduler change. `assign()`'s `pick()` scores
tasks `(urgent, distance, priority)` where `urgent` is 0 only for P_SURVIVAL tasks from
`SWEEP_DEADLINE_HOUR = 16`; cross-zone help (pass 2) is limited to `HELP_RADIUS = 5` steps.
Rationale and measurement in the CHANGELOG. Market lessons recorded here because they bound what
the executor can convert: market inventory has no decay, only shop consumption (`24 /
TOWN_SHOP_SELL_INTERVAL = 6` units/day per consuming shop, 12 for single-product shops, 1/day
from the town centre); contested units are quoted against the pre-commit inventory, so selling
first beats holding; strawberry's curve is steep (T = 100). Two farms with 33 fertilized
strawberry tiles produce ~66 units/day against ~30 of drain even with five consumers.

Benchmarks: arena vs champ-h2 30 games (seeds 13000+) 30–0, $91,039 vs $78,246, max $115,521;
held-out ladder 1.00 / 0.83 / 0.08 / 0.17 / 0.00 / 0.42; paired champion worlds −$1.2k ± 4.5k
(neutral); p99 0.40 ms. Not shipped from this round: paced selling (−$30k on the traced world),
demand-anchored strawberry cap (−$2.7k ± 4.6k).

## champ-h2 — animal chores first (2026-09-10, submitted)

Same playbook and executor as champ-h1; three changes from tracing nightly per-animal
fed/cared state on a champion world (fed 11/15 and cared 9/15 on day 16, 9/16 on day 24):

- `scheduler.P_ANIMAL = 1.5` for FEED and CARE — between harvest and tend. Engine rule: on a
  production night an animal yields `1 + pending_care_bonus` (one banked per fed+cared day,
  cap `max_held`) **only if fed that day**; otherwise the bank is zeroed. So an unfed sheep
  loses 3 wool, not 1.
- Zone round split into passes: (1) own-zone tasks below filler priority (animal deliveries
  first), (2) any task below filler anywhere, (3) filler watering in own zone then anywhere.
  Fixes the workload skew from centre-first pasture placement.
- `_champ_herd_targets`: sheep `2 + (day − 6)` from day 7 (4/6/8 by days 8/10/12);
  `CHAMP_WHEAT_ENGINE = 32`.

Benchmarks: solo on champion worlds $83.9k (h1 $80.4k); wool $32,064 / 162 units (h1 $16,960 /
116); nightly coverage 14–16 of 16; arena vs champ-h1 20 games seeds 9000+: 16–4, $79,320 vs
$76,034, max $115,856; held-out ladder 1.00 / 0.75 / 0.00 / 0.08 / 0.00 / 0.17; moves 147–190
for 84–105 actions per day (cross-zone help costs ~35–55 moves). Open: milk 181 vs 224–265
units, wheat 372 vs 600 units.

## champ-h1 — decoded champion playbook (2026-09-09, submitted id 56128207, rated 690 after 24h)

**Why it exists.** The rating ladder (`rl/session_bench.py ladder --heldout`: our win rate vs
recorded opponents replayed on their true worlds, binned by the opponent's Kaggle rating) read
identically for rl-v9 and for g13, the 200-iteration rating-curriculum run: 0.83 / 0.08 / 0 /
0 / 0 / 0. Opponents rated 1000+ earned $104–145k on their worlds while we earned $51–54k on
the same worlds. Training was not the constraint; the economy the executor could express was.

**Decode.** `tools/decode_sessions.py` replays harvested sessions (exact fidelity) and logs
both sides' per-day state and clipped cash flow. The 1500+/2500+ profile ($117k): 2 cows +
2 sheep + 12 melon + 7 wheat on day 0; days 1–5 live on animal fertilizer ($400–590/day —
every animal yields one unit nightly, fed or not); the day-6 wool pays for the first quadrant
the same day; milk day 8; melons day 10 ($12–15k); second quadrant day 10–11; 33 strawberries
by day 12; 8 cows + 7–9 sheep by day 10–14 bought before demand is revealed; 20–30 standing
wheat tiles replanted every 4 days (600 units, $25k); 55 of ~58 plantable tiles standing days
14–26; 12–19 carrots planted days 22–27. rl-v9 on the same worlds: 1 cow on day 0, no sheep
until a yarn shop appears, land on day 10–11, income under $1k/day on days 11–16 vs their
$4–5k/day, 34 → 23 tiles after day 14.

**Architecture.** `planner.make_champion_plan` replaces the old speculative BC teacher and
runs as the primary policy when the bundle carries `CHAMPION_PLAN = True`
(`rl/export.py --clear --champion --freeze champ-h1`); the reactive planner stays the
SafeAgent fallback. Executor changes, each found by an hour-by-hour trace of the champion plan
on a champion world:

| Symptom (traced) | Fix |
|---|---|
| Cow carried around for 3 days, pasture empty | animal carriers place first, anywhere |
| Herd starved at $3 cash on days 1–4 | fertilizer is cash: no wheat/carrot fertilizing at base price, shed trips at ≤2 units early; feed bought at any price ≤ $120 down to the last dollar |
| Land bought day 9–11 although wool cash arrives day 6 mid-day | land is a standing wish; the market controller buys the turn it is affordable |
| 20–27 empty tiles with seeds in stock, 0–7 plantings/day | watering only where the engine pays (one-shot growth window, fertilized production night, survival); planting tasks on every empty tile with the quota enforced at execution |
| 200 moves for 60 actions a day | zone-local wheat/fertilizer provisioning; survival override pulls only idle units; even zone split (12 hands had 11 zones) |
| Fertilizer sold before FERTILIZE tasks could use it | `plan.fert_reserve` from the task list |
| Herd starving late at $40k cash | two days of feed stock, no "pricey" guard |
| Strawberry seeds beat cows to mid-day cash | animal buys planned as full deficits (market buys animals before seeds) |

**Benchmarks.** Solo on 8 champion worlds: $80.4k (rl-v9 $57k). Arena vs rl-v9, 20 games on
fresh seeds 7000+: **20–0, $100,345 mean / $119,824 max** vs $59,182. Mirror: symmetric
$57–97k. Held-out rating ladder: **1.00 / 0.75 / 0.08 / 0.00 / 0.00 / 0.33**. p99 turn
0.32 ms. Labor on day 18: 109–121 productive actions (champions 113–156), 143–165 moves
(champions 123–160).

**Remaining gaps vs the $117k profile.** Wool $17k vs $35–40k (sheep reach 4 on day 10 vs
their 6; 2 sheep from day 0 give champions 8 productions each), wheat 382 vs 600 units,
carrots $3k vs $5–13k, 5–8 animals unfed at night on days 18–26, milk $22k vs $29k.

**Lessons.** Trace before tuning: every executor defect here was invisible in aggregates and
obvious in a per-unit hourly table. Fertilizer is the bootstrap currency. Watering everything
daily is the single largest labor waste in this engine.

## Scripted lineage (heuristic planner)

The scripted agents share one architecture (`kagent/`): a **day-level planner** (quotas, buys,
hires, land, sell floors), a **turn-level scheduler** (greedy nearest-unit assignment over
priority tasks: survival > harvest > tend > grow > filler), and a **market controller**
(≤10 priority-ordered orders; sells sized by exact marginal revenue from the engine-mirrored
price function). Economic ground truth that drives all of them:

- Only WHEAT and EGG have log glut curves (never crash); melon is the best early $/action
  (~$150) with ~158 units of shared depth; fertilizer is free daily income per animal with a
  monotonically falling price; strawberry/milk/wool are traps *unless* town shops push their
  drain past T, then they hold $200–300.
- Hands cost fib(n) per day: ~14/day optimal, the 15th/16th hire cost $610/$987.
- Day 29 has no end-of-day refresh: everything must be hand-dropped and sold by step 718.

### v1 — first full economy
- **Strategy**: melon rush days 0–2 (≈16–20 tiles) → sell waves from day 10 → convert cash to
  a goose flock (eggs + fertilizer) → wheat engine for feed → day 27–29 liquidation with
  floors decaying to $1 by step 718.
- **Benchmarks**: solo vs starter $36,197 (20 games, 20–0).
- **Lessons**: feed logistics break at scale (one carrier can't feed 20 coops); melon sell
  floors held too high strand capital; the goose pipeline dies silently when the 10-order
  market cap truncates BUY_ANIMAL.
- **Caveat**: bundle predates the isolation fix — never play v1 against itself locally.

### v2 — demand-adaptive livestock, hardened logistics
- **Adds**: cow targets = f(#milk shops), sheep targets = f(#yarn stores), strawberries at 2+
  consuming shops; adaptive milk/wool floors; bounded market-order appender (animals before
  seeds); goose delivery in batches of 3 with working-stock drop exemptions; feed buffer
  buying; fertilizer floor $1 (earliest sale is always best — zero town demand — and a fert
  hoard once filled the shed so the midnight overflow destroyed the eggs); shed pressure
  valve (>80 items → dump biggest stacks); isolated bundler.
- **Benchmarks**: beats v1 20–0 ($26.2k vs $8.7k contested); solo $40,430.
- **Lessons**: the shed cap (100) is a throughput bound — anything that blocks selling
  destroys produce at midnight; mirror matches halve every shared pool.

### v3 — ruminant CARE (final scripted form) · Kaggle 55817540
- **Adds**: cows/sheep fed daily + CARE (each fed+cared day banks +1 product paid on the next
  production — ~$125/action for milk in demand games); geese stay on cheap alternate-day
  feeding (a single unfed day resets the care bank, so goose care requires daily feeding —
  measured net-negative).
- **Benchmarks**: beats v2 17–3 (p=0.0026); solo $44,856–$45,545; tournament sweep of v1/v2.
- **Failed experiments** (all lost 10-seed gates; do not retry): 16 hands/day, GEESE_TARGET 40,
  daily goose feed+care, buying feed wheat at any price, skipping low-price fertilizer collects.
- **Public score**: ~497 (rating still converging when superseded).

---

## Learned lineage (neural day-planner over scripted executors)

Shared architecture: a small MLP (`kagent/nnpolicy.py`, pure-stdlib inference) replaces the
planner's strategic fields **once per game-day**; the proven scripted scheduler/market
controller executes the micro. Masked multi-discrete heads (mask-before-softmax); heads are
**absolute standing targets** (crop tiles, herd sizes) — delta-style heads made BC collapse
to do-nothing plans. SafeAgent degrades neural → heuristic → PASS, so worst case ≈ v3.
Training (`rl/train.py`): behavior cloning from the heuristic, then PPO (GAE, clip 0.2,
annealed portfolio-delta shaping → sparse final-money+win) with parallel torch-free rollout
workers; stabilizers: 3-iteration value-head warmup, LR 1.5e-4, KL early-stop 0.03.

### rl-v1 — generation 1 · Kaggle 55825229
- **Net**: 86 features → 96×96 tanh → 11 heads (48 logits) + value; ~20k params; p99 0.8 ms.
- **Training** (runs r1/r2): BC 120 games from v3, PPO 50 iters × 96 episodes vs
  {v3 50%, starter 15%, snapshots 35%}; reward money/2000 + 8·win; best checkpoint iter 19
  (fixed-seed eval 20/20 vs v3); later iterations oscillated to 0 — best-checkpoint selection
  is essential.
- **Benchmarks**: 50–0 vs v3 across two blocks ($42.7k / $41.1k vs ~$24k); tournament 32/32
  at $45.5k; solo vs starter $54.5k ± 2.5k (very consistent).
- **Weaknesses found in play** (user-observed, root-caused): the land head *overwrites* the
  heuristic land logic with no fallback → never expands when shop demand rises; idle workers
  passed instead of pre-positioning. Beatable by a targeted exploiter (see rl2-it59).
- **Public score**: ~486.

### rl2-it59 — "the exploiter" (league member, never submitted)
- g2b iteration 59, trained 40% against the static rl-v1 bundle. Beats rl-v1 35–15 (p≈0.007)
  but **loses to v3 3–17** — a strategy cycle (rl2-it59 > rl-v1 > v3 > rl2-it59).
- **Lesson**: gating/training on a single incumbent breeds a counter-strategy, not a better
  agent; also, fixed-seed 20-game evals suffer winner's curse (the eval-picked "best"
  checkpoint actually lost its arena gate while this snapshot won — sweep `w{it}.json`
  snapshots in the arena before trusting `best.pt`).

### rl-v2 — generation 2 (g2d best, iter 9) · league member, not submitted
- **Adds over rl-v1**: 133 features — per-product *supply-vs-demand gap* (my + opponent's
  projected 7-day output vs town drain/day from the live shop list), price curvature, land
  economics; 13 heads — hires to 16, premium/staple sell-floor scale heads; neural-path land
  safety nets (crunch fallback + demand-pressure trigger); idle pre-positioning. Profit-first
  reward (money/1000 + 2·win) and profit-based checkpoint selection; in-run league growth;
  patience-based convergence stopping (all first used here).
- **Benchmarks**: beats v3 17–3 ($43.2k vs $26.8k) — cycle vs the heuristic closed; money-ties
  rl-v1 ($41.0k vs $41.5k, 10–20 h2h); solo $48.6k ± 14.7k with max **$78.1k** — the
  demand-adaptive ceiling is real but the variance (over-commitment before shop draws are
  known) kept rl-v1 on Kaggle.
- **Scripted-side negative result**: the land-pressure trigger added to the *heuristic*
  planner regressed it 10–17 (re-created day-0 cash starvation); it lives only in the neural
  plan applier.

### rl-v3 — generation 3 (g3 best, iter 19) · Kaggle 55829852
- **Training**: 4-gate league {v3 heuristic, rl-v1, rl2-it59, rl-v2} — 30% of episodes vs a
  random incumbent bundle, 12% vs in-run league bests, rest heuristic/starter/snapshots;
  selection by mean profit across all gates on rotating eval seeds; converged (patience 5)
  at iteration 44; peak at iteration 19 with eval profit $60.8k and min-win 0.80.
- **Benchmarks** (fresh seeds):
  | Gate | rl-v3 | Opponent | H2H |
  |---|---|---|---|
  | rl-v1 (two blocks) | **$59,337 ± 2,678** (min $53.8k > rl-v1's max) | $40,062 | **50–0** |
  | v3 | **$63,648** | $26,295 | **20–0** |
  | rl-v2 | **$59,224** | $47,670 | 16–4 |
  | Tournament (4 agents) | **$62,650 mean** | — | **24/24** |
  | Mirror | $63.4k / $58.9k | — | clean |
  | Latency | p99 1.2 ms (1s budget) | — | 17/17 tests |
- **Why it wins**: rl-v2's demand-adaptive earnings ceiling with better-than-rl-v1 consistency —
  the league punished both over-commitment (rl-v2 in pool) and conservatism (rl-v1 in pool),
  and profit-first selection stopped optimizing for exploiting any single opponent.

### gen-4 (g4/g4b/g4c) — experiments, not shipped (tag `g4-experiments`)
- **Adds** (all inherited by gen-5): `econ.forecast_drain` demand forecasting (expected drain
  from future random shop unlocks), carrying-cost + terminal unsold-value penalties,
  labor-utilization features, architecture v2 (192→128 GELU+LayerNorm policy trunk with a
  **decoupled** torch-only value net), LR scheduling (linear decay + plateau cooling).
- **Outcome**: every candidate lost its fresh-seed gate to rl-v3 (best 2–18). The stability
  machinery was proven (10 consecutive improving evals, zero collapses — the old
  breakout→collapse signature eliminated) but the stabilized run converged below the champion.
- **Pitfall recorded**: head VALUE semantics are part of trained weights — a "rl-v3 rebuild"
  on updated head values lost 0–10 to the true frozen bundle. Champions ship only from
  `arena/agents/<name>/main.py`.

### rl-v5 — generation 5 (g5 iter-99) · Kaggle 55853641 — CURRENT PROFIT CHAMPION
- **The meta source**: decoded top-performer episode 101761797 (finals $105k/$96k, watched by
  the team): day-0 speculative cows/sheep before any shop exists, strawberries scaled to 30-36
  tiles, 14-15 cows, land by day 6, ~12 hands, investment-first through day ~10, zero geese,
  late carrots into the pet-cafe drain, total liquidation.
- **Training**: BC mixes 50% a `speculative=True` planner variant recreating that meta with
  50% the reactive teacher (validated $40-70k solo standalone); PPO with the 4-gate league,
  softened penalties (CARRY 0.005, UNSOLD 0.2), LR 2.5e-4 hot-then-scheduled; head ceilings
  wheat 40 / strawberry 32 / cow 16 / sheep 14 / carrot 26. Converged at iter 109 (patience 6),
  best at iter 99 — 22 evals, zero collapses.
- **Benchmarks** (fresh seeds):
  | Gate | rl-v5 | Opponent | H2H |
  |---|---|---|---|
  | Solo vs starter | **$85,238 ± 8,451 / max $98,378** | $3,541 | 20–0 |
  | League gates (heuristic, rl-v1, exploiter, rl-v2) | $50-72k per gate | — | 90–100% |
  | rl-v3 head-to-head | $59,258 (max **$97,841**) | $67,692 | 3–17 |
  | Mirror | $55.6k / $53.7k | — | clean |
- **Standing**: profit champion by the project's stated objective; rl-v3 remains the
  head-to-head duelist. Both rate independently on Kaggle. Next frontier: a gen-6 league
  containing both champions.

---

## Reproducing / extending

```bash
# Verify any version
uv run python arena/run_games.py --a rl-v3 --b rl-v1 --n 15 --procs 10

# Next generation (league grows by one each time)
PYTHONPATH=. .venv/bin/python -u rl/train.py --resume rl/runs/g2/bc.pt \
  --iters 90 --episodes 96 --procs 10 --patience 5 \
  --opp-bundles arena/agents/rl-v1/main.py,arena/agents/rl2-it59/main.py,arena/agents/rl-v2/main.py,arena/agents/rl-v3/main.py \
  --run rl/runs/g4

# Export + freeze + submit
PYTHONPATH=. uv run python rl/export.py rl/runs/g4/best_weights.json --freeze rl-v4
uv run kaggle competitions submit kaggriculture -f main.py -m "rl-v4: ..."
```

Operational notes: always run training with `-u` (a buffered log once masqueraded as a hang);
sweep `w{it}.json` snapshots in the arena before trusting `best.pt`; never mirror v1/v2-era
bundles against themselves (pre-isolation `sys.modules` bug).
