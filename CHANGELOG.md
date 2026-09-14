# Changelog

All agent versions for the Kaggle **Kaggriculture** competition, newest first.
Detailed per-version documentation: [docs/VERSIONS.md](docs/VERSIONS.md).
Benchmarks are arena results on fresh seeds, both player positions, unless noted.

## worker routing — 2026-09-14 · not shipped (five negatives, and a methodology correction)

Chased the measured labour gap (our hands 59.4% MOVE / 34.0% productive vs a 2500+ opponent's
42.2% / 46.0%; 1.25 moves per productive action vs 0.76). **Nothing improved against a real
opponent.** Movement was attributed per assignment pass first (`scheduler.MOVE_STATS`, 3 games):

| pass | MOVE | action | move% | share of all movement |
| --- | --- | --- | --- | --- |
| zone (own zone, real work) | 7,260 | 6,388 | 53.2% | 61.2% |
| idlehelp (cross-zone, r≤5) | 1,999 | 486 | 80.4% | 16.9% |
| filler | 1,446 | 372 | 79.5% | 12.2% |
| shuttle (shed trips) | 760 | 206 | 78.7% | 6.4% |
| morning (zone provisioning) | 224 | 1,025 | 17.9% | 1.9% |

The help and filler passes really do run at four moves per action — 29% of all movement for 10%
of the actions — but every attempt to reclaim it lost money:

| change | vs champ-h8 (twin) | paired vs pub-ahmed-v38, same worlds |
| --- | --- | --- |
| filler walk capped at 3 tiles (1 after hour 16) | 19–11 | **−$7,387 ± 3,126** |
| CARE crash-pivot removed (always care) | 11–19 | −$544 ± 1,734 |
| CARE threshold 0.35 → 0.15 of base | 14–16 | +$1,263 ± 1,631 |
| idle hands pick by value per turn, `value/(d+1)` | 5–25 | **−$4,928 ± 2,702** |
| zones split by work (occupied tiles weighted ×2/×3/×5) | 11–19 / 1–29 / 1–29 | not run |
| `MIN_ZONE_TILES` 5 → 8 (9 zones, hands share) | **24–6, p=0.0014, +$4,481** | **−$4,490 ± 3,333** |

Readings:
- **The executor is at a local optimum.** Equal-area contiguous serpentine zones beat every
  relaxation. Work-weighted zones fail because a zone with sparse occupancy becomes
  geographically *larger*, so the hand walks further — the opposite of the intent. Value-per-turn
  ranking fails because the far P_TEND/P_HARVEST action really is worth five turns of walking and
  the nearby filler action really is worth almost nothing. Movement is the price of covering a
  55–75 tile farm with 11 hands, not slack.
- **The CARE gap is a symptom, not a cause.** We CARE 210 times to their 417 only because the
  crash-pivot fires once wool and milk collapse — and at $1–50 a unit the extra product genuinely
  does not repay the turns. Removing the pivot is worth nothing.
- **Methodology: never gate an executor change on the twin arena.** `MIN_ZONE_TILES = 8` beat a
  copy of ourselves 24–6 at p=0.0014 and was $4,490 *worse* against a real agent on the same
  worlds. Two agents sharing one price ladder can trade wins that come entirely from sale timing
  against each other. Executor changes must be gated paired against `arena/agents/pub-*`.

## champ-h8 — 2026-09-14 · SUBMITTED · stop churning the feed buffer

One-line planner fix worth **+$2,376 a game**: the sell path reserved `animals + 2` wheat while
the buy path topped the buffer up to `feed_days * animals`, so every day we sold the gap and
bought it straight back. Over a season that was **459 units bought for $19,037** against 488
sold for $21,290 — a round trip that also walked the wheat price against us twice, since
`BUY_PRODUCT` quotes at inventory−1 and both farms drain the same pool. The reserve is now
`max(animals + 2, feed_days * animals)`; purchases fall to 165 units.

- **Gates**: 37–13 vs champ-h7 on fresh seeds 21000+ (sign test **p = 0.0009**, $82,315 vs
  $79,940); 22–8 on the 19100 seeds; against a 2500+ public agent our mean rises $73,710 →
  $79,879. Paired champion worlds +$1,037 ± 3,325 (not resolvable at n=24); held-out ladder
  0.67 / 0.58 / 0.25 / 0.17 / 0.17 / **0.17** — the 2500+ bin is the first non-zero we have had,
  the middle bins are down within noise.

### How it was found: the public notebooks are replay tapes, and they beat us 0–30

Reading the competition's public code (`tools/fetch_public_agents.py` installs them as arena
opponents) changed the picture. Most highly-voted notebooks are **720-step replay tapes of
2500+ ladder players** plus a thin online wrapper — boatlee's V16-RC5 is a majority vote over
three recorded episodes with a one-turn "sell early" front-run, salemali7's "2900+" is a
byte-identical fork of it, and the yhay81 / tschinkel "routers" pick between 5–13 tapes at
144-turn boundaries using a fitted decision tree on public state. champ-h7 loses **0–30 to every
one of them**, by $19k (boatlee) to $44k (ahmed-v38).

An exact ledger (`_commit_unit` instrumented, so fills not requests) against ahmed-v38 shows the
farms are nearly identical — same 33 strawberries, same ~57 tiles, same 3 quadrants, same ~11
hands — and the whole $38k is revenue conversion and cost:

| gap | us | them |
| --- | --- | --- |
| wheat bought | 459 units / $19,037 | 172 units / $6,028 |
| milk | 131 units / $10,707 | 216 units / $19,002 |
| fertilizer | 201 units / $10,590 | 351 units / $17,504 |
| melon | $159/unit | $240/unit |
| eggs | none | 76 units / $5,189 |
| productive share of worker turns | 34.0% | 46.0% |

### The market is a per-unit lockstep, and the price ladders invalidate part of the playbook

`_process_market` walks both players' orders **one unit at a time**, quoting both the same
pre-commit inventory price, so there is no first-seller premium *within* a unit — but every unit
sold raises inventory and walks the price down. Selling capacity per product (from I0, before
town drain) is wildly uneven:

| product | base | 40 units | 100 units | 300 units |
| --- | --- | --- | --- | --- |
| melon | 250 | $9.8k @ $235 | $21.7k @ $152 | $26.6k @ $1 |
| wool | 200 | $6.8k @ $112 | $8.0k @ $1 | $8.2k @ $1 |
| milk | 160 | $4.8k @ $78 | $6.2k @ $1 | $6.4k @ $1 |
| strawberry | 120 | $3.3k @ $45 | $3.8k @ $1 | $4.0k @ $1 |
| egg | 50 | $1.8k @ $44 | $4.4k @ $42 | $12.6k @ $40 |
| fertilizer | 100 | $3.8k @ $92 | $9.0k @ $80 | $21.0k @ $40 |

Wool and milk — the products our herd targets are built around — are the two that collapse
fastest, and in the traced game both farms had wool at **$1/unit from day 15**. Egg and
fertilizer never collapse, which is why animals are worth more as fertilizer factories than for
their headline product. Wool is consumed *only* by YARN_STORE, so with no yarn store its entire
season capacity is ~$8k no matter how many sheep are bought.

### Negative results from the same round

- **Sell discipline is not the lever.** Sweeping `premium_scale` over 0.35 / 0.6 / 1.0 / 1.4
  against ahmed-v38 (24 games each) gives $73.7k / $72.7k / $72.5k / $67.8k — flat to worse. The
  shipped 0.35 is already right.
- **`wheat_floor` 30 is a no-op**: both farms buy wheat constantly, so its price sits at $40–46
  all game and a $30 floor never binds (results identical to 6 significant figures).
- **Geese still lose**, −$3,845 vs baseline against ahmed-v38, despite eggs having the best
  ladder in the game and a goose laying every day from day 4 for $300. This is an executor
  problem, not an economics one, and it is the clearest open lead.
- **Capping filler-task walking distance** (3 tiles, 1 after hour 16) was 19–11 vs the churn fix
  (p = 0.20) but −$7.4k against ahmed-v38 — reverted. The diagnosis behind it still stands: our
  hands spend **59.4% of turns moving vs their 42.2%**, need 1.25 moves per productive action vs
  their 0.76, and in hours 17–23 we move 65% of the time while they move 40% and simply PASS.
  Fixing that is worth ~700 actions a game and needs a better idea than a distance cap.

## champ-h7 — 2026-09-14 · SUBMITTED · first price-reactive rule (sheep on wool scarcity)

The adaptive layer's first rule that reads the **market** rather than the town: at day 12, if
wool is at or above base price (i.e. the opponent has not flooded it), raise the sheep cap from
8 to 10 for the season.

- **How it was found**: the day-9 farm is not a usable proxy for opponent strength (everyone in
  the top-heavy corpus plays the same opening: 20 vs 19 strawberry tiles for 2500+ vs <1500
  players), so opponent pressure is read where it shows — prices. The bandit gained price
  conditions with a decision day (`rl/bandit.py --arms 'sheep:10@P_WOOL>=1.0@d12'`) and
  two-phase paired batches: the twin runs first and its recorded day-6/9/12/15 contexts decide
  which conditional arms each world is eligible for (the deviated game is identical up to the
  decision day, so no games are wasted).
- **Measurements**: Kaggle run (4,800 samples, twin champ-h5, opponents ≥ 2000) **+$1,034 ±
  151** (n=714); local run +$753 ± 224 (n=321); the day-15 variant +$532 ± 140 (n=595, final). Other
  price arms: wheat engine 40 when wheat ≥ 1.1× base +$125 ± 64 (real, negligible); cows 10 when
  milk ≥ base −$567 ± 242; endgame carrots from day 22 when carrots ≥ base −$986; *holding*
  premium stock when strawberries are scarce (×0.7) −$1,438 — the first-seller rule holds even
  in scarcity. Strawberry/milk-crash reallocations (carrots, wheat, dump) never fired at day 12
  and rarely at day 15: the overnight drain keeps hour-0 prices above 0.7× base, so a crash
  signature needs our own realized sale prices instead (open).
- **Gates vs champ-h6**: arena 13–13–4 (both farms flood wool against a copy of ourselves, so the
  rule rarely fires there); 24 paired champion worlds **+$721 ± 416**, fired in 5 with **+$3,460**
  each; held-out ladder ≥ champ-h6 in every bin (0.75 / 0.67 / 0.50 / 0.17 / 0.17 / 0.00).
- Mid-season herd *caps* are no-ops once the ramp has completed (cows 8 by day 8–10, sheep 8 by
  day 12); only increases and crop/market knobs can act at day 12+.

## top-games knob bandit — 2026-09-12 · not shipped (negative)

Second free-compute job (10.7 h, 5,400 paired samples): the shipped and borderline knobs
re-measured with `--min-rating 2500` — opponents drawn only from 2500+-rated recorded scripts on
the new corpus, twin champ-h5. Nothing helps against the top of the field: premium ×0.2
+$75 ± 15, early strawberries 24 +$67 ± 379, second-quadrant margin $1,500 +$66 ± 233; premium
×0.5 −$36; strawberries 36 −$178, 40 −$740; wheat engine 40 −$739; carrot day 24/26 −$854/−$818
(25 stays optimal); opening melons 16 −$872; sheep 6 −$1,645; cows 10 −$1,873 (−$5,150 with no
milk consumer); geese 2 −$3,484. Against strong opponents the remaining gains are not in plan
constants; the next candidates are opponent-conditioned rules (mining: champions plant fewer
strawberries and far more carrots against 2500+ opponents), which need the opponent's visible
day-6/9 farm as the in-game condition.

## champ-h6 — 2026-09-12 · SUBMITTED · first conditional rule, from champion mining

The adaptive layer's first *conditional* knob, found by imitation mining and verified by the
counterfactual bandit.

- **Mining** (`tools/mine_knobs.py`, 16,738 seats rated 2500+ from the new corpus): champions
  set their herd and crops by the two shops open on day 6 — sheep 5 / 11 / 12 and cows 7 / 9 / 10
  for 0 / 1 / 2 consumers of wool / milk, strawberries 33 / 34 / 38, carrots 10 / 20 / 31 — with
  within-context margin z-scores of 5–9; geese 0 always; more wheat is worse in every context;
  against 2500+ opponents they plant fewer strawberries and far more carrots.
- **Verification** — conditional bandit arms applied only in worlds whose day-6 shops match
  (twin champ-h5, opponents ≥ 2000, 2,212 samples): **cows 6 when no milk consumer is open:
  +$1,989 ± 551** (n=365). The rest of the champion settings do **not** transfer to our
  executor: sheep 11 with a yarn store −$2,567, sheep 5 without −$1,703, strawberries 38 with
  two consumers −$1,529, cows 10 with two milk consumers −$1,358, early strawberries 12 −$3,156,
  wheat engine 24 −$493, carrot cap 30 ≈ 0. Imitation tells us *where* to look; only the paired
  measurement says what works on our farm.
- **Gates vs champ-h5**: arena 18–10–2; 24 paired champion worlds +$3,309 ± 1,867 — the rule
  fired in 10 of them and gained +$7,942 on average there; held-out ladder ≥ champ-h5 in every
  bin. Note the ladder now runs on the new 19k-session held-out pool (much stronger recorded
  opponents), so its absolute numbers reset: champ-h5 reads 0.67 / 0.67 / 0.33 / 0.17 / 0.00 /
  0.00 there, champ-h6 0.67 / 0.67 / 0.42 / 0.17 / 0.08 / 0.00.

## knob bandit round 3 — 2026-09-12 · not shipped (plan constants exhausted)

First job on free Kaggle compute (7.5 h, 5,400 paired samples, twin = champ-h5): the borderline
round-2 arms re-measured — second-quadrant cash margin $1,500 +$375 ± 171, $800 +$150 ± 145;
early strawberries 24 +$259 ± 294, 21 −$50 ± 257; opening melons 16 −$3,132 ± 1,366. Nothing
clears the bar. Three rounds: +$3.5k, +$1.4k, ~$0 — the fixed plan's constants are tuned out.
Next lever is *conditional* (day-6 shop draw) knobs from `tools/mine_knobs.py`, under
verification.

## data + compute — 2026-09-12 · official daily episodes on free Kaggle compute

- **Free-tier compute** (`tools/kaggle_job/`): any of our jobs runs as a private Kaggle script
  kernel (4 vCPU, 12 h). The private GitHub repo ships as a private Kaggle Dataset tarball
  (`push.sh`, versioned per push; `code-only` mode leaves a running kernel alone); the corpus is
  the `kaggriculture-datasets` Dataset; job steps are injected into the script at push time.
  Verified: tests pass on Kaggle, bandit batches run at local per-core speed.
- **Official daily episodes** (`kaggle/kaggriculture-episodes-index` → 44 daily datasets, ~21 GB
  each, 936 GB total). `tools/episodes_pipeline.py` mounts 3 days at a time into its own kernel
  (mounts cost no quota), `tools/extract_episodes.py` reduces each 32 MB replay to a ~20 KB
  session (seed, aligned action scripts, finals, shops, teams, ratings) — 657 replays in 185 s;
  fidelity EXACT — and only the compact output is merged, with a ledger for resumability.
  Raw replays carry no ratings and the community crawl has zero overlap with them, so ratings
  join from the leaderboard by team name.
- **Result**: 28 days (2026-08-15 → 09-11, engine 1.32.7) → **18,801 sessions**, corpus now
  **19,484**. The official datasets are top-heavy — seats by rating bin: <600: 482 · 600–1000:
  205 · 1000–1500: 3,915 · 1500–2000: 5,752 · 2000–2500: 8,940 · **2500+: 16,738** (2,936
  unrated) — i.e. the "train on top games" corpus; held-out (episode % 5 == 0) keeps ~7,900 seats
  across all bins for the ladder. `rl/bandit.py --min-rating` selects training worlds by the
  recorded opponent's rating.

## champ-h5 — 2026-09-11 · SUBMITTED · premium sell floors ×0.35 (knob bandit round 2)

Round 2 of the knob bandit with **champ-h4 as the twin** (its knobs became the plan's defaults;
verified identical play) over a widened set of 21 arms / 11 knobs, 4,680 paired samples including
two focused follow-ups on the premium floor scale.

- **Premium floor scale is monotone below 1**: ×0.5 +$844 ± 121 (n=541), **×0.35 +$1,391 ± 203**
  (n=215), ×0.2 +$617 ± 164, ×0.1 +$1,330 ± 153 — a plateau around +$1.2k; ×0.85 −$792. Across
  rounds: ×1.0 → ×0.7 → ×0.35 is worth ~+$3.3k. In this market the first seller takes the price,
  so premium products (strawberry / milk / wool) should be sold almost as soon as they exist.
- Everything else measured is neutral or worse against champ-h4: carrot day 24 (−$846) and 26
  (−$1,258) bracket the shipped 25; feed stock 1 day −$7.9k, 3 days −$0.9k; liquidation day 26
  +$13, 28 −$942; wheat floor 20/30 ≈ 0; wheat engine 28 −$591; opening melons 8/16 ±$900 (n.s.);
  $1.5k margin before the second quadrant +$466 ± 307 (n.s.); early strawberries 24 +$457 ± 536,
  12 −$4.2k; strawberry target 36 −$88 (the interim +$2.3k "with consumers" signal faded to
  +$508 ± 452). Diminishing returns are visible: round 1 ≈ +$3.5k of knobs, round 2 ≈ +$1.4k.
- **Gates vs champ-h4**: arena **20–10** ($101.7k vs $100.9k); 24 paired champion worlds
  **+$76 ± 347** (no harm against 2500-rated scripts — the gain comes against everyone else, where
  the rating is decided); held-out ladder unchanged 1.00 / 0.83 / 0.17 / 0.17 / 0.00 / 0.50.

## champ-h5 candidate (strawberries by demand) — 2026-09-11 · not shipped

Final knob table (3,600 paired samples): premium floors ×0.7 **+$1,949 ± 157** (n=501), carrot
day 25 **+$1,545 ± 116** (n=498) — champ-h4's knobs at ~12σ. The only surviving context rule,
strawberry target 40 when ≥1 strawberry consumer is open (+$881 ± 391, n=418; −$3,423 ± 668
with none), was tested as champ-h5c on top of champ-h4: arena 14–14–2, paired +$658 ± 1,901 —
consistent with the measured effect but unverifiable at gate sample sizes. No herd knob (cows
6/10, sheep 6/10, geese 2/3) pays in any demand context; standing carrot/tomato patches never pay.
`KnobBanditPolicy` gained explicit `conditions` (min/max consumers on the day-6 context) for
interpretable rules.

## champ-h4 — 2026-09-11 · SUBMITTED · first bandit-measured adaptive knobs

The adaptive layer, finally with evidence. Two learning approaches failed first and are recorded
in `docs/VERSIONS.md`: a residual PPO planner (offset heads over the champion plan, four runs)
converged back onto the prior — whole-episode credit over ~60 deviations per game is too weak
against ±$30k outcomes — and a single-day counterfactual bandit (4,800 paired samples) found no
positive one-day tweak within ±$500, because the plan's standing quotas refill any one-day change
the next day. What worked: **season-long knobs** of the champion plan, one per game, applied from
day 6 (shops revealed), each paired against the unchanged plan on the identical world
(`rl/bandit.py collect --knobs`).

- At 1,200 paired samples, two knobs are unconditionally positive at ~5σ: **premium sell floors
  ×0.7** (sell strawberry/milk/wool sooner — the first seller takes the price in this market),
  +$1,836 ± 380; **endgame carrot switch on day 25 instead of 22**, +$1,572 ± 249. Shipped as
  constant rules through `KnobBanditPolicy` (`arena/agents/champ-h4/rules.json`).
- Context-dependent signal already visible: strawberry target 40 is **+$2,274 ± 927 when at least
  one strawberry consumer is open and −$3,548 with none**; cows 10 only pays with milk consumers.
  Those need the regression fit (collection continues on the 9 surviving arms).
- Clearly harmful, measured: hires −2 (−$11.7k), 16 standing carrots (−$7.7k), strawberries 21
  (−$5.7k), hires +2 (−$4.4k), geese 3, carrot day 18, wheat engine 24, premium floors ×1.3.
- **Results**: arena vs champ-h3 **29–1**, $90.7k vs $86.1k; **24 paired champion worlds +$1,478 ±
  692 (18/24)** — the first candidate positive on both; held-out ladder **1.00 / 0.83 / 0.17 /
  0.17 / 0.00 / 0.50** (champ-h3 …0.08 / 0.17 / 0.00 / 0.42); p99 4.6 ms.

## champ-h3 — 2026-09-11 · SUBMITTED · sweep executor

Measured on our five worst worlds against the recorded occupant of our own seat: they got
105–159 productive actions a day out of the same crew with 94–140 moves and idle time to spare;
champ-h2 got 71–105 with 156–184 moves and no idle. The zone round scored tasks by
(priority, distance), so a hand walked four tiles to a harvest and four back to a water next
door. Now (distance, priority): finish what is on and around the current tile, move on;
survival tasks (dies tonight) jump the queue from hour 16; cross-zone help within 5 steps.

- Productive actions 105–128/day (from ~100), moves 128–169 (from 148–171), 58 tiles standing
  on day 18 (from 48); wheat +123 units, carrots +40, strawberries +21 per game.
- **Arena vs champ-h2: 30–0**, $91.0k vs $78.2k. **Held-out ladder 1.00 / 0.83 / 0.08 / 0.17 /
  0.00 / 0.42** (best yet; $108k earned in the 600–1000 bin, was $94k). p99 turn 0.40 ms.
- Paired champion worlds: neutral (−$1.2k ± 4.5k). The extra output lands in markets that the
  2500-rated scripts already flood: inventory only recovers through shop consumption (6 units a
  day per consumer) and the first seller takes the price. Two market experiments failed for the
  same reason and were reverted: paced selling (hold above a floor until the final week) let the
  opponent sell first and lost $30k on the traced world; capping strawberry tiles by the demand
  anchor was neutral. Converting throughput against strong opponents needs the adaptive
  allocation layer, not more executor.

## champ-h3 experiments — 2026-09-10 · not shipped (negative results)

Milk and labor were the next levers after champ-h2; each candidate was measured on **24
paired champion worlds** (same sessions, same seats) against the shipped champ-h2, with
hooks that reproduce champ-h2 exactly (+$0) as the control. Per-world outcomes swing by
±$30k between near-identical policies (paired s.d. ≈ $26–28k; arena s.d. ≈ $17k), so 8-world
readings earlier in the day ($88k) were noise.

| Candidate | Δ vs champ-h2 (24 paired worlds) | Verdict |
|---|---|---|
| cows before sheep in the buy order | −$6.0k ± 2.5k (better 6/24) | breaks the day-6 wool → land bootstrap |
| workload-weighted zones (pasture tiles ×4) | −$2.2k ± 5.0k | neutral; state-based weights drifted zones mid-day and starved a sheep |
| cash reserve for the first quadrant | −$4.9k to −$8.9k | delays the cow ramp |
| care skipped below 15% of base (was 35%) | +$0.2k ± 1.1k | neutral |
| melon window-watering at harvest priority | melon avg price unchanged (~$170–230) | the shared price is set by the opponent's day-10 dump |
| demand-conditional crops (strawberries scaled to consumers, standing carrots/tomatoes) + 3 geese in egg worlds | +$3.7k ± 4.9k | geese cost −$30k in the worlds that bought them: their daily feed/care/collect/egg load broke the cow/sheep care regime |
| same without geese | +$3.4k ± 5.3k (16/24); **arena vs champ-h2 12–18**, $79.8k vs $81.1k | not shippable |

Kept: the finding that outcome variance, not mean economy, is now the dominant term — the same
playbook earns $46k–$136k on champion worlds depending on the shop draw and on the opponent's
market pressure (in the bad draws the recorded champions still hold ~$90k). Code reverted to
the champ-h2 state.

## champ-h2 — 2026-09-10 · SUBMITTED (id 56152624) · animal chores first, champion sheep ramp

Pure heuristic (no neural net), same playbook as champ-h1 with the three levers the decode
left open — feeding, sheep, wheat — after tracing nightly fed/cared coverage per animal.

- **Feed/care above routine watering** (`P_ANIMAL = 1.5`). Engine: a fed+cared production
  yields 1 + banked care days (a sheep cared for 3 days gives 4 wool, not 1) and an *unfed*
  production night forfeits the whole bank. Coverage was 9–11 of 16 animals on days 16–28;
  now 14–16 of 16.
- **Two-pass zone round**: real work in your zone, then real work anywhere, then filler
  watering. Pastures are built centre-first, so 16 animals sat in 2–3 overloaded zones while
  other hands watered plants that weren't due.
- Sheep ramp at the recorded pace (4 by day 8, 6 by day 10, 8 by day 12); wheat engine 32.
- **Results**: solo on champion worlds $80.4k → $83.9k; wool $17k/116u → **$32k/162u**
  (champion range $35–44k); arena vs champ-h1 **16–4** (p = 0.012), $79.3k vs $76.0k, max
  $115.9k; held-out ladder 1.00 / 0.75 / 0.00 / 0.08 / 0.00 / 0.17 (unchanged within noise).
- Still open vs the $117k profile: milk $21k vs $29k (181 vs 224–265 units), wheat 372 vs
  600 units (labor-bound), 1000–2500 rating bins still ~0.

## champ-h1 — 2026-09-09 · SUBMITTED (id 56128207, rated 690 after 24h) · decoded champion playbook (heuristic)

The session-replay track's rating ladder showed rl-v9 and every trained successor
(g10–g13) frozen at 0.83 / 0.08 / 0 / 0 / 0 / 0 win rate against recorded opponents
binned by Kaggle rating — a ~600-rated economy that no training signal moved. So this
version comes from **decoding what 1500+/2500+ players do** (`tools/decode_sessions.py`
replays 683 harvested sessions on their true seeds and profiles both sides per day) and
fixing the executor until it could express that playbook.

- **Playbook** (`planner.make_champion_plan`): day 0 = 2 cows + 2 sheep + 12 melon + 7 wheat;
  days 1–5 funded by animal fertilizer (~$100/unit, produced fed or not); first quadrant
  bought the turn the day-6 wool lands; strawberries 18 → 33 by day 13 (fertilized: 2 units
  per production); 8 cows / 8 sheep by day 10–14 bought before demand is revealed; a 28-tile
  rolling wheat engine; carrots on freed tiles from day 22.
- **Executor fixes found by hour-by-hour tracing**: animal carriers place first (a cow was
  hauled around for 3 days); fertilizer treated as cash (no wheat/carrot fertilizing at base
  price, early shed trips, reserve for strawberry tasks); feed bought at any sane price with
  two days of stock (the herd starved at $40k cash behind a "pricey" guard); watering only
  where the engine pays (growth window / fertilized production night / survival) — daily
  watering had starved planting of labor; zone-local wheat/fertilizer provisioning; planting
  tasks on every empty tile; even zone split (12 hands had 11 zones); survival override only
  pulls idle units (it was re-targeting walking hands every turn: 200 moves per 60 actions).
- **Results**: solo on champion worlds $80.4k (rl-v9: $57k there); arena vs rl-v9 **20–0**,
  **$100,345 mean / $119,824 max** vs $59,182; held-out rating ladder
  **1.00 / 0.75 / 0.08 / 0.00 / 0.00 / 0.33**; p99 turn 0.32 ms.
- Next: this heuristic becomes the BC teacher; remaining gaps vs $117k champions are sheep
  timing (wool $17k vs $35k+), wheat volume (382 vs 600 units), and late-game feeding.

## rl-v5 — 2026-08-28 · SUBMITTED (id 55853641) · profit champion

Generation-5 **speculative-invest planner**, built from the decoded leaderboard meta.

- BC teacher mix: 50% a new *speculative* heuristic recreating the observed top-performer
  strategy (episode 101761797, finals $105k/$96k: day-0 ruminants on expected demand,
  30+ strawberry tiles, 14-16 cows, no geese, investment-first through day ~10) + 50% the
  proven reactive teacher; PPO then learns when each playbook applies via the demand
  forecast features.
- Head ceilings raised to the meta (wheat 40 tiles, strawberry 32, cows 16, sheep 14,
  carrots 26); softened carrying/unsold penalties; hot-then-scheduled LR (2.5e-4).
- **Results**: solo **$85,238 mean / $98,378 max** (rl-v3: $68,671); contested best game
  $97,841; 90-100% wins vs every league gate except rl-v3 head-to-head (3-17, whose
  counter-style feeds on rl-v5's economy). Both champions stay on Kaggle — submissions
  rate independently. 22 evals, zero training collapses.

## gen-4 experiments — 2026-08-28 · not shipped (tag g4-experiments)

Demand forecasting (`econ.forecast_drain`), carrying/unsold penalties, labor-utilization
features, architecture v2 (192/128 GELU+LayerNorm policy + decoupled value net), and the
LR scheduler all land here — but every gen-4 candidate lost its fresh-seed gate to rl-v3
(best 2-18). The stability machinery was validated (10 straight improving evals, zero
collapses) and became the foundation gen-5 shipped on. Also recorded: changing head VALUE
semantics silently remaps older policies' choices — champions ship only from their frozen
`arena/agents/<name>/main.py` bundles.

## rl-v3 — 2026-08-28 · SUBMITTED (id 55829852)

Generation-3 **league-trained, profit-first neural day-planner**.

- Trained against a 4-gate league {v3 heuristic, rl-v1, rl2-it59 exploiter, rl-v2};
  new best checkpoints join the opponent pool during the run (league growth).
- Profit-first objective: terminal reward dominated by final money (win bonus cut to a
  tiebreaker); checkpoint selection by **mean profit across all gates** on rotating eval seeds.
- Convergence stopping: run self-terminates after 5 evals without profit improvement
  (stopped at iter 44; best at iter 19).
- **Results**: 50–0 vs rl-v1 ($59.3k ± 2.7k vs $40.1k), 20–0 vs v3 ($63.6k), 16–4 vs rl-v2,
  tournament 24/24 at $62.7k mean, healthy mirror ($63.4k/$58.9k), step p99 1.2 ms.

## rl-v2 — 2026-08-28 · league member (not submitted)

Generation-2 profit-first planner (g2d best) + executor upgrades.

- 133 features (was 86): per-product **supply-vs-town-demand gap** (mine + opponent's projected
  7-day output vs shop drain), price curvature, land economics. 13 action heads (was 11):
  hires widened to 16, learned premium/staple **sell-floor scale** heads.
- Neural-path land safety nets (crunch fallback + demand-pressure trigger) fixing rl-v1's
  refusal to buy land; idle-worker pre-positioning; locked-tile walk-back.
- Training stabilizers after gen-2 collapses: value-head warmup, LR 1.5e-4, KL early-stop.
- **Results**: beats v3 17–3 ($43.2k), money-ties rl-v1 ($41.0k vs $41.5k; 10–20 h2h),
  high variance (solo max $78.1k, std $14.7k). Kept as league opponent.
- Side artifact **rl2-it59** ("the exploiter", g2b iter 59): beats rl-v1 35–15 but loses to v3
  3–17 — proof that training vs one static incumbent breeds a strategy cycle; league member.

## rl-v1 — 2026-08-27 · submitted (id 55825229, public ~486)

First **learned** agent: neural day-planner over the scripted executors.

- Tiny MLP (86 features → 96×96 → 11 masked heads + value), pure-Python inference (~30
  forward passes/episode, p99 0.8 ms). Heads are ABSOLUTE standing targets (crop tiles,
  herd sizes, hires, land, melon floor); scripted scheduler/market executes the micro.
- BC-initialized from v3, PPO vs {v3, starter, snapshots}; best checkpoint (iter 19) exported.
- Critical fix: bundle no longer touches `sys.modules` (two agents in one process — including
  Kaggle's self-play validation — used to overwrite each other).
- **Results**: 50–0 vs v3 across blocks ($42.7k/$41.1k vs ~$24k), tournament 32/32,
  solo vs starter $54.5k ± 2.5k.
- Known weaknesses (fixed in rl-v2+): never buys land under demand growth; beatable by a
  targeted exploiter.

## v3 — 2026-08-27 · submitted (id 55817540, public ~497)

Scripted heuristic, final form.

- CARE regime for ruminants: cows/sheep fed daily + cared (banks +1 product/day ≈ $125/action);
  geese stay on cheap alternate-day feeding.
- **Results**: beats v2 17–3; solo vs starter $44.9k mean; then-tournament sweep.
- Negative results recorded: 16 hands/day, 40 geese, daily goose care, aggressive feed-wheat
  buying, skipping cheap fertilizer collects — all lost head-to-head gates.

## v2 — 2026-08-27

Demand-adaptive livestock + market-controller hardening.

- Cows/sheep bought when milk/wool shops unlock (town drain >> T holds prices at $250+);
  strawberries planted at 2+ consuming shops; milk/wool adaptive sell floors.
- Fixed silent 10-order-cap truncation (goose pipeline freeze), goose delivery batching,
  feed buffer buys, fertilizer floor → $1 (a fert hoard blocked the shed; midnight overflow
  destroyed produce), shed pressure valve.
- Isolated bundler (no `sys.modules`).
- **Results**: beats v1 20–0 ($26.2k vs $8.7k contested); solo $40.4k.

## v1 — 2026-08-27

First full scripted economy.

- Melon rush (days 0–2) → goose/egg factory conversion; wheat feed engine; carrot/tomato
  filler; fib-priced hand hiring (~14/day); marginal-revenue sell sizing from the exact
  engine-mirrored price function; day-29 liquidation.
- **Results**: solo vs starter $36.2k mean (starter: $3.5k).

## M0 / demo — 2026-08-27

- Engine-mirrored `constants.py`/`econ.py` with parity tests against the installed engine.
- Arena harness (seeded batch runner, paired sign tests, tournament), submission bundler,
  demo melon agent preserved as baseline ($5,957 solo).
