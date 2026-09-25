# How the top players beat us, what we have tried, and what is left to try

*Research note, 2026-09-25. Every number here is either measured in this
repository (file named) or measured today on the standing evaluation: 785
held-out recorded games, band-stratified across six rating bins, played
paired (same worlds, same seats) with McNemar on discordant games. The
shipped champion, the RL nets and today's development tree were all put
through that same instrument for the first time this week.*

---

## 1. The summary, in five findings

1. **The development tree is the strongest agent we have, and it has never
   been shipped.** On the 785 held-out games, today's `kagent/` champion plan
   wins 182 (0.232); with champ-h8's knob rules on top, 190 (0.242). The
   frozen champ-h8 bundle wins 144 (0.183); the frozen champ-h1, 46 (0.059).
   That is +5.9 points over the shipped champion (z +3.69, p 0.0003, five of
   six bands) and +17 over h1. A month of planner and executor work is
   sitting in the tree. *Bundle it and submit it.*
2. **The RL day-planner track is closed for win rate.** Its best configuration,
   the land-capped control, wins 58–63 of 785 across three independent
   training seeds (0.074–0.080). That is 10.7 ± 0.3 points below champ-h8 and
   15 below the dev tree, worse in every band. Seven valid board-trunk arms
   never reliably beat the no-board control and are seven times more
   seed-variable. The one thing that survived replication is a rule, not a
   model: never buy the fourth quadrant, +1.7 ± 0.3 points and +$4.2k of
   margin, and the dev tree's champion plan already contains it.
3. **What top players do differently is measured and specific:** they move
   half as much per action (0.71 vs our 1.45 moves per productive action),
   haul half as much (PICKUP 194 vs 348 a game), harvest 489 times to our 290,
   scale strawberry, milk and wool output with demand (4.1× across the demand
   range where ours is 1.0–1.8×), grow four products we barely touch (wheat
   3× ours, carrot 3×, tomato and eggs absent from our farms), peak their
   board on day 9 where we peak on day 21, and hold stock longer before the
   first sale. They open the third quadrant on day 9–11 every game and the
   fourth only once the first three are full, on days 15–18.
4. **The public top agents are mostly replay tapes with thin online
   wrappers**, and routers that switch between 5–13 tapes on public state.
   Our own tape selector rated 1,750 on the ladder, higher than any heuristic
   or net we have shipped; the tape lineage has several free, untested
   improvements (§5.G).
5. **The highest-value untried levers are heuristic, cheap, and testable at
   inference in an hour each** on the dev champion plan: watering and chore
   triage by product value; demand-responsive volume; hold-stock and
   sell-later rules; the capacity-gated fourth quadrant; harvest-collection
   priority. Each is ranked in §5 with its evidence, cost, and test design.

---

## 2. Where every agent stands on the same 785 games

| agent | wins | win rate | money | margin | note |
|---|---|---|---|---|---|
| dev tree + h8 knob rules | 190 | **0.242** | $84,125 | −$9,734 | today's `kagent/`, unshipped |
| dev tree, plain champion plan | 182 | 0.232 | $83,231 | −$10,801 | knob layer adds +1.0 pt, n.s. |
| champ-h8 (frozen, strongest shipped) | 144 | 0.183 | $80,802 | −$13,894 | champion plan + 4 knob rules |
| RL board planner cloned from top players, seed 1234 | 141 | 0.180 | $80,817 | −$16,161 | K below: BC from 11,350 top-band sessions, trainable trunk, PPO |
| RL no-board planner cloned from top players, seed 1234 | 124 | 0.158 | $80,900 | −$17,299 | same BC, no board |
| capped RL control, 3 seeds | 58–63 | 0.074–0.080 | $64.7k | −$36.8k | BC from our own heuristic; same PPO budget as the two rows above |
| champ-h1 (frozen, recorded as submitted) | 46 | 0.059 | $72,137 | −$29,765 | plain champion plan of weeks ago |
| RL control, uncapped (B2) | 47 | 0.060 | $61,072 | −$41,068 | |
| frozen-trunk RL, uncapped (B2) | 21 | 0.027 | $55,150 | −$58,495 | worst |

Sources: `docs/stage_b5_preregistration.md` §6–13, `docs/results/*.log`.
The eval's own error bar is ±0.008–0.015 on a win rate; between training
seeds the no-board RL control varies by 0.003 and board nets by 0.023–0.026.

---

## 3. How the top players are more effective than our agent

### 3.1 Execution: they move less and haul less for the same work

`docs/executor_problems.md`, `tools/replay_probe.py`; op counts from
`CHANGELOG.md`.

| | M & M & P & Q (rated 3,093) | pub-ahmed-v38 | champ-h8 |
|---|---|---|---|
| moves per productive action | **0.71** | 0.84 | **1.45** |
| actions per tile-day | 2.10 | 2.02 | 1.67 |
| moves per tile-day | 1.43 | 1.61 | 2.39 |
| productive share of worker turns | | 46.0% | 34.0% |

Per game, ours vs MMPQ: WATER 859 / 1,195 · HARVEST **292 / 470** · CARE
199 / 329 · FERTILIZE 112 / 220 · **PICKUP 348 / 194** · long hauls of four
tiles or more 343 / 162. On 94.9% of the turns where they work and we move,
our nearest task is at distance zero: the deficit is not routing to distant
work, it is hauling and re-hauling. Twelve routing mechanisms and four
throughput raisers were tried against this and all read null or negative
(§4); the executor's movement budget optimised perfectly was measured as
worth about $0, and the hauling floor that fixed pickup-per-feed lost
$12,026. So the movement gap is real, large, and not closable by routing
alone; it is coupled to *what* the planner asks for.

**Turn-by-turn measurement, 2026-09-25** (twelve M & M & P & Q replays, both
seats; our RL control and today's dev champion plan on two traced worlds;
code and full report in the session scratchpad `research/`). The executor
gap is the same for both of our agents, so it belongs to the scheduler,
not the planner:

| per game | M & M & P & Q | their opponents | our RL control | our dev champion |
|---|---|---|---|---|
| moves per productive op | **0.71** | 0.88 | 1.24 | 1.40 |
| productive share of unit-steps | **55%** | 51% | 44% | 41% |
| effective chores per unit-day | **13.3** | 12.1 | 10.6 | 9.9 |
| chores per tile visit | **1.98** | 1.92 | 1.56 | 1.48 |
| next chore on the same or an adjacent tile | 89% | 86% | 79% | 78% |
| pickups | **194** | 216 | 395 | 380 |
| pickups per FEED | **0.55** | 0.60 | 1.03 | 1.09 |
| feeds from wheat harvested in hand, no shed trip | **26%** | 1% | 0% | 0% |
| effective harvests | **470** | 476 | 270 | 263 |
| effective plantings | **266** | 253 | 165 | 145 |
| one-time-crop harvest latency, median steps after peak | **0** | 0 | 22.5 | 23.5 |
| replant within one step of harvest | **78%** | 89% | 55% | 66% |
| WATER ops with no effect | **16%** | 22% | 23% | 26% |
| idle share of unlocked tiles | **12%** | 14% | 28% | 17% |
| price at sale over the game's mean price | **1.044** | 0.991 | 0.990 | 0.983 |
| revenue share in days 27–29 | **23%** | 16% | 15% | 10% |
| hands at noon / verified wages | 8.6 / **$4.3k** | 9.6 / $5.5k | 9.1 / $6.1k | 8.9 / $4.4k |

Three method facts the measurement established: the yield bonus of a
one-time crop is applied by the WATER op itself, only at wheat ages 2–4,
carrot 2–3 and melon 6–10; units respawn at the shed at hour 0, so any move
at hour 23 is wasted (5.5% of our moves, 0% of theirs); and M & M & P & Q
DROPs and SELLs in the same step, so the shed never holds stock long.

What the top agent does turn by turn that ours does not: a fixed daily
script (all seed purchases and hires at hours 0–1, one pickup per unit at
hours 0–2 and then no shed trips); **chore chains on a tile** before moving
(FEED, CARE, COLLECT on an animal; WATER, HARVEST, PLANT, WATER on a crop,
the second WATER because a seed left dry on its planting day dies that
night); wheat rows placed beside pastures so a unit harvests and feeds
without visiting the shed; harvest of one-time crops on the frame they
peak, replanted the next step; alternate-day watering outside the bonus
window and daily inside it; fertilizer collected from every animal daily
and put on wheat (220 FERTILIZE a game to our 112); selling the instant
produce reaches the shed, at 4% above the game's mean price, with $5.6k
liquidated on the last decision step. Our own day 12, read the same way:
the farmer ferried six sheep one or two at a time to pastures being built
on land bought that morning, while hands walked back to the shed for one to
three wheat at a time.

### 3.2 Volume: they respond to demand, we do not

`docs/market_congestion.md`, `docs/market_structure.md`. Product output as
town demand rises across its range:

| product | ours | top players |
|---|---|---|
| wool, drain 1 → 25 | 102 → 187 (1.8×) | 131 → 534 (**4.1×**) |
| milk | 2.1× | 3.3× |
| strawberry | 236 → 228 (**1.0×**) | 2.1× |

Units per seat, ours vs top: WHEAT 178 / **496**, CARROT 39 / **131**,
TOMATO **0** / 51, EGG **0** / 91; parity on strawberry, milk, wool, melon.
We are at parity in the four markets that crash and three times behind or
absent in the four that do not. We are also a net wheat *importer* at about
$13k a game (net wheat position −$1,237 vs ahmed's +$11,708).

### 3.3 Waste and harvest

`CHANGELOG.md` waste audit, `docs/metrics_catalogue.md`, today's traces:
harvests 290 vs 489 a game; stranded unharvested yield 2.7 vs 0.0; in a
traced win 67 yield units died on our plants against the opponent's 7; crop
death rate 24.2% vs 11.3%; strawberry death 75% vs 33%. Pre-yield drought
losses are small and level with the opponent's (4–10 plants, under $1k); the
waste is harvest not collected in time, not plants unwatered.

### 3.4 Land and labour

Today's replay measurements, `docs/stage_b5_preregistration.md` §0 and the
M & M & P & Q analysis: the top agent opens the third quadrant on day 9–11 in
every game and the fourth in 8 of 12 games, on days 15–18, only when the
first three are full (idle tiles 1–2), with the same ten hands, and then
fills it (occupied tiles 71–73 → 77–86). Our RL planner opened the fourth
around day 12–14 with 16–21 tiles idle and never filled it; the champion
plan stops at three. Hands per worked quadrant: theirs 3–4.5, ours 2.2–3.5.
Their hands serve about eight occupied tiles each, ours six and a half.

### 3.5 Timing: they build early and sell late

`docs/phi_ood_and_phase_timing.md` (514 replayed ladder sessions):

| landmark | ladder median | 2500+ | M & M & P & Q | champ-h8 |
|---|---|---|---|---|
| board peaks on day | 9 | 9 | 9 | **21** |
| first sale on day | 12 | 9 | **16** | 20–22 |

Board-peak day is the landmark most correlated with rating (Spearman
−0.36): stronger players peak twelve days before we do. M & M & P & Q's one
extreme trait is holding stock longest before the first drawdown, the third
independent confirmation that selling later than feels right pays here
(melon: harvesting at first yield lost $1,482; median sell day correlates
+0.136 with winning). Cash at day 10: champ-h8 $16, ladder $2,425,
M & M & P & Q $3,713, 2500+ band $5,063 — the divergence is one day wide.

### 3.6 The opponent is half the bank

Muting the opponent's supply is worth $52,652 a game (z 17.5), half the
bank, 97% of it through strawberry, milk and wool. Their board predicts
their supply (36% of forecast error); their shed does not (1.5%). Every
RL board arm that trained its trunk *raised the opponent's bank* (B2 +$3.3k,
B3 +$6.0k, B5b +$5.4k, B5c +$7.6k): the learned planner yields market share.
The dev champion plan lowers it (−$836 vs champ-h8).

### 3.7 What the top public agents are made of

`CHANGELOG.md`: most highly voted public agents are 720-step replay tapes of
2500+ ladder players plus a thin online wrapper (majority vote over three
episodes, a one-turn "sell early" front-run); the strongest are routers that
choose among 5–13 tapes at 144-turn boundaries with a fitted tree on public
state. M & M & P & Q is a genuinely reactive agent (27% action agreement
across worlds). Our tape selector rated 1,750; its lineage is where our
highest ladder rating came from.

---

## 4. What has been tried, by area, with the verdict

Condensed from the full ledger in `docs/*.md`, `CHANGELOG.md`, ~360 arena
arms and ~130 ladder result files. Money figures are paired gates unless
stated.

**Planner and allocation.** Shipped: the decoded champion playbook (h1),
animal chores first (h2), sweep executor (h3), four knob rules (h4–h7,
+$1.0k to +$2.0k each), the feed-buffer fix (h8), the forward-market rent
rule (**+$3,185 ± 768 replicated, +$4,336 on the fixed executor, win 0.302 →
0.347**), future-shop demand (+$676). Refuted: every single-product volume
knob, feed-days above 2 (monotone to −$12.7k), moving the allocation toward
the LP (**−$31,728, z −22.5**), the LP as a label source (r −0.34), demand
books (−$6.6k herd), hiring as a constant (−$4k to −$19k), closed-loop hiring
(−$11k), the fourth quadrant (−$6k twice), melon fertilisation (−$14.9k),
the glut penalty (−$1.2k), every derivative-free plan search (closed on
budget: ~50,000 games per $3k). Held: hiring as a per-day schedule (+$1.2k
money, flat on wins).

**Executor, routing, hauling.** Twelve routing mechanisms and four
throughput raisers null or negative; global matching shipped rating-neutral
(+$1.4k, rating 819 vs 810); cross-turn commitment refuted (−$2.2k); the
underfoot fix shipped (obedience 85 → 92%, money-neutral); pasture placement
near the shed −$11.3k; hauling floor −$12.0k; shifting capacity earlier
−$15.9k; fixed zones cut the 2500+ win rate from 0.20 to 0.07; market-order
slot reservation down to −$15.8k. The executor's whole movement budget was
bounded at about $0, and freed turns convert at zero.

**Market and selling.** Sell floors are inert (bind on 2.5% of sizing calls,
0–1 discordant worlds); harvesting melon early −$1.5k; geese substitution
cut wins 8 → 2 of 144; market denial 0 of 120 games flipped; the premium
scale sweep is flat. **Geese as an addition to the dev tree (25 September,
`docs/results/rules_geese_eval.log`): 4 geese days 6–18 −8.0 pts (z −6.1,
24 arm-only vs 87 base-only games), 6 geese −12.6 pts (z −9.0), worse in
every rating band against the dev tree with h8's rules (0.242).** Our bank
moved −$1.4k / −$5.5k while the recorded opponent's rose +$5.1k / +$7.7k:
feeding and caring for the geese displaced strawberry, milk and carrot
volume (−66, −51, −57 units on a traced 2000–2500 game, with 3 of 8 sheep
unfed at noon and two animals escaped by day 29), and the opponent sells
into those same markets, so the price relief is theirs. Eggs earned $8k on
that game. Third geese refutation, now with the mechanism; the top players
carry ~6 geese because their chores cost half of ours (§3.1), not because
eggs pay for the labour at our rate.
**Culls on the dev tree (26 September, `docs/results/rules_cull_eval.log`,
same 785 games, paired against the dev tree with h8's rules):** stop
feeding cows when milk falls under 0.35 of base (from day 12) +1.0 pt (z
+1.5, p 0.18, 18 vs 10 discordant, 3/6 bands, bank +$1.5k but the opponent
+$1.3k); sheep when wool falls under 0.5 of base −0.4 (n.s.); stop feeding
cows and sheep from day 25 **−7.5 pts (z −7.2, 0/6 bands)**; all three
together −6.9. The endgame cull that the top agent makes loses for us: a
fed cow or sheep still returns its last two productions against a wheat a
day, and what we withdraw from the market the recorded opponent sells
into. Price-triggered culls are neutral to slightly positive and not
adopted; the endgame cull is closed.

**Learned models.** The executor BC ranker fits at 69.4% and gated null
(later +$1.2k, z 1.95, on the fixed executor: the only learned component
ever to read positive); raw-action BC $35; the pointer network −$33k; value
heads wrong-signed on 3 of 4 arms; the board CNN clears its kill criterion
on standing counts and has no consumer; BoardNet opponent-supply forecast
**+$2,731 ± 538, win z 3.44** as the rent rule's input (the one learned arm
to clear the win metric) but not better than the arithmetic rule head to
head; capacity channels killed (+R², 0% MAE, −44% out of distribution); the
temporal board stack refuted twice on flow, passes on margin; the full
neural planner −$22.9k; fourteen PPO generations plateaued at rating 810;
self-play 96% became 3 of 144 externally. This week: the frozen trunk
−3.3 points, the trainable trunk parity, the deployment-mix curriculum −2.3,
the phase potential −1.9, the land cap +1.7 replicated, and the whole RL
track 10.7 points below champ-h8.

**Reward shaping and curricula.** The trainer's three policy-distorting
terms were priced (40% and 238:12 of the win signal) and replaced by PBRS;
PBRS is a provable no-op for every episode-return gate we ship on; a potential
fitted on self-play scored −0.398 R² out of self-play; at 20 iterations the
shaping is the objective and the phase potential halved wins. Rating
curricula read identically to no curriculum; the deployment-mix curriculum
starved the win signal to 1–5 wins per 64 episodes.

**Opponent modelling, tapes, bandits.** Opponent shed oracle 1.5% of
forecast error (retired); the direct game-theoretic contest proxy null; tapes
+$20k, the day-6 selector rated 1,750 (identical bytes 1,706: rating noise
±45); per-world tape selection is unlearnable (oracle equals the shuffled
null); cross-lineage splices collapse; tape-to-executor handoffs at days 6,
10, 14 lose $38–53k; the knob-bandit space is flat after 30,272 samples, and
knob signs transfer across executors (3/3) while magnitudes shrink up to 9×.

**The recurring law.** Every fitted model that looked right on held-out data
and was then intervened on failed the paired gate; every gain that shipped
was arithmetic on a rule the README or the corpus states. The board track
this week is the same law with a replication attached.

---

## 5. Hidden approaches: what we have not tried, ranked

Ranking criterion: measured upside × plausibility, divided by cost, with the
test design each needs. **Every heuristic item is tested the same way: at
inference on the dev champion plan, paired against `dev_champion` on the
same 785 games (`rl/session_eval.py --arm x=dev:champion`), an hour a test,
before any training.** A rule now has to beat 0.232, not 0.080.

### A. Ship the dev tree — now, no experiment needed
+5.9 points over champ-h8 on the standing eval, on five of six bands. The
ladder has disagreed with local gates twice before, so submit, watch the
rating, and keep the frozen h8 as the fallback. Cost: a bundle build.

### B0. The executor's daily script, copied from measurement — five concrete rules
The turn-by-turn table above turns "moves 1.45 vs 0.71" into mechanisms,
each a scheduler rule, each with its own mechanism check in the metric
code, and each cheap to test at inference on the dev champion plan:

1. **Harvest one-time crops on the frame they peak, replant the next step.**
   Our median latency is 22–23 steps, theirs 0; 44 yield units a game decay
   on our plants against their 0.5. Harvest priority is already P=1; the lag
   is in how ripeness is detected, so this is a detection fix, not a
   priority change.
2. **Chore chains on one tile before moving**: FEED→CARE→COLLECT on an
   animal, WATER→HARVEST→PLANT→WATER on a crop. We do 1.5 chores per tile
   visit, they do 2.0. *Caution:* a "chaining" throughput arm lost $6,385
   and cross-turn commitment lost $2,162, so the test must be this exact
   within-visit bundle with chores-per-visit as the check, not a commitment
   to a tile across turns.
3. **Feed from wheat harvested in hand**: place wheat rows beside pastures
   and let the harvesting unit feed without a shed trip. 26% of their feeds,
   0% of ours; pickups per feed 0.55 vs 1.03–1.09; 190 fewer pickups a game.
   The earlier hauling floor (−$12k) forced fewer pickups without changing
   *where the wheat was*; this changes the layout, which is why it may not
   repeat that loss. Mechanism check: the feeds-from-hand share.
4. **Water to the bonus window**: daily at wheat ages 2–4, carrot 2–3,
   melon 6–10, alternate days outside it. 23–26% of our WATER ops do
   nothing; 16% of theirs. Coverage inside the window is already equal, so
   this frees labour rather than adding yield.
5. **No moves at hour 23**, and PASS once the day's work is done rather than
   drifting: 5.5% of our moves are structurally wasted.

Expected size: unknown individually; together they account for the
measured 1,200 extra unit-steps a game and 200 fewer harvests, on the same
executor the dev tree ships. Test each alone and then the bundle, at
inference, paired against `dev_champion`.

### B. Watering and chore triage by product value — untested, reallocates labour
`docs/spatial_program.md` §4: the only untried lever that *reallocates*
rather than adds labour. Today survival and tending tasks are ordered by
ripeness and distance; ordering them by `econ.price(product, inventory)`
protects the tiles worth the most when hands are short, which is every
mid-game day (hands per work 0.47–0.58 at days 11–15). Expected size:
unknown, plausibly thousands, because 24% of our crops die against their
11% and 75% of our strawberries against their 33%. Test: one scheduler
switch, inference, paired.

### C. Demand-responsive volume — the largest measured output gap
Our strawberry output is constant across a sixfold demand range; theirs
doubles; wool 1.8× vs 4.1×; tomato and eggs absent, wheat and carrot a third
of theirs. The single-product *constant* knobs all refuted, and that is the
point: the top players' volume is a *function of demand*, not a constant.
The anchors in `kagent/features.py` already compute supply-equals-demand
targets from the drain forecast minus the opponent's projected supply; the
rule to test is making the champion plan's standing targets for wheat,
carrot, tomato and eggs track those anchors (the four uncrashable markets)
rather than fixed values. Expected size: the market-structure note prices
wheat and eggs as annuities; the ahmed ledger puts our net-wheat deficit at
$13k a game. Test: inference, paired, per product so a bad one is dropped.

### D. Sell on the drop, at the price, and liquidate at the end
The top agent sells the instant produce reaches the shed (54% of its SELL
steps coincide with a DROP, 14% of ours), at 4.4% *above* the game's mean
price where we sell 1–2% below, and takes 23% of its revenue in the last
three days to our 10–15%. It also liquidates $5.6k of carried produce on
the last decision step; that part is not our gap — both of our agents end
with an empty shed (0–2 units in all four traced games), so the leftover
is already handled. Melon timing confirms the direction
(harvest-at-first-yield −$1.5k). The sell floors are inert, so the lever is
not floors: it is *when* produce is dropped and sold relative to the price
path. Test: inference, paired; watch margin as well as wins, with the
price-at-sale ratio as the mechanism check.

**Sell timing, measured 26 September** (scratch `sell_impact.py` over the
three traced games and the twelve M & M & P & Q replays; engine
`_process_market` fills SELL orders unit by unit at the running price, both
players interleaved). Our orders for the four steep-curve products cut the
price within one step to 0.62 (strawberry) and 0.53 (wool) of the first
unit's price; the top agent's orders lose 6–9%. Our realised price over the
product's game mean is 0.89, theirs 1.04, on 1,087 against 1,639 units,
and 68% of our units sell at hours 0–3 against 54% of theirs, 39% of
theirs at hours 20–23 against 11% of ours. The clean comparison is inside
the same market: on our three traced games the recorded opponents realise
0.85–1.03 of the mean strawberry price to our 0.68–0.89 and 1.36–1.50 on
melon to our 1.11, selling 33–42% of their strawberries in the evening.
Two rules were built and checked on the traces:
- `sell_pace` (planner knob, off): cap a steep-curve product's order at
  the units that keep the within-order drop under N% when shops drain it.
  Order impact 0.6–0.8 → 0.7–0.9, realised price and money flat (±0.5%):
  the level of the market is set by both farms' supply against the drain,
  not by our batch size. No eval arm.
- `evesell` (executor flag, off): a unit carrying ≥ 3 sellable units within
  5 tiles walks in to DROP by hour 22, so the produce sells at 22–23 before
  the hour-0 dumps. Evening sale share 0.1–0.2 → 0.3–0.45; realised
  strawberry price 0.89 → 0.94 alone and 0.90 → 1.08 with the four executor
  rules on one game, worse alone on another; harvests −3 to −21 a game for
  the walk. Ambiguous on traces (single games move with the shop re-roll),
  so it is in the paired queue as `evesell_four` against `exec_four`.
Melon is a different story: the opponents' 1.4× melon price is the day-10
harvest race (they clear melons on day 10 at $271; we finish on day 13),
which is harvest timing, not sell timing.

### E. The capacity-gated fourth quadrant — the top agent's own pattern
Open the fourth quadrant only when the first three are full (idle < 3), the
day is at least 14, and cash after purchase leaves working capital; fill it
with short-cycle crops (the planting cut-offs already force that after day
13–19). This is what M & M & P & Q do in 8 of 12 games. On the RL control the
fixed cap alone was worth +1.7; the dev plan already caps at three, so the
question here is whether a *conditional* fourth adds to it. Already built
as `KAGG_MAX_QUADRANTS`; the conditional mode is a small extension. Test:
inference, three-way (cap 3 / gate / no cap), paired.

### F. Harvest collection priority — the single largest volume gap
489 harvests vs 290, and yield dying on plants. Throughput raisers have all
failed, so this is not "harvest more often"; it is *collect ripe yield before
it expires* as a survival-priority task, and it interacts with D (stock is
worthless unsold). Test: inference, paired, with the stranded-yield count
from `tools/game_trace.py` as the mechanism check.

### G. The tape lineage — several free improvements, the highest rating we ever had
1. Re-rank the 243-arm tape screen by μ − 0.5·sd instead of mean margin:
   ranking by margin cost 15.6 points of held-out win rate and selected an
   exploiter. Data on disk, zero compute.
2. A day-26 endgame handoff from tape to executor: handoffs were tried at
   days 6, 10, 14 where the tape is strongest, never at 26, its only
   sustained loss.
3. Cross-branch splicing within the 58-tape cluster at days 7–12: 378
   zero-mismatch edges, 163 novel scripts, free kill test not run.
4. Add live public agents at 2000+ to the ladder pool (ladder Spearman vs
   rating is +0.94 already).
5. A router in the public style: 5–13 tapes switched on public state with a
   fitted tree. Ours selected on the day-6 yarn-store draw and rated 1,750;
   the public routers rate higher.

### H. The one learned component that cleared the win metric, wired in
BoardNet's opponent-supply forecast as the rent rule's input gated
**+$2,731 ± 538 with win z 3.44** over 1,131 worlds, and the temporal stack
passes on margin. It never beat the arithmetic rule head to head, but it was
never tried *on the dev tree* nor with the signed market inventory the
encoder now carries. This is the only model-side item in this list, and it
is a forecast feeding a rule, not a policy.

### I. Executor policy head over legal (unit, op, tile) triples — the correct critique, unbuilt
`docs/research_review_2026-09-17.md`, `docs/policy_spec.md`: a Maia-shaped
imitation head over enumerated legal triples, trained on top players'
per-turn actions, would target the movement and hauling gap directly
(0.71 vs 1.45 moves per action) in a way the task-ranker BC could not,
because the ranker chose among *our* tasks. Cost is high (a new action
space and a new dataset), the prior is the BC ranker's null, and the upside
is the one gap nothing heuristic has moved. Rank it after B–F.

### J. Risk-aware endgame rule
`rl/value_weights.json` exists and no shipped agent uses it: when P(win) is
below 0.5 late, take variance; above, kill it. The tape selector's P(win)
curve recovers on day 13 and loses on day 26; the champion's decays to zero
by day 19. Cheap; test at inference.

### K. The board model on the whole top-player corpus — first seed landed
Per the standing request: the trunk pretrained on margin over all 11,350
top-band training sessions, the planner readout cloned from the top
players' *own* implied choices with the board (every earlier BC cloned our
board-blind heuristic), then PPO with a trainable trunk under the land cap on
three independent seeds, and one no-board planner cloned from the same top
games. Evaluated paired against champ-h8 and the capped control.
**Expectation, stated in advance:** the top-player readout is the one board
input never tried and could move the RL planner several points; the record
says it will not reach the dev tree's 0.232, because the gap is the planner
architecture and the executor, not data volume — the most data-rich trunk
was the worst arm, and the trainable trunk's drift toward yielding market
share appeared in every run.

**Result, local minimal test (seed 1234, 25 September 22:41,
`docs/results/top_board_local_eval.log`).** The dataset is 681,000 samples
from 11,350 sessions with a seat rated ≥ 2500, the 2,861 held-out eval
sessions excluded before replay; the trunk reaches held-out margin R² +0.28
to +0.31; the cloned readouts reproduce the top players' exact day plan on
80.7 % (board) and 79.8 % (no board) of validation samples. PPO then ran
with the identical budget as the capped control (20 iterations, 64
episodes, land cap 3, varied worlds, seed 1234; 12 policy updates per
iteration, worker/trainer KL 4e-7).

| arm | wins / 785 | win rate | money | vs capped control (same games) | vs champ-h8 | vs dev tree + h8 rules |
|---|---|---|---|---|---|---|
| top-player board planner | 141 | 0.180 | $80,817 | **+9.9 pts, z +7.1, 6/6 bands, +$15.8k bank, opponent −$4.7k** | −0.4, p 0.88 | −6.2, z −3.9 |
| top-player no-board planner | 124 | 0.158 | $80,900 | **+7.8 pts, z +6.0, 6/6 bands, +$15.9k bank, opponent −$3.4k** | −2.5, p 0.13 | −8.4 |
| capped control (our heuristic's BC) | 63 | 0.080 | $64,981 | — | −10.3 | −16.2 |

Board versus no-board on the same games: +2.2 pts (z +1.6, p 0.135, 2/6
bands), inside one board seed's between-seed noise (0.023–0.026); the two
Kaggle seeds decide it. Reading: the only thing that separates the no-board
row from the capped control is *whose plans the readout was cloned from*
(the top players' instead of our heuristic's), and that alone is worth +7.8
points and $15.9k of bank — the largest single move the RL track has made,
larger than the land cap (+1.7), the potential (−1.9) and the curriculum
(−2.3) combined, and it won in every rating band although the clone set was
the top band only. Half of the pre-registered expectation held: the
top-player readout did move the planner, and the RL planner still does not
reach the dev tree (0.180 versus 0.232–0.242, z −3.9). What it did reach is
parity with the frozen champ-h8. The gap that remains to the dev tree is
the same size as the executor and planner changes made since h8, which is
consistent with the executor being the remaining difference.

---

## 6. What is running, and how to read it when it lands

- `tools/top_board_pipeline.sh` (K above), five stages, resumable, about six
  hours: dataset → trunk → BC readouts → PPO ×4 → paired eval. Read only
  the paired table against `champ_h8` and `cap_control`; the pre-registered
  bar for the board planner is beating the capped control on wins in ≥ 2 of 3
  seeds, and for relevance to the ladder, approaching `dev_champion`.
  **Seed 1234 landed (K): both arms clear the capped control by 8–10 points
  in every band.** `tools/top_board_seeds.sh` queues seeds 1235 and 1236
  (board and no-board) after the rule tests, with a paired eval into
  `rl/runs/top_board/seeds_eval.json`; that decides board versus no-board.
- Kaggle kernel `gideonoba/kaggriculture-topboard` (`tools/kaggle_job/job_topboard.json`):
  the same pipeline, seed 1234, on the kernel's corpus with the eleven new
  daily datasets indexed in-kernel; its eval is per-bin 100 against the
  champ-h8 bundle and `dev:champion`. It is the free-compute path the
  daily-download training will run on.
- `tools/episodes_pipeline.py run --since 2026-09-14`: the eleven daily
  episode datasets (14–24 September, ~7,000 episodes, top score 3,100–3,240)
  the corpus has never seen, extracted in a Kaggle kernel four days at a
  time and merged into `rl/datasets/sessions` on return. The first kernel
  (version 16) is pushed. Use them to re-run K and to refresh the held-out
  eval; the corpus otherwise ends on 13 September.

---

## 7. Method notes that now hold for every experiment

- Evaluate per rating band on held-out recorded sessions, paired, ~785
  games, McNemar; never read the in-training win-vs-gate or an 18-game eval.
- Include `dev:champion` and the frozen champion bundle as arms in any eval
  that could lead to a submission.
- One training seed is not a result for a board net (between-seed sd
  0.023–0.026); it is for the no-board control (0.003).
- The trainer refuses on a worker/trainer policy mismatch, logs its update
  count, loads the trunk after the resume, and keeps the trunk out of BC;
  read `policy_updates` and `onpolicy_kl` before any PPO number.
- Check a behavioural switch on one traced game (`tools/game_trace.py`)
  before launching a multi-hour arm; land alone is bought through three code
  paths.
- **Pairing does not hold the town's shops fixed.** The engine seeds one
  RNG per day from (world seed, day), spends one draw per *empty tile on
  both farms* for weed spawning, then draws the new shop on unlock days
  (every third day, up to eight instances, with replacement). Any arm that
  changes the empty-tile count on an unlock day re-rolls that day's shop;
  on a traced geese game the two arms diverged from day 9 (PET_CAFE versus
  YARN_STORE) and wool sold at $240 all game for one arm and $1 from day
  16 for the other, moving both banks by tens of thousands. The seed is in
  `env.info`, not in the agent's configuration or observation, so the draw
  is unforecastable and the re-roll is uniform: a paired *mean* over 785
  games is unbiased, but a single traced pair cannot attribute its
  difference to the change, and the recorded opponent's actions were taken
  under the recorded shops, so the tape is incoherent with a re-rolled
  market. Instrument improvement not yet built: force the recorded
  session's shop sequence in the eval environment (one patched engine
  call), which fixes demand across arms and restores the tape's coherence.
  Do it between eval campaigns; it changes every reused arm.
