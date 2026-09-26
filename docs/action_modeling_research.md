# Modelling the executor's actions from the top players' games

*Research note, 2026-09-26 (Fork C of the executor workflow). Code under
`rl/execnet/`; tests in `tests/test_execnet.py`. Every number below is
measured in this repository; the verdict on any model is the paired
995-game instrument (`rl/session_eval.py` against
`rl/runs/draw995/baselines.json`), never held-out accuracy.*

## 1. What was tried before, and why each failed

| attempt | what it learned | held-out fit | paired result | why it could not close the gap |
|---|---|---|---|---|
| `tools/exec_bc.py` — task ranker cloned from M & M & P & Q | which of the ~60 tasks *our* `build_tasks` proposes a unit walks to (label: the unit's next work action that day) | 69.4% of their choices (our rule 27.9%) | **null** over 4,800 paired games; +$1.2k (z 1.95) on the fixed executor | it can only rank what our planner proposed; the top agent's chains (WATER→HARVEST→PLANT→WATER, FEED→CARE→COLLECT on one visit) and its no-shed-trip feeding are not among the candidates |
| `tools/raw_bc.py` — raw action-type classifier (28 classes) | the op type from a fixed feature vector | 75.5% | **$35** | it picks "WATER" and then hunts for a legal tile; the tile choice, which is the whole movement budget, is not modelled |
| `tools/ptrnet.py` — pointer network unit→task | a joint assignment over the candidate set | — | **−$33k** | same candidate set as the ranker, plus a decode that over-commits; the cost function was the problem, not the matching (`make_assign_variant.py`) |
| value-ranking of actions (four heads) | V(s) after each candidate action | chance out of sample | closed | the fitted V cannot order actions |

The recurring law of the project holds for all four: a fit to observed
behaviour that looked right on held-out data did not move the paired gate.
The design left untried (research note §5 I) is a policy over the legal
(unit, op, tile) triples — the unit decides *what to do where it stands or
where to step*, not which of our proposals to serve.

## 2. execnet: the design

- **Sample** = one (step, unit) of a top player's game. Features
  (`rl/execnet/features.py`): the unit's 5×5 neighbourhood as 24 tile
  channels (locked / off-board / bare / weed / coop / pasture / plant with
  crop one-hot, age, yield, watered, dry / animal with kind one-hot, fed,
  cared, fertiliser waiting, yield), plus 31 scalars (clock, position,
  shed distance and adjacency, farmer flag, crew size, pocket wheat /
  fertiliser / produce / animal, seeds, shed wheat / fertiliser / total,
  money, plants per crop, unfed and fed animals, bare tiles, weeds, empty
  structures, unlocked quadrants).
- **Label** = the op the recorded unit chose, one of 18: the four moves,
  PASS, WATER, HARVEST, PLANT, FEED, CARE, FERTILIZE, DIG, PICKUP, DROP,
  PLACE, COLLECT_FERTILIZER, BUILD_COOP, BUILD_PASTURE.
- **Legality mask** from the verified engine rules (shed ops before the
  LOCKED guard; tile ops need an owned tile; PLANT needs a seed; FEED needs
  wheat in the pocket; HARVEST needs yield and, for plants, maturity). On
  the replays 100% of the recorded ops are legal under the mask.
- **Our label on the same state**: today's `kagent/` champion plan and
  scheduler are run on every recorded observation, so every sample also
  carries what our executor would have done. That gives the
  top-player-versus-us confusion for free, and a baseline accuracy for the
  model to beat.
- **Model** (`rl/execnet/model.py`): MLP, tile window 600 → 256 → 256,
  scalars 31 → 64, head 320 → 256 → 18 logits, mask applied to the logits,
  cross-entropy, split by game.
- **Deployment** (`rl/execnet/policy.py`): the day plan and the market
  orders are exactly `HeuristicPolicy`'s; after `scheduler.assign` the
  model's masked argmax replaces a unit's action when the op is in the
  take-over set (moves, PASS, WATER, HARVEST, FEED, CARE, FERTILIZE, DIG,
  COLLECT_FERTILIZER, DROP) and the model's probability is at least the
  confidence knob. Ops that need an item or a crop (PLANT, PICKUP, PLACE,
  BUILD) stay with the scheduler. Arm syntax
  `dev:execnet:<weights.pt>[:<rules.json>[:<conf>]]` in `rl/session_eval.py`,
  `--execnet` in `tools/game_trace.py`.

## 3. Dataset

`rl/execnet/dataset.py --n 160 --replays replays/mmpq --replays replays/boey
--replays replays/dsm`, 9 minutes on 2 processes (`tools/fast_env`): 198
games, 1,412,583 (step, unit) samples, 0.01–0.03% of recorded ops illegal
under the mask (dropped at training). The 160 sessions are the
highest-rated seats in the corpus not in the eval's held-out draw (mean
rating 3,223); the replays are Fork A's fetches, with the seat chosen by the
directory's team.

| player | games | samples | rating | moves | PASS | WATER | HARVEST | PLANT | FEED | CARE | FERT | PICKUP | DROP | COLLECT | our scheduler agrees |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sessions ≥ 3000 (many teams) | 160 | 1,149,945 | 3223 | 0.45 | 0.02 | 0.18 | 0.07 | 0.04 | 0.05 | 0.05 | 0.03 | 0.03 | 0.01 | 0.06 | 0.34 |
| DSM | 14 | 100,192 | replay | 0.43 | 0.04 | 0.15 | 0.07 | 0.04 | 0.06 | 0.06 | 0.03 | 0.03 | 0.02 | 0.07 | 0.36 |
| Boey | 12 | 84,532 | replay | 0.44 | 0.03 | 0.15 | 0.07 | 0.04 | 0.06 | 0.06 | 0.03 | 0.03 | 0.02 | 0.07 | 0.37 |
| M & M & P & Q | 12 | 77,914 | replay | 0.39 | 0.05 | 0.18 | 0.07 | 0.04 | 0.05 | 0.05 | 0.03 | 0.03 | 0.01 | 0.06 | 0.39 |

The three named players and the 3000+ field have the same op mix within a
point or two: 39–45% of unit-steps are moves, 15–18% WATER, 7% HARVEST,
6–7% COLLECT_FERTILIZER, 5–6% each FEED and CARE. Our scheduler, run on
the same observations, chooses the same op as the top player on 34–39% of
unit-steps.

## 4. Held-out fit and the disagreement with our scheduler

`rl/execnet/train.py --epochs 8` (154 s, 2 threads): held-out 20 games /
141,786 samples, train 178 games / 1,270,578. Accuracy on the top
players' ops: **model 0.776**, our scheduler 0.355; moves only, model
0.663; chores only (non-move, non-PASS), **model 0.876, ours 0.458**.
Model and scheduler agree on 0.356 of rows. Accuracy by model confidence:
≥ 0.5 → 0.822 on 89% of rows, ≥ 0.7 → 0.901 on 68%, ≥ 0.9 → 0.962 on 46%.

| op | support | model precision | model recall | our scheduler precision | our scheduler recall |
|---|---|---|---|---|---|
| NORTH | 19,070 | 0.71 | 0.69 | 0.23 | 0.24 |
| SOUTH | 12,426 | 0.69 | 0.63 | 0.17 | 0.25 |
| EAST | 11,415 | 0.75 | 0.64 | 0.15 | 0.30 |
| WEST | 19,637 | 0.74 | 0.67 | 0.22 | 0.23 |
| PASS | 3,516 | 0.67 | 0.65 | 0.13 | 0.01 |
| WATER | 24,571 | 0.84 | 0.93 | 0.81 | 0.55 |
| HARVEST | 10,147 | 0.77 | 0.81 | 0.29 | 0.36 |
| PLANT | 5,369 | 0.90 | 0.95 | 0.83 | 0.27 |
| FEED | 7,442 | 0.83 | 0.96 | 0.74 | 0.62 |
| CARE | 7,418 | 0.88 | 0.92 | 0.90 | 0.46 |
| FERTILIZE | 3,951 | 0.62 | 0.50 | 0.42 | 0.14 |
| DIG | 680 | 0.66 | 0.65 | 0.59 | 0.19 |
| PICKUP | 3,881 | 0.65 | 0.75 | 0.29 | 0.52 |
| DROP | 2,002 | 0.92 | 0.95 | 0.66 | 0.39 |
| PLACE | 1,331 | 0.91 | 0.88 | 0.90 | 0.21 |
| COLLECT_FERTILIZER | 8,505 | 0.85 | 0.88 | 0.55 | 0.50 |
| BUILD_COOP | 109 | 0.73 | 0.61 | 0.22 | 0.04 |
| BUILD_PASTURE | 316 | 0.82 | 0.89 | 0.08 | 0.28 |

**What the disagreement says about our executor.** Read the scheduler's
recall column as "when a top player does X here, how often would we":
WATER 0.55, FEED 0.62, CARE 0.46, HARVEST 0.36, PLANT 0.27, DIG 0.19,
FERTILIZE 0.14, PASS 0.01. The largest single disagreements after the
move directions are *WATER → we move* (about 8,000 held-out rows: the top
player waters the plant it stands on, we walk off), then HARVEST and
CARE the same way. That is the chores-per-visit gap (ours 1.5, theirs
2.0–2.2) stated per decision: the top players finish the tile; our sweep
re-decides every turn and leaves. The direction disagreements (NORTH vs
WEST etc.) are the routing gap and are not interpretable one row at a
time.

## 5. Take-over trace and the paired eval

One traced game (session 95488440, band 2000–2500; the dev tree with h8's
rules scores $75,875 against the recorded opponent's $109,372), the model
replacing the scheduler's unit action under the take-over rules of §2:

| take-over mode | confidence | our bank | opponent's bank | chores / visit | moves / productive op | productive ops | harvests | fertilize | weed tile-days |
|---|---|---|---|---|---|---|---|---|---|
| dev tree (no model) | — | $75,875 | $109,372 | 1.43 | 1.38 | 2,836 | 294 | 99 | 30 |
| chores only | 0 | $72,518 | $95,775 | 1.73 | **0.86** | 3,502 | 370 | 137 | 82 |
| chores only | 0.5 | $107,709 | $147,568 | 1.75 | 0.94 | 3,427 | 353 | 126 | 114 |
| chores and moves | 0.7 | $75,020 | $100,616 | 1.66 | 1.11 | 3,085 | 313 | 92 | 80 |
| chores and moves (4-epoch model) | 0 | $13,670 | $124,642 | 1.34 | 1.75 | 2,250 | 298 | 20 | 80 |
| full (moves, PLANT, PICKUP resolved) | 0.5 | $24,958 | $144,370 | 1.45 | 2.19 | 1,971 | 320 | 27 | 172 |

Two facts. First, the model must not steer the walking: whenever it
overrides moves the scheduler's routes to the shed and to the plan's
tiles break (provisioning collapses, feeds 77 a game in full mode) and the
game is lost by $100k. Second, letting it decide only *what to do where the
unit stands* is the first executor change that moves the measured gap
itself: moves per productive op 1.38 → 0.86 (the top agent 0.71), chores
per visit 1.43 → 1.73 (theirs 2.2), productive ops +23%, harvests +26%,
fertilise +38%, with a cost in dead plants (weed tile-days 30 → 82) and
idle tiles. Money on a single traced game is not a verdict (the
opponent's bank moves with the shop re-roll); the 8-epoch model in chores
mode at confidence 0 scores $77,204 on this game and $92,617 on the
band-1000–1500 game where the dev tree scores $62,026.

**Paired eval queued** (`rl/runs/draw995/run_execnet.sh`, detached, after
batch 2 and Fork B's batch 3 so it does not share the 5 procs; expected to
start about 12:00 and land about 13:15): arms `execnet_chores` (confidence
0), `execnet_chores_c50` (0.5) and `execnet_all_c70` (chores and moves,
0.7), against `dev_knobs_h8` and `champ_h8` reused from
`rl/runs/draw995/baselines.json`, output `rl/runs/draw995/execnet_eval.json`.
The bar is the same as for every rule: p < 0.05 on discordant games, four
of six bands, money in the same direction. Not shippable as is: the
policy needs torch at play time, so a numpy export (as `tools/export_ptrnet.py`
did for the pointer net) is required before any bundle.

## 6. Where this leaves the executor question

- The top players' per-unit decisions are 78% predictable from a 5×5
  window and a few counts, chores 88%; our scheduler reproduces 36% of
  them. The executor gap is not noise in their play, it is a policy we can
  read off.
- The half of the gap that is *what to do on the tile* transfers to our
  planner's world at least mechanically (the chores-only take-over above).
  The half that is *where to walk* does not transfer by replacement,
  because the walking serves the day plan and the provisioning that the
  scheduler owns; a movement model has to be trained with the plan's
  targets as inputs, or the plan has to be cloned with it (the top-player
  BC planner of research note §K is that other half).
- Next steps if the paired eval reads positive: numpy export for the
  bundle; add the plan's task targets (nearest task tile, its op) to the
  features and re-test movement take-over; train on the held-out draw's
  complement of all 7,600 3000+ games rather than 160.

---

# v2: the destination model (Maia-style vocabulary)

*2026-09-26. v1 modelled the op a unit takes THIS step, out of eighteen.
Steering movement with it was catastrophic (feeds 351 → 77) because a unit
re-decided its direction every step with no memory of where it was going.
v2 changes the vocabulary, not the plumbing: the model predicts a
**destination**, the way Maia predicts a from-square-to-square move, and the
scheduler does the walking.*

## Design

**Vocabulary (102 classes).** For each (step, unit) the label is where that
unit performs its **next tile op**: the tile index (0–99), `SHED` when its
next op is a shed op, or `STAY` when it acts on the tile it is standing on
or nothing happens before the day ends. Units respawn at the shed at hour 0,
so the day boundary ends the horizon. A second head predicts the op class at
that destination and reads 0.907 held-out.

**Candidate set = our own request map.** The legal destinations are the
tiles our `kagent.scheduler.build_tasks` has a pending task on, plus the
shed, plus stay. The model is therefore a **ranker over the tasks we
already generate**, not a free-form policy — the hybrid the record supports.
The design only works if the leaders actually go where we would consider
going, and they do: **only 0.6% of their destinations have no pending
request of ours** (1.1% on the broad set). We are not missing the work; we
choose differently.

**Features (33 planes of 10×10, our farm only).** The 24 tile channels of
v1, the unit's position, and 8 request planes — one per op class — carrying
the task's priority (a survival task reads 1.0, a filler task 0.5). Plus the
v1 scalar vector. A ~104k-parameter CNN: a 1×1 conv gives one logit per
tile, a pooled head gives the SHED and STAY logits, and the mask is applied
to the 102.

## Curation: train on seats that play like the leaders

A rating floor mixes in seats that win some other way, which blurs the
target. Instead every candidate seat is replayed once and kept only if it
clears the three leaders' executor profile: chores per visit ≥ 1.7, moves
per productive op ≤ 0.95, pickups per feed ≤ 0.7, wheat units per harvest
≥ 4.0 (the leaders read 1.9–2.2, 0.71–0.83, 0.46–0.55, 4.4–4.7).

| | candidates | kept | dropped |
|---|---|---|---|
| sessions rated ≥ 2900 | 260 | 150 | 110 |
| M & M & P & Q / Boey / DSM replays | 38 | 38 | 0 |
| **total** | **298** | **188** (1,351,873 samples) | **110** |

Of the 110 drops, **106 fail on wheat units per harvest alone** (mean 3.84
against the kept set's 4.21) — they do not water the crop on the last day of
its bonus window, the same defect our own executor had until this week. The
other three metrics barely separate the two groups (chores per visit 1.95
kept vs 1.95 dropped), so at 2900+ the *routing* style is near-universal and
only the *watering discipline* distinguishes the best seats.

## Diagnostic: agreement with our scheduler is FLAT across the ladder

For every sample we also record what our own scheduler would do, as a
three-way destination class (stay / shed / move). Recorded play against our
scheduler, by rating band:

| band | samples | agreement with our scheduler | recorded STAY share | our STAY share |
|---|---|---|---|---|
| 0–600 | 128,568 | 0.568 | 0.602 | 0.389 |
| 600–1000 | 155,844 | 0.593 | 0.602 | 0.433 |
| 1000–1500 | 289,419 | 0.584 | 0.564 | 0.345 |
| 1500–2000 | 290,904 | 0.590 | 0.567 | 0.343 |
| 2000–2500 | 294,359 | 0.577 | 0.571 | 0.346 |
| 2500+ | 292,740 | 0.591 | 0.577 | 0.360 |
| the three leaders | 261,084 | **0.622** | 0.610 | 0.406 |
| curated-profile seats | 1,081,999 | 0.592 | 0.594 | 0.356 |

**The agreement does not rise with rating**: it sits at 0.57–0.62 in every
band, including 0–600. And every band stands still about 60% of the time
while our scheduler stands still 34–43% of the time. Our executor is an
outlier against the *whole recorded population*, not just against the top.
Two consequences: a broad-band warm-up would **not** mostly re-teach our own
habits (the worry behind the conditioning design does not materialise), and
"walk less, work where you stand" is not an elite trick — it is the game's
normal play and we alone do not do it.

## Held-out fit (curated model, 23 held-out games, 164,808 samples)

| metric | model | our scheduler |
|---|---|---|
| destination accuracy | **0.714** | — |
| within one tile | 0.734 | — |
| stay / shed / move agreement | **0.793** | 0.597 |
| STAY precision / recall | 0.792 / 0.910 | — |
| SHED precision / recall | 0.727 / 0.617 | — |
| op head accuracy | 0.907 | — |
| exact tile, on tile labels only | 0.402 | — |

Confidence covers the decisions well: at 0.4 the model is asked on 91% of
rows at 0.752 accuracy, at 0.6 on 67% at 0.844, at 0.8 on 41% at 0.928.

**Where we disagree, in one number.** On the 98,616 held-out decisions where
a leader STAYS and works, our scheduler stays 50,695 times and **walks away
41,728 times**; the model stays 89,742 times. When a leader moves, we move
80% of the time. So the entire disagreement is the chores-per-visit gap,
decision by decision: they finish the tile, we leave it.

## Take-over on the three reference games

The scheduler keeps its logistics passes (shuttles, provisioning, morning
pickups, survival overrides) and all the walking; for every other unit the
model picks the destination. One guard was needed: a leader's STAY means
"you are already where the work is", but our units often stand on nothing,
and obeying STAY there simply idles them — ungated, a traced game fell to
$30.6k against a $62.0k base. STAY is now obeyed only when there IS work
underfoot.

| per game (b2 / b3 / b4) | dev tree | dest, ungated | dest, confidence 0.4 | leaders |
|---|---|---|---|---|
| money | 62.0 / 71.8 / 75.9k | 30.6 / 79.3 / 58.1k | **69.1 / 91.8 / 76.0k** | — |
| moves per productive op | 1.47 / 1.31 / 1.38 | 1.04 / 1.04 / 1.02 | **1.08 / 1.08 / 1.08** | 0.71–0.83 |
| chores per visit | 1.34 / 1.50 / 1.43 | 1.53 / 1.61 / 1.57 | 1.52 / 1.53 / 1.51 | 1.9–2.2 |
| productive ops | 2709 / 2979 / 2836 | 3316 / 3324 / 3315 | 3274 / 3279 / 3278 | — |
| harvests | 240 / 296 / 294 | 302 / 276 / 283 | 315 / 290 / 315 | 470–522 |
| weed tile-days | 26 / 53 / 30 | 73 / 104 / 102 | 74 / 71 / 52 | 3–61 |
| idle tile share | .114 / .136 / .138 | .177 / .189 / .196 | .177 / .155 / .180 | .037–.065 |

Gated, the take-over is at or above the base on all three games (+11%, +26%,
+0.1%) and moves the measured gap: **moves per productive op 1.31–1.47 →
1.08** and productive ops +12–21%. The cost is tidiness — weed tile-days
roughly double and the idle tile share rises — because the freed turns go
into chores and feeds rather than into keeping plants watered. Single-game
money is not a verdict (the shop draw re-rolls under pairing), which is why
both designs go to the 995-game instrument.

## The conditioned variant (Maia-2 style): all bands, profile as an input

The curated model sees only tidy farms, and at deployment ours is untidy
(13% idle tiles, 1.4 moves per op). The second design keeps **every**
candidate seat — 208 seats drawn 42 per rating band, all six bands, plus the
188 curated games, 347 games and 2.45M samples — and instead hands the model
the seat's own executor profile as a **conditioning input**: the four
curation metrics for that game, z-scored across the training seats and
**concatenated to the scalar feature vector** (35 scalars instead of 31).
At play time the vector is clamped to the leaders' value, so the model is
asked "act like a seat with this profile" on our states.

Elite clamp (mean of the 38 leader seats), raw → z against the population:
chores per visit 1.96 → +0.52, moves per productive op 0.785 → −0.81,
pickups per feed 0.492 → −0.41, wheat units per harvest 4.55 → +1.26.

| held-out | curated (23 games, 165k rows) | conditioned (48 games, 340k rows) |
|---|---|---|
| destination accuracy | **0.714** | 0.648 |
| stay / shed / move | **0.793** | 0.708 |
| our scheduler, same rows | 0.597 | 0.592 |
| op head | 0.907 | 0.883 |
| coverage / accuracy at confidence 0.4 | 91% / 0.752 | 86% / 0.707 |

The conditioned model fits *worse* — but its held-out set spans all six
bands, which is a harder and noisier target than the curated set's tidy
games, and it trained on 200k rows for 3 epochs against 300k for 4 (this box
has 6 cores and the paired eval holds 5). The interesting result is that the
ranking **reverses at deployment**:

| per game (b2 / b4) | dev tree | curated, conf 0.4 | conditioned, conf 0.4 |
|---|---|---|---|
| money | 62.0 / 75.9k | 69.1 / 76.0k | **85.4 / 96.2k** |
| moves per productive op | 1.47 / 1.38 | 1.08 / 1.08 | **1.04 / 0.99** |
| productive ops | 2709 / 2836 | 3274 / 3278 | **3315 / 3396** |
| harvests | 240 / 294 | 315 / 315 | 288 / **344** |
| weed tile-days | 26 / 30 | 74 / 52 | **57 / 54** |
| idle tile share | .114 / .138 | .177 / .180 | **.145 / .159** |

The conditioned model is better on money (+38% and +27% over the dev tree),
closer to the leaders' 0.71–0.83 moves per productive op, and — the point of
the design — noticeably **tidier** than the curated model, because it has
seen untidy farms and knows what to do on them. Held-out accuracy ranked the
two the other way, which is the project's recurring law once more: fit is
not the verdict.

## Queued

`rl/runs/draw995/run_execnet2.sh` (detached, waits for the land4 arm and for
the five procs to free): three arms on the 995-game draw against the
replayed `dev_knobs_h8` (0.313) and `champ_h8` (0.230) —
`execnet2_curated_c40`, `execnet2_cond_c40`, `execnet2_cond_c60`. The bar is
the usual one: p < 0.05, at least four of six bands, money in the same
direction. The arm to beat on this draw is now `ship2_hv` at 0.354, not the
dev tree.

## Which design to carry forward

**The conditioned one.** Three reasons, in order of weight. It takes over
better on our own states, which is the only thing that can reach the paired
gate. It keeps the whole corpus, so the training set grows with the corpus
instead of shrinking to the tidy tail (188 games against 347, and the
curation threw away 106 seats for one metric). And the band diagnostic
removes the reason to curate at all: agreement with our scheduler is flat at
0.57–0.62 across every band, so the low bands are **not** teaching the model
our habits — they are teaching it the same "stand and finish the tile"
behaviour the leaders show, in states that look like ours. Curation remains
useful as the source of the *clamp* (the elite profile vector), not as a
filter on the data.

The route stays the hybrid: heuristic routing and logistics, heuristic
planner, learned destination choice under a confidence gate. The next step
after the paired result, if it lands positive, is the disagreement gate —
act on the model only where it disagrees with the scheduler AND is
confident, which is where all the measured value sits (41,728 of 98,616
"leader stays, we walk" decisions).

---

## Hybrid: learned planner + learned executor

The two learned halves wired together, with nothing heuristic left but the
scheduler's logistics passes and the market controller: the **Kaggle-kernel
board net** (`rl/runs/kaggle_topboard/out/out/top_board_s1234/eval.json`,
BC on 971k top-player samples then 20 PPO iterations on the full corpus)
makes the day plan, and the **conditioned destination model**
(`rl/datasets/execnet2_cond/execnet2_cond.pt`) chooses chores and
destinations.

**Wiring.** `rl/execnet/policy.make_plan_provider(planner, rules, weights)`
returns the day-plan half: `"champion"` (the default, unchanged),
`"knobs"` (the knob bandit) or `"weights"` (a neural planner, temperature
0). `parse_arm_fields` is now shared by `rl/session_eval.py`,
`tools/game_trace.py` and `rl/rollout.py` so the three cannot drift, and it
tells a rules JSON from a planner-weights JSON *by content* (`"policy"` vs
`"kind"`/`"conditions"`), so every `dev:execnet:` arm string that worked
before parses to exactly what it parsed to before. A trunk-carrying planner
gets `provider.board` fed before every plan in
`rl/rollout._make_dev_policy_agent`, mirroring `_make_learner_agent`;
without it nnpolicy silently drops the board columns.

**Traces, the three reference games (mean).**

| per game | base (dev tree) | kernel planner + shipped scheduler | hybrid |
|---|---|---|---|
| final money | 69,893 | **82,363** | 66,175 |
| opponent money | 90,997 | 102,547 | 100,170 |
| moves per productive op | 1.39 | 1.46 | **1.10** |
| chores per visit | 1.42 | 1.41 | **1.57** |
| productive ops | 2,841 | 2,840 | **3,372** |
| harvests | 277 | 278 | **339** |
| plantings | 153 | 153 | **183** |
| feeds | 357 | 337 | 415 |
| fertilize ops | 116 | 92 | 79 |
| weed tile-days | 36 | 32 | 53 |
| idle share of tiles | 0.130 | 0.303 | 0.305 |
| BUILD_COOP / PLACE GOOSE / egg harvests | 0 / 0 / 0 | **0 / 0 / 0** | **0 / 0 / 0** |

Per game: base 62.0k / 71.8k / 75.9k, kernel planner 100.8k / 87.8k /
58.5k, hybrid 70.0k / 75.3k / 53.3k.

**Viable, but the executor half subtracts.** The hybrid is not
catastrophic (money 95% of base, feeds 116% of base), so it goes to the
paired instrument. But the planner alone is +18% on money while the hybrid
is −5%, and the executor half moves every executor metric the right way
while costing money — the fourth time in this project that a fitted model
improved its own mechanism and lost on the outcome. That is exactly why
the eval is decomposed into `hybrid` and `kernel_planner`.

**The planner will not buy a goose either, and it is not a masking bug.**
`goose_target` is a real head with values `[0, 8, 16, 24, 32]`, unmasked on
every day we checked, and the trained planner chose **0 on all 30 days of
all 3 games** while moving `cow_target` freely. Two things follow. The BC
labels do carry the leaders' geese (`rl/replay_bc._targets_to_choices`
reads the recorded seat's own end-of-day herd), so this is not the label
pipeline. But the head cannot express what the leaders actually run: they
keep 5–6 geese, and the first non-zero option is 8 — a 33–60% overshoot,
and 8 geese is the size that lost 8–12 points when it was tested. A head
whose only choices are "none" and "more than anyone runs" will learn
"none". Fixing the goose head's granularity is a prerequisite for any
learned planner to reach the egg line.

**Queued:** `rl/runs/draw995/run_hybrid.sh` — arms `hybrid` and
`kernel_planner` against `ship2_hv` (champ-h10, 0.354), with
`dev_knobs_h8` and `champ_h8` reused; out
`rl/runs/draw995/hybrid_eval.json`. It waits for batch 4 and for the five
processes to free, so it is last in the chain.


## Paired verdict, v1 (26 September 14:40, `docs/results/draw995_execnet.log`)

995 games, paired, against the dev tree (0.313) and champ-h10 (0.354).

| arm | wins / 995 | vs dev tree | vs champ-h10 |
|---|---|---|---|
| chores only, conf 0 | 209 | −10.3 pts (z −6.9) | −14.4 (z −9.2) |
| chores only, conf 0.5 | 231 | −8.0 (z −5.5) | −12.2 (z −8.0) |
| all ops, conf 0.7 | 97 | −21.5 (z −15.6) | −25.6 (z −17.6) |

The model is not a bad fit: 0.776 agreement with the top players where our
scheduler manages 0.355, and when it steers the chores it reproduces their
headline statistics (moves per productive op 1.38 → 0.86, chores per visit
1.43 → 1.73, harvests +26%). It still loses 10 to 14 points of win rate.
Per-op imitation of the executor is closed; see
`kaggriculture-learned-executor-refuted-again` in memory.


## Paired verdict, v2 destination model (26 September 16:44)

`docs/results/draw995_execnet2.log`, 995 games, vs the dev tree (0.313).

| arm | wins / 995 | delta | z |
|---|---|---|---|
| curated, conf 0.4 | 116 | −19.6 pts | −14.3 |
| conditioned, conf 0.4 | 175 | −13.7 | −9.9 |
| conditioned, conf 0.6 | 231 | −8.0 | −6.5 |

Two things are settled. The conditioned design beats the curated one by 6
to 12 points at matched confidence, which is the band-agreement diagnostic
paying off: training on every band and clamping the executor profile at
play time is right, and filtering the corpus by profile is wrong. And the
result is monotone in the confidence gate — the more of the executor the
model owns, the worse the agent gets, with the best arm the one that defers
most. Changing the action space from per-op to per-destination did not
rescue it, so the failure is not the vocabulary. The learned executor is
closed at both levels; the executor gains that survive are the two rules
that add production without moving labour at all.
