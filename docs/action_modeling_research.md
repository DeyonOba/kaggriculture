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
