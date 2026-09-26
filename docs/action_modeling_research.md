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

(filled in below once the build lands)

## 4. Held-out fit and the disagreement with our scheduler

(filled in below)

## 5. Take-over trace and the paired eval

(filled in below)
