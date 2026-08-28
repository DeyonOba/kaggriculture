# Agent Version Documentation

Reference documentation for every Kaggriculture agent version: strategy, architecture,
training configuration, benchmarks, and the lessons each one taught. Quick history:
[CHANGELOG.md](../CHANGELOG.md). Research → design mapping: [rl/RESEARCH.md](../rl/RESEARCH.md).

All money figures are final bank (the game's score: only sold product counts). Arena numbers
are fresh-seed games, both player positions. Frozen bundles live in `arena/agents/<name>/main.py`.

---

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

### rl-v3 — generation 3 (g3 best, iter 19) · Kaggle 55829852 — CURRENT
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
