# Changelog

All agent versions for the Kaggle **Kaggriculture** competition, newest first.
Detailed per-version documentation: [docs/VERSIONS.md](docs/VERSIONS.md).
Benchmarks are arena results on fresh seeds, both player positions, unless noted.

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
