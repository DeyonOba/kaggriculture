# rlv4 — a learnable agent, one testable stage at a time

*2026-09-30. Written after the readout → PPO line plateaued at −13 to −17 vs h11 on the 144-world paired instrument, with three decode bugs found by tracing single games. Everything below is anchored to a number we have measured; the papers are cited where they change a design choice.*

## 0. The ground rules every stage inherits

1. **The verdict is the paired gate** (`rl/session_eval.py`, 144 held-out tape worlds, 24 per rating band, McNemar on discordant games), never an in-loop evaluation. This week the in-loop "best" was the worst checkpoint on the gate, twice; the 1.00-vs-heuristic evals were against the legacy planner; every net arm before 2026-09-29 played without the four shipped executor rules.
2. **Every stage is a pure addition on h11** (`dev:rlv3:base` = the champion plan + `lastwater, h23, harvest_visit, shedguard` + the knob layer = 0.458 on the instrument, +25.7 over `main`'s champion). The control arm must equal h11 to the dollar (it does: 144/144).
3. **Trace one game before gating.** A switch that fires and moves nothing (pipeline, tend_end) and a switch that moves the wrong thing (pasture builds for unaffordable cows) were both caught in a 10-second trace, never by the gate.
4. **A learned target needs the planner's gate on its way to the board.** Twice this week a head that matched the leaders in aggregate lost on our farm until it went through the champion's own bound (hires −16 → −9; herd: 4–5 empty pastures gone).
5. **Watch the opponent's bank.** Every supply lever we have tried raised the opponent's bank more than ours (care_all +$1.3k ours / +$2.4k theirs; the readout +$7–11k theirs). The market is one shared inventory; "more produce" is a donation unless it out-runs their share.
6. **Size the gate to the effect.** 144 worlds resolve ±8 points at p<0.05; a +2–3 point lever needs ~1,000 games per arm (memory: gate size). Decide the gate size before the arm runs.

## 1. What is learnable here, and what the record says about each

| lever | measured fact | learnable form | status |
|---|---|---|---|
| **Planning schedule (3-day standing stock)** | v5 readout at the leaders' floor on S0; −18 vs h11 when applied; PPO +7.6 then plateau 0.31–0.33 | the 10 applied heads, decoded by `readout.commit` with the planner's gates | built; no-board only |
| **Board state** | CNN over the two-farm tensor beats ridge-on-counts +0.116/+0.137 R² on all 8 kinds (first model here to clear its own kill criterion); board arms 7× more seed-variable than no-board | `tools/board_tensor.py` planes (T_crop_103), `rl/pretrain_trunk.py` trunk on final margin, `rl/session_bc.py` corpus with boards | built, never gated with flags |
| **Hiring** | hands track *harvests* r=.764, not plantings r=.300; `CHAMP_HIRES` is a fixed 30-day schedule; the hires head with the planner's bound gates −9 | hires = f(harvest load due in 3 days, plants due, animals), fitted on the leaders | not built |
| **Sell timing** | engine fills unit by unit at the running price; town drains every 4th step; leaders realise 0.85–1.03 of the mean price to our 0.68–0.89; holding at a *fixed* floor lost $30k; the shed drop destroys 39–59 units/game; market price is an exact identity of supply | reservation price relative to the exact forecast, per product, learned as a finite-horizon MDP | `rlv3/selling.py` schedule exists, never validly gated |
| **Seed buying** | follows `plant_quota` through `market.build_orders` (60% of cash, cutoff table); 11.8 seeds stranded a game (fixed by `late_plant`, null on wins) | inherits the crop heads; no separate model | wired |
| **Melon / the early race** | melon is in no shop (fixed demand), sq-curve trap; the race: they clear 60 units on day 10 at $271, we finish day 13 at $97; burst harvesting lost $1,482; the planner already reads the opponent's melon commitment (`_projected_melons(view.opp)`) | minimal melon count given the opponent's *visible* day-1 commitment and our harvest capacity, chosen at CVaR | not built |
| **Risk per game** | `kagent/risk.py` CVaR on the exact drain lattice; `alpha_for_lead`; the readout tilt was structurally null (ordered, never moved quantities); PBRS on the lead HALVED wins (B4); win-first + pv potential is the best training objective (B2) | P(win \| state) head → posture → *quantities* (variance up when behind) | pieces built, never connected to quantities |
| **Opponent policy** | their board is fully visible; their pending supply is solvable from public market inventory (~5% error through day 14); congestion costs $52k/game (z 17.5); the leaders are learned agents that change week to week (4th quadrant 7.5%→32.5% in a week) | conditional model of *their* season from their visible opening; timing our sells around their harvests | belief model built, unused by any arm |
| **Executor** | movement optimised perfectly converts at ~$0; the four rules are the only executor changes that ever paid; CARE 6–10 of 15–16 fed animals a day (leaders 15–19 of 15–19) — `care_all` −4.2 n.s. with the donation signature | per-unit actions from the board (the Lux AI recipe) — the largest untested piece | not built; the earlier per-op imitation refutation ran flagless and is suspect |

## 2. The order you asked for, and why it is also the right order

**Stage 1 — Board recommendation for the planning schedule (warm-up).** The readout today sees 196 scalars; it cannot see *where* anything is, and the two farm-specific facts we keep hitting — labour at a distance, ground near the shed — are spatial. The board trunk exists and is pretrained on the one target that has passed anything (final margin, `rl/datasets/trunk_margin_top.pt`).

- Model: `PlannerNet(board=True)`: trunk (103×10×10 → embedding) ⊕ the 196 scalars → the 16 v5 heads; decode unchanged (`readout.commit` with the hires and herd bounds).
- Data: `rl/session_bc.py --heads-version 5 --horizon 3 --arm T_crop_103` on the top-band sessions (the corpus builder already emits the board planes), then `rlv3/finetune.py` on the newest harvest (`replays/now`, 26 games) — the loop that fixed the opening (0.320 → 0.191 on held-out games).
- Tests, in order: (a) S0 stage TVD ≤ the leaders' floor (offline; a board readout that is *farther* from the leaders than the scalar one is a bug); (b) `tools/bc_tile_error.py` held-out tile error vs the no-board readout; (c) `tools/net_ood.py` — train on the deployment mix; (d) the paired gate `dev:rlv3:readout:readout=<board.pt>` vs h11 with the no-board readout (`all3`, 0.278) reused, **three seeds** because board arms are 7× more seed-variable; success = better than `all3` at p<0.05 on the pooled seeds *and* the opponent's bank not up. Kill: if (a) fails, stop before (d).
- Literature: this is the standard imitation warm-start; the Lux AI Season-1 winner used a fully-convolutional policy with a frozen-teacher KL to stabilise self-play, and the top Kaggle sim-competition recipes are imitation from top replays + self-play + rules hybrids (see Sources). Our KL prior is a fixed 0.05; the offline-to-online literature adapts the BC weight on performance — here that signal must be the paired gate, not the in-loop eval.

*Stage 1 result (2026-09-30):* frozen margin trunk — tile error 1.12 vs the scalar readout's 1.11, null. Trunk trained end-to-end on the targets (`rlv3/finetune.py --train-trunk`) — tile error 1.01 (−9%, better on all nine heads); three-seed paired gate, 432 games: **+0.025 ± 0.022 over the scalar readout (z 1.15, p 0.29, same sign on every seed), opponent's bank −$3.5k, ours −$1.2k; −0.118 vs h11 (z −4.93)**. Not significant; the board copies the leaders better and the copied plan still loses through our executor. Stage 1 closed; the board goes to Stage 8. Found on the way: the readout's $400 dawn-cash reserve (`bound_buys`) zeroes every animal purchase on the leaders' sub-$400 dawns — S0 opening 0.572 for *both* readouts at this commit — a within-day purchase rule is a lever of its own.

**Stage 2 — Sell timing, contest stage.** "Don't sell plants at low prices during the contest" is a reservation-price rule, and it is learnable *because the market is deterministic*: price at any future step is an exact function of the inventory path, our sells are known, and the opponent's pending supply is solvable from public inventory through day 14. That makes selling a finite-horizon MDP per product — state (our stock, market inventory, days left, their solved supply), action (sell n now / hold), reward realised cash — the same structure as the perishable-goods pricing results, where the optimal reservation *falls* with inventory and time. Solve it by dynamic programming on the exact model first (no learning), then Q-learn only the residual against real opponents.

- Guardrails from the record: the shed cap (holding past it destroys units — `shedguard` is the only sell-side rule that ever paid), the final-day dump (the leaders request 2,842 units in hours 18–23 and end with 3.6 unsold; we end with 7.2), and the $30k loss of a *fixed* floor — the reservation must be relative to the forecast, never a constant.
- Tests: `tools/sell_metrics.py` realised-price ratio per product (theirs 0.85–1.03) on traces; then a pure-addition gate through the existing `rlv3/selling.py` seam (`sell="slots"` then `"full"`), with the opponent's bank in the readout. Kill: realised price up with our bank down.

**Stage 3 — Seed buying from the board's recommendation.** Nothing to build: seeds already follow `plant_quota` (bought this turn, plantable next, under the cutoff table). The test is that the crops head from Stage 1, decoded through `commit`, ends games with ≤ 1 stranded seed (the leaders: 0.7) — the endgame scorecard reads it directly.

**Stage 4 — Hiring fired by the harvest/plant load.** The lag is one day (hired at hour 0, working the same day), the load is *observed* (crop ages give the productions due in the next three days; the quota gives the plantings), so no filter is needed — a regression from the leaders' games: hands tomorrow = f(harvests due, plantings due, animals, cash). Apply through the planner's own bound (`bound_hires`). Tests: held-out R² vs the constant schedule; the paired gate with `apply=hires`; kill if the learned schedule is *lower* than `CHAMP_HIRES` in the contest (we are labour-short, hands/work 0.47–0.58 on days 11–15).

**Stage 5 — Melon: the minimal planting that does not lose the race.** By day 1 the opponent's melon commitment is on their board. Fit from replays: P(we clear before they do \| our n, their n, our hands) and the payoff at the drain, choose the smallest n whose CVaR_α margin is non-negative (`kagent/risk.py`). Test: the `apply=crops` gate restricted to the melon head; kill on the opponent's bank rising.

**Stage 6 — Risk per game.** The P(win) head is the value net PPO already trains; the posture is `alpha_for_lead`; what was never built is posture → *quantities*. When behind, widen the head distributions (temperature on the sampled plan) and shift toward the high-variance assets (the sq-curve products); when ahead, protect (annuities: wheat/egg). Train with the CVaR/distributional policy-gradient objective only after the deterministic posture gates positive; the B4 lesson is that a shaped potential becomes the objective at 20 iterations.

**Stage 7 — Intercepting the opponent.** The cheap version is a conditional table from replays: their visible day-0/1 opening → their season (what they plant, when they sell), and our sells timed around their harvests. Model-based opponent modelling (MBOM) that imagines improving opponents is for opponents that adapt within a game; ours are fixed policies within a game and change between weeks — the weekly re-harvest + fine-tune is the adaptation that matters.

**Stage 8 — The learnable executor.** The one thing every winning sim-competition agent has that we don't: unit actions come from the same network that sees the board. Our split (learned plan, hand executor) caps what imitation can transfer — the leaders' mix works because their executor extracts ~1,000 one-time units to our 388. The per-op imitation refutation (−10.3) ran with net arms flagless; it needs one re-gate before it counts.

## 3. The three suggested methods, placed honestly

- **MDP / Q-learning.** The day-level plan is an MDP with a 16-head action; over 261 logits Q-learning is factored per head and is not better than the PPO we run. Where Q-learning fits is Stage 2: a small discrete action (sell n / hold), an exact transition model, a finite horizon — solve by value iteration first, then Q-learn the residual. Reference: the finite-season perishable-goods pricing MDPs (Sources).
- **ADMM.** The allocation problem does decompose per asset with coupling constraints (cash, tiles, hands), which is what ADMM is for. But the LP-optimal allocation, executed by our scheduler, lost $31,728 and the leaders' own mix gated flat: the missing constraint is *extraction* — what our executor actually harvests per tile. ADMM is worth a run only with executor-calibrated extraction rates as constraints, as an offline planner arm through the existing `alloc` seam, and it is killed if it points where the LP pointed.
- **Kalman filter.** State estimation for hidden, noisy state. Here the market is an exact identity and the opponent's supply is solvable through day 14 — there is nothing to filter until the solution degrades. The right first step is a measurement: price-forecast error at horizon 3 vs the exact identity, by day; build an EKF over the opponent's pending supply only if the error after day 14 is material.

## 4. What starts now

Stage 1, commands as they will run (board corpus → board readout → S0 → tile error → OOD → three-seed gate):

```
PYTHONPATH=. uv run python rl/session_bc.py --heads-version 5 --horizon 3 --arm T_crop_103 --n 5000 --procs 4 --out rl/runs/headfix/bc_v5_h3_board.jsonl
KAGG_BOARD=1 PYTHONPATH=. uv run python rl/train.py --heads-version 5 --board --trunk-init rl/datasets/trunk_margin_top.pt --bc-session rl/runs/headfix/bc_v5_h3_board.jsonl --iters 0 --run rl/runs/headfix/readout_v5_board
PYTHONPATH=. uv run python rlv3/s0_check.py --preset readout --weights rl/runs/headfix/readout_v5_board/bc.pt --out rl/runs/headfix/s0_board.json
KAGG_BOARD=1 PYTHONPATH=. uv run python rl/session_eval.py --reuse h11=... --reuse all3=rl/runs/readout_buys.json:all3 --baseline h11 --arm board=dev:rlv3:readout:readout=rl/runs/headfix/readout_v5_board/bc.pt --per-bin 24 --procs 4 --seed 7   # then seeds 8, 9
```

## Sources

- Lux AI Season 1 solutions (fully-convolutional policy, frozen-teacher KL, imitation from top replays): https://www.kaggle.com/c/lux-ai-2021/discussion/278508 , https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 , https://github.com/shmyak-ai/lux-ai
- "Winning Kaggle's Reinforcement-Learning Competitions": https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio
- Adaptive Behavior Cloning Regularization for Stable Offline-to-Online RL: https://arxiv.org/abs/2210.13846 ; The Three Regimes of Offline-to-Online RL: https://arxiv.org/pdf/2510.01460
- Model-Based Opponent Modeling: https://arxiv.org/abs/2108.01843 ; Deep MARL challenges (opponent modelling survey): https://arxiv.org/pdf/2106.15691
- Risk-sensitive RL via policy gradient (CVaR): https://arxiv.org/pdf/1810.09126 ; distributional CVaR policy gradients with convergence: https://arxiv.org/html/2405.14749v1 ; risk-constrained RL with percentile criteria: https://jmlr.org/papers/volume18/15-636/15-636.pdf
- Dynamic pricing for perishable goods as a finite-season MDP / Q-learning: https://ieeexplore.ieee.org/document/4680975/ ; deep RL for pricing and ordering of perishables: https://doi.org/10.3390/app15052421
- Risk-averse stochastic optimisation (CVaR) for farm management: https://arxiv.org/pdf/2208.04840
- ADMM for distributed resource allocation (survey): https://arxiv.org/pdf/2401.15607 ; multi-farmer crop planning under uncertainty: https://www.sciencedirect.com/science/article/abs/pii/S036083522030718X
- Kalman filtering for demand/inventory estimation: https://www.sciencedirect.com/science/article/abs/pii/S0278612520301205 ; EKF parameter estimation in supply chains: https://doi.org/10.3390/math14162906
- RL crop-management environments (gym-DSSAT, CropGym) — agronomy, not markets; cited for what the field has, not for a design choice: https://openreview.net/pdf?id=4vNeWNQc3U , https://arxiv.org/abs/2104.04326
