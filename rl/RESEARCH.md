# Research notes → RL design decisions

Sources reviewed before building the learning policy (2026-08-27):

1. **AlphaStar Unplugged: Large-Scale Offline RL** (Mathieu et al.) — https://arxiv.org/html/2308.03526v1
   - All their successful approaches are *one-step* offline methods anchored to behavior cloning;
     fancy offline RL (BCQ, CRR, return-conditioned BC) underperformed plain BC + fine-tuning.
   - Autoregressive/decomposed action heads; evaluate against a *pool* of reference opponents
     (robustness = min win rate), not a single baseline.
   - → We BC-initialize from the strongest scripted agent (v3) before any RL, and evaluate against
     {v3, starter, past snapshots}.

2. **RAISocketAI — first DRL winner of the IEEE microRTS competition** — https://arxiv.org/html/2402.08112v1
   - Multi-discrete decomposed action space with invalid-action masking (logits to -1e9).
   - Shaped→sparse reward *curriculum*; self-play then fine-tuning vs scripted bots; <1M-param nets
     are competitive; competition losses traced to inference timeouts.
   - → Masked multi-discrete heads; portfolio-delta shaping annealed to sparse final-money+win;
     tiny MLP (~20k params) that runs in pure Python well under the 1s/step budget.

3. **A Closer Look at Invalid Action Masking in Policy Gradient Algorithms** (Huang & Ontañón) —
   https://arxiv.org/abs/2006.14171
   - Masking invalid actions *before* the softmax yields unbiased policy gradients and much better
     sample efficiency than penalizing invalid actions.
   - → `features.masks()` is computed identically at train and inference time; BC targets are
     projected onto the mask.

4. **The AI Economist** (Zheng et al.) — https://arxiv.org/abs/2108.02755
   - PPO trains economically rational agents in market simulations with endogenous prices; agents
     maximize money/utility. Two-level (leader/follower) structure works.
   - → PPO as the algorithm; reward = what the game scores (final bank) plus a win bonus, since the
     leaderboard is decided by head-to-head wins.

5. **Hierarchical RL / action abstraction** (survey material; e.g. https://arxiv.org/pdf/2002.02080)
   - A high-level policy over temporally extended macro-actions/scripted skills collapses the
     horizon and improves credit assignment.
   - → The network decides once per day (30 decisions/episode instead of 720×17 unit micro-actions):
     standing targets for crops/herd, hiring, land, melon sell floor. The proven scripted
     scheduler/market controller executes the micro. Safety clamps (feed capacity, budgets,
     planting windows, liquidation) stay scripted so no sampled action can wreck an episode.

Also consulted: Kaggle simulation-competition writeups (Lux AI S1 winner used IMPALA-style
actor-critic with shaped→sparse curriculum, https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021;
rules-based agents won several seasons, which is why the scripted executor stays in the loop).

## What the agent "knows" and optimizes

- **Rules**: encoded twice — hard rules in the engine-mirrored constants/masks/safety clamps
  (planting windows, feed survival, budgets, the 10-order market cap), and soft dynamics learned
  from features (market inventory displacement vs T per product, prices vs base, town shop counts,
  opponent's public farm state).
- **Maximized**: terminal bank (the game's score) + win bonus (the leaderboard's currency), with
  annealed portfolio-value shaping only to ease early credit assignment.
- **Minimized** implicitly: dead capital (unsold stock at the horizon), animal escapes, and
  oversupply into crashed markets — all visible to the net through its features and punished by
  the terminal objective.
