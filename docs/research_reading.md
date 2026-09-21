# Research reading list

External papers surfaced by the 2026-09-16 research workflow (4 parallel agents:
tape splicing, policy collapse, probabilistic modelling, architectures). Internal
file references are stripped; only outside sources are listed, each with the
reason it bears on this project.

Ordered by how directly it touches something we have measured.

---

## 1. Trajectory stitching — the splice idea, done properly

The workflow measured that our 58-tape cluster is a **trie**: field actions are
byte-identical through step 151, all 23 branch events fall between steps 152 and
324, and from day 14 the tapes are permanently separated. So splicing at step 144
is a tautology (it just *is* tape B), and only **cross-branch** splices in days
7–12 create anything new. These papers are the literature for doing that.

- **Hepburn & Montana, *Model-based Trajectory Stitching for Improved Offline
  RL*** — [arXiv:2211.11603](https://arxiv.org/abs/2211.11603), and the follow-up
  [arXiv:2212.04280](https://arxiv.org/abs/2212.04280)
  Stitch where a probabilistic state-**reachability** test passes and a value
  estimate improves. We have an exact simulator, so our reachability test can be
  exact rather than probabilistic — a strictly stronger position than the paper's.

- **Li et al., *DiffStitch*** — [arXiv:2402.02439](https://arxiv.org/abs/2402.02439)
  A "qualification module" rejects a stitched trajectory when any predicted state
  deviates from the dynamics model by more than δ. Our zero-tile-mismatch rule is
  the exact analogue, measured instead of learned.

- **Eysenbach, Salakhutdinov & Levine, *Search on the Replay Buffer***
  — [arXiv:1906.05253](https://arxiv.org/abs/1906.05253) (NeurIPS 2019)
  Build a graph whose nodes are observed states and whose edges are reachability,
  then plan by graph search. Our days-7–12 edge graph is literally this graph.

- **Wang et al., *Robust Offline Imitation Learning Through State-level
  Trajectory Stitching*** — [arXiv:2503.22524](https://arxiv.org/abs/2503.22524)
  Learns a state-**similarity metric** to decide which state-action pairs may be
  stitched. The learned version of our hand-written tile signature, which we know
  has false positives (a 1-tile mismatch cost $12–13k while 0-tile cost under $400).

- ***Graph-Assisted Stitching for Offline Hierarchical RL***
  — [arXiv:2506.07744](https://arxiv.org/abs/2506.07744)
  Builds the stitch graph in a state/latent space rather than a trajectory space.

- **Brandfonbrener et al., *When Does Return-Conditioned Supervised Learning Work
  for Offline RL?*** (2022) — [arXiv:2206.01079](https://arxiv.org/abs/2206.01079)
  RCSL / Decision Transformer **provably cannot stitch** in general; it needs
  near-determinism plus coverage. Read this before hoping a sequence model will
  stitch for you — it has to be done explicitly.

- **Wu, Wang & Hamaya, *Elastic Decision Transformer***
  — [arXiv:2307.02484](https://arxiv.org/abs/2307.02484)
  Stitching performed **at test time** by shortening the conditioning history
  when the trajectory so far is sub-optimal. The direct analogue of a late,
  state-conditioned splice.

- **Buro, *Toward Opening Book Learning*** (ICGA 1999), and **Chaslot et al.,
  *Meta Monte-Carlo Tree Search for Automatic Opening Book Generation*** (IJCAI'09)
  Grow a book by expanding the leaf with the best measured outcome. Our route DAG
  *is* an opening book, and this is its growth rule.

---

## 2. Why RL-from-BC collapses — the "melon farmer" problem

- **Wolczyk et al.** — [arXiv:2402.02868](https://arxiv.org/abs/2402.02868)
  Forgetting happens on the states the student **stops visiting**. This is the
  sharpest framing of the collapse: it is a coverage problem, not a KL problem.

- **Policy-constraint / improvement mismatch in TD3+BC**
  — [arXiv:2211.11802](https://arxiv.org/abs/2211.11802)
  Why a behaviour-cloning penalty on the *action distribution* fails to prevent
  drift in the *trajectory* distribution.

- **Adaptive BC regularization** — [arXiv:2210.13846](https://arxiv.org/abs/2210.13846)

- **Schmitt et al., *Kickstarting Deep RL***
  — [arXiv:1803.03835](https://arxiv.org/abs/1803.03835)
  Teacher-driven warm start with a decaying auxiliary loss.

- **Entropy collapse / target-entropy control**
  — [arXiv:2605.11491](https://arxiv.org/html/2605.11491v1)
  Target-entropy control beats a fixed entropy bonus — relevant to a policy that
  converges onto one crop.

- **Perolat & Leibo et al., common-pool resource appropriation**
  — [arXiv:1707.06600](https://arxiv.org/abs/1707.06600)
  Multi-agent over-appropriation of a shared resource. This is our price ladder:
  both farms sell into one pool, which is why monoculture is an attractor and why
  the LP says solo $255k vs mirrored $147k.

---

## 3. Objectives, evaluation, and the traps we keep hitting

- **AlphaStar Unplugged** — [arXiv:2308.03526](https://arxiv.org/abs/2308.03526)
  Two things for us: evaluate by **min win rate over a reference pool** (our
  worst-opponent objective), and the finding that sophisticated offline RL
  *underperformed* plain BC + fine-tuning on a far larger corpus than ours.

- **Kumar, Hong, Singh & Levine, *When Should We Prefer Offline RL Over
  Behavioral Cloning?*** — [arXiv:2204.05618](https://arxiv.org/abs/2204.05618)
  On the mismatch between the statistic you optimise and the statistic you are
  scored on — exactly our margin-vs-win-rate problem.

- **Berman, Zhang & Zhao, *Valuing Winners: When and How to Correct for Selection
  Bias in Randomized Experiments***
  — [arXiv:2605.18887](https://arxiv.org/abs/2605.18887)
  The maximum over noisy arms is upward-biased **even when all true effects are
  equal**. We demonstrated this live: tape 271 screened at 83.3% (n=24) and
  confirmed at 41.9% (n=360). Read this before trusting any screen.

- **Karwowski et al., *Goodharting***
  — [arXiv:2310.09144](https://arxiv.org/pdf/2310.09144)
  A proxy improves, then **reverses** past a threshold. Our margin proxy did
  exactly this.

- **Lilian Weng, reward hacking**
  — https://lilianweng.github.io/posts/2024-11-28-reward-hacking/

- **PEX — freeze the offline policy, expand rather than overwrite**
  — [arXiv:2302.00935](https://arxiv.org/abs/2302.00935)
  Keeps the BC policy intact and adds to it, instead of fine-tuning it away.

---

## 4. From the earlier RL plan (still unread, still relevant)

- **Cobbe et al., Procgen** — [arXiv:1912.01588](https://arxiv.org/pdf/1912.01588)
  ~10k levels needed to close the generalisation gap.
- **Expert Iteration (ExIt)** — search as the improvement operator, supervised
  learning to internalise it.
- **Potential-based reward shaping** — [arXiv:1106.5267](https://arxiv.org/pdf/1106.5267)
  Provably preserves the optimal policy.
- **Distributionally robust RL** — [arXiv:1902.08708](https://arxiv.org/pdf/1902.08708)
  Worst-case over an uncertainty set; validated here (selector −$2,852 → +$1,014).
- **gym-invmgmt** / **MARLIM** — [arXiv:2308.01649](https://arxiv.org/pdf/2308.01649)
  Rule-based warm start; RL ≈ good heuristics on single-echelon inventory.

---

## 5. Routing (closed, but for completeness)

- **Multi-robot task allocation, Hungarian vs greedy**
  — [arXiv:2303.04894](https://arxiv.org/pdf/2303.04894)
  Greedy ≈ 0.93× Hungarian in practice. We implemented this (`match-all`); it was
  the session's only marginal ship and rated inside the noise.
- **Anticipatory / regret-based stochastic VRP**
  — [arXiv:1502.01972](https://arxiv.org/pdf/1502.01972) — never implemented.
- **Maia-2, *A Unified Model for Human-AI Alignment in Chess***
  — [arXiv:2409.20553](https://arxiv.org/abs/2409.20553)
  Conditions on the **opponent's skill level** as a model input, and its value
  head predicts the game **outcome** rather than a score. Both match where this
  project ended up; the auxiliary-head trick (legal moves as multi-hot plus
  structured sub-labels) is the part we have not tried.

---

## Reading note

The three papers most likely to generate a *new* idea rather than confirm an old
one: **Search on the Replay Buffer** (1906.05253) for state-collision mining over
all 21,517 sessions, **Brandfonbrener** (2206.01079) for why sequence models
won't rescue us, and **Berman** (2605.18887) for why our screens keep lying.

---

## 6. Spatial world models and auxiliary signal — the line we have NOT passed

Named 2026-09-21. Sections 1–5 cover stitching, RL-from-BC collapse,
objectives, the earlier RL plan and routing. The board-state encoder / board
CNN line (`docs/spatial_program.md`, `architecture_board_state.md`,
`board_state_plan.md`, `capacity_channels_killed.md`) was written up in method
detail with no citation anywhere; these are the papers behind it.

The intended arc is **BC → RL → a Maia-style spatial policy**: a *distribution*
over board representations that guides play, rather than a scalar head over
aggregates. We have never got past the first arrow.

- **Jaderberg et al., *Reinforcement Learning with Unsupervised Auxiliary
  Tasks* (UNREAL)** — [arXiv:1611.05397](https://arxiv.org/pdf/1611.05397).
  Auxiliary heads (pixel/feature control, reward prediction, value replay)
  densify a sparse reward into many per-step signals sharing one encoder.
  **Why it fits here:** our reward is one number at step 719 against a
  per-world sd of $34,749 — the sparsest possible signal, and the measured
  reason residual PPO collapsed (whole-episode credit over ~60 deviations).
  The auxiliary targets are already built and already validated: the board
  CNN predicts all eight standing-count kinds at +0.116/+0.137 R² over
  ridge, and `tools/price_forecast.py` shows the market is an exact identity.
  **What to watch:** UNREAL's claim is sample-efficiency on a shared encoder,
  not a better policy. `docs/capacity_channels_killed.md` is the local
  cautionary case — +0.0187 flow R² at p = 0.0001 bought 0% forecast MAE and
  −44% out-of-distribution.

- **Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2)**
  — [arXiv:2010.02193](https://arxiv.org/pdf/2010.02193). Categorical latents,
  policy trained entirely in imagination inside the learned model.
  **Why it fits here:** the expensive thing in this project is the rollout —
  every gate is whole-season re-simulation at ~1.3 s a game because
  `Environment(steps=…)` cannot faithfully resume mid-game. A world model
  good enough to imagine days 10→30 would turn a 30-minute gate into a
  minute. **What to watch:** the engine is already an exact simulator; a
  learned model can only be *cheaper*, never more accurate. So the gain is
  compute, and it has to be validated against the real engine on a paired
  gate — `tools/fast_env.py` already bought 1.70× bit-identically, which is
  the honest baseline any learned model must beat by more than its own error.

- **Maia-2 / the Maia line** — [arXiv:2409.20553](https://arxiv.org/abs/2409.20553),
  already cited in section 3 for skill-conditioning. The unrealised piece is
  the *spatial* one: a distribution over board representations conditioned on
  the opponent's skill, rather than a value head over aggregates.

### The specific use case: what is the opponent short of?

Both farms sell into one `market["inventory"]`, and that single shared
variable is worth **$52,652 a game** (`kaggriculture-market-congestion`).
Their board is fully visible; their shed is not, and is worth 1.5% even as an
oracle. So "what are they lacking, given the shop draw and their board" is
computable from public state, and it is the highest-value unexploited signal
we have measured.

**But the simplest version of it already gated null.** `k-diff5` —
`rent − λ × (their visible holding of that product)`, the direct
game-theoretic contest proxy — scored **+$362 ± 1,136 (z = 0.32)** against
champ-h8, while the plain rent rule it sits on is +$4,336 (z = 6.85). Pricing
a tile at its own harvest-day price already absorbs most of what the
opponent's board would tell you, because their supply lands in the same
inventory the forecast uses.

That is the bar any spatial opponent model has to clear: **not "does it
predict their board" — the board CNN already does that at +0.116 R² — but
"does it beat `k-diff5`'s null after the rent rule has taken its share".**

### Workflow this line must follow

    spec -> research (papers) -> use-case ideas -> experiment
         -> evaluate/test -> commit -> train -> test (over baseline)

The two steps this project has historically skipped are **evaluate/test
before train** and **test over baseline** at the end. The standing rules:
gate paired on identical worlds with `tools/ladder_gate.py`, read **win
rate** and not margin, replicate on a second independent block, and treat a
capability or accuracy score as a diagnostic only — `p1-mel100` moved
capability +50pp and money −$14,897 on the same change.
