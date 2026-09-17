# Research review: the Maia-shaped policy, a trained reward, and what our budget allows

2026-09-17. Prompted by the thesis that the gem here is a *trainable reward
system* over a Maia-style board encoding — enumerate the legal moves, add the
board (tiles, crops, livestock, unlocked quadrants) and the context (day), and
let the model output a **distribution**, so uncertainty is represented rather
than argmaxed away.

This reviews the literature against our own measurements. Three of the four
sections change what I said earlier today.

---

## 1. The idea is right about the POLICY HEAD, and today's failure proves it

Maia's defining property is not that it predicts moves — it is **what it
predicts over**. Maia-1 encodes the board as spatial feature planes and outputs
a distribution over the *legal move space*; Maia-3's Chessformer uses board
squares as tokens with an **attention-based source→destination policy head**
([Maia-2](https://ar5iv.labs.arxiv.org/html/2409.20553),
[Maia-3](https://github.com/CSSLab/maia3),
[Maia KDD'20](https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf)).
The distribution is over *concrete alternatives in context*.

**Neither thing we built does that.**

| ours | output | what it cannot express |
| --- | --- | --- |
| `tools/exec_bc.py` | ranks ~60 tasks **our `build_tasks` proposed** | anything our planner did not think of |
| `tools/raw_bc.py` | **fixed 28-class** action type | *which tile* — it picks "WATER" then hunts for a legal one |

The Maia analogue for this game is a score over enumerated **(unit, op, tile)**
triples generated from *the engine's own legality guards*
(`kaggriculture.py:334-530`), softmaxed into a distribution. That removes the
planner bottleneck and makes the output genuinely probabilistic over the real
decision. **It is untried, and it is the correct critique of what I built.**

## 2. But accuracy is not our bottleneck — we already beat the state of the art

Maia-3 reaches **57.1% move-matching accuracy** on human chess moves, which is
SOTA with a quarter of the previous parameter count.

Our cluster clone, today: **75.5%** held-out action accuracy (14.0% baseline).

And it scores **$35** against champ-h8's $166,321.

So a better policy head would raise a number that is already higher than the
field's flagship result and is not the thing that is broken. What is broken is
the closed loop: at 75.5% per-action accuracy over 719 steps, the policy leaves
the champion state distribution within three days and never returns — it PASSes
on 85% of turns. Higher fidelity made it **more** brittle, not less (66.4% →
75.5% accuracy, $90 → $35).

## 3. The trained reward has a named failure mode, and we have hit it four times

This is the part of the thesis the literature is most pointed about.
[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
states it plainly: a reward model is an imperfect proxy, and optimising against
it too hard *degrades* ground-truth performance — Goodhart's law with a fitted
curve. Reward models are learned from data, so they overfit like any model, and
the policy then exploits the proxy's weaknesses.

We have already produced four instances of exactly this, with the fit and the
outcome moving in opposite directions:

| model | observational fit | interventional result |
| --- | --- | --- |
| BC task ranker | 69.4% accuracy (rule: 27.9%) | gated **null**, 4,800 paired games |
| market absorption table | correct on price and demand | **3 rejections** (geese, melon, tomato) |
| value head V(s) | **4.8× tighter** statistic than money | **wrong sign on 3 of 4** arms |
| raw-action BC (cluster) | **75.5%** accuracy | **$35** |

Our `tools/value_head.py` *is* a trained reward model, and it prefers agents the
ladder says lose. The literature's mitigation is
[uncertainty via ensembles](https://arxiv.org/pdf/2310.02743) — which is also
the user's own instinct about capturing uncertainty, arrived at independently.

**So the thesis is right that uncertainty matters, and the correction is that a
learned reward must be validated interventionally** — against the sign and size
of a paired gate — never on held-out prediction accuracy.

## 4. I was too broad when I said RL is unaffordable — that was model-free only

Earlier today I priced RL at 1e6–1e8 episodes and concluded 7.9–786 days. That
figure is **model-free PPO**, and using it to dismiss RL generally was wrong.

Two corrections:

**(a) Sample-efficient RL is measured in STEPS, not episodes.** The Atari-100k
benchmark is 100,000 environment *steps*, and transformer world models reach
strong play inside it ([IRIS](https://arxiv.org/pdf/2209.00588),
[TWM](https://arxiv.org/pdf/2303.07109)). Our game is 720 steps, so 100k steps
is **139 games** — about ten minutes of our compute. Different regime entirely.

**(b) We do not need a learned world model at all.** Dreamer/MuZero learn
dynamics because the simulator is unavailable or slow. **We have the exact
simulator**, at 4.08 s/game. That points at the AlphaZero family, where *search*
is the improvement operator and the network only has to learn the prior and the
value.

And our own record already endorses that family: the **only** thing in this
project that ever converted compute into a replicated out-of-sample gain was a
**rollout** — the hiring schedule, +$3,287 ± 1,528 over 192 fresh worlds. That
is search as a policy-improvement operator, done once, by hand, on one knob.

The directly relevant result is
[Gumbel AlphaZero](https://arxiv.org/pdf/2302.12359): standard AlphaZero
**does not guarantee policy improvement when the simulation budget is small**,
and Gumbel's modified root selection restores that guarantee. That is precisely
our constraint. [MCTS as regularized policy
optimization](http://proceedings.mlr.press/v119/grill20a/grill20a.pdf) gives the
theory for why the search distribution is a better training target than the raw
prior.

---

## What this implies, ranked

1. **Search, not imitation.** The improvement operator should be a rollout or a
   small-budget tree search over the exact simulator, with the network learning
   the prior and value. Gumbel-style root selection because our budget is small.
2. **The policy head should be Maia-shaped** — a distribution over enumerated
   legal (unit, op, tile) triples from the engine's guards, not a fixed class
   set and not our planner's proposals. This is the thesis's real contribution
   and it is untried.
3. **Any learned reward must be gated interventionally**, with ensemble
   uncertainty, because our one existing reward model already fails the sign
   test 3 times in 4.
4. **Evaluation stays the binding constraint.** Per-world sd $34,749; paired
   $17,168 at corr 0.95. Nothing above changes that, and search quality is
   invisible without it.

## The honest caveat on 13 days

Everything in §4 is a *build*, and the record is that builds here take longer
than the value they return. `tape-sel` at ~1690–1750 is the submission and
nothing this week has come near it. The defensible version of this programme is
a **narrow** one: a Gumbel-style rollout over the tape lineage's day-6 decision
(where `tape-sel` already branches on a fitted rule and where a per-world oracle
was measured at ~$18k/game above the best fixed tape), rather than a
policy-from-scratch. That is search applied to the one decision we know is
worth the most, on the agent we would actually ship.

## Sources

- [Maia-2: A Unified Model for Human-AI Alignment in Chess](https://ar5iv.labs.arxiv.org/html/2409.20553)
- [Maia-3 / Chessformer](https://github.com/CSSLab/maia3)
- [Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/pdf/2310.02743)
- [Targeted Search Control in AlphaZero (Gumbel Zero discussion)](https://arxiv.org/pdf/2302.12359)
- [Monte-Carlo tree search as regularized policy optimization](http://proceedings.mlr.press/v119/grill20a/grill20a.pdf)
- [Transformers are Sample-Efficient World Models (IRIS)](https://arxiv.org/pdf/2209.00588)
- [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109)
- [Action Guidance: Sparse and Shaped Rewards for RTS Games](https://arxiv.org/pdf/2010.03956)
- [Efficient Planning in Combinatorial Action Spaces](https://arxiv.org/pdf/2302.04376)
