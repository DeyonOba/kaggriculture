# Modelling the world: hidden features, relationships, uncertainty

Written 2026-09-17, after a day in which **eight experiments were run and eight
were null or negative**. The value of that day is not the eight rejections, it
is what they say about *why* we cannot find anything — and that diagnosis sets
this programme.

---

## What today established, and what it implies

| finding | implication for this programme |
| --- | --- |
| per-world sd **$34,749**; paired $17,168 (corr 0.95) | the instrument, not the ideas, is the bottleneck |
| resolving a $3,000 effect at 50 evaluations needs **51,250 games** | we cannot afford to *measure*, let alone search |
| terminal money collapses **720 steps into one scalar** | we are discarding almost all the information each game produces |
| MELON is in no shop — found by *reading* `SHOPS` | hidden static structure is still there and still pays |
| three demand-side predictions all lost (geese, melon, tomato) | production **reallocates** fixed labour; the binding constraint is not modelled |
| the 24-game screen **anti-predicts** the ladder (ρ = −1.00) | our selection statistics are actively misleading |
| day 23 costs **−$4,623 in one day**, found by accident | discovery is currently luck, not process |

Two sentences summarise it. **We are measuring a $3k signal with a $35k ruler.**
And **we find things by accident.**

So the world model's first job is not to plan — it is to **make measurement
cheap and discovery systematic**. Planning comes after.

---

## Theme A — Model the world so measurement stops costing 50,000 games

### W1. Score a candidate by ΔV over the trajectory, not by terminal money

**The idea.** `tools/value_head.py` already fits V(s) = P(win | s) per day on
3,000 real games, and `rl/value_weights.json` is on disk. A game gives us 30
day-boundary states. Scoring a candidate by the **mean per-day ΔV against the
base on the same world** uses 30 observations per game instead of 1.

This is the evaluation analogue of an actor-critic baseline: the terminal
outcome is an unbiased but maximally noisy estimator of policy quality, and a
fitted value function is a lower-variance one.

**Free kill test, on data already on disk.** We have candidates with known
terminal paired money: `g-goose4` (−, gated), `g-tom9` (−1,171 ± 618),
`g-tom14` (−478 ± 726), `tape-271/162/277/112` (−8,523 / −8,277 / −7,871 /
−5,112 vs tape-129), and the four bake-off points. Compute each one's ΔV score
and correlate with its terminal paired money.

- If **corr > 0.7 and ΔV's se is materially lower**, every future experiment
  gets cheaper by the variance ratio. At the 3× that pairing gave, the 51,250-
  game search becomes ~6,000 games and runs locally in two hours.
- If corr is low, ΔV is measuring something else and the idea dies for $0.

**RESULT: REJECTED, 2026-09-17** — and the reason reshapes this whole document.

The V-score *is* much tighter. On the same games it separates candidates from
the base with |z| 2.5–7.2 against money's 0.6–2.1, a mean ratio of 4.8, which
would be **22.9× fewer games** for the same power. That half of the claim holds.

But it measures the wrong thing. Scored on LADDER worlds and compared with the
ladder money it must predict:

| arm | ladder money (n=270) | V diff | V \|z\| |
| --- | --- | --- | --- |
| tape-271 | −8,523 | **+0.0150** | 0.91 |
| tape-162 | −8,277 | **+0.0367** | 1.95 |
| tape-277b | −7,871 | −0.0284 | 1.35 |
| tape-112b | −5,112 | **+0.0550** | 3.09 |

**Three of four arms get a positive V-diff where the ladder says negative**,
and V is most confident (z = 3.09) about the arm that loses $5,112. A ranking
statistic that inverts the sign three times in four is unusable no matter how
tight it is.

(An earlier arena run against `pub-ahmed-v38` appeared to validate V at
r = +0.77. That run also said all four tapes BEAT tape-129 by $3.2k–$11.3k,
which the ladder contradicts — it was two estimators agreeing on the known
inverting instrument. Ground truth has to be the ladder.)

**Why it fails, and why that matters more than the experiment.** The head is
fitted on recorded games, so it is **correlational**. Features that co-occurred
with winning in the corpus — inventory, herd size, cash — carry positive
weight. An agent that accumulates those features without converting them scores
well on V and badly on money. Using a model fit on observational data to
predict the effect of an *intervention* is precisely where that breaks.

---

## The law this project keeps rediscovering

Three times today, in three unrelated places:

| model | observational fit | interventional result |
| --- | --- | --- |
| behaviour cloning | **69.4%** held-out action accuracy vs our rule's 27.9% | gated **NULL** over 4,800 paired games |
| market absorption table | correct about price and demand, verified against the engine | **three** rejections (geese, melon, tomato) |
| value head V(s) | fits win probability, and is 4.8× tighter as a statistic | **wrong sign on 3 of 4** arms |

**Observational fit does not transfer to interventional effect.** A descriptive
model of the world tells you what *co-occurs* with winning; it cannot tell you
what *causes* a win, and every one of these three was used to predict the
effect of a change.

This is now a standing constraint on the rest of this programme:

> **Any world model must be validated interventionally** — does it predict the
> sign and size of a *paired gate result* on held-out worlds? — never on
> held-out prediction accuracy. A model that predicts outcomes well and
> interventions badly is worse than no model, because it is confidently wrong.

It also means the variance problem has no modelling shortcut. Pairing works
*because* it is interventional: same world, same opponent, one thing changed.
That is why it is the only thing that has ever given a trustworthy number here.

### W2. Decompose the $34,749 — what is the irreducible floor?

Pairing removes the *world* term and leaves $17,168. What is the rest? Replay
the identical agent pair on the identical world with a **one-turn
perturbation** and measure how the outcome distribution spreads. That isolates
chaotic divergence from genuine policy difference.

**Why it matters:** if divergence alone is ~$17,000, then no modelling reduces
it, ~150-world evaluations are a hard floor, and we should stop designing
experiments that need more precision than that. If it is ~$5,000, the residual
is structured and W1/W3 can attack it. Either answer changes what we attempt.
~100 games.

### W3. Decisive-moment detector — make discovery a scan, not an accident

Day 23 (−$4,623 in one day) was found by reading a balance sheet for a
different purpose. Automate it: scan replays for the largest per-day |ΔV|,
aggregated across opponents and rating bins, and report the days and the state
deltas that move win probability most.

**Kill test.** Does it independently rediscover the two decisive days we
already know — day 10 (melon clear, +$12,078 to ahmed) and day 23? If it
cannot find known answers, it will not find unknown ones.

---

## Theme B — Relationships, stated mathematically

### M1. Re-solve the LP with the TRUE revenue curves

`docs/lp_bound.md` puts the ceiling at $147k, but it prices output linearly.
We now have the real revenue function: concave, per product, parameterised by
`T`, `above_func`, `above_target` — and melon's demand is **fixed at 1.0
unit/day forever** because it is in no shop. Revenue for selling *q* units is
∫ price(I₀+x) dx, not `q · base`.

Re-solve with (a) true concave revenue, (b) a labour constraint in
worker-actions per day, (c) the tile budget.

**Why this is the right shape for the three failed market experiments.** Geese,
melon and tomato each failed the same way: adding production *reallocates*
fixed labour. An LP with an explicit labour row prices that reallocation and
returns the optimal mix **and the shadow price of labour**. It is pure
computation — no games, no noise, no gate.

**Kill test.** Does the corrected LP reproduce champ-h8's actual product mix?
If it says our mix is already optimal under the labour constraint, the entire
market direction closes with a proof instead of three rejections — which is a
better outcome than three more rejections.

### M2. Elasticity map from the archive we already have

`tools/knob_archive.py` reads 30,272 paired counterfactual samples. That is
∂(margin)/∂(knob) for ~40 arms, free. Extend to *environment* parameters by
patching engine constants and re-running paired worlds. Produces the
quantitative version of `docs/market_structure.md`.

### M3. The shadow price of a worker-action, by day

The recurring explanation for every failure is "fixed capacity". Nobody has
priced it. From M1's dual, or empirically by granting one extra action on day
*d*. If labour's shadow price spikes on particular days, that is where to
spend — and it would explain day 23 directly.

---

## Theme C — Uncertainty, priced

### U1. Distributional payback, not mean payback

Every investment in the agent is a constant: 2 quadrants, land on day 5, $100
cash margin, 25% of cash for hires. No variance, no payback horizon, no
ahead/behind. For land #2, each animal and each hire, estimate the
**distribution** of payback from existing rollout and bandit data, then invest
when P(payback > cost) clears a threshold.

### U2. Make the threshold depend on ahead/behind

The value head's own rationale: ahead $30k on day 27 you liquidate and kill
variance; behind $30k you take the gamble. The two banks correlate +0.73, so
variance decides many games — and **nothing in the agent prices it**. Scale
risk appetite by V(s). This is the first genuinely risk-aware rule the agent
would have.

### U3. Use the opponent belief in a decision

`tools/opp_belief.py` solves their supply to ~5% through day 14 and has **never
been used in a decision**, only in analysis. The melon market is a two-player
fixed pot; the sell and plant decisions are exactly where their projected
supply belongs.

---

## Order, and why

1. **W1** — free, and multiplies everything downstream.
2. **W2** — ~100 games, tells us the precision floor and therefore which
   experiments are worth designing at all.
3. **M1** — free, pure computation, and either closes the market direction with
   a proof or points at the one mix we have not tried.
4. **W3** — cheap, and converts discovery from luck into a scan.
5. **U1/U2** — the first risk-aware decision rule; gate on the ladder with
   `tools/ladder_gate.py` (paired, McNemar).
6. **M2/M3/U3** as they become relevant.

## Standing rules this programme inherits

- Gate on the **ladder**, paired, with `tools/ladder_gate.py`. The 3-opponent
  panel inverts and the 24-game screen anti-predicts.
- Check `tools/knob_archive.py` **before** gating any knob.
- An absorption or demand table predicts **revenue, not wins** — three
  rejections on that point.
- Rating noise is ±45 points; per-world sd is $34,749. Size the measurement to
  the effect before running it, not after.
