# Experiment trajectory

Written 2026-09-17. ~13 days to the deadline.

**Rule: one idea, one branch, run to exhaustion. Do not move on until the idea is
refuted as a class — not when one variant failed.** The lines that were followed
through produced findings; the sixteen executor mechanisms that got one shot each
produced nothing.

Process: `.claude/skills/experiment-loop` (Spec → Research → Implement →
Evaluate → Critic → Submit → Loop), with `build-variant`, `gate-candidate` and
`replay-autopsy` for the individual steps.

---

## Where we actually are

| | rating |
| --- | --- |
| tape-sel | **1750.3** (re-submit of identical bytes: 1706.1) |
| tape-sel3 | 1671.9 |
| tape-v2 | 1521.5 |
| tape-112 | 1330.6 |
| champ-h8 / match-all | 810.4 / 819.0 |

Two lineages: **tapes ~1500–1750**, **executor ~810**. Nothing has moved either
in a week.

### What is settled

- The executor's task MIX and priority bands already match a rated-3,093 agent
  (band shares within ~3pp; our planner proposes their work, ABSENT only 6.6%).
- Raising throughput loses money — four independent attempts.
- Twelve routing mechanisms: null. A learned ranker at 69.4% vs our rule's
  27.9%: null over 4,800 paired games.
- Allocation is flat (LP-optimal allocation ties champ-h8). Sell floors inert.
- Hiring already sits at the champion median (4.2 opening / 11.2 late), and the
  LP independently converges on the same 2 quadrants and ~10 hands.

### What is NOT settled, despite my earlier wording

`docs/executor_problems.md` lists two named variants that have **never been
attempted**. I called the executor a dead end while those were outstanding; that
was premature.

---

## The queue

Each entry: branch, the falsifiable claim, the kill test, and what would close it.

### E1 — `exec-commit` : cross-turn worker commitment

**Claim.** A worker that commits to a target across turns reaches 2.39 → ≤1.9
moves/tile without lowering average action value.

**Why it is not the refuted work.** Every variant so far re-decides every turn;
`pick` re-chooses as other units claim, so a unit zig-zags. Zone assignment is
already sticky (`unit_index % n_zones`), but the TASK is not, and no variant has
carried per-worker state. The dawn-tour attempt committed to a *route* computed
at hour 0; this commits to a *target* and re-plans only when it is taken or
completed.

**Watch out.** Forcing the partition to a fixed size destroyed the agent
(`z-stable`: ladder win 0.20 → 0.07 at 2500+), because the partition tracking the
live crew is what lets the lone morning worker cover ground. Commitment must not
freeze the *partition*.

**Kill test.** Structural first: moves/tile and actions/tile on 4 seeds. If
moves/tile does not drop below 2.2, stop. Then ladder money, n=40/bin.

### E2 — `exec-joint` : change planner and executor together

**Claim.** The executor is co-adapted to our planner's layout, so the pair moves
where neither moves alone.

**Evidence it is real.** On a foreign farm champ-h8's work share collapses
40.5% → 12.4% — and that farm is not task-poor (our `build_tasks` finds MORE
tasks there, 98 vs 66 at day 10). This is the mechanical reason every executor
change in isolation gates flat.

**Cheapest form.** Take the crop/herd targets from a tape's recorded farm (the
planner side) and run our executor on them; or the reverse. Both directions are
one variant each.

**Kill test.** Ladder money vs champ-h8. If neither direction moves it, the
coupling hypothesis is refuted and the executor closes for real.

### E3 — `tape-cohort` : finish cohort 3017.2

**Claim.** More arms from the one cohort that transfers beat tape-sel.

**Status.** 243 arms screened at n=24; 7 confirmed at n=360; two replicated
(112 at 70.8%, 277 at 69.6%) against the shipped arm's 61.5%. **tape-112 then
rated 1330.6** — the panel inverted.

**What changes.** Re-rank the whole screen by **ladder money**, not panel win
rate. The screen data already exists (`rl/datasets/tapes/unscreened_winrate.json`);
only the top candidates need re-measuring on the ladder. Cheap, and it directly
tests whether the new instrument would have avoided the tape-112 mistake.

### E4 — `tape-stitch` : cross-branch routes, days 7–12

**Claim.** A route assembled from segments of several tapes beats any single tape
on average.

**Sharpened by measurement.** The 58-tape cluster is a **trie** — byte-identical
through step 151, 23 branch events in steps 152–324. So splicing at step 144 is a
tautology (it *is* tape B), and only **cross-branch** splices create new scripts.
378 zero-mismatch cross-branch edges were found from sampling 14 of 719 steps,
yielding 163 scripts that are none of the 58.

**Why it survives the selector null.** That null killed per-world *selection*
(oracle +$41,666 vs shuffled null +$41,951 ± 745). A route is ONE fixed script
played in every world — a main effect, with no interaction to noise-fit.

**Kill test.** Enumerate zero-mismatch edges over all 144 steps of days 7–12 and
count DISTINCT resulting scripts. Under ~200, the space is too small; costs
one 58-game snapshot pass.

### E5 — `instrument` : make the ladder trustworthy

**Claim.** Ladder money predicts rating well enough to gate on.

**Status.** Spearman **+0.94** over 6 scored submissions vs the panel's inversion.
But its opponents are **replayed scripts that cannot react**, which is why win
rates sit at 78–87% for agents rated 1330–1750, and why it compresses.

**Work.** Add live public agents to the pool; weight bins toward 2000+; re-check
Spearman as new submissions score. This is infrastructure — it multiplies every
other line and is the cheapest thing on this list.


### E6 — `risk-value` : price uncertainty instead of ignoring it

**Claim.** A risk-aware objective picks better arms and better investments than
the mean does.

**Already proven, free, on data on disk.** Rank the 58 panel arms on half the
worlds, score the pick on the other half, 40 random splits:

| ranked by | held-out win | worst-opp win | held-out margin |
| --- | --- | --- | --- |
| mean margin | **0.484** | **0.037** | **+18,006** |
| win rate | 0.630 | 0.154 | +8,077 |
| Phi(mu/sd) | 0.619 | 0.108 | +9,417 |
| **mu − 0.5·sd** | **0.640** | 0.118 | +8,808 |

Ranking by mean gets the HIGHEST held-out margin and the LOWEST win rate — it
reliably selects an exploiter (worst-opponent win 0.037). Penalising variance is
worth **+15.6pp of held-out win rate**. Mean and spread are near-independent
(corr −0.18), so spread is a separate lever we have never pulled.

**The agent has no uncertainty terms at all.** Every investment is a constant:
2 quadrants, day 5, $100 margin, 25% of cash for hires, 60% for seeds. No
variance, no payback horizon, no ahead/behind, no opponent state.

**What to build.** `rl/value_weights.json` is a fitted `P(win | state, day)` over
56 features from 3,000 games, and **no shipped agent uses it**. It was tested as
a SEARCH objective (comparing agents) and lost to margin, |t| 1.86 vs 7.64 —
because squashing to [0,1] compresses between-agent tails. That does not refute
it as a WITHIN-GAME decision signal, which is what it was built for:

    P(win) < 0.5  -> take variance   (buy SE, over-hire, volatile crop)
    P(win) > 0.5  -> kill variance   (liquidate early, hold the safe line)

We currently buy SE never, which is right on average (the LP agrees: labour binds
29/30 days) and wrong exactly when behind — ~40% of games at 1750, ~96% at 810.

**Kill test.** Re-rank the existing 243-arm screen by `mu − 0.5·sd` and check
whether the arms it picks beat the mean-ranked picks on the LADDER. Free; the
screen data is already on disk.

### E7 — `endgame` : the tape's only sustained loss is days 26-29

**Claim.** A reactive endgame layer over the last ~4 days recovers the ~0.2 of
win probability the tape gives back.

**Evidence** (`tools/value_head.py attribute`, P(win) by day):

| day | 7 | 9 | 12 | 13 | 17 | 21 | 26 | 27 | 29 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tape-sel (1750) | 0.535 | 0.583 | 0.340 | **0.649** | 0.535 | 0.572 | 0.525 | **0.326** | 0.334 |
| champ-h8 (810) | **0.511** | 0.372 | 0.244 | 0.339 | 0.084 | 0.002 | 0.000 | 0.000 | 0.000 |

Two different shapes. **tape-sel dips and RECOVERS** (0.340 → 0.649 over days
12-13) and holds ~0.5 for twenty-six days; its worst day is **26 (−0.199)** and
its steepest fall **25→27 (−0.207)**. **champ-h8 peaks at day 7 at 0.511 — an
even game — then decays monotonically and is dead by day 19.**

So the opening is not our problem, and for the tape lineage the endgame is.

**Why it is not the refuted hybrid.** Tape→our-executor handoffs were tried at
days **6, 10 and 14** and refuted — exactly where the value head says the tape is
STRONGEST (0.5-0.65). Those handoffs replaced good play with worse. **Day 26 was
never tried**, and it is the only region where the tape declines. ~96 turns.

**Kill test.** Hand off at day 26 to champ-h8's executor and to a liquidation-only
controller; ladder money vs tape-sel. If neither beats it, the decline is
inherent to the position rather than to the script.

### E8 — `market-timing` : sell into the ladder before they do

Added 2026-09-17 from `docs/market_structure.md`, which found the parameter
nobody had read: **MELON is in no shop.** Town demand is a flat 1.0 unit/day all
season; both farms supply 142 against 30 absorbed. The price runs 271 (day 10)
→ 87 (day 14) and never recovers, so the same melon is worth 3× more depending
only on *when* it is sold. Same structure, weaker, for FERTILIZER: zero town
demand, 485 units supplied, price ends at ~4.

**Correction: the melon half of this is already closed, and I proposed it again
before checking.** Selling melon earlier has been gated twice and rejected
(harvest burst `tools/make_burst_variant.py`, realised price 179.7 vs 185.4;
fertilizer-for-timing `tools/make_ripen_variant.py`, −$5,199 ± 1,812 vs all
three), and the recorded verdict is *do not retry*.

What the table adds is the **mechanism**, which changes the question rather
than reopening it. Melon is a **fixed pot that shrinks when contested**: total
melon revenue is the integral of one price ladder split between two sellers, so
every unit either farm adds shrinks the pot for both, and selling earlier only
moves the collapse earlier. That is precisely what was observed when we joined
the day-10 dump — ahmed's own realised price fell $247 → $244 alongside ours.
A prisoner's dilemma, and entering harder loses.

**So the live question is the opposite one: are 12 opening melon tiles too
many?** At base price melon is 118.3 $/tile-day; at the realised margin it is
50.4, behind wheat. `melon_opening` is already a knob, so this is cheap.

| sub-experiment | cost | status |
| --- | --- | --- |
| **fewer** opening melon tiles (`melon_opening` 12 → 6/8/10) | cheap | not started |
| sell FERTILIZER early (zero town demand, monotone-decreasing) | cheap | not started |
| FERTILIZE ongoing wheat (+3 free units/action) | cheap | not started |
| more TOMATO late (day 14–18), the one crop that improves with time | cheap | not started |
| ~~reactive melon dump~~ | — | **closed**, twice-rejected, mechanism now known |
| **geese instead of sheep/cows** | done | **REJECTED**, see below |

Note the standing warning before building any of these: the demand table
predicts revenue, not wins, and the geese result below is what that costs.

**`g-goose4` — REJECTED.** Predicted from the absorption table: EGG is the
largest unclaimed pool ($18,817) and the only animal market whose price survives
volume (`log`, 0.2 — still 80% of base 400 units past I0), while MILK and WOOL
collapse to the price floor of 1 within ~100 units. `KNOB_DEFAULTS` has
`geese: 0`. Under a labour bind a goose also yields 1 egg per feed-action
against a sheep's 1 wool per 3.

Gated paired vs three externals, 48 worlds each, base `champ-h8`:

| opponent | paired diff | se |
| --- | --- | --- |
| pub-ahmed-v38 | +4,173 | 3,330 |
| pub-tschinkel-psr | +2,807 | 2,890 |
| pub-boatlee-v16 | **−4,853** | 2,360 |
| POOLED | +709 | 1,688 |

Money is a wash, and **win rate falls 8/144 → 2/144**. McNemar on the 6
discordant worlds: the candidate wins **0** of them, p = 0.0312. The loss is
concentrated in boatlee, the only opponent champ-h8 ever beats — exactly the
games that carry rating.

Confirmed on the ladder (631 held-out recorded seats, same sessions both runs)
— **worse in all six rating bins**, which is 1/64 by the sign test before the
magnitudes are even counted:

| opponent rating | champ-h8 | g-goose4 |
| --- | --- | --- |
| <600 | 0.55 (58/105) | 0.50 (53/105) |
| 600–1000 | 0.54 (25/46) | 0.52 (24/46) |
| 1000–1500 | **0.25 (30/120)** | **0.10 (12/120)** |
| 1500–2000 | 0.12 (14/120) | 0.06 (7/120) |
| 2000–2500 | 0.21 (25/120) | 0.17 (20/120) |
| 2500+ | 0.17 (20/120) | 0.12 (15/120) |
| **total** | **172/631 = 27.3%** | **131/631 = 20.8%** |

Both instruments agree for once, so this is not the panel inverting. This is
the herd-book failure again in miniature (`b1fceba`: herd book −$6,637), and
the market analysis did not save it. **Recorded so the next market-derived idea
is not sold on the table alone.** The absorption table predicts *revenue*, and
revenue is not the metric — it says what a market would pay for a unit, not
what producing that unit costs in the labour that is actually binding.

### Stage map — what to optimise, when

| stage | champ-h8 | tape-sel | implication |
| --- | --- | --- | --- |
| days 0-7 | rises to 0.511 | flat ~0.50 | opening is FINE, leave it |
| days 7-17 | **monotonic collapse to 0.084** | dips and recovers | the executor lineage's whole problem |
| days 18-25 | ~0.00, already decided | holds ~0.5 | endgame knobs here are wasted for champ-h8 |
| days 26-29 | 0.000 | **0.525 → 0.326** | the tape lineage's only sustained loss |

---

## Ordering

**E6's free test is already done** (above) and **E7 is the sharpest single target
we have** — the tape lineage is our best agent and its loss is localised to four
days. Then **E5** (hours, and every other line depends on it), then **E1 and E2 in
parallel** on separate branches — they are the only untried executor items and
E2 is the one that could reopen the whole lineage. **E3** is cheap and reuses
existing data. **E4** is the largest but has a free kill test; run that first and
only build if it passes.

## Standing constraints

- Rating noise is **±45 points**; sub-50 differences are meaningless.
- 5 submissions/day, **only the most recent 2 active**.
- Gate on the instrument that correlates, at the size the metric needs
  (~1,000 games/arm for win rate), replicated on a second block.
- Record negatives with their numbers. The commit log is the experiment record.
