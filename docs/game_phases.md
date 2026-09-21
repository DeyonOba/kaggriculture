# Game phases: goals, and how to measure each one

Written 2026-09-17. This did not exist — the stage map lived only inside
`docs/experiment_trajectory.md`, and nothing measured the **balance sheet** along
the way. Every experiment in this project has been scored on final money, which
cannot tell a day that invested well from a day that did nothing.

---

## The phase boundaries, and where they come from

`tools/value_head.py attribute` gives P(win) per day from a head fitted on 3,000
real games. Run on our two lineages:

| day | 7 | 9 | 12 | 13 | 17 | 21 | 26 | 27 | 29 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tape-sel (1750) | 0.535 | 0.583 | 0.340 | **0.649** | 0.535 | 0.572 | 0.525 | **0.326** | 0.334 |
| champ-h8 (810) | **0.511** | 0.372 | 0.244 | 0.339 | 0.084 | 0.002 | 0.000 | 0.000 | 0.000 |

| phase | days | champ-h8 | tape-sel | the goal of this phase |
| --- | --- | --- | --- | --- |
| **A. Opening** | 0–7 | rises to **0.511** | flat ~0.50 | convert cash into productive assets; we are already even here |
| **B. Contest** | 7–17 | **collapses to 0.084** | dips to 0.34, **recovers to 0.65** | the whole game. champ-h8 never recovers from a dip; the tape does |
| **C. Consolidation** | 18–25 | ~0.00, decided | holds ~0.5 | protect a lead / keep optionality |
| **D. Endgame** | 26–29 | 0.000 | **0.525 → 0.326** | liquidate without giving it back — the tape's only sustained loss |

Two things follow immediately. **The opening is not our problem** — we are an even
game at day 7. And **endgame knobs are wasted on champ-h8**, whose P(win) is
already ~0 by day 18, while they are exactly where the tape bleeds.

---

## The measurement gap: investment vs leftover capital

Final money is a single number at day 29. It cannot distinguish:

- a day that spent $3,000 on a pasture that pays from day 12, from
- a day that spent $3,000 on seeds that never ripen, from
- a day that spent nothing and banked it.

All three look identical until the season ends. **So we cannot currently afford
to try a losing-looking idea**, because we have no way to tell a deliberate
investment from a mistake until it is too late to attribute.

What a per-day balance sheet would carry:

| quantity | how to compute | what it answers |
| --- | --- | --- |
| **cash** | `farm["money"]` | liquid position |
| **inventory** | shed + every unit inventory, at live market prices | value we hold but have not realised |
| **deployed** | standing crops × seed cost + animals × cost + structures + land paid | capital committed to assets |
| **book value** | cash + inventory + deployed | total position, comparable across strategies |
| **burn** | book value change across the day | did this day create or destroy value |
| **risk** | fraction of book value in assets that have not yet paid back | exposure |

`holdings()` in `tools/opp_belief.py` already computes shed + unit inventories
correctly (reading the shed alone is badly wrong — produce sits in a unit's hands
and is DROPped mid-turn). Prices are public in `obs.market`. Everything needed
exists; nothing assembles it.

**This is worth building before the next planner experiment.** Phase A is where
investment decisions concentrate (land day 5, herd from day 2, the opening crop
mix), and it is precisely the phase where P(win) says we are fine — so a change
there has to be judged on the balance sheet, not on a final bank that is
dominated by phases B–D.

---

## What each phase currently decides, and with what rule

| phase | decisions | rule | uncertainty term |
| --- | --- | --- | --- |
| A | opening crop mix, land #1 (day 5), herd ramp, hires 4–5/day | `CHAMP_OPENING`, `CHAMP_LAND_DAY`, `CHAMP_HIRES`, `_champ_herd_targets` | **none** |
| B | land #2 (~day 11), herd to target, strawberry target, melon harvest | constants + `hires_delta` knob | **none** |
| C | wheat engine, fertilizer use, sell floors | constants | **none** |
| D | `liquidation_day`, `carrot_day` 25, endgame carrots | constants | **none** |

Every investment is a constant: 2 quadrants, day 5, $100 cash margin, 25% of cash
for hires, 60% for seeds. No variance, no payback horizon, no ahead/behind, no
opponent state. See E6 in `docs/experiment_trajectory.md` — ranking by mean
margin instead of a risk-aware statistic costs **15.6 percentage points of
held-out win rate** on data already on disk.

---

## How to use this

- **Judge a phase-A change on book value and burn**, not final money.
- **Judge a phase-B change on P(win) at day 17**, which is where champ-h8 is
  already decided.
- **Do not spend effort on phase C/D for the executor lineage** — it is at 0.000
  by day 18. Phase D is a tape-lineage target only.
- Gate everything on the ladder (`gate-candidate` skill); the 3-opponent panel
  inverts.

---

## Measured 2026-09-22: how much of the outcome each phase actually decides

The table at the top of this document is P(win) per day for two *specific*
lineages. This is the complementary measurement — how much of the final
result is *determined* by the end of each phase, across 400 games with
randomised openings (`rl/datasets/propagation_perturb.npz`), read off the
per-day potential fitted by `tools/fit_potential.py` (six splits, by game,
held-out).

Φ's own error floor is **R² = 0.900 at day 29** — the season is over and it
still misses a tenth. So "resolved" is quoted both raw and as a share of what
is knowable at all.

| phase | days | margin resolved | of knowable | win AUC start → end | gain |
| --- | --- | --- | --- | --- | --- |
| **A opening** | 0–7 | 14.8% | **16.4%** | 0.554 → 0.638 | +0.084 |
| **B contest** | 7–17 | 25.9% | **28.8%** | 0.638 → 0.796 | **+0.158** |
| **C consolidation** | 18–25 | 32.6% | **36.2%** | 0.795 → 0.948 | +0.153 |
| **D endgame** | 26–29 | 10.1% | **11.3%** | 0.959 → 0.997 | +0.038 |

```
day  7   83.9% of the knowable margin still open   win AUC 0.638
day 12   78.1%                                     win AUC 0.687
day 17   55.1%                                     win AUC 0.796
day 21   35.7%                                     win AUC 0.876
day 25   16.3%                                     win AUC 0.948
```

**Three things follow, and the third is the one that matters.**

**1. The opening decides almost nothing.** 84% of the knowable margin is
still open at day 7 and 78% at day 12. That is an independent confirmation,
on different data and a different estimator, of `docs/reward_design.md`
Rule 2 — *"through day 12 the money lead is a coin flip", r ≈ 0.00,
sign-correct 50%*. Two measurements, two datasets, same answer: **do not
shape, reward, or search the opening on any outcome proxy.** There is nothing
there yet to be right about.

**2. Raw resolution peaks in consolidation, not contest** — C resolves 36.2%
of the knowable margin against B's 28.8%. Taken alone that would argue for
spending effort on days 18–25.

**3. It would be the wrong conclusion for us, and the table at the top of
this document is why.** champ-h8's own P(win) falls 0.511 → **0.084 across
phase B** and is **0.000 by day 21**. The 36.2% that resolves in C is margin
we have already lost by the time C starts: it is resolving *how badly*, not
*whether*. Since the rating is win-based, **phase B (days 7–17) is the only
phase where an intervention can still change the result**, and it is exactly
the phase where the tape dips to 0.34 and recovers to 0.65 while we do not.

So the per-stage policy, stated once:

```
A  days 0-7    nothing to reward. 84% still open, cash is a coin flip.
               Shape on investment STATE only, or not at all.
B  days 7-17   the whole game. Every reward, every search, every schedule
               degree of freedom belongs here.
C  days 18-25  resolves the most margin and decides no wins FOR US.
               Margin theatre. Do not spend search here.
D  days 26-29  11.3%, and champ-h8 is at P(win) 0 long before it.
```

### One thing this measurement does NOT license

A tempting corollary is to score search candidates by rolling only to a phase
boundary and reading Φ, instead of playing all 30 days — 1.7× to 3.8× cheaper
per evaluation. The arithmetic that makes it look free is wrong: `1 − R²` is
Φ's **prediction error against the outcome**, not i.i.d. noise that averages
out with more games. The part of it that correlates with the candidate is
**bias**, and no sample size removes a bias. Truncated scoring is a
hypothesis — *does Φ rank candidates the way the full game does?* — and the
test is to score a dozen existing arms both ways and compare the orderings.
It is cheap. It has not been run. Until it has, every gate plays the full
season.
