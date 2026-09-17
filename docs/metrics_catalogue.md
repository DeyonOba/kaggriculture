# Metrics catalogue: what we can measure, and what happened when we optimised it

2026-09-17. The thesis behind this document: every constant in the agent is a
candidate variable, and anything we can *measure* per day is a candidate
objective to minimise or maximise as the season progresses.

That thesis is right in principle and has been tested four times. **It paid
once.** This catalogue records every mechanical metric we can compute, its
current value, the opponent's value where known, and — the important column —
what happened when someone tried to turn it into a lever.

Two kinds of metric are mixed here and they are not equally trustworthy:

- **Mechanical** — a fact about the trajectory (crops died, moves made, tiles
  empty). Cannot be wrong, only irrelevant.
- **Fitted** — a model's opinion (V(s), BC accuracy, demand tables). These have
  failed to predict interventions **five times today**; see
  `docs/world_modeling.md`. Never optimise one without a paired gate.

---

## 1. Constants tested as per-day schedules

The direct test of "constants are really variables".

| lever | as a CONSTANT | as a per-day SCHEDULE | verdict |
| --- | --- | --- | --- |
| **hires** | refuted both ways: −2 is −$11,746 ± 1,829, +2 is −$4,408 ± 1,384 | **+$3,287 ± 1,528** over 192 fresh worlds | **PROVEN VARIABLE** — shipped as `CHAMP_HIRES` |
| **straw** | negative at every value: 21 −$5,667, 27 −$3,714, 36 −$152, 40 −$600 | **REJECTED TWICE**: −$5,809 ± 1,715 and −$6,227 ± 2,130 | not variable |
| **land** | null: `land2_margin` 800/1500/4000 → +150/+294/−620 | +$215 — an order of magnitude under the gate's se | not variable |
| **wheat** | negative: 24 −$614 ± 241, 40 −$502 ± 160 | **UNTESTED** | unknown |

**The straw result is the cautionary one.** Two rollouts, 6 worlds and 16
worlds, disagreed on *direction* — one found more strawberries, one fewer —
each gaining ~$5k in-sample and losing ~$6k out. Doubling the evaluation worlds
did not reduce the overfit, it moved it. A schedule has 30 free parameters and
our per-world sd is $34,749; that combination overfits unless the lever is
unusually sharp, which hiring was and strawberry is not.

So: **one of four levers is genuinely a variable.** Treat "make it a schedule"
as a hypothesis needing the same gate as anything else, not as a free win.

---

## 2. Mechanical metrics, measured

Ours vs `pub-ahmed-v38` unless noted. Tools in `tools/`.

### Execution

| metric | ours | theirs | tool | tried as a lever? |
| --- | --- | --- | --- | --- |
| work share of unit-turns | 40.5% | 50.0% | `commit_probe` | yes — `j-tile` raised actions 2,791→3,057 for **+$465** |
| move share | **58.6%** | 42.2% | `commit_probe` | 12 routing mechanisms, all null |
| idle (PASS) share | 0.9% | 7.8% | `commit_probe` | — |
| moves per action | 1.79 | 1.06 | `commit_probe` | yes, repeatedly |
| moves per distinct tile | 2.64 | 1.88 | `commit_probe` | E1 — **refuted structurally** |
| hop distance between worked tiles | 1.92 | 1.60 | `commit_probe` | untried |
| actions per tile visit | 1.58 | 1.88 | `commit_probe` | same-tile-first, +$465 |
| trips per tile-day | 1.35 | 1.32 | scratch | — (null, near-identical) |
| runs per tile | 1.06 | 1.04 | `commit_probe` | — (null) |
| no-op rate | **0.2%** | — | `action_audit` | — nothing to fix |
| doomed-tile actions | 1.2% | — | `action_audit` | E7 — ~$2,700, not pursued |
| **realised $ per marginal action** | **$1.75** | LP says **$13.00** | LP dual + j-tile | **the central gap** |

### Crops and land

| metric | ours (champ-h8) | tape-sel | tool | tried? |
| --- | --- | --- | --- | --- |
| crops planted | 88.0 | 41.3 | `crop_deaths` | — |
| **crop death rate** | **24.2%** | 11.3% | `crop_deaths` | **untried** |
| — strawberry | **75%** (18.7 of 25) | 33% | `crop_deaths` | schedule rejected twice |
| — wheat / carrot / melon | 3% / 4% / 0% | 0% | `crop_deaths` | — |
| empty unlocked tiles, day 16 | ~9 | 1–3 | `tile_census` | — (a symptom of deaths) |
| weeds at day 28 | 14.3 | ~2 | `tile_census` | — |
| weeds spawned on empty tiles | ~0 | ~0 | `tile_census` | — irrelevant |

### Market

| metric | value | tool | tried? |
| --- | --- | --- | --- |
| supply ÷ absorption, MELON | **4.72×** | `market_headroom` | 7 melon fixes, all rejected |
| supply ÷ absorption, EGG / TOMATO | 0.20 / 0.12 | `market_headroom` | geese and tomato — **both rejected** |
| unclaimed revenue, under-supplied | $48,723 | `market_headroom` | — predicts revenue, not wins |
| destroyed by over-supply | $36,398 | `market_headroom` | — |
| unsold inventory at day 29 | **$0** | scratch (E7) | nothing to recover |
| book value / burn / locked % | per day | `balance_sheet` | day-23 loss found, unexplained |

### Instruments (metrics about our measurements)

| metric | value | why it matters |
| --- | --- | --- |
| per-world sd, ladder margin | **$34,749** | sets every sample size |
| paired sd (same world, σ=0.10) | **$17,168**, corr 0.95 | always pair |
| rating noise, identical bytes | **sd 46** over 4 samples | <90 points is noise |
| winner's curse, best of N | `sqrt(2 ln N) · se` | best-of-11 at 36 worlds = $4,130 |

---

## 3. What this catalogue says to do next

The single number that organises everything is **$1.75 vs $13.00** — our
marginal worker-action realises a seventh of what the LP's does. Every lever
tried so far either *adds* actions (hires, hands, quadrants) or *moves* them
(12 routing mechanisms, same-tile-first). None has raised the value of the
marginal one.

The two candidates that would:

1. **Watering triage.** Strawberry needs ~10 consecutive waterings before its
   first unit and dies on two misses; we plant 33 in one week and lose 75% of
   them. Planting fewer is refuted and scheduling them is refuted — but
   *choosing which ones to water* has never been tried, and it needs **no extra
   labour**. This is the only untried lever that reallocates rather than adds.
2. **Wheat as a schedule** — the one constant→variable cell still empty.

Both must clear `tools/ladder_gate.py` (paired, McNemar, held-out) before
anything ships, and `tools/knob_archive.py` must be checked first.
