# State capture: what the board actually shows, and what we have been throwing away

2026-09-22. Workflow: **spec → research → use-case → experiment → evaluate →
commit → train → test over baseline.** Stages 1–3 are this document. Stage 4
is `tools/state_panel.py`.

---

## 0 The question that started it, answered first

*Why do we issue only ~0.5 market orders per turn on days 0–5?*

Measured over 6 games of `x-fut10`, orders per **day** and cash at dawn:

```
day      0      1      2      3      4      5      6      7      8      9     10     11
orders  11.0   2.0   12.0   11.0   14.3   17.5   23.5   20.8   29.7   27.0   30.2   28.2
cash  $3,000    $2    $25    $76   $324   $234   $186   $106    $23   $110   $993 $4,010
```

**We are not order-limited. We are broke.** The $3,000 opening is spent on day
0 and the bank runs at $2–$338 for the next nine days. `MAX_MARKET_ORDERS` is
10 per turn and we touch it in 1 turn out of 144 on days 7–9. There is no cap
to relieve and no decision to fix; there is nothing to buy with.

And the field is the same — median cash at hour 12, days 1–9: **champ-h8 $261,
ladder $389, M & M & P & Q $526**, and it does not track rating (band
1000–1500 averages $440, band 2500+ $349). Running the opening at zero cash is
how the game is played, not a defect.

**The divergence is one day wide and it is day 10**, when the first harvest
wave lands:

```
day 10 cash   champ-h8 $16    ladder $2,425   MMPQ $3,713   band 2500+ $5,063
day 11        champ-h8 $6,124 ladder $10,937  MMPQ $9,633   band 2500+ $12,408
```

We arrive at the first revenue wave with $16 and never close the gap (day 29:
$83,817 against the ladder's $86,169 and MMPQ's $101,781). That is the same
event as `peak_board` at day 21 against a field median of 9 — seen from the
cash side instead of the board side. **The question is not "why so few orders
early"; it is "why is day 10 worth so much less to us".**

---

## 1 Spec

Build a state representation that a model can learn the day-10 divergence
from. Three requirements, all of them from things measured this week:

1. **Not a frozen snapshot.** Rates are invisible in one frame: how fast their
   board is filling, whether a price is rising or falling, whether a crop was
   just watered or has been dry for two days.
2. **Not fitted on self-play.** A Φ fitted on champ-h8 self-play scored
   **−0.398 R² on replayed ladder games**, below a two-line hand-written rule
   (`docs/phi_ood_and_phase_timing.md` §1). Training data is replayed ladder
   sessions or it is worthless.
3. **Evaluated per rating band, never pooled and never top-only.** Every
   board-tensor experiment to date sampled with a rating **floor**
   (`tools/s3_flow.py::_pick(n, min_rating, seed)`), and 14,211 of the
   corpus's 21,054 sessions sit in the 2500+ band — so "min-rating 2000" is
   in practice "top games". That is exactly the mistake `tools/net_ood.py`
   was written to catch: a net trained on both-seats-2800+ closed **+71%
   in-distribution and −13%** against champ-h8.

## 2 Research: what the prior refutation actually covered

The temporal board stack is **recorded as closed**: `T_temporal_95` scored
−0.0058 and `T_crop_103` −0.0029 against `E_full_83` on market flow, and
history bought +10–14% R² on standing counts (melon +0.26, animals +0.00)
without carrying. That verdict stands *for what it tested*.

What it did not test:

- **a band-stratified sample.** Every arm was drawn with a rating floor, so
  the comparison lives almost entirely inside the 2500+ band.
- **per-band reporting.** A stack that helps in three bands and hurts in one
  reads as null when pooled.
- **the day-10 event specifically.** The arms were scored on pooled market
  flow across all 30 days, where day 10 is one thirtieth of the loss.

So the re-opening is justified on distribution and resolution, not on hope —
and it must be pre-committed to killing the idea again if the per-band table
comes back flat.

## 3 The feature audit: what the observation carries vs what we encode

Every key in the observation, against `tools/board_tensor.py`.

### Per-tile — PLANT

| field | encoded? | channel |
| --- | --- | --- |
| `crop` | yes | one-hot 0–4 |
| `planted_day` | yes, as age | 5 |
| `yield_units` | yes | 6 |
| `watered_today` | yes | 7 |
| `consecutive_unwatered` | yes | 8 |
| (derived) days to first yield | yes | 9 |
| **`max_lifespan_step`** | **NO** | — the tile's **death deadline** |
| **`fertilized_until_day`** | **NO** | — fertiliser expiry, and the one-time yield bonus lives here |

### Per-tile — ANIMAL

| field | encoded? | channel |
| --- | --- | --- |
| `animal` | yes | one-hot 10–12 |
| `fed_today` | yes | 13 |
| `cared_today` | yes | 14 |
| `fertilizer_available` | yes | 15 |
| `yield_units` | yes | 16 |
| **`placed_day`** | **NO** | — no animal age, so no maturity |
| **`consecutive_unfed`** | **NO** | — **the animal death clock** |
| **`pending_care_bonus`** | **NO** | — banked yield multiplier, not yet realised |

The asymmetry is the finding: the **plant** death clock is a channel and the
**animal** death clock is not, in a game where CARE doubles animal yield and
`pending_care_bonus` is exactly how much of that is banked and invisible.

### Globals (`scalars`, 24 values)

| quantity | encoded? | note |
| --- | --- | --- |
| day, hour | yes | |
| own money | yes | clipped at $60,000 |
| own hand count, quadrant count | yes | |
| price per product | yes | clipped at 3× base |
| **shop IDENTITY** | **NO — only `len(shops)/8`** | the eight types demand different products: WHEAT is in 5 of 8 and MELON in **0**. This is precisely the structure the +$676 future-demand term priced. A count cannot express it. |
| **opponent money** | **NO** | fully visible, and the money lead is the dominant late signal (r 0.552 → 0.892) |
| **`hires_today`, both farms** | **NO** | their labour spend, visible every turn |
| **own `seeds`** | **NO** | `scalars2` adds the shed and not the seeds |
| **own per-unit `inventories`** | **NO** | what each hand is carrying |
| market inventory | **damaged** | `min(abs(inv − 10000), 600)/600` — **the `abs()` destroys the sign**, so a glut and a shortage encode identically, and the ±600 clip saturates long before a real glut does |

The market-inventory encoding is the one to fix first regardless of anything
else here: the opponent's supply costs **$52,652/game (z 17.52)** through that
single shared variable, and we hand the model its absolute value.

### Not observable, and worth stating so nobody looks for it

Opponent `shed`, `seeds` and unit `inventories` are private. Their supply is
recoverable from public market inventory at ~5% through day 14 — which is one
more reason the inventory channel should carry sign and range.

## 4 The capture

Three frames — **today, day−1, day−2** — and the deltas between them.

```
per frame, per farm   24 tile channels (existing) + 3 new:
                        max_lifespan_step remaining / 30
                        consecutive_unfed / 2
                        pending_care_bonus / 4
                      + fertilized_until_day - day, clipped
deltas                12 rate-bearing channels x 2 lags:
                        yield_units, occupancy, watered, fed, cared,
                        money, and the 6 market-flow scalars
globals               day, own+opp money, own+opp hires_today, quads,
                      8 shop-identity bits, seeds(5), shed(12),
                      per product: price/base AND signed inventory
                      deviation (no abs, wider clip)
```

**Budget.** ~115 channels x 10 x 10 at float16 = **23 KB a frame**; 30 days x 2
seats = 60 frames a session, so **1.4 MB a session** and the 10 GB ceiling
holds **~7,200 sessions** — more than the corpus has. The binding constraint
is not memory, it is **band availability**: the six `ladder_gate` bands hold
101 / 14 / 637 / 1,610 / 4,329 / 14,211 sessions. Band 1 (600–1000) has
**fourteen**. Equal-size bands are impossible.

**So: train on the natural mix, evaluate per band.** That is not a compromise,
it is the recorded result — retraining on the natural mix turned −$546 into
−$276 parity where the 2800+ slice had gone −13% out of distribution.

## 5 Evaluation protocol

- split **by session**, never by row, and never by seat (the two seats share a
  world);
- train on the natural rating mix;
- report R² and AUC **per band**, plus a day-10 slice, and only then a pooled
  number;
- an arm ships only if it beats `E_full_83` in **no fewer than four of the six
  bands** and does not lose in any band by more than it wins in the best one;
- and then it still has to clear `tools/ladder_gate.py` twice, because
  everything in this document is observational until it does.

## 6 Kill criteria, written before the experiment

The temporal stack is dead again, and stays dead, if any of these holds:

1. per-band Δ against `E_full_83` is within ±0.01 R² in four or more bands;
2. it wins only in the 2500+ band — that is the prior result restated, not a
   new one;
3. the day-10 slice does not move, since that is the event the whole
   representation was built for.

---

## 7 Experiment and verdict: the temporal stack is refuted again, per band

`tools/temporal_bands.py`, on the 514-session ladder panel (30,840 rows).
Same 48 metrics, then with their 1-day and 2-day differences appended. Split
by session, four repeats, per-day ridge, α chosen inside train.

| band | rows | now only (48) | now + Δ1 (96) | now + Δ1 + Δ2 (144) |
| --- | --- | --- | --- | --- |
| <1000 | 4,500 | +0.239 / 0.810 | +0.233 / 0.814 | +0.218 / 0.819 |
| 1000–1500 | 6,420 | +0.260 / 0.757 | +0.244 / 0.751 | +0.239 / 0.749 |
| 1500–2000 | 5,070 | +0.133 / 0.724 | +0.125 / 0.717 | +0.115 / 0.708 |
| 2000–2500 | 4,470 | +0.189 / 0.728 | +0.193 / 0.736 | +0.203 / 0.747 |
| **2500+** | 4,260 | **−0.019 / 0.666** | −0.028 / 0.668 | −0.021 / 0.678 |
| **day 10 only** | | **+0.112 / 0.630** | +0.094 / 0.616 | +0.079 / 0.617 |

Against the §6 kill criteria, written before the run:

1. **within ±0.01 R² in four or more bands** — Δ1 gains >0.01 in **0 of 5**
   bands; Δ1+Δ2 in **1 of 5**. **Fires.**
2. wins only at the top — it does not even do that: the one band that gains is
   2000–2500, and the 2500+ band stays negative. Moot.
3. **the day-10 slice does not move** — it moves the wrong way, +0.112 →
   +0.079 R². **Fires.**

**So the 10 GB build is not justified, and the temporal stack stays dead** —
now on ladder data, band-stratified, with the criteria fixed in advance. The
earlier verdict (`T_temporal_95` −0.0058, `T_crop_103` −0.0029) was measured
on an effectively top-games-only sample; this replaces it with a stronger one
rather than overturning it.

One honest wrinkle: the lags do add a little **ranking** information while
costing calibration — AUC is up in 3 of 5 bands (+0.009, +0.019, +0.013) as R²
falls. Small, and criterion 1 fires regardless, but it is the reason to record
the AUC column rather than a single pooled R².

### The result that vindicates the band rule

**The 2500+ band is the hardest to predict of all of them, by a wide margin:
R² −0.019 and AUC 0.666, against +0.239 / 0.810 in the <1000 band.** Every
board-tensor experiment this project has run was drawn with a rating floor —
in practice, top games only. Reading any of them as a statement about "how
predictable the game is" was reading the hardest slice and assuming it
generalised down. The band choice does not shift the conclusion slightly; it
inverts it.

### Where this leaves the representation

The hypothesis was *"we are training on frozen board state"*. The measurement
says the freezing is not the problem: three frames buy nothing a linear model
can use, in any band, including at the exact day the divergence happens.

What the §3 audit found is a different thing, and it survives untouched — the
**frozen state itself is incomplete**. Every gap below is a static channel,
not history, and none of them was tested by this experiment because none of
them is in the 48-metric panel:

```
animal `consecutive_unfed`   the animal death clock  (plants have one, animals do not)
animal `placed_day`          no animal age -> no maturity
`pending_care_bonus`         banked yield multiplier, invisible
`max_lifespan_step`          the plant's death deadline
`fertilized_until_day`       fertiliser expiry
shop IDENTITY                8 bits, replaced by a count -- and WHEAT is in 5
                             shops, MELON in 0; this is exactly the structure
                             the +$676 future-demand term priced
opponent money, hires_today  visible, unencoded
market inventory SIGN        `abs(inv - 10000)` makes a glut and a shortage
                             identical, on the variable that costs $52,652/game
```

That is the next experiment, and it is cheap: add the channels, re-run the
same per-band table, keep the same kill criteria. No new dataset, no 10 GB,
no history.

---

## 8 The missing static channels: measured, and they FAIL the bar

`tools/missing_features.py` adds the 35 audited quantities as per-day scalars
and `tools/ladder_panel.py --out ladder_panel_x.npz` rebuilds the 514-session
panel with them (48 → 83 metrics). All 35 verified live, none constant.

> One near-miss worth recording: three of the eight shop identifiers I wrote
> from memory (`PIZZERIA`, `YARN_SHOP`, `SMOOTHIE_BAR`) do not exist in the
> engine — the real ones are `PIZZA_SHOP`, `YARN_STORE`, `SMOOTHIE_SHOP`.
> Those bits would have read zero all season and silently voided the entire
> shop-identity arm. `SHOPS_L` now reads `kagent.constants.SHOPS`.

Group ablation, per band, four splits, held-out **sessions**, ΔR² / ΔAUC
against the 48-metric base:

| band | +clocks | +banked | +opp | +seeds | **+minv** | +ALL |
| --- | --- | --- | --- | --- | --- | --- |
| <1000 | −0.013 / −0.002 | +0.000 / +0.002 | −0.040 / +0.004 | −0.003 / −0.005 | **+0.035 / +0.010** | −0.059 / +0.019 |
| 1000–1500 | −0.006 / −0.003 | −0.002 / +0.000 | +0.008 / +0.005 | −0.005 / −0.000 | −0.009 / +0.000 | −0.009 / +0.002 |
| 1500–2000 | −0.005 / −0.004 | +0.002 / +0.003 | −0.023 / +0.001 | +0.009 / +0.002 | −0.012 / −0.004 | −0.009 / −0.006 |
| 2000–2500 | −0.007 / −0.000 | +0.012 / +0.012 | +0.037 / +0.007 | −0.001 / +0.008 | **+0.012 / +0.013** | +0.044 / +0.026 |
| 2500+ | +0.005 / +0.001 | −0.006 / +0.003 | −0.004 / −0.006 | −0.015 / −0.006 | **+0.011 / +0.014** | −0.016 / +0.011 |
| **verdict (≥4/5 bands, R²)** | 0/5 **FAIL** | 1/5 **FAIL** | 1/5 **FAIL** | 0/5 **FAIL** | **3/5 FAIL** | 1/5 **FAIL** |

**Every arm fails the pre-registered bar.** The death clocks — the gap that
looked most obviously wrong, since plants have a death clock and animals do
not — buy **nothing, in any band**. `pending_care_bonus` and the seed counts
buy nothing. The best is the signed market inventory at 3 of 5 bands, which is
the one the audit predicted would matter most and which still misses.

A harness bug was found and fixed in the middle of this: `+opp` first read
**−1.683 R²** in the <1000 band, which was `per_day_fit` extrapolating a
near-degenerate column without the support clamp `tools/fit_potential.py`
already had. Clamped, it reads −0.040. The verdicts did not change, so the
conclusion is robust to it — but the first table was wrong and this is the
third time this exact failure mode has appeared.

### The thing I am not allowed to do, and the test that would be legitimate

The criterion was written on **R²**. On **AUC** the picture is different:
`+minv` gains in 4 of 5 bands and `+ALL` in 4 of 5, with the largest gains at
the top (+0.026 at 2000–2500, +0.011 at 2500+). And there is a real argument
that AUC is the right metric here, because **the rating is win-based** and
five separate results have shown margin and win coming apart.

That argument had to be made *before* seeing the table. Switching the
criterion now would be exactly the post-hoc selection this document's §6
exists to prevent, so the recorded verdict stands: **FAIL**.

What is legitimate is to pre-register AUC and run it on **data these arms have
never touched**. Recorded now, before that draw:

> **H-AUC.** On a fresh, disjoint draw of ladder sessions, `+minv` improves
> AUC against the 48-metric base by more than +0.005 in at least 3 of the 4
> bands that can be re-drawn disjointly (1000–1500, 1500–2000, 2000–2500,
> 2500+). Bands <1000 and 600–1000 cannot be re-drawn — the corpus holds 101
> and 14 sessions in total — so they are excluded by availability, not by
> choice. Any other outcome kills the signed-inventory channel.

### H-AUC on fresh sessions: the signed market inventory PASSES

Second draw, seed 7, **377 sessions none of these arms had seen** (overlap
with draw 1 stripped; it was concentrated in the <1000 band, which holds only
115 sessions in total and therefore cannot be re-drawn — that is why the four
test bands were fixed in advance).

AUC change against the 48-metric base, per band:

| band | +clocks | +banked | +opp | +seeds | **+minv** | +ALL |
| --- | --- | --- | --- | --- | --- | --- |
| 1000–1500 | +0.0012 | −0.0020 | +0.0013 | +0.0013 | **+0.0057** | +0.0023 |
| 1500–2000 | +0.0004 | −0.0004 | +0.0031 | −0.0040 | +0.0035 | +0.0051 |
| 2000–2500 | −0.0009 | −0.0016 | +0.0057 | −0.0006 | **+0.0066** | +0.0081 |
| **2500+** | +0.0033 | −0.0016 | +0.0011 | −0.0018 | **+0.0143** | +0.0108 |
| **H-AUC (≥3/4 > +0.005)** | 0/4 fail | 0/4 fail | 1/4 fail | 0/4 fail | **3/4 PASS** | 3/4 pass |

**`+minv` passes**, on data it had never touched, against a criterion fixed
before the draw, and its largest gain (**+0.0143**) is in the 2500+ band —
the hardest one, where the base model is at R² −0.100 and AUC 0.643. `+ALL`
also passes, but never beats `+minv` by enough to justify 35 columns against
9, and it is the arm that drags R² down.

Everything else fails on fresh data exactly as it failed on the first draw.
**The death clocks are dead.** The plant/animal asymmetry looked like an
obvious bug and buys nothing in any band on either draw; so do
`pending_care_bonus`, the fertiliser expiry, the seed counts and the
opponent's labour.

### What this finding is, and what it is not

The one real gap in the audit is the one predicted a priori: `scalars()`
encodes `min(abs(inv − 10000), 600) / 600`, and **the `abs()` throws away the
sign of the single shared variable that costs $52,652 a game (z 17.52)**. A
glut and a shortage are handed to the model as the same number.

It is **not a shipped win and cannot be gated**, and that distinction matters:

- `kagent/planner.py` reads `view.market_inv` directly and has always had the
  sign. The agent is unaffected.
- the `abs()` lives only in `tools/board_tensor.py::scalars`, i.e. in the
  **encoder every learned model here has ever consumed** — the board CNN, the
  value heads, the BC ranker, the flow nets. All of them were trained on a
  market channel with half its information removed.

So the finding re-opens the fitted-model track rather than adding money. Every
"a net cannot beat the rule" result on record was measured with this channel
damaged, and that is now a known defect rather than a mystery.

**Not fixed in `tools/board_tensor.py` yet, deliberately.** Changing the
semantics of 9 of the 24 scalar channels silently invalidates every cached
`rl/datasets/*.npz` built with the old encoder, and `arm_channels` addresses
channels by absolute offset. That is a rebuild decision, not a drive-by edit.

---

## 9 Encoder v2, the rebuild, and putting the board inside the RL loop

Three changes, in the order they have to happen.

**1. `scalars()` carries the sign.** `tools/board_tensor.py`:

```python
# was:  min(abs(inv - 10000), 600) / 600      glut and shortage IDENTICAL
# now:  tanh((inv - 10000) / 1500)            opposite signs, no hard clip
```

Verified: inventory 8,500 → **−0.762**, inventory 11,500 → **+0.762**. Under
v1 both read 0.60.

`ENCODER_VERSION = 2` is now a module constant, every built dataset stamps it
as `ENC`, and the bump rule is written down: **bump when a channel's semantics
change, not when its count does.** A silently-reinterpreted channel is
indistinguishable from a real result, and that is the failure this guards.

**2. The dataset is rebuilt band-stratified, not behind a floor.**
`tools/s3_flow.py --bands` draws equally per rating band and stores the
session rating per row so every evaluation can be split by band.

```
rl/datasets/s3_flow_v2band.npz   encoder v2, 21,500 rows, 1,075 sessions
  band      0-1000    2,300 rows    115 sessions   (the whole corpus has 115)
  band   1000-1500    4,800 rows    240
  band   1500-2000    4,800 rows    240
  band   2000-2500    4,800 rows    240
  band       2500+    4,800 rows    240
  ratings 197-3203, median 1945
```

Against the old `--min-rating` builds, whose median sat in the 2500+ band.

Two build bugs fixed on the way, both of which produced silent wrong data:

- `_session_rating` read the first 4 KB of the session file for `"scores"`.
  `scores` sits **after** the two 720-step action scripts, i.e. megabytes in,
  so every rating came back `NaN` and the per-band split would have been
  empty. It now reads the index.
- `pool.map` buffered every worker's `(n, 177, 10, 10)` float32 in the parent
  — ~1.5 GB at 1,075 sessions — and broke the pipe. Workers now return
  float16 and `build` streams with `imap`.

**3. The board now enters RL training, and is learned there.**

The policy had never seen it. `rl/rollout.py` recorded only
`kagent.features` day-level aggregates — 168 numbers in which the board is a
set of counts — while `tools/board_tensor.py` sat offline feeding flow nets.

- `rl/rollout.py`: with `KAGG_BOARD=1` every recorded decision also carries
  `board_pair_full(obs, seat)` as float16, (91, 10, 10), ~18 KB a decision and
  ~35 MB an iteration at 64 episodes. Off by default, verified inert when off.
- `rl/common.py`: `BoardTrunk` (two 3×3 convs, mean **and** max pooling, 48-d
  output, 38,736 parameters) and `PlannerNet(n_features, board=True)`, which
  concatenates the trunk output onto the feature vector and raises rather than
  silently dropping the board if it is missing.

It is trained **end to end with the policy**, not pretrained and frozen. A
frozen trunk fitted on a corpus is a self-play potential by another name, and
that scored **−0.398 R²** out of distribution. The representation has to move
with the states the policy actually visits — which is the whole point of
putting the encoder in the loop rather than in a dataset.

Mean-and-max pooling because several channels matter as **existence** rather
than as a count: one ripe melon, one starving cow.

### Why this reopens the RL track rather than decorating it

Every learned model here — the board CNN, the four value heads, the BC
ranker, the flow nets — consumed a market channel with its sign removed, and
was evaluated on a sample drawn behind a rating floor from a corpus that is
two-thirds top games. Both defects were ours. Neither is evidence about what a
model can do on this game, and the recorded verdicts that rest on them
("V-ranking is CLOSED", "capacity channels KILLED", "BC gated null") are
**suspect for reasons that have nothing to do with the models**.

That does not make them wrong. It makes them untested.

---

## 10 The temporal arms on the repaired dataset: both positive, and not yet believable

`tools/s3_flow.py --ablate` on `s3_flow_v2band.npz` (encoder v2,
band-stratified, 21,500 rows / 1,075 sessions), paired within seed on
identical held-out games.

| seed | `E_full_83` | `T_temporal_95` | Δ | `T_crop_103` | Δ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8155 | 0.8152 | **−0.0003** | 0.8219 | **+0.0064** |
| 1 | 0.8035 | 0.8202 | **+0.0167** | 0.8096 | **+0.0061** |
| **mean** | **0.8095** | **0.8177** | **+0.0082** | **0.8157** | **+0.0062** |

Recorded verdict before this: `T_temporal_95` **−0.0058**, `T_crop_103`
**−0.0029**. **Both arms have changed sign.**

### Why that is not yet a result

**Two seeds give no usable standard error, and the tool says so itself.** The
`t = +34.10` the table prints for `T_crop_103` comes from two paired
differences that happen to agree to four decimals (+0.0064, +0.0061). At one
degree of freedom that is a coincidence of two numbers, not evidence. The
honest reading is: `T_crop_103` is **consistent** across both seeds;
`T_temporal_95` is **not** (−0.0003 then +0.0167), and its mean is carried
entirely by one seed.

**Two things changed at once, and the comparison to the old verdict is
therefore confounded.** The old numbers came from a dataset that was
floor-drawn *and* encoder v1. This one is band-stratified *and* encoder v2,
on different sessions. So the sign flip cannot be attributed to the market
channel repair — that would need v1 and v2 built over the *same* sessions,
which is one more build and is the clean test.

**No per-band table.** This run predates the per-band reporting now wired into
`ablate`; it reports pooled R² only, which is the exact shape of number that
hid the problem in the first place.

### What the run does establish

The absolute level: **held-out R² ≈ 0.81 on 10-day opponent flow**, on
band-stratified data spanning ratings 197–3203. Whatever the arms do relative
to each other, the encoder predicts the opponent's market flow well across the
whole ladder and not only at the top.

### The test that would settle it

```
--ablate --arms E_full_83 T_temporal_95 T_crop_103 --seeds 0 1 2 3 4 5
```

six seeds for a usable se, plus the per-band table, on this dataset. ~18 fits
at ~17 minutes. And separately, a v1-encoder build over the **same 1,075
sessions**, which is the only way to attribute anything to the `abs()` repair
rather than to the sampling change.
