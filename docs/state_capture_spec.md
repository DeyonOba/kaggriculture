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
