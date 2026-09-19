# One traversal, every metric — draft spec for approval

2026-09-19. Not yet built. This is the design to approve or amend.

## The problem this solves

Fifteen-plus tools each replay the corpus separately to compute one metric.
`opp_coupling`, `market_clock`, `price_forecast`, `market_capacity`,
`volatility`, `glut_target`, `placement_ceiling`, `haul_probe`, `slot_cost`,
`target_horizon`, `k_ablation`, `s3_flow`, `net_ood`, `fit_risk`,
`sales_autopsy` — every one of them pays the same replay cost for a different
slice of the same state. A replay is ~2 s, so each full-corpus pass is ~2.4 h
at 5 procs, and we have never done one: every measurement above has been on
40–400 games, which is **0.2–1.9% of the corpus**.

A single traversal that records everything once turns every one of those from
a 2.4-hour replay into a sub-second query on a table.

**The sample-size argument is the real payoff.** The three best agents sit
within $411 ± 759 of each other; `k-comp5 − r-mrg100` is z = 0.54 on 1,131
worlds. Nothing separates at n = 400. Metrics computed over 21,054 sessions
are ~7× tighter on the same quantity.

## Corpus

```
21,054 sessions indexed
17,876 with ratings (85%)       <- daily dumps carry finals but no scores
```

**Open issue, needs a decision:** the 15% without ratings cannot be stratified
into ladder bins. Options are (a) compute rating-free metrics for all 21,054
and rating-conditioned ones for 17,876, (b) backfill from `manifest.csv` if it
carries ratings, (c) drop them. Default in this draft is (a), with a `rated`
flag per row so any query can filter.

## Output schema

One `.npz` per shard, fixed-shape arrays, no ragged structures. `D = 31` days,
`S = 2` seats, `P = 9` products, `K = 8` kinds.

| array | shape | contents |
| --- | --- | --- |
| `game` | (N, 8) | episode, seed, rated, min/max rating, final money ×2, winner |
| `day_market` | (N, D, P) | market inventory |
| `day_price` | (N, D, P) | price |
| `day_cap` | (N, D, P) | absorption capacity above 60% of base |
| `day_regen` | (N, P) | daily drain (shop-dependent, fixed per game) |
| `day_shops` | (N, D) | unlocked shop count |
| `seat_day` | (N, D, S, 6) | money, quadrants, hands, work, moves, hires |
| `seat_day_kind` | (N, D, S, K) | standing counts per crop/animal |
| `seat_day_sell` | (N, D, S, P) | **gross** sells (units) |
| `seat_day_buy` | (N, D, S, P) | **gross** buys (units) |
| `seat_day_rev` | (N, D, S, P) | realised revenue |
| `seat_day_shed` | (N, D, S, P) | shed contents (privileged) |
| `seat_act` | (N, S, 10) | action census: water/feed/care/pickup/plant/harvest/sell/hire/move/pass |
| `seat_slots` | (N, D, S, 3) | orders issued, orders filled, suppressed by the 10-cap |

Float32 throughout. Size: ~21k × 31 × 2 × 9 × 4 B ≈ 47 MB per seat-day-product
array, six of those ≈ 280 MB, plus the rest ≈ **~400 MB total**, shardable.

## Metrics, and which level each one serves

Every entry below is a **query on the table above**, not a new replay.

### L0 — win rate
- `win` = final money us > them, per game
- `margin` = us − them
- conditioned on rating bin where `rated`

### L1 — margin, not absolute money
- margin distribution and sign split
- **opponent coupling** — the one metric needing a *second* pass, because it is
  a counterfactual (mute their SELL into `market["inventory"]`, keep their
  cash identical). Known: $52,652/game, z = 17.52, on 40 games. Worth
  re-measuring at scale; costs a second traversal, so flagged as optional.

### L2 — sell timing
- `realised $/unit ÷ base` per product per seat = `seat_day_rev / seat_day_sell`
- **glut day** = first day `day_market[p] > I0`; known fertilizer d2, wool d7,
  melon/milk d11, strawberry d23, wheat/carrot/tomato/egg never
- **season utilisation** = total both-seat sells ÷ (cap at d0 + 30×regen)
- known ceiling: the floor binds on 2.5% of sizing calls — this level is
  CLOSED, and these metrics exist to confirm it holds at 21k rather than 40

### L3 — ordering
- saturated turns/game, suppressed SELLs, value at then-current price
- from `seat_slots`; known 21 turns, 46.8 SELLs, $29,009 — CLOSED, a delay not
  a loss ($55/seat stranded)

### L4 — investment
- `$/tile-day` per crop = revenue ÷ (standing count × days standing)
- herd size vs shop-draw demand — the known failure is **ours flat at 236
  strawberries across a sixfold demand range**
- quadrant unlock timing vs final margin (**the untested one** — flagged as
  unexplored in the metric hierarchy)

### L5 — risk / demand-supply ratio
- **projected inventory vs I0 at each product's harvest day** — the quantity
  the rent rule consumes, and the one thing that paid
- **volatility** per product = sd of price ÷ mean, and sd of glut day across
  games; this is what "bet on stable items" needs
- shop-draw exposure: P(no wool shop) = 34.4%, from `day_shops`
- **demand/supply ratio** = (cap + regen×H) ÷ projected supply from both boards

### L6 — farming and routing
- `moves/work`, `pickup/feed` from `seat_act`; known ours 1.43 / 1.07 vs the
  3000-band's 0.91 / 0.52–0.63 — BOUNDED AT ~$0, kept for regression only

## What this does NOT do

It records **state**, not counterfactuals. Any metric requiring an
intervention — opponent coupling, placement ceiling, action value, the paired
ladder gate — still needs its own run. The traversal makes the *observational*
metrics free; it does not make the *interventional* ones cheaper, and those
are the ones that have actually decided things here.

That boundary is the whole lesson of the last day: a p = 0.0001 observational
result moved the decision metric by 0%. **A bigger sample makes an
observational metric more precise, not more predictive.** This spec buys
precision on the quantities we already trust and nothing on the quantities we
do not.

## Plan

| phase | what | where | cost |
| --- | --- | --- | --- |
| **A** | `tools/metric_harvest.py`, schema above, `--shard i/n` | local | build + smoke on 20 games |
| **B** | validate: recompute 3 known numbers and require they match — capacity utilisation, glut days, moves/work | local | 400 games, ~15 min |
| **C** | full traversal, sharded | Kaggle (4 kernels × 11 h) or local overnight | ~2.4 h local at 5 procs |
| **D** | `TM_full_157` train on the larger corpus | Kaggle | 1 kernel |
| **E** | re-run L0–L6 queries at 21k, report what moved vs the 40–400-game numbers | local | minutes |

Phase B is non-negotiable: three metrics already have published values, and if
the traversal does not reproduce them it is wrong.

## What to approve

1. **Schema** — anything missing, anything not worth 47 MB?
2. **The unrated 15%** — include with a flag (default), backfill, or drop?
3. **Opponent coupling** — is the second traversal worth it, or keep at n=40?
4. **Order** — is D (train `TM_full_157`) blocking, or does it run in parallel
   with C? They are independent kernels.
5. **Kaggle vs overnight local** for phase C.
