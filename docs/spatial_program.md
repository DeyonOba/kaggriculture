# The spatial programme: a board encoder, per-day metrics, and a horizon plan

2026-09-17, evening. Written after eleven nulls, and after a correction I owe
the record.

## The correction

I argued that "representation is not the bottleneck", citing the
enumerated-legal-move experiment where masking changed accuracy by 0.0006.
That experiment tested a spatial **action head**, and E-A had already shown
this game has no source→destination choice (a unit acts only on the tile it
stands on). It said nothing about a spatial **encoder**. I ran the two claims
together and they are not the same.

Checked directly, the case for an encoder is strong:

| model | what it sees of the 10×10 board |
| --- | --- |
| `raw_bc.feats` | the unit's tile + 4 adjacent = **5 of 100 tiles**, plus 7 scalar counts |
| `kagent/features` | day-level aggregates, **no per-tile layout** |
| `value_head` | the same aggregates |

**Nothing we have ever built sees the layout** — which tiles neighbour which,
how clustered the animals are, where the empties sit. And the one deficit we
have measured most precisely is purely spatial:

```
hop distance between consecutive worked tiles   ours 1.92   ahmed 1.60
moves per distinct tile                         ours 2.64   ahmed 1.88
```

A gap none of our models can even represent. That is a real argument for a
board encoder and I was wrong to wave it away.

## The distinction that makes it viable

A CNN **policy** would hit the wall behaviour cloning hit: 75.5% per-action
accuracy compounded over 719 steps gave $35, because errors accumulate with no
correction path. Raising accuracy does not fix compounding.

A CNN **metric predictor** is a different object. One input, one output, no
loop, no accumulation. Predicting "how many crops die this week" or "what is
tomorrow's realised price" is ordinary supervised learning on a spatial input,
and the failure mode that killed the policy does not apply.

**So: encoder yes, controller not yet.** Use it to see, not to act, until it
has earned the right.

---

## 1. Metrics we have not measured

Everything in `docs/metrics_catalogue.md` is aggregate or per-day. These are
spatial or temporal and all are cheap:

| metric | why it might matter | cost |
| --- | --- | --- |
| **animal clustering** — mean pairwise distance, count within 3 tiles | one loaded unit could feed a cluster; a scattered herd forces shed round-trips. PICKUP is 12.5% of our actions vs their 5.6% | free, scratch tool exists |
| **work-span per unit-day** — bounding box of one unit's worked tiles | measures whether units own a region or wander. Zone assignment is sticky, the task is not | cheap |
| **path overlap between units** | two units covering the same ground is duplicated travel | cheap |
| **ripe→harvested latency** | yield sits on the tile losing nothing, but `max_held` caps it; latency past the cap is pure loss | cheap |
| **harvest→sold latency** | melon loses 2.4× between day 10 and day 14; we know the ladder collapses, not how fast we clear | cheap |
| **realised price ÷ best price available that day** | execution slippage on selling, distinct from the volume gap | cheap |
| **idle capital** — cash not deployed, by day | the balance sheet has cash/inventory/deployed but never asks what idle cash cost | free |
| **assets past payback** — animals/land whose cumulative yield exceeds cost | tells us when an investment stopped being a decision | cheap |

## 2. A state-aware model per day

Input: the board as **channels over a 10×10 grid** — the natural encoding, and
the one thing we have never tried.

```
channel  content
  0-4    crop one-hot (wheat, carrot, tomato, strawberry, melon)
  5      crop age / max_yield_day
  6      yield_units / max_yield
  7      watered_today
  8      consecutive_unwatered      <- the death predictor
  9-11   animal one-hot (goose, cow, sheep)
 12-13   fed_today, cared_today
 14      fertilizer_available
 15      structure (coop/pasture, empty)
 16      weed
 17      locked
 18      unit occupancy count
 19      distance to shed (static)
```

Plus a scalar vector (day, hour, cash, prices, shop draw) broadcast or
concatenated at the head. Stack the last *k* days on the channel axis for the
temporal half — a plain Conv stack is enough at 10×10; a transformer is
overkill for 100 tokens and we have no compute to spare.

**Targets, all mechanical and all already computable:** crops that will die
this week, tiles that will go idle, next-day yield by product, day-end cash.

**Validation, and this is the part that matters.** Held-out prediction accuracy
is *not* the test — five fitted models passed that and failed to move money.
The test is whether the encoder's output changes a decision that then clears
`tools/ladder_gate.py`. Until then it is instrumentation, not policy.

## 3. The horizon plan: assets, capital, stocks

`tools/balance_sheet.py` already tracks cash / inventory / deployed / book /
burn / locked per day. What it does not do is look **forward**. The pieces to
add:

- **payback clock per asset** — days until cumulative yield covers cost, against
  days remaining. The planner has last-plant guards for crops and nothing for
  animals or land.
- **stock projection** — holdings + standing crop yield vs `forecast_drain`,
  which `market_headroom` computes for the season but not as a rolling horizon.
- **idle-capital cost** — cash held × the LP's marginal return, which we now
  know is $13/action-equivalent on labour.

## 4. The immediate test, and it comes from your framing

You said: minimise weeds from crops not watered **when they are in demand and
have high priority**. That last clause is the part my triage got wrong.

My rule ordered rescues by **time to yield**. It cut strawberry deaths 75% →
60% and still lost $1,028 ± 511 — because **strawberry's product is worthless
at the margin**. Its `above_target` is 1.6 on a linear curve: the price hits
the floor of 1 within ~100 units past I0, and both farms already oversupply it.
Saving a strawberry saves a unit the market will not pay for.

The rule that follows from your clause is different: **triage by product
value**, not by ripeness. Water the crop whose output is *under*-supplied.
From `market_headroom`:

| product | supply ÷ absorption | marginal price ÷ base |
| --- | --- | --- |
| TOMATO | 0.12 | **2.38** |
| WHEAT | 0.38 | 1.74 |
| CARROT | 0.43 | 1.36 |
| STRAWBERRY | 0.95 | 0.91 |
| MELON | **4.72** | 0.50 |

A dying wheat is worth 1.74× base at the margin; a dying strawberry 0.91× and
falling. We have never connected the market table to the executor — every
market finding so far was tested by changing **what we plant**, and every one
lost because planting displaces. Changing **what we water** displaces far less.

That is the next experiment: `P_SURVIVAL` and `P_TEND` ordering weighted by
`econ.price(product, current_inventory)` rather than by distance or ripeness.
It is one anchor in the same frozen scheduler, gateable tonight.

---

## Standing rules this inherits

- Gate on `tools/ladder_gate.py` — paired, McNemar, held-out.
- Check `tools/knob_archive.py` before touching any knob.
- A metric improving is not a result: watering triage improved its metric by 15
  points and lost money at z = −2.01.
- Mechanical beats fitted, and neither is exempt from the gate.
