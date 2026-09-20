# Policy spec: two heads over one spatial trunk

2026-09-20. To be reviewed before either head is written. Every shape below
is fixed by a measurement or an engine guard, and the ones that are not are
marked OPEN.

---

## Why this shape

Three prior attempts and what each could not express:

| built | output | could not express |
| --- | --- | --- |
| `exec_bc` | ranks ~60 tasks **our `build_tasks` proposed** | anything our planner did not think of |
| `raw_bc` | **fixed 28-class action type** | *which tile* — picks WATER, then hunts for a legal one |
| `ptrnet` | pointer over the same ~60 tasks | same planner bottleneck |

All three reached high accuracy (69.4%, 75.5%, 59.7%) and gated null to
catastrophic. The common defect is the output space, not the fit.

This spec is the [Gym-μRTS](https://arxiv.org/pdf/2105.13807) UAS formulation
with a spatial encoder: a distribution over **concrete legal alternatives**,
masked by the engine's own guards.

---

## 0. Trunk — shared

```
in    board_pair_units(obs, seat, ...)        (181, 10, 10)
      181 = 91 board/global + 12 plateau-lag + 20 crop-lag
            + 54 capacity + 4 unit-logistics
arch  conv stem -> N residual blocks -> feature map F  (d, 10, 10)
out   F, plus pooled g = concat(mean(F), amax(F))       (2d,)
```

`boardnet.py` already implements stem + ResBlocks + mean/amax pooling. The
trunk is reused verbatim; only the heads are new.

**Why a CNN and not a transformer over 100 tokens:** locality is the
structure we want and a conv gives it for free. Recorded in
`board_tensor.py` and unchanged.

---

## 1. Unit head — per unit, not per cell

**Settled by measurement, not preference.** GridNet emits one action per
cell. Here 24.3% of unit-frames are on a shared cell and **74.9% of those
groups do different things**, dominated by movement direction. One action
per cell would force co-located units to move identically, so they would
*stay* co-located — the error feeds back rather than averaging out. Measured
cost: **8.28% of all unit-decisions**.

```
for each unit i at (x, y):
  in    F[:, y, x]                      (d,)    trunk feature AT ITS CELL
        unit_state(obs, seat, i, priv)  (18,)   its own bag + position
  arch  concat -> MLP(d + 18 -> h -> 25)
  mask  legal_ops(obs, seat, i, priv)   (25,)   EXACT, verified 0/25
  out   masked softmax over 25 ops
```

25 ops: `PASS`, 4 moves, `WATER HARVEST FEED CARE FERTILIZE DIG
COLLECT_FERTILIZER DROP BUILD_COOP BUILD_PASTURE`, `PLANT:{5 crops}`,
`PLACE:{3 animals}`, `PICKUP:{WHEAT, FERTILIZER}`.

**The bag must be a head input, not a plane.** Four ops guard on the acting
unit's inventory (`FEED` a WHEAT, `FERTILIZE` a FERTILIZER, `PLACE` the
animal, `DROP` anything), so a policy that cannot see its own bag cannot see
what its mask is conditioned on.

Runs 1–12 times per turn. Negligible.

---

## 2. Market head — a sequence, not a choice

The engine takes a **list** of orders and truncates hard: `q[:max_orders]`
with `maxMarketOrdersPerTurn = 10`. So the output is a sequence of up to ten
slots, decoded autoregressively.

```
per slot k = 0..9:
  in    g (2d,)  +  slot embedding  +  running order summary
  out   categorical over 21 op-items + STOP
        then a quantity head conditioned on the chosen op-item
```

**Op-item vocabulary (21), from `_parse_order` and `_commit_unit`:**

| op | items | order form | guard |
| --- | --- | --- | --- |
| `SELL` | 9 products | `[op, item, n]` | shed holds it |
| `BUY_PRODUCT` | WHEAT, FERTILIZER only | `[op, item, n]` | money ≥ price **and** shed not full |
| `BUY_SEED` | 5 crops | `[op, item, n]` | money ≥ price (seeds bypass the shed) |
| `BUY_ANIMAL` | 3 animals | `[op, item, n]` | money ≥ price **and** shed not full |
| `HIRE` | — | `[op]` | atomic, resolved before per-unit orders |
| `BUY_LAND` | — | `[op]` | atomic |

**Quantity — OPEN.** `n` is an unbounded integer. Two candidates:

- **discrete buckets** `{1, 2, 3, 5, 8, 13, 20}` — simple, and matches the
  measured haul distribution (corpus units-per-PICKUP p90 = 6, max 18)
- **a scalar head** with a learned scale — finer, harder to mask

Buckets are the default until something argues otherwise.

**Orders commit one unit at a time in lockstep with the opponent**
(`_process_market`), so a 10-unit SELL interleaves with theirs and the price
walks between them. The head therefore controls *exposure*, not a fill —
already the recorded lesson that `SELL order[2]` is a request, not a fill.

---

## 3. Value head

```
in   g (2d,)  ->  MLP -> 1  ->  sigmoid
out  V(s) = P(this seat finishes with more money)
```

`tools/value_head.py` already fits this, and its verdict is the warning
label: **4.8× tighter than money as a statistic, wrong SIGN on 3 of 4 arms.**
It was fit on the recorded corpus, so it learned what co-occurs with winning.
Here it must be trained on **search/self-play outcomes** — counterfactuals,
not co-occurrence. Same architecture, different labels; that difference is
the whole point.

---

## 4. The three fallback modes

All three share trunk and heads; they differ only in how the output reaches
the engine.

| mode | rule | risk |
| --- | --- | --- |
| **residual** | heuristic proposes; policy may override a unit only when its max masked logit exceeds τ | starts at champ-h8; τ is a tuned constant, and five tuned constants failed this week |
| **gated** | policy acts; falls back to `build_tasks` when entropy is high | same τ problem, more upside |
| **pure** | policy acts always | highest ceiling; 14 prior RL generations never beat the heuristic |

Run all three at small scale, tracking improvement, as instructed.

---

## 5. What is NOT decided

- **Quantity encoding** (buckets vs scalar) — default buckets
- **Trunk width/depth** — reuse `boardnet` defaults, tune only if capacity
  is measurably binding
- **Training algorithm** — PPO with invalid-action masking is the μRTS
  result; expert iteration on `rollout_plan` is the only thing that has ever
  replicated here. These are not exclusive: rollout can label, PPO can
  polish.
- **τ** for residual/gated modes

## 6. The honest prior

Every piece below the heads is now verified: masks 0/25 disagreements
against the engine, encoder 0/2,185 mismatches, co-location settled on
measurement. Infrastructure has been reliable here.

What has failed **eleven times** is the next step — a fitted model beating
the closed form on the same consumer. Nothing in this spec changes that
prior. It only removes the three excuses the previous attempts had: the
planner bottleneck, the missing legality masking, and the absent spatial
encoder.

The gate does not move: `tools/ladder_gate.py`, ≥360 games, replicated on a
second block, win rate as well as money.
