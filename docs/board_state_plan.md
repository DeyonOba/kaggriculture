# Board-state encoder: what the design gets right, and what bounds it

2026-09-18, branch `opponent-targets`. Response to
`docs/architecture_board_state.md`.

---

## 1. What the design gets right

**Stacking both farms into one (C, 10, 10) tensor is the single best idea in
it, and it is the one thing we have independently measured as valuable.**
`tools/board_tensor.py` today encodes ONE farm (`board_tensor(obs, seat=0)`,
24 channels). The forward-model ablation prices the opponent's board at **36%
of k=5 forecast error** (20.8 → 13.3). Every model we have built has been
half-blind, and this fixes it.

Broadcasting global scalars across the grid is also correct and standard — day
index, prices, both banks. Cheap, and convolution cannot otherwise see them.

The **belief-training** option (§Key Considerations 2: auxiliary head predicts
hidden state from visible board, supervised by the replay's ground truth) is
the methodologically right way to handle hidden state — *if* there is hidden
state worth predicting. There is not; see §3.

## 2. Three corrections, each with the measurement behind it

### 2a. The shed-belief GRU is the centrepiece, and it is worth ~nothing

The design devotes its most expensive component — a recurrent branch tracking
harvest/sell events into an "Est_Opponent_Inventory" channel — to the
opponent's hidden shed. We tested that with an **oracle**:

| | k=1 | k=3 | k=5 | k=10 |
| --- | --- | --- | --- | --- |
| + our shed | 4.3 | 7.7 | 13.0 | 29.0 |
| + their shed, **known perfectly** | 4.3 | 7.8 | **12.8** | 28.3 |

**1.5% at k=5, and no better than noise elsewhere.** A GRU that inferred their
shed *perfectly* would buy 1.5%. The reason is structural: their shed is
harvested-minus-sold, and both terms are already visible — harvests as
`yield_units` vanishing from their tiles, sales as market-inventory
increments. The information is in the board, so a belief head re-derives what
the board already told us.

**Cut the recurrent branch and the belief channels.**

### 2b. Do not train on actions

The Dataset predicts the next action (`_encode_action`). That is the one target
we have measured to fail, twice:

| attempt | per-action accuracy | played result |
| --- | --- | --- |
| behaviour cloning | 69.4% | null gate |
| raw-action BC | 75.5% | **$35** |

719 steps of closed-loop control compound, and DAgger's fix needs a queryable
expert while ours are recordings. This is the project's recurring law:
observational fit does not transfer to interventional effect. **Predict a
state quantity with a consumer, never an action.**

### 2c. The schema in the code is not this game's

Worth fixing before anything is written against it:

| in the doc | actually |
| --- | --- |
| `obs['players']`, `player_data['farm']` | `obs['farms'][i]['tiles'][y][x]` |
| `farmer_pos` | `farm['farmer']` (a `[x, y]` list), `farm['hands']` |
| `CROP_TYPES = CORN, BARLEY, SOY, SUNFLOWER` | `WHEAT, CARROT, TOMATO, STRAWBERRY, MELON` |
| `ANIMAL_TYPES = CHICKEN, PIG` | `GOOSE, COW, SHEEP` |
| `market.get(good)` | `obs['market']['prices'][item]` and `['inventory'][item]` |
| "Ground Items: Fertilizer_Count" | no ground items; `fertilizer_available` is a per-animal-tile bool |
| `replay['steps'][t][0]['observation']` | **sessions store ACTIONS ONLY** |

That last one is the big practical one. `rl/datasets/sessions/*.json` carries
`{episode, seed, teams, finals, shops, actions, scores}` and **no states at
all**. Every board tensor must be recovered by replaying the action scripts
(`~1.4 s/game` locally). That is the real cost of the dataset, not the fitting.

**One channel is missing and it is the most predictive one we know of:**
`consecutive_unwatered`. The design has `Is_Watered`; the engine's own
integral of dry days is what our crop-death CNN's 33.8× lift rests on, and it
is why the state is Markov at all.

## 3. Three ceilings, all measured

Before building an encoder, it is worth knowing what any encoder could win.

**Ceiling 1 — opponent supply: 17–20%, and we already hold ~80% of it.**
Giving the forecast the opponent's ACTUAL future market flow every day, which
no encoder can beat:

| | k=1 | k=3 | k=5 | k=10 |
| --- | --- | --- | --- | --- |
| projected from their board (today) | 4.3 | 7.7 | 13.0 | 29.0 |
| **PERFECT opponent supply** | 4.7 | 6.6 | **10.4** | 24.1 |
| no opponent at all | 4.4 | 12.0 | 20.8 | 44.2 |

Arithmetic on their visible board already closes 36% of the gap; perfection
closes another 20%, and is *worse* at k=1. The encoder would be competing for
that last slice — of one input, to a forecast, that feeds one decision.

**Ceiling 2 — history: $1 of $29, depth flat.** `tools/price_forecast.py` L7:
1 day and 5 days of board history give the same number. The engine integrates
history into the observation. (History *does* pay for opponent **targets** —
strawberry R² 0.56 → 0.87 at 10 days — but targets feed the forecast, so they
inherit ceiling 1.)

**Ceiling 3 — board → P(win): the board makes it WORSE.** Predicting the
winner between two 2800+ seats, ridge held out by game, 80 games:

| day | money only | + both boards | base rate |
| --- | --- | --- | --- |
| 9 | 0.438 | 0.575 | 0.525 |
| 15 | 0.475 | 0.588 | 0.525 |
| 21 | 0.662 | 0.575 | 0.525 |
| 27 | **0.800** | 0.637 | 0.525 |

Between matched players the winner is near-unpredictable until the money gap
opens in the last week, and **adding board features degrades the late-game
prediction**. This is consistent with the variance decomposition: among
matched agents the world explains 56% and the agent 6.9%.

## 4. If we build it anyway — staged, with kill criteria

Sized for the 12 days remaining. Each stage is a day or less and can stop the
next.

**S1 — two-farm tensor (half a day).** Extend `tools/board_tensor.py` from
`(24, 10, 10)` single-farm to `(~50, 10, 10)` with both farms stacked plus
broadcast globals. Keep `consecutive_unwatered`. Reuse the existing
replay harness. *No kill criterion — this is infrastructure and is correct
regardless.*

**S2 — predict the opponent's standing counts at H=5 and H=10 (one day).**
Target already validated as learnable by ridge: strawberry R² 0.87, goose
0.90, sheep 0.70 at 10 days of history. **Kill if the CNN does not beat the
ridge-on-counts baseline** — if spatial structure adds nothing over counts,
the whole encoder premise is wrong and we stop here.

**S3 — feed it into `_rent_inventory` (half a day).** The consumer already
exists and already gates positive. Replace the projected opponent supply term
with the model's. **Kill if forecast MAE does not move at least halfway from
13.0 toward the 10.4 oracle** — below that it cannot matter downstream.

**S4 — gate paired on the ladder, two blocks, judged on WIN RATE.** The rent
rule's own history is the warning: +$2,605 money at z = 5.2 and z = 1.15 on
wins. **Kill unless wins move**, because rating is win-based.

**Compute.** The bottleneck is replay, not fitting: 21k sessions ≈ 1.6 h
locally at 5 procs, or ~6.5 h on one Kaggle kernel (4 vCPU, ~4.5 s/game)
against an 11 h budget; four parallel slugs on disjoint seed blocks cut it to
~1.6 h. **A GPU buys nothing for the replay half.** Cache the tensors once —
20k games × ~25 usable days × 50 × 100 floats is ~10 GB at float32, so store
`uint8`/sparse or sample days.

## 5. The alternative the evidence favours

Both replicated wins in this project — the hiring rollout (+$3,287 ± 1,528)
and the rent rule (+$2,605 ± 500) — came from replacing a **constant** with a
state-dependent rule computed by **arithmetic**, not from a fitted model. Every
fitted model we have gated (BC 69.4%, raw BC 75.5%, the value head, the market
table, the legality mask) has failed to transfer.

The largest unexplained gap is not in the board: it is **884 rating points
between tape-sel (1694) and champ-h8 (810)**. The rent rule improves the
lineage that does *not* ship. Porting the forward market model into the tape
lineage's day-6 selector — so the shipped agent can use the one thing that
gated positive twice — has a larger expected value than the encoder, and is
a day of work.

**My recommendation: do S1 regardless (it is correct and cheap), run S2 as the
decisive test of the encoder premise, and put the rest of the time into the
shipped lineage.** If S2's CNN beats ridge-on-counts by a clear margin, the
encoder earns S3 and S4 on its own evidence rather than on prior belief.
