# Executor work: what is measured, what is ruled out, what is left

Branch `executor-v2`. Baseline is `champ-h8`. Everything here was measured against
`arena/agents/pub-*` — public competition agents, most of them replay tapes of 2500+ ladder
players — because the twin arena cannot gate this kind of change (see **Gating** below).

## The instruments

| tool | what it answers |
| --- | --- |
| `tools/exec_gate.py` | Does a candidate beat the baseline, paired on identical worlds, against every external opponent? |
| `tools/fetch_public_agents.py` | Installs the external opponents (their code is not committed). |
| `scheduler.MOVE_STATS` (set `MOVE_STATS_ENABLED`) | Which assignment pass spends the walking. |
| session scratch `h2h3.py` | Exact per-item money ledger, by wrapping the engine's `_commit_unit`. |
| session scratch `moves.py` | Per-pass move census over N games, running the package in-process. |

## What the labour gap actually is

Against `pub-ahmed-v38` on seed 19100, with near-identical farms (same 33 strawberries, ~57
standing tiles, 3 quadrants, ~11 hands):

| | champ-h8 | opponent |
| --- | --- | --- |
| unit-turns | 6,836 | 6,749 |
| MOVE share | 59.4% | 42.2% |
| productive share | 34.0% | 46.0% |
| productive actions | 2,818 | 3,391 |
| tile-visits | 1,703 | 1,828 |
| **actions per tile-visit** | **1.65** | **1.86** |
| visits yielding 4+ actions | 4% | 12% |
| moves before each action | 1.25 | 0.76 |

Movement by pass (3 games):

| pass | MOVE | action | move% | share of all movement |
| --- | --- | --- | --- | --- |
| zone (own zone, real work) | 7,260 | 6,388 | 53.2% | 61.2% |
| idlehelp (cross-zone, r≤5) | 1,999 | 486 | 80.4% | 16.9% |
| filler | 1,446 | 372 | 79.5% | 12.2% |
| shuttle (shed trips) | 760 | 206 | 78.7% | 6.4% |
| morning (zone provisioning) | 224 | 1,025 | 17.9% | 1.9% |

**The gap is not travel order — it is work per tile.** We make almost as many visits as they do
(1,703 vs 1,828); we extract less from each. Their 12% of visits with 4+ actions are animal
tiles run FEED → CARE → COLLECT_FERTILIZER → HARVEST in one stop; they hold 17 animals to our 14
and never skip CARE.

## Ruled out (do not retry)

Every row was gated paired against `pub-ahmed-v38` on identical worlds.

| change | vs twin | vs external |
| --- | --- | --- |
| filler walk capped at 3 tiles (1 after hour 16) | 19–11 | −$7,387 ± 3,126 |
| CARE crash-pivot removed | 11–19 | −$544 ± 1,734 |
| CARE threshold 0.35 → 0.15 | 14–16 | +$1,263 ± 1,631 |
| idle hands pick by value per turn `v/(d+1)` | 5–25 | −$4,928 ± 2,702 |
| zones split by occupied-tile weight ×2 / ×3 / ×5 | 11–19 / 1–29 / 1–29 | — |
| `MIN_ZONE_TILES` 5 → 8 / 9 / 10 / 12 | **24–6** / 0–30 / 2–28 / 0–30 | **−$4,490 ± 3,333** |
| `SWEEP_ORDER` forward-only serpentine sweep | — | pooled +$1,982 ± 3,154, but −$5,826 vs ahmed |

Why each failed:
- **Work-weighted zones** make a sparsely-planted zone geographically *larger*, so the hand walks
  further — the opposite of the intent.
- **Value-per-turn ranking** is wrong because the far P_TEND/P_HARVEST action really is worth five
  turns of walking and the nearby P_FILLER action really is worth almost nothing.
- **`SWEEP_ORDER`** (implemented, default off) ranks a distant task by how far forward it lies
  along the zone's serpentine instead of by Manhattan distance. It does not move the number it was
  built to move: MOVE share 58.8% / 59.2% / 58.7% at backstep 1 / 2 / 4 against a 58.3% baseline.
  Greedy-nearest inside a contiguous ~7-tile zone is already close to the optimal tour, so there is
  no ordering slack to recover. Gated anyway for a hard number: pooled +$1,982 ± 3,154 over 48
  paired worlds, negative against ahmed-v38 (−$5,826) and positive against the other two, i.e.
  noise. Kept in the tree, default off, as the documented answer to "did anyone try route
  ordering".
- **CARE** is a symptom, not a cause: we care half as often only because the crash-pivot fires once
  wool and milk collapse, and at $1–50 a unit the extra product does not repay the turns.

## Gating (the important correction)

`MIN_ZONE_TILES = 8` beat a copy of our own champion **24–6 at p = 0.0014, +$4,481 a game**, and
was **$4,490 ± 3,333 worse** against a real agent on the same worlds. Two agents sharing one price
ladder trade wins that come entirely from selling at different moments against each other, so any
change that moves when goods reach the shed can win the twin arena while being absolutely worse.

    PYTHONPATH=. uv run python tools/exec_gate.py --cand var-x --base champ-h8 --n 8

Ship only if the paired mean is positive against **every** opponent and the pooled mean clears
about two standard errors.

Budget the run: per-world spread against these opponents is ~$20k, so one opponent at `--n 8`
(16 paired worlds) has a standard error near $5k. That resolves a $10k effect, not a $2k one. Use
`--n 8` to triage a batch of ideas and re-run the survivor at `--n 25` before believing it.

## What is left, in order of expected value

1. **Raise work per visit rather than cutting travel.** We are at 1.65 actions per tile-visit
   against 1.86. Travel is ~1 move per tile visited and is close to irreducible; the return is in
   what each stop is worth. Concretely: more animals per pasture visit only pays if the product
   sells, so this is coupled to (2).
2. **Fix what the farm produces, not how it is tended.** Wool caps near $8k a season and milk near
   $6.3k because their price ladders collapse (wool is consumed only by YARN_STORE); egg and
   fertilizer never collapse. Our herd targets sit on the two steepest ladders. This is a planner
   change, and the evidence says it dominates anything left in the executor.
3. **Geese remain the clearest unexplained loss.** A goose costs $300, lays every day from day 4,
   and eggs have the flattest ladder in the game, yet adding three costs us $3,845 against
   `pub-ahmed-v38`. Something in coop building or goose servicing is wrong; a per-pass move census
   with geese enabled would localise it.
4. **Morning dispersal** is the one untouched movement cost: hours 4–5 run 84% / 79% MOVE for us
   against 61% / 54% for them. Hands are hired daily and spawn at the shed. Staging the first
   pickup on the way to the zone, or assigning zone indices by distance from the shed, is untested.
