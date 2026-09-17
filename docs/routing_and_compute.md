# Routing, and whether compute is the constraint

Two questions, both now answered with measurements rather than argument:

1. Would a compiled language help?
2. Is the ~$22k movement gap to the tape reachable by better routing?

## 1. Compute: the harness was the cost, not the game

`cProfile` on one champ-h8 vs `pub-ahmed-v38` game (24.85 s under the profiler,
~3.1 s without):

| component | time | share | what it is |
| --- | --- | --- | --- |
| `copy.deepcopy` | 13.13 s | 53% | `__get_shared_state` clones the whole state per agent per step |
| `structify` | 7.36 s | 30% | state rebuilt into `Struct` twice per step |
| `jsonschema` | 1.27 s | 5% | every action validated; the schema itself re-checked 1,448×/game |
| both agents' `act()` | ~2.5 s | 10% | our scheduler **and** the opponent's |
| kaggriculture interpreter | small | | the actual game |

Over two thirds of a game is `kaggle_environments` plumbing. **A C or Rust
rewrite of the scheduler would optimise the 10%**, and could not ship regardless
— Kaggle submissions are a Python `main.py` and the in-game work is trivial
anyway (2-opt over ~8 stops is microseconds).

The lever was the harness, and it is pure Python. `tools/fast_env.py`:

- replaces `deepcopy` with a direct recursive clone (the state is JSON-shaped,
  so `deepcopy`'s reduce/memo/dispatch machinery is all overhead);
- promotes the interpreter's returned state **in place** instead of rebuilding
  it. The rebuild's only real job was resyncing `Struct.__dict__` — `Struct`
  mirrors entries into `__dict__`, and only `__setattr__` keeps the two sides in
  step, so an interpreter writing `farm["money"] = x` leaves `farm.money` stale;
- skips per-action schema validation, keeping default-filling.

**Result: 3.11 → 1.84 s/game, 1.70×, bit-identical rewards** on every seed
(`tools/fast_env.py --verify` runs patched and stock in separate processes and
compares). Wired into `arena/run_games.py`; `KAG_FAST_ENV=0` restores stock.

This multiplies every offline search left: rollout, gating, selector fitting,
expert-iteration data generation.

## 2. Routing: eleven mechanisms, all null

The gap is real and well-measured: 976 excess moves a game, 12.4 moves per
worker-day against the tape's 9.4, on the *same* tile coverage (5.4 vs 5.2
tiles). At the LP's ~$23/action that is ~$22k, and it is essentially the whole
part of the champ-h8/tape gap that allocation and labour supply do not explain.

| # | mechanism | result |
| --- | --- | --- |
| 1–5 | zone relaxations: walk caps, zone sizing, priority bands, idle-help scoring | null |
| 6 | value-driven prize-collecting `pick()` | much worse |
| 7 | bigger shed pickup batches | null |
| 8 | whole-day committed tours (NN seed → budget truncate → 2-opt) | worse |
| 9 | global matching, pass 1 (`match-p1`) | **−$1,119 ± 957, REJECT** |
| 10 | global matching, all passes (`match-all`) | +$1,913 ± 1,349, then +$884 ± 1,458 on a fresh block — null |
| 11 | crop layout clustering (`lay-block`, `lay-zone`) | mechanism does not fire |

Attempts 9 and 10 were worth making because every pass in `scheduler.assign`
reads `for idx in sorted(free)` — units claim tasks in **unit-index order**, so
unit 0 takes the task nearest unit 0 even when unit 5 is standing on it. That is
arbitrary-order greedy, which in theory can be arbitrarily worse than optimal
matching. Replacing it with global-greedy matching over the identical candidate
set and cost changed nothing measurable. (Literature is consistent with this:
greedy runs ~0.93× Hungarian on multi-robot target tracking, and in one
agricultural field-layout study greedy *beat* Hungarian on completion time.)

Attempt 11 tested the remaining structural explanation: `build_tasks` assigns
crops round-robin (`wanted[i % len(wanted)]`), so each crop is smeared across
the farm, and same-crop same-day plantings ripen together — a scattered stop set
that no router can compact. Holding the crop multiset bit-identical and
clustering the tiles instead moved nothing, because work is **already** local
per worker-day (mean pairwise distance ~2.9 tiles on a 10×10 farm); the zone
partition localises it. There was no scatter to remove.

### Method note

A 4–6 seed smoke test is not evidence. `match-p1` smoked **+$7,201 on 4 seeds**
and gated **−$1,119 on 96**. Structural metrics (moves/action, work spread) are
low-variance and do settle on a handful of seeds; money does not.

### Standing interpretation

The tape is **open-loop**: a fixed script optimised offline in its own world,
paying nothing for adaptivity. Our scheduler is closed-loop and re-targets every
turn. The excess movement looks like the price of that adaptivity, which is an
architectural property, not a routing defect. Routing is closed.
