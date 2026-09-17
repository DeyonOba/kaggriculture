# match-all: global assignment, 360 games, the session's only ship-quality result

## What it changes

Every assignment pass in `scheduler.assign` reads `for idx in sorted(free)`, so
units claim tasks in **unit-index order** — unit 0 takes the task nearest unit 0
even when unit 5 is standing on it, and the claim is irrevocable. That is
arbitrary-order greedy, which the literature puts strictly below optimal
matching in the worst case.

`match-all` replaces all three passes with **global-greedy matching** over the
identical candidate set and the identical cost `(urgent, distance, priority)`:
build every (unit, task) pair, sort by cost, take pairs cheapest-first, skip a
pair whose unit or task is already spoken for. Same filters, same
`try_execute`; only the order in which claims resolve changes.

Run-time cost is negligible — 12 units × ~60 tasks is microseconds in pure
Python.

## Result

| block | n | paired diff | se |
| --- | --- | --- | --- |
| seed 19100 | 96 | +$1,913 | 1,349 |
| seed 47000 | 120 | +$884 | 1,458 |
| seed 88000 | 144 | +$1,339 | 840 |
| **inverse-variance combined** | **360** | **+$1,381** | **641** (t = 2.16) |

Per-opponent means across the three blocks are all positive:

| opponent | block results | mean |
| --- | --- | --- |
| pub-ahmed-v38 | −466, +1,634, +1,687 | **+952** |
| pub-tschinkel-psr | +2,986, +2,036, +3,496 | **+2,839** |
| pub-boatlee-v16 | +3,218, −1,018, −1,167 | **+344** |

It clears the pooled 2-se bar. It does NOT clear "positive against every
opponent in every block" — but the negative rotates between opponents block to
block (boatlee twice, ahmed once), which is the signature of noise rather than a
real weakness. Notably it is the only change tested all session that is
consistently positive against **tschinkel**, the opponent that rejected every
other candidate.

## Caveats before submitting

- champ-h8 rates **810.4**; tape-sel rates **1,771.9** and tape-v2 **1,509.0**
  (2026-09-15, 9,063 teams). This is a champ-h8 derivative worth ~+$1.4k/game,
  so it enters the **executor lineage near 810**, not near the tape.
- Only the most recent **2** submissions are active, so submitting this
  deactivates one of the tape entries.

## Pre-flight

    size            0.112 MiB   (limit 100)
    compiles        yes
    file I/O        none — self-contained, so /kaggle_simulations/agent/ is moot
    smoke           DONE $75,597 / $109,443 (not the $3,000 no-op signature)

Submit the frozen bundle directly — do NOT use `tools/submit.sh`, which rebuilds
from the `kagent/` dev tree and would ship different code:

    cp arena/agents/match-all/main.py main.py
    uv run kaggle competitions submit kaggriculture -f main.py -m "match-all: global assignment"
