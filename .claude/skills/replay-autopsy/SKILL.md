---
name: replay-autopsy
description: Characterise any Kaggriculture agent from its replays — tape-or-reactive, moves/action, turn budget, farm layout, true sales. Use when studying a leaderboard opponent, a downloaded episode, or one of our own agents.
---

# Replay autopsy

One command replaces the ad-hoc scripts. **Always start here before theorising
about an opponent.**

```bash
# a downloaded opponent
PYTHONPATH=. uv run python tools/replay_probe.py --dir replays/mmpq --team "M & M & P & Q"

# one of ours, played locally
PYTHONPATH=. uv run python tools/replay_probe.py --agent champ-h8 --opp pub-ahmed-v38 --n 6

# pick probes: tape,budget,layout,ops,sales
PYTHONPATH=. uv run python tools/replay_probe.py --dir replays/x --team T --probes tape,budget
```

## Getting the replays

The leaderboard URL carries the submission id (`?submissionId=NNN`). One
targeted call, never a crawl — broad crawling has caused 429s:

```python
import sys; sys.path.insert(0, "tools")
from harvest_replays import list_episodes
eps = list_episodes(56254996)          # -> 150 episodes, ids + both agents' ratings
```
Then `uv run kaggle competitions replay <id> -p replays/<name>` with a ~1.5s sleep.
The replay's `info` carries **TeamNames** and the **seed**, so their games are
reproducible in our engine.

## Reading the output

**moves/action is the number that separates strength.** Reference points:

| agent | moves/action | note |
| --- | --- | --- |
| M & M & P & Q | **0.71** | reactive, rated 3,093 |
| pub-ahmed-v38 | 0.84 | |
| a replay tape | 0.85 | |
| champ-h8 | **1.45** | ours |

**tape probe** — pairwise identical-action rate over the first 40 turns across
*different seeds*. A tape is ~100%; a reactive agent is ~27%. This is how we
established the top of the leaderboard is not all tapes.

**byday probe** — every other metric is a season average. champ-h8 is WORST on
days 2-8 (1.79-1.95 moves/action) and IMPROVES to 1.14 by day 26; a tape steps
down from 1.23 to 0.66 at day 12. So the executor does NOT degrade during the
day 7-17 window where champ-h8's P(win) collapses — it improves. Those are two
separate problems.

**workers probe** — per-worker load spread is NOT a lever: tape-sel runs a 9.1x
spread across 14 units and beats champ-h8's 3.0x across 12.

**layout probe** — wide worker-day span with LOW moves/action means *sweeping*;
narrow span with HIGH moves/action means *oscillating*. M & M & P & Q sweeps
(span 12.87 @ 0.71); champ-h8 oscillates (span 7.56 @ 1.45).

## Traps this tool already avoids

- `order[2]` in a SELL action is a **requested** quantity, not a fill — the
  engine stops at the seller's stock. Counting orders made one agent look like
  it sold 17,045 carrots. The sales probe uses conservation instead.
- Seat is identified by **TeamNames**, not submission id; replays carry no
  submission ids.
- Seat 1's observation has **`step: None`** (only seat 0 carries it).
- `hands` are **day-scoped** — re-hired each morning, so an hour-0 snapshot
  legitimately shows zero.

## Related

- `tools/decision_diff.py` — what would OUR agent do from THEIR states, turn by
  turn. Self-checks at 100% against our own recorded actions; run that check
  whenever the harness changes.
- `tools/taskset_diff.py` — what work does our planner propose vs what they do.
- `tools/winrate_archive.py` — re-score every past experiment by WIN RATE.
  The rating is win-based; champ-h8 loses 96.5% of games at −$31k mean margin,
  so margin gains move the size of a loss.
