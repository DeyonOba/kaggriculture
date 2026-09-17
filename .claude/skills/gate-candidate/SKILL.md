---
name: gate-candidate
description: Decide whether a Kaggriculture candidate is actually better — which instrument to use, how many games, and which results to distrust. Use before claiming any improvement or spending a submission slot.
---

# Gating a candidate

**Four instruments have failed here. Pick the right one or the result is worse
than nothing — it costs a submission slot and a day.**

## The instrument, by lineage

| you are testing | use | why |
| --- | --- | --- |
| a TAPE | `rl/session_bench.py ladder --heldout`, read **mean money** | Spearman **+0.94** vs rating over 6 scored submissions |
| an EXECUTOR change | `tools/exec_gate.py` (margin) **and** the ladder | exec_gate alone has inverted; see below |
| anything, as a screen | `tools/tape_splice.py screen` / cheap n | ranks only, never gates |

```bash
PYTHONPATH=. uv run python rl/session_bench.py ladder \
    --agent arena/agents/<name>/main.py --n 40 --procs 6 --heldout
```
Validate any new instrument against these known outcomes before trusting it:
tape-sel **1750.3** > tape-sel3 **1671.9** > tape-v2 **1521.5** > tape-112
**1330.6** > champ-h8 **810.4**.

## The four failures, so they are not repeated

1. **Margin vs win rate.** The rating is win-based. champ-h8 loses 96.5% of
   games at a mean margin of −$31,049, so a $1–5k lever moves the *size of a
   loss*. `match-all` gated +$1,381 ± 641 over 360 games and took wins from
   5/144 to 4/144. `exec_gate` now prints win rate and warns on disagreement.
2. **Gate size.** champ-h8's own baseline win rate was 3.5% on one seed block
   and 7.3% on another. Detecting 5%→8% needs ~1,058 games/arm (~11 min with
   `tools/fast_env.py`). We used 144–192 for weeks.
3. **Self-play does not transfer.** An opening at 96.1% against a champ-h8 twin
   scored −$11,566 and 3/144 wins externally. Two copies share one price ladder
   and trade wins through relative selling timing alone.
4. **The 3-opponent panel INVERTS.** `tape-112` won 70.8% against
   ahmed/tschinkel/boatlee over 720 games (z = 3.75) and rated **191 points
   below** the arm it beat. Maximising over three fixed agents selects the arm
   that exploits those three.

## Sizing

- **Screen** (rank many arms): small n is fine, but the max over K noisy arms is
  upward-biased by ~+30pp at K=243, n=24. Tape 271 screened **83.3%** and
  confirmed **41.9%**. A screen can never gate.
- **Confirm**: ≥360 games/arm on FRESH seeds, then **replicate on a second
  independent block**. Both survivors that replicated (112, 277) held; four of
  seven candidates shrank 14–41 points.
- **Paired tests**: same worlds both arms → use **McNemar on discordant pairs**,
  not a two-proportion z.

## Reading the result

- **Rating noise is ±45 points.** The same bytes scored 1750.3 and 1706.1.
  Ignore sub-50-point differences. Scores also drift after first appearing.
- A 4–6 seed money result is **not evidence**: `match-p1` smoked +$7,201 on 4
  seeds and gated −$1,119 on 96. Structural metrics (moves/action, work share,
  actions/tile) ARE low-variance and do settle on a few seeds — use those to
  screen mechanism, never money.
- **Mechanism firing ≠ money.** Four changes raised work share and all lost
  money. Always check both.

## Before a submission slot

5/day, **only the most recent 2 are ACTIVE** — a submit evicts the next-oldest
active entry. Pre-flight every bundle:

```bash
# size <= 100 MiB, compiles, no file I/O, and NOT the $3,000 no-op
```
See the `build-variant` skill. `$3,000` exactly = the bundle raised on import and
kaggle-environments silently no-opped every turn.
