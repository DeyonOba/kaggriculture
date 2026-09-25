# Stage B5 — the labour-gated fourth quadrant, pre-registered

## 0 Where it comes from

Two traced B2 games against 1000–1500 opponents (`tools/game_trace.py`,
days 10–25): we field the same eleven hands as the opponent, open a fourth
quadrant, and carry 16–21 idle tiles all mid-game against their 4–5 on three
quadrants; hands per worked quadrant 2.2–3.5 against their 4–5. DIG counts
and pre-yield drought losses are level with the opponent's, so idle ground is
the practice gap that survives measurement. The fourth quadrant gated
−$6,266 / −$6,017 replicated (memory: labour gates land), and the neural
planner's land head buys it regardless. `apply_to_plan` also carries a
"land crunch" fallback that buys the next quadrant when idle tiles fall below
12, which is precisely the state observed.

## 1 The one change

`KAGG_MAX_QUADRANTS=3` (`kagent/features.py`): once three quadrants are
owned, the land head's buy choice is masked and the crunch fallback is
suppressed. Default 0, no cap, so bundles and every prior run are unchanged.
Tests in `tests/test_land_gate.py`: no cap is byte-identical behaviour; at
the cap neither the head nor the crunch buys; below the cap purchases still
go through; the switch is read at import.

Applied to **all three B2 agents** — control, frozen trunk, trainable trunk —
from B2's own BC checkpoints, with B2's configuration throughout (`--seed
1234`, 20 × 64 episodes, `--reward win`, frontier mix, pv potential). Each
capped arm is paired against its own B2 counterpart on the same 785 held-out
games, reused from `rl/runs/stage_b2/session_eval.json`; only the three
capped arms are played.

## 2 Hypotheses and stop rule

- **H-land (primary)**: `land_control` beats `b2_control` on win rate,
  p < 0.05, positive in ≥ 4 of 6 bands, **and** money moves the same way.
  Wins and money moving together is the signature that idle ground is the
  mechanism; money up with wins flat repeats the board arms' pattern and is
  read as a loss-size effect only.
- **H-land-board**: the same test for `land_frozen` vs `b2_frozen` and
  `land_trainable` vs `b2_trainable`. Reported per arm; the primary test is
  the control.
- **H-consolidation (what passing would mean)**: if H-land passes,
  consolidation — knowing when to stop investing in land, and by extension in
  risky crops — is a lever the planner must learn or be given, and the next
  question is whether a learned cap (hands-conditioned) beats the fixed one.
- **Guards** as before: `policy_updates > 0` after warm-up, `|onpolicy_kl|`
  within tolerance, `bad == 0`; the reused baselines must match the draw game
  by game.

Power as before: paired se ≈ 0.008 at these win rates; a 0.025 delta
resolves at z ≈ 3. A null on the control closes the fixed cap at this budget
and says the idle ground is a symptom of something upstream (planting
quotas, hire timing) rather than the land purchase itself.

## 3 Not changed

The trunk file still predates the held-out exclusion; the BC readout is still
the heuristic's; the mechanism (idle tiles, hands per quadrant) is verified
after the fact with `tools/game_trace.py` on the same two worlds.

## 4 First run (2026-09-25, eval landed 04:01): the cap LEAKED — partial manipulation

**Mechanism check failed.** The capped control traced on the two B2 worlds
still held 3.7 and 3.9 quadrants over days 10–25 (B2 control: 3.8 on both).
A **third** purchase path was uncovered: `apply_land_pressure`, called at the
end of the neural plan path, buys the next quadrant on days 2–24 whenever
idle tiles fall below pending plantings plus two and cash exceeds the price
plus $1,500. The first gate test used a day-0 state, outside that window, so
it passed with the leak live. The cap therefore blocked the head and the
crunch but not the pressure net; the arms were partially capped, buying the
fourth quadrant later or less often rather than never.

Guards held (KL ≤ 2e-7, 11–12 updates per iteration, no zero-update
iteration), so the numbers are real, but they measure a weaker manipulation
than pre-registered and are **not read against the stop rule**:

| capped arm vs its B2 counterpart, 785 paired games | Δ win | z | p | our bank | opp bank | margin |
|---|---|---|---|---|---|---|
| control (54 vs 47 wins) | +0.009 | +1.15 | 0.32 | +$3,939 | +$1,583 | +$2,355 |
| frozen (29 vs 21) | +0.010 | +2.00 | 0.077 | +$127 | −$620 | +$747 |
| trainable (28 vs 41) | −0.017 | −2.20 | 0.041 | −$7,509 | +$5,397 | −$12,905 |

For the record: the partially-capped control is the highest held-out win
rate of any arm to date (0.069) and the first arm where wins, money and
margin all moved the same way, concentrated in the 2500+ band (+0.029). The
partially-capped trainable board arm went the other way on all three.

**Fix:** the pressure net is now behind `land_allowed` too, with a
mid-season test that exercises it. **Mechanism check, fixed gate, B2 control
on the two traced worlds:** max quadrants 3 on both; idle tiles days 10–25
15.9 → 4.1 and 20.7 → 1.9 (the opponent's level); plants + animals 75.0 →
69.6 and 73.1 → 72.1; hands unchanged; final bank $101.5k → $104.7k and
$91.4k → $104.1k. Two games, so anecdotal, but the manipulation now does
what it says.

## 5 Rerun (B5b), pre-registered before it ran

`tools/stage_b5b.sh`, two tests in one queue, cheapest first:

- **H-cap-rule (inference-time cap).** B2's three nets, weights untouched,
  played with `KAGG_MAX_QUADRANTS=3`, each paired against its own B2 games
  (reused). Tests consolidation as a *rule the planner is given*, isolated
  from retraining noise. Pass: `cap_control` beats `b2_control` on win rate,
  p < 0.05, positive in ≥ 4 of 6 bands, with money moving the same way.
  Board arms reported per arm.
- **H-land (retrained), as in §2**, into `rl/runs/stage_b5b`, same BC
  checkpoints, same configuration, same rule.

If the rule passes and retraining does not add to it, consolidation is a
constraint to be given, not learned, at this budget. If retraining adds to
it, the planner can use the freed labour. If neither passes, the idle ground
was a symptom, not a cause, and the two traced games were lucky.
