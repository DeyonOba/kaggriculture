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
