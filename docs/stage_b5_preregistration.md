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

## 6 Result, inference-time cap (2026-09-25, eval landed 04:59): H-cap-rule PASSES

B2's three nets unchanged, `KAGG_MAX_QUADRANTS=3` at inference, 785 paired
held-out games each, baselines reused game-by-game; `bad == 0` throughout.

| net under the cap vs the same net uncapped | wins | Δ win rate | se | z | discordant | p | 95% CI | bands ≥ 0 | our bank | opp bank | margin (z) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 62 vs 47 | +0.019 | 0.007 | +2.80 | 22 / 7 | 0.008 | [+0.006, +0.032] | 5 of 6, 4 positive | +$3,806 | −$784 | +$4,590 (+9.9) |
| frozen trunk | 29 vs 21 | +0.010 | 0.004 | +2.32 | 10 / 2 | 0.039 | [+0.002, +0.019] | 6 of 6, 3 positive | +$4,675 | −$63 | +$4,738 (+11.0) |
| trainable trunk | 59 vs 41 | +0.023 | 0.007 | +3.31 | 24 / 6 | 0.001 | [+0.009, +0.037] | 6 of 6, 5 positive | +$4,992 | −$758 | +$5,750 (+12.0) |

**H-cap-rule: PASSES** on the control — p 0.008, positive in four of six
bands (the one negative band is 600–1000, 21 games, one flip), and money
moves with wins: bank up, opponent's bank slightly down, margin +$4,590 at
z +9.9. The same rule improves all three nets by a similar amount, +$4.6k to
+$5.8k of margin and +1.0 to +2.3 points of win rate, which is what a
mechanism looks like rather than a fit. Near-miss losses inside $5k double
for the control (11 → 22): the cap moves games toward the margin.

**First rule in the board track to raise held-out wins**, and the largest
win-rate gain of any arm in the project's paired evals. The capped control
at 0.079 is the best held-out configuration to date. The capped trainable
trunk is at parity with the capped control on wins (−0.004, p 0.80) with
+$2.4k of bank and a −$1.0k margin: the board still does not add wins.

**Consolidation reading (the user's hypothesis):** knowing when to stop
investing in land is worth about two points of win rate and $4.6k of margin
as a fixed rule at inference, with no retraining. Whether retraining under
the constraint adds to it is the second half of this queue.

## 7 Result, retrained under the cap (2026-09-25, eval landed 12:01)

Guards: KL ≤ 1.9e-7, 10.7–13.1 updates per iteration, no zero-update
iteration, `bad == 0`. Same 785 games; B2 nets and the inference-cap nets
reused game-by-game.

| retrained under cap vs … | Δ win | z | discordant | p | bands + | our bank | opp bank | margin (z) |
|---|---|---|---|---|---|---|---|---|
| control vs B2 control (63 vs 47) | +0.020 | +2.54 | 28 / 12 | 0.017 | 4 of 6 | +$3,909 | −$492 | +$4,401 (+6.0) |
| control vs inference-cap control (63 vs 62) | +0.001 | +0.15 | 22 / 21 | 1.00 | 1 of 6 | +$103 | +$292 | −$189 (−0.3) |
| frozen vs B2 frozen (68 vs 21) | +0.060 | +6.16 | 54 / 7 | <0.0001 | 5 of 6 | +$13,929 | −$10,263 | +$24,192 (+22.2) |
| frozen vs inference-cap frozen (68 vs 29) | +0.050 | +4.91 | 52 / 13 | <0.0001 | 5 of 6 | +$9,254 | −$10,200 | +$19,454 (+17.9) |
| trainable vs B2 trainable (27 vs 41) | −0.018 | −2.34 | 11 / 25 | 0.029 | 1 of 6 | −$7,054 | +$5,402 | −$12,457 (−11.4) |
| trainable vs inference-cap trainable (27 vs 59) | −0.041 | −4.49 | 10 / 42 | <0.0001 | 0 of 6 | −$12,046 | +$6,160 | −$18,207 (−16.3) |
| frozen vs control, both retrained under cap | +0.006 | +0.64 | 33 / 28 | 0.61 | 2 of 6 | +$4,097 | +$1,733 | +$2,364 (+2.6) |

**H-land (retrained control): PASSES** — +2.0 points, p 0.017, four of six
bands, money with it. **Retraining adds nothing to the rule for the
control** (+0.001 against the inference-time cap): for the control,
consolidation is a constraint to be given, not learned, at this budget.

**The frozen trunk retrained under the cap is the best agent to date:** 68
of 785, 0.087, money $69k, margin −$34k against B2 control's −$41k. It gains
+6.0 points over the same net uncapped and +5.0 over the same net with the
cap at inference only, on five of six bands, and the opponent's bank falls
by $10k: with the land purchase removed, this policy learned to compete for
the shared market rather than yield it. Against the retrained capped control
it is parity on wins (+0.6 points, p 0.61) and +$2.4k of margin (z +2.6).

**The trainable trunk retrained under the cap got worse** on every measure,
handing the opponent +$5–6k. The trunk-drift sign pattern now holds in three
runs: every trainable trunk trained by PPO raised the opponent's bank (B2
+$3.3k, B3 +$6.0k, B5b +$5.4k) while every frozen trunk did not.

**Caveat that applies to every arm in this track:** one training seed each.
The eval se (0.008–0.010) is the eval's noise for a fixed net; training-run
variance is unmeasured. The +6.0-point frozen result is the largest effect
seen and the one that most needs replication across training seeds before
it is believed at that size.

## 8 Replication (B5c), pre-registered before it ran

`tools/stage_b5c.sh`: the capped control and the capped frozen-trunk arm,
two more training seeds each (1235, 1236), with `--vary-worlds` so each
replicate re-seeds the world seeds and the opponent draw as well as the
minibatch order — an independent training run, not the same worlds
reshuffled. Same B2 BC checkpoints, same configuration. The four new nets
are played on the same 785 games; B2 control, B2 frozen and the seed-1234
capped nets are reused.

- **H-rep-frozen**: the capped frozen arm's paired delta over B2 frozen is
  positive in all three seeds, and the three-seed mean is at least +0.03
  (half the seed-1234 size). Reported: mean ± between-seed sd of the delta,
  of money and of the opponent's bank.
- **H-rep-control**: the capped control's paired delta over B2 control is
  positive in all three seeds; three-seed mean reported with its sd.
- **H-rep-order**: capped frozen vs capped control, same seed, parity or
  better on wins in at least two of three seeds.
- Guards as before, per run.

A seed-1234 result that the other two seeds do not approach is reported as
a lucky draw, and the between-seed sd becomes the error bar every future arm
in this track must clear.

## 9 Replication result (2026-09-25, eval landed 16:21)

Guards clean on all four new runs (KL ≤ 2.5e-7, 11.8–13.1 updates per
iteration, no zero-update iteration). Same 785 games; per seed, paired.

**Capped control vs B2 control — REPLICATES, tightly.**

| seed | wins | Δ win | p | bands + | bank | opp bank | margin |
|---|---|---|---|---|---|---|---|
| 1234 | 63 vs 47 | +0.020 | 0.017 | 4 | +$3,909 | −$492 | +$4,401 |
| 1235 | 59 vs 47 | +0.015 | 0.073 | 4 | +$3,956 | −$784 | +$4,740 |
| 1236 | 58 vs 47 | +0.014 | 0.117 | 4 | +$2,966 | −$633 | +$3,599 |
| mean ± between-seed sd | | **+0.017 ± 0.003** | | | +$3,610 | −$636 | +$4,246 |

Sign holds 3 of 3; three-seed mean 8.7 between-seed se from zero and ~3.6
eval-se from zero. H-rep-control passes. The land cap is worth **+1.7
points of win rate and +$4.2k of margin**, and the control's training-run
variance is small (sd 0.003).

**Capped frozen trunk vs B2 frozen — sign replicates, size does not.**

| seed | wins | Δ win | p | bands + | bank | opp bank | margin |
|---|---|---|---|---|---|---|---|
| 1234 | 68 vs 21 | +0.060 | <0.0001 | 5 | +$13,929 | −$10,263 | +$24,192 |
| 1235 | 36 vs 21 | +0.019 | 0.003 | 5 | +$8,777 | −$5,377 | +$14,154 |
| 1236 | 30 vs 21 | +0.012 | 0.049 | 5 | +$5,905 | +$1,926 | +$3,979 |
| mean ± sd | | +0.030 ± 0.026 | | | +$9,537 | −$4,572 | +$14,108 |

Sign holds 3 of 3 and the mean sits exactly at the +0.03 bar, but the two
fresh seeds are +0.019 and +0.012: **the seed-1234 result was a lucky
training draw**, and the "competes for the market" reading (opponent −$10k)
does not hold on the fresh seeds (−$5k, +$2k).

**Capped frozen vs capped control, same seed — FAILS: the board arm is
worse.**

| seed | wins | Δ win | p | bank | opp bank | margin |
|---|---|---|---|---|---|---|
| 1234 | 68 vs 63 | +0.006 | 0.61 | +$4,097 | +$1,733 | +$2,364 |
| 1235 | 36 vs 59 | −0.029 | 0.006 | −$1,102 | +$6,912 | −$8,013 |
| 1236 | 30 vs 58 | −0.036 | 0.0002 | −$2,983 | +$14,064 | −$17,047 |
| mean ± sd | | −0.020 ± 0.023 | | +$4 | +$7,570 | −$7,565 |

H-rep-order fails, one of three. On average the frozen board arm is two
points *worse* than the capped control and hands the opponent $7.6k more —
the yielding pattern, now seen in a frozen trunk too. The board arm's
between-seed sd (0.023–0.026) is seven times the control's: **board nets
are unstable across training runs**, and every prior board-arm conclusion
in this track (B2 freeze order, B5b's "best agent") sat inside that noise.

**What survives replication:** the land cap on the no-board control, +1.7 ±
0.3 points and +$4.2k margin, three independent runs at 58–63 of 785 against
B2's 47. **What does not:** any board arm beating the control. The board
track is closed for win rate: across seven valid board arms (B2 frozen and
trainable, B3 trainable, B5b frozen and trainable, B5c frozen ×2) none
reliably beats the no-board control, the mean is worse, the variance is
large, and the market-yielding sign recurs.

**Carry forward:** the capped control, no board. Next: the deployment check
against the shipped champion on the same games (needs a bundle arm in the
eval script), then the capacity gate as a third land mode, then longer
training on this configuration. The between-seed sd of 0.003 for the control
is the error bar future control-family arms must clear; anything with a
board needs three seeds before it is read at all.

## 10 Deployment check, pre-registered before it ran

`tools/deploy_check.sh`. The shipped champion bundles `champ-h8` (the
strongest local champion, the base every gate uses) and `champ-h1` (the one
recorded as submitted) play *our* side on the same 785 held-out games,
through a new bundle arm in `rl/session_eval.py` and `rl/rollout.py`
(`learner_bundle`). The three capped controls (seeds 1234–1236) and B2's
uncapped control are reused game-by-game. Baseline `champ-h8`.

- **H-deploy**: the capped control beats `champ-h8` on win rate, p < 0.05,
  positive in ≥ 4 of 6 bands, in at least two of three seeds, with the
  three-seed mean positive. Money reported alongside; the ladder is
  win-based, so wins decide.
- **H-deploy-h1**: the same against `champ-h1`, reported.
- Passing makes the capped control a submission candidate, with
  `MAX_QUADRANTS = 3` baked into the bundle at export (a bundle has no
  environment). Failing says the RL track has not yet caught the heuristic
  champion on the ladder's own terms, and the cap should be tested on the
  champion itself.
