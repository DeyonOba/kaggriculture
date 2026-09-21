# Changelog

All agent versions for the Kaggle **Kaggriculture** competition, newest first.
Detailed per-version documentation: [docs/VERSIONS.md](docs/VERSIONS.md).
Benchmarks are arena results on fresh seeds, both player positions, unless noted.

## melon glut risk — 2026-09-21 · the sd table IS inverted, and fixing it loses money

The critique is right and stronger than stated: melon is in **no shop at all** (README Town
Buildings — bakery/pizza/brunch/yarn/ice-cream/pet-cafe/smoothie/farmers-market, melon in none),
so its only demand is the town centre's flat 1/day. `k-comp5`'s `_RISK_SD` is fitted on the
**shop draw**, so zero shop demand reads as zero variance:

| crop | shops | base | P(I0+T) | glut drop | k-comp5 sd |
| --- | --- | --- | --- | --- | --- |
| **MELON** | **0** | 250 | **$1** | **100%** | **0.00** ← least risky |
| STRAWBERRY | 4 | 120 | $1 | 99% | 0.16 |
| CARROT | 3 | 35 | $10 | 71% | 1.07 |
| TOMATO | 2 | 60 | $24 | 60% | **1.60** ← most risky |
| WHEAT | 5 | 25 | $20 | 20% | 0.09 |

Exactly inverted. `tools/make_glut_variant.py` replaces the sd with the structural glut
coefficient (fractional price drop after one field's supply, computed from `MARKET_PARAMS`, no
fitting) and drops the shop-draw decay, since glut risk never resolves and for melon cannot.

| arm | pooled, 692 worlds | z | win |
| --- | --- | --- | --- |
| `x-glut25` (λ 0.25) | **−$580 ± 424** | −1.37 | 0.350 / 0.358 |
| `x-glut50` (λ 0.50) | **−$1,230 ± 520** | −2.37 | 0.355 / 0.335 |

**Monotone in λ** (0 → $0, 0.25 → −$580, 0.50 → −$1,230) — a real effect, and harmful.

**Why: `_rent_pick` already charges for glut.** It prices the whole block against the inventory
its own units create (`econ.sell_revenue(crop, inv[crop][k], total)`), walking the price down
unit by unit along `above_func`. Melon's 100% coefficient is not missing information, it is the
same information — the penalty charges it twice, and halving the rent of the $250-base crop the
opening is built around is a tax on the best tile in the game. This also explains
`k-comp5 − r-mrg100` = z 0.54 on 1,131 worlds: any risk term bolted onto a rent that already
prices self-impact is near-redundant, and the damage scales with λ.

Sixth instance of the recurring law — observation correct, intervention refuted. The half the
rent rule genuinely cannot see is the **opponent's** future supply into the same pool; that is
`k-diff`, and it gated +$362 ± 1,136 (z 0.32).

## hiring rollout ported + gated — 2026-09-21 · money +$1,215 (z 2.76), wins FLAT -> hold

Re-rolled on `x-rentfx` rather than champ-h8 (a schedule is only meaningful for the stack it
was fitted on; `--base` is now a flag so this cannot be silent again). Rollout on 8 evaluation
worlds: −$26,984 → −$22,036 (**+$4,948**), moving six days — 2 (4→2), 3 (5→7), 7 (9→8),
26–28 (10→11). Days 10–25 untouched: the 11-hand plateau is already optimal.

| block | money vs `x-rentfx` | z | win | McNemar |
| --- | --- | --- | --- | --- |
| seed 0 | +$939 ± 797 | 1.18 | 0.350 → 0.361 | 28 c / 24 b |
| seed 1 | +$930 ± 768 | 1.21 | 0.344 → 0.350 | 23 c / 21 b |
| seed 2 | +$1,776 ± 718 | 2.47 | 0.335 → 0.312 | 25 c / 33 b |
| **pooled, 1,038 worlds** | **+$1,215 ± 440** | **+2.76** | **0.343 → 0.341** | **76 c / 78 b, p 0.94** |

**Money clears 2 se; wins are dead flat.** Verdict **hold, not ship** — the rating is
win-based, and `match-all` already taught us a +$1,381 margin gain can cost head-to-head wins.
Contrast the rent rule on the same instrument: +$4,336 **and** +4.5pp win rate at p 0.0027.

**In-sample +$4,948 → out-of-sample +$1,215, a 75% shrink**, exactly as `rollout_hires`'
docstring predicted (cost-improvement only guarantees the evaluation worlds). The original
+$3,287 ± 1,528 was probably the same shape; the per-day hiring schedule is worth about a
quarter of its in-sample number. `x-hirefx` built and gated; `kagent/` unchanged.

### `docs/research_reading.md` section 6 written

The gap flagged earlier is closed with the papers named: **UNREAL**
([arXiv:1611.05397](https://arxiv.org/pdf/1611.05397)) for densifying a reward that is one
number at step 719 against a per-world sd of $34,749 — the measured reason residual PPO
collapsed — with auxiliary targets that already exist and already validate (board CNN
+0.116/+0.137 R², market an exact identity); **DreamerV2**
([arXiv:2010.02193](https://arxiv.org/pdf/2010.02193)) for imagination rollouts, whose only
possible gain here is compute since the engine is already an exact simulator, so it must beat
`fast_env`'s bit-identical 1.70× by more than its own error; and the **Maia** line
(arXiv:2409.20553) for the unrealised spatial piece — a distribution over board
representations rather than a value head over aggregates.

The intended arc **BC → RL → Maia-style spatial policy** has never got past the first arrow.
Recorded with the bar the opponent-model use case has to clear: `k-diff5`
(`rent − λ × their visible holding`) is the direct contest proxy and gated **+$362 ± 1,136
(z 0.32)** — pricing a tile at its own harvest-day price already absorbs most of what their
board says, because their supply lands in the same inventory the forecast reads. And the
workflow is written down: spec → research → use-case ideas → experiment → evaluate/test →
commit → train → test over baseline, with the two steps this project skips called out.

## rent planner on the fixed executor — 2026-09-21 · +$4,336 ± 633, z 6.85, WIN RATE +4.5pp

### The risk analysis was already applied, and its increment is null

`tools/fit_risk.py` / `volatility.py` / `make_risk_variant.py` exist and were applied to
`_rent_pick` in three modes. The recorded arms look like risk wins — `k-comp5` +$4,017 ± 1,040
(z 3.86), `k-lear5` +$3,163 ± 1,012 (z 3.13) vs champ-h8 at 270 worlds — but every `k-*` arm is
built on `r-mrg100`, which already carries `_rent_pick`; champ-h8 does not. Those are **rent +
risk** against a base with neither. `docs/metric_harvest_spec.md` already separated them:
**`k-comp5 − r-mrg100` is z = 0.54 on 1,131 worlds.** Risk-adjusting the rent rule does nothing,
at 4× the sample size anything else here has used — consistent with the fitted variance model
being weak (best R² 0.135, negative for TOMATO and WOOL): *risk level is knowable, risky states
are not.* Nothing to re-run.

### What it sat on was never shipped

`_rent_pick` prices every plantable crop at **its own harvest-day price with self-impact** and
takes the best $/tile-day — the forward market model, this project's first fitted-model win
(+$3,185 ± 768, replicated; "+$2,605 ± 500 … the one thing that paid"). It lives in `r-mrg100`,
which differs from champ-h8 in `planner` alone, and it is **not in the champion**.

`x-rentfx` = `h12-ufh` + that planner:

| block | money vs `h12-ufh` | z | win | McNemar |
| --- | --- | --- | --- | --- |
| seed 0 | **+$4,208 ± 908** | 4.63 | 0.303 → **0.350** | 34 c / 18 b, p 0.037 |
| seed 1 | **+$4,464 ± 883** | 5.05 | 0.301 → **0.344** | 32 c / 17 b, p 0.044 |
| **pooled, 692 worlds** | **+$4,336 ± 633** | **+6.85** | **0.302 → 0.347** | **66 c / 35 b, p 0.0027** |

**The only result in this audit that clears the bar on the metric the rating actually uses.**
Everything else was null-positive at best: underfoot +$336 (z 0.51), hoist +$880 (z 1.34),
BC +$1,233 (z 1.95). Landed in `kagent/planner.py`; **`dev-exec` verifies at +$0 / 0 discordant
worlds against `x-rentfx`**.

### The pattern is now three for three

`docs/board_state_plan.md` called it: both replicated wins here — the hiring rollout
(+$3,287 ± 1,528) and the rent rule — came from replacing a **constant** with a state-dependent
rule computed by **arithmetic**, not a fitted model. This session confirms it from the other
side: residual planner head inert (+$4 ± 6, 0 discordant), full neural planner −$22,887, melon
fertilisation copied from the corpus −$14,897, arithmetic rent rule **+$4,336**.

**Next: the other unshipped arithmetic rule** — the per-day hiring schedule from
`tools/rollout_hires.py` (+$3,287 ± 1,528 out-of-sample) replacing `CHAMP_HIRES`' 30 hardcoded
constants. Same shape, same evidence class, never run on the fixed executor.

## old planner weights on the fixed executor — 2026-09-21 · the residual head is INERT

The BC ranker flipped −$1,052 → +$1,233 on `h12-ufh`, so the trained **planner** heads deserved
the same retest: `rl/runs/g10..g14e/best_weights.json` were all scored on the broken dispatch
and abandoned. `rl/export.py` cannot answer it (it writes `kagent/weights.py` and builds from
the dev tree, not the gated bundle), so `tools/make_neural_variant.py` patches the frozen
bundle's `weights` module directly. `policy.py` dispatches
`BANDIT_RULES → CHAMPION_PLAN → _LAYERS`, so a neural arm needs both off — which makes
`h13-weights` (rules off, no net) the honest baseline rather than `h12-ufh`.

| arm | head | money | z | win | discordant |
| --- | --- | --- | --- | --- | --- |
| `n1-g14e` | residual offsets over the champion plan | **+$4 ± 6** | 0.66 | 0.277 | **0 / 0** |
| `n1-g13` | full neural planner (rl-v9 lineage) | **−$22,887 ± 1,105** | **−20.71** | **0.176** | 38 / 3, p<0.0001 |

**The residual head is inert, not held back.** Zero discordant worlds in 346, ±$4 pooled, $3 on
a single full game. It reproduces `docs/VERSIONS.md`'s own gen-14 account: the BC prior was too
peaked (P(zero offset) 0.996), and after softening it learned mostly to *stop* deviating, its
argmax playing identically to champ-h3. The weights encode "do exactly what the champion plan
says", so no executor fix can revive them. `n1-g13` simply prices the abandoned rl-v9 lineage
against the current field.

**This does not contradict the BC result.** The BC ranker replaces `pick`'s ordering *inside*
the executor, so a disobedient executor destroys its signal — real coupling, and it flipped. A
planner head sits *above* the executor and emits quotas; its failure here is that it learned
the identity function. Different layer, different failure, and only the executor-coupled one
transfers. **There is no trained planner artifact worth reviving — planner work has to be
re-fitted, not re-scored.**

Also: `docs/research_reading.md` gains a section 6 flagging that the board-state encoder / board
CNN line — the most recent modelling work, written up in method detail across four docs — has
**no citation anywhere**. Recorded as an explicit TODO rather than filled with a guessed
reference.

New: `tools/make_neural_variant.py`.

## dev-tree drift gated — 2026-09-21 · it was ONE line; melon fertilisation REFUTED at −$14,897

### 100% of the drift is `weights.py`

`tools/make_swap_variant.py` swaps exactly one `kagent/` module into the gated bundle. All
arms vs `h12-ufh`, 346 held-out worlds:

| arm | dev diff lines | money | win |
| --- | --- | --- | --- |
| `h13-weights` | 2 | **−$538 ± 433** | 0.303 → **0.277** (14 b-only / 5 c-only, p 0.064) |
| `h13-planner` | 82 | **+$0**, 0 discordant | — |
| `h13-econ` / `h13-features` / `h13-market` / `h13-rest` | 36 | **+$0**, 0 discordant each | — |
| `h13-all` | 120 | −$538 ± 433 | identical to `h13-weights` |

`h13-all` reproducing `h13-weights` exactly proves the decomposition is complete: every other
module is inert, including the 82 planner diff lines (comments and unreachable branches).

The cause: `BANDIT_RULES = None` in `kagent/weights.py`. `premium_scale`/`carrot_day` survive as
`KNOB_DEFAULTS`, so what was lost is the two **conditional** rules — champ-h6's cows 6 when no
milk consumer (+$1,989 ± 551) and champ-h7's sheep 10 at day 12 on wool ≥ base (+$1,034 ± 151).
They fire in ~40% of worlds, hence a −$538 pooled cost that shows up mainly as **9 net lost
wins in 346**. Restored from the frozen champion. **`dev-exec` now gates +$0 with 0 discordant
worlds vs `h12-ufh`** — the dev tree and the gated artifact are behaviourally identical and
`tools/submit.sh` is safe to build from.

**Stale cache, caught.** The first re-gate after restoring read −$2,058 ± 564 — `ladder_gate`'s
cache keys on `(agent, n, seed)` and encodes neither the corpus nor the agent's BYTES, so a
rebuilt bundle reuses the old agent's numbers. Second guise of the same failure. The cache now
stamps a SHA-1 of `main.py` and recomputes on mismatch.

### Melon fertilisation: +50pp capability, −$14,897

`FERTILIZE.one_time` was the suite's worst rule (20% / 0% on every executor). Of its ten mined
cases **5 are MELON at ages 6–8 and `want_fert` has no melon clause**; the rest are wheat/carrot
blocked by the `fert_price < 40/50` guard. Adding
`(crop == "MELON" and 6 <= age <= 8 and fert_price < 100)`:

| | capability | rigid | ladder vs h12-ufh | win |
| --- | --- | --- | --- | --- |
| `p1-mel60` | 20% | 0% | **+$0**, 0 discordant | 0.303 |
| `p1-mel100` | **70%** | **30%** | **−$14,897 ± 1,939 (z −7.68)** | **0.220**, p < 0.0001 |

The largest single regression in this project. `p1-mel60` being a no-op shows fertilizer never
drops below $60 inside a melon window, so the clause only fires near base — on 12–16 opening
melon tiles across three days, i.e. up to ~48 FERTILIZE actions, each burning a ~$100 asset we
otherwise sell, each a `P_TEND` turn taken from strawberry fertilisation (2 × ~$165), each
inflating `plan.fert_reserve` so the stock is withheld from the market too. And the payoff was
never there: melon `first_yield_day` is 10 and the engine refuses HARVEST before it, so
reaching the cap at age 8 frees nothing — fertilizer buys only insurance against a missed
watering, and in-window watering is already 92%.

**Fifth instance of the recurring law, and the sharpest.** The 2000+ seats really do fertilise
melons at ages 6–8 — that is where the cases came from — and copying it costs $15k a game. A
20% capability score says the executor *cannot* do a thing; it never said the thing was worth
doing.

**So the planner work does not start from "what does the corpus do that we don't."** It starts
from the knobs the transfer gate put back in scope — the arms whose champ-h8 effect was inside
~2 se, the only ones the old executor could have been carrying: `straw_target 40`, the
`wheat_engine` variants, `land2_margin`, `carrot_cap`, `straw_early`.

New: `tools/make_swap_variant.py`, `tools/make_fert_variant.py`.

## task.DIG 60% -> 80% — 2026-09-21 · a second scorer off-by-one, plus claim-order

Three causes, one of them mine.

1. **A second scorer off-by-one (1 of 4 cases).** `task.DIG` has `settle_days = 0`, so its
   window is `[T, end of day]` — and the mined cases cluster late. `#05` is day 15 **hour 23**:
   a one-turn window, on which the executor **did** DIG (`step 383, unit 2, DIG`; weed cleared
   384). `done` was only evaluated inside the window, but an action on step S first shows up in
   step S+1's observation, so every command obeyed on its last turn scored as a failure.
   Fixed (`done` now checked to `settle_end + 1`, injection still stops at `settle_end`):
   **DIG 60% → 70%**.
2. **Claim-order on the tile you are standing on (1 more case).** The Part-8 underfoot branch
   lives inside pass 1's per-unit loop, and `claimed` is set on the step *toward* a task, not on
   arrival — so a low-index distant unit reserves the weed through its own-zone `pick` and the
   unit already standing on it is refused (`#00`: unit 4 on the weed at 116, went SOUTH, dug at
   129). `--hoist` runs the underfoot check as its own pass before any zone pick, skipping
   animal carriers so `deliver.PLACE` keeps priority: **DIG 70% → 80%**.
3. **Weeds are P_FILLER by design.** `weed_pri = P_GROW if len(empties) - cursor < 6 else
   P_FILLER`, and the underfoot branch takes `pri < P_FILLER` only — deliberately, since
   `v-tile` grabbed filler underfoot and gated −$4,907. Hence unprompted DIG 70% vs commanded
   80%: the executor *can* dig on demand and declines when there is spare ground. That is an
   economics question, not an execution one.

| | champ-h8 | h11-uf | **h12-ufh** | match-all | a-commit |
| --- | --- | --- | --- | --- | --- |
| `task.DIG` | 60% | 70% | **80%** | 70% | 90% |
| `task.BUILD_COOP` | 71% | **100%** | 86% | 86% | 100% |
| `deliver.FERTILIZE` | 82% | 94% | **100%** | 82% | 88% |
| **ALL** | **89%** | 95% | **96%** | 90% | 94% |

**Gate:** `h12-ufh` +$1,281 ± 903 and +$480 ± 952 → pooled **+$880 ± 656 (z 1.34)**, wins 45 vs
47; `h11-uf` pooled +$336 ± 658. Both positive in both blocks, neither clears 2 se. The hoist
costs `task.BUILD_COOP` 100 → 86% and its money edge over h11-uf is inside noise — **landed in
`kagent/scheduler.py` because it is the principled fix for the measured mechanism, not because
the money resolved.** `kagent/` now reads obedience **96%**, DIG 80%, FERTILIZE 100%,
`pytest tests/` 25 passed / 2 xfailed.

**Third scorer bug in one instrument.** (1) `op[1]` IndexError crashed `build_tasks` into
SafeAgent's all-PASS; (2) a `--cmd` run overwrote the full results file; (3) `done` missed the
last turn. All three made the executor look worse than it is and none announced itself. The
wrapper can no longer raise, failures land in `harness_error` and print a loud warning, subset
runs merge. Lesson: a hook inside the agent's own call path sits inside `SafeAgent`'s blast
radius, so it must be incapable of raising.

## CORRECTION — 2026-09-21 · a harness bug understated every obedience number

Found by reading the replays. `exec_commands.run_case` computed
`case["_kind"] = (case.get("op") or [None,"GOOSE"])[1]`, which raises **IndexError for every
argument-less op** (WATER, HARVEST, DIG, CARE, COLLECT_FERTILIZER, BUILD_*, DROP). The wrapper
replaces `build_tasks`, which runs inside `policy.act`, and **SafeAgent catches everything and
emits all-PASS** — so on the commanded turn the agent did nothing and the command was not
injected until T+1. A crashed harness was indistinguishable from a refused order. A second
instance (`plan.PLANT` reading the crop from `op[1]` on cases mined from `BUY_SEED`) hit 42
more. Both fixed; the wrapper can no longer raise, failures land in `harness_error` and `run`
prints a loud warning; subset runs now merge instead of truncating the results file.

| | reported | **corrected** |
| --- | --- | --- |
| champ-h8 | 85% | **89%** |
| `h11-uf` | 92% | **95%** |
| `dev-exec` (`kagent/`) | 90% | **95%** |
| `match-all` | 85% | **90%** |
| `a-commit` | 87% | **92%** |
| `h10-cmt` | 83% | **87%** |

h11-uf fixes **15 of champ-h8's 24**, breaks 2, leaving **11**.

**What changes.** `match-all` is NOT identical to champ-h8 — 90% vs 89%, and it lifts
`task.DIG` 60→70% and `BUILD_COOP` 71→86%. The earlier "index-order claiming is refuted,
identical on 213/217" was measured on contaminated runs and is too strong: claim order is a
**minor** contributor, not a non-contributor. `plan.HIRE` and `plan.PLANT` are 100%, not 75%
and 96% — the whole plan layer obeys perfectly.

**What does not change.** Every ladder/money gate (the harness plays no part): the underfoot
fix's +$337 ± 658, `h10-cmt` −$2,162, the knob transfer, the BC flip. Also the no-op audit and
the entire unprompted rule suite — `rule_tests.py` injects nothing and never touched this path,
so 79%→82% capability and 32%→51% rigid stand. Commitment still fixes nothing (87%, below
baseline) and the zone partition was still the dominant cause.

**The four replays that prompted it.** Three were the renderer's off-by-one, not bad tiles:
(1,5) d27 was WHEAT dying that night and we DO water it; (4,1) d10 was empty, not a crop, and
we build the coop 3 turns late; (9,1) d28 tomato was harvested 4 turns late. The fourth is
real: (1,0) d16, a strawberry with `fertilized_until_day 18` on its production night — a unit
stood on it and walked SOUTH. README Harvest Yields makes that a 2-unit yield instead of 1, so
it costs a ~$165 unit.

**Indexing.** `day` and `hour` are both 0-indexed (README; engine `obs0.hour = next_step %
turns_per_day`), hour 0–23, so "turn 22" is the 23rd turn. The off-by-one is the renderer:
`env.steps[t]`'s observation is the state AFTER step t's actions. `window_for()` now subtracts
one and renders 2 in-game days of lead-in plus 10 turns of tail (~2–4 MB per case).

## 4th executor + BC re-opened — 2026-09-21 · knobs transfer, BC does NOT

### h11-uf as a fourth executor: the knob conclusion holds

| knob | champ-h8 | match-all | a-commit | **h11-uf** |
| --- | --- | --- | --- | --- |
| geese 4 | −5,618 ± 885 | −4,952 ± 886 | −6,178 ± 881 | **−6,206 ± 953** |
| carrot_standing 16 | −4,855 ± 854 | −4,705 ± 840 | −2,193 ± 885 | **−3,791 ± 904** |
| hires_delta +2 | −3,194 ± 687 | −2,742 ± 696 | −372 ± 901 | **−3,548 ± 631** |

**E0 vs E3 Spearman +0.94, sign 3/3 on every arm champ-h8 resolves.** The better-obeying
executor agrees with champ-h8 more closely than any other pair except the near-identical
E0/E1 — and `hires_delta`, whose magnitude collapsed on a-commit (−$3,194 → −$372), comes back
to −$3,548. a-commit was the outlier, not "a better executor". Closed knob negatives at >2 se
stay closed.

### BC flips sign on the fixed executor — the premise holds HERE

`exec_bc.py export --mode banded` swaps the learned ranker in for `pick`'s
`(urgent, distance, priority)` score and changes nothing else. Same weights both times.

| arm | base | money/game | z | win |
| --- | --- | --- | --- | --- |
| `bc-h8` | champ-h8 | **−$1,052 ± 902** | −1.17 | 0.295 → 0.277 |
| `bc-h11` block 0 | h11-uf | +$888 ± 906 | 0.98 | 0.303 → 0.318 |
| `bc-h11` block 1 | h11-uf | +$1,578 ± 880 | 1.79 | 0.301 → 0.327 |
| **pooled, 692 worlds** | h11-uf | **+$1,233 ± 632** | **+1.95** | 60 c-only vs 46 b-only |

**Same ranker, different executor: a $2,285/game swing.** The recorded BC null (z = +0.69 over
4,800 worlds) was measured inside an executor that disobeys 15% of direct orders — the ranker
picked better targets and the zone partition overruled it. Does not clear the bar yet
(z 1.95 vs the ~2 se rule, McNemar p 0.21), but it is the only learned component in this
project that has ever read positive, and it is now the strongest open lead. Next: a third
block, then re-fit the ranker on candidate sets from the *fixed* `assign`.

### Remaining executor failures: 18 of 217

16 held over, 2 newly broken (`deliver.FEED#03`, `task.DIG#08`). All 14 tile cases still
*arrive* on the tile and **10 of 14 are commanded at hours 18–23** — the residue is end-of-day
logistics (section 6 sends loaded units to the shed from hour 21; the shuttle grabs units
before every work pass), not zone blindness. Caveat: `ASSIGN_LABEL` still reports them as
`zone` because the underfoot branch now lives inside pass 1, so the stage attribution is no
longer separable — the hour concentration is the reliable signal. Replays for all 18 at
`tests/replays/h11-uf-commands/<command>/`.

## underfoot fix — 2026-09-21 · SHIPPED to kagent/scheduler.py · obedience 85% -> 92%

The fix for the cause found above. One branch in pass 1, between the delivery check and the
own-zone check: if there is an unclaimed task **on the tile I am standing on**, any zone,
`pri < P_FILLER`, do it. Two narrowings, both because `v-tile` already gated −$4,907 — it runs
*after* deliveries (so `deliver.PLACE` stays 100%), and it excludes FILLER.

| | champ-h8 | `h11-uf` |
| --- | --- | --- |
| commands obeyed | 85% | **92%** |
| median latency | 5–9 turns | **2** |
| unprompted capability T3 | 79% | **82%** |
| unprompted **rigid** T3 | 32% | **51%** |

Per command: WATER 73→87, HARVEST 78→94, FERTILIZE 76→94, COLLECT_FERTILIZER 92→100,
BUILD_COOP 71→86, FEED 58→75. DIG (60%) and HIRE (75%) unmoved.

**Ladder, two blocks: +$454 ± 874 and +$219 ± 984; pooled +$337 ± 658 (z 0.51), wins 46 vs 44.**
Null on money, null on wins, **positive in both blocks and negative in neither** — and it fixes
the measured defect while breaking nothing. Not a money win; the honest claim is that the
executor now does what it is told at no cost. Rigid agreement nearly doubling is the sharpest
evidence the right thing changed. For contrast on the same instrument: `v-tile` −$4,907,
`h10-cmt` −$2,162, `h9-carry` −$521.

`h11-uf2` (pri < 2, urgent only) is strictly worse: 90% obeyed, +$19 ± 879, win 0.280.

### Landed in `kagent/scheduler.py` — but `kagent/` is not the gated artifact

Suites re-run from a bundle built out of the dev tree: obedience 85→**90%**, capability
79→**82%**, rigid 32→**51%**, `pytest tests/` 25 passed / 2 xfailed.

| arm | money vs champ-h8 | win |
| --- | --- | --- |
| `h11-uf` (frozen champ-h8 + this fix) | **+$454 ± 874** | 0.295 → 0.303 |
| `dev-exec` (all of `kagent/`) | **−$777 ± 869** | 0.295 → **0.275** |

Same fix, same block, opposite sign. The gap is everything else the dev tree has drifted by —
planner, econ, market, features — never gated, currently ≈ **$1,200/game** worse than the
frozen champion. `tools/submit.sh` rebuilds from `kagent/`, so **submitting from it today
would ship a −$777 agent.** Gate the rest of the drift before any submission.

New: `tools/make_underfoot_variant.py`. Result files are now per-agent
(`rl/rule_results_<agent>.json`, `rl/exec_command_results_<agent>.json`) so a later run cannot
clobber a baseline — it clobbered champ-h8's once before this was fixed.

## the 28 unobeyed commands — 2026-09-21 · commitment REJECTED, the zone partition is the cause

Three variants on the frozen champ-h8 bundle, gated on 346 held-out ladder worlds paired
against champ-h8 and re-run through the 217-command obedience suite.

| arm | change | money/game | z | obeyed | fixes / breaks |
| --- | --- | --- | --- | --- | --- |
| `h10-cmt` | commit to a P_SURVIVAL target across turns | **−$2,162 ± 572** | **−3.78** | 83% | **0 / 32, +4 broken** |
| `h10-cmts` | + committed units exempt from the shed shuttle | **−$2,133 ± 571** | **−3.74** | 83% | **0 / 32, +4 broken** |
| `h10-pad1` | route over unbought ground only at depth ≤ 1 | +$34 ± 20 | +1.71 | 85% | 0 / 0 |

- **Commitment is refuted as the mechanism.** It fixes none of the 32 failures and costs
  $2,162/game at z = −3.78. In hindsight it could not work: commitment only helps a unit that
  already had the task assigned last turn, and the 27 failures are units that were **never
  given the task** — they were standing on it. `--shield` rules the shed shuttle out too.
  (`a-commit` fixes 16/32 but breaks 12 and replaces the whole three-pass assignment with a
  value-density scorer, so its gain was never commitment.)
- **The cause, measured.** `ASSIGN_LABEL` (the scheduler's own per-stage debug hook) says the
  diverting stage is `zone` — pass 1, the own-zone round — **68%** of the time. Direct test:
  the commanded tile is in a **FOREIGN zone 65%** of the time. Pass 1 searches
  `zone_tasks[idx % n_zones]` only, so a unit physically standing on the tile is refused it
  because the tile belongs to another worker's zone, and pass 1 sends it back to its own.
  That is `executor_problems.md` **§4**, listed there as untried.
- **Three mechanisms, three verdicts:** index-order claiming (§3) **refuted** — `match-all` is
  identical on 213/217; re-deciding every turn (§2) **refuted** — `h10-cmt` above; the zone
  partition (§4) **confirmed at 65%**. Nothing shipped: the nearest previously-tried fix,
  `v-tile` (same-tile pass before the band passes), gated −$4,907, and §4's own note is that
  planner and executor have to change together.
- **Locked-ground padding.** Measured *before* building: 96.5% of locked-ground unit-turns are
  already at depth 1 and only 25 moves per 4 games go deeper, so this makes a near-invariant
  into an invariant. `h10-pad1` (multi-source BFS depth grid in `_step_toward`, cached on the
  quadrant key) eliminates the single genuine deep excursion; the residual depth-2 turns are
  all `(5,5)` at hour 0, where the engine's `_spawn_hand` places a hand regardless of locks.
  +$34 ± 20, zero discordant worlds, nothing broken — free, and a guarantee rather than a win.
  Locked tiles are passable, not obstacles, so the detour costs no distance; what it buys is
  steps taken over workable ground (`actions per tile arrival`, ours 1.67 vs 2.10).
- New: `tools/make_survcommit_variant.py`, `tools/make_pad_variant.py`,
  `tests/test_locked_ground.py` (asserts: no tile op on LOCKED, shed ops only from
  shed-access tiles, routing within the padding, no silently-voided PLANT or market order).

## commanded intents + transfer gate — 2026-09-21 · it ARRIVES and walks away; the negatives stay closed

Two follow-ups to the rule suite: attribute its 21% misses between planner and executor, and
test the premise that nothing fitted on champ-h8 transfers.

### `tools/exec_commands.py` — the executor disobeys 15% of direct orders

Inject the intent into the SAME mined states and see if it lands. Three layers, all through
the real machinery: a `Task` at `P_SURVIVAL` on a named tile (routing + execution), a `Task`
whose `need` is only in the shed (provisioning + carry), and a `DayPlan` field re-asserted
every turn (`build_tasks` + market controller). 217 commands, 15 intents.

- **champ-h8 obeys 85%.** Every purely economic PLAN intent lands 100% — buy the quadrant,
  build the coop, buy the animal — so `plan.COOP` at 100% obeyed against 41% unprompted means
  the entire coop gap is **the planner never asking**. Same for `plan.BUY_LAND` (100/50),
  `task.CARE` (100/83), `task.BUILD_PASTURE` (100/50).
- **Every failure is a TASK on a named tile**, and `task.WATER` is *worse* commanded (73%)
  than unprompted (87%) — the unprompted test accepts any qualifying tile, the command names
  one. It waters *a* plant and steps off *the* one. `deliver.FEED` 58% vs 92%.
- **27 of 27 unobeyed tile commands had a unit standing ON the tile**, which then emitted a
  move. Never once did it fail to arrive. That is `executor_problems.md` §1 demonstrated at
  `P_SURVIVAL`, where nothing outranks the order.
- **Which executor change fixes it:** the same 217 commands on three executors sharing one
  planner. `match-all` (global matching) is **indistinguishable** — identical latency on
  213/217, the same 32 failures — which **refutes the claim-order explanation** (§3); if
  index-order claiming were the cause, resolving claims cheapest-pair-first would fix it.
  `a-commit` (cross-turn commitment, §2, listed as *untried*) does move it: WATER 73→87%,
  FEED 58→75%, BUILD_COOP 71→86%, HARVEST 78→89%, overall 85→87% — and costs
  `deliver.PLACE` 100→60%, because a committed unit will not divert to deliver the animal in
  its pocket. `a-commit` gated null on money, so obedience and money are separable: treat the
  obedience number as a diagnostic, never a gate.

### `tools/knob_transfer.py` — "nothing transfers" is too strong on sign, right on magnitude

6 knobs × 3 executors (champ-h8 / match-all / a-commit, one planner, `scheduler.py` only),
each arm gated against **its own** executor on the same 180 held-out worlds.

| knob | champ-h8 | match-all | a-commit |
| --- | --- | --- | --- |
| geese 4 | −5,618 ± 885 | −4,952 ± 886 | −6,178 ± 881 |
| carrot_standing 16 | −4,855 ± 854 | −4,705 ± 840 | −2,193 ± 885 |
| hires_delta +2 | −3,194 ± 687 | −2,742 ± 696 | −372 ± 901 |
| straw_target 40 | −753 ± 602 | −440 ± 543 | −1,095 ± 719 |
| wheat_engine 40 | +61 ± 359 | +156 ± 332 | −459 ± 210 |
| melon_opening 16 | +0 ± 0 | +0 ± 0 | +0 ± 0 |

Spearman +1.00 (E0/E1) and +0.71 (E0/E2, E1/E2); **sign agreement 3/3 on every arm champ-h8
resolves at >2 se**. `melon_opening 16` is already the default, so its +$0 ± 0 across all
three is a genuine zero control.

- **Every resolved knob keeps its sign across a materially different executor** — a-commit
  changes 144 of 217 commanded outcomes. Geese do not become good on a better executor; they
  get worse. The single sign disagreement (`wheat_engine`) is +$61 ± 359 on champ-h8, i.e.
  noise.
- **Magnitudes do move**: `hires_delta` shrinks 9× (−$3,194 → −$372) and stops resolving at
  all. So "hires +2 is catastrophic" was partly executor-carried; "hires +2 is bad" was not.
- **Rule this buys**: reopen a closed experiment only if its effect was inside ~2 se. Geese,
  standing carrots, ±2 hires, straw 21, carrot_day 18, premium 1.3 stay closed; the
  borderline set (straw 40, wheat_engine variants, land2_margin, carrot_cap, straw_early)
  goes back in scope.
- **Not gated:** BC. A cloned policy couples to the executor far more tightly than a scalar
  knob, so `exec_bc.py`'s null could still be executor-carried — re-run that one on `a-commit`
  before deciding. Also: 6 knobs, one seed block, and all three executors share a planner.

- **`tools/noop_audit.py` — the executor emits essentially no illegal actions.** 4 games,
  27,328 ops emitted, **5 silent no-ops (0.02%)**, all `WATER: not_a_plant` (a same-turn race
  where the tile stopped being a plant). **0** tile ops attempted on LOCKED ground, **0**
  PLANT requests voided by the atomic-seed rule, **0** market orders past the cap. It also
  *uses* both permissive locked-tile rules rather than avoiding locked ground — 895 moves
  from a locked tile and 290 shed ops from a locked shed-access tile (a hand spawns on (5,4),
  locked until NE is bought, and picks up from there anyway). This closes the cheapest
  explanation for the 15% disobedience: it is not malformed orders, it is choosing a
  different *valid* action.
- Replays are now filed per action: `tests/replays/<agent>/<ACTION>/<case>.html`, with the
  14.7 MB viewer shared once per agent directory.

Replay filenames carry the target, day, turn and seat
(`DIG.clears_a_weed-04-x4y9-d12-t20-p1.html`).

## rule test suite — 2026-09-21 · TDD ground truth; the executor OMITS, it does not misfire

`tools/rule_tests.py` turns README.md into a test suite. One rule per action; a case exists
only where a **2000+ rated seat performed it** and the README precondition held, so the corpus
defines correctness. Mid-game resume is unfaithful, so each case replays both recorded scripts
from step 0 on the session's own seed, hands OUR executor the wheel at step T, judges it, then
PASSes to the end; our agent is called on every step from 0 so its hour-0 day plans evolve
along the recorded trajectory. **286 cases, 28 rules, 40 reference seats, six day bands.**

Two verdicts — RIGID (exactly step T, exactly that tile) and CAPABILITY (any qualifying tile
until the opportunity closes) — each at three tiers: emitted, accepted by the engine, and the
README rule's **value materialised**.

- **champ-h8: capability T3 79%, rigid T3 32%.**
- **`T1 == T2 == T3` in 27 of 28 rules.** Whenever it emits an action where the precondition
  holds, the engine accepts it *and* the value lands — 0 no-ops, 0 mistimings in 286 cases.
  **Every failure is pure omission**: a state where it never emitted the action anywhere, all
  day. That is a much narrower diagnosis than "the executor is faulty", and it reframes the
  `moves/action` work: a worker that never proposes the task cannot be routed to it.
- **The one exception is CARE** — 92% accepted against 83% paid. README Animal Care banks the
  bonus only on a day the animal was *also* fed; one day-26 case cares an animal that then goes
  unfed, spending the action for nothing. Same coupling `docs/executor_problems.md` §6 found
  from the other side.
- Worst: `BUILD_COOP` 14% (champ-h8 sets `geese = 0`, so it never builds one),
  `FERTILIZE.one_time` 20%, then `BUILD_PASTURE` / `DIG` / `DROP` / `BUY_LAND` at 50%.
  Clean 100%: `HARVEST.one_time_ripe`, `PICKUP`, `PLACE.animal_on_structure`, all four moves,
  `COLLECT_FERTILIZER`, `HIRE`.
- **Max yields, answered:** it gets every structural rule right and the *cap* rule wrong in
  three places, all the same omission — it never compares `yield_units` to the cap before
  pricing an action. Watering priority is window-membership only, so a melon gets `P_TEND`
  waterings at ages 11–12 (9–12 fertilised) for zero units; `max_held` is read into a local and
  never used, so a cared sheep's `4 + 4 -> min(6, 8)` destroys two wool and the bank that paid
  for them (`HARVEST.animal_before_clip` missed on both late-game cases); and README's Object
  Types "Time to Max Yield" (melon 10) is not the engine's `max_yield_day` (12), which the
  executor uses for both the window and the ripeness test.
- **Artifacts:** every case writes a watchable replay to `tests/replays/<agent>/` — a windowed
  render around the test (1.4–4 MB) and, for failures, the whole season (~30 MB). 347 files,
  2.3 GB. The 14.7 MB visualiser is externalised once per directory as a classic `defer`
  script so the pages still open from `file://`; `tests/replays/serve.sh` serves them if a
  browser objects.
- **This is a capability test, not a value test.** 100% means the executor *can* do the thing
  where a strong player did, not that doing it more earns money — four recorded cases say the
  opposite. Find gaps here; decide with `tools/ladder_gate.py`.

## animal chain vs README — 2026-09-21 · two inert fixes shipped, one refuted

Audit of the `BUY_ANIMAL -> PICKUP -> walk -> PLACE` chain in champ-h8's executor against
`README.md`. Four assumptions are genuinely false; **none of them is worth money**, and the
one that changes behaviour makes the agent worse. Full write-up:
[docs/animal_chain_audit.md](docs/animal_chain_audit.md).

- **FA-1 / FA-2 — a head in a carrier's pocket is invisible to the planner.**
  `make_champion_plan` reads `herd + shed` for the buy target and
  `animal_buys + shed` for the structure pipeline, while `scheduler.build_tasks` counts
  `shed + pockets` in `available`. The two halves of the chain disagree: the buy path orders a
  $300–500 replacement for a head we already own, the build path never queues it a coop, and
  README Terrain says `DIG` cannot recover the tile once an animal lands on it.
  **Gated bit-identical: +$0, 0 discordant worlds of 346.** The day plan is rebuilt only at
  hour 0 and the engine empties every inventory at midnight
  (`private["inventories"] = [{}]`, kaggriculture.py:882), so the missing term is always zero.
  Shipped into `kagent/` anyway — `tools/rollout_plan.py` and the MPC provider re-plan
  off-cadence and would hit it.
- **FA-4 — the shed-overflow drop can name an animal.** The block counts cargo *excluding*
  `GOOSE/COW/SHEEP` and then picks `max(inv, key=inv.get)` over the *whole* inventory; README
  resolves `PLACE`'s animal branch first, so on a matching empty structure that puts one head
  on the tile and nothing in the shed. Reachable — the build queue sorts empties by distance
  to the shed and all four shed-access tiles tie for the minimum (9 of 46 structures land on
  them) — but latent: 0 occurrences in six traced games, **+$0, 0 discordant worlds**. Shipped
  as a guard.
- **FA-3 — `batch = 2` for animals. REFUTED, not shipped.** README Animals: `PLACE` "places
  one animal ... The `n` argument is ignored", and the code's own comment two lines up says
  "any unit may carry **one** to its pasture". The rule reading is right; the inference that
  head #2 is dead weight is not. Two head in one pocket is **one** shed trip serving **two**
  structures, which cluster by the shed. One head per carrier cost **12 extra PICKUP actions
  per 3 games for the same 48 placements** and made midnight bounces *worse* (1 → 7). Gated
  over **two blocks, 692 held-out ladder worlds**: money −$521 ± 601 (z −0.87, null) and
  **wins 54 base-only vs 34 candidate-only, McNemar p = 0.042** — a replicated regression on
  the metric the rating actually uses.
- **Also measured and rejected:** capping the structure build at
  `min(animal_buys, bank // cost, shed_room)` — the faithful reading of "BUY_ANIMAL is an
  order that can be refused". The champion plan buys animals mid-day on wool/milk cash by
  design, so hour-0 bank is near zero and the gate suppressed almost every build;
  `stranded_shed_turns` doubled 107 → 216. Only 4 of 24 GOOSE orders actually go unfilled.
- **Note for anyone reading the goose lead:** champ-h8 sets `knobs["geese"] = 0`, so the
  COOP→GOOSE branch never executes in the shipped champion (0 `BUY_ANIMAL GOOSE` in 3 traced
  games). `GEESE_TARGET = 32` and friends live in the *reactive* `make_day_plan`, which is
  only the SafeAgent fallback. The chain measurements above use `g-goose6` to load it.
- New: `tools/make_chain_variant.py`, `tests/test_animal_chain.py` (asserts the four README
  rules against a bundle; the FA-3 pair is `xfail` with the refutation in the reason string).

## tape-sel — 2026-09-14 · SUBMITTED · day-6 selector, +$5.3k/+$4.9k over tape-v2

Two arms over the 58-tape cluster, chosen at step 144: **a yarn store among the first two shops**
takes one continuation, anything else takes another. Gated against tape-v2 on two fresh blocks:

| block | ahmed | tschinkel | boatlee | pooled |
| --- | --- | --- | --- | --- |
| 71000 | +$7,222 | +$1,599 | +$7,219 | **+$5,346 ± 1,335** (69/96) |
| 84000 | +$2,906 | +$6,580 | +$5,219 | **+$4,902 ± 1,769** (60/96) |

Cumulative against champ-h8 that is roughly +$25k to +$28k a game.

**Why a yarn store is the signal.** YARN_STORE is the *only* consumer of wool in the game and it
takes 12 units a day, so its presence at day 6 decides whether wool has any season-long demand at
all. It is the single largest swing in the demand draw, which is also why the one conditional rule
that ever survived on the heuristic lineage (champ-h7) was wool-conditioned.

**Training panel** (`tools/tape_panel.py`, 2,320 games = 58 tapes × 2 opponents × 20 seeds,
recording the full day-6 context with each outcome). Held out by seed parity:

| | held-out margin |
| --- | --- |
| random tape from the cluster | −$6,128 |
| best fixed tape chosen on train | +$6,325 |
| tape-v2 (shipped) | +$9,266 |
| per-world oracle | +$27,179 |

Leave-one-world-out selector scores: yarn key **+$14,191**, milk-consumer key +$10,410, two-plus
strawberry shops +$7,782, exact shop pair +$1,314, against +$229 for picking the best fixed tape
the same way. The exact shop pair is *worse* than a coarse binary — 34 distinct pairs over 40
worlds is too fine to estimate. Tape quality correlates +0.714 between the two halves of the
panel, which is what makes any of this estimable.

Caveat kept in view: the yarn arm was fitted on only 8 worlds out of 58 arms, and the key itself
was chosen after trying four. Both fresh-block gates are the real evidence, not the panel.

### How the source agents actually make tapes — measured, not assumed

The 58 cluster seats are 58 distinct episodes, with **58 distinct seeds, 34 distinct day-6 shop
pairs and 23 distinct opponents**, yet byte-identical field actions for days 0–5. The source agent
is **open-loop**: it ignores shops, weeds, prices and opponent entirely for the first six days.
Its later branching does not track the shop draw either — each day-7 branch spans 2–8 different
shop pairs, and only 1 of 58 seats sits in a branch with a unique pair.

So these are recordings, not derivations. The community method (stated openly in the public
notebooks) is to reconstruct a tape from public replays by majority vote across several episodes
of a strong submission; our library is built the same way from our own corpus. Nobody visible is
computing a schedule from first principles, which is why "precompute our own" has no example to
copy — and why a genuine state-conditioned selector adds something the recorded play never had.

### Can we substitute our own schedule after the opening? No

Tape opening then champ-h8's planner, mean margin vs pub-ahmed-v38 over 4 seeds:

| | margin |
| --- | --- |
| champ-h8 alone | −$47,576 |
| tape-v2 alone | −$11,601 |
| tape → ours at day 6 | −$53,433 |
| tape → ours at day 10 | −$46,333 |
| tape → ours at day 14 | −$38,625 |

Handing over is worse than either pure agent at day 6, and the later the handover the less the
damage — our planner is worse than the tape at every point in the season, and it also inherits a
farm it did not plan. The tape's value is season-long choreography, not a strong opening.

## tape-v2 — 2026-09-14 · SUBMITTED · +$20k/+$23k, the first SHIP verdict of the project

A replay tape from episode 107599243 seat 0 (rating 3029, recorded $126,474) in our own corpus,
with weed repair only. Gated against champ-h8 on **two independent fresh seed blocks**:

| block | ahmed-v38 | tschinkel | boatlee | pooled |
| --- | --- | --- | --- | --- |
| 52000 | +$19,241 | +$30,515 | +$11,151 | **+$20,302 ± 2,946** (66/84) |
| 63000 | +$19,219 | +$31,826 | +$19,186 | **+$23,410 ± 2,162** (75/84) |

Positive against every opponent in both blocks. For scale, every heuristic change of the previous
fortnight was worth between −$11k and +$2.4k, and champ-h8's shipped feed-buffer fix was +$2,376.

### What made this one work where `tape-v1` reached only parity

**Tapes cannot be spliced across lineages.** Switching from one tape to another at day 6 collapses
the agent — $80k and $83k alone become $12k and $34k spliced — because the two farms have
different tile layouts by then and every later action lands on the wrong square.

**But 58 of our 301 tapes share byte-identical field actions over days 0–5**, all from rating-3029
seats. Within that cluster splicing is **lossless**: A→B at step 144 reproduces B-alone to the
dollar. That is the compatible set a selector needs, and it exists in our corpus without any
curation.

**Screening the cluster changed the estimate completely.** Over 12 worlds against two public
agents (696 games):

| | mean margin |
| --- | --- |
| random tape from the cluster | −$9,290 |
| best fixed tape (129) | **+$2,449** |
| per-world oracle | +$18,275 |

Even a *random* member of this lineage beats champ-h8's ≈ −$30,000 margin by $20k. `tape-v1` was
chosen from the whole library by recorded margin, which selects for having beaten a weak opponent;
the lineage matters far more than the individual score.

**The selector is not built yet and the headroom is large**: a perfect per-world choice among the
58 is worth a further **+$15,826** over the best fixed tape. tape-v2 is the best *fixed* member,
so the whole selection gain is still on the table.

### Why tapes beat our executor, measured

Worker-assignment structure of today's top-10 rated seats against ours:

| | top 10 | champ-h8 |
| --- | --- | --- |
| actions per worker-day | **11.83** | 9.52 |
| distinct tiles per worker-day | 5.84 | 5.58 |
| spread of those tiles | 6.06 | 6.17 |
| moves per worker-day | **9.16** | 12.35 |
| specialisation | 0.78 | 0.80 |

Our zone executor already reproduces their spatial structure — same tiles, same spread, same
specialisation. They extract 2.03 actions per tile to our 1.71 while walking three steps less per
worker-day. They chain work *and* stay on the urgent task, which a precomputed whole-day schedule
can do and greedy per-turn assignment provably cannot: chaining greedily cost us $6,385 precisely
because it delayed farm-wide urgent work.

## replay-tape architecture — 2026-09-14 · `tools/tape_agent.py`; parity, selector still to come

First architecture change rather than a tuning round: follow a recorded 719-step action script
from a 3000+ seat in our own corpus instead of running the hand-written planner.

**Replay is exact.** A tape replayed in its own world against its own recorded opponent reproduces
the recorded bank to the dollar (episode 107273702, $125,546). The corpus is faithful, so anything
lost afterwards is transfer, not extraction.

**Transfer is where the money goes, and it is idiosyncratic.** Screening 12 tapes over 4 seeds
against two external agents, mean bank in foreign worlds against what the tape scored at home:

| recorded | transfers to | rating |
| --- | --- | --- |
| $152,723 | **$91,697** (win 0.38) | 3028 |
| $158,977 | $89,247 | 3016 |
| $165,971 | $78,238 | 3029 |
| $174,335 | $69,444 | 3016 |
| $155,147 | **$32,799** | 3125 |

Recorded score, margin and rating all fail to predict transfer — the $155k / 3125-rated tape is
the worst of the set. Tapes must be screened empirically, and the two obvious ranking keys are
traps: ranking by *margin* selects for having beaten a weak opponent.

**Desync, diagnosed.** Weeds spawn at only 0.005 per empty tile per day, so spatial repair is
nearly free but nearly worthless. The real failure is economic: the engine drops **every** PLANT
of a crop whose requests exceed seeds in hand, so one unaffordable `BUY_SEED` voids a whole turn
of planting, the tile stays empty, its harvest never happens, and the next purchase fails too.
A transplanted tape showed 134 seeds unplanted, 8 structures empty, 2 animals placed, $35k against
a recorded $116k. Two repairs tried: topping up the seed order is a no-op when cash is the binding
constraint and −$11k when it fires; raising that cash by selling the shed is **−$8k to −$70k a
seed**, because it dumps into a market the tape has already crashed. Both left in the tool, off.

**Result: parity.** `arena/agents/tape-v1` (episode 108431672, rating 3028) gated over 84 paired
worlds: +$5,476 vs ahmed-v38, +$2,246 vs tschinkel, −$9,588 vs boatlee, **pooled −$622 ± 3,688**.
A transplanted 3028-rated script is worth about what our hand-tuned planner is worth — no more.
Its win rate against opponents champ-h8 loses to 0-for-60 is 0.38, so the shape of its play is
better; the level is not.

**Reading.** Most of a tape's strength lived in its own world, and open-loop replay surrenders it.
That is exactly why the public routers carry 5–13 tapes and switch on public state: the value is
in *selection*, not in any one script. The library (301 tapes, day-6 shop pairs indexed, 62
distinct) and the screening harness are built; the selector is the next step and now has a
measured baseline to beat.

## waste audit and pasture layout — 2026-09-14 · `tools/waste_audit.py`; layout rejected

Audited the hypothesis that we pay for seed that never yields, and that the top agents plan a
spatial layout we do not. Per game against `pub-ahmed-v38`:

| | ours | theirs |
| --- | --- | --- |
| seeds bought / cost | 183 / $6,047 | 242 / $6,677 |
| plantings | 175 | 242 |
| harvests | 290 | 489 |
| tiles dying to WEED | 41.0 | 19.0 |
| DIG turns clearing them | 32 | 36 |
| tiles stranded unharvested at the end | 2.7 | 0.0 |
| mean shed-distance, crops / pastures | 4.39 / **2.31** | 4.60 / **1.76** |

**The headline death number is not waste, and reads backwards.** Splitting deaths by cause: ours
are strawberry 33.7, wheat 5.7, carrot 1.7. The engine sets `max_lifespan_step` on an ongoing crop
*only after its final production* (`production_count == max_yield`), so a strawberry that dies of
lifespan has delivered all four harvests — normal end of life. Our 33.7 means essentially all 33
strawberries completed; their 19 means fourteen of theirs did not. On the crop that matters most
we are **ahead**, and we spend fewer turns clearing weeds than they do (32 vs 36).

**The genuine waste is small and specific**: 5.7 wheat plus 1.7 carrot dying before harvest, and
7.0 drought deaths (a plant dies at `consecutive_unwatered >= 2`) against their 1.0 — about 7
tiles and ~$200 of seed a game. Real, worth fixing, but not a $30k line item.

**The layout observation is correct and acting on it loses badly.** Their pastures genuinely sit
closer to the shed (1.76 vs 2.31), which matters because animal chores — FEED needs wheat carried
from the shed, plus CARE and COLLECT_FERTILIZER — repeat daily for the season while a crop tile is
visited a handful of times. Reserving the shed-adjacent empties for pastures moved ours to 2.05,
so the mechanism works, and it was **−$11,334 ± 2,849** over 72 paired worlds (reserve caps 2, 4
and 6 are identical because the reserve is bounded by the herd still to be bought). Those nearest
tiles are the ones we plant *first*; holding them back delays planting more than the saved travel
is worth. Reverted.

## joint plan search — 2026-09-14 · nothing found, and the value head is refuted as an objective

`tools/plan_search.py`. Two claims tested; both fail, and the first fails cleanly enough to be
worth keeping.

**Claim 1: mid-season win probability is a cheaper objective than final money.** False. Separating
the incumbent from a knob known to be bad (+4 hands) over 66 games a side:

| metric | separation \|t\| |
| --- | --- |
| final margin | **6.53** |
| V day 16 | 1.41 |
| V day 14 | 0.25 |
| V day 12 | 0.08 |

The head predicts a *single game's* outcome well (r = +0.90 with margin at day 16, +0.38 with the
win) but separates *agents* four times worse, because squashing into [0,1] compresses exactly the
tails where between-agent signal lives. Across the 29 searched configurations mean V and final
margin correlate **−0.26** — it is not merely weaker, it is anti-correlated as an objective. The
value head remains what it has been useful as: a diagnostic that localised the season to days
2–17 and the day-10 melon clear. It is not a training signal. The objective reverted to margin.

**Claim 2: the gap is joint, so moving several knobs together finds what single levers missed.**
Also false, at least within reach. 29 joint configurations over 8 knobs (cows, sheep, geese,
strawberry target, wheat engine, standing carrots, premium scale, feed days), 24 games each
against all three external agents:

| | |
| --- | --- |
| incumbent | −$30,555 |
| best found | −$29,974 (`premium_scale` 0.6 + `wheat_engine` 36) |
| improvement | **+$581 = 0.16 se** |
| spread across configs | $13,939 |
| sd across configs | $3,823, against a per-config se of $3,550 |

The landscape is **flat to within noise**: the between-config sd barely exceeds the measurement
error of a single config, so essentially all apparent variation is sampling. Joint moves find no
more than single moves did. What the search does reliably identify is how to *lose* — the worst
three configurations (−$41k to −$44k) all set `feed_days: 1` or pair `geese: 4` with `cows: 10`,
confirming the feed buffer and the goose/executor interaction as real, already-known cliffs.

**Conclusion.** The knob space does not contain the answer, jointly or separately, and the value
head cannot cheapen the search for one. champ-h8 stands.

## days 0–5 opening — 2026-09-14 · rejected on replication; the opening is not the gap

The one window never tested, because it was copied verbatim from champion replays. Traced every
purchase, hire, plant and harvest for both seats over days 0–5 against `pub-ahmed-v38`.

**Two real differences, one of them a genuine bug.** On day 0 we hire **3 hands to their 5** off
the identical $3,000. The cause: `plan.hires_target` is gated by
`workload = sum(crops) + 2 * animals + 10`, which counts the farm we *have* — bare ground on day 0
— while the same turn plants 19 tiles and places 4 animals. So the gate reads 10, `10 // 3 = 3`,
and caps a schedule that asks for 5. Adding the opening's own work to that count restores 5 hires.

**It does not pay.** First gate, 72 paired worlds: +$381 ± 1,602 (+3,101 / +151 / −2,109) —
`hold`. Replicated on 108 fresh paired worlds: **−$2,353 ± 1,470, negative against all three**
(−2,316 / −3,016 / −1,726). Rejected and reverted. Two extra hands on day 0 cost more in wages and
market-order slots than the extra planting returns. Gating `day <= 0`, `<= 1` and `<= 3` produced
byte-identical agents, which confirms the gate only ever bound on day 0.

**The second difference is not a gap at all.** They run a 2-day wheat rotation from day 2
(harvest 3–4, replant 3–4, buy 3–4 seed daily) where we hold to the 4-day cycle. The engine says
ours is right: wheat's water window opens at age 2, so an age-2 harvest yields ~2 units and an
age-4 harvest ~4, i.e. **1.0 units per tile-day either way** — but the fast rotation pays double
the seed and double the planting labour for it. Their 163 plantings to our 104 is a cost, not an
edge.

**Days 0–5 are therefore clean**, and with them the last untested window closes. Every lever this
project can reach has now been measured: opening, crop mix, herd, land, hands, assignment rule,
sell discipline, delivery, harvest timing. The $30–40k gap against a 2500+ agent is not in any
single one of them.

## opportunistic delivery — 2026-09-14 · rejected on replication (and a lesson about gate size)

The last idea left in the day-10 delivery problem: stop paying for delivery. A hand already
standing on a shed tile drops whatever it holds regardless of `DROP_THRESHOLD` — one action, zero
travel — instead of walking off still carrying the day's cash.

**The mechanism works exactly as designed.** Peak units carried on days 10–11 fell 66 → 55 and
58 → 50, and day-11 cash rose $83 → $582 and $98 → $2,346.

**The economics do not.** First gate, 48 paired worlds: **+$2,409 ± 2,370**, positive against all
three opponents (+2,988 / +166 / +4,074), 31/48 better. That is `hold`, not `ship`, so it was
replicated on 144 fresh paired worlds from a different seed block:

| opponent | paired diff |
| --- | --- |
| pub-ahmed-v38 | +$104 ± 1,829 |
| pub-tschinkel-psr | −$2,931 ± 2,054 |
| pub-boatlee-v16 | +$844 ± 1,932 |
| **pooled (144)** | **−$661 ± 1,121** |

No effect. The first result was seed-specific noise, and the gate's two-standard-error rule is
what stopped it shipping — worth recording, because a 48-world +$2,409 looks exactly like the
+$2,376 that made champ-h8 real.

A **one-step detour** version (deliver when within 1 move of the shed) was **−$5,591 ± 3,311**,
which completes the picture: free delivery is worth ~0, one move of travel is worth −$5.6k, and a
carried-value trigger forcing full trips was −$4 to −$5k. Delivery timing is not a lever at any
price, because the produce reaches the market within a day or two either way and the price ladder
recovers overnight.

**The day-10 divergence is now fully explored and closed.** Harvesting is not the gap (we match
their harvest count), priority is not the gap (sprint failed), labour is not the gap (sprint and
hires both failed), and delivery is not the gap (all three variants failed). What remains is that
they simply have more cash *by* day 10 — their day-5 bank is $689 to our $382, before any melon —
so the compounding starts earlier. That points back at days 0–5, the one window nothing in this
project has ever touched, because the opening is the part we copied verbatim from champion replays.

## crew sprint and value-triggered shed trips — 2026-09-14 · both rejected, cause re-identified

Built the day 10–11 crew sprint: `Task` gained a `sprint` flag, `build_tasks` tags one-shot tiles
that are harvestable and die within 3 days once the block reaches 6 tiles, and `assign` gained a
pass 0 where every free hand takes the nearest tagged task anywhere on the farm, zones resuming
when the block is clear. **It made things worse and did not even clear the block faster** (day-11
melons 7 and 5 remaining against the baseline's 6 and 6; finals $41k/$38k against $94k/$52k). The
whole crew converging on melons starves the strawberry fertilize window, which is ages 9–15 —
exactly days 10–15 — and that doubles strawberry production.

**The instrumented day then overturned the diagnosis.** We already harvest the same number of
melon tiles they do on day 10 (12 harvest actions against their 15). What differs is where the
goods sit: our workers peak at **66 units carried on day 10 and 54 on day 11**, theirs at 63 then
**23**. They deliver and sell; we hold. The market controller sizes sells on the *shed*, so
produce in a worker's pocket is invisible to it and the cash lands days late.

Cause: the shed trip triggers on a unit COUNT (`DROP_THRESHOLD = 10`). One melon harvest is 6
units worth ~$1,400 — under the threshold — so the hand kept working with the day's cash in its
pocket. Replacing the trigger with carried market value was gated at three levels and **all three
lose**, monotonically:

| carried value forcing a shed trip | pooled paired diff |
| --- | --- |
| $400 | −$4,007 ± 1,954 |
| $600 | −$4,899 ± 1,847 |
| $1,000 | −$4,254 ± 2,429 |

Extra shed trips cost more in travel than the earlier cash is worth, which is consistent with the
routing round: on a 10×10 board with the shed at the centre, every logistics trip is ~4 moves.
Both changes reverted; the scheduler is unchanged.

**What stands after this round.** The day-10 divergence is real and is about *delivery*, not
harvesting or priority — but neither a labour sprint nor an earlier drop rule captures it,
because both pay in travel what they gain in timing. The remaining lever is to make harvested
produce sellable without a dedicated trip: sell from the tile the hand is standing on when it is
shed-adjacent, or route the return leg of an existing task through the shed. That is a change to
how logistics is *combined* with work, not a threshold.

## day 2–17 localised: the day-10 melon clear — 2026-09-14 · diagnosis kept, fix rejected

Used the value head to ask *what*, not just *when*. `value_head.py diagnose` replaces one part of
our feature vector with the opponent's **inside a single day's model** and reports the
win-probability uplift, so model disagreement cancels. Ten games vs `pub-ahmed-v38`, mean uplift
if our side matched theirs over days 2–17:

| if we matched theirs | mean | concentrated at |
| --- | --- | --- |
| money | **+0.099** | d11 +0.32, d16 +0.23, d17 +0.37 |
| herd value | +0.030 | d3 +0.11, d4 +0.13 |
| herd counts | +0.027 | d11 +0.10 |
| wheat tiles | +0.025 | d11 +0.17, d12 +0.28 |
| geese | +0.023 | steady +0.05…+0.08 from d11 |
| crops (all) | +0.006 | — |
| hands, quadrants, carrots, tomatoes | **0.000** | — |
| melons | −0.020 | ours is better |

**Farm composition is worth nothing.** Matching their tiles, herd, hands and quadrants exactly
moves win probability by under 0.01, which independently explains why the demand book failed: we
were rearranging the one thing that was never the problem. The deficit is *cash*, and the
structural gaps that convert to cash are wheat throughput and geese.

**The single largest divergence of the season is one event on day 10.** Day-by-day land use is
near-identical, but at day 11 they hold **$17,253 to our $62**. Both farms carry 12 melons at
yield 5 on day 10 (engine: watering tops a one-shot crop up from age 6, melons die at day 13).
Our harvest rule only fires at yield 6, so each tile needs WATER *then* HARVEST the same day — 24
serial actions — and we clear 6 tiles, dribbling the rest into days 11–13. They clear all 12
during day 10. Dribbling also sells into a market we are ourselves crashing: **$159/unit against
their $240**. That day-11 cash is what funds their wheat engine, and our wheat never recovers
(13.8 tiles/day vs 17.4; 104 plantings vs 163; 290 harvests vs 489; 194 empty tile-days vs 141).

**Fix attempted and rejected.** Harvesting one unit short (`yields >= max_yield - 1`) does clear
the melons a day earlier, but gated paired over 48 worlds it is **−$973 ± 1,340** (−$4,818 against
ahmed-v38). Ranking the ripening water at `P_HARVEST` so the water/harvest pair completes the same
day did not clear day 10 either — the constraint is labour capacity on that one day, not priority.
Both reverted. The correct fix concentrates the whole crew on the melon block for days 10–11,
which the current zone executor cannot express.

## win-probability value head — 2026-09-14 · new instrument (`tools/value_head.py`)

V(s) = P(this seat finishes with more money | s), fitted supervised on the recorded corpus, so a
game can be scored decision by decision instead of by final bank. Unlike a chess engine we need
no self-play to calibrate: 3,000 replayed sessions give 180,000 labelled states (both seats,
every day boundary), and the label is who actually won.

- **Model**: one L2 logistic regression per day over 59 features — both farms' public state (tiles
  by crop, herd, hands, quadrants, standing crop value), our own shed, market prices as a fraction
  of base, and consumers per product. Held out on `episode % 5 == 0`.
- **Accuracy 0.682 against a 0.633 margin-sign baseline; log-loss 0.5607 against 0.6632.** It is
  well calibrated across the whole range: the 0.0–0.1 bucket wins 4.8% of the time, 0.4–0.5 wins
  45.3%, 0.9–1.0 wins 96.6%. Accuracy by phase: 0.541 on day 1, 0.647 on day 10, 0.726 on day 20,
  0.913 on day 29 — the season really is undecided early.
- **The edge over the margin baseline peaks at days 10–13** (+0.147, +0.084, +0.110, +0.122),
  exactly where the state predicts the outcome long before the bank does. On day 12 the strongest
  signals are melon and strawberry tile counts, not money; only by day 20 does margin lead.

**First finding, and it reframes where to spend effort.** Against `pub-ahmed-v38` our win
probability decays from 0.50 to 0.053 by day 17 and 0.010 by day 19, and then does not move at
all: days 20–29 contribute ±0.006 between them. The steepest two-day falls are day 15→17
(−0.246), day 1→3 (−0.160), day 16→18 (−0.160) and day 7→9 (−0.154). So **the game against a
2500+ agent is over by day 19**, and every endgame knob we have tuned — `liquidation_day` 27,
`carrot_day` 25, endgame carrots, final-push selling — operates in a region that is already
decided. The decisive window is days 2–17. Sanity check: against champ-h1, which we beat 8–8, the
same trajectory rises to 0.999 instead.

Caveat recorded in the tool's output: each day has its own model, so a single day's ΔV mixes real
state change with disagreement between two fits (visible as the bounce at days 6 and 12, where the
shop draw enters). The level trajectory is the robust read; per-day deltas are indicative only.

    PYTHONPATH=. uv run python tools/value_head.py build --n 3000 --procs 6   # ~50 min
    PYTHONPATH=. uv run python tools/value_head.py fit                        # writes rl/value_weights.json
    PYTHONPATH=. uv run python tools/value_head.py attribute --a champ-h8 --b pub-ahmed-v38 --n 8

## demand book and capacity — 2026-09-14 · not shipped (the agent is at a local optimum)

Refreshed the corpus with the official episodes for 12–13 September (+1,309 sessions, 20,793
total) and mined what the current top meta actually invests in. It sizes **every** product to the
shops consuming it, read at day 6, where we conditioned only cows and sheep (medians, 1,857 seats
rated 2900+ in yesterday's games, by consumer count 0 / 1 / 2):

| investment | 0 | 1 | 2 | ours |
| --- | --- | --- | --- | --- |
| strawberries by day 13 | 21 | 29 | 35 | 33 fixed |
| cows by day 18 | 6 | 9 | 12 | 6 or 8 |
| sheep by day 20 | 5 | 12 | 18 | 8, or 10 on a rule |
| geese by day 18 | 2 | 3 | 5 | 0 |
| carrots, season total | 22 | 51 | 73 | ~43 |
| tomatoes, season total | 4 | 9 | 14 | 0 |
| wheat, season plantings | 139 | 148 | 156 | ~114 |

Holding, from replaying those games with the market commit instrumented: they carry ~20
strawberries and ~23 melons to midnight and release 40–65% of each product's revenue in the first
six hours, after the overnight town drain lifts the price. Strawberry cleared at 1.19× base,
wheat 1.44×, carrot 1.44×, tomato 1.29×; milk 0.63× and wool 0.75×. Run on champ-h8, **our
realized prices match or beat theirs** (wool 0.91×, milk 0.79×, strawberry 0.96×) — the whole gap
is volume: wheat $7.0k against their $27.6k, carrots $2.3k against $13.0k, tomato and egg zero
against $6.7k and $9.4k.

`planner._apply_demand_book` implements the table (flags `BOOK_HERD` / `BOOK_CROPS` /
`BOOK_WHEAT`, all default off). **Every capacity and allocation change fails**, gated paired
against three external agents with `tools/exec_gate.py` (48 paired worlds each):

| change | pooled paired diff | verdict |
| --- | --- | --- |
| demand book, crops only | +$231 ± 3,186 | flat |
| 4th quadrant (`CHAMP_LAND_BUYS` 2 → 3) | +$325 ± 2,318 | flat |
| demand book, crops + 4th quadrant | −$5,955 ± 2,901 | reject |
| demand book, herd only | −$6,637 ± 2,808 | reject |
| demand book, all groups | −$6,909 ± 3,160 | reject |
| +2 hands from day 6 | −$5,229 ± 2,085 | reject |
| +4 hands from day 6 | −$19,344 ± 1,967 | reject |

The crops book does exactly what it says — carrots 37 → 126 units, tomatoes 0 → 42 — but **wheat
falls 181 → 100**: it reallocates a fixed capacity rather than adding volume. Land does not
relieve it (25 more tiles, same hands, no gain) and neither does labour (hiring is strongly
negative and monotone in the wrong direction, so we are at or past the optimum hand count).

**Reading.** Production is bound jointly by labour throughput and tiles, and the top meta's
targets only work on top of their throughput: they convert ~6,800 unit-turns into 3,391 productive
actions where we get 2,818. Copying a stronger player's build order without their execution
capacity spreads the same labour thinner. Combined with the routing round below — five assignment
changes, all neutral or negative — the agent is at a local optimum in **every single-lever
direction we can reach**: hands, land, crop mix, herd size, assignment rule, sell discipline.
Closing the remaining $30–40k needs a jointly different plan, not another knob.

## worker routing — 2026-09-14 · not shipped (five negatives, and a methodology correction)

Chased the measured labour gap (our hands 59.4% MOVE / 34.0% productive vs a 2500+ opponent's
42.2% / 46.0%; 1.25 moves per productive action vs 0.76). **Nothing improved against a real
opponent.** Movement was attributed per assignment pass first (`scheduler.MOVE_STATS`, 3 games):

| pass | MOVE | action | move% | share of all movement |
| --- | --- | --- | --- | --- |
| zone (own zone, real work) | 7,260 | 6,388 | 53.2% | 61.2% |
| idlehelp (cross-zone, r≤5) | 1,999 | 486 | 80.4% | 16.9% |
| filler | 1,446 | 372 | 79.5% | 12.2% |
| shuttle (shed trips) | 760 | 206 | 78.7% | 6.4% |
| morning (zone provisioning) | 224 | 1,025 | 17.9% | 1.9% |

The help and filler passes really do run at four moves per action — 29% of all movement for 10%
of the actions — but every attempt to reclaim it lost money:

| change | vs champ-h8 (twin) | paired vs pub-ahmed-v38, same worlds |
| --- | --- | --- |
| filler walk capped at 3 tiles (1 after hour 16) | 19–11 | **−$7,387 ± 3,126** |
| CARE crash-pivot removed (always care) | 11–19 | −$544 ± 1,734 |
| CARE threshold 0.35 → 0.15 of base | 14–16 | +$1,263 ± 1,631 |
| idle hands pick by value per turn, `value/(d+1)` | 5–25 | **−$4,928 ± 2,702** |
| zones split by work (occupied tiles weighted ×2/×3/×5) | 11–19 / 1–29 / 1–29 | not run |
| `MIN_ZONE_TILES` 5 → 8 (9 zones, hands share) | **24–6, p=0.0014, +$4,481** | **−$4,490 ± 3,333** |

Readings:
- **The executor is at a local optimum.** Equal-area contiguous serpentine zones beat every
  relaxation. Work-weighted zones fail because a zone with sparse occupancy becomes
  geographically *larger*, so the hand walks further — the opposite of the intent. Value-per-turn
  ranking fails because the far P_TEND/P_HARVEST action really is worth five turns of walking and
  the nearby filler action really is worth almost nothing. Movement is the price of covering a
  55–75 tile farm with 11 hands, not slack.
- **The CARE gap is a symptom, not a cause.** We CARE 210 times to their 417 only because the
  crash-pivot fires once wool and milk collapse — and at $1–50 a unit the extra product genuinely
  does not repay the turns. Removing the pivot is worth nothing.
- **Methodology: never gate an executor change on the twin arena.** `MIN_ZONE_TILES = 8` beat a
  copy of ourselves 24–6 at p=0.0014 and was $4,490 *worse* against a real agent on the same
  worlds. Two agents sharing one price ladder can trade wins that come entirely from sale timing
  against each other. Executor changes must be gated paired against `arena/agents/pub-*`.

## champ-h8 — 2026-09-14 · SUBMITTED · stop churning the feed buffer

One-line planner fix worth **+$2,376 a game**: the sell path reserved `animals + 2` wheat while
the buy path topped the buffer up to `feed_days * animals`, so every day we sold the gap and
bought it straight back. Over a season that was **459 units bought for $19,037** against 488
sold for $21,290 — a round trip that also walked the wheat price against us twice, since
`BUY_PRODUCT` quotes at inventory−1 and both farms drain the same pool. The reserve is now
`max(animals + 2, feed_days * animals)`; purchases fall to 165 units.

- **Gates**: 37–13 vs champ-h7 on fresh seeds 21000+ (sign test **p = 0.0009**, $82,315 vs
  $79,940); 22–8 on the 19100 seeds; against a 2500+ public agent our mean rises $73,710 →
  $79,879. Paired champion worlds +$1,037 ± 3,325 (not resolvable at n=24); held-out ladder
  0.67 / 0.58 / 0.25 / 0.17 / 0.17 / **0.17** — the 2500+ bin is the first non-zero we have had,
  the middle bins are down within noise.

### How it was found: the public notebooks are replay tapes, and they beat us 0–30

Reading the competition's public code (`tools/fetch_public_agents.py` installs them as arena
opponents) changed the picture. Most highly-voted notebooks are **720-step replay tapes of
2500+ ladder players** plus a thin online wrapper — boatlee's V16-RC5 is a majority vote over
three recorded episodes with a one-turn "sell early" front-run, salemali7's "2900+" is a
byte-identical fork of it, and the yhay81 / tschinkel "routers" pick between 5–13 tapes at
144-turn boundaries using a fitted decision tree on public state. champ-h7 loses **0–30 to every
one of them**, by $19k (boatlee) to $44k (ahmed-v38).

An exact ledger (`_commit_unit` instrumented, so fills not requests) against ahmed-v38 shows the
farms are nearly identical — same 33 strawberries, same ~57 tiles, same 3 quadrants, same ~11
hands — and the whole $38k is revenue conversion and cost:

| gap | us | them |
| --- | --- | --- |
| wheat bought | 459 units / $19,037 | 172 units / $6,028 |
| milk | 131 units / $10,707 | 216 units / $19,002 |
| fertilizer | 201 units / $10,590 | 351 units / $17,504 |
| melon | $159/unit | $240/unit |
| eggs | none | 76 units / $5,189 |
| productive share of worker turns | 34.0% | 46.0% |

### The market is a per-unit lockstep, and the price ladders invalidate part of the playbook

`_process_market` walks both players' orders **one unit at a time**, quoting both the same
pre-commit inventory price, so there is no first-seller premium *within* a unit — but every unit
sold raises inventory and walks the price down. Selling capacity per product (from I0, before
town drain) is wildly uneven:

| product | base | 40 units | 100 units | 300 units |
| --- | --- | --- | --- | --- |
| melon | 250 | $9.8k @ $235 | $21.7k @ $152 | $26.6k @ $1 |
| wool | 200 | $6.8k @ $112 | $8.0k @ $1 | $8.2k @ $1 |
| milk | 160 | $4.8k @ $78 | $6.2k @ $1 | $6.4k @ $1 |
| strawberry | 120 | $3.3k @ $45 | $3.8k @ $1 | $4.0k @ $1 |
| egg | 50 | $1.8k @ $44 | $4.4k @ $42 | $12.6k @ $40 |
| fertilizer | 100 | $3.8k @ $92 | $9.0k @ $80 | $21.0k @ $40 |

Wool and milk — the products our herd targets are built around — are the two that collapse
fastest, and in the traced game both farms had wool at **$1/unit from day 15**. Egg and
fertilizer never collapse, which is why animals are worth more as fertilizer factories than for
their headline product. Wool is consumed *only* by YARN_STORE, so with no yarn store its entire
season capacity is ~$8k no matter how many sheep are bought.

### Negative results from the same round

- **Sell discipline is not the lever.** Sweeping `premium_scale` over 0.35 / 0.6 / 1.0 / 1.4
  against ahmed-v38 (24 games each) gives $73.7k / $72.7k / $72.5k / $67.8k — flat to worse. The
  shipped 0.35 is already right.
- **`wheat_floor` 30 is a no-op**: both farms buy wheat constantly, so its price sits at $40–46
  all game and a $30 floor never binds (results identical to 6 significant figures).
- **Geese still lose**, −$3,845 vs baseline against ahmed-v38, despite eggs having the best
  ladder in the game and a goose laying every day from day 4 for $300. This is an executor
  problem, not an economics one, and it is the clearest open lead.
- **Capping filler-task walking distance** (3 tiles, 1 after hour 16) was 19–11 vs the churn fix
  (p = 0.20) but −$7.4k against ahmed-v38 — reverted. The diagnosis behind it still stands: our
  hands spend **59.4% of turns moving vs their 42.2%**, need 1.25 moves per productive action vs
  their 0.76, and in hours 17–23 we move 65% of the time while they move 40% and simply PASS.
  Fixing that is worth ~700 actions a game and needs a better idea than a distance cap.

## champ-h7 — 2026-09-14 · SUBMITTED · first price-reactive rule (sheep on wool scarcity)

The adaptive layer's first rule that reads the **market** rather than the town: at day 12, if
wool is at or above base price (i.e. the opponent has not flooded it), raise the sheep cap from
8 to 10 for the season.

- **How it was found**: the day-9 farm is not a usable proxy for opponent strength (everyone in
  the top-heavy corpus plays the same opening: 20 vs 19 strawberry tiles for 2500+ vs <1500
  players), so opponent pressure is read where it shows — prices. The bandit gained price
  conditions with a decision day (`rl/bandit.py --arms 'sheep:10@P_WOOL>=1.0@d12'`) and
  two-phase paired batches: the twin runs first and its recorded day-6/9/12/15 contexts decide
  which conditional arms each world is eligible for (the deviated game is identical up to the
  decision day, so no games are wasted).
- **Measurements**: Kaggle run (4,800 samples, twin champ-h5, opponents ≥ 2000) **+$1,034 ±
  151** (n=714); local run +$753 ± 224 (n=321); the day-15 variant +$532 ± 140 (n=595, final). Other
  price arms: wheat engine 40 when wheat ≥ 1.1× base +$125 ± 64 (real, negligible); cows 10 when
  milk ≥ base −$567 ± 242; endgame carrots from day 22 when carrots ≥ base −$986; *holding*
  premium stock when strawberries are scarce (×0.7) −$1,438 — the first-seller rule holds even
  in scarcity. Strawberry/milk-crash reallocations (carrots, wheat, dump) never fired at day 12
  and rarely at day 15: the overnight drain keeps hour-0 prices above 0.7× base, so a crash
  signature needs our own realized sale prices instead (open).
- **Gates vs champ-h6**: arena 13–13–4 (both farms flood wool against a copy of ourselves, so the
  rule rarely fires there); 24 paired champion worlds **+$721 ± 416**, fired in 5 with **+$3,460**
  each; held-out ladder ≥ champ-h6 in every bin (0.75 / 0.67 / 0.50 / 0.17 / 0.17 / 0.00).
- Mid-season herd *caps* are no-ops once the ramp has completed (cows 8 by day 8–10, sheep 8 by
  day 12); only increases and crop/market knobs can act at day 12+.

## top-games knob bandit — 2026-09-12 · not shipped (negative)

Second free-compute job (10.7 h, 5,400 paired samples): the shipped and borderline knobs
re-measured with `--min-rating 2500` — opponents drawn only from 2500+-rated recorded scripts on
the new corpus, twin champ-h5. Nothing helps against the top of the field: premium ×0.2
+$75 ± 15, early strawberries 24 +$67 ± 379, second-quadrant margin $1,500 +$66 ± 233; premium
×0.5 −$36; strawberries 36 −$178, 40 −$740; wheat engine 40 −$739; carrot day 24/26 −$854/−$818
(25 stays optimal); opening melons 16 −$872; sheep 6 −$1,645; cows 10 −$1,873 (−$5,150 with no
milk consumer); geese 2 −$3,484. Against strong opponents the remaining gains are not in plan
constants; the next candidates are opponent-conditioned rules (mining: champions plant fewer
strawberries and far more carrots against 2500+ opponents), which need the opponent's visible
day-6/9 farm as the in-game condition.

## champ-h6 — 2026-09-12 · SUBMITTED · first conditional rule, from champion mining

The adaptive layer's first *conditional* knob, found by imitation mining and verified by the
counterfactual bandit.

- **Mining** (`tools/mine_knobs.py`, 16,738 seats rated 2500+ from the new corpus): champions
  set their herd and crops by the two shops open on day 6 — sheep 5 / 11 / 12 and cows 7 / 9 / 10
  for 0 / 1 / 2 consumers of wool / milk, strawberries 33 / 34 / 38, carrots 10 / 20 / 31 — with
  within-context margin z-scores of 5–9; geese 0 always; more wheat is worse in every context;
  against 2500+ opponents they plant fewer strawberries and far more carrots.
- **Verification** — conditional bandit arms applied only in worlds whose day-6 shops match
  (twin champ-h5, opponents ≥ 2000, 2,212 samples): **cows 6 when no milk consumer is open:
  +$1,989 ± 551** (n=365). The rest of the champion settings do **not** transfer to our
  executor: sheep 11 with a yarn store −$2,567, sheep 5 without −$1,703, strawberries 38 with
  two consumers −$1,529, cows 10 with two milk consumers −$1,358, early strawberries 12 −$3,156,
  wheat engine 24 −$493, carrot cap 30 ≈ 0. Imitation tells us *where* to look; only the paired
  measurement says what works on our farm.
- **Gates vs champ-h5**: arena 18–10–2; 24 paired champion worlds +$3,309 ± 1,867 — the rule
  fired in 10 of them and gained +$7,942 on average there; held-out ladder ≥ champ-h5 in every
  bin. Note the ladder now runs on the new 19k-session held-out pool (much stronger recorded
  opponents), so its absolute numbers reset: champ-h5 reads 0.67 / 0.67 / 0.33 / 0.17 / 0.00 /
  0.00 there, champ-h6 0.67 / 0.67 / 0.42 / 0.17 / 0.08 / 0.00.

## knob bandit round 3 — 2026-09-12 · not shipped (plan constants exhausted)

First job on free Kaggle compute (7.5 h, 5,400 paired samples, twin = champ-h5): the borderline
round-2 arms re-measured — second-quadrant cash margin $1,500 +$375 ± 171, $800 +$150 ± 145;
early strawberries 24 +$259 ± 294, 21 −$50 ± 257; opening melons 16 −$3,132 ± 1,366. Nothing
clears the bar. Three rounds: +$3.5k, +$1.4k, ~$0 — the fixed plan's constants are tuned out.
Next lever is *conditional* (day-6 shop draw) knobs from `tools/mine_knobs.py`, under
verification.

## data + compute — 2026-09-12 · official daily episodes on free Kaggle compute

- **Free-tier compute** (`tools/kaggle_job/`): any of our jobs runs as a private Kaggle script
  kernel (4 vCPU, 12 h). The private GitHub repo ships as a private Kaggle Dataset tarball
  (`push.sh`, versioned per push; `code-only` mode leaves a running kernel alone); the corpus is
  the `kaggriculture-datasets` Dataset; job steps are injected into the script at push time.
  Verified: tests pass on Kaggle, bandit batches run at local per-core speed.
- **Official daily episodes** (`kaggle/kaggriculture-episodes-index` → 44 daily datasets, ~21 GB
  each, 936 GB total). `tools/episodes_pipeline.py` mounts 3 days at a time into its own kernel
  (mounts cost no quota), `tools/extract_episodes.py` reduces each 32 MB replay to a ~20 KB
  session (seed, aligned action scripts, finals, shops, teams, ratings) — 657 replays in 185 s;
  fidelity EXACT — and only the compact output is merged, with a ledger for resumability.
  Raw replays carry no ratings and the community crawl has zero overlap with them, so ratings
  join from the leaderboard by team name.
- **Result**: 28 days (2026-08-15 → 09-11, engine 1.32.7) → **18,801 sessions**, corpus now
  **19,484**. The official datasets are top-heavy — seats by rating bin: <600: 482 · 600–1000:
  205 · 1000–1500: 3,915 · 1500–2000: 5,752 · 2000–2500: 8,940 · **2500+: 16,738** (2,936
  unrated) — i.e. the "train on top games" corpus; held-out (episode % 5 == 0) keeps ~7,900 seats
  across all bins for the ladder. `rl/bandit.py --min-rating` selects training worlds by the
  recorded opponent's rating.

## champ-h5 — 2026-09-11 · SUBMITTED · premium sell floors ×0.35 (knob bandit round 2)

Round 2 of the knob bandit with **champ-h4 as the twin** (its knobs became the plan's defaults;
verified identical play) over a widened set of 21 arms / 11 knobs, 4,680 paired samples including
two focused follow-ups on the premium floor scale.

- **Premium floor scale is monotone below 1**: ×0.5 +$844 ± 121 (n=541), **×0.35 +$1,391 ± 203**
  (n=215), ×0.2 +$617 ± 164, ×0.1 +$1,330 ± 153 — a plateau around +$1.2k; ×0.85 −$792. Across
  rounds: ×1.0 → ×0.7 → ×0.35 is worth ~+$3.3k. In this market the first seller takes the price,
  so premium products (strawberry / milk / wool) should be sold almost as soon as they exist.
- Everything else measured is neutral or worse against champ-h4: carrot day 24 (−$846) and 26
  (−$1,258) bracket the shipped 25; feed stock 1 day −$7.9k, 3 days −$0.9k; liquidation day 26
  +$13, 28 −$942; wheat floor 20/30 ≈ 0; wheat engine 28 −$591; opening melons 8/16 ±$900 (n.s.);
  $1.5k margin before the second quadrant +$466 ± 307 (n.s.); early strawberries 24 +$457 ± 536,
  12 −$4.2k; strawberry target 36 −$88 (the interim +$2.3k "with consumers" signal faded to
  +$508 ± 452). Diminishing returns are visible: round 1 ≈ +$3.5k of knobs, round 2 ≈ +$1.4k.
- **Gates vs champ-h4**: arena **20–10** ($101.7k vs $100.9k); 24 paired champion worlds
  **+$76 ± 347** (no harm against 2500-rated scripts — the gain comes against everyone else, where
  the rating is decided); held-out ladder unchanged 1.00 / 0.83 / 0.17 / 0.17 / 0.00 / 0.50.

## champ-h5 candidate (strawberries by demand) — 2026-09-11 · not shipped

Final knob table (3,600 paired samples): premium floors ×0.7 **+$1,949 ± 157** (n=501), carrot
day 25 **+$1,545 ± 116** (n=498) — champ-h4's knobs at ~12σ. The only surviving context rule,
strawberry target 40 when ≥1 strawberry consumer is open (+$881 ± 391, n=418; −$3,423 ± 668
with none), was tested as champ-h5c on top of champ-h4: arena 14–14–2, paired +$658 ± 1,901 —
consistent with the measured effect but unverifiable at gate sample sizes. No herd knob (cows
6/10, sheep 6/10, geese 2/3) pays in any demand context; standing carrot/tomato patches never pay.
`KnobBanditPolicy` gained explicit `conditions` (min/max consumers on the day-6 context) for
interpretable rules.

## champ-h4 — 2026-09-11 · SUBMITTED · first bandit-measured adaptive knobs

The adaptive layer, finally with evidence. Two learning approaches failed first and are recorded
in `docs/VERSIONS.md`: a residual PPO planner (offset heads over the champion plan, four runs)
converged back onto the prior — whole-episode credit over ~60 deviations per game is too weak
against ±$30k outcomes — and a single-day counterfactual bandit (4,800 paired samples) found no
positive one-day tweak within ±$500, because the plan's standing quotas refill any one-day change
the next day. What worked: **season-long knobs** of the champion plan, one per game, applied from
day 6 (shops revealed), each paired against the unchanged plan on the identical world
(`rl/bandit.py collect --knobs`).

- At 1,200 paired samples, two knobs are unconditionally positive at ~5σ: **premium sell floors
  ×0.7** (sell strawberry/milk/wool sooner — the first seller takes the price in this market),
  +$1,836 ± 380; **endgame carrot switch on day 25 instead of 22**, +$1,572 ± 249. Shipped as
  constant rules through `KnobBanditPolicy` (`arena/agents/champ-h4/rules.json`).
- Context-dependent signal already visible: strawberry target 40 is **+$2,274 ± 927 when at least
  one strawberry consumer is open and −$3,548 with none**; cows 10 only pays with milk consumers.
  Those need the regression fit (collection continues on the 9 surviving arms).
- Clearly harmful, measured: hires −2 (−$11.7k), 16 standing carrots (−$7.7k), strawberries 21
  (−$5.7k), hires +2 (−$4.4k), geese 3, carrot day 18, wheat engine 24, premium floors ×1.3.
- **Results**: arena vs champ-h3 **29–1**, $90.7k vs $86.1k; **24 paired champion worlds +$1,478 ±
  692 (18/24)** — the first candidate positive on both; held-out ladder **1.00 / 0.83 / 0.17 /
  0.17 / 0.00 / 0.50** (champ-h3 …0.08 / 0.17 / 0.00 / 0.42); p99 4.6 ms.

## champ-h3 — 2026-09-11 · SUBMITTED · sweep executor

Measured on our five worst worlds against the recorded occupant of our own seat: they got
105–159 productive actions a day out of the same crew with 94–140 moves and idle time to spare;
champ-h2 got 71–105 with 156–184 moves and no idle. The zone round scored tasks by
(priority, distance), so a hand walked four tiles to a harvest and four back to a water next
door. Now (distance, priority): finish what is on and around the current tile, move on;
survival tasks (dies tonight) jump the queue from hour 16; cross-zone help within 5 steps.

- Productive actions 105–128/day (from ~100), moves 128–169 (from 148–171), 58 tiles standing
  on day 18 (from 48); wheat +123 units, carrots +40, strawberries +21 per game.
- **Arena vs champ-h2: 30–0**, $91.0k vs $78.2k. **Held-out ladder 1.00 / 0.83 / 0.08 / 0.17 /
  0.00 / 0.42** (best yet; $108k earned in the 600–1000 bin, was $94k). p99 turn 0.40 ms.
- Paired champion worlds: neutral (−$1.2k ± 4.5k). The extra output lands in markets that the
  2500-rated scripts already flood: inventory only recovers through shop consumption (6 units a
  day per consumer) and the first seller takes the price. Two market experiments failed for the
  same reason and were reverted: paced selling (hold above a floor until the final week) let the
  opponent sell first and lost $30k on the traced world; capping strawberry tiles by the demand
  anchor was neutral. Converting throughput against strong opponents needs the adaptive
  allocation layer, not more executor.

## champ-h3 experiments — 2026-09-10 · not shipped (negative results)

Milk and labor were the next levers after champ-h2; each candidate was measured on **24
paired champion worlds** (same sessions, same seats) against the shipped champ-h2, with
hooks that reproduce champ-h2 exactly (+$0) as the control. Per-world outcomes swing by
±$30k between near-identical policies (paired s.d. ≈ $26–28k; arena s.d. ≈ $17k), so 8-world
readings earlier in the day ($88k) were noise.

| Candidate | Δ vs champ-h2 (24 paired worlds) | Verdict |
|---|---|---|
| cows before sheep in the buy order | −$6.0k ± 2.5k (better 6/24) | breaks the day-6 wool → land bootstrap |
| workload-weighted zones (pasture tiles ×4) | −$2.2k ± 5.0k | neutral; state-based weights drifted zones mid-day and starved a sheep |
| cash reserve for the first quadrant | −$4.9k to −$8.9k | delays the cow ramp |
| care skipped below 15% of base (was 35%) | +$0.2k ± 1.1k | neutral |
| melon window-watering at harvest priority | melon avg price unchanged (~$170–230) | the shared price is set by the opponent's day-10 dump |
| demand-conditional crops (strawberries scaled to consumers, standing carrots/tomatoes) + 3 geese in egg worlds | +$3.7k ± 4.9k | geese cost −$30k in the worlds that bought them: their daily feed/care/collect/egg load broke the cow/sheep care regime |
| same without geese | +$3.4k ± 5.3k (16/24); **arena vs champ-h2 12–18**, $79.8k vs $81.1k | not shippable |

Kept: the finding that outcome variance, not mean economy, is now the dominant term — the same
playbook earns $46k–$136k on champion worlds depending on the shop draw and on the opponent's
market pressure (in the bad draws the recorded champions still hold ~$90k). Code reverted to
the champ-h2 state.

## champ-h2 — 2026-09-10 · SUBMITTED (id 56152624) · animal chores first, champion sheep ramp

Pure heuristic (no neural net), same playbook as champ-h1 with the three levers the decode
left open — feeding, sheep, wheat — after tracing nightly fed/cared coverage per animal.

- **Feed/care above routine watering** (`P_ANIMAL = 1.5`). Engine: a fed+cared production
  yields 1 + banked care days (a sheep cared for 3 days gives 4 wool, not 1) and an *unfed*
  production night forfeits the whole bank. Coverage was 9–11 of 16 animals on days 16–28;
  now 14–16 of 16.
- **Two-pass zone round**: real work in your zone, then real work anywhere, then filler
  watering. Pastures are built centre-first, so 16 animals sat in 2–3 overloaded zones while
  other hands watered plants that weren't due.
- Sheep ramp at the recorded pace (4 by day 8, 6 by day 10, 8 by day 12); wheat engine 32.
- **Results**: solo on champion worlds $80.4k → $83.9k; wool $17k/116u → **$32k/162u**
  (champion range $35–44k); arena vs champ-h1 **16–4** (p = 0.012), $79.3k vs $76.0k, max
  $115.9k; held-out ladder 1.00 / 0.75 / 0.00 / 0.08 / 0.00 / 0.17 (unchanged within noise).
- Still open vs the $117k profile: milk $21k vs $29k (181 vs 224–265 units), wheat 372 vs
  600 units (labor-bound), 1000–2500 rating bins still ~0.

## champ-h1 — 2026-09-09 · SUBMITTED (id 56128207, rated 690 after 24h) · decoded champion playbook (heuristic)

The session-replay track's rating ladder showed rl-v9 and every trained successor
(g10–g13) frozen at 0.83 / 0.08 / 0 / 0 / 0 / 0 win rate against recorded opponents
binned by Kaggle rating — a ~600-rated economy that no training signal moved. So this
version comes from **decoding what 1500+/2500+ players do** (`tools/decode_sessions.py`
replays 683 harvested sessions on their true seeds and profiles both sides per day) and
fixing the executor until it could express that playbook.

- **Playbook** (`planner.make_champion_plan`): day 0 = 2 cows + 2 sheep + 12 melon + 7 wheat;
  days 1–5 funded by animal fertilizer (~$100/unit, produced fed or not); first quadrant
  bought the turn the day-6 wool lands; strawberries 18 → 33 by day 13 (fertilized: 2 units
  per production); 8 cows / 8 sheep by day 10–14 bought before demand is revealed; a 28-tile
  rolling wheat engine; carrots on freed tiles from day 22.
- **Executor fixes found by hour-by-hour tracing**: animal carriers place first (a cow was
  hauled around for 3 days); fertilizer treated as cash (no wheat/carrot fertilizing at base
  price, early shed trips, reserve for strawberry tasks); feed bought at any sane price with
  two days of stock (the herd starved at $40k cash behind a "pricey" guard); watering only
  where the engine pays (growth window / fertilized production night / survival) — daily
  watering had starved planting of labor; zone-local wheat/fertilizer provisioning; planting
  tasks on every empty tile; even zone split (12 hands had 11 zones); survival override only
  pulls idle units (it was re-targeting walking hands every turn: 200 moves per 60 actions).
- **Results**: solo on champion worlds $80.4k (rl-v9: $57k there); arena vs rl-v9 **20–0**,
  **$100,345 mean / $119,824 max** vs $59,182; held-out rating ladder
  **1.00 / 0.75 / 0.08 / 0.00 / 0.00 / 0.33**; p99 turn 0.32 ms.
- Next: this heuristic becomes the BC teacher; remaining gaps vs $117k champions are sheep
  timing (wool $17k vs $35k+), wheat volume (382 vs 600 units), and late-game feeding.

## rl-v5 — 2026-08-28 · SUBMITTED (id 55853641) · profit champion

Generation-5 **speculative-invest planner**, built from the decoded leaderboard meta.

- BC teacher mix: 50% a new *speculative* heuristic recreating the observed top-performer
  strategy (episode 101761797, finals $105k/$96k: day-0 ruminants on expected demand,
  30+ strawberry tiles, 14-16 cows, no geese, investment-first through day ~10) + 50% the
  proven reactive teacher; PPO then learns when each playbook applies via the demand
  forecast features.
- Head ceilings raised to the meta (wheat 40 tiles, strawberry 32, cows 16, sheep 14,
  carrots 26); softened carrying/unsold penalties; hot-then-scheduled LR (2.5e-4).
- **Results**: solo **$85,238 mean / $98,378 max** (rl-v3: $68,671); contested best game
  $97,841; 90-100% wins vs every league gate except rl-v3 head-to-head (3-17, whose
  counter-style feeds on rl-v5's economy). Both champions stay on Kaggle — submissions
  rate independently. 22 evals, zero training collapses.

## gen-4 experiments — 2026-08-28 · not shipped (tag g4-experiments)

Demand forecasting (`econ.forecast_drain`), carrying/unsold penalties, labor-utilization
features, architecture v2 (192/128 GELU+LayerNorm policy + decoupled value net), and the
LR scheduler all land here — but every gen-4 candidate lost its fresh-seed gate to rl-v3
(best 2-18). The stability machinery was validated (10 straight improving evals, zero
collapses) and became the foundation gen-5 shipped on. Also recorded: changing head VALUE
semantics silently remaps older policies' choices — champions ship only from their frozen
`arena/agents/<name>/main.py` bundles.

## rl-v3 — 2026-08-28 · SUBMITTED (id 55829852)

Generation-3 **league-trained, profit-first neural day-planner**.

- Trained against a 4-gate league {v3 heuristic, rl-v1, rl2-it59 exploiter, rl-v2};
  new best checkpoints join the opponent pool during the run (league growth).
- Profit-first objective: terminal reward dominated by final money (win bonus cut to a
  tiebreaker); checkpoint selection by **mean profit across all gates** on rotating eval seeds.
- Convergence stopping: run self-terminates after 5 evals without profit improvement
  (stopped at iter 44; best at iter 19).
- **Results**: 50–0 vs rl-v1 ($59.3k ± 2.7k vs $40.1k), 20–0 vs v3 ($63.6k), 16–4 vs rl-v2,
  tournament 24/24 at $62.7k mean, healthy mirror ($63.4k/$58.9k), step p99 1.2 ms.

## rl-v2 — 2026-08-28 · league member (not submitted)

Generation-2 profit-first planner (g2d best) + executor upgrades.

- 133 features (was 86): per-product **supply-vs-town-demand gap** (mine + opponent's projected
  7-day output vs shop drain), price curvature, land economics. 13 action heads (was 11):
  hires widened to 16, learned premium/staple **sell-floor scale** heads.
- Neural-path land safety nets (crunch fallback + demand-pressure trigger) fixing rl-v1's
  refusal to buy land; idle-worker pre-positioning; locked-tile walk-back.
- Training stabilizers after gen-2 collapses: value-head warmup, LR 1.5e-4, KL early-stop.
- **Results**: beats v3 17–3 ($43.2k), money-ties rl-v1 ($41.0k vs $41.5k; 10–20 h2h),
  high variance (solo max $78.1k, std $14.7k). Kept as league opponent.
- Side artifact **rl2-it59** ("the exploiter", g2b iter 59): beats rl-v1 35–15 but loses to v3
  3–17 — proof that training vs one static incumbent breeds a strategy cycle; league member.

## rl-v1 — 2026-08-27 · submitted (id 55825229, public ~486)

First **learned** agent: neural day-planner over the scripted executors.

- Tiny MLP (86 features → 96×96 → 11 masked heads + value), pure-Python inference (~30
  forward passes/episode, p99 0.8 ms). Heads are ABSOLUTE standing targets (crop tiles,
  herd sizes, hires, land, melon floor); scripted scheduler/market executes the micro.
- BC-initialized from v3, PPO vs {v3, starter, snapshots}; best checkpoint (iter 19) exported.
- Critical fix: bundle no longer touches `sys.modules` (two agents in one process — including
  Kaggle's self-play validation — used to overwrite each other).
- **Results**: 50–0 vs v3 across blocks ($42.7k/$41.1k vs ~$24k), tournament 32/32,
  solo vs starter $54.5k ± 2.5k.
- Known weaknesses (fixed in rl-v2+): never buys land under demand growth; beatable by a
  targeted exploiter.

## v3 — 2026-08-27 · submitted (id 55817540, public ~497)

Scripted heuristic, final form.

- CARE regime for ruminants: cows/sheep fed daily + cared (banks +1 product/day ≈ $125/action);
  geese stay on cheap alternate-day feeding.
- **Results**: beats v2 17–3; solo vs starter $44.9k mean; then-tournament sweep.
- Negative results recorded: 16 hands/day, 40 geese, daily goose care, aggressive feed-wheat
  buying, skipping cheap fertilizer collects — all lost head-to-head gates.

## v2 — 2026-08-27

Demand-adaptive livestock + market-controller hardening.

- Cows/sheep bought when milk/wool shops unlock (town drain >> T holds prices at $250+);
  strawberries planted at 2+ consuming shops; milk/wool adaptive sell floors.
- Fixed silent 10-order-cap truncation (goose pipeline freeze), goose delivery batching,
  feed buffer buys, fertilizer floor → $1 (a fert hoard blocked the shed; midnight overflow
  destroyed produce), shed pressure valve.
- Isolated bundler (no `sys.modules`).
- **Results**: beats v1 20–0 ($26.2k vs $8.7k contested); solo $40.4k.

## v1 — 2026-08-27

First full scripted economy.

- Melon rush (days 0–2) → goose/egg factory conversion; wheat feed engine; carrot/tomato
  filler; fib-priced hand hiring (~14/day); marginal-revenue sell sizing from the exact
  engine-mirrored price function; day-29 liquidation.
- **Results**: solo vs starter $36.2k mean (starter: $3.5k).

## M0 / demo — 2026-08-27

- Engine-mirrored `constants.py`/`econ.py` with parity tests against the installed engine.
- Arena harness (seeded batch runner, paired sign tests, tournament), submission bundler,
  demo melon agent preserved as baseline ($5,957 solo).
