# Adaptive executor: parallel branches

Three ideas, one branch each, run in parallel. Each is allowed to fail; the
point is breadth, because the top of the leaderboard is probably not doing what
we are doing.

Why not tapes. A tape is a recorded 719-step script from someone else's episode.
It decays: it is decodable by anyone with the same public replays, it is tuned
to the opponent population that existed when it was recorded, and it cannot
react at all. If the environment or the opponent mix shifts between submission
and scoring, a fixed script has no recourse. Keep it as the fallback only.

## Evidence that adaptation has signal HERE

The day-6 tape selector has **no** per-world signal: real oracle +$41,666 vs an
interaction-shuffled null +$41,951 +/- 745. Which tape wins in which world is
noise, so selecting between tapes is unlearnable.

Executor-level adaptation is different, and measured. `c-chain` (allow CARE
before FEED on one tile visit) gated:

    pub-ahmed-v38       +8,072 +/- 1,843   26/32
    pub-tschinkel-psr   -5,891 +/- 2,407   13/32
    pub-boatlee-v16        +44 +/- 1,545   12/32

Both extremes individually significant. The optimum genuinely DIFFERS by
opponent, so a policy that identifies which regime it is in and switches has
real money available -- roughly the spread, not the pooled mean.

## Branch 1 — `exec-visit`: a new executor, visit-major instead of band-major

Measured defect. `scheduler.assign` orders work by PRIORITY BAND and uses
distance only as a tie-break within a band, so pass 1 sends a worker across its
zone for a P_GROW task while a P_FILLER task sits under its feet.

    moves                        3,864/game
    ...with work underfoot       2,846/game   (73.6% of moves)
    tasks abandoned underfoot    4,110/game

champ-h8 lands 38.2% of its actions with zero intervening moves; ahmed lands
49.6% and the tape 0.85 moves/action against our 1.45. This is the gap, and all
eleven previous routing attempts operated INSIDE the band structure, which is
why none of them could reach it.

Change: a worker finishes the tile it is standing on before it walks. Survival
work still preempts.

## Branch 2 — `exec-adaptive`: opponent regime detection + response bank

Shape from the literature: OPS-DeMo (arxiv 2406.06500) keeps an Assumed Opponent
Policy Bank and a Response Policy Bank and switches on a decaying error
estimate; DPOA (Sci. 2022) trains a deliberately diverse response set for
unknown adversaries.

Ours: classify the opponent online from public state, then pick the executor
variant that gated best against that class. We already hold the pieces --
`tools/opp_belief.py` (validated supply inference) and a growing set of gated
executor variants with per-opponent results.

Open question the branch must answer FIRST: is the regime identifiable early
enough to matter? Note the saturation feature is already refuted -- tschinkel
leaves the MOST milk room (197 vs ahmed's 114) and is where caring LOSES, so
whatever separates them is not market headroom.

## Branch 3 — `exec-varied`: deliberate variability

Randomise tie-breaks and a small set of parameters per game rather than shipping
one deterministic script. Two reasons: it cannot be decoded from replays the way
a tape can, and it hedges a shifted opponent population instead of committing to
the one we tuned against.

## Discipline carried over

- Screen a knob as a CONSTANT before training a head over it; screen the ORACLE
  form of a feature before building its estimator. Both paid for themselves.
- Measure a quantity's SCALE before choosing a threshold (three gates came back
  bit-identical because headroom runs to tens of units/day, not 0-1).
- A 4-6 seed money result is not evidence (match-p1: +$7,201 on 4 seeds,
  -$1,119 gated on 96). Structural metrics ARE low-variance; use those to screen.
- Gate on the WORST opponent, not the pooled mean.
