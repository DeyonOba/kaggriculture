# The board track: what is actually unanswered

Written 2026-09-29, before spending another day of compute on it. Every number
below is from the record, not from an argument.

## What the board track HAS established

`T_crop_103` on top-player boards, three local seeds, 785 paired games:

```
top_board_s1234   0.180   vs champ_h8  -0.004 +/- 0.017  p 0.879
top_board_s1235   0.168   vs champ_h8  -0.015 +/- 0.016  p 0.385
top_board_s1236   0.164   vs champ_h8  -0.019 +/- 0.016  p 0.267
dev_knobs_h8      0.242   vs champ_h8  +0.059 +/- 0.016  p 0.000
```

And on the 995-game instrument, the kernel's own board net:

```
kernel_board      0.249  $88,857   vs champ_h8 +0.019, z 1.27, p 0.230
champ_h8          0.230  $82,345
dev_knobs_h8      0.313  $86,408   vs champ_h8 +0.082, z 5.59
kernel_planner    0.217            vs champ-h10 -0.137, z -8.54, 6/6 bands
```

So: **indistinguishable from champ-h8 on every seed, beaten by the plain
heuristic on the same games, and 10.5 points under the shipped champion.** It
also banks the MOST money of any agent while winning least (margin -$11.2k vs
champ-h10's -$2.4k) -- the market-yielding signature every trained-trunk board
arm has shown (B2 +$3.3k, B3 +$6.0k, B5b +$5.4k, B5c +$7.6k to the opponent).

The one thing that did lift the RL net is the TOP-PLAYER CLONE, not the board:
0.074-0.080 -> 0.127-0.180. The board trunk is worth about +2.0 on top of it,
sign 2 of 3 seeds, inside its own between-seed noise.

## Q1 -- CAN THE ACTION SPACE EVEN SAY WHAT THE LEADERS DO? Answered today: no.

Measured over 1,200 leader dawns at hour 12, every label pushed through
`rl/replay_bc._targets_to_choices` and decoded BACK to absolute tiles, so
residual (v1/v2), fraction (v3) and integer (v4) encodings are compared fairly:

```
              exact   mean error        worst heads at v1
v1            37.8%   2.16 tiles        wheat 6.1% / 6.11 tiles
v2            42.7%   2.08 tiles        strawberry 9.1% / 6.78 tiles
v3            35.0%   3.51 tiles        hires 12.8% / 1.11
v4            98.7%   0.04 tiles
```

**`HEADS_VERSION = 1`, and `rl/session_bc.py` had no `--heads-version` flag at
all**, so every board net ever trained here was fitted to a target that names
the wrong number on 62% of dawns and misses by two tiles on average. v3 is
WORSE than v1, which the code comment already suspected: an absolute target of
2+ lands on the 1.5 fraction for every crop and animal, so the corpus teaches
"always ask for 1.5x the anchor".

This is not the usual observational/interventional trap. It is a
REPRESENTATIONAL CEILING: the net cannot learn what the label cannot say. It
is also the mechanism behind a finding already on record -- the trained planner
chose `goose_target = 0` on all 30 days of all 3 traced games, because the
first non-zero option was 8 while the leaders run 5-6.

Fixed: `--heads-version` added, verified end to end (v1 = 13 heads/57 logits,
v4 = 15 heads/240 logits, and the emitted LABELS differ in shape, which only
happens if the spawn worker got the version rather than the parent).

**What is still unknown: whether better labels buy anything.** BC accuracy is
exactly the kind of held-out number that has misled here before -- the cloned
executor hit 69.4% agreement and gated null.

## Q2 -- Does the board beat its own no-board control on the GOOD instrument?

**Never asked.** The 995-game gate submitted `kernel_board` and
`local_board_s1234` only; the no-board control has never been on that draw. The
question was settled on the 785 instrument at +2.0 mean, sign 2 of 3, inside
one seed's noise -- and the record's own note says "the two Kaggle seeds decide
it", and those two seeds were never run.

## Q3 -- Does any of it approach the SHIPPED champion?

The pre-registered bar was "beating the capped control in >= 2 of 3 seeds, and
for relevance to the ladder, approaching `dev_champion`." Every 995 number is
paired against `champ_h8` at 0.230. On the same draw `dev_knobs_h8` is 0.313,
and champ-h11 is around 0.44. **The gap to the live champion has never been
measured for the best board arm** -- only to an agent we stopped shipping.

## Q4 -- Is a top-band-only corpus a distribution mismatch?

The same shape has already failed once: a net trained on both-seats-2800+
closed +71% in-distribution and **-13% vs champ-h8**, because the gate draws
across six rating bins and five of them were unseen. The fix was retraining on
the natural mix. Training on the top boards only is that experiment again, and
"fine-tune on the top 10" narrows it further.

The counter-argument is on record too: the top-player clone is the ONLY
intervention measured to lift the RL net. So this is a real tension, not a
settled question, and it wants the OOD check (`tools/net_ood.py`) run before
any gate is read as a verdict.

## Q5 -- Does the trunk train, and does anyone check?

`val_r2` appears nowhere in the top-board logs. The trunk's held-out margin R2
is +0.2517 (kernel, 5 of 16 chunks -- budget exhausted) and +0.2750 local. The
Stage B record is that five board arms ran twenty iterations each with
`policy_updates = 0` and were read as null results for three runs, because the
trunk was never serialised and the worker played a 216-input policy fed 168
numbers. Eight guards exist now; they must be read before any number is.

## Q6 -- Eleven daily datasets the corpus has never seen

~7,000 episodes from 14-24 September, top score 3,100-3,240. The corpus ends on
13 September. The standing memory is that **every real jump here came from
ingesting an external corpus, and internal sweeps gate flat.** This is the
cheapest unexplored lever in the whole track and it is not a modelling change.

## The order these can be answered in

1. **Q1's second half** -- v1 vs v4 clone on the SAME sessions, compared in
   TILE space, not in choice-index space. (A v4 model at 100% index accuracy
   and a v1 model at 100% index accuracy are not equally right: v1's target is
   off by 2.16 tiles by construction. Comparing accuracies directly is the
   methodological trap here.)
2. **Q5** -- read `policy_updates`, `onpolicy_kl`, `val_r2` before anything.
3. **Q2** -- board vs no-board, same seeds, on the 995 draw.
4. **Q3** -- and against champ-h11, not champ-h8.
5. **Q4** -- `tools/net_ood.py` before reading any of it as a verdict.
