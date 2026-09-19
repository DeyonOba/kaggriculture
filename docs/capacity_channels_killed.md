# Capacity channels: a p=0.0001 R² win worth nothing

2026-09-19. `tools/s3_flow.py`, 177-channel build, 400 games.

The question was whether the encoder should carry market **capacity**
(`econ.units_sellable_above` — the exact number of units the market can absorb
before price falls through a floor) and a **temporal stack** of the opponent's
board. Both were computed somewhere in the project already and fed to nothing.

## The chain

Four measurements. Each one passes, and each fails to carry to the next.

| level | result | |
|---|---|---|
| history → their standing counts | +10–14% R² | PASS |
| history → their market flow | −0.0029 | closed |
| capacity → their market flow | **+0.0187, p=0.0001** | PASS |
| capacity → forecast MAE | **0%** | nothing |
| model → OOD forecast MAE | **−44%** | harmful |

## The R² win was real

`TM_full_157` (E + crop temporal stack + capacity) over `E_full_83`, six
paired seeds, all six positive:

```
mean +0.0187   sd 0.0044   se 0.0018   t(5) = +10.45   p = 0.0001
```

## And it moved the forecast by zero

Identical data, identical seed, identical 40 held-out games:

```
k=5 MAE    arithmetic 13.8
  E_full_83        12.1   closes 68%     83 channels
  TM_full_157      12.1   closes 67%    157 channels
  PERFECT          11.3
```

E is *better* at k=10 (25.2 vs 26.4) at half the channel count.

## Out of distribution it is worse than doing nothing

```
regime                arithmetic  BoardNet  PERFECT   gap closed
replay 2800+  IN-DIST      13.3      12.0     11.2       +65%
vs champ-h8       OOD       8.6       9.3      6.9       -44%
vs r-mrg100       OOD      10.1      10.5      8.0       -20%
```

Against both of our own agents, feeding the learned opponent-supply prediction
into the forward model makes the forecast **worse than the plain arithmetic
projection**. This is the deployment-distribution failure already on record
(+71% in-distribution, −13% vs champ-h8) repeating with a larger sign.

Running `tools/net_ood.py` is what turned a PASS into a kill, and it required
fixing its encoder first — it built a 91-channel tensor and could not score
any arm carrying temporal or capacity planes.

## Two corrections that fall out

**`E_full_83` was never better than `A_v1_72`.** +0.0012, t = +0.45, p = 0.67.
E had been treated as the best arm on a 2-seed +0.0077 that does not
replicate. The arms differ far less than the old unpaired reporting suggested.

**The oracle is not an upper bound at every horizon.** PERFECT is *worse* than
arithmetic at k=1 (4.9 vs 4.5), and `E_full_83` beats PERFECT at k=10 (25.2 vs
25.9). Opponent supply is not the binding error term away from k≈5, so
"% of the oracle gap closed" should only be quoted there.

## What is kept

Nothing is reverted. The encoder, the arms, `SIDES`, and the paired reporting
all stay — the measurement apparatus is what produced the kill, and three
tooling defects it exposed (unpaired stats on a paired design, an eval that
could not construct the winning tensor, an eval that loaded a stale model)
would have silently produced confident wrong answers on any future question.

What stops is spending remaining time on a **learned opponent-supply model**.
Every fitted model that has competed with the engine's closed form on the same
consumer has lost. The two zero-parameter closed forms remain the only things
that paid: the forward market model (+$3,185 ± 768, replicated) and computed
risk.

## Still unused rather than refuted

The SELL/BUY decomposition. `price_forecast.SIDES` and the `C`/`B` arrays now
carry gross sells and gross buys separately, because netting them erased 18%
of all market commits. At 400 games:

```
WHEAT       1,266,016 sells   923,905 buys   42.2%
FERTILIZER    771,960 sells   112,551 buys   12.7%
7 others    2,362,389 sells         0 buys    0.0%
```

Nothing consumes it yet. Note it is a *demand* signal, and every demand-side
argument in this project has lost so far — see `docs/`
and the market-structure notes — so it deserves a cheap test, not a build.
