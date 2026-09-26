# The three leaderboard leaders against our executor, measured

*26 September 2026. M & M & P & Q (12 replays, `replays/mmpq/`), Boey (14
replays, `replays/boey/`, submission 56521745, rated 3,113) and DSM (14
replays, `replays/dsm/`, submission 56564674, rated 3,016), fetched with
`tools/fetch_team_replays.py`; ours is today's dev tree with h8's knob rules
on three traced held-out games (`b2/b3/b4_base`), and the same games with
the five new switchable rules on (`allin`: last-day water, hour 23, harvest
on visit, spent, layout, fertiliser gate ×2). Numbers are per game, averaged.
Tools: `tools/tile_metrics.py` (replay teams, `--replay-glob`), `tools/sell_metrics.py`;
raw tables in `docs/results/top3_tile_metrics.txt` and `top3_sell_metrics.txt`.*

The three leaders win 11–12 of their 12–14 games here, mostly against each
other and the rest of the top ten, at $105–122k a game. Ours lose all three
traced games at $70k. Their opponents are not ours, so money is not
comparable; the executor metrics are.

## 1. Yield and harvest

| per game | MMPQ | Boey | DSM | ours today | ours, new rules |
|---|---|---|---|---|---|
| wheat units per harvest | 4.38 | 4.58 | 4.67 | 3.17 | 5.15 |
| carrot units per harvest | 2.94 | 3.26 | 3.41 | 2.00 | 2.98 |
| one-time crop units | 889 | 1,043 | 1,077 | 335 | 615 |
| one-time harvests taken unwatered on the window's last day | 9% / 5% | 2% / 3% | 1% / 5% | **100% / 100%** | 26% / 9% |
| replant within one step of a harvest | 0.78 | 0.86 | 0.88 | 0.60 | 0.53 |
| harvests | 470 | 511 | 522 | 277 | 332 |
| plantings | 266 | 276 | 271 | 153 | 178 |
| fertilize ops | 220 | 209 | 223 | 116 | 134 |

All three water a one-time crop on its last window day and harvest after
(the engine adds a unit per in-window water); we harvested first on every
one of them. All three harvest an ongoing crop on every production and
replant a bare tile the next step.

## 2. Labour and movement

| per game | MMPQ | Boey | DSM | ours today | ours, new rules |
|---|---|---|---|---|---|
| chores per tile visit | 2.08 | 1.90 | 1.91 | 1.42 | 1.59 |
| on animal tiles | 2.77 | 1.82 | 1.98 | 1.64 | 1.79 |
| on crop tiles | 1.76 | 1.95 | 1.85 | **1.19** | 1.40 |
| visits with two or more chores | 60% | 52% | 54% | 32% | 43% |
| moves per productive op | 0.71 | 0.83 | 0.81 | 1.39 | 1.10 |
| productive ops | 3,595 | 3,734 | 3,807 | 2,841 | 3,116 |
| moves | 2,557 | 3,096 | 3,074 | 3,934 | 3,414 |
| hour-23 moves, share of moves | 0.0% | 4.5% | 2.6% | 5.1% | 0.0% |
| hire orders / at hours 0–3 | 257 / 100% | 277 / 100% | 284 / 100% | 269 / 100% | 267 / 100% |

Per-game spread: chores per visit 1.91–2.28 (MMPQ), 1.80–2.01 (Boey),
1.79–2.01 (DSM); moves per productive op 0.67–0.77, 0.75–0.91, 0.75–0.88.
The crop-tile visit is where all three are alike and we are not: they
water, harvest and plant on one stand (1.8–2.0 chores), we do one thing and
leave (1.2). Only MMPQ never moves on the day's last turn.

## 3. Hauling and feeding

| per game | MMPQ | Boey | DSM | ours today | ours, new rules |
|---|---|---|---|---|---|
| shed ops (PICKUP+DROP) per unit-day | 8.9 | 10.1 | 10.5 | **14.2** | 16.4 |
| pickups per FEED | 0.55 | 0.46 | 0.48 | 0.99 | 1.10 |
| feeds from wheat harvested in hand | 29% | 16% | 16% | 2% | 4% |
| pickups at hours 0–3 / 4–11 | 100% / 1% | 63% / 23% | 64% / 25% | 82% / 11% | 80% / 7% |
| feed ops | 356 | 409 | 397 | 357 | 361 |
| what a unit does at hours 0–2, days 6–20 | PICKUP 47%, MOVE 34%, PASS 15% | MOVE 48%, COLLECT 21%, PICKUP 18% | MOVE 52%, PICKUP 18%, COLLECT 8%, HARVEST 8% | PICKUP 51%, MOVE 24%, FEED 10% | |

All three haul less than we do for the same feeding: half the pickups per
feed, a sixth to a third of feeds from wheat in hand. MMPQ does its whole
day's provisioning at hours 0–2 (one pickup per unit) and then never goes
back; Boey and DSM spread a quarter of their pickups over the morning.

## 4. Tile use, weeds and layout

| per game | MMPQ | Boey | DSM | ours today | ours, new rules |
|---|---|---|---|---|---|
| idle share of unlocked tiles at dawn | 6.5% | 4.0% | 3.7% | 13.0% | 17.8% |
| of which bare | 3.5% | 3.4% | 3.2% | 10.5% | 13.1% |
| of which weed | 2.2% | 0.2% | 0.1% | 1.7% | 4.1% |
| weed tile-days | 61 | **3** | **3** | 36 | 89 |
| water ops with no effect | 9% | 1% | 1% | 9% | 6% |
| animals at day 15 | 20.6 | 20.9 | 20.9 | 16.3 | 16.7 |
| plants at day 15 | 54 | 54 | 54 | 57 | 54 |
| wheat / strawberry / carrot / tomato / melon tiles at day 15 | 14 / 25 / 3 / 3 / 10 | 23 / 26 / 3 / 0 / 2 | 19 / 28 / 3 / 0 / 4 | 16 / 40 / 0 / 0 / 1 | 21 / 33 / 0 / 0 / 0 |
| wheat tiles orthogonally beside a pasture or coop | 46% | 36% | 42% | 9% | 68% |
| animal distance to the shed / plant distance | 3.5 / 4.2 | 2.1 / 4.7 | 2.1 / 4.7 | 2.5 / 4.4 | 2.2 / 4.4 |

Boey and DSM run a farm with almost no weeds and almost no wasted water;
MMPQ tolerates weeds (61 tile-days) and still beats us on idle ground. All
three carry four to five more animals than we do at day 15 on a farm with
12–15 fewer strawberry tiles, and put their wheat beside the animals. Only
MMPQ places animals away from the shed; Boey and DSM keep them as close as
we do, so the "animals far, crops near" layout is one player's habit, the
wheat-beside-pasture row is everyone's.

## 5. Selling

| per game | MMPQ | Boey | DSM | ours (three games) |
|---|---|---|---|---|
| realised price over the product's game-mean price, all products | 1.064 | 0.955 | 1.016 | 0.86–0.93 |
| share of units sold at hours 20–23 | 31% | 22% | 29% | 8–12% |
| strawberry: units per order / price after ÷ before / realised | 8.4 / 0.94 / 1.05 | 3.4 / 0.95 / 1.03 | 3.6 / 0.89 / 0.94 | 10–14 / 0.41–0.78 / 0.16–0.89 |
| milk: units per order / impact / realised | 6.1 / 0.88 / 1.04 | 3.4 / 0.95 / 0.89 | 3.8 / 0.85 / 0.94 | 7–8 / 0.51–0.98 / 0.88–1.05 |
| wool: units per order / impact / realised | 7.0 / 0.90 / 1.08 | 2.7 / 0.76 / 1.12 | 3.1 / 0.70 / 1.32 | 5–7 / 0.35–0.64 / 0.74–1.00 |
| melon: units per order / impact / realised | 13.6 / 0.86 / 0.95 | 4.6 / 0.98 / 1.17 | 6.8 / 0.93 / 1.18 | 10–13 / 0.79–0.87 / 1.11–1.12 |

Boey's order counts are inflated (3,669 SELL orders a game): it re-posts
the same request every step until it fills, so its "units per order" is
the request size, not the fill. Read the price columns, not the volumes.
Boey and DSM sell in three-to-four-unit requests and lose 5–11% per order
where our ten-unit dumps lose 22–59%; all three sell a quarter to a third
of their units in the evening, before the hour-0 dumps. Our own pacing
rule (`sell_pace`) did not recover the realised price on traces and the
evening run lost paired, so this family is measured rather than open.

## 6. What the three share and we lack, ranked

Evidence weight: a behaviour all three show is strong; two of three is
noted; one of three is weak. Sizes are the metric gap; the money each is
worth is unknown until it is paired on the 995-game draw, except where a
paired number already exists.

1. **Harvest after the last window water, then replant the next step** (all
   three: 1–9% unwatered last-day harvests, replant 0.78–0.88 within a step).
   Ours 100% / 0.60. Metric: one-time units 335 → 615 with the rule on.
   Paired on the old draw the water fix alone was +$2.1k, +0.9 pt.
2. **Two chores per crop visit** (all three: 1.76–1.95 chores on a crop
   stand; ours 1.19). The stand is water → harvest → plant → water. Metric:
   chores per visit, productive ops (+750–950 a game at their rate).
3. **Harvest every production of an ongoing crop** (all three: 470–522
   harvests; ours 277). Metric: harvests; realised price rises when produce
   reaches the market on its production day.
4. **Feed from the wheat in hand, wheat rows beside the pastures** (all
   three: 16–29% of feeds from hand, wheat beside an animal 36–46%, pickups
   per feed 0.46–0.55; ours 2%, 9%, 0.99). Metric: shed ops per unit-day
   14.2 → 9–10.5, about 190 fewer pickups a game.
5. **One provisioning run per unit per day** (MMPQ strictly at hours 0–2;
   Boey and DSM 64% at 0–3 and the rest by hour 11; none in the afternoon).
   Ours keep fetching all day (11% at 4–11, 8% after noon). Metric: shed
   ops per unit-day, moves per productive op.
6. **Fertilise at about 210–220 a game** (all three; ours 116). Fertilised
   wheat waters add two units instead of one. Metric: wheat units per
   harvest 4.4–4.7; the fertiliser gate ×2 reaches 4.8–5.2 on traces.
7. **Keep bare ground under 4%** (all three: 3.2–3.5% bare at dawn; ours
   10.5%). Follows from 1–3 rather than from a separate rule: the melon
   replant lag and the endgame cap are our two bare-ground events.
8. **No weeds** (Boey and DSM: 3 weed tile-days a game, 1% wasted waters;
   MMPQ 61, 9%; ours 36, 9%). Two of three. Their plants do not die dry:
   every plant is watered on schedule. Metric: weed tile-days, water ops
   with no effect.
9. **Four to five more animals by day 15, twelve fewer strawberry tiles**
   (all three: 20.6–20.9 animals, 25–28 strawberries; ours 16.3, 40). This
   is the planner's allocation, not the executor's; it is the largest
   structural difference on the board and is listed for the planner track.
10. **Sell in small requests, a quarter to a third in the evening** (Boey
    and DSM 3–4 unit requests; MMPQ 6–14; all three 22–31% evening). Our
    pacing and evening-run rules did not convert; measured, not open.
11. **No moves on the last turn** — MMPQ only (Boey 4.5%, DSM 2.6% of
    moves). Weak evidence; free for us either way.
12. **Animals away from the shed** — MMPQ only (3.5 tiles; Boey and DSM 2.1,
    ours 2.5). Weak evidence; not a shared pattern.

## 7. Where the three differ

MMPQ tolerates weeds and wasted water that Boey and DSM never allow; MMPQ
provisions once at dawn where the other two spread a quarter of their
pickups over the morning; MMPQ plants tomatoes and ten melons at day 15
where Boey plants two and DSM four; MMPQ's animal-tile stands are the
longest (2.8 chores: feed, care, collect, harvest) while Boey's and DSM's
are like ours; only MMPQ keeps the animals away from the shed and never
moves at hour 23. Boey re-posts market orders every step (a wrapper
artefact, not a strategy). The shared core is items 1–7 above.
