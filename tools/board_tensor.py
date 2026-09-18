"""The board as a tensor: (C, 10, 10), the representation nothing here has had.

Every model this project has built sees the board as scalars. `raw_bc.feats`
reads the unit's tile plus its four neighbours -- 5 of 100 tiles -- and seven
aggregate counts. `kagent/features` and the value head see day-level aggregates
only. So no model of ours can represent WHERE anything is, and the deficit we
have measured most precisely is exactly that: consecutive worked tiles 1.92
Manhattan steps apart against ahmed's 1.60, moves per distinct tile 2.64
against 1.88.

This encodes the farm the way it actually is -- a grid -- so a convolution can
see adjacency, clustering and distance, none of which a count can express.

Channels (C = 24):

     0-4   crop one-hot: wheat, carrot, tomato, strawberry, melon
     5     crop age / 20
     6     yield_units / 6
     7     watered_today
     8     consecutive_unwatered / 2      <- the death predictor
     9     days until this crop first yields / 12
    10-12  animal one-hot: goose, cow, sheep
    13     fed_today
    14     cared_today
    15     fertilizer_available
    16     animal yield_units / 6
    17     empty and unlocked
    18     weed
    19     locked
    20     structure (coop or pasture, no animal)
    21     shed tile
    22     unit occupancy / 4
    23     distance to nearest shed tile / 12   (static)

Deliberately a grid and not a token sequence: 100 cells with strong locality is
what convolution is for, and a transformer over 100 tokens would cost more for
structure we can hand to the kernel directly.
"""

import numpy as np

N_CH = 24
CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_L = ["GOOSE", "COW", "SHEEP"]
SHED = [(4, 4), (4, 5), (5, 4), (5, 5)]
FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}


def board_tensor(obs, seat=0):
    """-> float32 (N_CH, n, n) for one farm at one moment."""
    farm = obs["farms"][seat]
    tiles = farm["tiles"]
    n = len(tiles)
    day = obs["day"]
    t = np.zeros((N_CH, n, n), dtype="float32")

    # static distance-to-shed channel
    for y in range(n):
        for x in range(n):
            t[23, y, x] = min(abs(x - a) + abs(y - b) for a, b in SHED) / 12.0
    for x, y in SHED:
        if 0 <= y < n and 0 <= x < n:
            t[21, y, x] = 1.0

    for y, row in enumerate(tiles):
        for x, c in enumerate(row):
            if c == "LOCKED":
                t[19, y, x] = 1.0
                continue
            if c is None:
                t[17, y, x] = 1.0
                continue
            if not isinstance(c, dict):
                continue
            kind = c.get("kind")
            if kind == "WEED":
                t[18, y, x] = 1.0
            elif kind == "PLANT":
                crop = c.get("crop")
                if crop in CROPS_L:
                    t[CROPS_L.index(crop), y, x] = 1.0
                age = day - c.get("planted_day", day)
                t[5, y, x] = min(age, 20) / 20.0
                t[6, y, x] = min(c.get("yield_units", 0), 6) / 6.0
                t[7, y, x] = 1.0 if c.get("watered_today") else 0.0
                t[8, y, x] = min(c.get("consecutive_unwatered", 0), 2) / 2.0
                t[9, y, x] = max(0, FIRST_YIELD.get(crop, 0) - age) / 12.0
            elif "animal" in c:
                a = c.get("animal")
                if a in ANIMALS_L:
                    t[10 + ANIMALS_L.index(a), y, x] = 1.0
                t[13, y, x] = 1.0 if c.get("fed_today") else 0.0
                t[14, y, x] = 1.0 if c.get("cared_today") else 0.0
                t[15, y, x] = 1.0 if c.get("fertilizer_available") else 0.0
                t[16, y, x] = min(c.get("yield_units", 0), 6) / 6.0
            else:
                t[20, y, x] = 1.0          # empty coop / pasture

    units = [tuple(farm.get("farmer") or (0, 0))]
    units += [tuple(h) for h in (farm.get("hands") or [])]
    for ux, uy in units:
        if 0 <= uy < n and 0 <= ux < n:
            t[22, uy, ux] = min(t[22, uy, ux] * 4 + 1, 4) / 4.0
    return t


def scalars(obs, seat=0):
    """Context a convolution cannot see: time, cash, prices, shop draw."""
    from kagent.constants import MARKET_PARAMS
    farm = obs["farms"][seat]
    px = (obs.get("market") or {}).get("prices") or {}
    inv = (obs.get("market") or {}).get("inventory") or {}
    shops = list((obs.get("town") or {}).get("unlocked_shops", []))
    v = [obs["day"] / 30.0, obs["hour"] / 24.0,
         min(float(farm.get("money", 0)), 60000) / 60000.0,
         min(len(farm.get("hands") or []), 14) / 14.0,
         min(len(farm.get("unlocked_quadrants") or ["NW"]), 4) / 4.0,
         min(len(shops), 8) / 8.0]
    for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]:
        base = MARKET_PARAMS[p]["base"]
        v.append(min(float(px.get(p, base)) / base, 3.0) / 3.0)
        v.append(min(abs(float(inv.get(p, 10000)) - 10000), 600) / 600.0)
    return np.asarray(v, dtype="float32")


N_SCALAR = 6 + 9 * 2


# ---------------------------------------------------------------------------
# S1: both farms in ONE tensor.
#
# Everything above encodes a single farm. That was the deficit: the forward
# market model's ablation (tools/price_forecast.py) prices the OPPONENT's
# board at 36% of k=5 forecast error, and no model here has ever seen it.
#
# Layout, (N_PAIR, 10, 10):
#     0 .. 23      our farm      (board_tensor, seat)
#    24 .. 47      their farm    (board_tensor, 1 - seat)
#    48 .. 47+S    globals, broadcast across every cell so a convolution can
#                  see price and day at every pixel
#
# Their shed is deliberately absent: an ORACLE on it moved k=5 forecast error
# 13.0 -> 12.8, so there is nothing there to encode.
# ---------------------------------------------------------------------------

N_PAIR = 2 * N_CH + N_SCALAR


def board_pair(obs, seat=0):
    """-> float32 (N_PAIR, n, n): our farm, their farm, broadcast globals."""
    mine = board_tensor(obs, seat)
    theirs = board_tensor(obs, 1 - seat)
    n = mine.shape[-1]
    g = scalars(obs, seat)
    out = np.empty((N_PAIR, n, n), dtype="float32")
    out[:N_CH] = mine
    out[N_CH:2 * N_CH] = theirs
    out[2 * N_CH:] = g[:, None, None]
    return out


def counts(obs, seat=0):
    """Plain standing counts per farm -- the ridge baseline S2 must beat."""
    v = []
    for s in (seat, 1 - seat):
        farm = obs["farms"][s]
        c = dict.fromkeys(CROPS_L + ANIMALS_L, 0)
        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    c[t["crop"]] = c.get(t["crop"], 0) + 1
                elif t.get("animal"):
                    c[t["animal"]] = c.get(t["animal"], 0) + 1
        v += [float(c[k]) for k in CROPS_L + ANIMALS_L]
        v.append(min(float(farm.get("money", 0)), 60000) / 1000.0)
        v.append(float(len(farm.get("unlocked_quadrants") or ["NW"])))
    v.append(float(obs["day"]))
    v.append(float(len((obs.get("town") or {}).get("unlocked_shops", []))))
    return np.asarray(v, dtype="float32")


N_COUNT = 2 * (len(CROPS_L) + len(ANIMALS_L) + 2) + 2
