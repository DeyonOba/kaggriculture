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


# ---------------------------------------------------------------------------
# v2 encoder. Five defects measured on the 8,000-sample dawn dataset:
#
#   1. EIGHT OF 72 CHANNELS WERE IDENTICALLY ZERO. watered_today, fed_today
#      and cared_today are reset by the engine at midnight and we sample at
#      midnight, so all six (both farms) carried nothing; so did the `hour`
#      and hands-count scalars. Their information is not lost --
#      consecutive_unwatered is the engine's integral of the same thing and
#      survives the reset -- but the planes were dead weight.
#   2. THE WORKFORCE WAS INVISIBLE. `_end_of_day` clears farm["hands"], and
#      hiring resolves during hour 0, so a dawn observation holds ZERO hands
#      (measured: hour 0 -> 0.00, hour 2 -> 8.10). ch22 merged farmer and
#      hands, so it showed exactly one cell every sample.
#   3. Farmer and hands were indistinguishable.
#   4. Our OWN shed is legitimately visible and was unused, though the
#      forecast ablation prices it at 13.3 -> 13.0.
#   5. Dawn sampling cannot simply move: the planner decides at hour 0
#      (HeuristicPolicy.act rebuilds the plan when day_plan_day != view.day),
#      so mid-day states are ones the agent can never present. The fix is
#      YESTERDAY's mid-day snapshot, which Memory can carry to today's hour 0.
# ---------------------------------------------------------------------------

DEAD = (7, 13, 14)                       # watered_today, fed_today, cared_today
LIVE = [i for i in range(N_CH) if i not in DEAD]
N_LIVE = len(LIVE)                       # 21
MID_PLANES = 4                           # watered, fed, cared, hands (yesterday noon)
PRODUCTS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]


def _units(farm):
    f = farm.get("farmer") or (0, 0)
    return tuple(f), [tuple(h) for h in (farm.get("hands") or [])]


def farm_planes(obs, seat):
    """-> (N_LIVE + 1, n, n): live channels, with farmer and HANDS separated.

    ch22 of the base encoder merged both; here it keeps the farmer and the
    extra plane counts hands, so a convolution can tell a lone farmer from a
    crew of eight standing on the same tiles.
    """
    base = board_tensor(obs, seat)
    n = base.shape[-1]
    out = np.zeros((N_LIVE + 1, n, n), dtype="float32")
    out[:N_LIVE] = base[LIVE]
    farm = obs["farms"][seat]
    fx, fy = _units(farm)[0]
    idx22 = LIVE.index(22)
    out[idx22] = 0.0
    if 0 <= fy < n and 0 <= fx < n:
        out[idx22, fy, fx] = 1.0          # farmer only
    for hx, hy in _units(farm)[1]:
        if 0 <= hy < n and 0 <= hx < n:
            out[N_LIVE, hy, hx] = min(out[N_LIVE, hy, hx] * 8 + 1, 8) / 8.0
    return out


def mid_planes(obs, seat):
    """-> (MID_PLANES, n, n) from a MID-DAY observation: the layer midnight wipes."""
    farm = obs["farms"][seat]
    tiles = farm["tiles"]
    n = len(tiles)
    out = np.zeros((MID_PLANES, n, n), dtype="float32")
    for y, row in enumerate(tiles):
        for x, c in enumerate(row):
            if not isinstance(c, dict):
                continue
            if c.get("kind") == "PLANT":
                out[0, y, x] = 1.0 if c.get("watered_today") else 0.0
            elif c.get("animal"):
                out[1, y, x] = 1.0 if c.get("fed_today") else 0.0
                out[2, y, x] = 1.0 if c.get("cared_today") else 0.0
    for hx, hy in _units(farm)[1]:
        if 0 <= hy < n and 0 <= hx < n:
            out[3, hy, hx] = min(out[3, hy, hx] * 8 + 1, 8) / 8.0
    return out


def scalars2(obs, seat=0):
    """Globals with the two dead entries dropped and OUR OWN shed added."""
    v = list(scalars(obs, seat))
    v = [x for i, x in enumerate(v) if i not in (1, 3)]      # hour, hands count
    shed = (obs.get("private") or {}).get("shed") or {}
    for p in PRODUCTS_L:
        v.append(min(float(shed.get(p, 0)), 100.0) / 100.0)
    return np.asarray(v, dtype="float32")


N_SCALAR2 = (N_SCALAR - 2) + len(PRODUCTS_L)
N_PAIR2 = 2 * (N_LIVE + 1) + 2 * MID_PLANES + N_SCALAR2


def board_pair2(obs, seat=0, prev_mid=None):
    """-> (N_PAIR2, n, n). prev_mid: yesterday's MID-DAY observation, or None."""
    mine = farm_planes(obs, seat)
    theirs = farm_planes(obs, 1 - seat)
    n = mine.shape[-1]
    out = np.zeros((N_PAIR2, n, n), dtype="float32")
    k = N_LIVE + 1
    out[:k] = mine
    out[k:2 * k] = theirs
    if prev_mid is not None:
        out[2 * k:2 * k + MID_PLANES] = mid_planes(prev_mid, seat)
        out[2 * k + MID_PLANES:2 * k + 2 * MID_PLANES] = mid_planes(prev_mid, 1 - seat)
    out[2 * k + 2 * MID_PLANES:] = scalars2(obs, seat)[:, None, None]
    return out


# ---------------------------------------------------------------------------
# Superset encoder for ABLATION. Carries every channel any arm might want, so
# one dataset build serves all arms and each arm is a pure channel slice --
# no confound from differently-built datasets.
#
#   [0        : 24)   our farm, BASE 24 incl. the three reset-at-midnight planes
#   [24       : 25)   our hands, split out from the merged occupancy plane
#   [25       : 49)   their farm, BASE 24
#   [49       : 50)   their hands
#   [50       : 54)   our yesterday-MID-DAY planes   (watered, fed, cared, hands)
#   [54       : 58)   their yesterday-mid-day planes
#   [58       : 82)   globals, BASE 24 incl. the dead hour and hands-count
#   [82       : 91)   our own shed, 9 products
# ---------------------------------------------------------------------------

F_OURS = (0, 24)
F_OUR_HANDS = (24, 25)
F_THEIRS = (25, 49)
F_THEIR_HANDS = (49, 50)
F_OUR_MID = (50, 54)
F_THEIR_MID = (54, 58)
F_GLOBAL = (58, 82)
F_SHED = (82, 91)
N_FULL = 91

# indices of the structurally-dead planes inside a BASE 24-channel farm block
DEAD_IN_FARM = (7, 13, 14)
DEAD_IN_GLOBAL = (1, 3)          # hour, hands count


def _hands_plane(farm, n):
    out = np.zeros((1, n, n), dtype="float32")
    for hx, hy in _units(farm)[1]:
        if 0 <= hy < n and 0 <= hx < n:
            out[0, hy, hx] = min(out[0, hy, hx] * 8 + 1, 8) / 8.0
    return out


def board_pair_full(obs, seat=0, prev_mid=None):
    """-> (N_FULL, n, n). Every channel; arms are slices of this."""
    n = len(obs["farms"][seat]["tiles"])
    out = np.zeros((N_FULL, n, n), dtype="float32")
    for base, hbase, s in ((F_OURS, F_OUR_HANDS, seat),
                           (F_THEIRS, F_THEIR_HANDS, 1 - seat)):
        b = board_tensor(obs, s)
        # keep the merged occupancy plane as FARMER ONLY, hands get their own
        farm = obs["farms"][s]
        fx, fy = _units(farm)[0]
        b[22] = 0.0
        if 0 <= fy < n and 0 <= fx < n:
            b[22, fy, fx] = 1.0
        out[base[0]:base[1]] = b
        out[hbase[0]:hbase[1]] = _hands_plane(farm, n)
    if prev_mid is not None:
        out[F_OUR_MID[0]:F_OUR_MID[1]] = mid_planes(prev_mid, seat)
        out[F_THEIR_MID[0]:F_THEIR_MID[1]] = mid_planes(prev_mid, 1 - seat)
    out[F_GLOBAL[0]:F_GLOBAL[1]] = scalars(obs, seat)[:, None, None]
    shed = (obs.get("private") or {}).get("shed") or {}
    out[F_SHED[0]:F_SHED[1]] = np.asarray(
        [min(float(shed.get(p, 0)), 100.0) / 100.0 for p in PRODUCTS_L],
        dtype="float32")[:, None, None]
    return out


def _farm_idx(base, drop_dead):
    lo, hi = base
    return [lo + i for i in range(hi - lo)
            if not (drop_dead and i in DEAD_IN_FARM)]


def arm_channels(name):
    """Channel indices for one ablation arm."""
    g_all = list(range(*F_GLOBAL))
    g_live = [i for k, i in enumerate(g_all) if k not in DEAD_IN_GLOBAL]
    ours_all, theirs_all = _farm_idx(F_OURS, False), _farm_idx(F_THEIRS, False)
    ours_lv, theirs_lv = _farm_idx(F_OURS, True), _farm_idx(F_THEIRS, True)
    hands = list(range(*F_OUR_HANDS)) + list(range(*F_THEIR_HANDS))
    mid = list(range(*F_OUR_MID)) + list(range(*F_THEIR_MID))
    shed = list(range(*F_SHED))
    A = ours_all + theirs_all + g_all                       # 72, == v1
    B = ours_lv + theirs_lv + g_live                        # 64, dead dropped
    C = B + hands                                           # 66
    D = C + mid                                             # 74
    E = D + shed                                            # 83, == v2
    T_ARM = E + list(range(N_FULL, N_FULL + N_TEMPORAL))   # E + opponent lags
    T_DEEP = E + list(range(N_FULL, N_FULL + N_TEMPORAL_D))
    return {"A_v1_72": A, "B_drop_dead_64": B, "C_hands_66": C,
            "D_midday_74": D, "E_full_83": E, "F_shed_aux_83": E,
            "T_temporal_95": T_ARM, "T_crop_103": T_DEEP}[name]


ARMS = ["A_v1_72", "B_drop_dead_64", "C_hands_66", "D_midday_74",
        "E_full_83", "F_shed_aux_83", "T_temporal_95", "T_crop_103"]


# ---------------------------------------------------------------------------
# TEMPORAL opponent board. `board_pair_full` gives the opponent as ONE frame
# plus four planes of yesterday noon, and I justified that with the Markov
# result -- which was measured on PRICE, where the engine integrates history
# into the observation (consecutive_unwatered, cumulative market inventory).
#
# Opponent SUPPLY is a different target class and history measurably helps it.
# tools/target_horizon.py, predicting their standing counts 10 days out:
#
#     STRAWBERRY  0.56 -> 0.87     WHEAT   0.08 -> -0.15
#     GOOSE       0.54 -> 0.90     MELON   0.59 ->  0.07
#     SHEEP       0.44 -> 0.70     CARROT  0.38 ->  0.20
#     COW         0.39 -> 0.59
#
# History helps the PLATEAU quantities (strawberry, the animals) that ramp to
# a target and hold, and HURTS the rotated ones (wheat, melon, carrot) whose
# standing count oscillates. So the temporal stack carries only the plateau
# planes, not the whole board -- feeding the rotated ones back in is the part
# measured to be harmful.
#
# Lag spacing is geometric (1, 3, 10 days) rather than 1..10: the gain at H=10
# came from FIVE-to-TEN days back, and ten consecutive frames would be ten
# nearly-identical boards at 21 channels each.
# ---------------------------------------------------------------------------

PLATEAU_PLANES = [3, 10, 11, 12]     # strawberry one-hot, goose, cow, sheep
LAGS = (1, 3, 10)
N_TEMPORAL = len(PLATEAU_PLANES) * len(LAGS)

# `tools/k_ablation.py` then measured the SAME question directly -- per-kind
# R^2 on the opponent's standing count, with and against history, identical
# window, held out by game -- and came back the other way round:
#
#     gain from history, H=5 days:  melon +0.26  wheat +0.17  tomato +0.14
#                                   strawberry +0.07
#                                   cow +0.00  sheep +0.01  goose +0.00
#
# The animals are already predicted at R^2 0.82-0.94 from the current frame
# alone, because an animal is bought once and then stands there -- the count
# IS the forecast, and history has nothing to add. Crops rotate, so the phase
# of the rotation is only legible over time, and melon (the one crop no shop
# buys, on a 10-day clock) gains most of all.
#
# So the plateau restriction is backwards FOR THIS TARGET. It is not
# necessarily wrong for the one it was fitted on -- that was BoardNet's price
# head, a different label -- so both sets stay reachable and the old arm keeps
# its exact channel list. HIST_PLANES is what the forward model should use.
HIST_PLANES = [0, 1, 2, 3, 4]        # crop one-hot: wheat carrot tom straw melon
LAGS_DEEP = (1, 3, 5, 10)            # measured best; adding d5 is worth +0.024
N_TEMPORAL_D = len(HIST_PLANES) * len(LAGS_DEEP)


def temporal_opp(hist, seat, n=10, planes=None, lags=None):
    """-> (len(planes)*len(lags), n, n) from `hist`, oldest-last past dawns.

    hist[0] is the most recent PAST day; a lag beyond what we hold repeats the
    oldest available frame rather than zero-padding, so early game looks like
    "nothing has changed yet" instead of "the board was empty".
    """
    planes = PLATEAU_PLANES if planes is None else planes
    lags = LAGS if lags is None else lags
    out = np.zeros((len(planes) * len(lags), n, n), dtype="float32")
    if not hist:
        return out
    for li, lag in enumerate(lags):
        o = hist[min(lag - 1, len(hist) - 1)]
        try:
            b = board_tensor(o, 1 - seat)
        except Exception:
            continue
        for pi, ch in enumerate(planes):
            out[li * len(planes) + pi] = b[ch]
    return out


N_PAIR_T = N_FULL + N_TEMPORAL
N_PAIR_TD = N_FULL + N_TEMPORAL_D


def board_pair_temporal(obs, seat=0, prev_mid=None, hist=None):
    """board_pair_full plus the opponent's plateau planes at lags 1, 3, 10."""
    base = board_pair_full(obs, seat, prev_mid=prev_mid)
    n = base.shape[-1]
    out = np.zeros((N_PAIR_T, n, n), dtype="float32")
    out[:N_FULL] = base
    out[N_FULL:] = temporal_opp(hist or [], seat, n)
    return out


def board_pair_temporal_deep(obs, seat=0, prev_mid=None, hist=None):
    """board_pair_full plus the opponent's CROP planes at lags 1, 3, 5, 10."""
    base = board_pair_full(obs, seat, prev_mid=prev_mid)
    n = base.shape[-1]
    out = np.zeros((N_PAIR_TD, n, n), dtype="float32")
    out[:N_FULL] = base
    out[N_FULL:] = temporal_opp(hist or [], seat, n,
                                planes=HIST_PLANES, lags=LAGS_DEEP)
    return out
