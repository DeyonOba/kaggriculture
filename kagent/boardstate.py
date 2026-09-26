"""Board tensor for a trunk-carrying planner, built INSIDE the bundle.

Why this module exists: `rl/common.to_weights_data` used to refuse board nets
outright, because the b85 payload carried only blocks+out and the bundle had
no way to build the (C,10,10) tensor the trunk eats. So every board-trunk
planner we ever trained was unsubmittable. This is the missing half; the
other half is the trunk serialisation in `to_weights_data`/`decode_weights`.

It is a PORT of `tools/board_tensor.py` for the one arm the trainer uses,
`T_crop_103`, and it has to stay bit-identical to it: the reference is what
the weights were trained against, so any drift is a silent train/serve skew
of exactly the kind that cost us every Stage B board arm.
`tests/test_board_export.py::test_builder_matches_the_trainer` pins the two
together on real observations and is the guard against that drift.

Layout, identical to `board_pair_temporal_both` (123 channels), of which the
arm selects 103:

    [0  : 24)  our farm, base 24 channels
    [24 : 25)  our hands
    [25 : 49)  their farm
    [49 : 50)  their hands
    [50 : 54)  our yesterday-midday planes (watered, fed, cared, hands)
    [54 : 58)  their yesterday-midday planes
    [58 : 82)  globals
    [82 : 91)  our shed
    [91 :103)  opponent plateau planes x lags (NOT in T_crop_103, left zero)
    [103:123)  opponent crop one-hot x lags (1, 3, 5, 10)

Every module-level name is `_BS_`/`_bs_` prefixed: the bundler execs all
modules into ONE shared namespace, so an unprefixed `FIRST_YIELD` here would
silently overwrite the planner's.
"""

from kagent.constants import MARKET_PARAMS

_BS_NP = None


def _bs_np():
    """numpy, imported on first use.

    Deliberately NOT a module-level import: the bundler execs every module at
    load, so an unconditional `import numpy` would make the whole agent
    un-importable anywhere numpy is missing, even a board-free bundle that
    never touches it. `nnpolicy.forward_board` imports it the same way.
    """
    global _BS_NP
    if _BS_NP is None:
        import numpy
        _BS_NP = numpy
    return _BS_NP

# Encoder v2 is what every current net was trained with (tools/board_tensor.py
# defaults KAGG_ENC to 2); pinned here rather than read from the environment,
# because a submitted agent must not depend on one.
_BS_N_CH = 24
_BS_CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
_BS_ANIMALS = ["GOOSE", "COW", "SHEEP"]
_BS_PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER"]
_BS_SHED = [(4, 4), (4, 5), (5, 4), (5, 5)]
_BS_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10,
                   "MELON": 10}

_BS_DEAD_IN_FARM = (7, 13, 14)     # watered_today, fed_today, cared_today
_BS_DEAD_IN_GLOBAL = (1, 3)        # hour, hands count
_BS_MID_PLANES = 4

_BS_F_OURS = (0, 24)
_BS_F_OUR_HANDS = (24, 25)
_BS_F_THEIRS = (25, 49)
_BS_F_THEIR_HANDS = (49, 50)
_BS_F_OUR_MID = (50, 54)
_BS_F_THEIR_MID = (54, 58)
_BS_F_GLOBAL = (58, 82)
_BS_F_SHED = (82, 91)
_BS_N_FULL = 91

_BS_PLATEAU_PLANES = [3, 10, 11, 12]
_BS_LAGS = (1, 3, 10)
_BS_HIST_PLANES = [0, 1, 2, 3, 4]
_BS_LAGS_DEEP = (1, 3, 5, 10)
_BS_OFF_PLATEAU = _BS_N_FULL                                    # 91
_BS_OFF_CROP = _BS_N_FULL + len(_BS_PLATEAU_PLANES) * len(_BS_LAGS)   # 103
_BS_N_PAIR_TB = _BS_OFF_CROP + len(_BS_HIST_PLANES) * len(_BS_LAGS_DEEP)  # 123


def _bs_units(farm):
    f = farm.get("farmer") or (0, 0)
    return tuple(f), [tuple(h) for h in (farm.get("hands") or [])]


def _bs_board_tensor(obs, seat=0):
    """-> float32 (24, n, n) for one farm at one moment."""
    np = _bs_np()
    farm = obs["farms"][seat]
    tiles = farm["tiles"]
    n = len(tiles)
    day = obs["day"]
    t = np.zeros((_BS_N_CH, n, n), dtype="float32")

    for y in range(n):
        for x in range(n):
            t[23, y, x] = min(abs(x - a) + abs(y - b) for a, b in _BS_SHED) / 12.0
    for x, y in _BS_SHED:
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
                if crop in _BS_CROPS:
                    t[_BS_CROPS.index(crop), y, x] = 1.0
                age = day - c.get("planted_day", day)
                t[5, y, x] = min(age, 20) / 20.0
                t[6, y, x] = min(c.get("yield_units", 0), 6) / 6.0
                t[7, y, x] = 1.0 if c.get("watered_today") else 0.0
                t[8, y, x] = min(c.get("consecutive_unwatered", 0), 2) / 2.0
                t[9, y, x] = max(0, _BS_FIRST_YIELD.get(crop, 0) - age) / 12.0
            elif "animal" in c:
                a = c.get("animal")
                if a in _BS_ANIMALS:
                    t[10 + _BS_ANIMALS.index(a), y, x] = 1.0
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


def _bs_scalars(obs, seat=0):
    """Context a convolution cannot see: time, cash, prices, shop draw."""
    np = _bs_np()
    farm = obs["farms"][seat]
    px = (obs.get("market") or {}).get("prices") or {}
    inv = (obs.get("market") or {}).get("inventory") or {}
    shops = list((obs.get("town") or {}).get("unlocked_shops", []))
    v = [obs["day"] / 30.0, obs["hour"] / 24.0,
         min(float(farm.get("money", 0)), 60000) / 60000.0,
         min(len(farm.get("hands") or []), 14) / 14.0,
         min(len(farm.get("unlocked_quadrants") or ["NW"]), 4) / 4.0,
         min(len(shops), 8) / 8.0]
    for p in _BS_PRODUCTS:
        base = MARKET_PARAMS[p]["base"]
        v.append(min(float(px.get(p, base)) / base, 3.0) / 3.0)
        # signed, tanh at 1500 -- encoder v2 (the abs() of v1 handed the model
        # a glut and a shortage as the same number)
        v.append(float(np.tanh((float(inv.get(p, 10000)) - 10000.0) / 1500.0)))
    return np.asarray(v, dtype="float32")


def _bs_mid_planes(obs, seat):
    """-> (4, n, n) from a MID-DAY observation: the layer midnight wipes."""
    np = _bs_np()
    farm = obs["farms"][seat]
    tiles = farm["tiles"]
    n = len(tiles)
    out = np.zeros((_BS_MID_PLANES, n, n), dtype="float32")
    for y, row in enumerate(tiles):
        for x, c in enumerate(row):
            if not isinstance(c, dict):
                continue
            if c.get("kind") == "PLANT":
                out[0, y, x] = 1.0 if c.get("watered_today") else 0.0
            elif c.get("animal"):
                out[1, y, x] = 1.0 if c.get("fed_today") else 0.0
                out[2, y, x] = 1.0 if c.get("cared_today") else 0.0
    for hx, hy in _bs_units(farm)[1]:
        if 0 <= hy < n and 0 <= hx < n:
            out[3, hy, hx] = min(out[3, hy, hx] * 8 + 1, 8) / 8.0
    return out


def _bs_hands_plane(farm, n):
    np = _bs_np()
    out = np.zeros((1, n, n), dtype="float32")
    for hx, hy in _bs_units(farm)[1]:
        if 0 <= hy < n and 0 <= hx < n:
            out[0, hy, hx] = min(out[0, hy, hx] * 8 + 1, 8) / 8.0
    return out


def _bs_farm_idx(base, drop_dead):
    lo, hi = base
    return [lo + i for i in range(hi - lo)
            if not (drop_dead and i in _BS_DEAD_IN_FARM)]


def _bs_arm_channels(name="T_crop_103"):
    """Channel indices for one arm, identical to tools/board_tensor."""
    g_all = list(range(*_BS_F_GLOBAL))
    g_live = [i for k, i in enumerate(g_all) if k not in _BS_DEAD_IN_GLOBAL]
    ours_lv = _bs_farm_idx(_BS_F_OURS, True)
    theirs_lv = _bs_farm_idx(_BS_F_THEIRS, True)
    hands = list(range(*_BS_F_OUR_HANDS)) + list(range(*_BS_F_THEIR_HANDS))
    mid = list(range(*_BS_F_OUR_MID)) + list(range(*_BS_F_THEIR_MID))
    shed = list(range(*_BS_F_SHED))
    e_full = ours_lv + theirs_lv + g_live + hands + mid + shed      # 83
    if name == "E_full_83":
        return e_full
    if name == "T_crop_103":
        return e_full + list(range(_BS_OFF_CROP, _BS_N_PAIR_TB))
    if name == "T_temporal_95":
        return e_full + list(range(_BS_OFF_PLATEAU, _BS_OFF_CROP))
    raise ValueError("boardstate: unsupported board arm %r" % (name,))


def _bs_temporal_opp(out, base_ch, hist, seat, n, planes, lags):
    """Fill `out[base_ch:...]` with the opponent's `planes` at each lag.

    hist[0] is the most recent PAST dawn; a lag beyond what we hold repeats
    the oldest frame, so early game reads as "nothing has changed yet"
    instead of "the board was empty".
    """
    if not hist:
        return
    cache = {}
    for li, lag in enumerate(lags):
        idx = min(lag - 1, len(hist) - 1)
        b = cache.get(idx)
        if b is None:
            try:
                b = _bs_board_tensor(hist[idx], 1 - seat)
            except Exception:
                continue
            cache[idx] = b
        for pi, ch in enumerate(planes):
            out[base_ch + li * len(planes) + pi] = b[ch]


def board_arm_tensor(obs, seat=0, prev_mid=None, hist=None, arm="T_crop_103"):
    """-> float16 (len(arm), 10, 10), the tensor `nnpolicy.forward_board` eats.

    Mirrors `rl.rollout._board`: board_pair_temporal_both(...) sliced by
    arm_channels(arm), cast to float16. The plateau block (91..102) is left
    zero because T_crop_103 does not select it; ask for T_temporal_95 and it
    is filled.
    """
    np = _bs_np()
    n = len(obs["farms"][seat]["tiles"])
    out = np.zeros((_BS_N_PAIR_TB, n, n), dtype="float32")

    for base, hbase, s in ((_BS_F_OURS, _BS_F_OUR_HANDS, seat),
                           (_BS_F_THEIRS, _BS_F_THEIR_HANDS, 1 - seat)):
        b = _bs_board_tensor(obs, s)
        farm = obs["farms"][s]
        fx, fy = _bs_units(farm)[0]
        b[22] = 0.0                       # merged occupancy -> farmer only
        if 0 <= fy < n and 0 <= fx < n:
            b[22, fy, fx] = 1.0
        out[base[0]:base[1]] = b
        out[hbase[0]:hbase[1]] = _bs_hands_plane(farm, n)

    if prev_mid is not None:
        out[_BS_F_OUR_MID[0]:_BS_F_OUR_MID[1]] = _bs_mid_planes(prev_mid, seat)
        out[_BS_F_THEIR_MID[0]:_BS_F_THEIR_MID[1]] = _bs_mid_planes(prev_mid, 1 - seat)

    out[_BS_F_GLOBAL[0]:_BS_F_GLOBAL[1]] = _bs_scalars(obs, seat)[:, None, None]
    shed = (obs.get("private") or {}).get("shed") or {}
    out[_BS_F_SHED[0]:_BS_F_SHED[1]] = np.asarray(
        [min(float(shed.get(p, 0)), 100.0) / 100.0 for p in _BS_PRODUCTS],
        dtype="float32")[:, None, None]

    h = hist or []
    channels = _bs_arm_channels(arm)
    if arm == "T_temporal_95":
        _bs_temporal_opp(out, _BS_OFF_PLATEAU, h, seat, n,
                         _BS_PLATEAU_PLANES, _BS_LAGS)
    if arm == "T_crop_103":
        _bs_temporal_opp(out, _BS_OFF_CROP, h, seat, n,
                         _BS_HIST_PLANES, _BS_LAGS_DEEP)
    return out[channels].astype("float16")


def observe(memory, view, obs):
    """Record the dawn and midday frames the lag planes need. Every step."""
    if getattr(memory, "bs_dawn", None) is None:
        memory.bs_dawn, memory.bs_mid = {}, {}
    if view.hour == 0:
        memory.bs_dawn[view.day] = obs
    elif view.hour == 12:
        memory.bs_mid[view.day] = obs


def build(memory, view, obs, arm="T_crop_103"):
    """The tensor for THIS step, from the frames `observe` has banked."""
    dawn = getattr(memory, "bs_dawn", None) or {}
    mid = getattr(memory, "bs_mid", None) or {}
    past = [dawn[d] for d in range(view.day - 1, -1, -1) if d in dawn]
    return board_arm_tensor(obs, view.player, prev_mid=mid.get(view.day - 1),
                            hist=past, arm=arm)


_BS_ARM_BY_CHANNELS = {83: "E_full_83", 95: "T_temporal_95", 103: "T_crop_103"}


def arm_for_trunk(board):
    """Board arm implied by a decoded trunk's input-channel count.

    Weights do not record which arm they were trained on, so it is read back
    off `c1w` (shape (32, in_ch, 3, 3)). Building a tensor of the wrong width
    is the failure this prevents: `forward_policy`'s guard would catch a
    wrong FEATURE width, but a wrong BOARD width would be a silently
    different trunk input.
    """
    c1w = board["c1w"]
    in_ch = len(c1w[0])
    try:
        return _BS_ARM_BY_CHANNELS[in_ch]
    except KeyError:
        raise ValueError(
            "boardstate: trunk wants %d input channels; known arms are %s"
            % (in_ch, sorted(_BS_ARM_BY_CHANNELS)))
