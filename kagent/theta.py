"""THETA: the champion's strategy as a searchable parameter vector.

The planner is not an emergent thing. Its behaviour is a table of constants --
how many strawberry tiles to stand up, which day the endgame carrots start,
how many head of each animal, which executor rules run -- and `memory.knobs`
already overrides every one of them (`planner.KNOB_DEFAULTS`). Those constants
ARE the strategy, so the strategy is a point in R^n and finding a better one is
a search, not an argument.

WHY A SEARCH AND NOT ANOTHER LEVER. The project record is eleven single-lever
arms, of which two won. The losses share a shape: a lever is moved, it pays its
cost, and the thing that would have made it pay sits at zero in another
coordinate. `geese` is the clean case -- it was refuted at +4 and +6 head while
`animal_visit` was off, so every goose produced one egg a day into a harvest
rule that only collects at two, and while the coop it needs competed for ground
against a strawberry target tuned for a farm with no coop on it. Those three
coordinates are one decision and cannot be tested one at a time.

WHAT IS IN AND WHAT IS OUT. In: a constant the planner or scheduler already
reads, that names a STRUCTURAL choice -- how much of something, on which day,
with which rule running. Out: anything fitted from prices (the sell floors
themselves, the drain forecast, the rent model), because those are derived from
an exact identity and searching them would be refitting a solved problem with a
worse instrument; and anything the engine fixes (seed costs, yield tables,
shop draw schedule). `premium_scale` and `fert_gate` are in despite touching
price, because they are SCALES ON a derived quantity -- a policy about how
hard to hold out, not an estimate of anything.

BOUNDS come from the ground truth where it speaks: the leaders' observed counts
(memory: head grids are mis-spaced -- 2-3 geese, 1-4 carrots, 4-8 melon, 9-10
hands) widened so the champion's own value and the leaders' are both interior,
never at an edge. A coordinate pinned at its bound is a bound that was wrong.
"""
from __future__ import annotations

import math

# name, lo, hi, kind. "int" rounds, "bool" thresholds at 0.5, "float" passes.
KNOB_SPEC = [
    # --- the opening -------------------------------------------------------
    ("melon_opening",   0,    16,    "int"),    # champ 12; leaders 4-8
    # --- the day 5-13 block ------------------------------------------------
    ("straw_early",     0,    40,    "int"),    # champ 18
    ("straw_target",    0,    44,    "int"),    # champ 33; leaders 13-24
    # --- the herd ----------------------------------------------------------
    ("cows",            0,    12,    "int"),    # champ 8
    ("sheep",           0,    12,    "int"),    # champ 8
    ("geese",           0,    12,    "int"),    # champ 0; leaders 5-9
    ("geese_day",       0,    14,    "int"),    # champ 6
    ("feed_days",       1,     4,    "int"),    # champ 2
    # --- the crop engine ---------------------------------------------------
    ("wheat_engine",    0,    60,    "int"),    # champ 32
    ("carrot_standing", 0,    24,    "int"),    # champ 0; leaders 1-18
    ("tomato_standing", 0,    24,    "int"),    # champ 0
    ("carrot_day",      6,    29,    "int"),    # champ 25
    ("carrot_cap",      0,    30,    "int"),    # champ 20
    # --- labour and land ---------------------------------------------------
    ("hires_delta",    -3,     4,    "int"),    # champ 0; leaders 9-10 hands
    ("land2_margin",    0,  4000,    "int"),    # champ 100
    ("land4",           0,     1,    "bool"),   # champ 0
    # --- holding out -------------------------------------------------------
    ("premium_scale", 0.05,  1.20,   "float"),  # champ 0.35
    ("wheat_floor",     1,    60,    "int"),    # champ 1 (= sell all)
    ("fert_gate",     0.25,  3.00,   "float"),  # champ 1.0
    ("liquidation_day", 18,   29,    "int"),    # champ 27
]

# Executor rules are coordinates too. The shipped four are ON at the champion's
# point, so the search starts where the agent stands and can switch any of them
# off; the rest are off and can come on. `tilequeue` is excluded -- it is a
# measured -24.3 and carries a whole allocation mechanism, not a rule.
EXEC_SPEC = [
    ("lastwater",     "bool"),   # shipped ON
    ("h23",           "bool"),   # shipped ON
    ("harvest_visit", "bool"),   # shipped ON
    ("shedguard",     "bool"),   # shipped ON
    ("animal_visit",  "bool"),   # OFF -- the gate that makes a goose half-useless
    ("chain",         "bool"),
    ("weeds",         "bool"),
    ("evesell",       "bool"),
    ("spent",         "bool"),
    ("handfeed",      "bool"),
    ("oneshed",       "bool"),
    ("hv_underfoot",  "bool"),
    ("nodrift",       "bool"),
    ("dailywater",    "bool"),
    ("endwater",      "bool"),
]

EXEC_NAMES = [n for n, _ in EXEC_SPEC]
KNOB_NAMES = [n for n, _, _, _ in KNOB_SPEC]
NAMES = KNOB_NAMES + EXEC_NAMES
DIM = len(NAMES)


def champion():
    """The shipped agent's own point: planner defaults + the shipped exec rules.

    Read from the live modules rather than copied, so the search starts where
    the agent actually stands and a change to the champion is not silently
    left behind in a duplicate table.
    """
    from kagent import planner
    d = {n: planner.KNOB_DEFAULTS[n] for n in KNOB_NAMES}
    try:
        from kagent.weights import EXEC_RULES
    except Exception:
        EXEC_RULES = []
    on = set(EXEC_RULES or [])
    d.update({n: (n in on) for n in EXEC_NAMES})
    return d


# ------------------------------------------------------------ the unit cube
# CEM runs in [0,1]^n so one sigma means the same thing in every coordinate
# and no bound needs a hand-picked step. The cast back to int/bool happens on
# the way out, which is also where two different vectors can land on the same
# strategy -- harmless, and cheaper than a mixed-integer search.
def _bounds(name):
    for n, lo, hi, kind in KNOB_SPEC:
        if n == name:
            return float(lo), float(hi), kind
    return 0.0, 1.0, "bool"


# Quantisation for the float coordinates. Without it a perturbation can be a
# 2% move sitting inside a 4.5% standard error -- a candidate spent, 2.3
# minutes of games, and nothing learned. Measured: premium_scale 0.35 -> 0.3431
# scored +0.000 +/- 0.000, bit-identical to the champion on all 144 games.
FLOAT_STEP = {"premium_scale": 0.05, "fert_gate": 0.10}


def _quantise(name, v):
    st = FLOAT_STEP.get(name)
    return round(round(v / st) * st, 4) if st else round(float(v), 4)


def to_unit(theta):
    """theta dict -> [0,1]^n."""
    u = []
    for name in NAMES:
        lo, hi, kind = _bounds(name)
        v = theta[name]
        v = float(bool(v)) if kind == "bool" else float(v)
        u.append(0.0 if hi == lo else min(1.0, max(0.0, (v - lo) / (hi - lo))))
    return u


def from_unit(u):
    """[0,1]^n -> theta dict, rounded onto the coordinate's own type."""
    out = {}
    for name, x in zip(NAMES, u):
        lo, hi, kind = _bounds(name)
        v = lo + min(1.0, max(0.0, float(x))) * (hi - lo)
        if kind == "int":
            v = int(round(v))
        elif kind == "bool":
            v = bool(v >= 0.5)
        else:
            v = _quantise(name, v)
        out[name] = v
    return out


KINDS = [k for _n, _lo, _hi, k in KNOB_SPEC] + [k for _n, k in EXEC_SPEC]
BOOL_AT = [i for i, k in enumerate(KINDS) if k == "bool"]


def sample(rng, mu, sigma):
    """One candidate: Gaussian on the continuous coordinates, BERNOULLI on the
    binary ones.

    A binary coordinate cannot be searched through a Gaussian. `to_unit` puts
    an off flag at 0.0 and `from_unit` thresholds at 0.5, so flipping it needs
    a draw more than two sigma out: at sigma 0.22 that is a 1.2% event, and at
    the sigma CEM contracts to it is never. `animal_visit` -- the harvest rule
    that decides whether a goose's daily egg is collected at all -- would have
    sat off for the whole run while the search believed it was exploring it.

    So the binary coordinates carry a PROBABILITY, not a mean with a spread,
    which is how the cross-entropy method handles combinatorial coordinates:
    sample Bernoulli(p), refit p to the elite frequency. Both live in the same
    [0,1] vector, and `from_unit` reads either.
    """
    out = []
    for i, (m, s) in enumerate(zip(mu, sigma)):
        if KINDS[i] == "bool":
            out.append(1.0 if rng.random() < m else 0.0)
        else:
            out.append(min(1.0, max(0.0, rng.gauss(m, s))))
    return out


# --------------------------------------------------- coupled by MECHANISM
# Which coordinates feed the same plan field or the same chain in the engine.
# This is read off the code, not off a belief about which values are good: a
# goose's egg is PRODUCED by `geese` on a coop built from `geese`, and
# COLLECTED by `animal_visit` (without it the scheduler waits for held >= 2 and
# a goose makes one a day), so the three are one decision and a sampler that
# cannot select them together cannot express it.
#
# MEASURED, which is why this exists: under uniform subset sampling `geese` and
# `animal_visit` are each selected in about 7% of candidates, so both land in
# 0.5% -- half a candidate in a ninety-six candidate run. Accumulation does not
# rescue it either, because each half LOSES alone (geese without collection
# -0.083, rl/runs/cem_theta_uniform) and so never enters the mean for the other
# to build on. That is the same "unreachable by construction" shape the project
# has hit four times.
GROUPS = {
    "goose":     ["geese", "geese_day", "animal_visit"],
    "herd":      ["cows", "sheep", "feed_days", "animal_visit", "handfeed"],
    "ground":    ["melon_opening", "straw_early", "straw_target", "wheat_engine",
                  "carrot_standing", "tomato_standing", "carrot_day", "carrot_cap"],
    "labour":    ["hires_delta", "land2_margin", "land4"],
    "holding":   ["premium_scale", "wheat_floor", "fert_gate", "liquidation_day"],
    "harvest":   ["lastwater", "h23", "harvest_visit", "animal_visit",
                  "dailywater", "endwater"],
    "logistics": ["shedguard", "evesell", "oneshed", "spent", "hv_underfoot",
                  "nodrift", "chain", "weeds"],
}
GROUP_IDX = {g: [NAMES.index(n) for n in ns] for g, ns in GROUPS.items()}

# A coordinate that cannot change anything while another sits at zero. Also
# mechanism: `geese_day` gates a target that is 0, so moving it is a no-op --
# measured at +0.000 +/- 0.000 across 144 games.
DEPENDS = {"geese_day": "geese"}


def _inert(i, mu):
    need = DEPENDS.get(NAMES[i])
    if need is None:
        return False
    j = NAMES.index(need)
    return _decode_at(j, mu[j]) in (0, False)


def sample_block(rng, mu, sigma, block=4, group_p=0.6):
    """One candidate that perturbs a random SUBSET of the coordinates.

    MEASURED, not assumed. The first generation of a dense search moved 15-22
    of the 35 coordinates in every candidate and ALL ELEVEN LOST, from -0.097
    to -0.410 paired win rate, monotone in how far the candidate had travelled
    (rl/runs/cem_theta_dense). That is the narrow basin the project already had
    on record -- a plan perturbed at sigma 0.25 costs $18,944 -- and it means a
    dense sampler spends its whole budget discovering that the champion is a
    local optimum, which is already known.

    Perturbing a few coordinates keeps the candidate INSIDE the basin while
    still searching the joint: a subset of size three can hold {geese,
    geese_day, animal_visit} together, which is the combination every previous
    single-lever goose arm could not express. Joints larger than the block size
    are reached by ACCUMULATION -- a good coordinate enters the mean through
    the elites, and later candidates perturb on top of it.

    A binary coordinate that is selected is FLIPPED outright rather than
    redrawn: selection is the exploration, so the flip should be certain when
    it happens.
    """
    if rng.random() < group_p:
        # A MECHANISM at a time: pick one group and move two or more of it
        # together, so a combination whose halves each lose can still be tried.
        pool = GROUP_IDX[rng.choice(list(GROUP_IDX))]
        pool = [i for i in pool if not _inert(i, mu)] or pool
        k = rng.randint(min(2, len(pool)), min(int(block), len(pool)))
    else:
        pool = [i for i in range(len(mu)) if not _inert(i, mu)]
        k = rng.randint(1, max(1, int(block)))
    out = list(mu)
    for i in rng.sample(pool, min(k, len(pool))):
        if KINDS[i] == "bool":
            out[i] = 0.0 if mu[i] >= 0.5 else 1.0
        else:
            out[i] = _perturb(rng, i, mu[i], sigma[i])
    return out


def _decode_at(i, u):
    """The value coordinate i takes at unit position u."""
    lo, hi, kind = _bounds(NAMES[i])
    v = lo + min(1.0, max(0.0, u)) * (hi - lo)
    return int(round(v)) if kind == "int" else (v >= 0.5 if kind == "bool"
                                                else _quantise(NAMES[i], v))


def _perturb(rng, i, m, s):
    """A draw that actually CHANGES coordinate i.

    Selecting a coordinate is the decision to probe it, so a draw that decodes
    back to the value it already had wastes the probe -- and the waste is not
    spread evenly. A coordinate sitting on a bound loses half its draws to the
    clip: `geese` starts at 0, so every negative draw rounds back to 0, and it
    was probed in 16 of 600 candidates where `animal_visit` got 45. The
    coordinates at a floor are exactly the ones that have never been tried.
    """
    lo, hi, _kind = _bounds(NAMES[i])
    cur = _decode_at(i, m)
    for _ in range(8):
        u = min(1.0, max(0.0, rng.gauss(m, s)))
        if _decode_at(i, u) != cur:
            return u
    # Cornered: step one quantum off the bound, into the interior.
    step = 1.0 / (hi - lo) if hi > lo else 0.0
    return min(1.0, m + step) if m <= 0.5 else max(0.0, m - step)


def spec(theta):
    """theta dict -> the rollout arm spec, {"kind","knobs","exec"}.

    The exec flags go on `spec["exec"]`, which `rl/rollout.run_episode` already
    applies per episode and restores afterwards, so a worker playing several
    candidates in a row does not leak one arm's executor into the next.
    """
    knobs = {n: theta[n] for n in KNOB_NAMES}
    flags = sorted(n for n in EXEC_NAMES if theta.get(n))
    return {"kind": "theta", "knobs": knobs, "exec": flags}


def describe(theta, ref=None):
    """One line per coordinate that DIFFERS from `ref` (default: champion)."""
    ref = champion() if ref is None else ref
    rows = []
    for n in NAMES:
        a, b = ref.get(n), theta.get(n)
        if a != b:
            rows.append(f"{n}={b}" + (f" (was {a})" if a is not None else ""))
    return ", ".join(rows) or "= champion"


def distance(a, b):
    """Euclidean distance in the cube -- how far a point has travelled."""
    ua, ub = to_unit(a), to_unit(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ua, ub)))
