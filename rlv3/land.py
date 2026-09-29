"""The land decision: learned, self-correcting, and ONE-WAY SAFE.

Land cannot be sold back, so the failure mode of a badly fitted score has to
be one-way too. The structure here is therefore: HARD GATES FIRST, learned
score LAST. A theta can only ever VETO a purchase the gates already allowed;
it can never authorise one they refused. The worst a wrong theta does is buy
LESS land.

WHY THE GATES ARE MEASUREMENTS, NOT PROJECTIONS. The fourth quadrant gated
-$6,266 and -$6,017 on two independent blocks despite a monotone 21,054-game
correlation favouring it (3+ days earlier to the third quadrant wins 77.7%),
because hands-per-work sat at 0.47-0.58 on days 11-15 and $4,000 bought idle
ground. Labour GATES land. So the gate is on the ratio we have MEASURED on
ground we already own -- a 3-day rolling occupied share, and hands against
pending work -- not on a projection of the ground about to be owned. That is
also the self-correction: an over-bought farm reads its own idleness the next
dawn and stops.

theta_0 (rlv3/params/allocation_v0.json) reproduces champ-h10 exactly:
max_quadrants 3, day_min 5, day_max 22, no capital gate, floors at 0, w = 0,
b = +1 -- "always buy once the gates pass", which IS the champion's standing
wish from day 5. `rlv3/tests/test_allocation.py::test_theta0_matches_champion`
asserts that over a sweep of days, cash levels and quadrant counts.

"Learned" means theta is moved by TOMORROW's training loop and written back to
a params JSON; the JSON IS the checkpoint, so training resumes from the last
file and a compute limit costs nothing. It is never fitted on replay outcomes
or held-out accuracy: the project's recurring law is that BC (69.4%), the
market demand table and the value head (4.8x tighter, wrong sign 3 of 4) all
fit the data and failed to predict the effect of a change. theta moves only on
a PAIRED GATE.

THE THIRD PURCHASE PATH. Land is bought through three paths in this repo --
the planner head, the crunch fallback, and `planner.apply_land_pressure`.
Stage B5 ran three hours with only two of them gated and the capped control
still held 3.8 quadrants. `decide` is the ONLY path rlv3 uses: the champion
plan never calls `apply_land_pressure` (only `features.apply_to_plan` does,
on the neural path), and `rlv3.allocation.apply_to_plan` overwrites
`plan.land_buys` wholesale, so the leak cannot re-open here.
"""

import json
import os

from kagent.constants import LAND_ORDER, LAND_PRICES, TURNS_PER_DAY
from kagent import features

# Feature order is part of the checkpoint contract: theta["w"] is indexed by
# it, so append at the END and never reorder.
FEATURE_NAMES = (
    "day_frac",            # (day - day_min) / (day_max - day_min), clipped
    "quad_idx",            # index of the quadrant being considered (0,1,2)
    "cash_headroom",       # (money - price) / price, clipped to [-1, 4]
    "util3",               # 3-day rolling occupied share of unlocked tiles
    "hands_per_work",      # measured hands / pending work, clipped to [0, 3]
    "idle_share",          # yesterday's PASS share of unit-turns
    "crop_capacity_frac",  # today's plantable tiles / unlocked tiles
    "lead_norm",           # projected money lead / 20k, clipped to [-2, 2]
    "opp_quads",           # quadrants the opponent owns / 4
)


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class LandDecider:
    """Hard gates, then a monotone linear score. Checkpoints as JSON."""

    def __init__(self, theta):
        self.theta = dict(theta or {})
        self.theta.setdefault("version", 0)
        self.theta.setdefault("w", [0.0] * len(FEATURE_NAMES))
        self.theta.setdefault("b", 1.0)
        self.theta.setdefault("day_min", 5)
        self.theta.setdefault("day_max", 22)
        self.theta.setdefault("max_quadrants", 3)
        self.theta.setdefault("util_floor", 0.0)
        self.theta.setdefault("hands_per_work_floor", 0.0)
        self.theta.setdefault("capital", None)
        self.theta.setdefault("lam", 0.0)
        self.theta.setdefault("margin", {})

    # -- checkpoint in / out (the JSON IS the checkpoint) --------------------

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        return cls(data.get("land", data))

    def save(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.theta, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # -- features ------------------------------------------------------------

    def features(self, view, memory, alloc):
        t = self.theta
        day = view.day
        span = max(1, int(t["day_max"]) - int(t["day_min"]))
        idx = next_index(view)
        price = LAND_PRICES[min(idx, len(LAND_PRICES) - 1)]
        unlocked = max(1, 25 * len(view.me.unlocked_quadrants))
        util = measured_utilisation(view, memory)
        hpw = measured_hands_per_work(view, memory, alloc)
        cap = getattr(alloc, "crop_capacity", 0) or 0
        try:
            lead = float((getattr(alloc, "trace", None) or {}).get("lead", 0.0))
        except Exception:
            lead = 0.0
        return [
            _clip((day - int(t["day_min"])) / float(span), 0.0, 1.0),
            float(idx),
            _clip((view.me.money - price) / float(max(1, price)), -1.0, 4.0),
            _clip(util, 0.0, 1.0),
            _clip(hpw, 0.0, 3.0),
            _clip(float(getattr(memory, "yesterday_idle", 0.0) or 0.0), 0.0, 1.0),
            _clip(cap / float(unlocked), 0.0, 1.0),
            _clip(lead / 20000.0, -2.0, 2.0),
            _clip(len(view.opp.unlocked_quadrants) / 4.0, 0.0, 1.0),
        ]

    def score(self, view, memory, alloc):
        """w . x + b. Positive authorises the purchase the gates allowed."""
        x = self.features(view, memory, alloc)
        w = self.theta.get("w") or []
        total = float(self.theta.get("b", 0.0))
        for i in range(min(len(w), len(x))):
            total += float(w[i]) * x[i]
        return total

    # -- the decision --------------------------------------------------------

    def decide(self, view, memory, alloc):
        """-> (buy, margin, why). Gates run FIRST; the score can only veto."""
        t = self.theta
        why = {"gate": None, "score": None}
        idx = next_index(view)
        why["quad_idx"] = idx
        if idx >= len(LAND_ORDER):
            why["gate"] = "no_quadrants_left"
            return False, 0, why
        if len(view.me.unlocked_quadrants) >= int(t["max_quadrants"]):
            why["gate"] = "max_quadrants"
            return False, 0, why
        if not features.land_allowed(view):
            # KAGG_MAX_QUADRANTS, the env cap the capped control runs under.
            why["gate"] = "env_cap"
            return False, 0, why
        if getattr(alloc, "trace", {}).get("liquidation"):
            why["gate"] = "liquidation"
            return False, 0, why
        if not (int(t["day_min"]) <= view.day <= int(t["day_max"])):
            why["gate"] = "day_window"
            return False, 0, why
        capital = t.get("capital")
        price = LAND_PRICES[idx]
        if capital is not None and view.me.money < price + float(capital):
            why["gate"] = "capital"
            return False, 0, why
        util = measured_utilisation(view, memory)
        why["util3"] = round(util, 4)
        if util < float(t["util_floor"]):
            why["gate"] = "util_floor"
            return False, 0, why
        hpw = measured_hands_per_work(view, memory, alloc)
        why["hands_per_work"] = round(hpw, 4)
        if hpw < float(t["hands_per_work_floor"]):
            why["gate"] = "hands_per_work_floor"
            return False, 0, why
        s = self.score(view, memory, alloc)
        why["score"] = round(s, 4)
        if s < 0.0:
            why["gate"] = "score_veto"
            return False, 0, why
        why["gate"] = "pass"
        margins = t.get("margin") or {}
        margin = margins.get(str(idx), margins.get(idx))
        return True, (int(margin) if margin is not None else None), why


# --- the measurements the gates are made of ---------------------------------

def next_index(view):
    """Index into LAND_ORDER / LAND_PRICES of the next quadrant to buy."""
    owned = view.me.unlocked_quadrants
    return sum(1 for q in LAND_ORDER if q in owned)


def observe_dawn(memory, view):
    """Roll the rlv3 measurement ledger. MEASUREMENTS ONLY -- never a target.

    That is the whole self-correction: nothing an allocator decided yesterday
    is carried forward, so an over-ordered plan corrects itself the next dawn
    simply because the board it reads already contains the mistake.
    """
    st = getattr(memory, "rlv3", None)
    if not isinstance(st, dict) or st.get("day", 0) > view.day:
        st = {"day": -1, "occupied": [], "chore": {}}
        memory.rlv3 = st
    if st["day"] != view.day:
        st["day"] = view.day
        st["occupied"] = (st["occupied"] + [occupied_share(view)])[-3:]
    return st


def occupied_share(view):
    """Share of our unlocked tiles carrying something right now."""
    unlocked = 25 * max(1, len(view.me.unlocked_quadrants))
    bare = sum(1 for _ in view.me.empty_tiles()) + sum(1 for _ in view.me.weeds())
    return _clip((unlocked - bare) / float(unlocked), 0.0, 1.0)


def measured_utilisation(view, memory):
    """3-day rolling occupied share of the ground we ALREADY own."""
    st = getattr(memory, "rlv3", None)
    hist = (st or {}).get("occupied") or []
    if not hist:
        return occupied_share(view)
    return sum(hist) / float(len(hist))


def measured_hands_per_work(view, memory, alloc=None):
    """Hands against pending work -- the ratio the fourth quadrant lost on.

    Work is what the farm can actually be asked to do today: the plantings the
    allocation wants, the harvests that are ripe, and two chores per animal.
    Hands come from `memory.yesterday_hands` (the measured peak the ledger in
    `kagent.policy.HeuristicPolicy.act` already rolls), falling back to the
    crew standing on the board.
    """
    hands = int(getattr(memory, "yesterday_hands", 0) or 0)
    if hands <= 0:
        hands = max(0, len(view.me.units) - 1)
    work = 0
    if alloc is not None:
        try:
            work += sum(max(0, int(v)) for v in alloc.plant_quota.values())
        except Exception:
            pass
    day, step = view.day, view.step
    from kagent.constants import CROPS
    for _, _, t in view.me.plants():
        cd = CROPS.get(t.get("crop"))
        if cd is None:
            continue
        age = day - t.get("planted_day", day)
        if not cd["ongoing"]:
            if age >= cd["max_yield_day"]:
                work += 1
        elif age >= cd["first_yield_day"] or 0 <= t.get("max_lifespan_step", -1) <= step + TURNS_PER_DAY:
            work += 1
    work += 2 * sum(1 for _ in view.me.animals())
    return hands / float(max(1, work))
