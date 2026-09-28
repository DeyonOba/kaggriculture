"""REACTIVE strategy: theta as a function of the observed state.

A constant vector is closed. `tools/cem_theta.py` searched all 35 of the
champion's own constants over 98 candidates and ~15,000 games and found
nothing: best +0.028 +/- 0.049 on the search draw, +0.000 +/- 0.047 on a fresh
one. The champion is a sharp local optimum of its own parameters.

What a constant vector CANNOT express is the thing the leaders visibly do --
change plan on the market draw. The town unlocks one shop from eight known
types on a known schedule, so which products have demand is the single largest
thing that differs between two games, and it is fully observable by day 6.

    theta(s) = theta_base, then at each decision day, for every RULE that
               fires on features.context(s):  knob := value

This is the shape `KnobBanditPolicy` already has, and the machinery is proven
to reach the plan. The difference is where the rules come from: the shipped
four were FITTED from counterfactual-bandit gain models and never gated
individually. Here they are SEARCHED and each one is gated on the games it
actually fires in.

WHY THE FIRING SET IS THE WHOLE PROBLEM. A rule that fires in 30% of games and
is worth +10 points there reads as +3 points over the full draw, and the
instrument's standard error is 4.5. Averaging a rule over games where it never
fired is a procedure for producing nulls -- which is a fair description of
several entries in this project's record. So a rule is scored on its OWN games,
paired, with the partition decided by the champion's run (tools/context_census)
so that "it fired" cannot depend on the rule's own outcome.
"""
from __future__ import annotations

from kagent import planner

# Fields of `features.context` a rule may test. Scalars are compared directly;
# mappings are compared at one key. Nothing here is a judgement about which
# condition is useful -- the search decides that. The opponent's fields are in
# because their farm is fully public and their policy is the one quantity in
# this game that genuinely has to be estimated rather than derived.
SCALAR_FIELDS = ("money", "land", "hands", "shed_total", "opp_money", "opp_land")
MAP_FIELDS = ("consumers", "price_ratio", "crops", "herd", "opp_crops", "opp_herd")

DECISION_DAYS = (6, 9, 12, 15)
OPS = (">=", "<=")


def value_of(ctx, field, key=None):
    """The number a rule compares against, or None if absent."""
    if field in SCALAR_FIELDS:
        v = ctx.get(field)
        return None if v is None else float(v)
    d = ctx.get(field)
    if not isinstance(d, dict):
        return None
    return float(d.get(key, 0))


def fires(rule, ctx):
    """Does this rule's condition hold on `ctx`?"""
    if int(rule["day"]) != int(ctx.get("day", -1)):
        return False
    v = value_of(ctx, rule["field"], rule.get("key"))
    if v is None:
        return False
    return v >= float(rule["thr"]) if rule["op"] == ">=" else v <= float(rule["thr"])


def firing_set(rule, rows):
    """Indices of census rows where `rule` fires. No games are played: the
    champion's context at each day is a property of the world."""
    out = []
    for i, row in enumerate(rows):
        ctx = (row.get("ctx") or {}).get(str(rule["day"]))
        if ctx is not None and fires(rule, ctx):
            out.append(i)
    return out


def describe(rule):
    key = f"[{rule['key']}]" if rule.get("key") else ""
    return (f"d{rule['day']} {rule['field']}{key} {rule['op']} {rule['thr']}"
            f" -> {rule['knob']}={rule['value']}")


class ReactivePlanProvider:
    """theta_base from day 0; rules applied at their own decision day.

    Rules are evaluated ONCE per decision day, against the context as it stands
    at hour 0 before the day's plan is built -- the same instant the census
    records, so a rule gated on the census fires on the same states in play.
    A knob set by a rule stays set: `memory.knobs` carries to season end, which
    is what makes these season-long decisions rather than per-day noise.
    """

    def __init__(self, knobs, rules=None):
        self.base = dict(knobs or {})
        self.rules = list(rules or [])
        self.fired = []

    def make_plan(self, view, memory):
        from kagent import features

        if getattr(memory, "knobs", None) is None:
            memory.knobs = dict(self.base)
        day = view.day
        if self.rules and day in {int(r["day"]) for r in self.rules}:
            seen = getattr(memory, "reactive_days", None)
            if not isinstance(seen, set):
                seen = set()
                memory.reactive_days = seen
            if day not in seen:
                seen.add(day)
                ctx = features.context(view)
                for r in self.rules:
                    if fires(r, ctx):
                        memory.knobs[r["knob"]] = r["value"]
                        self.fired.append((day, describe(r)))
        return planner.make_day_plan(view, memory, speculative=True)


# ------------------------------------------------------- when a knob is live
# A knob set after its own phase has passed changes nothing, and a rule that
# changes nothing costs a full gate to learn that. These windows are read off
# planner.make_champion_plan's own day gates -- they are facts about when the
# plan consults each knob, not opinions about when it should.
#
# Measured on a six-rule smoke: THREE of six candidates came back bit-identical
# to the champion (delta +0.000 +/- 0.000, money +0), e.g. `sheep=0` at day 15,
# where the sheep ramp min(cap, 2+(day-6)) has already reached the cap at day
# 12 and the buy target is max(0, target - herd). Half the budget, spent on
# nothing.
#
# The herd caps carry a second, tighter bound for DECREASES: lowering a cap
# below the herd you already own is inert, so the window ends when the ramp
# tops out (cows at day 8, sheep at day 12) rather than when buying stops.
def _alive_windows():
    from kagent import planner as P
    return {
        "melon_opening":   (0, 1),
        "straw_early":     (P.CHAMP_LAND_DAY, 9),
        "straw_target":    (P.CHAMP_LAND_DAY, 13),
        "carrot_standing": (6, P.CARROT_LAST_PLANT_DAY),
        "tomato_standing": (6, P.TOMATO_LAST_PLANT_DAY),
        "wheat_engine":    (2, P.WHEAT_LAST_PLANT_DAY),
        "carrot_day":      (0, P.CARROT_LAST_PLANT_DAY),
        "carrot_cap":      (0, P.CARROT_LAST_PLANT_DAY),
        "cows":            (0, 8),      # ramp tops out at day 8
        "sheep":           (0, 12),     # ramp tops out at day 12
        "geese":           (0, 18),
        "geese_day":       (0, 18),
        "feed_days":       (0, 29),
        "hires_delta":     (0, 29),
        "land2_margin":    (0, 8),
        "land4":           (0, P.LAND4_DAY_MAX),
        "premium_scale":   (0, 29),
        "wheat_floor":     (0, P.FLOOR_DECAY_DAY),
        "fert_gate":       (0, 29),
        "liquidation_day": (0, 29),
    }


ALIVE = None


def knob_alive(knob, day):
    """Can setting `knob` on `day` still change the plan?"""
    global ALIVE
    if ALIVE is None:
        ALIVE = _alive_windows()
    lo, hi = ALIVE.get(knob, (0, 29))
    return lo <= day <= hi


# ------------------------------------------------------- the rule space
def thresholds_from(rows, day, field, key=None, max_cuts=3):
    """Candidate thresholds taken from the DATA, not chosen.

    A condition is only worth playing if it actually splits the draw, so the
    cuts are the observed quantiles of that field on that day. Integer-valued
    fields (shop consumer counts) use their distinct values instead, because a
    quantile between 1 and 2 consumers is a threshold no world can sit on.
    """
    vals = []
    for row in rows:
        ctx = (row.get("ctx") or {}).get(str(day))
        if ctx is None:
            continue
        v = value_of(ctx, field, key)
        if v is not None:
            vals.append(v)
    if not vals:
        return []
    vals.sort()
    distinct = sorted(set(vals))
    if len(distinct) <= 1:
        return []                       # constant across the draw: untestable
    if all(float(v).is_integer() for v in distinct) and len(distinct) <= 6:
        return [float(v) for v in distinct[1:]]
    return [vals[int(len(vals) * q)] for q in
            [(i + 1) / (max_cuts + 1) for i in range(max_cuts)]]


def enumerate_rules(rows, knob_values, min_fire=0.15, max_fire=0.85,
                    days=DECISION_DAYS, fields=None):
    """Every (condition, knob, value) whose condition splits the draw usefully.

    `min_fire`/`max_fire` are the testability filter and the reason this
    function exists: a rule firing in 5% of games cannot be resolved, and one
    firing in 98% is a constant vector wearing a condition -- and constants are
    closed. Both are dropped before a single game is played.
    """
    keys = []
    for f in (fields or MAP_FIELDS):
        seen = set()
        for row in rows:
            for ctx in (row.get("ctx") or {}).values():
                d = ctx.get(f)
                if isinstance(d, dict):
                    seen.update(d)
        keys += [(f, k) for k in sorted(seen)]
    keys += [(f, None) for f in (SCALAR_FIELDS if fields is None else ())]

    n = len(rows)
    out = []
    for day in days:
        for field, key in keys:
            for thr in thresholds_from(rows, day, field, key):
                for op in OPS:
                    probe = {"day": day, "field": field, "key": key, "op": op,
                             "thr": thr, "knob": "_", "value": 0}
                    share = len(firing_set(probe, rows)) / max(1, n)
                    if not (min_fire <= share <= max_fire):
                        continue
                    for knob, values in knob_values.items():
                        if not knob_alive(knob, day):
                            continue        # the plan no longer reads it
                        for v in values:
                            out.append({**probe, "knob": knob, "value": v,
                                        "fire_share": round(share, 3)})
    return out
