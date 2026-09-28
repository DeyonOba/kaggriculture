"""A reactive rule must fire on exactly the states the census says it will.

The whole gate rests on one identity: the games where a rule is SCORED are the
games where it FIRES, and that set is computed from the champion's recorded
context without playing anything. If the two ever disagree -- the census says a
rule fires and in play it does not, or vice versa -- then every reactive
result is a comparison between two different partitions and means nothing.

The identity holds for a SINGLE rule because the arm plays bit-identically to
the champion up to that rule's own decision day, so the context it evaluates is
the context the census recorded. It does NOT automatically hold for a candidate
carrying two rules on different days: the first rule changes the board, so the
second rule's context is no longer the census's. `test_two_rules_break_the_
partition` pins that as a known limit rather than a lurking bug.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULE = {"day": 6, "field": "consumers", "key": "EGG", "op": ">=", "thr": 1.0,
        "knob": "geese", "value": 4}


def test_fires_only_on_its_own_day():
    from kagent import reactive as R

    ctx = {"day": 6, "consumers": {"EGG": 2}}
    assert R.fires(RULE, ctx)
    assert not R.fires(RULE, {**ctx, "day": 9}), "a rule must not fire off its day"


def test_both_operators_and_the_absent_key():
    from kagent import reactive as R

    ctx = {"day": 6, "consumers": {"EGG": 0}}
    assert not R.fires(RULE, ctx)
    assert R.fires({**RULE, "op": "<=", "thr": 0.0}, ctx)
    # a product no shop has yet is 0, not missing -- the rule must still resolve
    assert R.fires({**RULE, "op": "<=", "thr": 0.0}, {"day": 6, "consumers": {}})


def test_scalar_and_opponent_fields_resolve():
    from kagent import reactive as R

    ctx = {"day": 9, "opp_money": 12000, "opp_herd": {"COW": 6}}
    assert R.fires({"day": 9, "field": "opp_money", "key": None, "op": ">=",
                    "thr": 10000, "knob": "cows", "value": 4}, ctx)
    assert R.fires({"day": 9, "field": "opp_herd", "key": "COW", "op": ">=",
                    "thr": 5, "knob": "cows", "value": 4}, ctx)


def test_the_alive_window_matches_the_planners_own_gates():
    """These windows exist to stop the search buying bit-identical games. If a
    planner constant moves and this table does not, the filter silently starts
    dropping live rules or paying for dead ones."""
    from kagent import planner as P, reactive as R

    assert R.knob_alive("melon_opening", 0) and not R.knob_alive("melon_opening", 6)
    assert R.knob_alive("straw_target", P.CHAMP_LAND_DAY)
    assert not R.knob_alive("straw_target", 14), "straw target is read only to day 13"
    assert R.knob_alive("tomato_standing", P.TOMATO_LAST_PLANT_DAY)
    assert not R.knob_alive("tomato_standing", P.TOMATO_LAST_PLANT_DAY + 1)
    # the herd ramps reach their cap before buying stops, so LOWERING a cap
    # after that is inert -- measured as three bit-identical candidates of six
    assert R.knob_alive("sheep", 12) and not R.knob_alive("sheep", 15)
    assert R.knob_alive("cows", 8) and not R.knob_alive("cows", 9)


def test_provider_applies_the_rule_at_its_day_and_not_before():
    from kagent import planner, reactive as R

    seen = []

    def fake(view, memory, speculative=False):
        seen.append((view.day, dict(getattr(memory, "knobs", None) or {})))
        return "plan"

    class _View:
        def __init__(self, day):
            self.day = day

    class _Mem:
        knobs = None

    import kagent.features as F
    orig_plan, orig_ctx = planner.make_day_plan, F.context
    planner.make_day_plan = fake
    F.context = lambda v: {"day": v.day, "consumers": {"EGG": 2}}
    try:
        prov = R.ReactivePlanProvider({"geese": 0, "cows": 8}, [RULE])
        mem = _Mem()
        for day in (0, 5, 6, 7):
            prov.make_plan(_View(day), mem)
    finally:
        planner.make_day_plan, F.context = orig_plan, orig_ctx

    assert seen[0][1]["geese"] == 0, "base vector must be live from day 0"
    assert seen[1][1]["geese"] == 0, "rule must not fire before its day"
    assert seen[2][1]["geese"] == 4, "rule did not fire on its own day"
    assert seen[3][1]["geese"] == 4, "a knob set by a rule must persist"
    assert len(prov.fired) == 1, "a rule must be evaluated once, not every day"


def test_a_rule_that_never_fires_is_the_theta_arm():
    from kagent import planner, reactive as R

    seen = []

    def fake(view, memory, speculative=False):
        seen.append(dict(getattr(memory, "knobs", None) or {}))
        return "plan"

    class _View:
        day = 6

    class _Mem:
        knobs = None

    import kagent.features as F
    orig_plan, orig_ctx = planner.make_day_plan, F.context
    planner.make_day_plan = fake
    F.context = lambda v: {"day": 6, "consumers": {"EGG": 0}}   # condition false
    try:
        R.ReactivePlanProvider({"geese": 0}, [RULE]).make_plan(_View(), _Mem())
    finally:
        planner.make_day_plan, F.context = orig_plan, orig_ctx
    assert seen == [{"geese": 0}], "a non-firing rule must leave the plan untouched"


def test_firing_set_is_a_pure_lookup_over_the_census():
    from kagent import reactive as R

    rows = [{"ctx": {"6": {"day": 6, "consumers": {"EGG": 2}}}},
            {"ctx": {"6": {"day": 6, "consumers": {"EGG": 0}}}},
            {"ctx": {"9": {"day": 9, "consumers": {"EGG": 2}}}}]   # no day-6 ctx
    assert R.firing_set(RULE, rows) == [0]


def test_enumerate_drops_conditions_that_do_not_split_the_draw():
    """A rule firing in 5% of games cannot be resolved at se 0.045, and one
    firing in 98% is a constant wearing a condition -- and constants are
    closed. Both must be gone before anything is played."""
    from kagent import reactive as R

    rows = ([{"ctx": {"6": {"day": 6, "consumers": {"EGG": 2}}}}] * 19
            + [{"ctx": {"6": {"day": 6, "consumers": {"EGG": 0}}}}])   # 95/5 split
    out = R.enumerate_rules(rows, {"geese": [4]}, min_fire=0.20, max_fire=0.85,
                            days=(6,), fields=("consumers",))
    assert out == [], f"a 95/5 condition is untestable but survived: {out}"


def test_two_rules_break_the_partition_and_that_is_known():
    """A KNOWN LIMIT, pinned so it is not rediscovered as a mystery.

    The census records the champion's context. A candidate with one rule plays
    the champion's game up to that rule's day, so the context matches. With two
    rules on different days, the first changes the board and the second's
    context is no longer the champion's -- and shop draws re-roll on any board
    difference (memory: shop draws RE-ROLL under pairing). Multi-rule
    candidates therefore need their own census, not this one.
    """
    from kagent import reactive as R

    days = {r["day"] for r in [RULE, {**RULE, "day": 12, "knob": "cows"}]}
    assert len(days) > 1
    assert R.DECISION_DAYS == (6, 9, 12, 15)


def test_dev_reactive_dispatches_in_a_fresh_interpreter():
    """multiprocessing starts `spawn`, so the branch must live in the module."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from rl import rollout\n"
        "from kagent import theta as T\n"
        "spec = {'kind': 'reactive', **T.spec(T.champion()), 'rules': [%r]}\n"
        "fn, switched = rollout._make_dev_policy_agent(spec)\n"
        "assert callable(fn) and switched is False\n"
        "print('OK')\n" % (REPO, RULE)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_session_eval_parses_a_reactive_arm():
    src = open(os.path.join(REPO, "rl", "session_eval.py")).read()
    assert '"reactive"' in src
    assert "dev:reactive:<spec.json>" in src, "the error message must advertise it"


def test_rounds_nest_so_survivors_are_not_replayed():
    from kagent import reactive as R
    from tools.sh_reactive import draw_games

    rows = [{"bin": i % 6, "ctx": {"6": {"day": 6, "consumers": {"EGG": 2 if i % 3 else 0}}}}
            for i in range(300)]
    fire = R.firing_set(RULE, rows)
    a, b, c = (draw_games(RULE, rows, n, fire) for n in (8, 16, 40))
    assert a == b[:8] and b == c[:16], "rounds do not nest; survivors get replayed"
    assert set(c) <= set(fire), "a drawn game does not fire"


def test_the_arm_parser_does_not_let_the_spec_overwrite_its_own_kind():
    """The bug this pins: `{"kind": "reactive", **spec}` is wrong, because a
    spec written by theta.spec() carries "kind": "theta" and the spread puts it
    back. The arm then dispatches the THETA branch, every rule is dropped, and
    the gate reports the champion's own number -- which looks like a clean null
    rather than a broken arm. It cost a 30-minute three-arm gate to notice,
    and only because three different rule sets scored identically to the
    dollar."""
    src = open(os.path.join(REPO, "rl", "session_eval.py")).read()
    i = src.index('parts[1] == "reactive"')
    block = src[i:i + 900]
    assert '{**json.load(fh), "kind": "reactive"}' in block, \
        "the spec must not be able to overwrite kind"
    assert '{"kind": "reactive", **json.load(fh)}' not in block


def test_a_theta_spec_round_trips_into_a_reactive_arm_with_its_rules():
    """End to end through the real parser shape: a theta.spec() with rules
    bolted on must dispatch REACTIVE and keep every rule."""
    import tempfile

    from kagent import theta as T

    spec = T.spec(T.champion())
    spec["rules"] = [RULE]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    try:
        with open(path) as fh:
            parsed = {**json.load(fh), "kind": "reactive"}
        assert parsed["kind"] == "reactive"
        assert len(parsed["rules"]) == 1

        from rl import rollout
        fn, _ = rollout._make_dev_policy_agent(parsed)
        assert callable(fn)
    finally:
        os.unlink(path)
