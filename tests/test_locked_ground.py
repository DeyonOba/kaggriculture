"""Invariants about unbought ground and silently-discarded actions.

README Movement + kaggriculture.py:325-345:

  movement                        allowed onto and across LOCKED tiles
  PICKUP / DROP / PLACE-into-shed allowed -- they resolve BEFORE the locked
                                  guard and use the tile as a standing
                                  position only
  every other op                  `if tile == "LOCKED": return` -- a SILENT
                                  no-op that still spends the unit-turn

Nothing in the engine reports a wasted turn, so these have to be asserted or
they rot. `tools/noop_audit.py` is the same check with a full report.

    PYTHONPATH=. uv run pytest tests/test_locked_ground.py -q
    KAGG_BUNDLE=h10-pad1 PYTHONPATH=. uv run pytest tests/test_locked_ground.py -q
"""
import collections
import os

import pytest

from tools import rule_harness as H

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.environ.get("KAGG_BUNDLE", "champ-h8")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
SHED_OPS = ("PICKUP", "DROP", "PLACE")
SHED_ACCESS = {(4, 4), (5, 4), (4, 5), (5, 5)}
PAD = 1


@pytest.fixture(scope="module")
def game():
    from kaggle_environments import make
    path = os.path.join(REPO, "arena", "agents", BUNDLE, "main.py")
    if not os.path.isfile(path):
        pytest.skip(f"bundle {BUNDLE} not built")
    rec = H.Recorder(0, record_all=True)
    rec.install()
    try:
        env = make("kaggriculture",
                   configuration={"seed": 71000, "episodeSteps": 720},
                   debug=False)
        env.run([path, "arena/agents/pub-ahmed-v38/main.py"])
    finally:
        rec.uninstall()
    return rec, env


def test_no_tile_op_on_locked_ground(game):
    """Every op but movement and the three shed ops is a silent no-op on a
    LOCKED tile -- and still costs the turn."""
    rec, _ = game
    bad = [(r["step"], r["unit"], r["op"], (r["x"], r["y"]))
           for r in rec.unit_ops
           if r["tile_before"] == "LOCKED"
           and r["op"][0] not in MOVES + SHED_OPS + ("PASS",)]
    assert not bad, (
        f"{len(bad)} unit-turns spent on a tile op over unbought ground "
        f"(each one is a silent no-op): {bad[:5]}"
    )


def test_shed_ops_over_locked_ground_are_legitimate(game):
    """The permissive half: shed ops from a locked shed-access tile are legal
    and load-bearing -- a hand spawns on (5,4), locked until NE is bought."""
    rec, _ = game
    off = [(r["step"], r["op"], (r["x"], r["y"]))
           for r in rec.unit_ops
           if r["tile_before"] == "LOCKED" and r["op"][0] in SHED_OPS
           and (r["x"], r["y"]) not in SHED_ACCESS]
    assert not off, f"shed op on locked ground away from the shed: {off[:5]}"


def test_routing_stays_within_the_padding(game):
    """Design constraint: a unit may stand on unbought ground only within PAD
    tiles of owned land.

    The one exception the engine forces on us is the spawn: `_spawn_hand`
    ignores locked status, and `(5,5)` is depth 2 while only NW is owned, so a
    hand can appear there at hour 0. Anything else is our routing choosing to
    cut through a quadrant where no tile action is legal.
    """
    rec, _ = game
    deep = [r for r in rec.unit_ops if r.get("locked_depth", 0) > PAD]
    spawned = [r for r in deep
               if r["step"] % 24 == 0 and (r["x"], r["y"]) in SHED_ACCESS]
    chosen = [(r["step"], r["unit"], r["op"], (r["x"], r["y"]),
               r["locked_depth"]) for r in deep if r not in spawned]
    if BUNDLE != "champ-h8" and "pad" in BUNDLE:
        assert not chosen, (
            f"{len(chosen)} unit-turns deeper than PAD={PAD} into unbought "
            f"ground that the spawn did not force: {chosen[:5]}"
        )
    else:
        # champ-h8 does not enforce it; record the rate so a regression shows
        n = len(rec.unit_ops)
        assert len(chosen) / max(1, n) < 0.01, (
            f"{len(chosen)}/{n} unit-turns routed deeper than PAD={PAD}"
        )


def test_no_silently_discarded_market_or_plant(game):
    """Two paths the engine hides: it rewrites ALL of a crop's PLANTs to PASS
    when a turn over-requests seeds, and truncates market orders past the cap.

    Alignment: steps[t]'s action was chosen while observing state t-1.
    """
    _, env = game
    over, dropped = 0, 0
    for t in range(1, len(env.steps)):
        act = env.steps[t][0].get("action") or {}
        if not isinstance(act, dict):
            continue
        ops = [act.get("farmer")] + list(act.get("hands") or [])
        want = collections.Counter(
            o[1] for o in ops
            if isinstance(o, list) and len(o) > 1 and o[0] == "PLANT")
        priv = env.steps[t - 1][0].observation.private
        seeds = priv.get("seeds", {}) if hasattr(priv, "get") else {}
        for crop, k in want.items():
            if k > seeds.get(crop, 0):
                over += k
        dropped += max(0, len(act.get("market") or []) - 10)
    assert over == 0, f"{over} PLANT requests voided by the atomic-seed rule"
    assert dropped == 0, f"{dropped} market orders dropped past the cap"
