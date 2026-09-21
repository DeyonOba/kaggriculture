"""README rules for the BUY_ANIMAL -> PICKUP -> PLACE chain.

Runs against the dev tree by default and against a frozen bundle with
`KAGG_BUNDLE=<name>` -- a candidate is always a patched bundle, and the dev
tree and the frozen champion have diverged before (see `build-variant`).

The rules under test, all from README.md and re-verified against the engine in
`docs/rules_verified.md`:

  Animals   "PLACE ... places one animal from inventory onto the tile. The `n`
            argument is ignored."  -> a carrier can only ever deliver ONE head
            per action.
  Shed      "At the end of the day, all items in all inventory will be added to
            shed inventory (if there is room). Anything that doesn't fit is
            discarded."  -> a head still in a pocket at midnight costs a round
            trip, and is destroyed outright at a full shed.
  Terrain   "DIG ... OR remove an *empty* goose coop / pasture. A coop or
            pasture with an animal on it cannot be dug."  -> a structure is a
            one-way commitment of a tile, so every animal that needs one must
            be visible to the planner that queues the build.

Two of these are `xfail`: the rule reading is right and the correction it
implies was MEASURED WORSE. See `docs/animal_chain_audit.md` -- reading a rule
correctly is not evidence about the counterfactual.

    PYTHONPATH=. uv run pytest tests/test_animal_chain.py -q
    KAGG_BUNDLE=champ-h8 PYTHONPATH=. uv run pytest tests/test_animal_chain.py -q
"""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.environ.get("KAGG_BUNDLE")
ANIMAL_KINDS = ("GOOSE", "COW", "SHEEP")

REFUTED = (
    "deliberate: the rule reading is right but the correction was measured "
    "WORSE -- one head per carrier costs 12 extra shed trips per 3 games for "
    "the same 48 placements and lost 54-34 on wins, McNemar p = 0.042 over "
    "692 ladder worlds. See docs/animal_chain_audit.md."
)


def load_bundle(name=BUNDLE):
    """Default: the dev tree (what `tools/submit.sh` would build).

    `KAGG_BUNDLE=<name>` instead loads `arena/agents/<name>/main.py`, because
    the dev tree and a frozen champion have diverged before and a candidate is
    always a patched bundle (see the `build-variant` skill).
    """
    if not name:
        from kagent import planner, scheduler
        from kagent.gamestate import Memory, View
        return {"View": View, "Memory": Memory,
                "make_champion_plan": planner.make_champion_plan,
                "build_tasks": scheduler.build_tasks,
                "assign": scheduler.assign}
    path = os.path.join(REPO, "arena", "agents", name, "main.py")
    if not os.path.isfile(path):
        pytest.skip(f"bundle {name} not built")
    outer = {"__name__": "kagent_bundle_test"}
    exec(compile(open(path).read(), path, "exec"), outer)
    # main.py execs every module into one shared dict and keeps it in `_G`;
    # the outer namespace only carries `agent`.
    return outer["_G"]


def blank_farm(unlocked=("NW",)):
    tiles = [["LOCKED"] * 10 for _ in range(10)]
    for y in range(10):
        for x in range(10):
            q = ("NW" if x < 5 else "NE") if y < 5 else ("SW" if x < 5 else "SE")
            if q in unlocked:
                tiles[y][x] = None
    return {
        "money": 5000.0,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": list(unlocked),
        "hires_today": 0,
    }


def make_obs(farm, shed=None, inventories=None, day=8, hour=6):
    products = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER"]
    return {
        "player": 0,
        "step": day * 24 + hour,
        "day": day,
        "hour": hour,
        "farms": [farm, blank_farm()],
        "market": {"inventory": {p: 10000 for p in products},
                   "prices": {p: 50 for p in products}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": [dict(i) for i in (inventories or [{}])],
        },
    }


def coop_at(farm, x, y, kind="COOP"):
    farm["tiles"][y][x] = {"kind": kind}


# --------------------------------------------------------------------------


@pytest.mark.xfail(reason=REFUTED, strict=False)
def test_never_picks_up_more_than_one_animal():
    """README Animals: PLACE takes exactly ONE head and ignores `n`.

    Four empty pastures and four cows in the shed is the shape that makes the
    provisioning step batch. Every PICKUP of an animal must ask for 1.
    """
    g = load_bundle()
    farm = blank_farm()
    farm["hands"] = [[4, 4], [4, 4], [3, 3]]
    for i, (x, y) in enumerate(((0, 0), (1, 0), (2, 0), (3, 0))):
        coop_at(farm, x, y, "PASTURE")
    obs = make_obs(farm, shed={"COW": 4, "WHEAT": 20})

    view = g["View"](obs)
    memory = g["Memory"]()
    memory.observe(view)
    plan = g["make_champion_plan"](view, memory)
    tasks = g["build_tasks"](view, plan)
    actions = g["assign"](view, plan, tasks)

    bad = [(idx, op) for idx, op in actions.items()
           if op and op[0] == "PICKUP" and op[1] in ANIMAL_KINDS and len(op) > 2
           and int(op[2]) > 1]
    assert not bad, (
        f"PLACE delivers one head and ignores n, so these pickups carry dead "
        f"weight: {bad}"
    )


@pytest.mark.xfail(reason=REFUTED, strict=False)
def test_never_stacks_two_animals_on_one_carrier():
    """Same rule, the other half: a unit already holding a head is not a
    candidate for another. The dispatcher picks the unit nearest the shed,
    which is exactly the one that just picked up."""
    g = load_bundle()
    farm = blank_farm()
    farm["hands"] = [[4, 4], [4, 4]]
    for x, y in ((0, 0), (1, 0), (2, 0), (3, 0)):
        coop_at(farm, x, y, "PASTURE")
    # unit 0 is already carrying a cow and is standing on the shed tile
    obs = make_obs(farm, shed={"COW": 3, "WHEAT": 20},
                   inventories=[{"COW": 1}, {}, {}])

    view = g["View"](obs)
    memory = g["Memory"]()
    memory.observe(view)
    plan = g["make_champion_plan"](view, memory)
    tasks = g["build_tasks"](view, plan)
    actions = g["assign"](view, plan, tasks)

    op = actions.get(0)
    assert not (op and op[0] == "PICKUP" and op[1] in ANIMAL_KINDS), (
        f"unit 0 already holds a COW and was sent to fetch another: {op}"
    )


def _herd_at_target_farm(n_placed=8, animal="SHEEP"):
    """Day 20: the cow ramp is over and the sheep target is 8, so a herd of 8
    placed sheep leaves the buy path with nothing to want. That isolates the
    pocket from the speculative `animal_buys` term."""
    farm = blank_farm(unlocked=("NW", "NE"))
    farm["hands"] = [[4, 4]]
    placed = 0
    for y in range(5):
        for x in range(5, 10):
            if placed >= n_placed:
                break
            farm["tiles"][y][x] = {
                "kind": "PASTURE", "animal": animal, "placed_day": 10,
                "yield_units": 0, "fed_today": True, "consecutive_unfed": 0,
                "cared_today": True, "fertilizer_available": False,
                "pending_care_bonus": 0,
            }
            placed += 1
    return farm


def test_pocketed_animal_gets_a_structure_queued():
    """README Terrain: DIG cannot remove an occupied structure, so the tile is
    a one-way commitment and the planner must see every animal that needs one.

    Herd at target (so nothing is on order), one sheep riding in a pocket, no
    empty pasture: that head is bought, alive and homeless, and the plan owes
    it a build. `scheduler.build_tasks` already counts pockets in `available`
    and will hand it a PLACE task; the planner has to agree.

    LATENT under the current cadence, and measured so: the day plan is built
    only when `memory.day_plan_day != view.day`, i.e. at hour 0, and the
    engine empties every unit inventory at midnight
    (`_drop_inventories_to_shed` then `private["inventories"] = [{}]`,
    kaggriculture.py:882). So no pocket is ever non-empty at plan time and the
    correction is bit-identical to champ-h8 over 346 ladder worlds. It is
    kept as a guard because any mid-day re-plan (MPC, rollout) walks into it.
    """
    g = load_bundle()
    farm = _herd_at_target_farm()
    obs = make_obs(farm, shed={"WHEAT": 20}, inventories=[{}, {"SHEEP": 1}],
                   day=20)

    view = g["View"](obs)
    memory = g["Memory"]()
    memory.observe(view)
    plan = g["make_champion_plan"](view, memory)

    pocketed = sum(view.unit_inv(i).get("SHEEP", 0) for i, _, _ in view.me.units)
    empty = sum(1 for _ in view.me.empty_structures())
    assert pocketed == 1 and empty == 0, "scenario drifted"
    assert plan.structure_builds.get("PASTURE", 0) >= 1, (
        "a pocketed sheep with no empty pasture owes 1 build; plan queued "
        f"{plan.structure_builds.get('PASTURE', 0)}"
    )


def test_pocketed_animal_is_not_re_bought():
    """The same blind spot on the buy side: `want` reads herd + shed, so a
    head in a pocket counts as missing and a replacement is ordered. At the
    target of 8 with 7 placed and 1 in hand the herd is already complete.

    LATENT for the same reason as the test above -- see its docstring."""
    g = load_bundle()
    farm = _herd_at_target_farm(n_placed=7)
    obs = make_obs(farm, shed={"WHEAT": 20}, inventories=[{}, {"SHEEP": 1}],
                   day=20)

    view = g["View"](obs)
    memory = g["Memory"]()
    memory.observe(view)
    plan = g["make_champion_plan"](view, memory)

    assert plan.animal_buys.get("SHEEP", 0) == 0, (
        "7 placed + 1 pocketed already meets the target of 8, but the plan "
        f"ordered {plan.animal_buys.get('SHEEP', 0)} more at $500 each"
    )


def test_shed_overflow_drop_never_names_an_animal():
    """README Animals: PLACE resolves the animal-placement branch FIRST when
    the unit stands on a matching empty structure. The shed-overflow path
    means to push cargo into the shed; naming an animal there puts one head on
    the tile and nothing in the shed."""
    g = load_bundle()
    farm = blank_farm(unlocked=("NW", "NE", "SW", "SE"))
    # a pasture on a shed-access tile -- the build queue sorts empties by
    # distance to the shed and all four shed tiles tie for the minimum
    coop_at(farm, 4, 4, "PASTURE")
    # shed nearly full so the DROP branch cannot take the whole load
    shed = {"WHEAT": 98, "COW": 1}
    obs = make_obs(farm, shed=shed, inventories=[{"COW": 2, "MILK": 1}], day=12)

    view = g["View"](obs)
    memory = g["Memory"]()
    memory.observe(view)
    plan = g["make_champion_plan"](view, memory)
    tasks = g["build_tasks"](view, plan)
    actions = g["assign"](view, plan, tasks)

    op = actions.get(0)
    if op and op[0] == "PLACE" and len(op) > 2:
        assert op[1] not in ANIMAL_KINDS, (
            f"shed-overflow drop named an animal while standing on a matching "
            f"empty structure: {op}"
        )
