"""opening_v2 (planner knob, off by default): a 4-hand crew on days 0-2, crops
that pay first planted first, tomorrow's hire bill kept in the bank, seeds in
lots, animal wants that the market buys as cash arrives while structures
follow what dawn cash affords.
    PYTHONPATH=. uv run pytest tests/test_opening_v2.py -q
"""
import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import market, planner  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402


def _obs(day=0, money=3000):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = day, 0, day * 24
    obs["farms"][0]["money"] = money
    return obs


def _plan(obs, **knobs):
    mem = Memory()
    if knobs:
        mem.knobs = dict(knobs)
    view = View(obs)
    return view, planner.make_champion_plan(view, mem), mem


def test_shipped_day_zero_is_unchanged_without_the_knob():
    view, plan, _ = _plan(_obs())
    assert plan.hires_target == 3 and plan.cash_floor == 0.0 and plan.seed_lot == 0
    assert list(plan.plant_quota)[0] == "MELON"


def test_v2_day_zero_crew_order_floor_and_lot():
    view, plan, _ = _plan(_obs(), opening_v2=1)
    assert plan.hires_target == 4
    assert list(plan.plant_quota) == ["WHEAT", "MELON"]         # pays in 2 days before pays in 10
    assert plan.plant_quota == {"WHEAT": 7, "MELON": 12}
    assert plan.cash_floor > 0 and plan.seed_lot == 2 * (1 + plan.hires_target)


def test_v2_wants_stand_but_structures_follow_dawn_cash():
    # day 4, $50 in the bank, fresh farm: the shipped plan wants 4 cows + 2
    # sheep (the market buys them as cash lands) AND builds six pastures for
    # them now -- ground for animals the bank cannot buy
    view, plan, _ = _plan(_obs(day=4, money=50))
    assert plan.animal_buys == {"SHEEP": 2, "COW": 4} and plan.structure_builds["PASTURE"] == 6
    # v2: the same wants, no structure until the cash is there
    view, plan2, _ = _plan(_obs(day=4, money=50), opening_v2=1)
    assert plan2.animal_buys == plan.animal_buys
    assert plan2.structure_builds == {"COOP": 0, "PASTURE": 0}
    # day 4, $2,000: pastures for what dawn cash covers (sheep first: 2 x 500,
    # then cows 2 x 400 fit in the remainder), not for all six
    view, plan3, _ = _plan(_obs(day=4, money=2000), opening_v2=1)
    assert 0 < plan3.structure_builds["PASTURE"] < 6


def test_market_leaves_the_floor_and_buys_seeds_in_lots():
    view, plan, mem = _plan(_obs(), opening_v2=1)
    plan.hires_target = 0                         # isolate the purchases
    plan.cash_floor = 2900.0                      # only $100 to spend
    orders = market.build_orders(view, plan, mem)
    seeds = [o for o in orders if o[0] == "BUY_SEED"]
    animals = [o for o in orders if o[0] == "BUY_ANIMAL"]
    assert animals == []                          # a cow costs 400 > 100
    assert all(o[2] <= plan.seed_lot for o in seeds)
    assert seeds and seeds[0][1] == "WHEAT"       # wheat seeds first


def test_lead_opening_is_the_leaders_day0_basket():
    from kagent import planner, market
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    view = View(obs)
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["lead_opening"] = 1; mem.knobs_decided = True
    plan = planner.make_champion_plan(view, mem)
    assert plan.plant_quota == {"MELON": 6, "WHEAT": 10}, plan.plant_quota
    assert plan.animal_buys == {"SHEEP": 3, "COW": 2}, plan.animal_buys
    assert plan.hires_target == 4 and plan.hire_all_cash
    # day 1 on a $4 dawn: 3 hands for $4, and the four remaining melons
    obs["day"], obs["hour"], obs["step"] = 1, 0, 24
    obs["farms"][0]["money"] = 4
    view = View(obs)
    plan = planner.make_champion_plan(view, mem)
    assert plan.hires_target == 3 and plan.plant_quota["MELON"] == 10
    orders = market.build_orders(view, plan, mem)
    assert [o[0] for o in orders].count("HIRE") == 3, orders
    # shipped: the workload cap holds day 0 at 3 hands and the $4 dawn hires none
    mem.knobs["lead_opening"] = 0
    obs["day"], obs["hour"], obs["step"] = 0, 0, 0
    obs["farms"][0]["money"] = 3000
    plan0 = planner.make_champion_plan(View(obs), mem)
    assert plan0.hires_target == 3 and plan0.plant_quota == {"MELON": 12, "WHEAT": 7} and plan0.animal_buys == {"SHEEP": 2, "COW": 2}


def test_lead_hires_is_the_leaders_day_schedule():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["farms"][0]["money"] = 5_000
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["lead_hires"] = 1; mem.knobs_decided = True
    for day, want in ((0, 5), (2, 5), (10, 12), (16, 12)):
        obs["day"], obs["hour"], obs["step"] = day, 0, day * 24
        plan = planner.make_champion_plan(View(obs), mem)
        assert plan.hires_target == want, (day, plan.hires_target)
    mem.knobs["lead_hires"] = 0
    obs["day"], obs["hour"], obs["step"] = 0, 0, 0
    assert planner.make_champion_plan(View(obs), mem).hires_target == 3           # shipped: the workload cap on day 0


def test_posture_protects_when_ahead_and_presses_when_behind():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 12, 0, 12 * 24
    obs["farms"][0]["money"] = 30_000
    obs["farms"][1]["money"] = 10_000
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["posture"] = 1; mem.knobs_decided = True
    ahead = planner.make_champion_plan(View(obs), mem).plant_quota
    obs["farms"][0]["money"], obs["farms"][1]["money"] = 10_000, 30_000
    behind = planner.make_champion_plan(View(obs), mem).plant_quota
    mem.knobs["posture"] = 0
    level = planner.make_champion_plan(View(obs), mem).plant_quota
    assert ahead.get("STRAWBERRY", 0) < level.get("STRAWBERRY", 0) < behind.get("STRAWBERRY", 0), (ahead, level, behind)
    assert ahead.get("WHEAT", 0) >= level.get("WHEAT", 0)


def test_press_niche_buys_land_and_plants_the_crop_the_opponent_left_out():
    from kagent import planner
    from kagent.constants import MARKET_PARAMS
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 14, 0, 14 * 24
    obs["farms"][0]["money"], obs["farms"][1]["money"] = 5_000, 30_000          # we are $25k behind
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW"]
    obs["town"]["unlocked_shops"] = ["PET_CAFE"]                                 # carrot demand beyond the town centre
    obs["market"]["inventory"]["CARROT"] = MARKET_PARAMS["CARROT"]["I0"] - 50    # not glutted
    # the opponent works a much bigger farm and has no carrots
    for y in range(10):
        for x in range(10):
            if obs["farms"][1]["tiles"][y][x] is None:
                obs["farms"][1]["tiles"][y][x] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 8, "watered_today": False,
                                                  "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1, "fertilized_until_day": -1}
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["press_niche"] = 1; mem.knobs_decided = True
    plan = planner.make_champion_plan(View(obs), mem)
    assert "SE" in plan.land_buys and plan.land_margin == 600, (plan.land_buys, plan.land_margin)
    assert plan.plant_quota.get("CARROT", 0) >= planner.NICHE_TILES, plan.plant_quota
    # ahead: nothing of it
    obs["farms"][0]["money"], obs["farms"][1]["money"] = 30_000, 5_000
    plan = planner.make_champion_plan(View(obs), mem)
    assert "SE" not in plan.land_buys and plan.plant_quota.get("CARROT", 0) < planner.NICHE_TILES
    # the opponent grows carrots too: no niche
    obs["farms"][0]["money"], obs["farms"][1]["money"] = 5_000, 30_000
    for x in range(3):
        obs["farms"][1]["tiles"][0][x]["crop"] = "CARROT"
    plan = planner.make_champion_plan(View(obs), mem)
    assert plan.plant_quota.get("CARROT", 0) < planner.NICHE_TILES


def test_straw_day_lets_strawberries_take_freed_tiles_early():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 3, 0, 72
    obs["farms"][0]["money"] = 2_000
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs_decided = True
    assert planner.make_champion_plan(View(obs), mem).plant_quota.get("STRAWBERRY", 0) == 0    # shipped: from day 5
    mem.knobs["straw_day"] = 2
    assert planner.make_champion_plan(View(obs), mem).plant_quota.get("STRAWBERRY", 0) > 0


def test_egg_niche_fires_on_the_shop_draw_and_an_empty_opponent_coop_line():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 8, 0, 8 * 24
    obs["farms"][0]["money"] = 3_000
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["egg_niche"] = 1; mem.knobs_decided = True
    obs["town"]["unlocked_shops"] = ["BAKERY"]                                   # eggs demanded
    plan = planner.make_champion_plan(View(obs), mem)
    assert plan.animal_buys.get("GOOSE", 0) >= 1 and plan.structure_builds["COOP"] >= 1, (plan.animal_buys, plan.structure_builds)
    obs["farms"][1]["tiles"][0][0] = {"kind": "COOP"}                            # the opponent has a coop: no niche
    assert planner.make_champion_plan(View(obs), mem).animal_buys.get("GOOSE", 0) == 0
    obs["farms"][1]["tiles"][0][0] = None
    obs["town"]["unlocked_shops"] = ["YARN_STORE"]                               # no egg demand: no niche
    assert planner.make_champion_plan(View(obs), mem).animal_buys.get("GOOSE", 0) == 0


def test_land4_niche_loads_the_fourth_quadrant_with_the_demanded_crop():
    from kagent import planner
    from kagent.constants import MARKET_PARAMS
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 16, 0, 16 * 24
    obs["farms"][0]["money"] = 20_000
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    obs["town"]["unlocked_shops"] = ["PET_CAFE"]
    obs["market"]["inventory"]["CARROT"] = MARKET_PARAMS["CARROT"]["I0"] - 50
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs.update({"land4": 1, "land4_niche": 1}); mem.knobs_decided = True
    plan = planner.make_champion_plan(View(obs), mem)
    assert plan.plant_quota.get("CARROT", 0) >= planner.NICHE_TILES, plan.plant_quota
    mem.knobs["land4_niche"] = 0
    assert planner.make_champion_plan(View(obs), mem).plant_quota.get("CARROT", 0) < planner.NICHE_TILES


def test_land4_early_asks_for_the_fourth_quadrant_from_its_day_on_a_small_margin():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["farms"][0]["money"] = 5_000
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW"]
    obs["farms"][0]["hands"] = [[4, 4]] * 3                                    # a small crew: the gated version would refuse
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs.update({"land4_early": 1, "land4_day": 12, "land4_margin": 600}); mem.knobs_decided = True
    for day, want in ((11, False), (12, True), (20, True)):
        obs["day"], obs["hour"], obs["step"] = day, 0, day * 24
        plan = planner.make_champion_plan(View(obs), mem)
        assert ("SE" in plan.land_buys) == want, (day, plan.land_buys)
        if want:
            assert plan.land_margin == 600
    mem.knobs["land4_early"] = 0
    obs["day"], obs["hour"], obs["step"] = 12, 0, 12 * 24
    assert "SE" not in planner.make_champion_plan(View(obs), mem).land_buys


def test_quad4_plans_the_fourth_quadrant_from_the_market_and_delivers_it_there():
    from kagent import planner, scheduler
    from kagent.constants import MARKET_PARAMS
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 13, 3, 13 * 24 + 3
    obs["farms"][0]["money"] = 6_000
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    obs["farms"][0]["farmer"] = [4, 4]
    obs["private"]["seeds"] = {"WHEAT": 30, "CARROT": 30, "MELON": 5}
    farm = obs["farms"][0]
    for y in range(10):
        for x in range(10):
            farm["tiles"][y][x] = None                                        # everything bare, all four quadrants
    obs["town"]["unlocked_shops"] = ["PET_CAFE"]                             # carrots demanded beyond the centre
    obs["market"]["inventory"]["CARROT"] = MARKET_PARAMS["CARROT"]["I0"] - 20
    obs["market"]["prices"]["CARROT"] = 40.0
    obs["market"]["prices"]["MELON"] = 10.0                                   # tanked
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs["quad4"] = 1; mem.knobs_decided = True
    view = View(obs)
    plan = planner.make_champion_plan(view, mem)
    se = plan.quad_quota.get("SE")
    assert se and sum(se.values()) == 25 and se.get("WHEAT", 0) == round(0.67 * 25) and se.get("CARROT", 0) == 25 - round(0.67 * 25), plan.quad_quota
    assert "MELON" not in se
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}                        # no pasture taking an SE tile
    tasks = scheduler.build_tasks(view, plan)
    se_plants = [t for t in tasks if t.op[0] == "PLANT" and t.x >= 5 and t.y >= 5]
    assert len(se_plants) == 25 and {t.op[1] for t in se_plants} == {"WHEAT", "CARROT"}, len(se_plants)
    # the farm-wide quota never reaches the fourth quadrant's tiles
    plan.plant_quota = {"MELON": 5}
    tasks = scheduler.build_tasks(view, plan)
    assert not [t for t in tasks if t.op == ["PLANT", "MELON"] and t.x >= 5 and t.y >= 5]
    # no quadrant plan without the knob, or when SE is not owned
    mem.knobs["quad4"] = 0
    assert not planner.make_champion_plan(view, mem).quad_quota
    mem.knobs["quad4"] = 1
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW"]
    assert not planner.make_champion_plan(View(obs), mem).quad_quota


def test_quad4_hires_and_tiles_scale_the_crew_and_cap_the_quadrant_plan():
    from kagent import planner
    from kagent.gamestate import Memory, View
    import copy
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.reset(2)
    obs = copy.deepcopy(env.state[0].observation)
    obs["day"], obs["hour"], obs["step"] = 14, 0, 14 * 24
    obs["farms"][0]["money"] = 20_000
    farm = obs["farms"][0]
    for y in range(10):
        for x in range(10):
            farm["tiles"][y][x] = None
    mem = Memory(); mem.knobs = dict(planner.KNOB_DEFAULTS); mem.knobs.update({"quad4": 1, "quad4_hires": 2, "quad4_tiles": 12}); mem.knobs_decided = True
    farm["unlocked_quadrants"] = ["NW", "NE", "SW"]
    base = planner.make_champion_plan(View(obs), mem)
    farm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    plan = planner.make_champion_plan(View(obs), mem)
    assert plan.hires_target == base.hires_target + 2, (plan.hires_target, base.hires_target)
    assert sum(plan.quad_quota["SE"].values()) == 12, plan.quad_quota
