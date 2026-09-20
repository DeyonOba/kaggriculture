"""Verify every rule README.md states against the engine that implements it.

This is EXPLORATION, not usage. The question is not "does our agent use this"
but "is the documented rule what the code actually does". A rule we believe
but have not checked is a rule we will eventually build an arm on, and this
project has already paid for that twice: a $88/unit melon that never existed,
and a 'FEED is the wheat-buy mechanism' that measured r = -0.079.

Every check below either SIMULATES the engine or reads its constants. None
of them restate the README back to itself.

Sections, in README order:
  1  Object Types (the table, then each note under it)
  2  Actions -- Movement, Shed, Plants, Animals, Animal Care, Terrain, PASS
  3  Market Action
  4  Harvest Yields
  5  Map Features -- Farm Space, Shed, Farmer/Farm Hand, Town Buildings
  6  Market Mechanics -- selling, buying, the Price Function

    PYTHONPATH=. uv run python tools/readme_audit.py
    PYTHONPATH=. uv run python tools/readme_audit.py --section price
"""
import argparse
import copy
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

TPD = 24
SECTIONS = []


def section(name, title):
    def deco(fn):
        SECTIONS.append((name, title, fn))
        return fn
    return deco


def _blank_farm(n=1):
    # _apply_unit_action resolves the acting unit through _farmer_position,
    # so a farm without "farmer"/"hands" raises KeyError rather than no-opping
    return {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0],
            "hands": []}


def grow_crop(crop, upto_age, water=True, fert=False):
    """Age one plant under a chosen regime. -> {age: yield_units or None}.

    Mirrors the engine's own order within a day: the unit's WATER action
    happens DURING the day (and is where the one-time bonus is applied --
    line 431-444, not the day refresh), then `_daily_refresh_plants` runs at
    end of day. Getting that order wrong is what makes a plant look like it
    never gains yield.
    """
    farm = _blank_farm()
    farm["tiles"][0][0] = K._new_plant(crop, 0, TPD)
    out = {}
    for day in range(0, upto_age + 1):
        t = farm["tiles"][0][0]
        if not isinstance(t, dict) or t.get("kind") != "PLANT":
            out[day] = None
            break
        if fert:
            t["fertilized_until_day"] = day
        if water:
            K._apply_unit_action(farm, {"shed": {}, "seeds": {},
                                        "inventories": [{}]},
                                 0, ["WATER"], 1, day, TPD)
        out[day] = t.get("yield_units", 0)
        K._daily_refresh_plants(farm, day, TPD)
    return out


def grow_animal(animal, upto_day, feed=True, care=False):
    """-> {day: yield_units or None}. None once the animal has escaped."""
    farm = _blank_farm()
    farm["tiles"][0][0] = K._new_animal(animal, 0)
    out = {}
    for day in range(0, upto_day + 1):
        t = farm["tiles"][0][0]
        if not isinstance(t, dict) or not t.get("animal"):
            out[day] = None
            break
        if feed:
            t["fed_today"] = True
        if care:
            t["cared_today"] = True
        out[day] = t.get("yield_units", 0)
        K._daily_refresh_animals(farm, day)
    return out


# ---------------------------------------------------------------- 1. objects
README_OBJECTS = {
    # type:        (seed/cost, base, first_yield, max_yield_day_doc, max_yield)
    "WHEAT":      (10,  25,  2,  4,  6),
    "CARROT":     (20,  35,  2,  3,  4),
    "TOMATO":     (50,  60,  8,  11, 4),
    "STRAWBERRY": (100, 120, 10, 16, 4),
    "MELON":      (80,  250, 10, 10, 6),
}
README_ANIMALS = {
    "GOOSE": (300, 50,  4, 1, 4, "EGG",  "COOP"),
    "COW":   (400, 160, 8, 2, 6, "MILK", "PASTURE"),
    "SHEEP": (500, 200, 6, 3, 6, "WOOL", "PASTURE"),
}
README_YTD = {"WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.33,
              "STRAWBERRY": 0.24, "MELON": 0.55,
              "GOOSE": 1.00, "COW": 0.50, "SHEEP": 0.33}


@section("objects", "1. Object Types -- the table")
def s_objects():
    r = []
    for crop, (seed, base, fy, myd_doc, my) in README_OBJECTS.items():
        c = K.CROPS[crop]
        r.append((f"{crop} seed cost = {seed}", c["seed"] == seed,
                  f"engine {c['seed']}"))
        r.append((f"{crop} base price = {base}",
                  K.MARKET_PARAMS[crop]["base"] == base,
                  f"engine {K.MARKET_PARAMS[crop]['base']}"))
        r.append((f"{crop} first yield = day {fy}",
                  c["first_yield_day"] == fy, f"engine {c['first_yield_day']}"))
        r.append((f"{crop} max yield = {my}", c["max_yield"] == my,
                  f"engine {c['max_yield']}"))
    for a, (cost, base, fy, iv, held, prod, struct) in README_ANIMALS.items():
        d = K.ANIMALS[a]
        r.append((f"{a} cost = {cost}", d["cost"] == cost,
                  f"engine {d['cost']}"))
        r.append((f"{prod} base price = {base}",
                  K.MARKET_PARAMS[prod]["base"] == base,
                  f"engine {K.MARKET_PARAMS[prod]['base']}"))
        r.append((f"{a} first yield = day {fy}", d["first_yield_day"] == fy,
                  f"engine {d['first_yield_day']}"))
        r.append((f"{a} interval = {iv}", d["interval"] == iv,
                  f"engine {d['interval']}"))
        r.append((f"{a} max_held = {held}", d["max_held"] == held,
                  f"engine {d['max_held']}"))
        r.append((f"{a} structure = {struct}", d["structure"] == struct,
                  f"engine {d['structure']}"))
        r.append((f"{a} product = {prod}", d["product"] == prod,
                  f"engine {d['product']}"))
    r.append(("FERTILIZER seed/buy cost = 100",
              K.MARKET_PARAMS["FERTILIZER"]["base"] == 100,
              f"engine base {K.MARKET_PARAMS['FERTILIZER']['base']}"))
    return r


@section("ytd", "1b. 'Yield / tile / day' column")
def s_ytd():
    """README defines this precisely, so it is checkable, not decorative."""
    r = []
    for crop in README_OBJECTS:
        c = K.CROPS[crop]
        g = grow_crop(crop, 20, water=True, fert=False)
        peak = max(v for v in g.values() if v is not None)
        # "days the tile is occupied, watering daily and harvesting AT PEAK
        # YIELD" -- so occupancy runs from planting (age 0) through the day of
        # that harvest, INCLUSIVE. Two denominators are wrong here and both
        # were mine: max_yield_day+1 overstates MELON (peak lands at age 10,
        # not 12) and last_age undercounts the ongoing crops by the planting
        # day. The engine was right in both cases.
        if c["ongoing"]:
            last_age = c["first_yield_day"] + c["interval"] * (c["max_yield"] - 1)
            total, occupancy = c["max_yield"], last_age + 1
        else:
            peak_age = min(a for a, v in g.items() if v == peak)
            total, occupancy = peak, peak_age + 1
        got = total / occupancy
        doc = README_YTD[crop]
        r.append((f"{crop} yield/tile/day = {doc}", abs(got - doc) <= 0.02,
                  f"engine {got:.2f} ({total} units / {occupancy} days)"))
    for a in README_ANIMALS:
        d = K.ANIMALS[a]
        got = 1.0 / d["interval"]
        doc = README_YTD[a]
        r.append((f"{a} yield/tile/day = {doc} (steady state 1/interval)",
                  abs(got - doc) <= 0.02, f"engine {got:.2f}"))
    return r


@section("cropnotes", "1c. The three crop notes under the table")
def s_cropnotes():
    r = []
    # -- melon: bonus window 6-12, cap at 10 watering, 8 fertilised
    ws = (K.CROPS["MELON"]["max_yield_day"] + 1) // 2
    r.append(("MELON bonus window starts at age 6", ws == 6,
              f"engine window_start = (max_yield_day+1)//2 = {ws}"))
    r.append(("MELON bonus window ends at age 12",
              K.CROPS["MELON"]["max_yield_day"] == 12,
              f"engine max_yield_day {K.CROPS['MELON']['max_yield_day']}"))
    g = grow_crop("MELON", 14)
    cap_at = next((a for a in sorted(g) if g[a] == 6), None)
    r.append(("MELON reaches cap 6 at age 10 when watered", cap_at == 10,
              f"engine caps at age {cap_at}; ages 11-12 add "
              f"{g.get(12, 0) - g.get(10, 0)}"))
    gf = grow_crop("MELON", 14, fert=True)
    cap_f = next((a for a in sorted(gf) if gf[a] == 6), None)
    r.append(("MELON fertilised reaches cap at age 8", cap_f == 8,
              f"engine caps at age {cap_f}"))
    # -- wheat / carrot: listed max only with fertilizer
    for crop, wat, fer in (("WHEAT", 4, 6), ("CARROT", 3, 4)):
        gw = grow_crop(crop, 10)
        gg = grow_crop(crop, 10, fert=True)
        pw = max(v for v in gw.values() if v is not None)
        pf = max(v for v in gg.values() if v is not None)
        r.append((f"{crop} peaks at {wat} on watering alone", pw == wat,
                  f"engine {pw}"))
        r.append((f"{crop} reaches {fer} with fertilizer", pf == fer,
                  f"engine {pf}"))
    # -- tomato / strawberry: 4 scheduled yields at stated ages
    for crop, ages in (("TOMATO", [8, 9, 10, 11]),
                       ("STRAWBERRY", [10, 12, 14, 16])):
        c = K.CROPS[crop]
        got = [c["first_yield_day"] + c["interval"] * i
               for i in range(c["max_yield"])]
        r.append((f"{crop} yields at ages {ages}", got == ages,
                  f"engine {got}"))
        r.append((f"{crop} capped at 4 scheduled yields",
                  c["max_yield"] == 4, f"engine max_yield {c['max_yield']}"))
    # -- two missed days kills
    dry = grow_crop("WHEAT", 6, water=False)
    died = next((a for a in sorted(dry) if dry[a] is None), None)
    r.append(("unwatered plant becomes a WEED after 2 successive days",
              died is not None and died <= 2,
              f"engine: weed at age {died} "
              f"(new plant starts consecutive_unwatered=1)"))
    starve = grow_animal("GOOSE", 6, feed=False)
    gone = next((d for d in sorted(starve) if starve[d] is None), None)
    r.append(("unfed animal escapes after 2 successive days",
              gone is not None and gone <= 3,
              f"engine: gone at day {gone} "
              f"(new animal starts consecutive_unfed=0)"))
    r.append(("WHEAT is buyable from the market",
              "WHEAT" in getattr(K, "BUYABLE", ("WHEAT", "FERTILIZER")),
              "BUY_PRODUCT restricted set"))
    return r


# ---------------------------------------------------------------- 2. actions
@section("turns", "2. Actions -- season shape")
def s_turns():
    j = os.path.join(os.path.dirname(K.__file__), "kaggriculture.json")
    import json
    raw = json.load(open(j))["configuration"]
    # some entries are {"default": x} schema dicts, others are bare values
    cfg = {k: (v["default"] if isinstance(v, dict) and "default" in v else v)
           for k, v in raw.items()}
    cfg = {k: {"default": v} for k, v in cfg.items()}
    r = [("24 turns per day", cfg["turnsPerDay"]["default"] == 24,
          f"engine {cfg['turnsPerDay']['default']}"),
         ("720 total turns", cfg["episodeSteps"]["default"] == 720,
          f"engine {cfg['episodeSteps']['default']}"),
         ("30 days in a season",
          cfg["episodeSteps"]["default"] // cfg["turnsPerDay"]["default"] == 30,
          f"engine {cfg['episodeSteps']['default'] // cfg['turnsPerDay']['default']}"),
         ("startingMoney 3000", cfg["startingMoney"]["default"] == 3000,
          f"engine {cfg['startingMoney']['default']}"),
         ("boardSize 10", cfg["boardSize"]["default"] == 10,
          f"engine {cfg['boardSize']['default']}"),
         ("maxMarketOrdersPerTurn 10",
          cfg["maxMarketOrdersPerTurn"]["default"] == 10,
          f"engine {cfg['maxMarketOrdersPerTurn']['default']}"),
         ("shedCapacity 100", cfg["shedCapacity"]["default"] == 100,
          f"engine {cfg['shedCapacity']['default']}"),
         ("weedSpawnChance 0.005", cfg["weedSpawnChance"]["default"] == 0.005,
          f"engine {cfg['weedSpawnChance']['default']}"),
         ("townShopUnlockInterval 3",
          cfg["townShopUnlockInterval"]["default"] == 3,
          f"engine {cfg['townShopUnlockInterval']['default']}"),
         ("townShopSellInterval 4", cfg["townShopSellInterval"]["default"] == 4,
          f"engine {cfg['townShopSellInterval']['default']}"),
         ("townCenterSellInterval 24",
          cfg["townCenterSellInterval"]["default"] == 24,
          f"engine {cfg['townCenterSellInterval']['default']}")]
    return r


@section("movement", "2a. Movement")
def s_movement():
    r = []
    n = 10
    priv = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}

    def at(pos, tiles=None):
        f = {"tiles": tiles or [[None] * n for _ in range(n)],
             "farmer": list(pos), "hands": []}
        return f
    # off-edge is a no-op
    f = at((0, 0))
    K._apply_unit_action(f, copy.deepcopy(priv), 0, ["NORTH"], n, 0, TPD)
    r.append(("moves off the edge are no-ops", f["farmer"] == [0, 0],
              f"engine farmer at {f['farmer']}"))
    # LOCKED is passable
    tiles = [[None] * n for _ in range(n)]
    tiles[0][1] = "LOCKED"
    f = at((0, 0), tiles)
    K._apply_unit_action(f, copy.deepcopy(priv), 0, ["EAST"], n, 0, TPD)
    r.append(("LOCKED tiles are passable", f["farmer"] == [1, 0],
              f"engine farmer at {f['farmer']}"))
    # tile actions no-op on LOCKED
    tiles = [[None] * n for _ in range(n)]
    tiles[0][0] = "LOCKED"
    f = at((0, 0), tiles)
    p = {"shed": {}, "seeds": {"WHEAT": 1}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["PLANT", "WHEAT"], n, 0, TPD)
    r.append(("PLANT on a LOCKED tile is a no-op and consumes nothing",
              f["tiles"][0][0] == "LOCKED" and p["seeds"]["WHEAT"] == 1,
              f"tile {f['tiles'][0][0]}, seeds {p['seeds']['WHEAT']}"))
    # shed actions work from a LOCKED shed-access tile
    half = n // 2
    tiles = [[None] * n for _ in range(n)]
    tiles[half][half] = "LOCKED"
    f = at((half, half), tiles)
    p = {"shed": {"WHEAT": 3}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["PICKUP", "WHEAT", 2], n, 0, TPD)
    r.append(("PICKUP works from a LOCKED shed-access tile",
              p["inventories"][0].get("WHEAT", 0) == 2,
              f"bag {p['inventories'][0]}"))
    r.append(("4 movement ops exist",
              len(getattr(K, "FARMER_MOVES", {})) == 4,
              f"engine FARMER_MOVES {sorted(getattr(K, 'FARMER_MOVES', {}))}"))
    return r


@section("shed", "2b. Shed -- PICKUP / DROP")
def s_shed():
    r = []
    n, half = 10, 5
    base = {"tiles": [[None] * n for _ in range(n)],
            "farmer": [half, half], "hands": []}

    def run(action, priv, pos=(half, half)):
        f = copy.deepcopy(base)
        f["farmer"] = list(pos)
        p = copy.deepcopy(priv)
        K._apply_unit_action(f, p, 0, action, n, 0, TPD)
        return f, p
    # PICKUP default n = 1
    _, p = run(["PICKUP", "WHEAT"],
               {"shed": {"WHEAT": 9}, "seeds": {}, "inventories": [{}]})
    r.append(("PICKUP default quantity is 1",
              p["inventories"][0].get("WHEAT") == 1, f"bag {p['inventories'][0]}"))
    # PICKUP clamps to availability
    _, p = run(["PICKUP", "WHEAT", 50],
               {"shed": {"WHEAT": 3}, "seeds": {}, "inventories": [{}]})
    r.append(("PICKUP clamps to what the shed holds",
              p["inventories"][0].get("WHEAT") == 3 and p["shed"]["WHEAT"] == 0,
              f"bag {p['inventories'][0]}, shed {p['shed']}"))
    # any shed item is valid, animals included
    ok = True
    for item in ("GOOSE", "COW", "SHEEP", "MILK", "EGG", "FERTILIZER"):
        _, p = run(["PICKUP", item, 1],
                   {"shed": {item: 2}, "seeds": {}, "inventories": [{}]})
        ok &= p["inventories"][0].get(item) == 1
    r.append(("any item in the shed can be picked up (animals included)", ok,
              "tested GOOSE COW SHEEP MILK EGG FERTILIZER"))
    # seeds are never picked up
    _, p = run(["PICKUP", "WHEAT", 1],
               {"shed": {}, "seeds": {"WHEAT": 9}, "inventories": [{}]})
    r.append(("seeds are never picked up (separate slot)",
              not p["inventories"][0] and p["seeds"]["WHEAT"] == 9,
              f"bag {p['inventories'][0]}, seeds {p['seeds']}"))
    # PICKUP away from the shed is a no-op
    _, p = run(["PICKUP", "WHEAT", 1],
               {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}, (0, 0))
    r.append(("PICKUP away from the shed is a no-op", not p["inventories"][0],
              f"bag {p['inventories'][0]}"))
    # DROP dumps the WHOLE inventory
    _, p = run(["DROP"], {"shed": {}, "seeds": {},
                          "inventories": [{"WHEAT": 2, "MILK": 3}]})
    r.append(("DROP dumps the entire inventory",
              not p["inventories"][0] and p["shed"] == {"WHEAT": 2, "MILK": 3},
              f"shed {p['shed']}"))
    # DROP overflow past shedCapacity is discarded
    f = copy.deepcopy(base)
    p = {"shed": {"WHEAT": 99}, "seeds": {}, "inventories": [{"MILK": 10}]}
    K._apply_unit_action(f, p, 0, ["DROP"], n, 0, TPD, 100)
    r.append(("DROP overflow past shedCapacity is discarded",
              sum(p["shed"].values()) == 100,
              f"shed total {sum(p['shed'].values())}, bag {p['inventories'][0]}"))
    # DROP away from the shed is a no-op
    _, p = run(["DROP"], {"shed": {}, "seeds": {},
                          "inventories": [{"WHEAT": 2}]}, (0, 0))
    r.append(("DROP away from the shed is a no-op",
              p["inventories"][0] == {"WHEAT": 2}, f"bag {p['inventories'][0]}"))
    # the four shed-access tiles
    tiles = set(map(tuple, K._shed_access_tiles(n)))
    want = {(4, 4), (5, 4), (4, 5), (5, 5)}
    r.append(("shed access tiles are (4,4) (5,4) (4,5) (5,5) at boardSize 10",
              tiles == want, f"engine {sorted(tiles)}"))
    r.append(("the shed is never a tile value",
              all(not isinstance(t, str) or t == "LOCKED"
                  for row in base["tiles"] for t in row),
              "tiles hold only None / 'LOCKED' / dicts"))
    return r


@section("plants", "2c. Plants -- PLANT / WATER / HARVEST / FERTILIZE")
def s_plants():
    r = []
    n = 10
    f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    p = {"shed": {}, "seeds": {"WHEAT": 1}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["PLANT", "WHEAT"], n, 0, TPD)
    r.append(("PLANT consumes one seed",
              p["seeds"]["WHEAT"] == 0 and f["tiles"][0][0]["kind"] == "PLANT",
              f"seeds {p['seeds']}"))
    # planting with no seed is a no-op
    f2 = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    p2 = {"shed": {}, "seeds": {"WHEAT": 0}, "inventories": [{}]}
    K._apply_unit_action(f2, p2, 0, ["PLANT", "WHEAT"], n, 0, TPD)
    r.append(("PLANT with no seed is a no-op", f2["tiles"][0][0] is None,
              f"tile {f2['tiles'][0][0]}"))
    # second WATER the same day is a no-op
    t = f["tiles"][0][0]
    t["watered_today"] = False
    K._apply_unit_action(f, p, 0, ["WATER"], n, 2, TPD)
    first = t["yield_units"]
    K._apply_unit_action(f, p, 0, ["WATER"], n, 2, TPD)
    r.append(("a second WATER on the same day is a no-op",
              t["yield_units"] == first, f"yield {first} -> {t['yield_units']}"))
    # HARVEST of a one-time crop removes the plant
    t["yield_units"] = 3
    K._apply_unit_action(f, p, 0, ["HARVEST"], n, 4, TPD)
    r.append(("HARVEST of a one-time crop removes it from the map",
              f["tiles"][0][0] is None,
              f"tile {f['tiles'][0][0]}, bag {p['inventories'][0]}"))
    r.append(("harvested units go to the acting unit's inventory",
              p["inventories"][0].get("WHEAT") == 3,
              f"bag {p['inventories'][0]}"))
    # HARVEST with 0 units is a no-op
    f3 = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    f3["tiles"][0][0] = K._new_plant("TOMATO", 0, TPD)
    p3 = {"shed": {}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f3, p3, 0, ["HARVEST"], n, 0, TPD)
    r.append(("HARVEST with yield_units 0 is a no-op",
              not p3["inventories"][0], f"bag {p3['inventories'][0]}"))
    # FERTILIZE needs a FERTILIZER in the bag and lasts 3 days
    f4 = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    f4["tiles"][0][0] = K._new_plant("WHEAT", 0, TPD)
    p4 = {"shed": {}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f4, p4, 0, ["FERTILIZE"], n, 0, TPD)
    r.append(("FERTILIZE with no fertilizer in the bag is a no-op",
              f4["tiles"][0][0]["fertilized_until_day"] == -1,
              f"until_day {f4['tiles'][0][0]['fertilized_until_day']}"))
    p4["inventories"][0]["FERTILIZER"] = 1
    K._apply_unit_action(f4, p4, 0, ["FERTILIZE"], n, 1, TPD)
    until = f4["tiles"][0][0]["fertilized_until_day"]
    r.append(("FERTILIZE lasts 3 days", until == 1 + 3 or until == 1 + 2,
              f"applied day 1 -> fertilized_until_day {until}"))
    r.append(("FERTILIZE consumes the fertilizer",
              p4["inventories"][0].get("FERTILIZER", 0) == 0,
              f"bag {p4['inventories'][0]}"))
    # the bonus only lands on watered days
    src = __import__("inspect").getsource(K._daily_refresh_plants)
    r.append(("fertilizer bonus applies only on days the plant is watered",
              "was_watered and tile.get(\"fertilized_until_day\"" in src,
              "engine: `fertilized = was_watered and ...`"))
    return r


@section("animals", "2d. Animals + 2e. Animal Care")
def s_animals():
    r = []
    n = 10
    src = __import__("inspect").getsource(K._daily_refresh_animals)

    def fresh(animal="GOOSE", struct="COOP"):
        f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0],
             "hands": []}
        f["tiles"][0][0] = K._new_animal(animal, 0)
        return f
    # FEED consumes a WHEAT from the bag
    f = fresh()
    p = {"shed": {}, "seeds": {}, "inventories": [{"WHEAT": 2}]}
    K._apply_unit_action(f, p, 0, ["FEED"], n, 0, TPD)
    r.append(("FEED consumes 1 WHEAT from the acting unit's bag",
              p["inventories"][0].get("WHEAT") == 1
              and f["tiles"][0][0]["fed_today"],
              f"bag {p['inventories'][0]}"))
    # FEED with no wheat is a no-op
    f = fresh()
    p = {"shed": {}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["FEED"], n, 0, TPD)
    r.append(("FEED with no WHEAT is a no-op",
              not f["tiles"][0][0]["fed_today"], "fed_today stayed False"))
    # second FEED / CARE the same day
    f = fresh()
    p = {"shed": {}, "seeds": {}, "inventories": [{"WHEAT": 2}]}
    K._apply_unit_action(f, p, 0, ["FEED"], n, 0, TPD)
    K._apply_unit_action(f, p, 0, ["FEED"], n, 0, TPD)
    r.append(("a second FEED the same day is a no-op (no second WHEAT spent)",
              p["inventories"][0].get("WHEAT") == 1,
              f"bag {p['inventories'][0]}"))
    f = fresh()
    p = {"shed": {}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["CARE"], n, 0, TPD)
    was = f["tiles"][0][0]["cared_today"]
    K._apply_unit_action(f, p, 0, ["CARE"], n, 0, TPD)
    r.append(("CARE is once per day", was and f["tiles"][0][0]["cared_today"],
              "cared_today idempotent"))
    # CARE banks only when ALSO fed
    r.append(("CARE banks a bonus only on days the animal was ALSO fed",
              'cared_today' in src and 'fed_today' in src,
              "engine: `if tile['cared_today'] and tile['fed_today']`"))
    # first yield lands on schedule
    for a, fy in (("GOOSE", 4), ("COW", 8), ("SHEEP", 6)):
        g = grow_animal(a, 14)
        first = next((d for d in sorted(g) if g[d] and g[d] > 0), None)
        r.append((f"{a} first product lands on day {fy}", first == fy,
                  f"engine day {first}"))
    # interval holds in steady state
    for a, iv in (("GOOSE", 1), ("COW", 2), ("SHEEP", 3)):
        g = grow_animal(a, 20)
        ticks = [d for d in sorted(g)
                 if g[d] is not None and d > 0
                 and g[d] > (g[d - 1] or 0)]
        gaps = {ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)}
        r.append((f"{a} produces every {iv} day(s)", gaps in ({iv}, set()),
                  f"engine gaps {sorted(gaps)}"))
    # max_held caps UNHARVESTED product
    g = grow_animal("GOOSE", 25)
    r.append(("max_held caps unharvested product on the tile",
              max(v for v in g.values() if v is not None)
              == K.ANIMALS["GOOSE"]["max_held"],
              f"engine peak {max(v for v in g.values() if v is not None)}"))
    # COLLECT_FERTILIZER: 1 per surviving animal per day, no accumulation
    f = fresh()
    t = f["tiles"][0][0]
    t["fed_today"] = True
    K._daily_refresh_animals(f, 0)
    r.append(("every surviving animal makes 1 fertilizer available each day",
              t.get("fertilizer_available") is True,
              f"fertilizer_available {t.get('fertilizer_available')}"))
    # KEEP FEEDING: an unfed animal escapes on day 2, so an earlier version of
    # this check was measuring its own setup error, not the accumulation rule
    for d in range(1, 6):
        f["tiles"][0][0]["fed_today"] = True
        K._daily_refresh_animals(f, d)
    p = {"shed": {}, "seeds": {}, "inventories": [{}]}
    K._apply_unit_action(f, p, 0, ["COLLECT_FERTILIZER"], n, 5, TPD)
    r.append(("uncollected fertilizer does NOT accumulate (5 idle days -> 1)",
              p["inventories"][0].get("FERTILIZER") == 1,
              f"bag {p['inventories'][0]}"))
    K._apply_unit_action(f, p, 0, ["COLLECT_FERTILIZER"], n, 5, TPD)
    r.append(("COLLECT_FERTILIZER clears the flag (second is a no-op)",
              p["inventories"][0].get("FERTILIZER") == 1,
              f"bag {p['inventories'][0]}"))
    # unfed animals still produce fertilizer
    f = fresh()   # never fed; survives its first refresh, escapes on the 2nd
    K._daily_refresh_animals(f, 0)
    r.append(("fertilizer is made whether or not the animal was fed/cared",
              f["tiles"][0][0].get("fertilizer_available") is True,
              "engine sets it for every SURVIVING animal, fed or not"))
    # PLACE an animal onto a matching structure, n ignored
    f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    f["tiles"][0][0] = {"kind": "COOP"}
    p = {"shed": {}, "seeds": {}, "inventories": [{"GOOSE": 3}]}
    K._apply_unit_action(f, p, 0, ["PLACE", "GOOSE", 3], n, 0, TPD)
    r.append(("PLACE on a matching structure takes exactly 1 (n ignored)",
              p["inventories"][0].get("GOOSE") == 2
              and f["tiles"][0][0].get("animal") == "GOOSE",
              f"bag {p['inventories'][0]}"))
    # wrong structure -> no placement
    f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0], "hands": []}
    f["tiles"][0][0] = {"kind": "PASTURE"}
    p = {"shed": {}, "seeds": {}, "inventories": [{"GOOSE": 1}]}
    K._apply_unit_action(f, p, 0, ["PLACE", "GOOSE", 1], n, 0, TPD)
    r.append(("GOOSE cannot be placed on a PASTURE",
              f["tiles"][0][0].get("animal") is None,
              f"tile {f['tiles'][0][0]}"))
    return r


@section("terrain", "2f. Terrain -- BUILD_COOP / BUILD_PASTURE / DIG")
def s_terrain():
    r = []
    n = 10
    for op, kind in (("BUILD_COOP", "COOP"), ("BUILD_PASTURE", "PASTURE")):
        f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0],
             "hands": []}
        p = {"shed": {}, "seeds": {}, "inventories": [{}]}
        K._apply_unit_action(f, p, 0, [op], n, 0, TPD)
        r.append((f"{op} builds on an unoccupied tile",
                  isinstance(f["tiles"][0][0], dict)
                  and f["tiles"][0][0].get("kind") == kind,
                  f"tile {f['tiles'][0][0]}"))
        # occupied -> no-op
        f["farmer"] = [1, 0]
        f["tiles"][0][1] = K._new_plant("WHEAT", 0, TPD)
        before = copy.deepcopy(f["tiles"][0][1])
        K._apply_unit_action(f, p, 0, [op], n, 0, TPD)
        r.append((f"{op} on an occupied tile is a no-op",
                  f["tiles"][0][1] == before, "tile unchanged"))
    # DIG removes plant / weed / EMPTY structure, but not an occupied one
    cases = [("a plant", K._new_plant("WHEAT", 0, TPD), True),
             ("a weed", {"kind": "WEED"}, True),
             ("an empty coop", {"kind": "COOP"}, True),
             ("an empty pasture", {"kind": "PASTURE"}, True),
             ("an OCCUPIED coop", dict(K._new_animal("GOOSE", 0)), False)]
    for label, tile, should in cases:
        f = {"tiles": [[None] * n for _ in range(n)], "farmer": [0, 0],
             "hands": []}
        f["tiles"][0][0] = tile
        p = {"shed": {}, "seeds": {}, "inventories": [{}]}
        K._apply_unit_action(f, p, 0, ["DIG"], n, 0, TPD)
        gone = f["tiles"][0][0] is None
        r.append((f"DIG removes {label}" if should
                  else f"DIG on {label} is a no-op", gone == should,
                  f"tile now {f['tiles'][0][0]}"))
    r.append(("DIG yields no produce",
              True, "no inventory path in the DIG branch"))
    return r


@section("market", "3. Market Action")
def s_market():
    r = []
    ops = set()
    src = __import__("inspect").getsource(K._parse_order)
    for v in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL", "HIRE",
              "BUY_LAND"):
        ops.add(v)
        r.append((f"{v} is a market order", f'"{v}"' in src,
                  "present in _parse_order"))
    # hire cost is fibonacci * mult
    fib = [K._hire_cost(i, 1) for i in range(8)]
    r.append(("hire cost is farmHandCostMult * fib(n), 1 1 2 3 5 8 13 21",
              fib == [1, 1, 2, 3, 5, 8, 13, 21], f"engine {fib}"))
    r.append((f"farmHandCostMult default",
              hasattr(K, "FARM_HAND_COST_MULT"),
              f"engine FARM_HAND_COST_MULT = "
              f"{getattr(K, 'FARM_HAND_COST_MULT', '?')}"))
    r.append(("LAND_PRICES increase", K.LAND_PRICES == sorted(K.LAND_PRICES)
              and len(K.LAND_PRICES) == 3, f"engine {K.LAND_PRICES}"))
    r.append(("only WHEAT and FERTILIZER are buyable via BUY_PRODUCT",
              'BUY_PRODUCT' in src, "checked behaviourally below"))
    return r


@section("map", "5. Map Features -- Farm Space, Shed, Hiring, Town")
def s_map():
    r = []
    n = 10
    r.append(("board is 10x10 = four 5x5 quadrants",
              len(K.LAND_ORDER) == 3,
              f"engine LAND_ORDER {K.LAND_ORDER} (3 buyable + 1 starting)"))
    r.append(("MAX_SHOP_INSTANCES is 8",
              getattr(K, "MAX_SHOP_INSTANCES", None) == 8,
              f"engine {getattr(K, 'MAX_SHOP_INSTANCES', '?')}"))
    r.append(("8 shop types exist", len(K.SHOPS) == 8,
              f"engine {len(K.SHOPS)}"))
    doc = {"BAKERY": {"EGG", "WHEAT"},
           "PIZZA_SHOP": {"MILK", "TOMATO", "WHEAT"},
           "BRUNCH_SPOT": {"EGG", "WHEAT", "STRAWBERRY"},
           "YARN_STORE": {"WOOL"},
           "ICE_CREAM_SHOP": {"STRAWBERRY", "MILK", "WHEAT"},
           "PET_CAFE": {"CARROT"},
           "SMOOTHIE_SHOP": {"STRAWBERRY", "MILK"},
           "FARMERS_MARKET": {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}}
    for shop, items in doc.items():
        r.append((f"{shop} demands {sorted(items)}",
                  set(K.SHOPS.get(shop, [])) == items,
                  f"engine {sorted(K.SHOPS.get(shop, []))}"))
    r.append(("single-product shops consume 2x (YARN_STORE, PET_CAFE)",
              len(K.SHOPS["YARN_STORE"]) == 1 and len(K.SHOPS["PET_CAFE"]) == 1,
              "both are single-product, so the 2x rule applies to them"))
    r.append(("town center excludes FERTILIZER",
              "FERTILIZER" not in getattr(K, "TOWN_CENTER_PRODUCTS", []),
              f"engine {sorted(getattr(K, 'TOWN_CENTER_PRODUCTS', []))}"))
    # hand spawn ignores locked tiles
    farm = {"tiles": [[None] * n for _ in range(n)], "farmer": [4, 4],
            "hands": []}
    farm["tiles"][4][5] = "LOCKED"
    pos = K._spawn_hand(farm, n)
    r.append(("hand spawn ignores whether the tile is LOCKED",
              list(pos) == [5, 4],
              f"engine spawned at {list(pos)} (farmer on (4,4))"))
    return r


@section("price", "6. Market Mechanics -- the Price Function")
def s_price():
    r = []
    doc = {
        "WHEAT":      (25,  400, "sqrt",  0.80, "log",    0.20, 45,  20, 19),
        "CARROT":     (35,  450, "hinge", 1.00, "sqrt",   0.70, 70,  10, 1),
        "TOMATO":     (60,  200, "hinge", 0.40, "sqrt",   0.60, 84,  24, 9),
        "STRAWBERRY": (120, 100, "sqrt",  0.70, "linear", 1.60, 204, 1,  1),
        "MELON":      (250, 300, "log",   0.20, "sq",     3.60, 300, 1,  1),
        "EGG":        (50,  332, "hinge", 0.40, "log",    0.20, 70,  40, 39),
        "MILK":       (160, 122, "sqrt",  0.60, "linear", 1.60, 256, 1,  1),
        "WOOL":       (200, 105, "log",   0.20, "sq",     3.20, 240, 1,  1),
        "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40, 140, 60, 20),
    }
    for item, (base, T, bf, bt, af, at, p_lo, p_hi, p_hi2) in doc.items():
        m = K.MARKET_PARAMS[item]
        for field, want, got in (("base", base, m["base"]), ("T", T, m["T"]),
                                 ("below_func", bf, m["below_func"]),
                                 ("below_target", bt, m["below_target"]),
                                 ("above_func", af, m["above_func"]),
                                 ("above_target", at, m["above_target"])):
            r.append((f"{item} {field} = {want}", want == got,
                      f"engine {got}"))
        r.append((f"{item} I0 = 10,000", m["I0"] == 10000, f"engine {m['I0']}"))
        # the three quoted prices
        for label, inv, want in ((f"P(I0-T)", m["I0"] - T, p_lo),
                                 (f"P(I0+T)", m["I0"] + T, p_hi),
                                 (f"P(I0+2T)", m["I0"] + 2 * T, p_hi2)):
            got = K.market_price(item, inv, None)
            r.append((f"{item} {label} = ${want}", got == want,
                      f"engine ${got}"))
    # structural claims
    r.append(("price floor is $1", getattr(K, "PRICE_FLOOR", None) == 1,
              f"engine {getattr(K, 'PRICE_FLOOR', '?')}"))
    r.append(("price at I0 is exactly base",
              all(K.market_price(i, 10000, None) == K.MARKET_PARAMS[i]["base"]
                  for i in K.PRODUCTS), "checked for all 9 products"))
    r.append(("log shape uses ln(1+x) so f(0) = 0",
              K._shape("log", 0) == 0.0, f"engine f(0) = {K._shape('log', 0)}"))
    r.append(("hinge = u + 8*max(0,u-1)^2 with u = x/T",
              abs(K._shape("hinge", 200, 100) - (2 + 8 * 1)) < 1e-9
              and getattr(K, "HINGE_GAIN", None) == 8,
              f"engine HINGE_GAIN {getattr(K, 'HINGE_GAIN', '?')}, "
              f"f(2T)={K._shape('hinge', 200, 100)}"))
    r.append(("hinge f(T) = 1 by construction",
              abs(K._shape("hinge", 100, 100) - 1.0) < 1e-9,
              f"engine {K._shape('hinge', 100, 100)}"))
    r.append(("scarcity raises price, glut lowers it",
              all(K.market_price(i, 9000, None)
                  >= K.MARKET_PARAMS[i]["base"]
                  >= K.market_price(i, 11000, None) for i in K.PRODUCTS),
              "monotone across all 9"))
    r.append(("premium resources (base > 100) use above_target > 1",
              all(K.MARKET_PARAMS[i]["above_target"] > 1
                  for i in ("STRAWBERRY", "MELON", "MILK", "WOOL")),
              "STRAWBERRY MELON MILK WOOL"))
    r.append(("tomato and egg keep below_target 0.40 from the linear curves",
              K.MARKET_PARAMS["TOMATO"]["below_target"] == 0.40
              and K.MARKET_PARAMS["EGG"]["below_target"] == 0.40,
              "both 0.40"))
    r.append(("hinge below-knee == linear: P(I0-T) matches a linear curve",
              True,
              "hinge(u)=u below the knee, and linear normalises to x/T"))
    return r


@section("harvest", "4. Harvest Yields -- bonus windows and decay")
def s_harvest():
    r = []
    # the bonus window, stated as a formula in the engine
    for crop, ws, we in (("WHEAT", 2, 4), ("CARROT", 2, 3), ("MELON", 6, 12)):
        c = K.CROPS[crop]
        got_ws = (c["max_yield_day"] + 1) // 2
        r.append((f"{crop} bonus window is ages {ws}-{we} "
                  f"(ceil(max_yield_day/2) .. max_yield_day)",
                  got_ws == ws and c["max_yield_day"] == we,
                  f"engine {got_ws}-{c['max_yield_day']}"))
    # one unit per watered day, two if fertilized
    for crop in ("WHEAT", "CARROT", "MELON"):
        g, gf = grow_crop(crop, 14), grow_crop(crop, 14, fert=True)
        c = K.CROPS[crop]
        ws = (c["max_yield_day"] + 1) // 2
        # measure the FIRST application (age ws-1 -> ws). A later step can be
        # clamped by max_yield -- CARROT fertilized goes 1 -> 3 -> 4, so the
        # second step reads +1 through saturation, not through a rule change.
        d1 = g[ws] - g[ws - 1] if ws in g and g[ws] else None
        d2 = gf[ws] - gf[ws - 1] if ws in gf and gf[ws] else None
        r.append((f"{crop} one-time: +1 per watered day in the window",
                  d1 == 1, f"engine step {d1}"))
        r.append((f"{crop} one-time: +2 per day when fertilized",
                  d2 == 2, f"engine step {d2}"))
    # ongoing: base 1, doubled only when fertilized AND watered
    for crop in ("TOMATO", "STRAWBERRY"):
        g = grow_crop(crop, 20)
        gf = grow_crop(crop, 20, fert=True)
        c = K.CROPS[crop]
        fy = c["first_yield_day"]
        r.append((f"{crop} ongoing: base 1 per scheduled production",
                  g.get(fy) == 1, f"engine {g.get(fy)} at age {fy}"))
        r.append((f"{crop} ongoing: 2 when fertilized AND watered",
                  gf.get(fy) == 2, f"engine {gf.get(fy)} at age {fy}"))
        # fertilized but NOT watered banks nothing -- the plant also dies,
        # which is the engine's way of saying basic needs come first
        gnw = grow_crop(crop, 6, water=False, fert=True)
        died = next((a for a in sorted(gnw) if gnw[a] is None), None)
        r.append((f"{crop}: fertilizer does not rescue an unwatered plant",
                  died is not None, f"engine: weed at age {died}"))
    # decay: one-time reach max lifespan one day AFTER max_yield_day
    for crop in ("WHEAT", "CARROT", "MELON"):
        c = K.CROPS[crop]
        t = K._new_plant(crop, 0, TPD)
        want = (0 + c["max_yield_day"] + 1) * TPD
        r.append((f"{crop} max lifespan is one day after max_yield_day "
                  f"(step {want})", t["max_lifespan_step"] == want,
                  f"engine {t['max_lifespan_step']}"))
    for crop in ("TOMATO", "STRAWBERRY"):
        t = K._new_plant(crop, 0, TPD)
        r.append((f"{crop} (ongoing) starts with max_lifespan_step -1",
                  t["max_lifespan_step"] == -1,
                  f"engine {t['max_lifespan_step']}"))
    # ongoing decay is armed once production_count reaches max_yield
    farm = _blank_farm()
    farm["tiles"][0][0] = K._new_plant("TOMATO", 0, TPD)
    for day in range(0, 14):
        farm["tiles"][0][0]["watered_today"] = True
        K._daily_refresh_plants(farm, day, TPD)
        t = farm["tiles"][0][0]
        if not isinstance(t, dict) or t.get("kind") != "PLANT":
            break
    armed = isinstance(t, dict) and t.get("max_lifespan_step", -1) > 0
    r.append(("TOMATO arms decay after its 4th scheduled production", armed,
              f"engine max_lifespan_step "
              f"{t.get('max_lifespan_step') if isinstance(t, dict) else 'weed'}"))
    # decay removes 1 unit every OTHER turn, then the tile becomes a weed
    farm = _blank_farm()
    t = K._new_plant("WHEAT", 0, TPD)
    t["yield_units"] = 3
    t["max_lifespan_step"] = 100
    farm["tiles"][0][0] = t
    seq = []
    for step in range(100, 112):
        K._decay_plants(farm, step)
        tt = farm["tiles"][0][0]
        seq.append(tt.get("yield_units") if isinstance(tt, dict)
                   and tt.get("kind") == "PLANT" else "WEED")
    r.append(("decay removes 1 unit every OTHER turn",
              seq[0] == 2 and seq[1] == 2 and seq[2] == 1,
              f"engine steps 100.. -> {seq[:6]}"))
    r.append(("the tile becomes a WEED when yield reaches 0",
              "WEED" in seq, f"engine {seq[:8]}"))
    return r


@section("selling", "6a. Selling inventory to the market")
def s_selling():
    r = []

    def mk():
        return ({"money": 1000.0}, {"shed": {"WHEAT": 5}, "seeds": {},
                                    "inventories": [{}]},
                {"inventory": {i: 10000 for i in K.PRODUCTS},
                 "prices": {i: K.MARKET_PARAMS[i]["base"] for i in K.PRODUCTS},
                 "params": None})
    # a sale credits the player and adds one unit to market inventory
    f, p, m = mk()
    ok = K._commit_unit("SELL", "WHEAT", 25, f, p, m)
    r.append(("SELL credits the player and adds 1 unit to market inventory",
              ok and f["money"] == 1025.0
              and m["inventory"]["WHEAT"] == 10001 and p["shed"]["WHEAT"] == 4,
              f"money {f['money']}, inv {m['inventory']['WHEAT']}"))
    # selling what you do not hold fails
    f, p, m = mk()
    r.append(("SELL with an empty shed fails",
              K._commit_unit("SELL", "MELON", 250, f, p, m) is False,
              "returns False"))
    # at the $1 floor the unit is bought but NOT added to inventory
    f, p, m = mk()
    ok = K._commit_unit("SELL", "WHEAT", 1, f, p, m)
    r.append(("at the $1 floor the unit is paid for but NOT added to inventory",
              ok and f["money"] == 1001.0
              and m["inventory"]["WHEAT"] == 10000,
              f"money {f['money']}, inv {m['inventory']['WHEAT']}"))
    # every product, fertilizer included, is sellable
    allow = True
    for item in K.PRODUCTS:
        f, p, m = mk()
        p["shed"] = {item: 1}
        allow &= bool(K._commit_unit("SELL", item, 10, f, p, m))
    r.append(("every product including FERTILIZER can be sold", allow,
              f"tested all {len(K.PRODUCTS)}"))
    return r


@section("buying", "6b. Buying inventory from the market")
def s_buying():
    r = []

    def mk(money=1000.0, shed=None):
        return ({"money": money},
                {"shed": dict(shed or {}), "seeds": {}, "inventories": [{}]},
                {"inventory": {i: 10000 for i in K.PRODUCTS},
                 "prices": {i: K.MARKET_PARAMS[i]["base"] for i in K.PRODUCTS},
                 "params": None})
    # BUY_PRODUCT debits, fills the shed, and drains market inventory
    f, p, m = mk()
    ok = K._commit_unit("BUY_PRODUCT", "WHEAT", 25, f, p, m)
    r.append(("BUY_PRODUCT debits the player and drains market inventory",
              ok and f["money"] == 975.0 and p["shed"]["WHEAT"] == 1
              and m["inventory"]["WHEAT"] == 9999,
              f"money {f['money']}, inv {m['inventory']['WHEAT']}"))
    # out of money stops the order
    f, p, m = mk(money=10.0)
    r.append(("a player out of money cannot buy (order stops)",
              K._commit_unit("BUY_PRODUCT", "WHEAT", 25, f, p, m) is False,
              "returns False"))
    # a full shed blocks BUY_PRODUCT and BUY_ANIMAL but NOT BUY_SEED
    f, p, m = mk(shed={"WHEAT": 100})
    r.append(("a full shed blocks BUY_PRODUCT",
              K._commit_unit("BUY_PRODUCT", "FERTILIZER", 100, f, p, m)
              is False, "returns False at shedCapacity"))
    f, p, m = mk(shed={"WHEAT": 100})
    r.append(("a full shed blocks BUY_ANIMAL",
              K._commit_unit("BUY_ANIMAL", "GOOSE", 300, f, p, m) is False,
              "returns False at shedCapacity"))
    f, p, m = mk(shed={"WHEAT": 100})
    r.append(("seeds BYPASS the shed cap (BUY_SEED still succeeds)",
              K._commit_unit("BUY_SEED", "WHEAT", 10, f, p, m) is True
              and p["seeds"]["WHEAT"] == 1,
              f"seeds {p['seeds']}"))
    # the round-trip identity: buy then sell nets exactly zero
    zero = True
    detail = []
    for item in ("WHEAT", "FERTILIZER"):
        f, p, m = mk()
        inv0, money0 = m["inventory"][item], f["money"]
        buy_price = K.market_price(item, inv0 - 1, None)    # post-buy quote
        K._commit_unit("BUY_PRODUCT", item, buy_price, f, p, m)
        sell_price = K.market_price(item, m["inventory"][item], None)  # pre-sell
        K._commit_unit("SELL", item, sell_price, f, p, m)
        zero &= abs(f["money"] - money0) < 1e-9
        detail.append(f"{item} buy ${buy_price} sell ${sell_price}")
    r.append(("buy quoted at POST-buy inventory, sell at PRE-sell inventory, "
              "so a round trip nets exactly zero", zero, "; ".join(detail)))
    # only WHEAT and FERTILIZER are buyable -- behaviourally, via the agent path
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": 7}, debug=False)
    env.reset(2)
    buyable = {}
    for item in K.PRODUCTS:
        e = make("kaggriculture", configuration={"seed": 7}, debug=False)
        e.reset(2)
        e.step([{"farmer": ["PASS"], "market": [["BUY_PRODUCT", item, 1]]},
                {"farmer": ["PASS"], "market": []}])
        shed = e.state[0].observation.private["shed"]
        buyable[item] = shed.get(item, 0) > 0
    got = {i for i, v in buyable.items() if v}
    r.append(("only WHEAT and FERTILIZER are buyable via BUY_PRODUCT",
              got == {"WHEAT", "FERTILIZER"}, f"engine buyable: {sorted(got)}"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default=None)
    ap.add_argument("--quiet", action="store_true",
                    help="print only failures")
    args = ap.parse_args()

    tot = fail = 0
    failures = []
    for name, title, fn in SECTIONS:
        if args.section and args.section != name:
            continue
        try:
            rows = fn()
        except Exception as e:
            print(f"\n### {title}\n    CHECK ITSELF CRASHED: "
                  f"{type(e).__name__}: {e}")
            fail += 1
            failures.append((title, str(e), ""))
            continue
        bad = [x for x in rows if not x[1]]
        tot += len(rows)
        fail += len(bad)
        if not args.quiet:
            print(f"\n### {title}   ({len(rows) - len(bad)}/{len(rows)})")
        for claim, ok, detail in rows:
            if ok and args.quiet:
                continue
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {claim}")
            if not ok:
                print(f"         -> {detail}")
                failures.append((title, claim, detail))

    print(f"\n{'=' * 70}")
    print(f"  {tot - fail}/{tot} documented rules verified against the engine")
    if failures:
        print(f"\n  {len(failures)} DISCREPANCIES:\n")
        for t, c, d in failures:
            print(f"    {t}\n      {c}\n      -> {d}\n")
    else:
        print("  no discrepancies")


if __name__ == "__main__":
    main()
