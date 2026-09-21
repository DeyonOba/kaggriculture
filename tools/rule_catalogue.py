"""One test rule per README action: precondition, acceptance, and payment.

Every rule is a direct reading of `README.md`, re-verified against the engine
in `docs/rules_verified.md`. Three tiers, scored independently:

  T1 emitted   our executor produced the op where the precondition held
  T2 accepted  the engine did NOT silently no-op it (right tile, right
               preconditions, item actually in the bag)
  T3 paid      the README rule's VALUE materialised -- the yield unit landed,
               the care bonus banked, the seed survived the night, the fill
               was above the floor. T2 without T3 is the whole class of bug
               this project keeps shipping.

A rule's `precond` reads a single engine record (the tile and bags either side
of one op), so the same predicate can be evaluated on the reference seat's
action (to mine a case) and on ours (to score capability on any tile).

`ctx` gives the payment check the rest of the world: whole-grid snapshots for
the settle window, the post-refresh grid for each day, and every op our seat
emitted.
"""
from kagent.constants import ANIMALS, CROPS

TPD = 24
ANIMAL_KINDS = ("GOOSE", "COW", "SHEEP")
STRUCT_OF = {k: v["structure"] for k, v in ANIMALS.items()}


# ---------------------------------------------------------------- helpers ---
def _plant(rec):
    t = rec["tile_before"]
    return t if isinstance(t, dict) and t.get("kind") == "PLANT" else None


def _animal(rec):
    t = rec["tile_before"]
    return t if isinstance(t, dict) and "animal" in t else None


def _age(rec, t):
    return rec["day"] - t["planted_day"]


def _window(cd):
    """README Harvest Yields: bonus starts at half `max_yield_day`, rounded up."""
    return (cd["max_yield_day"] + 1) // 2, cd["max_yield_day"]


def _produces_tonight(rec, t, cd):
    """Ongoing crop: is tonight's refresh a scheduled production?"""
    since = rec["day"] + 1 - t["planted_day"] - cd["first_yield_day"]
    if since < 0 or cd["interval"] <= 0 or since % cd["interval"]:
        return False
    return since // cd["interval"] + 1 <= cd["max_yield"]


def _animal_produces_tonight(rec, t):
    a = ANIMALS[t["animal"]]
    since = rec["day"] + 1 - t["placed_day"] - a["first_yield_day"]
    return since >= 0 and since % a["interval"] == 0


def _tile_after_eod(ctx, day, x, y):
    grid = ctx["eod_grid"].get(day)
    return None if grid is None else grid[y][x]


def _delta(before, after, item):
    return after.get(item, 0) - before.get(item, 0)


def _later_ops(ctx, rec, ops=None, x=None, y=None, until=None):
    """Our seat's ops after this record, optionally filtered by op and tile."""
    out = []
    for r in ctx["our_ops"]:
        if r["step"] <= rec["step"]:
            continue
        if until is not None and r["step"] > until:
            continue
        if ops and r["op"][0] not in ops:
            continue
        if x is not None and (r["x"], r["y"]) != (x, y):
            continue
        out.append(r)
    return out


# ------------------------------------------------------------------ rules ---
class Rule:
    def __init__(self, name, op, readme, precond, accepted, paid,
                 settle_days=0, arg=None, kind="unit"):
        self.name = name
        self.op = op
        self.arg = arg           # required op[1] (crop / animal / item), or None
        self.kind = kind         # "unit" | "market"
        self.readme = readme     # the sentence this rule is testing
        self.precond = precond
        self.accepted = accepted
        self.paid = paid
        self.settle_days = settle_days   # extra days our executor keeps driving

    def matches(self, rec):
        return rec["op"][0] == self.op and (
            self.arg is None or (len(rec["op"]) > 1 and rec["op"][1] == self.arg))


RULES = []


def rule(*a, **kw):
    RULES.append(Rule(*a, **kw))


# -- WATER --------------------------------------------------------------------
def _water_bonus_pre(rec):
    t = _plant(rec)
    if not t or t.get("watered_today"):
        return False
    cd = CROPS.get(t["crop"])
    if not cd or cd["ongoing"]:
        return False
    lo, hi = _window(cd)
    # README: watering adds a unit only INSIDE the window and only while the
    # plant is under its cap -- a melon caps at age 10 though the window runs
    # to 12, so ages 11-12 are worth nothing.
    return lo <= _age(rec, t) <= hi and t.get("yield_units", 0) < cd["max_yield"]


rule("WATER.bonus_window", "WATER",
     "Harvest Yields: one-time crops gain one unit per watered day inside the "
     "bonus window, capped at max_yield",
     _water_bonus_pre,
     lambda rec: bool(rec["tile_after"] and rec["tile_after"].get("watered_today")),
     lambda rec, ctx: (rec["tile_after"].get("yield_units", 0)
                       > rec["tile_before"].get("yield_units", 0)))


def _water_survival_pre(rec):
    t = _plant(rec)
    return bool(t and not t.get("watered_today")
                and t.get("consecutive_unwatered", 0) >= 1)


rule("WATER.survival", "WATER",
     "Watering: two successive unwatered days turn the plant into a WEED at "
     "end of day",
     _water_survival_pre,
     lambda rec: bool(rec["tile_after"] and rec["tile_after"].get("watered_today")),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and t.get("kind") == "PLANT")(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


def _water_fert_ongoing_pre(rec):
    t = _plant(rec)
    if not t or t.get("watered_today"):
        return False
    cd = CROPS.get(t["crop"])
    if not cd or not cd["ongoing"]:
        return False
    return (t.get("fertilized_until_day", -1) >= rec["day"]
            and _produces_tonight(rec, t, cd)
            and t.get("yield_units", 0) < cd["max_yield"])


rule("WATER.fertilised_production", "WATER",
     "Harvest Yields: an ongoing crop's scheduled yield doubles to 2 only if "
     "it is fertilised AND watered that day",
     _water_fert_ongoing_pre,
     lambda rec: bool(rec["tile_after"] and rec["tile_after"].get("watered_today")),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and t.get("kind") == "PLANT"
                       and t.get("yield_units", 0)
                       - rec["tile_after"].get("yield_units", 0) >= 2)(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


# -- FERTILIZE ----------------------------------------------------------------
def _fert_ongoing_pre(rec):
    t = _plant(rec)
    if not t or rec["inv_before"].get("FERTILIZER", 0) < 1:
        return False
    cd = CROPS.get(t["crop"])
    if not cd or not cd["ongoing"]:
        return False
    return (t.get("fertilized_until_day", -1) < rec["day"]
            and t.get("yield_units", 0) < cd["max_yield"])


rule("FERTILIZE.ongoing", "FERTILIZE",
     "Plants: FERTILIZE doubles the per-day yield bonus for 3 days, and the "
     "bonus only applies on days the plant is also watered",
     _fert_ongoing_pre,
     lambda rec: (rec["tile_after"].get("fertilized_until_day", -1) >= rec["day"]
                  and _delta(rec["inv_before"], rec["inv_after"], "FERTILIZER") == -1),
     # paid only if a later production actually doubled, which needs watering
     lambda rec, ctx: any(
         isinstance(_tile_after_eod(ctx, d, rec["x"], rec["y"]), dict)
         and _tile_after_eod(ctx, d, rec["x"], rec["y"]).get("kind") == "PLANT"
         and _tile_after_eod(ctx, d, rec["x"], rec["y"]).get("yield_units", 0)
         - (ctx["yield_at"](d, rec["x"], rec["y"]) or 0) >= 2
         for d in range(rec["day"], rec["day"] + 3)),
     settle_days=3)


def _fert_onetime_pre(rec):
    t = _plant(rec)
    if not t or rec["inv_before"].get("FERTILIZER", 0) < 1:
        return False
    cd = CROPS.get(t["crop"])
    if not cd or cd["ongoing"]:
        return False
    lo, hi = _window(cd)
    return (t.get("fertilized_until_day", -1) < rec["day"]
            and lo <= _age(rec, t) <= hi
            and t.get("yield_units", 0) < cd["max_yield"])


rule("FERTILIZE.one_time", "FERTILIZE",
     "Harvest Yields: a fertilised one-time crop adds 2 units per watered day "
     "instead of 1",
     _fert_onetime_pre,
     lambda rec: (rec["tile_after"].get("fertilized_until_day", -1) >= rec["day"]
                  and _delta(rec["inv_before"], rec["inv_after"], "FERTILIZER") == -1),
     lambda rec, ctx: any(
         r["tile_after"].get("yield_units", 0) - r["tile_before"].get("yield_units", 0) >= 2
         for r in _later_ops(ctx, rec, ("WATER",), rec["x"], rec["y"])
         if isinstance(r["tile_after"], dict)),
     settle_days=2)


# -- HARVEST ------------------------------------------------------------------
def _harvest_onetime_pre(rec):
    t = _plant(rec)
    if not t:
        return False
    cd = CROPS.get(t["crop"])
    if not cd or cd["ongoing"]:
        return False
    age = _age(rec, t)
    if age < cd["first_yield_day"] or t.get("yield_units", 0) <= 0:
        return False
    # ripe: at the cap, or the window has closed and more watering cannot help
    return t["yield_units"] >= cd["max_yield"] or age >= cd["max_yield_day"]


rule("HARVEST.one_time_ripe", "HARVEST",
     "Plants: HARVEST of a one-time crop removes it and the units go to the "
     "acting unit's bag; decay starts one day after max_yield_day",
     _harvest_onetime_pre,
     lambda rec: (rec["tile_after"] is None
                  and _delta(rec["inv_before"], rec["inv_after"],
                             rec["tile_before"]["crop"])
                  == rec["tile_before"].get("yield_units", 0)),
     # paid = harvested before decay had eaten any of it
     lambda rec, ctx: (rec["tile_before"].get("max_lifespan_step", -1) < 0
                       or rec["step"] < rec["tile_before"]["max_lifespan_step"]))


def _harvest_ongoing_pre(rec):
    t = _plant(rec)
    if not t:
        return False
    cd = CROPS.get(t["crop"])
    if not cd or not cd["ongoing"]:
        return False
    # README: yield_units is capped at max_yield, so a tile at the cap loses
    # the next scheduled production outright
    return (t.get("yield_units", 0) >= cd["max_yield"] - 1
            and _produces_tonight(rec, t, cd))


rule("HARVEST.ongoing_before_clip", "HARVEST",
     "Harvest Yields: an ongoing crop's yield_units are capped at max_yield, "
     "so an unharvested tile at the cap loses tonight's production",
     _harvest_ongoing_pre,
     lambda rec: (isinstance(rec["tile_after"], dict)
                  and rec["tile_after"].get("yield_units", 0) == 0),
     lambda rec, ctx: (lambda t: isinstance(t, dict)
                       and t.get("yield_units", 0) > 0)(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


def _harvest_animal_pre(rec):
    t = _animal(rec)
    if not t or t.get("yield_units", 0) <= 0:
        return False
    a = ANIMALS[t["animal"]]
    if not _animal_produces_tonight(rec, t):
        return False
    # tonight's base 1 + the banked care bonus will clip against max_held
    incoming = 1 + (t.get("pending_care_bonus", 0) if t.get("fed_today") else 0)
    return t["yield_units"] + incoming > a["max_held"]


rule("HARVEST.animal_before_clip", "HARVEST",
     "Object Types: max_held caps UNHARVESTED product on the tile, and Animal "
     "Care: the banked bonus is spent on the production even when it clips",
     _harvest_animal_pre,
     lambda rec: (isinstance(rec["tile_after"], dict)
                  and rec["tile_after"].get("yield_units", 0) == 0
                  and _delta(rec["inv_before"], rec["inv_after"],
                             ANIMALS[rec["tile_before"]["animal"]]["product"]) > 0),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and "animal" in t
                       and t.get("yield_units", 0)
                       <= ANIMALS[t["animal"]]["max_held"] - 1)(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


# -- ANIMALS ------------------------------------------------------------------
def _care_pre(rec):
    t = _animal(rec)
    return bool(t and not t.get("cared_today"))


rule("CARE.banks_a_bonus", "CARE",
     "Animal Care: at end of day, if the animal was both fed AND cared for, "
     "pending_care_bonus increments by 1",
     _care_pre,
     lambda rec: bool(rec["tile_after"] and rec["tile_after"].get("cared_today")),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and "animal" in t
                       and t.get("pending_care_bonus", 0)
                       > rec["tile_after"].get("pending_care_bonus", 0))(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


def _feed_pre(rec):
    t = _animal(rec)
    return bool(t and not t.get("fed_today")
                and rec["inv_before"].get("WHEAT", 0) >= 1)


rule("FEED.keeps_it_alive", "FEED",
     "Animals: FEED consumes 1 WHEAT from the acting unit's bag; two "
     "successive unfed days and the animal escapes, unrecoverable",
     _feed_pre,
     lambda rec: (bool(rec["tile_after"] and rec["tile_after"].get("fed_today"))
                  and _delta(rec["inv_before"], rec["inv_after"], "WHEAT") == -1),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and "animal" in t
                       and t.get("consecutive_unfed", 9) == 0)(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


def _collect_fert_pre(rec):
    t = _animal(rec)
    return bool(t and t.get("fertilizer_available"))


rule("COLLECT_FERTILIZER.use_it_or_lose_it", "COLLECT_FERTILIZER",
     "Animals: every surviving animal makes 1 fertilizer available at end of "
     "day and it does NOT accumulate -- uncollected is lost",
     _collect_fert_pre,
     lambda rec: _delta(rec["inv_before"], rec["inv_after"], "FERTILIZER") == 1,
     # paid = it reached the shed (sellable) rather than being discarded at
     # midnight; README Shed: overflow past shedCapacity is discarded
     lambda rec, ctx: ctx["shed_at_eod"](rec["day"]).get("FERTILIZER", 0)
     > rec["shed_before"].get("FERTILIZER", 0) - 1,
     settle_days=1)


def _place_animal_pre(rec):
    t = rec["tile_before"]
    if not isinstance(t, dict) or "animal" in t:
        return False
    if t.get("kind") not in ("COOP", "PASTURE"):
        return False
    return any(rec["inv_before"].get(k, 0) >= 1
               and STRUCT_OF[k] == t["kind"] for k in ANIMAL_KINDS)


rule("PLACE.animal_on_structure", "PLACE",
     "Animals: standing on a matching unoccupied structure places ONE animal "
     "from inventory; the n argument is ignored",
     _place_animal_pre,
     lambda rec: isinstance(rec["tile_after"], dict) and "animal" in rec["tile_after"],
     lambda rec, ctx: (lambda t: isinstance(t, dict) and "animal" in t)(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])),
     settle_days=1)


# -- TERRAIN ------------------------------------------------------------------
def _build_pre(struct, kinds):
    def f(rec):
        if rec["tile_before"] is not None:
            return False
        held = sum(rec["shed_before"].get(k, 0) + rec["inv_before"].get(k, 0)
                   for k in kinds)
        return held >= 1
    return f


rule("BUILD_COOP.for_a_goose_we_hold", "BUILD_COOP",
     "Terrain: BUILD_COOP adds a coop to an unoccupied tile; DIG cannot "
     "remove it again once an animal is on it",
     _build_pre("COOP", ("GOOSE",)),
     lambda rec: isinstance(rec["tile_after"], dict)
     and rec["tile_after"].get("kind") == "COOP",
     lambda rec, ctx: any(
         isinstance(ctx["grid_at"](s), list)
         and isinstance(ctx["grid_at"](s)[rec["y"]][rec["x"]], dict)
         and "animal" in ctx["grid_at"](s)[rec["y"]][rec["x"]]
         for s in ctx["snap_steps"] if s > rec["step"]),
     settle_days=1)

rule("BUILD_PASTURE.for_stock_we_hold", "BUILD_PASTURE",
     "Terrain: BUILD_PASTURE adds a pasture to an unoccupied tile; DIG cannot "
     "remove it again once an animal is on it",
     _build_pre("PASTURE", ("COW", "SHEEP")),
     lambda rec: isinstance(rec["tile_after"], dict)
     and rec["tile_after"].get("kind") == "PASTURE",
     lambda rec, ctx: any(
         isinstance(ctx["grid_at"](s), list)
         and isinstance(ctx["grid_at"](s)[rec["y"]][rec["x"]], dict)
         and "animal" in ctx["grid_at"](s)[rec["y"]][rec["x"]]
         for s in ctx["snap_steps"] if s > rec["step"]),
     settle_days=1)


rule("DIG.clears_a_weed", "DIG",
     "Map Features: weeds spawn on empty tiles and must be cleared before the "
     "land can be used; Terrain: DIG removes a weed",
     lambda rec: isinstance(rec["tile_before"], dict)
     and rec["tile_before"].get("kind") == "WEED",
     lambda rec: rec["tile_after"] is None,
     # paid = the freed tile was actually used before the settle horizon
     lambda rec, ctx: any(
         isinstance(ctx["grid_at"](s), list)
         and isinstance(ctx["grid_at"](s)[rec["y"]][rec["x"]], dict)
         for s in ctx["snap_steps"] if s > rec["step"]),
     settle_days=1)


# -- PLANTING -----------------------------------------------------------------
def _plant_pre(rec):
    if rec["tile_before"] is not None or len(rec["op"]) < 2:
        return False
    return rec["seeds_before"].get(rec["op"][1], 0) >= 1


rule("PLANT.survives_its_first_night", "PLANT",
     "Watering: a new seed starts with consecutive_unwatered = 1, so a seed "
     "planted and left unwatered that same day is a weed by morning",
     _plant_pre,
     lambda rec: (isinstance(rec["tile_after"], dict)
                  and rec["tile_after"].get("kind") == "PLANT"
                  and _delta(rec["seeds_before"], rec["seeds_after"],
                             rec["op"][1]) == -1),
     lambda rec, ctx: (lambda t: isinstance(t, dict) and t.get("kind") == "PLANT")(
         _tile_after_eod(ctx, rec["day"], rec["x"], rec["y"])))


# -- SHED ---------------------------------------------------------------------
def _pickup_pre(rec):
    if len(rec["op"]) < 2:
        return False
    return rec["shed_before"].get(rec["op"][1], 0) >= 1


rule("PICKUP.gets_what_it_asked_for", "PICKUP",
     "Shed: PICKUP moves up to n of an item from the shed into the acting "
     "unit's inventory, clamped to what the shed holds",
     _pickup_pre,
     lambda rec: _delta(rec["inv_before"], rec["inv_after"], rec["op"][1]) > 0,
     # paid = it was consumed or banked before midnight rather than riding
     # around; README Shed: the midnight drop returns it, overflow discarded
     lambda rec, ctx: bool(_later_ops(
         ctx, rec, ("FEED", "FERTILIZE", "PLACE", "DROP"),
         until=(rec["day"] + 1) * TPD - 1)))


def _drop_pre(rec):
    return sum(rec["inv_before"].values()) > 0


rule("DROP.nothing_is_discarded", "DROP",
     "Shed: DROP dumps the entire inventory into the shed and anything past "
     "shedCapacity is discarded -- there is no overflow holding area",
     _drop_pre,
     lambda rec: sum(rec["shed_after"].values()) > sum(rec["shed_before"].values()),
     lambda rec, ctx: (sum(rec["shed_after"].values())
                       - sum(rec["shed_before"].values())
                       == sum(rec["inv_before"].values())))


# -- MOVEMENT -----------------------------------------------------------------
for _d in ("NORTH", "SOUTH", "EAST", "WEST"):
    rule(f"MOVE.{_d.lower()}_is_not_off_board", _d,
         "Movement: moves off the edge of the board are no-ops; locked tiles "
         "are passable",
         lambda rec: True,
         lambda rec: rec["pos_after"] != [rec["x"], rec["y"]],
         lambda rec, ctx: rec["pos_after"] != [rec["x"], rec["y"]])


# -- MARKET -------------------------------------------------------------------
def market_rule(name, op, readme, paid, settle_days=1, item=None):
    RULES.append(Rule(name, op, readme,
                      lambda rec: True,
                      lambda rec: rec["ok"],
                      paid, settle_days=settle_days, arg=item, kind="market"))


market_rule("SELL.above_the_floor", "SELL",
            "Market Mechanics: the sell price is quoted at the pre-sell "
            "inventory and floored at $1; a unit sold at $1 is not even added "
            "to market inventory",
            lambda rec, ctx: rec["price"] > 1, settle_days=0)

market_rule("BUY_SEED.gets_planted", "BUY_SEED",
            "Market Action: BUY_SEED purchases N units; seeds live in their "
            "own slot and are consumed directly by PLANT",
            lambda rec, ctx: any(
                r["op"][0] == "PLANT" and r["op"][1] == rec["item"]
                and r["step"] > rec["step"] for r in ctx["our_ops"]),
            settle_days=2)

market_rule("BUY_ANIMAL.reaches_a_structure", "BUY_ANIMAL",
            "Market Action + shed cap: a bought animal lands in the shed and "
            "only earns once PLACEd on a matching structure",
            lambda rec, ctx: any(
                r["op"][0] == "PLACE" and len(r["op"]) > 1
                and r["op"][1] == rec["item"] and r["step"] > rec["step"]
                and isinstance(r["tile_after"], dict)
                and "animal" in r["tile_after"] for r in ctx["our_ops"]),
            settle_days=2)

market_rule("BUY_PRODUCT.wheat_is_fed_not_resold", "BUY_PRODUCT",
            "Market Mechanics: only WHEAT and FERTILIZER can be bought back, "
            "and an immediate buy-then-sell nets exactly zero",
            lambda rec, ctx: any(
                r["op"][0] == "FEED" and r["step"] > rec["step"]
                for r in ctx["our_ops"]),
            settle_days=1, item="WHEAT")

market_rule("HIRE.the_hand_does_work", "HIRE",
            "Hiring: a hand costs farmHandCostMult * fib(n) and disappears at "
            "end of day, so a hand that does not work is a pure loss",
            lambda rec, ctx: ctx["worked_units"](rec["step"] // TPD) > 0,
            settle_days=0)

market_rule("BUY_LAND.the_quadrant_gets_used", "BUY_LAND",
            "Farm Space: BUY_LAND unlocks a 5x5 segment for $1k/$2k/$4k; tile "
            "actions no-op on a locked tile",
            lambda rec, ctx: ctx["new_quadrant_used"](rec["step"]),
            settle_days=2)


BY_NAME = {r.name: r for r in RULES}
