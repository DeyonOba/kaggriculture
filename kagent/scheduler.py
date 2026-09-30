"""Turn-level task generation and ZONE-BASED unit assignment.

Field measurement (episode 106290169 vs a $175k opponent) showed the old
greedy nearest-task assignment walking 3.2 steps per productive action vs the
leaders' 1.2, starving distant quadrants entirely (SW: 72 actions all game).
This executor gives each worker a contiguous serpentine ZONE for the day and
a fixed round inside it — fertilize > water > feed > care > harvest > plant —
so travel is intra-zone, every unlocked quadrant gets dedicated labor, and
the care regime (the 3x animal-output multiplier) is guaranteed rather than
priority-contested. Survival tasks keep a global override; logistics
(pickups, deliveries, drops, liquidation shuttles) stay global.
"""

import os

from kagent.constants import ANIMALS, CROPS, MARKET_PARAMS, SHED_ACCESS, TURNS_PER_DAY
from kagent.planner import FINAL_STEP

# Task priorities (lower = more urgent).
P_SURVIVAL = 0
P_HARVEST = 1
P_ANIMAL = 1.5   # feed/care: a fed+cared production is 3-4x a bare one
P_TEND = 2
P_GROW = 3
P_FILLER = 5

PLANT_LAST_HOUR = 21  # leave >= 2 turns to water the fresh plant
DROP_THRESHOLD = 10   # mid-game inventory size that triggers a shed trip
EARLY_DROP_THRESHOLD = 2  # days 0-9: every fertilizer unit is the day's cash
MIN_ZONE_TILES = 5
SWEEP_DEADLINE_HOUR = 16  # from here survival tasks (dies tonight) jump the sweep
HELP_RADIUS = 5           # cross-zone help only within this many steps

# --- Switchable executor rules (research note §5 B0 and the per-quadrant
# queue), all OFF by default so the shipped behaviour is byte-identical.
# Turn them on with KAGG_EXEC="lastwater,chain,weeds,h23,quadq" in a
# process, or "exec": [...] in a dev arm's rules JSON (rl/rollout.py).
#   lastwater  water a one-time crop on the last bonus-window day BEFORE
#              harvesting it (the engine adds a yield unit per in-window
#              WATER; shipped code harvested at hour 0 of that day and lost
#              the unit: wheat 3 of 4, carrot 2 of 3)
#   chain      finish the tile underfoot (harvest, plant, water, feed ...)
#              before any shed shuttle or provisioning trip
#   weeds      dig weeds at planting priority when the plan wants more tiles
#              than are bare
#   h23        no moves on the day's last turn (units respawn at the shed)
#   quadq      one work queue per unlocked quadrant, hands allocated in
#              proportion to the quadrant's pending chores, survival /
#              harvest / animal chores before tend / grow, nearest within
#   evesell    a unit carrying EVESELL_MIN_CARRY sellable units within
#              EVESELL_MAX_DIST of the shed walks in and DROPs so the
#              produce is on the market before the hour-0 dumps (the engine
#              fills sells unit by unit; measured, the recorded opponents
#              sell 33-42% of their strawberries in the evening to our
#              14-19% and realise 0.85-1.03 of the mean price to our
#              0.68-0.89 in the same markets)
#   harvest_visit  harvest an ongoing crop whenever it holds a unit, at tend
#              priority, so the unit already there for the water takes the
#              produce (the top agent: 470 harvests a game to our 290)
#   spent      DIG an ongoing crop that has made its last production and is
#              harvested clean, at planting priority, before it decays into
#              a weed and idles the tile (strawberries from day ~23)
#   layout     put the wheat quota on bare tiles ADJACENT to pastures and
#              coops, other crops elsewhere, so the unit that harvests the
#              wheat feeds the animals from hand without a shed trip (26% of
#              the top agent's feeds, 0% of ours)
#   handfeed   feed from the harvest: the morning zone provisioning skips
#              animals with a harvestable wheat tile orthogonally adjacent
#              (until HANDFEED_UNTIL; survival feeds are never skipped), and
#              a unit holding wheat feeds an unfed animal next to it before
#              anything else. Metric: feeds from wheat in hand (ours 1-5%,
#              top agent 26-30%), pickups per feed (1.0-1.2 vs 0.5)
#   oneshed    consolidated provisioning: the farm's whole wheat and
#              fertilizer need is fetched by the fewest units nearest the
#              shed (five wheat / three fertilizer each) at hours
#              0-ONESHED_LAST_HOUR, none later except for a starving animal,
#              instead of every zone hand making its own 1-3 wheat pickup;
#              the early-drop threshold on days 0-9 rises from 2 to 5 units.
#              Metric: wheat pickups (ours 266-296 a game, top agent 144),
#              pickups per feed (1.0-1.2 vs 0.5)
#   hv_underfoot  harvest_visit restricted to the tile a unit is standing
#              on: the 1-2 unit harvest of an ongoing crop is taken by the
#              unit already there, never walked to, so it cannot take turns
#              from watering (all-five tripled weed tile-days). Metric:
#              harvests up, weed tile-days flat
#   nodrift    a unit with nothing to do PASSes instead of walking back to
#              its zone: every unit respawns at the shed at hour 0, so the
#              walk home buys nothing (the top agent: 0.71 moves per
#              productive op, ours 1.1-1.4). Metric: moves per productive op
#   dailywater water every ongoing crop every day at tend priority instead
#              of alternate days as filler: a plant dies after two dry
#              nights, the alternate-day scheme leans on the survival water
#              the next day always landing, and on our traces 33-42
#              strawberries a game die into weeds where Boey and DSM lose
#              none (weed tile-days 3 vs our 36). Metric: weed tile-days,
#              plant deaths, harvests
#   endwater   from ENDWATER_DAY, an ongoing crop with a production still
#              due before the season ends is watered daily at animal
#              priority (above tend, below harvest): the weed census on the
#              traced games shows 32-53 strawberries a game dying of two dry
#              nights on days 21-28, when the liquidation harvests and the
#              shed shuttles saturate the crew and filler waters are
#              skipped; Boey and DSM lose none. Metric: weed tile-days on
#              days 20-29, late strawberry harvests
#   animal_visit  harvest an animal whenever it holds ANY produce, at tend
#              priority, the same free-labour argument that made
#              `harvest_visit` worth +4.1 pts on crops: a unit visits every
#              animal every day anyway to FEED and CARE, so the shipped
#              `held >= 2` threshold makes produce sit a day for nothing.
#              It matters most for GEESE (interval 1, max_held 4: one egg a
#              day, so half of them always wait) and is why the three
#              leaders harvest eggs 60-77 times a game where we harvest 0.
#   shedguard  count CARRIED stock toward the shed-pressure valve from hour
#              ONESHED-late every day, not only in the final push. README,
#              Shed: at end of day every unit inventory is added to the shed
#              "if there is room. Anything that doesn't fit is discarded", and
#              the engine's _drop_inventories_to_shed runs unconditionally. Our
#              guards only covered MID-DAY drops, so the crew's pockets were
#              never counted: a ground-truth review measured 39 units a game
#              destroyed with no flags and 59 with the three SHIPPED rules
#              (wheat 38, strawberry 19, melon 2 on one seed) -- roughly $4-5k,
#              larger than lastwater's measured +$2.1k gain. harvest_visit
#              alone REDUCES overflow; combined with lastwater the extra wheat
#              tips it over. Metric: units discarded at end of day.
#   tend_end   liquidation keeps TENDING. From LIQUIDATION_DAY the shipped
#              rules stop fertilising and harvest any tile holding a unit;
#              with the flag, days 27-28 run the everyday rules (fertiliser
#              windows, max-yield / lastwater harvests) and the strip-
#              everything harvest is kept for the final day. Measured on the
#              fresh leaders' winner seats (40 games, replays/now): FERTILIZE
#              16.4 / 13.8 / 1.0 on days 27/28/29, WATER 46 / 43 / 21,
#              HARVEST rising 33 / 39 / 37; the champion on tape worlds:
#              FERTILIZE 0 / 0 / 0, WATER 21 / 19 / 4, HARVEST falling
#              30 / 23 / 9. One-time crops are unaffected (ripe is ripe); the
#              strip takes every ONGOING crop's 1-2 units at harvest
#              priority, ahead of the water and fertiliser the same crew
#              would do, and our WATER halves the day it starts. Metric:
#              fertilise and harvest counts on days 27-29.
#   late_plant seeds already in the bag are planted during liquidation on
#              any crop whose FIRST yield lands by day 29 (wheat and carrot:
#              2 days, so day 27), and on the final day a one-time crop
#              holding units is harvested even below max (the shipped `ripe`
#              rule waits for max yield, which a day-27 planting never
#              reaches). Measured: the champion ends with 11.8 unplanted
#              seeds a game (8.8 carrot, bought on day 26 for a quota the
#              day-27 planting stop then stranded); the fresh leaders end
#              with 0.7 and plant 4.6 tiles on day 27. Metric: seeds left at
#              the end, day-27 plantings, day-29 harvests.
#   care_all   CARE is emitted for every kept animal not yet cared for,
#              without waiting for its FEED to land first. README, Animal
#              Care: the bonus is banked at END OF DAY "if the animal was
#              both fed AND cared for that day" -- order inside the day is
#              irrelevant -- and the shipped rule only emits CARE once
#              fed_today is already set, so with a busy crew the care is
#              deferred and often never happens. Measured (days 23-27): the
#              champion FEEDs 15-16 animals a day and CAREs 6-10; the fresh
#              leaders FEED 15-19 and CARE 15-19. Metric: CARE per FEED.
#   radial     crops plant from the OUTSIDE in: bare tiles within RADIAL_RING
#              Manhattan steps of the four shed-access tiles are planted only
#              when nothing farther is bare, so the ring stays free for the
#              coops and pastures (which already take the nearest empties).
#              tools/placement_heatmap.py, champ-h11 self-play vs the top-10%
#              corpus, days 10-25: livestock mean shed distance ours 2.76 vs
#              theirs 1.83, crops 4.20 vs 4.58; their crop map has a hole at
#              distance <= 2 and ours puts pastures in the far corner because
#              day-0 planting takes the near tiles first. FEED needs a wheat
#              delivery from the shed every day; WATER carries nothing.
#              Metric: livestock mean shed distance, feed trips.
#   executor2  kagent/executor2.py replaces assign(): min-cost assignment of
#              units to the same jobs with COMMITMENT (a unit keeps its job
#              unless a strictly better one appears), the tile underfoot first,
#              farm-wide morning provisioning, fetch-aware costs. Same jobs,
#              same purchases. rlv3/EXECUTOR_V2.md. Metric: tools/plan_ledger.py
#              delivery %, productive turn share, then the paired gate.
#   approve    the plan's purchases are APPROVED: seeds for the whole quota are
#              bought above the cash floor (no 60%-of-cash ration). Ledger on
#              world 0: the herd then grows 3 instead of 4 on days 2-8 and 6
#              instead of 12 on day 9 -- capital allocation, kept separate.
#   replant    the plan by TURN for one-time tiles: a wheat/carrot/melon tile
#              that comes off today is replanted with the same crop the turn
#              it is bare, seeds bought at dawn for the expected harvests. The
#              champion quota is target minus STANDING and the ripe tiles
#              still count as standing at dawn, so before day 10 a harvested
#              tile waits for the next dawn (one day lost per 4-day wheat
#              cycle). Capped by the ground the plan's own asks leave bare;
#              inside each crop's planting window; never in liquidation.
#              MEASURED INERT on the champion plan (worlds 0-1, days 0-16):
#              the quota already exceeds the bare ground at every dawn (day 7
#              bare 17 vs STR 28, day 11 bare 26 vs STR 34), so the room is
#              zero and nothing is added; world 1 plays the identical game.
#              The quota is not the bottleneck -- cash and labour are.
#   size_*     kagent/sizing.py: the dawn plan is APPROVED against a budget
#              and a crew in a fixed claim order (feed, hands, seeds, animals,
#              land) and only the approved part is asked. size_budget = dawn
#              cash + today's expected sales, asks cut to it; size_feed = feed
#              at any price, first claim; size_hires = hands by workload;
#              size_ground = a bare tile whose planned crop is unaffordable is
#              planted with the best affordable crop. `sized` = all four.
#   care_price CARE regardless of the product's price. The shipped crash-pivot
#              drops the CARE when the product trades under 35% of base; the
#              yield ledger (2026-09-30) shows our care rate falling to 63%
#              on days 13-20 and 40% on days 21-28 while the leaders hold
#              82% and 65% with fed rates lower than ours. A cared animal
#              produces 3-4x a bare one and prices recover as the shops drain.
#              Gated alone: -0.014 (2 vs 4), our bank -889 -- cared units
#              sold into the crash do not pay for the turn.
#   hold_crash market: no crashed-market release. The shipped sell dumps a
#              product at any price once it trades under 30% of base; held,
#              it waits for its floor while the shops drain the market (the
#              shed-pressure valve and the final push still bypass). Paired
#              with care_price: an animal cannot be removed, so the question
#              is the SELL timing of what it keeps producing, not the care.

EXEC_FLAGS = ("lastwater", "chain", "weeds", "h23", "quadq", "evesell", "animal_visit", "shedguard",
              "harvest_visit", "spent", "layout",
              "handfeed", "oneshed", "hv_underfoot", "nodrift", "dailywater",
              "endwater", "tend_end", "late_plant", "care_all", "radial", "executor2", "approve",
              "replant", "size_budget", "size_feed", "size_hires", "size_ground", "sized", "care_price",
              "hold_crash")
RADIAL_RING = 2              # radial: crops keep off tiles this close to the shed access
ENDWATER_DAY = 20
EXEC = {k: False for k in EXEC_FLAGS}
LASTWATER_HARVEST_HOUR = 20  # from here harvest the crop even if unwatered
HANDFEED_UNTIL = 14          # after this hour the shed feeds what the harvest has not
ONESHED_LAST_HOUR = 2        # zone provisioning pickups only up to this hour
ONESHED_EARLY_DROP = 5       # days 0-9 pocket size that triggers a shed trip
EVESELL_MIN_CARRY = 3        # sellable units in the pocket that justify the run
EVESELL_MAX_DIST = 5         # farther units keep working; midnight drops them
EVESELL_ARRIVE_HOUR = 22     # DROP by this hour, sold at 22-23, before hour 0
SHEDGUARD_HOUR = 20          # from this hour, carried stock counts toward shed pressure


def set_exec(flags):
    """flags: names (iterable or comma string) or {name: bool}. Unknown raise."""
    if isinstance(flags, str):
        flags = [f for f in flags.split(",") if f.strip()]
    items = flags.items() if isinstance(flags, dict) else [(f.strip(), True) for f in flags]
    for k, v in items:
        if k not in EXEC:
            raise ValueError(f"unknown executor flag {k!r}; known: {EXEC_FLAGS}")
        EXEC[k] = bool(v)


def reset_exec():
    for k in EXEC:
        EXEC[k] = False


if os.environ.get("KAGG_EXEC"):
    set_exec(os.environ["KAGG_EXEC"])


class Task:
    __slots__ = ("pri", "x", "y", "op", "need")

    def __init__(self, pri, x, y, op, need=None):
        self.pri = pri
        self.x = x
        self.y = y
        self.op = op
        self.need = need  # item that must be in the unit's inventory


def _dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def _shed_ring(x, y):
    """Manhattan distance to the nearest of the four shed-access tiles."""
    return min(_dist(x, y, sx, sy) for sx, sy in ((4, 4), (5, 4), (4, 5), (5, 5)))


def _plant_order(rest):
    """radial: the ring around the shed is planted last."""
    if not EXEC["radial"]:
        return rest
    return ([e for e in rest if _shed_ring(*e) > RADIAL_RING]
            + [e for e in rest if _shed_ring(*e) <= RADIAL_RING])


def _step_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    if dx != 0:
        return "EAST" if dx > 0 else "WEST"
    return None


def _nearest_shed_tile(x, y):
    return min(SHED_ACCESS, key=lambda t: _dist(x, y, t[0], t[1]))


def _unlocked(view, x, y):
    return view.me.tile(x, y) != "LOCKED"


def zone_map(view):
    """Serpentine partition of unlocked tiles into contiguous worker zones.

    Returns (zone_of: {(x, y): zone_id}, n_zones). Deterministic given the
    farm state, so a unit's zone (unit_index % n_zones) is sticky within a
    day without any cross-turn state.
    """
    quads = [("NW", 0, 0), ("NE", 5, 0), ("SW", 0, 5), ("SE", 5, 5)]
    unlocked = view.me.unlocked_quadrants
    path = []
    for name, x0, y0 in quads:
        if name not in unlocked:
            continue
        for r in range(5):
            cols = range(5) if r % 2 == 0 else range(4, -1, -1)
            for c in cols:
                path.append((x0 + c, y0 + r))
    n_units = max(1, len(view.me.units))
    n_zones = max(1, min(n_units, len(path) // MIN_ZONE_TILES))
    # Even split (sizes differ by at most one): a ceil'd chunk left the last
    # zone EMPTY (75 tiles / 12 hands -> 11 zones), so one hand roamed the
    # whole farm every day.
    zone_of = {tile: i * n_zones // len(path) for i, tile in enumerate(path)}
    return zone_of, n_zones


def quadrant_queues(view, tasks, plan=None):
    """EXEC["quadq"]: one zone per unlocked quadrant, with the units
    allocated in proportion to the quadrant's pending chores (non-filler
    tasks; PLANT tasks -- one per bare tile -- are weighted so they sum to
    the day's remaining plant quota; every quadrant with work gets at least
    one unit when the crew allows). Returns (zone_of: {(x, y): zone},
    n_zones, uz: {unit: zone}). Ordering inside a queue is done by `pick`
    (priority class, then distance). Stateless per turn like zone_map."""
    quads = [("NW", 0, 0), ("NE", 5, 0), ("SW", 0, 5), ("SE", 5, 5)]
    unlocked = [q for q in quads if q[0] in view.me.unlocked_quadrants] or quads[:1]
    zone_of = {}
    for zi, (_, x0, y0) in enumerate(unlocked):
        for r in range(5):
            for c in range(5):
                zone_of[(x0 + c, y0 + r)] = zi
    n = len(unlocked)
    n_units = max(1, len(view.me.units))
    n_plant = sum(1 for t in tasks if t.op[0] == "PLANT")
    quota = sum(plan.plant_quota.values()) if plan is not None else n_plant
    plant_w = min(1.0, quota / n_plant) if n_plant else 0.0
    work = [0.0] * n
    for t in tasks:
        z = zone_of.get((t.x, t.y))
        if z is not None and t.pri < P_FILLER:
            work[z] += plant_w if t.op[0] == "PLANT" else 1.0
    total = sum(work)
    if total == 0:
        alloc = [n_units // n + (1 if i < n_units % n else 0) for i in range(n)]
    else:
        raw = [n_units * w / total for w in work]
        alloc = [int(r) for r in raw]
        for i in range(n):
            if work[i] > 0 and alloc[i] == 0 and sum(alloc) < n_units:
                alloc[i] = 1
        rem = n_units - sum(alloc)
        by_frac = sorted(range(n), key=lambda i: (raw[i] - int(raw[i]), work[i]), reverse=True)
        j = 0
        while rem > 0:
            alloc[by_frac[j % n]] += 1
            rem -= 1
            j += 1
        while rem < 0:
            k = max(range(n), key=lambda i: (alloc[i] - raw[i], alloc[i]))
            alloc[k] -= 1
            rem += 1
    uz = {}
    zi, left = 0, alloc[0]
    for idx, _, _ in view.me.units:
        while zi < n - 1 and left <= 0:
            zi += 1
            left = alloc[zi]
        uz[idx] = zi
        left -= 1
    return zone_of, n, uz


REPLANT_LAST_DAY = {"WHEAT": 25, "CARROT": 26, "MELON": 19}   # the market's planting windows


def replant_quota(view, plan):
    """replant: add today's one-time harvests back to the plan's quota, same
    crop, capped by the ground left after the plan's own asks. Called once at
    dawn, before the market buys the day's seeds."""
    if not EXEC["replant"] or plan.liquidation:
        return {}
    day = view.day
    me = view.me
    ripe = {}
    for _, _, t in me.plants():
        cd = CROPS.get(t["crop"])
        if cd is None or cd["ongoing"]:
            continue
        age = day - t.get("planted_day", day)
        if age >= cd["max_yield_day"] or 0 <= t.get("max_lifespan_step", -1) <= (day + 1) * TURNS_PER_DAY:
            ripe[t["crop"]] = ripe.get(t["crop"], 0) + 1
    if not ripe:
        return {}
    free0 = (sum(1 for _ in me.empty_tiles()) + sum(ripe.values())
             - sum(plan.structure_builds.values()))
    room = max(0, free0 - sum(plan.plant_quota.values()))
    added = {}
    for crop in ("WHEAT", "CARROT", "MELON"):
        n = ripe.get(crop, 0)
        if n <= 0 or day > REPLANT_LAST_DAY[crop] or day + CROPS[crop]["first_yield_day"] > 29:
            continue
        n = min(n, room)
        if n <= 0:
            break
        plan.plant_quota[crop] = plan.plant_quota.get(crop, 0) + n
        added[crop] = n
        room -= n
    return added


def build_tasks(view, plan):
    """Generate this turn's pending tile tasks."""
    tasks = []
    day = view.day
    me = view.me

    fert_targets = set()
    unit_tiles = {(ux, uy) for _, ux, uy in me.units}

    # late_plant: during liquidation the planner asks for nothing; put the
    # seeds already paid for into bare ground while their first yield still
    # lands inside the season.
    if EXEC["late_plant"] and plan.liquidation and not plan.plant_quota:
        room = len(list(me.empty_tiles()))
        late = {}
        for c in ("WHEAT", "CARROT"):
            if day + CROPS[c]["first_yield_day"] > 29 or room <= 0:
                continue
            n = min(view.seeds.get(c, 0), room)
            if n > 0:
                late[c] = n
                room -= n
        plan.plant_quota = late

    # --- Plants: water / harvest / fertilize ---------------------------------
    for x, y, t in me.plants():
        crop = CROPS.get(t["crop"])
        if crop is None:
            continue
        age = day - t["planted_day"]
        yields = t.get("yield_units", 0)

        decaying = 0 <= t.get("max_lifespan_step", -1) <= view.step + 24
        if crop["ongoing"]:
            # The engine sets max_lifespan_step on the night of the LAST
            # production; at that step the plant loses a unit every two
            # steps and becomes a WEED the moment it hits zero, so a plant
            # harvested clean turns into a weed at exactly that step. Under
            # `spent`, a finished plant is harvested at once and dug before.
            finished = EXEC["spent"] and t.get("max_lifespan_step", -1) >= 0
            # tend_end: the liquidation strip runs on the final day only
            liq_strip = plan.liquidation and (day >= 29 or not EXEC["tend_end"])
            if (yields >= crop["max_yield"] - 1
                    or (yields > 0 and (decaying or finished or liq_strip))):
                tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))
            elif yields >= 1 and (EXEC["harvest_visit"]
                                  or (EXEC["hv_underfoot"] and (x, y) in unit_tiles)):
                tasks.append(Task(P_TEND, x, y, ["HARVEST"]))
            if finished and yields == 0 and not plan.liquidation:
                # tend priority: the plant is a weed within 24-48 steps of
                # its last production, and a dig at planting priority
                # (measured: 29 of 33 strawberries still died) rarely
                # arrives in time
                tasks.append(Task(P_TEND, x, y, ["DIG"]))
                continue
        else:
            if (
                EXEC["lastwater"]
                and age == crop["max_yield_day"]
                and yields < crop["max_yield"]
                and not t.get("watered_today")
                and view.hour < LASTWATER_HARVEST_HOUR
            ):
                # Last day of the bonus window and not yet watered: the WATER
                # adds a yield unit, so it goes first at harvest priority (the
                # underfoot pass keeps the unit on the tile) and the HARVEST
                # is generated next turn once watered_today is set.
                tasks.append(Task(P_HARVEST, x, y, ["WATER"]))
                continue
            ripe = age >= crop["first_yield_day"] and (
                yields >= crop["max_yield"] or age >= crop["max_yield_day"]
            )
            # late_plant: on the final day anything a one-time crop holds is
            # taken; `ripe` waits for max yield, which a late planting never
            # reaches, and units left on the tile at the end are lost.
            last_call = EXEC["late_plant"] and day >= 29 and yields > 0
            if ripe or last_call or (yields > 0 and decaying and age >= crop["first_yield_day"]):
                tasks.append(Task(P_HARVEST if not decaying else P_SURVIVAL, x, y, ["HARVEST"]))
                continue

        # Fertilizer is cash (~$100/unit at base): only spend it on a crop
        # when the market has glutted it below what the extra yield returns.
        fert_price = view.market_prices.get("FERTILIZER", 100)
        # plan.fert_gate (knob, default 1.0) scales the price gates: the top
        # agent fertilises 220 times a game to our 112, mostly wheat, whose
        # fertilised window waters add 2 units instead of 1 (6 vs 4 a plant)
        gate = getattr(plan, "fert_gate", 1.0)
        # plan.fert_budget (rlv3 readout, v5): FERTILIZE actions the plan
        # wants today. None keeps the shipped price gate exactly. 0 means
        # none at all -- the leaders' opening. N > 0 means the crop-age
        # windows below still decide WHICH tile (engine facts about when
        # fertiliser adds yield) but the PRICE test is waived, because the
        # leaders fertilise 205 times a season at whatever the price is and
        # a price gate is how we stop at ~112. policy.act decrements it per
        # emitted op, so the cap holds across the day's turns.
        budget = getattr(plan, "fert_budget", None)
        if budget is not None and budget <= 0:
            pass                                  # phase says no fertiliser today
        elif (t.get("fertilized_until_day", -1) < day
              and (not plan.liquidation or (EXEC["tend_end"] and day < 29))):
            waive = budget is not None and budget > 0
            # Strawberries: a fertilized production yields 2 instead of 1
            # (~$165 each) for 3 days of cover — champions run ~270 units off
            # 33 tiles this way versus ~130 unfertilized.
            want_fert = (
                (t["crop"] == "STRAWBERRY" and 9 <= age <= 15 and (waive or fert_price < 150 * gate))
                or (t["crop"] == "TOMATO" and age in (8, 11) and (waive or fert_price < 80 * gate))
                or (t["crop"] == "WHEAT" and age == 2 and (waive or fert_price < 40 * gate))
                or (t["crop"] == "CARROT" and age == 2 and (waive or fert_price < 50 * gate))
            )
            if want_fert:
                tasks.append(Task(P_TEND, x, y, ["FERTILIZE"], need="FERTILIZER"))
                fert_targets.add((x, y))

    for x, y, t in me.plants():
        if t.get("watered_today"):
            continue
        crop = CROPS.get(t["crop"])
        if crop is None:
            continue
        age = day - t["planted_day"]
        if (x, y) in fert_targets and t.get("consecutive_unwatered", 0) == 0 and view.hour < 18:
            continue
        # Engine rules: water adds yield only inside a one-shot crop's growth
        # window, or on a fertilized ongoing crop's production night; any
        # other day it is pure survival, and a plant survives one dry day.
        # Watering everything daily is what starved planting of labor.
        if t.get("consecutive_unwatered", 0) >= 1:
            pri = P_SURVIVAL  # dies tonight if unwatered
        elif not crop["ongoing"]:
            in_window = (crop["max_yield_day"] + 1) // 2 <= age <= crop["max_yield_day"]
            pri = P_TEND if in_window else P_FILLER
        else:
            since_first = day + 1 - t["planted_day"] - crop["first_yield_day"]
            produces_tonight = since_first >= 0 and since_first % crop["interval"] == 0
            fertilized = t.get("fertilized_until_day", -1) >= day
            pri = P_TEND if (produces_tonight and fertilized) or EXEC["dailywater"] else P_FILLER
            if (EXEC["endwater"] and day >= ENDWATER_DAY and pri > P_ANIMAL
                    and t.get("max_lifespan_step", -1) < 0 and crop["interval"] > 0):
                # next production night and its ordinal; water daily while
                # one is still due before the last sellable morning (day 29)
                n_ahead = (-since_first) % crop["interval"]
                count = (since_first + n_ahead) // crop["interval"] + 1
                if day + n_ahead <= 28 and count <= crop["max_yield"]:
                    pri = P_ANIMAL
        tasks.append(Task(pri, x, y, ["WATER"]))

    # --- Animals: the care regime is first-class (3x output multiplier) --------
    feeding_day = day < 29
    # Culled kinds (plan.cull_kinds, from the champion plan's cull knobs) get
    # no FEED and no CARE: they escape after two unfed days, and the wheat
    # and the actions go to the crops. Existing yield is still harvested.
    culled = getattr(plan, "cull_kinds", None) or ()
    for x, y, t in me.animals():
        keep = t.get("animal") not in culled
        if feeding_day and keep and not t.get("fed_today"):
            # Feed EVERY animal EVERY day: the CARE bonus (banked only on
            # fed+cared days, paid only on fed production days) is what turns
            # 0.5 milk/cow/day into ~1.5. Survival cases stay hard-urgent.
            pri = P_SURVIVAL if t.get("consecutive_unfed", 0) >= 1 else P_ANIMAL
            tasks.append(Task(pri, x, y, ["FEED"], need="WHEAT"))
        if (feeding_day and keep and not t.get("cared_today") and day < 28
                and (t.get("fed_today") or EXEC["care_all"])):
            # Crash-pivot: caring boosts output we cannot profitably sell when
            # the product's market has collapsed — release those actions.
            product = ANIMALS.get(t.get("animal"), {}).get("product")
            price_ok = (
                EXEC["care_price"]
                or not product
                or view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product]["base"]
            )
            if price_ok:
                tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))
        if t.get("fertilizer_available"):
            tasks.append(Task(P_TEND, x, y, ["COLLECT_FERTILIZER"]))
        held = t.get("yield_units", 0)
        max_held = ANIMALS.get(t.get("animal"), {}).get("max_held", 4)
        # Harvest every production so yields never clip at max_held.
        if held >= 2 or (held > 0 and (plan.liquidation or day >= 27)):
            tasks.append(Task(P_HARVEST, x, y, ["HARVEST"]))
        elif EXEC["animal_visit"] and held >= 1:
            tasks.append(Task(P_TEND, x, y, ["HARVEST"]))

    # --- Growth: builds, animal placement, planting ------------------------------
    empties = sorted(
        me.empty_tiles(), key=lambda p: _dist(p[0], p[1], 4, 4) + _dist(p[0], p[1], 5, 5)
    )
    cursor = 0
    for structure, op in (("COOP", "BUILD_COOP"), ("PASTURE", "BUILD_PASTURE")):
        for _ in range(plan.structure_builds.get(structure, 0)):
            if cursor >= len(empties):
                break
            x, y = empties[cursor]
            cursor += 1
            tasks.append(Task(P_GROW, x, y, [op]))

    available = {}
    for kind in ("GOOSE", "COW", "SHEEP"):
        available[kind] = view.shed.get(kind, 0) + sum(
            view.unit_inv(i).get(kind, 0) for i, _, _ in me.units
        )
    for x, y, t in me.empty_structures():
        kinds = ("GOOSE",) if t["kind"] == "COOP" else ("COW", "SHEEP")
        for kind in kinds:
            if available.get(kind, 0) > 0:
                available[kind] -= 1
                tasks.append(Task(P_TEND, x, y, ["PLACE", kind], need=kind))
                break

    if view.hour <= PLANT_LAST_HOUR:
        # One PLANT task per empty tile, crops interleaved by remaining quota,
        # so every zone plants its own ground instead of all hands converging
        # on the few center-most empties. The daily quota is enforced at
        # execution time (assign.try_execute).
        wanted = [crop for crop, n in plan.plant_quota.items() for _ in range(n)]
        if wanted and EXEC["layout"] and plan.plant_quota.get("WHEAT", 0) > 0:
            # Wheat rows beside the animals: bare tiles orthogonally adjacent
            # to a pasture or coop take the wheat quota first, the rest of
            # the quota goes to the other bare tiles in the usual rotation.
            near = set()
            for sx, sy, st_ in me.iter_tiles():
                if isinstance(st_, dict) and st_.get("kind") in ("COOP", "PASTURE"):
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        near.add((sx + dx, sy + dy))
            rest = _plant_order(empties[cursor:])
            adj = [e for e in rest if e in near]
            far = [e for e in rest if e not in near]
            n_wheat = plan.plant_quota["WHEAT"]
            wheat_tiles = adj[:n_wheat]
            others = [c for c in wanted if c != "WHEAT"] or ["WHEAT"]
            for x, y in wheat_tiles:
                tasks.append(Task(P_GROW, x, y, ["PLANT", "WHEAT"]))
            remaining = far + adj[len(wheat_tiles):]
            for i, (x, y) in enumerate(remaining):
                tasks.append(Task(P_GROW, x, y, ["PLANT", others[i % len(others)]]))
        elif wanted:
            for i, (x, y) in enumerate(_plant_order(empties[cursor:])):
                tasks.append(Task(P_GROW, x, y, ["PLANT", wanted[i % len(wanted)]]))

    weed_pri = P_GROW if len(empties) - cursor < 6 else P_FILLER
    if EXEC["weeds"] and weed_pri == P_FILLER:
        # A weed on plantable ground is a lost tile-day whenever the plan
        # wants more tiles than are bare: dig it at planting priority.
        if len(empties) - cursor < sum(plan.plant_quota.values()) + 2:
            weed_pri = P_GROW
    for x, y, _ in me.weeds():
        tasks.append(Task(weed_pri, x, y, ["DIG"]))

    return tasks


HANDFEED_RADIUS = 3          # an animal this close to a ripe wheat tile is fed from its harvest


def _handfed_tiles(view):
    """Unfed animal tiles that today's wheat harvests can feed from hand
    (EXEC["handfeed"]): each harvestable WHEAT plant (age at its last window
    day or later, units on it) covers up to its units of the nearest unfed
    animals within HANDFEED_RADIUS. Animals never covered are left to the
    shed provisioning."""
    myd = CROPS["WHEAT"]["max_yield_day"]
    ready = []
    for x, y, t in view.me.plants():
        if t.get("crop") == "WHEAT" and view.day - t.get("planted_day", view.day) >= myd \
                and t.get("yield_units", 0) > 0:
            ready.append((x, y, int(t["yield_units"])))
    unfed = [(x, y) for x, y, t in view.me.animals() if not t.get("fed_today")]
    out = set()
    for wx, wy, units in ready:
        near = sorted(((_dist(wx, wy, ax, ay), ax, ay) for ax, ay in unfed
                       if (ax, ay) not in out and _dist(wx, wy, ax, ay) <= HANDFEED_RADIUS))
        for _, ax, ay in near[:units]:
            out.add((ax, ay))
    return out


def _feed_pickup_needs(view, zone_of, n_zones):
    """Wheat each unit should carry for the unfed animals in ITS zone."""
    needs = {}
    for x, y, t in view.me.animals():
        if view.day >= 29 or t.get("fed_today"):
            continue
        z = zone_of.get((x, y))
        if z is not None:
            needs[z] = needs.get(z, 0) + 1
    return needs


MOVE_STATS = {}  # debug: moves per assignment step (filled only when enabled)
ASSIGN_LABEL = {}  # debug: unit idx -> step that assigned its action this turn
MOVE_STATS_ENABLED = False


def _count_moves(actions, before, label):
    if not MOVE_STATS_ENABLED:
        return
    for idx, op in actions.items():
        if idx in before or not op:
            continue
        ASSIGN_LABEL[idx] = label
        key = label if op[0] in ("NORTH", "SOUTH", "EAST", "WEST") else "act:" + label
        MOVE_STATS[key] = MOVE_STATS.get(key, 0) + 1


def assign(view, plan, tasks):
    """Zone-based assignment. Returns {unit_idx: op_list}."""
    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    seeds_left = dict(view.seeds)
    shed_room = max(0, 100 - view.shed_total)
    zone_of, n_zones = zone_map(view)
    uz = {idx: idx % n_zones for idx, _, _ in me.units}  # unit -> zone
    if EXEC["quadq"]:
        zone_of, n_zones, uz = quadrant_queues(view, tasks, plan)

    final_push = view.step >= FINAL_STEP - 10
    aggressive = view.day >= 29 or final_push

    plant_requests = {}
    claimed = set()  # (x, y, op0) already assigned this turn

    def try_execute(idx, task):
        x, y = units[idx]
        key = (task.x, task.y, task.op[0])
        if key in claimed:
            return False
        if task.need and view.unit_inv(idx).get(task.need, 0) < 1:
            return False
        if task.op[0] == "PLANT":
            crop = task.op[1]
            allowed = min(seeds_left.get(crop, 0), plan.plant_quota.get(crop, 0))
            if plant_requests.get(crop, 0) + 1 > allowed:
                return False
        if (x, y) == (task.x, task.y):
            if not _unlocked(view, task.x, task.y):
                return False
            actions[idx] = task.op
            if task.op[0] == "PLANT":
                plant_requests[task.op[1]] = plant_requests.get(task.op[1], 0) + 1
        else:
            step = _step_toward(x, y, task.x, task.y)
            if not step:
                return False
            actions[idx] = [step]
        claimed.add(key)
        return True

    # Work underfoot, regardless of whose zone the tile belongs to. `pick`
    # below only searches the unit's own zone, and measured, the unit
    # standing on an unserved task tile is in a FOREIGN zone 65% of the time,
    # so it is refused the task and walks back to its own zone.
    _uf = {}
    for _t in tasks:
        if _t.pri < P_FILLER:
            _uf.setdefault((_t.x, _t.y), []).append(_t)

    def _underfoot(idx):
        _x, _y = units[idx]
        _inv = view.unit_inv(idx)
        _best = None
        for _t in _uf.get((_x, _y), ()):
            if (_t.x, _t.y, _t.op[0]) in claimed:
                continue
            if _t.need and _inv.get(_t.need, 0) < 1:
                continue
            if _best is None or _t.pri < _best.pri:
                _best = _t
        return _best

    # Carriers of an animal are skipped by the underfoot pass so the animal
    # in a pocket still reaches its structure first.
    _hold = {i for i, _, _ in me.units
             if any(view.unit_inv(i).get(_a, 0)
                    for _a in ("GOOSE", "COW", "SHEEP"))}

    if EXEC["chain"]:
        # ---- 0. Chore chain: a unit standing on a tile with an executable
        # chore finishes the tile before any shuttle or provisioning trip.
        # Measured on days 0-9: a carrot harvest tripped the early-drop
        # threshold, the unit walked to the shed with the tile bare, and the
        # replant came 20+ steps later; the top agent replants next step.
        for idx in sorted(free - _hold):
            _t = _underfoot(idx)
            if _t is not None and try_execute(idx, _t):
                free.discard(idx)

    # ---- 1. Liquidation / heavy-load shuttles (global) ----------------------
    # Fertilizer in a pocket is only working stock while a FERTILIZE task
    # wants it; otherwise it is unsold cash and goes to the shed.
    fert_wanted = any(t.need == "FERTILIZER" for t in tasks)
    for idx, x, y in me.units:
        if idx in actions:
            continue
        inv = view.unit_inv(idx)
        total = sum(
            n
            for item, n in inv.items()
            if not (
                item in ("GOOSE", "COW", "SHEEP")
                or (not aggressive and fert_wanted and item == "FERTILIZER" and n <= 3)
                or (not aggressive and item == "WHEAT" and n <= 5)
            )
        )
        if total == 0:
            continue
        threshold = 1 if (aggressive and view.hour >= 4) or view.step >= FINAL_STEP - 6 else DROP_THRESHOLD
        if view.day < 10:
            threshold = min(threshold, ONESHED_EARLY_DROP if EXEC["oneshed"] else EARLY_DROP_THRESHOLD)
        if plan.liquidation and not aggressive:
            threshold = 8
        if total < threshold:
            continue
        tx, ty = _nearest_shed_tile(x, y)
        if (x, y) == (tx, ty):
            if shed_room >= total or final_push:
                actions[idx] = ["DROP"]
                shed_room = max(0, shed_room - total)
            else:
                # README: PLACE resolves the animal-placement branch FIRST
                # when the unit stands on a matching empty structure, so
                # naming an animal here puts ONE animal on the tile and
                # nothing in the shed. Ship only what `total` counted as
                # cargo -- animals were deliberately excluded from it.
                cargo = {k: v for k, v in inv.items()
                         if k not in ("GOOSE", "COW", "SHEEP")}
                if not cargo:
                    continue
                item = max(cargo, key=cargo.get)
                n = min(cargo[item], shed_room)
                if n <= 0:
                    continue
                actions[idx] = ["PLACE", item, n]
                shed_room -= n
        else:
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else ["PASS"]
        free.discard(idx)

    _count_moves(actions, set(), "shuttle")
    _seen = set(actions)

    if EXEC["evesell"] and not aggressive:
        # ---- 1b. Evening shed run: produce in a pocket at midnight is
        # dropped by the engine and sold at hour 0 into both farms' dumps;
        # carried in by hour 22 it sells first. Only units close enough to
        # arrive in time, with enough in the pocket to be worth the walk.
        for idx in sorted(free):
            x, y = units[idx]
            inv = view.unit_inv(idx)
            carry = sum(n for item, n in inv.items()
                        if item not in ("GOOSE", "COW", "SHEEP", "WHEAT", "FERTILIZER"))
            if carry < EVESELL_MIN_CARRY:
                continue
            tx, ty = _nearest_shed_tile(x, y)
            d = _dist(x, y, tx, ty)
            # leave in time to arrive by EVESELL_ARRIVE_HOUR; a unit already
            # at the shed drops from the hour before, so it does not walk
            # off to a chore and back
            if d > EVESELL_MAX_DIST or view.hour < EVESELL_ARRIVE_HOUR - max(d, 1):
                continue
            if d == 0:
                if shed_room <= 0:
                    continue
                actions[idx] = ["DROP"]
                shed_room = max(0, shed_room - sum(inv.values()))
            else:
                step = _step_toward(x, y, tx, ty)
                if not step:
                    continue
                actions[idx] = [step]
            free.discard(idx)
        _count_moves(actions, _seen, "evesell")
        _seen = set(actions)

    handfed = _handfed_tiles(view) if EXEC["handfeed"] else set()
    if EXEC["handfeed"] or EXEC["oneshed"]:
        # ---- 1c. Wheat in hand feeds first: a unit holding wheat takes the
        # nearest unfed animal before any provisioning trip or zone pick --
        # within two tiles under handfeed (the harvester feeds its
        # neighbours), anywhere under oneshed (the few provisioned feeders
        # serve the whole farm, since no one else holds wheat).
        radius = 99 if EXEC["oneshed"] else 2
        for idx in sorted(free - _hold):
            x, y = units[idx]
            if view.unit_inv(idx).get("WHEAT", 0) < 1:
                continue
            best = None
            for t in tasks:
                if t.op[0] != "FEED" or (t.x, t.y, "FEED") in claimed:
                    continue
                d = _dist(x, y, t.x, t.y)
                if d <= radius and (best is None or (d, t.pri) < best[0]):
                    best = ((d, t.pri), t)
            if best is not None and try_execute(idx, best[1]):
                free.discard(idx)
        _count_moves(actions, _seen, "handfeed")
        _seen = set(actions)

    # ---- 1.5 Global need-provisioning: animals, fertilizer, and any wheat the
    # morning zone pickups missed. Without this, bought animals strand in the
    # shed forever (measured: 47 structures built, zero animals ever placed).
    # Only ANIMALS are provisioned globally here (rare, high-value, any unit
    # may carry one to its pasture). Wheat and fertilizer are fetched by the
    # zone that needs them (step 2) so hands never turn into farm-wide
    # couriers — measured: 150-200 zone moves/day for 40-60 actions.
    wants = {}
    carried = {}
    for t in tasks:
        if t.need in ("GOOSE", "COW", "SHEEP"):
            wants[t.need] = wants.get(t.need, 0) + 1
    for idx, _, _ in me.units:
        for item, n in view.unit_inv(idx).items():
            carried[item] = carried.get(item, 0) + n
    for item, want in sorted(wants.items()):
        shortage = min(want - carried.get(item, 0), view.shed.get(item, 0))
        batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
        while shortage > 0 and free:
            idx = min(
                free,
                key=lambda i: _dist(units[i][0], units[i][1],
                                    *_nearest_shed_tile(units[i][0], units[i][1])),
            )
            x, y = units[idx]
            tx, ty = _nearest_shed_tile(x, y)
            n = min(batch, shortage)
            if (x, y) == (tx, ty):
                actions[idx] = ["PICKUP", item, n]
            else:
                step = _step_toward(x, y, tx, ty)
                if not step:
                    break
                actions[idx] = [step]
            free.discard(idx)
            shortage -= n

    _count_moves(actions, _seen, "provision")
    _seen = set(actions)

    # ---- 2. Zone provisioning: each zone's hand fetches ITS zone's wheat and
    # fertilizer from the shed (any hour), so deliveries stay intra-zone.
    zone_needs = {}
    zone_starving = set()
    for t in tasks:
        if t.need in ("WHEAT", "FERTILIZER"):
            z = zone_of.get((t.x, t.y))
            if z is None:
                continue
            if t.op[0] == "FEED" and t.pri == P_SURVIVAL:
                zone_starving.add(z)
            # handfeed: the harvest next door feeds this animal; the shed
            # provisioning skips it until HANDFEED_UNTIL (never a survival feed)
            if (handfed and t.op[0] == "FEED" and t.pri != P_SURVIVAL
                    and (t.x, t.y) in handfed and view.hour < HANDFEED_UNTIL):
                continue
            zone_needs.setdefault(z, {}).setdefault(t.need, 0)
            zone_needs[z][t.need] += 1
    zone_carry = {}
    for idx, _, _ in me.units:
        z = uz[idx]
        for item in ("WHEAT", "FERTILIZER"):
            zone_carry.setdefault(z, {}).setdefault(item, 0)
            zone_carry[z][item] += view.unit_inv(idx).get(item, 0)
    shed_left = {item: view.shed.get(item, 0) for item in ("WHEAT", "FERTILIZER")}
    if EXEC["oneshed"]:
        # Consolidated provisioning: the farm's whole wheat and fertilizer
        # need is fetched by the fewest units nearest the shed, five wheat /
        # three fertilizer each, at hours 0-ONESHED_LAST_HOUR only (later
        # only for a starving animal). Measured: every one of our 12 hands
        # made its own 1-3 wheat pickup each morning (266-296 wheat pickups
        # a game for 350-373 feeds); the top agent makes 144 for 405.
        starving = sum(1 for t in tasks if t.op[0] == "FEED" and t.pri == P_SURVIVAL)
        for item, cap in (("WHEAT", 5), ("FERTILIZER", 3)):
            need = sum(v.get(item, 0) for v in zone_needs.values())
            need -= sum(view.unit_inv(i).get(item, 0) for i, _, _ in me.units)
            if view.hour > ONESHED_LAST_HOUR:
                if item != "WHEAT" or starving == 0:
                    continue
                need = min(need, starving)
            while need > 0 and free and shed_left[item] > 0:
                idx = min(free, key=lambda i: _dist(units[i][0], units[i][1],
                                                    *_nearest_shed_tile(units[i][0], units[i][1])))
                n = min(cap, need, shed_left[item])
                x, y = units[idx]
                tx, ty = _nearest_shed_tile(x, y)
                if (x, y) == (tx, ty):
                    actions[idx] = ["PICKUP", item, n]
                else:
                    step = _step_toward(x, y, tx, ty)
                    if not step:
                        break
                    actions[idx] = [step]
                shed_left[item] -= n
                need -= n
                free.discard(idx)
    else:
        for idx in sorted(free):
            z = uz[idx]
            for item in ("WHEAT", "FERTILIZER"):
                need = zone_needs.get(z, {}).get(item, 0) - zone_carry.get(z, {}).get(item, 0)
                n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)
                if n <= 0:
                    continue
                x, y = units[idx]
                tx, ty = _nearest_shed_tile(x, y)
                if (x, y) == (tx, ty):
                    actions[idx] = ["PICKUP", item, n]
                else:
                    step = _step_toward(x, y, tx, ty)
                    if not step:
                        continue
                    actions[idx] = [step]
                shed_left[item] -= n
                zone_carry[z][item] = zone_carry[z].get(item, 0) + n
                free.discard(idx)
                break

    _count_moves(actions, _seen, "morning")
    _seen = set(actions)

    # ---- 3. Zone rounds: each unit serves its own zone -----------------------
    zone_tasks = {}
    global_tasks = []
    for t in tasks:
        z = zone_of.get((t.x, t.y))
        if z is None:
            global_tasks.append(t)
        else:
            zone_tasks.setdefault(z, []).append(t)
    for z in zone_tasks:
        zone_tasks[z].sort(key=lambda t: (t.pri, t.y, t.x))

    # A unit carrying an ANIMAL places it FIRST, anywhere on the farm. Without
    # this the provisioning carrier fell back into its own zone's rounds and
    # hauled a cow around for days while the pasture sat empty (traced on
    # day 0-3). Wheat/fertilizer deliveries stay inside the zone round.
    need_tasks = [t for t in tasks if t.need in ("GOOSE", "COW", "SHEEP")]

    # SWEEP scoring: nearest task first, priority only as a tie-break. The
    # old (priority, distance) order walked a hand four tiles to a harvest
    # and four back to a water next door: 1.7 moves per action against the
    # leaders' ~1.0 (measured on our five worst worlds: 71-105 productive
    # actions a day vs their 105-159 with the same crew). Late in the day
    # survival tasks (dies tonight) jump the queue so the sweep never
    # strands them.
    late = view.hour >= SWEEP_DEADLINE_HOUR

    def pick(idx, pool, lo, hi, max_dist=None):
        """Nearest executable task in pool with lo <= pri < hi."""
        x, y = units[idx]
        inv = view.unit_inv(idx)
        best = None
        for task in pool:
            if not (lo <= task.pri < hi):
                continue
            key = (task.x, task.y, task.op[0])
            if key in claimed:
                continue
            if task.need and inv.get(task.need, 0) < 1:
                continue
            d = _dist(x, y, task.x, task.y)
            if max_dist is not None and d > max_dist:
                continue
            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            if EXEC["quadq"]:
                # priority queue: survival / harvest / animal chores before
                # tend / grow, nearest first inside the class
                score = (urgent, 0 if task.pri <= P_ANIMAL else 1, d, task.pri)
            else:
                score = (urgent, d, task.pri)
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None

    all_tasks = sorted(
        (t for pool in zone_tasks.values() for t in pool),
        key=lambda t: (t.pri, t.y, t.x),
    )

    # Pass 0 — the tile underfoot, ALL units, before anyone picks by zone.
    # Doing this inside pass 1's per-unit loop is not enough: `sorted(free)`
    # lets a DISTANT unit reach the task first through its own-zone pick,
    # claim it (`claimed` is set on the step TOWARD a task, not on arrival),
    # and the unit already standing on the tile is then refused. Measured on
    # the DIG residue: a unit stood on the weed at T, walked off, and the weed
    # was dug 14-37 turns later. (Under EXEC["chain"] this pass already ran
    # before the shuttles; units it served are no longer free.)
    for idx in sorted(free - _hold):
        _t = _underfoot(idx)
        if _t is not None and try_execute(idx, _t):
            free.discard(idx)

    # Pass 1 — deliveries, then the tile underfoot, then the unit's own zone.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = _underfoot(idx)
        if task is None:
            task = pick(idx, zone_tasks.get(uz[idx], []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "zone")
    _seen = set(actions)

    # Pass 2 — real work anywhere: animal-heavy zones (pastures cluster near
    # the shed) get help before anyone waters a plant that is not due.
    for idx in sorted(free):
        task = pick(idx, all_tasks + global_tasks, P_SURVIVAL, P_FILLER, max_dist=HELP_RADIUS)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "idlehelp")
    _seen = set(actions)

    # Pass 3 — filler work: own zone, then anywhere.
    for idx in sorted(free):
        task = pick(idx, zone_tasks.get(uz[idx], []), P_FILLER, 99)
        if task is None:
            task = pick(idx, all_tasks + global_tasks, P_FILLER, 99)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "filler")
    _seen = set(actions)

    # ---- 5. Survival override: nothing urgent goes unserved -----------------
    urgent = [
        t for t in all_tasks
        if t.pri == P_SURVIVAL and (t.x, t.y, t.op[0]) not in claimed
    ]
    if urgent:
        # Only genuinely idle units are pulled: re-targeting hands that are
        # already walking to their zone's work made every unfed animal or dry
        # plant a farm-wide emergency and hands paced between quadrants
        # (traced: 200 moves for 60 actions a day). Zone rounds already
        # serve P_SURVIVAL first inside each zone.
        movable = [idx for idx, op in actions.items() if op and op[0] == "PASS"] + list(free)
        for task in urgent:
            if not movable:
                break
            eligible = [
                i for i in movable
                if not task.need or view.unit_inv(i).get(task.need, 0) >= 1
            ]
            if not eligible:
                continue
            idx = min(eligible, key=lambda i: _dist(units[i][0], units[i][1], task.x, task.y))
            if try_execute(idx, task):
                movable.remove(idx)
                free.discard(idx)

    _count_moves(actions, _seen, "survival")
    _seen = set(actions)

    # ---- 6. Whoever is left: stage toward own zone / end-of-day shed --------
    for idx in list(free):
        x, y = units[idx]
        if sum(view.unit_inv(idx).values()) > 0 and view.hour >= TURNS_PER_DAY - 3:
            tx, ty = _nearest_shed_tile(x, y)
            step = _step_toward(x, y, tx, ty)
            actions[idx] = [step] if step else (["DROP"] if shed_room > 0 else ["PASS"])
        elif EXEC["nodrift"] and _unlocked(view, x, y):
            actions[idx] = ["PASS"]
        else:
            z = uz[idx]
            home = [tile for tile, zz in zone_of.items() if zz == z]
            if home and (x, y) not in home:
                hx, hy = min(home, key=lambda p: _dist(x, y, p[0], p[1]))
                step = _step_toward(x, y, hx, hy)
                actions[idx] = [step] if step else ["PASS"]
            elif not _unlocked(view, x, y):
                step = _step_toward(x, y, 4, 4)
                actions[idx] = [step] if step else ["PASS"]
            else:
                actions[idx] = ["PASS"]
        free.discard(idx)
    _count_moves(actions, _seen, "stage")

    if EXEC["h23"] and view.hour >= TURNS_PER_DAY - 1:
        # Units respawn at the shed at hour 0, so a move on the day's last
        # turn is wasted (5.5% of shipped moves; 0% of the top agent's).
        for idx, op in actions.items():
            if op and op[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
                actions[idx] = ["PASS"]

    return actions
