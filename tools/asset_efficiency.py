"""With the SAME crew, which farm does it manage better? Days 13-26 over K tape
worlds, per spec: unit-turns, productive ops, moves, idle, moves per productive
op; assets (plant tiles, animals) and how much of each is served per day
(watered at dusk, fed, cared, ripe one-time tiles left standing); income a day.
usage: PYTHONPATH=. uv run python tools/asset_efficiency.py K spec [spec...]"""
import sys, os, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
import plan_ledger as PL
from kagent.constants import CROPS, ANIMALS
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
PROD = {"PLANT", "WATER", "HARVEST", "FEED", "CARE", "FERTILIZE", "BUILD_PASTURE", "BUILD_COOP", "PLACE", "COLLECT_FERTILIZER", "DIG"}
LO, HI = 13, 26
K = int(sys.argv[1]); specs = sys.argv[2:]
def quad(x, y): return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
for spec in specs:
    c = collections.Counter(); byq = collections.defaultdict(collections.Counter); finals = []
    for k in range(K):
        steps, side, asks, name = PL.play(spec, k)
        finals.append(name.split("final ")[1].split(" (")[0])
        for d in range(LO, HI + 1):
            t0 = d * 24
            if t0 + 23 >= len(steps): break
            for t in range(t0, t0 + 24):
                a = steps[t][side].get("action") or {}
                ob = steps[t][side]["observation"]["farms"][side]
                pos = [tuple(ob["farmer"][:2])] + [tuple(h[:2]) for h in (ob.get("hands") or [])]
                units = [a.get("farmer") or ["PASS"]] + [h or ["PASS"] for h in (a.get("hands") or [])]
                for u, p in zip(units, pos):
                    op = u[0] if u else "PASS"; c["turns"] += 1
                    q = quad(*p); byq[q]["turns"] += 1
                    if op in MOVES: c["move"] += 1; byq[q]["move"] += 1
                    elif op in PROD: c["prod"] += 1; byq[q]["prod"] += 1; c[op] += 1
                    else: c["idle"] += 1
            eve = steps[t0 + 23][side]["observation"]["farms"][side]
            nxt = steps[min(t0 + 24, len(steps) - 1)][side]["observation"]["farms"][side]
            c["income_proxy"] += max(0.0, nxt["money"] - ob["money"])
            for y, row in enumerate(eve["tiles"]):
                for x, tile in enumerate(row):
                    if not isinstance(tile, dict): continue
                    q = quad(x, y)
                    if tile.get("kind") == "PLANT":
                        cd = CROPS.get(tile.get("crop"))
                        if not cd: continue
                        c["plant_days"] += 1; byq[q]["plant_days"] += 1
                        if tile.get("watered_today"): c["watered"] += 1; byq[q]["watered"] += 1
                        age = d - tile.get("planted_day", d)
                        if not cd["ongoing"] and age >= cd["max_yield_day"] and tile.get("yield_units", 0) > 0: c["ripe_left"] += 1
                    elif tile.get("animal"):
                        c["animal_days"] += 1; byq[q]["animal_days"] += 1
                        if tile.get("fed_today"): c["fed"] += 1
                        if tile.get("cared_today"): c["cared"] += 1
            c["quadrants"] += len(eve.get("unlocked_quadrants") or [1]); c["days"] += 1
    n = c["days"] / K
    print(f"\n=== {spec} ({K} worlds, days {LO}-{HI})  finals [{', '.join(finals)}]")
    print(f"  quadrants {c['quadrants']/c['days']:.2f} | per day: unit-turns {c['turns']/c['days']:.0f}, productive {c['prod']/c['days']:.0f} ({100*c['prod']/c['turns']:.0f}%), moves {c['move']/c['days']:.0f}, idle {c['idle']/c['days']:.0f}, moves per productive op {c['move']/max(1,c['prod']):.2f}")
    print(f"  assets a day: plant tiles {c['plant_days']/c['days']:.1f} watered {100*c['watered']/max(1,c['plant_days']):.0f}%, animals {c['animal_days']/c['days']:.1f} fed {100*c['fed']/max(1,c['animal_days']):.0f}% cared {100*c['cared']/max(1,c['animal_days']):.0f}%, ripe one-time left at dusk {c['ripe_left']/c['days']:.1f}")
    print(f"  ops a day: water {c['WATER']/c['days']:.0f} harvest {c['HARVEST']/c['days']:.0f} feed {c['FEED']/c['days']:.0f} care {c['CARE']/c['days']:.0f} plant {c['PLANT']/c['days']:.0f} fert {c['FERTILIZE']/c['days']:.0f} | assets per unit-turn {(c['plant_days']+3*c['animal_days'])/max(1,c['turns']):.2f} | money gained a day (positive deltas) ${c['income_proxy']/c['days']:,.0f}")
    print("  by quadrant: " + "  ".join(f"{q}: {byq[q]['turns']/c['days']:.0f} turns, {byq[q]['prod']/c['days']:.0f} prod, {byq[q]['move']/c['days']:.0f} moves, tiles {byq[q]['plant_days']/c['days']:.0f} watered {100*byq[q]['watered']/max(1,byq[q]['plant_days']):.0f}%, animals {byq[q]['animal_days']/c['days']:.0f}" for q in ("NW", "NE", "SW", "SE") if byq[q]["turns"]))
