"""Productive work per tile, from game traces (tools/game_trace.py) and
recorded replays: the mechanism checks for the executor rules in
kagent/scheduler.EXEC and for the per-quadrant queue.

    PYTHONPATH=. uv run python tools/tile_metrics.py \\
        --trace base=/tmp/base.json --trace rules=/tmp/rules.json \\
        --replay mmpq=replays/mmpq/xxx.json [--out metrics.json]

Per farm (our seat in a trace; the named team's seat in a replay):
  one-time harvests   units per HARVEST by crop (engine: 1 at planting + 1
                      per WATER inside the bonus window, x2 fertilized), and
                      the share harvested on the window's last day UNWATERED
                      (the forfeited unit)
  replant gap         decision steps from a one-time HARVEST to the next
                      PLANT on that tile; share within 1 step; median
  chores per visit    tile ops per stay on a tile (a stay = consecutive
                      steps at one position with >= 1 tile op)
  moves / productive  moves per tile-or-shed op
  idle tile share     (empty + weed + empty structure) / unlocked, at hour 0,
                      mean over days 1-29; weed tile-days
  water no effect     WATER on an already-watered tile, or a filler WATER on
                      a one-time crop outside its window with no dry day
  hour-23 moves       moves on the day's last turn (units respawn at hour 0)
  pickups per feed
  layout (day 15)     animals / plants per quadrant, mean Manhattan distance
                      to the nearest shed-access tile
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from game_trace import tile_code, CROPS_L, ANIMALS_L  # noqa: E402

SHED = [(4, 4), (5, 4), (4, 5), (5, 5)]
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
TILE_OPS = {"WATER", "HARVEST", "PLANT", "FEED", "CARE", "FERTILIZE", "DIG",
            "BUILD_COOP", "BUILD_PASTURE", "PLACE", "COLLECT_FERTILIZER"}
SHED_OPS = {"PICKUP", "DROP"}
ONE_TIME = {0: "WHEAT", 1: "CARROT", 4: "MELON"}
MAX_YIELD_DAY = {0: 4, 1: 3, 4: 12}
MAX_YIELD = {0: 6, 1: 4, 4: 6}


def _window(ci):
    myd = MAX_YIELD_DAY[ci]
    return (myd + 1) // 2, myd


def shed_dist(x, y):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in SHED)


def quad_of(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


# ---------------------------------------------------------------- loaders
def load_trace(path):
    """-> list of steps {day, hour, tiles, units: [(x, y)], acts: [op...]}, final."""
    with open(path) as fh:
        d = json.load(fh)
    steps = []
    for s in d["steps"]:
        me = s["me"]
        act = s["act"]["me"] or {}
        acts = [act.get("farmer") or ["PASS"]] + [h or ["PASS"] for h in (act.get("hands") or [])]
        units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
        steps.append({"day": s["day"], "hour": s["hour"], "tiles": me["tiles"],
                      "units": units, "acts": acts})
    return steps, d["final"]


def load_replay(path, seat=None, team="M & M & P & Q"):
    with open(path) as fh:
        d = json.load(fh)
    if seat is None:
        tn = d["info"].get("TeamNames")
        if isinstance(tn, str):
            tn = ast.literal_eval(tn)
        seat = list(tn).index(team)
    raw = d["steps"]
    steps = []
    for k in range(len(raw) - 1):
        obs = raw[k][0]["observation"]
        farm = obs["farms"][seat]
        day = int(obs["day"])
        act = raw[k + 1][seat].get("action") or {}
        acts = [act.get("farmer") or ["PASS"]] + [h or ["PASS"] for h in (act.get("hands") or [])]
        units = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands") or []]
        steps.append({"day": day, "hour": int(obs["hour"]),
                      "tiles": [[tile_code(c, day) for c in row] for row in farm["tiles"]],
                      "units": units, "acts": acts})
    fin = [float(raw[-1][s].get("reward") or 0.0) for s in range(2)]
    return steps, {"me": fin[seat], "opp": fin[1 - seat], "win": fin[seat] > fin[1 - seat]}


# ---------------------------------------------------------------- metrics
def analyse(steps, final):
    n = len(steps)
    ops = Counter()
    harvest_units = defaultdict(list)     # crop -> units per harvest
    lastday_unwatered = Counter()         # crop -> harvests on the window's last day, unwatered
    lastday_total = Counter()
    pending_replant = {}                  # (x, y) -> harvest step
    replant_gaps = []
    never_replanted = 0
    visits_ops = []
    water_bad = 0
    n_water = 0
    h23_moves = 0
    moves = 0
    productive = 0

    for k, s in enumerate(steps):
        tiles = s["tiles"]
        for (x, y), op in zip(s["units"], s["acts"]):
            if not op:
                continue
            o = op[0]
            ops[o] += 1
            if o in MOVES:
                moves += 1
                if s["hour"] == 23:
                    h23_moves += 1
                continue
            if o in TILE_OPS or o in SHED_OPS:
                productive += 1
            if not (0 <= x < 10 and 0 <= y < 10):
                continue
            t = tiles[y][x]
            if o == "HARVEST" and isinstance(t, list) and t[0] == 4 and t[1] in ONE_TIME and t[3] > 0:
                ci, age, yld, w = t[1], t[2], t[3], t[4]
                harvest_units[ONE_TIME[ci]].append(yld)
                lo, hi = _window(ci)
                if age == hi and yld < MAX_YIELD[ci]:
                    lastday_total[ONE_TIME[ci]] += 1
                    if not w:
                        lastday_unwatered[ONE_TIME[ci]] += 1
                pending_replant[(x, y)] = k
            elif o == "PLANT" and t == 1 and (x, y) in pending_replant:
                replant_gaps.append(k - pending_replant.pop((x, y)))
            elif o == "WATER" and isinstance(t, list) and t[0] == 4:
                n_water += 1
                ci, age, w, u = t[1], t[2], t[4], t[5]
                if w:
                    water_bad += 1
                elif ci in ONE_TIME and u == 0:
                    lo, hi = _window(ci)
                    if not (lo <= age <= hi):
                        water_bad += 1
    never_replanted = len(pending_replant)

    # visits: per day, per unit index, consecutive steps at one position
    by_day = defaultdict(list)
    for k, s in enumerate(steps):
        by_day[s["day"]].append(s)
    for day, ss in by_day.items():
        nu = max(len(s["units"]) for s in ss)
        for ui in range(nu):
            pos = None
            cnt = 0
            for s in ss:
                if ui >= len(s["units"]):
                    continue
                p = s["units"][ui]
                op = s["acts"][ui][0] if ui < len(s["acts"]) and s["acts"][ui] else "PASS"
                if p != pos:
                    if pos is not None and cnt > 0:
                        visits_ops.append(cnt)
                    pos, cnt = p, 0
                if op in TILE_OPS:
                    cnt += 1
            if pos is not None and cnt > 0:
                visits_ops.append(cnt)

    # land use at hour 0
    idle_shares = []
    empty_shares = []
    weed_shares = []
    estruct_shares = []
    weed_days = 0
    layout = None
    for s in steps:
        if s["hour"] != 0 or s["day"] < 1:
            continue
        flat = [t for row in s["tiles"] for t in row]
        unlocked = sum(1 for t in flat if t != 0)
        empty = sum(1 for t in flat if t == 1)
        weeds = sum(1 for t in flat if t == 2)
        estruct = sum(1 for t in flat if isinstance(t, list) and t[0] == 3)
        if unlocked:
            idle_shares.append((empty + weeds + estruct) / unlocked)
            empty_shares.append(empty / unlocked)
            weed_shares.append(weeds / unlocked)
            estruct_shares.append(estruct / unlocked)
        weed_days += weeds
        if s["day"] == 15:
            an = Counter()
            pl = Counter()
            ad = []
            pd = []
            for y, row in enumerate(s["tiles"]):
                for x, t in enumerate(row):
                    if isinstance(t, list) and t[0] == 5:
                        an[quad_of(x, y)] += 1
                        ad.append(shed_dist(x, y))
                    elif isinstance(t, list) and t[0] == 4:
                        pl[quad_of(x, y)] += 1
                        pd.append(shed_dist(x, y))
            layout = {"animals_per_quadrant": dict(an), "plants_per_quadrant": dict(pl),
                      "animal_shed_dist": st.mean(ad) if ad else None,
                      "plant_shed_dist": st.mean(pd) if pd else None}

    def r(a, b):
        return a / b if b else None

    return {
        "final_money": final["me"], "opp_money": final["opp"], "win": bool(final["win"]),
        "ops": dict(ops),
        "one_time_units_per_harvest": {c: r(sum(v), len(v)) for c, v in harvest_units.items()},
        "one_time_harvests": {c: len(v) for c, v in harvest_units.items()},
        "one_time_units_total": sum(sum(v) for v in harvest_units.values()),
        "lastday_unwatered_harvest_share": {c: r(lastday_unwatered[c], lastday_total[c])
                                            for c in lastday_total},
        "replant_gap_median": st.median(replant_gaps) if replant_gaps else None,
        "replant_within_1_step": r(sum(1 for g in replant_gaps if g <= 1), len(replant_gaps)),
        "replant_n": len(replant_gaps), "never_replanted": never_replanted,
        "chores_per_visit": r(sum(visits_ops), len(visits_ops)),
        "visits_ge2_share": r(sum(1 for v in visits_ops if v >= 2), len(visits_ops)),
        "moves_per_productive": r(moves, productive),
        "productive_ops": productive, "moves": moves,
        "idle_share_mean": st.mean(idle_shares) if idle_shares else None,
        "idle_empty_share": st.mean(empty_shares) if empty_shares else None,
        "idle_weed_share": st.mean(weed_shares) if weed_shares else None,
        "idle_structure_share": st.mean(estruct_shares) if estruct_shares else None,
        "weed_tile_days": weed_days,
        "water_no_effect_share": r(water_bad, n_water), "water_ops": n_water,
        "h23_moves": h23_moves, "h23_move_share": r(h23_moves, moves),
        "pickups_per_feed": r(ops["PICKUP"], ops["FEED"]),
        "harvest_ops": ops["HARVEST"], "plant_ops": ops["PLANT"],
        "layout_day15": layout,
    }


ROWS = [
    ("final money", "final_money", "{:,.0f}"), ("opponent money", "opp_money", "{:,.0f}"),
    ("win", "win", "{}"),
    ("wheat units / harvest", ("one_time_units_per_harvest", "WHEAT"), "{:.2f}"),
    ("carrot units / harvest", ("one_time_units_per_harvest", "CARROT"), "{:.2f}"),
    ("melon units / harvest", ("one_time_units_per_harvest", "MELON"), "{:.2f}"),
    ("one-time units total", "one_time_units_total", "{}"),
    ("last-day unwatered harvest share, wheat", ("lastday_unwatered_harvest_share", "WHEAT"), "{:.2f}"),
    ("last-day unwatered harvest share, carrot", ("lastday_unwatered_harvest_share", "CARROT"), "{:.2f}"),
    ("replant within 1 step", "replant_within_1_step", "{:.2f}"),
    ("replant gap median (steps)", "replant_gap_median", "{}"),
    ("never replanted", "never_replanted", "{}"),
    ("chores per visit", "chores_per_visit", "{:.2f}"),
    ("visits with >= 2 chores", "visits_ge2_share", "{:.2f}"),
    ("moves per productive op", "moves_per_productive", "{:.2f}"),
    ("productive ops", "productive_ops", "{}"), ("moves", "moves", "{}"),
    ("harvests", "harvest_ops", "{}"), ("plantings", "plant_ops", "{}"),
    ("idle share of unlocked tiles", "idle_share_mean", "{:.3f}"),
    ("  of which bare", "idle_empty_share", "{:.3f}"),
    ("  of which weed", "idle_weed_share", "{:.3f}"),
    ("  of which empty structure", "idle_structure_share", "{:.3f}"),
    ("weed tile-days", "weed_tile_days", "{}"),
    ("water ops with no effect", "water_no_effect_share", "{:.2f}"),
    ("hour-23 move share", "h23_move_share", "{:.3f}"),
    ("pickups per feed", "pickups_per_feed", "{:.2f}"),
    ("animal shed distance (day 15)", ("layout_day15", "animal_shed_dist"), "{:.2f}"),
    ("plant shed distance (day 15)", ("layout_day15", "plant_shed_dist"), "{:.2f}"),
    ("animals per quadrant (day 15)", ("layout_day15", "animals_per_quadrant"), "{}"),
    ("plants per quadrant (day 15)", ("layout_day15", "plants_per_quadrant"), "{}"),
]


def _get(m, key):
    if isinstance(key, tuple):
        v = m.get(key[0]) or {}
        return v.get(key[1]) if isinstance(v, dict) else None
    return m.get(key)


def print_table(results):
    labels = list(results)
    w = max(len(l) for l in labels) + 2
    print(f"{'metric':<42}" + "".join(f"{l:>{w}}" for l in labels))
    for name, key, fmt in ROWS:
        cells = []
        for l in labels:
            v = _get(results[l], key)
            cells.append("-" if v is None else fmt.format(v))
        print(f"{name:<42}" + "".join(f"{c:>{w}}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", action="append", default=[], help="label=path (game_trace.py JSON)")
    ap.add_argument("--replay", action="append", default=[],
                    help="label=path[:seat] (Kaggle replay; seat defaults to M & M & P & Q's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = {}
    for spec in args.trace:
        label, path = spec.split("=", 1)
        results[label] = analyse(*load_trace(path))
    for spec in args.replay:
        label, rest = spec.split("=", 1)
        seat = None
        if rest.count(":") and rest.rsplit(":", 1)[1].isdigit():
            rest, s = rest.rsplit(":", 1)
            seat = int(s)
        results[label] = analyse(*load_replay(rest, seat))
    if not results:
        ap.error("give at least one --trace or --replay")
    print_table(results)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print("->", args.out)


if __name__ == "__main__":
    main()
