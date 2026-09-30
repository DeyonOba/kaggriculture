"""Sizing ledger (2026-09-30): LEADERS=0 to skip the replays. PYTHONPATH=. uv run python tools/size_ledger.py champion [spec...]

How the plan is SIZED, day by day: dawn cash, the day's spend by category (list prices),
income earned within the day (money delta + spend), cash left at hour 23, hands vs standing work.
Leaders = winner seats of replays/now; ours = champion (h11) on tape worlds 0-3."""
import sys, os, json, glob, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
from kagent.constants import CROPS, ANIMALS, LAND_PRICES
from kagent import econ
BANDS = ((0, 0), (1, 2), (3, 6), (7, 10), (11, 15), (16, 22), (23, 29))
def standing(farm):
    plants = animals = bare = 0
    for row in farm["tiles"]:
        for c in row:
            if c is None: bare += 1
            elif isinstance(c, dict):
                if c.get("kind") == "PLANT": plants += 1
                elif c.get("animal"): animals += 1
    return plants, animals, bare
def day_rows(steps, seat):
    rows = []
    for d in range(30):
        t0 = d * 24; t1 = min(t0 + 23, len(steps) - 1)
        if t0 >= len(steps): break
        dawn = steps[t0][seat]["observation"]["farms"][seat]
        eve = steps[t1][seat]["observation"]["farms"][seat]
        nxt = steps[min(t0 + 24, len(steps) - 1)][seat]["observation"]["farms"][seat]
        spend = collections.Counter(); hires = 0; land = 0
        for t in range(t0, min(t0 + 24, len(steps))):
            a = steps[t][seat].get("action") or {}
            ob = steps[t][seat]["observation"]
            for m in a.get("market") or []:
                if not m: continue
                if m[0] == "BUY_SEED" and len(m) > 2 and m[1] in CROPS: spend["seeds"] += CROPS[m[1]]["seed"] * int(m[2])
                elif m[0] == "BUY_ANIMAL" and len(m) > 2 and m[1] in ANIMALS: spend["animals"] += ANIMALS[m[1]]["cost"] * int(m[2])
                elif m[0] == "HIRE" and len(m) > 1: hires += int(m[1]) if isinstance(m[1], int) else 1
                elif m[0] == "BUY_LAND": land += 1
                elif m[0] == "BUY_PRODUCT" and len(m) > 2: spend["feed"] += int(m[2]) * float(ob["market"]["prices"].get(m[1], 0))
        hands_after = len(steps[min(t0 + 2, len(steps) - 1)][seat]["observation"]["farms"][seat].get("hands") or [])
        spend["hires"] = econ.hire_total_cost(hands_after)
        owned = len(dawn.get("unlocked_quadrants") or [1])
        spend["land"] = sum(LAND_PRICES[max(0, min(2, owned - 1 + i))] for i in range(land)) if land else 0
        total = sum(spend.values())
        income = (nxt["money"] - dawn["money"]) + total
        plants, animals, bare = standing(dawn)
        rows.append(dict(day=d, dawn=dawn["money"], spend=total, seeds=spend["seeds"], animals=spend["animals"], hires=spend["hires"],
                         feed=spend["feed"], land=spend["land"], income=max(0.0, income), eve=eve["money"], hands=hands_after,
                         plants=plants, anim=animals, bare=bare))
    return rows
def table(title, allrows):
    print(f"\n=== {title}: {len(allrows)} games ===")
    print(f"{'days':>6s} {'dawn$':>7s} {'spend$':>7s} {'seeds':>6s} {'anim':>6s} {'hires':>6s} {'feed':>5s} {'land':>5s} {'income$':>8s} {'eve$':>7s} {'spend/dawn':>10s} {'>dawn':>6s} {'hands':>5s} {'plants':>6s} {'anim':>5s} {'bare':>4s} {'work/hand':>9s}")
    for lo, hi in BANDS:
        rs = [r for rows in allrows for r in rows if lo <= r["day"] <= hi]
        if not rs: continue
        n = len(rs); f = lambda k: sum(r[k] for r in rs) / n
        over = sum(1 for r in rs if r["spend"] > r["dawn"] + 1) / n
        ratio = sum(r["spend"] for r in rs) / max(1, sum(r["dawn"] for r in rs))
        work = sum(r["plants"] + 3 * r["anim"] for r in rs) / max(1, sum(1 + r["hands"] for r in rs))
        print(f"{lo:2d}-{hi:2d} {f('dawn'):7,.0f} {f('spend'):7,.0f} {f('seeds'):6,.0f} {f('animals'):6,.0f} {f('hires'):6,.0f} {f('feed'):5,.0f} {f('land'):5,.0f} {f('income'):8,.0f} {f('eve'):7,.0f} {ratio:10.2f} {over:6.0%} {f('hands'):5.1f} {f('plants'):6.1f} {f('anim'):5.1f} {f('bare'):4.1f} {work:9.1f}")
leaders = []
for f in (sorted(glob.glob("replays/now/*/*.json")) if os.environ.get("LEADERS", "1") == "1" else []):
    r = json.load(open(f)); st = r["steps"]; last = st[-1]
    w = 0 if (last[0].get("reward") or 0) >= (last[1].get("reward") or 0) else 1
    leaders.append(day_rows(st, w))
if leaders: table("LEADERS, winner seats (replays/now)", leaders)
import plan_ledger as PL
for spec in (sys.argv[1:] or ["champion"]):
    ours = []
    for k in range(4):
        steps, side, asks, name = PL.play(spec, k)
        ours.append(day_rows(steps, side))
    table(f"OURS, {spec} on tape worlds 0-3  [" + ", ".join(f"${r[-1]['eve']:,.0f}" for r in ours) + "]", ours)
print("\nwork/hand = (standing plant tiles + 3 x animals) per unit incl. the farmer; >dawn = share of days whose spend exceeded the dawn cash (funded within the day)")
