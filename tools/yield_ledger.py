"""Yield ledger (2026-09-30): LEADERS=0 to skip the replays. PYTHONPATH=. uv run python tools/yield_ledger.py champion [spec...]

Yield delivery: of the farm that is STANDING, how much of its daily potential is
worked? Read at hour 23 of each day: plant-days watered, animal-days fed and cared,
fertiliser left uncollected, production left clipped at max_held, ripe one-time tiles
left unharvested. Leaders = winner seats of replays/now; ours = specs on tape worlds 0-3."""
import sys, os, json, glob, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
from kagent.constants import CROPS, ANIMALS
BANDS = ((1, 6), (7, 12), (13, 20), (21, 28))
def scan(steps, seat):
    acc = collections.Counter()
    for d in range(1, 29):
        t = d * 24 + 23
        if t >= len(steps): break
        farm = steps[t][seat]["observation"]["farms"][seat]
        for row in farm["tiles"]:
            for c in row:
                if not isinstance(c, dict): continue
                if c.get("kind") == "PLANT":
                    cd = CROPS.get(c.get("crop"))
                    if not cd: continue
                    age = d - c.get("planted_day", d)
                    acc[(d, "plant_days")] += 1
                    if c.get("watered_today"): acc[(d, "watered")] += 1
                    if c.get("fertilized_until_day", -1) >= d: acc[(d, "fertilised")] += 1
                    if not cd["ongoing"] and age >= cd["max_yield_day"] and c.get("yield_units", 0) > 0: acc[(d, "ripe_unharvested")] += 1
                    if cd["ongoing"] and c.get("yield_units", 0) >= cd["max_yield"]: acc[(d, "plant_clipped")] += 1
                elif c.get("animal"):
                    ad = ANIMALS.get(c["animal"], {})
                    acc[(d, "animal_days")] += 1
                    if c.get("fed_today"): acc[(d, "fed")] += 1
                    if c.get("cared_today"): acc[(d, "cared")] += 1
                    if c.get("fertilizer_available"): acc[(d, "fert_uncollected")] += 1
                    if c.get("yield_units", 0) >= ad.get("max_held", 99): acc[(d, "animal_clipped")] += 1
    return acc
def table(title, accs):
    n = len(accs)
    print(f"\n=== {title} (n={n}) ===")
    print(f"{'days':>6s} {'plant-days':>10s} {'watered%':>8s} {'fert%':>6s} {'ripe-unharv':>11s} {'clipped':>7s} | {'animal-days':>11s} {'fed%':>5s} {'cared%':>6s} {'fert-uncoll':>11s} {'clipped':>7s}")
    for lo, hi in BANDS:
        c = collections.Counter()
        for a in accs:
            for (d, k), v in a.items():
                if lo <= d <= hi: c[k] += v
        pd, ad = max(1, c["plant_days"]), max(1, c["animal_days"])
        print(f"{lo:2d}-{hi:2d} {c['plant_days']/n:10.1f} {100*c['watered']/pd:7.0f}% {100*c['fertilised']/pd:5.0f}% {c['ripe_unharvested']/n:11.1f} {c['plant_clipped']/n:7.1f} | {c['animal_days']/n:11.1f} {100*c['fed']/ad:4.0f}% {100*c['cared']/ad:5.0f}% {c['fert_uncollected']/n:11.1f} {c['animal_clipped']/n:7.1f}")
leaders = []
for f in sorted(glob.glob("replays/now/*/*.json")):
    r = json.load(open(f)); st = r["steps"]; last = st[-1]
    w = 0 if (last[0].get("reward") or 0) >= (last[1].get("reward") or 0) else 1
    leaders.append(scan(st, w))
table("LEADERS, winner seats", leaders)
import plan_ledger as PL
for spec in (sys.argv[1:] or ["champion"]):
    ours = []
    for k in range(4):
        steps, side, asks, name = PL.play(spec, k)
        ours.append(scan(steps, side))
    table(f"OURS, {spec}, tape worlds 0-3", ours)
print("\nper game: plant-days and animal-days are standing counts summed over the band's days; watered/fed/cared read at hour 23; ripe-unharv = one-time tiles at max yield still standing at day end; clipped = production sitting at the cap")
