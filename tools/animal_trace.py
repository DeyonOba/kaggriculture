"""Opening trace, animal by animal (2026-09-30). PYTHONPATH=. uv run python tools/animal_trace.py leader replays/now/mmpq/episode-114976921-replay.json 0 | ours 0

Opening trace, animal by animal: when each animal was bought (cash at that
turn), placed (delay, tile, shed distance), first fed, first cared, first
harvested, and what it yielded by day 12; plus the day line (dawn cash, hands,
income, herd, plants, bare) for days 0-12. usage: animal_trace.py leader <replay> <seat> | ours <world>"""
import sys, os, json, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
from kagent.constants import ANIMALS, CROPS
from kagent import econ
mode = sys.argv[1]
if mode == "leader":
    steps = json.load(open(sys.argv[2]))["steps"]; seat = int(sys.argv[3]); label = f"LEADER {os.path.basename(sys.argv[2])} seat {seat}"
else:
    import plan_ledger as PL
    steps, seat, asks, name = PL.play("champion", int(sys.argv[2])); label = f"OURS h11 on {name}"
def farm(t): return steps[t][seat]["observation"]["farms"][seat]
def priv(t): return steps[t][seat]["observation"].get("private") or {}
def positions(t):
    f = farm(t)
    out = [tuple(f["farmer"][:2])]
    for h in f.get("hands") or []: out.append(tuple(h[:2]))
    return out
def acts(t):
    a = steps[t][seat].get("action") or {}
    return [a.get("farmer") or []] + list(a.get("hands") or []), a.get("market") or []
def sd(x, y): return min(abs(x - sx) + abs(y - sy) for sx, sy in ((4, 4), (5, 4), (4, 5), (5, 5)))
T = min(len(steps), 13 * 24)
# --- day line
print(f"\n=== {label} ===")
print(f"{'day':>3s} {'dawn$':>6s} {'hands':>5s} {'income$':>7s} {'spend$':>6s} {'herd@23':>7s} {'shed anim':>9s} {'plants':>6s} {'bare':>4s} {'struct':>6s} | buys (day)")
for d in range(0, 13):
    t0 = d * 24; t1 = min(t0 + 23, len(steps) - 1)
    if t0 >= len(steps): break
    f0, f1 = farm(t0), farm(t1)
    spend = 0.0; buys = collections.Counter()
    for t in range(t0, min(t0 + 24, len(steps))):
        units, mk = acts(t)
        ob = steps[t][seat]["observation"]
        for m in mk:
            if not m: continue
            if m[0] == "BUY_ANIMAL" and len(m) > 2: buys[m[1]] += int(m[2]); spend += ANIMALS[m[1]]["cost"] * int(m[2])
            elif m[0] == "BUY_SEED" and len(m) > 2: spend += CROPS[m[1]]["seed"] * int(m[2])
            elif m[0] == "BUY_PRODUCT" and len(m) > 2: spend += int(m[2]) * float(ob["market"]["prices"].get(m[1], 0))
            elif m[0] == "HIRE": spend += econ.hire_marginal_cost(farm(t).get("hires_today", 0))
            elif m[0] == "BUY_LAND": spend += 1000
    nxt = farm(min(t0 + 24, len(steps) - 1))
    income = nxt["money"] - f0["money"] + spend
    herd = sum(1 for row in f1["tiles"] for c in row if isinstance(c, dict) and c.get("animal"))
    structs = sum(1 for row in f1["tiles"] for c in row if isinstance(c, dict) and c.get("kind") in ("PASTURE", "COOP"))
    plants = sum(1 for row in f1["tiles"] for c in row if isinstance(c, dict) and c.get("kind") == "PLANT")
    bare = sum(1 for row in f1["tiles"] for c in row if c is None)
    shed_anim = sum(priv(t1).get("shed", {}).get(k, 0) for k in ANIMALS)
    hands = len(farm(min(t0 + 2, len(steps) - 1)).get("hands") or [])
    print(f"{d:3d} {f0['money']:6,.0f} {hands:5d} {income:7,.0f} {spend:6,.0f} {herd:7d} {shed_anim:9d} {plants:6d} {bare:4d} {structs:6d} | {dict(buys) or '-'}")
# --- animal by animal
placed = {}     # (x,y) -> dict
buy_queue = collections.defaultdict(list)   # kind -> [(t, cash)]
for t in range(0, T):
    units, mk = acts(t)
    for m in mk:
        if m and m[0] == "BUY_ANIMAL" and len(m) > 2:
            for _ in range(int(m[2])): buy_queue[m[1]].append((t, farm(t)["money"]))
    f = farm(t)
    for y, row in enumerate(f["tiles"]):
        for x, c in enumerate(row):
            if isinstance(c, dict) and c.get("animal") and (x, y) not in placed:
                kind = c["animal"]
                bt = buy_queue[kind].pop(0) if buy_queue[kind] else (None, None)
                placed[(x, y)] = {"kind": kind, "placed": t, "bought": bt[0], "cash_at_buy": bt[1], "feed": None, "care": None, "harv": None, "harvests": 0}
    pos = positions(t)
    for u, p in zip(units, pos):
        if not u or p not in placed: continue
        rec = placed[p]
        if u[0] == "FEED" and rec["feed"] is None: rec["feed"] = t
        if u[0] == "CARE" and rec["care"] is None: rec["care"] = t
        if u[0] == "HARVEST":
            rec["harvests"] += 1
            if rec["harv"] is None: rec["harv"] = t
def dh(t): return "-" if t is None else f"d{t // 24}h{t % 24:02d}"
print(f"\n{'#':>2s} {'kind':6s} {'bought':>7s} {'cash':>6s} {'placed':>7s} {'delay':>5s} {'tile':>6s} {'shed':>4s} {'1st feed':>8s} {'1st care':>8s} {'1st harv':>8s} {'harv<=d12':>9s}")
for i, ((x, y), r) in enumerate(sorted(placed.items(), key=lambda kv: kv[1]["placed"]), 1):
    delay = "-" if r["bought"] is None else f"{r['placed'] - r['bought']:d}t"
    cash = "-" if r["cash_at_buy"] is None else f"{r['cash_at_buy']:,.0f}"
    print(f"{i:2d} {r['kind']:6s} {dh(r['bought']):>7s} {cash:>6s} {dh(r['placed']):>7s} {delay:>5s} {str((x, y)):>6s} {sd(x, y):4d} {dh(r['feed']):>8s} {dh(r['care']):>8s} {dh(r['harv']):>8s} {r['harvests']:9d}")
# --- where day-0/1 money came from
sells = collections.Counter(); rev = 0.0
for t in range(0, 48):
    units, mk = acts(t)
    for m in mk:
        if m and m[0] == "SELL" and len(m) > 2: sells[m[1]] += int(m[2])
print("\nSELL requests days 0-1:", dict(sells))
