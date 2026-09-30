"""The precondition DESIGN_V4 set for any filter: how wrong is the opponent-supply
forecast that Stage 2 uses? Per product and day (6-26), on held-out sessions
(id % 5 == 0, never used for the prior), compare the realised units a seat
sold that day with (a) the corpus prior for the day band, (b) the in-game
blend Stage 2 uses (0.5 * (EMA_0.25 of past days + prior)), (c) persistence
(yesterday's units). MAE in units a day. usage: PYTHONPATH=. uv run python tools/supply_forecast_error.py [n]"""
import sys, os, json, random, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from kagent import opp_profile as OP
from kagent.constants import TURNS_PER_DAY
ITEMS = ("STRAWBERRY", "MILK", "WOOL")
idx = json.load(open(os.path.join(REPO, "rl", "datasets", "session_index.json")))
ids = sorted(k for k in idx if int(k) % 5 == 0)
random.Random(3).shuffle(ids)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
def band(day):
    for k in OP.UNITS_PER_DAY["STRAWBERRY"]:
        lo, hi = (int(x) for x in k.split("-"))
        if lo <= day <= hi: return k
    return "21-26"
err = {m: collections.defaultdict(list) for m in ("prior", "blend", "persist")}
level = collections.defaultdict(list)
used = 0
for sid in ids[:n]:
    path = os.path.join(REPO, idx[sid]["path"])
    if not os.path.exists(path): continue
    sess = json.load(open(path)); used += 1
    for seat in (0, 1):
        daily = {it: collections.Counter() for it in ITEMS}
        for t, a in enumerate(sess["actions"][seat]):
            if not isinstance(a, dict): continue
            d = t // TURNS_PER_DAY
            if d > 26: break
            for m in a.get("market") or []:
                if m and m[0] == "SELL" and len(m) > 2 and m[1] in ITEMS:
                    daily[m[1]][d] += min(30, int(m[2]))
        for it in ITEMS:
            ema = None
            for d in range(6, 27):
                real = daily[it][d]; prior = OP.UNITS_PER_DAY[it][band(d)]
                blend = prior if ema is None else 0.5 * (ema + prior)
                err["prior"][it].append(abs(real - prior)); err["blend"][it].append(abs(real - blend))
                err["persist"][it].append(abs(real - daily[it][d - 1])); level[it].append(real)
                ema = real if ema is None else 0.75 * ema + 0.25 * real
print(f"held-out sessions {used}, both seats, days 6-26 (MAE in units a day; mean realised in brackets)")
for it in ITEMS:
    mean = sum(level[it]) / len(level[it])
    print(f"  {it:10s} realised {mean:5.1f} | prior {sum(err['prior'][it])/len(err['prior'][it]):5.1f}  blend(Stage 2) {sum(err['blend'][it])/len(err['blend'][it]):5.1f}  persistence {sum(err['persist'][it])/len(err['persist'][it]):5.1f}")
