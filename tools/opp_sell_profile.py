"""Prior for the opponent's SELL timing: share of each product's units sold by
hour of day, and the mean units a day by day band, from both seats of the
recorded training sessions (id % 5 != 0; the held-out gate worlds are id % 5
== 0 and are never read). Written to kagent/data/opp_sell_profile.json for
kagent/selltiming.py. usage: PYTHONPATH=. uv run python tools/opp_sell_profile.py [n_sessions]"""
import sys, os, json, random, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from kagent.constants import PRODUCTS, TURNS_PER_DAY
BANDS = ((0, 5), (6, 12), (13, 20), (21, 26))
LAST_DAY = 26      # days 27-29 are the liquidation: 1,000-unit "sell everything" requests
ORDER_CAP = 30     # a SELL order is a REQUEST (not a fill): cap it at a shed's worth
idx = json.load(open(os.path.join(REPO, "rl", "datasets", "session_index.json")))
ids = sorted(k for k in idx if int(k) % 5 != 0)
random.Random(7).shuffle(ids)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
hour = {p: [0.0] * TURNS_PER_DAY for p in PRODUCTS}
daily = {p: {f"{lo}-{hi}": [0.0, 0] for lo, hi in BANDS} for p in PRODUCTS}
used = 0
for sid in ids[:n]:
    path = os.path.join(REPO, idx[sid]["path"])
    if not os.path.exists(path):
        continue
    sess = json.load(open(path))
    for seat in (0, 1):
        acts = sess["actions"][seat]
        per_day = collections.defaultdict(lambda: collections.Counter())
        for t, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            for m in a.get("market") or []:
                if m and m[0] == "SELL" and len(m) > 2 and m[1] in hour and t // TURNS_PER_DAY <= LAST_DAY:
                    units = min(ORDER_CAP, int(m[2]))
                    hour[m[1]][t % TURNS_PER_DAY] += units
                    per_day[t // TURNS_PER_DAY][m[1]] += units
        for d, c in per_day.items():
            for p, u in c.items():
                for lo, hi in BANDS:
                    if lo <= d <= hi:
                        daily[p][f"{lo}-{hi}"][0] += u
        for lo, hi in BANDS:
            for p in PRODUCTS:
                daily[p][f"{lo}-{hi}"][1] += (hi - lo + 1)
    used += 1
out = {"sessions": used, "hour_share": {}, "units_per_day": {}}
for p in PRODUCTS:
    tot = sum(hour[p]) or 1.0
    out["hour_share"][p] = [round(v / tot, 4) for v in hour[p]]
    out["units_per_day"][p] = {k: round(v[0] / max(1, v[1]), 3) for k, v in daily[p].items()}
json.dump(out, open(os.path.join(REPO, "kagent", "data", "opp_sell_profile.json"), "w"), indent=1)
print("sessions", used)
for p in ("STRAWBERRY", "MILK", "WOOL", "MELON", "WHEAT"):
    hs = out["hour_share"][p]
    print(f"  {p:10s} hours 0-3 {sum(hs[0:4]):.2f}  4-11 {sum(hs[4:12]):.2f}  12-19 {sum(hs[12:20]):.2f}  20-23 {sum(hs[20:24]):.2f} | units/day {out['units_per_day'][p]}")
