"""Stage 4: hands from the observed load (rlv3/DESIGN_V4.md). From the leaders'
winner seats: at each dawn the LOAD is observable -- harvests due in the next
three days (crop ages), plantings the bare ground can take, animals to feed
and care -- and the target is the hands they hired that day. Fit a linear
rule, report held-out R^2 and how it compares with CHAMP_HIRES on their days.
usage: PYTHONPATH=. uv run python tools/hire_load_fit.py"""
import sys, os, json, glob, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from kagent.constants import CROPS, ANIMALS
from kagent.planner import CHAMP_HIRES

def load_features(farm, day):
    harv3 = plants = 0
    for row in farm["tiles"]:
        for c in row:
            if not isinstance(c, dict): continue
            if c.get("kind") == "PLANT":
                cd = CROPS.get(c.get("crop"))
                if not cd: continue
                plants += 1
                age = day - c.get("planted_day", day)
                if cd["ongoing"]:
                    if age + 3 >= cd["first_yield_day"]: harv3 += 3 / max(1, cd["interval"])
                else:
                    if age < cd["max_yield_day"] <= age + 3: harv3 += 1
    animals = sum(1 for row in farm["tiles"] for c in row if isinstance(c, dict) and c.get("animal"))
    bare = sum(1 for row in farm["tiles"] for c in row if c is None)
    return dict(harv3=harv3, plants=plants, animals=animals, bare=bare, cash=farm["money"])

rows = []
for f in sorted(glob.glob(os.path.join(REPO, "replays/now/*/*.json"))):
    r = json.load(open(f)); st = r["steps"]; last = st[-1]
    w = 0 if (last[0].get("reward") or 0) >= (last[1].get("reward") or 0) else 1
    for d in range(0, 29):
        t0 = d * 24
        if t0 + 2 >= len(st): break
        farm = st[t0][w]["observation"]["farms"][w]
        hands = len(st[t0 + 2][w]["observation"]["farms"][w].get("hands") or [])
        x = load_features(farm, d); x["day"] = d; x["hands"] = hands; x["game"] = f
        rows.append(x)
games = sorted({r["game"] for r in rows})
def mat(rs):
    X = np.array([[1.0, r["harv3"], r["plants"], 3.0 * r["animals"], r["bare"], math.log1p(r["cash"]), r["day"]] for r in rs])
    y = np.array([r["hands"] for r in rs], dtype=float)
    return X, y
names = ["const", "harv3", "plants", "3*animals", "bare", "log_cash", "day"]
# leave-games-out CV (5 folds)
folds = [games[i::5] for i in range(5)]
sse = sst = 0.0; preds = {}
for k in range(5):
    tr = [r for r in rows if r["game"] not in folds[k]]; te = [r for r in rows if r["game"] in folds[k]]
    Xtr, ytr = mat(tr); Xte, yte = mat(te)
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    p = Xte @ beta
    sse += float(((yte - p) ** 2).sum()); sst += float(((yte - yte.mean()) ** 2).sum())
X, y = mat(rows); beta = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"rows {len(rows)} games {len(games)}; held-out R^2 {1 - sse / sst:.3f}")
print("coefficients:", {n: round(float(b), 3) for n, b in zip(names, beta)})
# the constant-schedule baseline on the same days
champ = np.array([CHAMP_HIRES[r["day"]] if r["day"] < len(CHAMP_HIRES) else 8 for r in rows], dtype=float)
print(f"CHAMP_HIRES vs their hands: R^2 {1 - float(((y - champ) ** 2).sum()) / sst:.3f}, mean their hands {y.mean():.2f} vs schedule {champ.mean():.2f}")
by_day = {}
for r, p in zip(rows, X @ beta):
    by_day.setdefault(r["day"], []).append((r["hands"], p, CHAMP_HIRES[r["day"]] if r["day"] < len(CHAMP_HIRES) else 8))
print(" day  their hands  fitted  CHAMP_HIRES")
for d in sorted(by_day):
    v = by_day[d]
    print(f"{d:4d} {np.mean([a for a, _, _ in v]):11.1f} {np.mean([b for _, b, _ in v]):7.1f} {v[0][2]:12d}")
json.dump({"names": names, "beta": [float(b) for b in beta]}, open(os.path.join(REPO, "kagent", "data", "hire_load_fit.json"), "w"), indent=1)
