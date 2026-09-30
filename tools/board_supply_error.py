"""Does the opponent's visible BOARD predict their daily supply better than the
history? On the 40 fresh replays (both seats visible), days 6-26, per product:
realised units the seat sold that day (requests capped at 30) vs (a) the corpus
prior, (b) Stage 2's blend of prior and EMA, (c) persistence, (d) the board:
features.supply_projection of THAT seat's farm at dawn over one day (held
yield + what ripens today), (e) board blended 50/50 with the EMA.
usage: PYTHONPATH=. uv run python tools/board_supply_error.py"""
import sys, os, json, glob, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from kagent import features, opp_profile as OP
from kagent.gamestate import FarmView
from kagent.constants import TURNS_PER_DAY
ITEMS = ("STRAWBERRY", "MILK", "WOOL")
def band(day):
    for k in OP.UNITS_PER_DAY["STRAWBERRY"]:
        lo, hi = (int(x) for x in k.split("-"))
        if lo <= day <= hi: return k
    return "21-26"
err = {m: collections.defaultdict(list) for m in ("prior", "blend", "persist", "board", "board+ema")}
level = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(REPO, "replays/now/*/*.json"))):
    st = json.load(open(f))["steps"]
    for seat in (0, 1):
        daily = {it: collections.Counter() for it in ITEMS}
        for t in range(1, len(st)):
            a = st[t][seat].get("action") or {}
            d = (t - 1) // TURNS_PER_DAY
            for m in a.get("market") or []:
                if m and m[0] == "SELL" and len(m) > 2 and m[1] in ITEMS:
                    daily[m[1]][d] += min(30, int(m[2]))
        for it in ITEMS:
            ema = None
            for d in range(6, 27):
                farm = FarmView(st[d * 24][seat]["observation"]["farms"][seat])
                board = features.supply_projection(farm, d, horizon=1).get(it, 0.0)
                real = daily[it][d]; prior = OP.UNITS_PER_DAY[it][band(d)]
                blend = prior if ema is None else 0.5 * (ema + prior)
                bema = board if ema is None else 0.5 * (board + ema)
                for k, v in (("prior", prior), ("blend", blend), ("persist", daily[it][d - 1]), ("board", board), ("board+ema", bema)):
                    err[k][it].append(abs(real - v))
                level[it].append(real)
                ema = real if ema is None else 0.75 * ema + 0.25 * real
print("40 replays x 2 seats, days 6-26, MAE in units a day")
print(f"{'item':11s} {'realised':>8s} {'prior':>6s} {'blend':>6s} {'persist':>7s} {'board':>6s} {'board+ema':>9s}")
for it in ITEMS:
    m = lambda k: sum(err[k][it]) / len(err[k][it])
    print(f"{it:11s} {sum(level[it])/len(level[it]):8.1f} {m('prior'):6.1f} {m('blend'):6.1f} {m('persist'):7.1f} {m('board'):6.1f} {m('board+ema'):9.1f}")
