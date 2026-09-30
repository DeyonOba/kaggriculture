"""Stage 5: the melon race, from the replays where both boards are visible.
Per game and seat: melons standing by day 1 (the commitment the opponent can
read), hands, the day the seat's melons are cleared (tiles fall to 0 after the
peak), melon units sold and the revenue at the exact identity on the observed
inventory. Who clears first, and what the second seat's melons fetch.
usage: PYTHONPATH=. uv run python tools/melon_race_fit.py"""
import sys, os, json, glob, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from kagent import econ

def melons(farm):
    return sum(1 for row in farm["tiles"] for c in row if isinstance(c, dict) and c.get("kind") == "PLANT" and c.get("crop") == "MELON")

rows = []
for f in sorted(glob.glob(os.path.join(REPO, "replays/now/*/*.json"))):
    r = json.load(open(f)); st = r["steps"]; last = st[-1]
    w = 0 if (last[0].get("reward") or 0) >= (last[1].get("reward") or 0) else 1
    per_seat = {}
    for seat in (0, 1):
        by_day = [melons(st[min(d * 24 + 23, len(st) - 1)][seat]["observation"]["farms"][seat]) for d in range(0, 20)]
        n1 = by_day[1]; peak = max(by_day[:12]); peak_day = by_day.index(peak)
        clear = next((d for d in range(peak_day, 20) if by_day[d] <= max(0, peak // 4)), None)
        hands = [len(st[min(d * 24 + 2, len(st) - 1)][seat]["observation"]["farms"][seat].get("hands") or []) for d in range(8, 13)]
        units = 0; rev = 0.0; first_sale = None
        for t in range(1, len(st)):
            a = st[t][seat].get("action") or {}
            prev = st[t - 1][seat]["observation"]
            priv = prev.get("private") or {}
            # the engine applies DROP before the market in the same step, so a
            # unit's load sells the turn it is dropped: cap fills at shed + pockets
            shed = priv.get("shed", {}).get("MELON", 0) + sum(i.get("MELON", 0) for i in priv.get("inventories", []))
            for m in a.get("market") or []:
                if m and m[0] == "SELL" and m[1] == "MELON" and len(m) > 2:
                    n = min(int(m[2]), shed); shed -= n
                    if n <= 0: continue
                    inv = prev["market"]["inventory"].get("MELON", 0)
                    units += n; rev += econ.sell_revenue("MELON", inv, n)
                    if first_sale is None: first_sale = (t - 1) // 24
        per_seat[seat] = dict(n1=n1, peak=peak, clear=clear, hands=sum(hands) / len(hands), units=units, rev=rev, first_sale=first_sale, win=(seat == w))
    for seat in (0, 1):
        me, op = per_seat[seat], per_seat[1 - seat]
        rows.append(dict(game=os.path.basename(f), seat=seat, **{k: me[k] for k in me}, opp_n1=op["n1"], opp_clear=op["clear"], opp_units=op["units"], opp_rev=op["rev"],
                         first=(me["clear"] is not None and (op["clear"] is None or me["clear"] < op["clear"]))))
print(f"{'game':30s} {'seat':>4s} {'win':>3s} {'n1':>3s} {'peak':>4s} {'clear':>5s} {'1st sale':>8s} {'hands8-12':>9s} {'units':>5s} {'$/unit':>6s} {'rev':>7s} | {'opp n1':>6s} {'opp clear':>9s} {'opp $/u':>7s} {'first?':>6s}")
for x in sorted(rows, key=lambda x: (x["game"], x["seat"])):
    pu = x["rev"] / x["units"] if x["units"] else 0; opu = x["opp_rev"] / x["opp_units"] if x["opp_units"] else 0
    print(f"{x['game'][:30]:30s} {x['seat']:4d} {'W' if x['win'] else '':>3s} {x['n1']:3d} {x['peak']:4d} {str(x['clear']):>5s} {str(x['first_sale']):>8s} {x['hands']:9.1f} {x['units']:5d} {pu:6.0f} {x['rev']:7,.0f} | {x['opp_n1']:6d} {str(x['opp_clear']):>9s} {opu:7.0f} {'yes' if x['first'] else 'no':>6s}")
# summary: does clearing first pay, and what commitment wins the race
firsts = [x for x in rows if x["first"]]; seconds = [x for x in rows if not x["first"] and x["clear"] is not None]
def mean(v): return sum(v) / max(1, len(v))
print(f"\nclear first: n {len(firsts)}, melon $/unit {mean([x['rev']/x['units'] for x in firsts if x['units']]):.0f}, revenue {mean([x['rev'] for x in firsts]):,.0f}, n1 {mean([x['n1'] for x in firsts]):.1f}, hands {mean([x['hands'] for x in firsts]):.1f}, win rate {mean([x['win'] for x in firsts]):.2f}")
print(f"clear second: n {len(seconds)}, melon $/unit {mean([x['rev']/x['units'] for x in seconds if x['units']]):.0f}, revenue {mean([x['rev'] for x in seconds]):,.0f}, n1 {mean([x['n1'] for x in seconds]):.1f}, hands {mean([x['hands'] for x in seconds]):.1f}, win rate {mean([x['win'] for x in seconds]):.2f}")
by_n = collections.defaultdict(list)
for x in rows: by_n[min(x["n1"], 14) // 2 * 2].append(x)
print("\nby melons standing at day 1 (both seats):")
for k in sorted(by_n):
    v = by_n[k]; print(f"  n1 {k:2d}-{k+1:2d}: n {len(v):2d}, first {mean([x['first'] for x in v]):.2f}, clear day {mean([x['clear'] for x in v if x['clear'] is not None]):.1f}, $/unit {mean([x['rev']/x['units'] for x in v if x['units']]):.0f}, revenue {mean([x['rev'] for x in v]):,.0f}, win {mean([x['win'] for x in v]):.2f}")
