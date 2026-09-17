"""Are we leaving unlocked tiles EMPTY?

The LP says tiles bind on 19/30 days. Persistent empty squares would mean they
do not bind at all, and that we are sitting on unused capacity in the quadrants
we have already paid for.
"""
import os, sys, collections, statistics as st
sys.path.insert(0, os.getcwd())
import tools.fast_env as fe; fe.install()
from kaggle_environments import make

def quad(x, y, n):
    return ("NW" if x < n//2 and y < n//2 else
            "NE" if x >= n//2 and y < n//2 else
            "SW" if x < n//2 and y >= n//2 else "SE")

SHED = {(4,4),(4,5),(5,4),(5,5)}

def scan(agent, opp="pub-ahmed-v38", seeds=(71000,71001,71002)):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for sd in seeds:
        env = make("kaggriculture", configuration={"seed": sd}, debug=False)
        env.run([f"arena/agents/{agent}/main.py", f"arena/agents/{opp}/main.py"])
        seen=set()
        for stp in env.steps[:-1]:
            o = stp[0].observation
            d = o["day"]
            if d in seen or o["hour"] != 12: continue
            seen.add(d)
            tiles = o["farms"][0]["tiles"]; n=len(tiles)
            c = collections.Counter()
            for y,row in enumerate(tiles):
                for x,t in enumerate(row):
                    if (x,y) in SHED: continue
                    q = quad(x,y,n)
                    if t == "LOCKED": c[(q,"locked")] += 1
                    elif t is None: c[(q,"EMPTY")] += 1
                    elif isinstance(t,dict):
                        k = t.get("kind")
                        if "animal" in t: c[(q,"animal")] += 1
                        elif k == "PLANT": c[(q,"crop")] += 1
                        elif k == "WEED": c[(q,"weed")] += 1
                        else: c[(q,"structure")] += 1
            for k,v in c.items(): per[d][k].append(v)
    return per

for agent in ("tape-sel","champ-h8"):
    per = scan(agent)
    print(f"\n{agent}: UNLOCKED tiles that sit EMPTY (mean of 3 seeds, midday)\n")
    print(f"  {'day':>4}{'NW empty':>10}{'NE empty':>10}{'SW empty':>10}{'SE empty':>10}"
          f"{'crops':>8}{'animals':>9}{'weeds':>7}")
    for d in sorted(per):
        if d % 4 and d not in (1,29): continue
        m = lambda k: st.mean(per[d][k]) if per[d].get(k) else 0
        tot = lambda s: sum(m((q,s)) for q in ("NW","NE","SW","SE"))
        print(f"  {d:>4}{m(('NW','EMPTY')):>10.1f}{m(('NE','EMPTY')):>10.1f}"
              f"{m(('SW','EMPTY')):>10.1f}{m(('SE','EMPTY')):>10.1f}"
              f"{tot('crop'):>8.1f}{tot('animal'):>9.1f}{tot('weed'):>7.1f}")
