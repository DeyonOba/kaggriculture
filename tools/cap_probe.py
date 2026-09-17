"""Unmeasured executor defects, from docs/spatial_program.md section 1.

The sharpest candidate is the CAP. The engine writes
    tile["yield_units"] = min(max_held, yield_units + base + bonus)
so an animal already at max_held DISCARDS the night's production. Same for
crops at max_yield. Every animal-day spent at the cap is output thrown away,
and it is invisible in the score.

Also measures ripe->harvested latency and per-unit work span, both listed as
unmeasured.
"""
import os, sys, collections, statistics as st
sys.path.insert(0, os.getcwd())
import tools.fast_env as fe; fe.install()
from kaggle_environments import make
from kagent.constants import ANIMALS, CROPS

def scan(agent, opp="pub-ahmed-v38", seeds=(71000,71001,71002)):
    at_cap=collections.Counter(); animal_days=collections.Counter()
    crop_cap=0; crop_days=0
    ripe_age=[]   # how long a ripe tile stays unharvested
    spans=[]
    for sd in seeds:
        env=make("kaggriculture",configuration={"seed":sd},debug=False)
        env.run([f"arena/agents/{agent}/main.py", f"arena/agents/{opp}/main.py"])
        ripe_since={}
        for stp in env.steps[:-1]:
            o=stp[0].observation
            if o["hour"]!=23: continue
            d=o["day"]
            for y,row in enumerate(o["farms"][0]["tiles"]):
                for x,t in enumerate(row):
                    if not isinstance(t,dict): continue
                    if "animal" in t:
                        a=ANIMALS[t["animal"]]; animal_days[t["animal"]]+=1
                        if t.get("yield_units",0) >= a["max_held"]:
                            at_cap[t["animal"]]+=1
                    elif t.get("kind")=="PLANT":
                        c=CROPS.get(t["crop"])
                        if c:
                            crop_days+=1
                            if t.get("yield_units",0) >= c["max_yield"]: crop_cap+=1
                    # ripe latency
                    k=(x,y)
                    if t.get("yield_units",0)>0:
                        ripe_since.setdefault(k,d)
                    elif k in ripe_since:
                        ripe_age.append(d-ripe_since.pop(k))
        # per-unit work span (bounding box of tiles worked in a day)
        from tools.commit_probe import unit_ops, unit_pos, MOVES
        byday=collections.defaultdict(lambda: collections.defaultdict(list))
        for t in range(1,len(env.steps)):
            pos,day=unit_pos(env.steps[t],0)
            for i,op in enumerate(unit_ops(env.steps[t],0)):
                if i<len(pos) and op not in MOVES and op!="PASS":
                    byday[day][i].append(pos[i])
        for day,units in byday.items():
            for i,ps in units.items():
                if len(ps)>1:
                    spans.append((max(p[0] for p in ps)-min(p[0] for p in ps)+1)
                                 *(max(p[1] for p in ps)-min(p[1] for p in ps)+1))
    return at_cap, animal_days, crop_cap, crop_days, ripe_age, spans

for agent in ("champ-h8","tape-sel"):
    ac,ad,cc,cd,ra,sp = scan(agent)
    tot_cap=sum(ac.values()); tot_days=sum(ad.values())
    print(f"\n{agent}: production lost to the CAP (3 seeds)\n")
    print(f"  {'animal':<8}{'animal-days':>13}{'days AT CAP':>13}{'% wasted':>10}")
    for a in ("GOOSE","COW","SHEEP"):
        if ad[a]:
            print(f"  {a:<8}{ad[a]:>13}{ac[a]:>13}{100*ac[a]/ad[a]:>9.0f}%")
    print(f"  {'TOTAL':<8}{tot_days:>13}{tot_cap:>13}{100*tot_cap/max(1,tot_days):>9.0f}%")
    print(f"  crop tile-days at max_yield: {cc}/{cd} = {100*cc/max(1,cd):.0f}%")
    print(f"  ripe -> harvested latency: mean {st.mean(ra) if ra else 0:.2f} days, "
          f"max {max(ra) if ra else 0}")
    print(f"  per-unit daily work span (bounding box): mean {st.mean(sp) if sp else 0:.1f} tiles")
