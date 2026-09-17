"""WHICH crops die, and when? A dead melon (seed 80, 6 units at up to $250) is
not a dead wheat (seed 10). If deaths are concentrated in expensive crops there
is a triage rule to write that costs no extra labour.
"""
import os, sys, collections, statistics as st
sys.path.insert(0, os.getcwd())
import tools.fast_env as fe; fe.install()
from kaggle_environments import make
from kagent.constants import CROPS, MARKET_PARAMS

def run(agent, opp="pub-ahmed-v38", seeds=(71000,71001,71002)):
    died=collections.Counter(); planted=collections.Counter()
    byday=collections.Counter()
    for sd in seeds:
        env=make("kaggriculture",configuration={"seed":sd},debug=False)
        env.run([f"arena/agents/{agent}/main.py", f"arena/agents/{opp}/main.py"])
        prev=None
        for stp in env.steps[:-1]:
            o=stp[0].observation
            if o["hour"]!=23: continue
            tiles=o["farms"][0]["tiles"]; day=o["day"]
            cur={}
            for y,row in enumerate(tiles):
                for x,t in enumerate(row):
                    if isinstance(t,dict):
                        cur[(x,y)]=("WEED" if t.get("kind")=="WEED" else
                                    (t.get("crop") if t.get("kind")=="PLANT" else "O"))
                    else: cur[(x,y)]=None
            if prev:
                for k,v in cur.items():
                    p=prev.get(k)
                    if v=="WEED" and p in CROPS:
                        died[p]+=1; byday[day]+=1
                    elif v in CROPS and p not in CROPS:
                        planted[v]+=1
            prev=cur
    n=len(seeds)
    return {k:v/n for k,v in died.items()}, {k:v/n for k,v in planted.items()}, {k:v/n for k,v in byday.items()}

for agent in ("champ-h8","tape-sel"):
    died,planted,byday=run(agent)
    print(f"\n{agent}: crop deaths from missed watering (3 seeds, per game)\n")
    print(f"  {'crop':<12}{'planted':>9}{'died':>8}{'death %':>9}{'seed lost':>11}{'yield lost @base':>18}")
    tot_seed=tot_yield=0
    for c in ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"):
        p,d=planted.get(c,0),died.get(c,0)
        if p<0.5 and d<0.5: continue
        seed=CROPS[c]["seed"]*d
        yld=CROPS[c]["max_yield"]*MARKET_PARAMS[c]["base"]*d
        tot_seed+=seed; tot_yield+=yld
        print(f"  {c:<12}{p:>9.1f}{d:>8.1f}{100*d/max(1,p):>8.0f}%{seed:>11,.0f}{yld:>18,.0f}")
    print(f"  {'TOTAL':<12}{sum(planted.values()):>9.1f}{sum(died.values()):>8.1f}"
          f"{'':>9}{tot_seed:>11,.0f}{tot_yield:>18,.0f}")
    top=sorted(byday.items(), key=lambda kv:-kv[1])[:6]
    print(f"  deaths by day (top): " + ", ".join(f"d{d}:{n:.1f}" for d,n in top))
