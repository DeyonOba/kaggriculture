"""The one op-level gap left unexplained: PICKUP 348 vs their 194.

Feeding needs 1 WHEAT in the UNIT's inventory, so every animal fed costs either
a carried wheat or a shed trip. scheduler.py:335 asks for batch=5 wheat per
PICKUP, so 328 feeds should need ~66 pickups. We do 348. Measure the real
batching: how many FEEDs does one shed trip actually serve?
"""
import os, sys, collections, statistics as st
sys.path.insert(0, os.getcwd())
import tools.fast_env as fe; fe.install()
from kaggle_environments import make
from tools.commit_probe import unit_ops, unit_pos, MOVES
SHED={(4,4),(4,5),(5,4),(5,5)}

for agent,seat in (("champ-h8",0),("pub-ahmed-v38",1)):
    picks=feeds=0; per_trip=[]; carried=[]
    for sd in (71000,71001,71002):
        env=make("kaggriculture",configuration={"seed":sd},debug=False)
        env.run(["arena/agents/champ-h8/main.py","arena/agents/pub-ahmed-v38/main.py"])
        # per unit: sequence of PICKUP / FEED events
        seq=collections.defaultdict(list)
        for t in range(1,len(env.steps)):
            pos,_=unit_pos(env.steps[t],seat)
            o=env.steps[t][seat].observation
            priv=o.get("private") or {}
            invs=priv.get("inventories") or []
            for i,op in enumerate(unit_ops(env.steps[t],seat)):
                if i>=len(pos): break
                if op=="PICKUP":
                    picks+=1
                    inv=(invs[i] if i<len(invs) else {}) or {}
                    carried.append(float(inv.get("WHEAT",0)))
                    seq[i].append("P")
                elif op=="FEED":
                    feeds+=1; seq[i].append("F")
        for i,s in seq.items():
            run=0
            for e in s:
                if e=="P":
                    if run: per_trip.append(run)
                    run=0
                else: run+=1
            if run: per_trip.append(run)
    n=3
    print(f"{agent:<16} PICKUP {picks/n:>6.0f}  FEED {feeds/n:>6.0f}  "
          f"feeds per shed trip {st.mean(per_trip) if per_trip else 0:>5.2f}"
          f"   wheat held at pickup {st.mean(carried) if carried else 0:>5.2f}")
