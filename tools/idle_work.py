"""With free labour, are idle workers out of work -- or is the executor failing
to assign them? Distinguishes a work-supply ceiling from an assignment ceiling."""
import sys, os, collections
sys.path.insert(0,'/Users/gideonoba/WorkSpace/OpenSource/kaggriculture')
import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

EXTRA=[0]; _eod=K._end_of_day
def eod(state, env, day):
    r=_eod(state,env,day); n=EXTRA[0]
    if n:
        o=state[0].observation; b=len(o.farms[0]["tiles"])
        o.farms[0]["hands"]=[list(K._default_spawn(b)) for _ in range(n)]
    return r
K._end_of_day=eod

def doable(t, seeds, day):
    if t is None:
        return "PLANT" if any(v>0 for v in (seeds or {}).values()) else None
    if not isinstance(t,dict): return None
    if t.get("kind")=="PLANT":
        cd=K.CROPS.get(t.get("crop")) or {}
        if t.get("yield_units",0)>0 and day-t.get("planted_day",day)>=cd.get("first_yield_day",0):
            return "HARVEST"
        if not t.get("watered_today"): return "WATER"
        return None
    if t.get("kind")=="WEED": return "DIG"
    if t.get("animal"):
        if not t.get("fed_today"): return "FEED"
        if t.get("fertilizer_available"): return "COLLECT"
        if not t.get("cared_today"): return "CARE"
        if t.get("yield_units",0)>0: return "HARVEST"
    return None

for extra in (0, 8, 16):
    EXTRA[0]=extra
    hit=collections.Counter(); n=0
    for sd in (61000,61001,61002):
        env=make("kaggriculture",configuration={"episodeSteps":720,"seed":sd},debug=False)
        env.run(["arena/agents/champ-h8/main.py","arena/agents/pub-ahmed-v38/main.py"]); n+=1
        for st in env.steps[:-1]:
            o=st[0]["observation"]; a=st[0].get("action") or {}
            me=o["farms"][0]; day=o.get("day",0)
            sd_=(o.get("private") or {}).get("seeds") or {}
            units=[me.get("farmer")]+list(me.get("hands") or [])
            acts=[a.get("farmer")]+list(a.get("hands") or [])
            tiles=me.get("tiles") or []
            # farm-wide doable work this turn
            avail=sum(1 for y in range(len(tiles)) for x in range(len(tiles))
                      if doable(tiles[y][x], sd_, day))
            for i,u in enumerate(units):
                if i>=len(acts): break
                x_=acts[i]
                if not (isinstance(x_,list) and x_ and x_[0]=="PASS"): continue
                hit["pass"]+=1
                hit["work_when_idle"]+=avail
                if avail>0: hit["idle_with_work"]+=1
    print(f"free hands {extra:>2}: PASS {hit['pass']/n:>6,.0f}   "
          f"of which with farm-wide work available: {hit['idle_with_work']/n:>6,.0f} "
          f"({hit['idle_with_work']/max(hit['pass'],1)*100:>3.0f}%)   "
          f"mean doable tiles at those moments {hit['work_when_idle']/max(hit['idle_with_work'],1):>5.1f}")
