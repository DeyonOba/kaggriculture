"""Waste audit: seeds paid for that never became money, and turns spent on it.

Tracks every tile's life: planted -> harvested (good) or died to WEED / still
standing at the end (waste). Reports seed cost burned, DIG turns spent clearing
the wreckage, and the spatial layout (where pastures and crops sit relative to
the shed) for both seats.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.getcwd())
import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

SHED = [(4, 4), (4, 5), (5, 4), (5, 5)]


def dist_shed(x, y):
    return min(abs(x - a) + abs(y - b) for a, b in SHED)


def audit(a, b, seeds, label_a="ours", label_b="theirs"):
    tot = [defaultdict(float), defaultdict(float)]
    layout = [defaultdict(list), defaultdict(list)]
    for sd in seeds:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd}, debug=False)
        env.run([a, b])
        prev = [{}, {}]
        for i, step in enumerate(env.steps):
            obs = step[0]["observation"]
            for s in (0, 1):
                cur = {}
                for y, row in enumerate(obs["farms"][s]["tiles"]):
                    for x, t in enumerate(row):
                        if isinstance(t, dict):
                            k = t.get("kind")
                            if k == "PLANT":
                                cur[(x, y)] = t.get("crop")
                                if i % 24 == 0:
                                    layout[s]["crop_d"].append(dist_shed(x, y))
                            elif k == "WEED":
                                cur[(x, y)] = "WEED"
                            elif k in ("PASTURE", "COOP"):
                                if i % 24 == 0:
                                    layout[s]["pasture_d"].append(dist_shed(x, y))
                # transitions: PLANT -> WEED means the tile died unharvested
                for pos, was in prev[s].items():
                    now = cur.get(pos)
                    if was not in (None, "WEED") and now == "WEED":
                        tot[s]["died"] += 1
                        tot[s]["died_seed_cost"] += K.CROPS[was]["seed"] if was in K.CROPS else 0
                prev[s] = cur
                a_ = step[s].get("action") or {}
                for act in [a_.get("farmer") or []] + list(a_.get("hands") or []):
                    if isinstance(act, list) and act:
                        if act[0] == "DIG":
                            tot[s]["dig"] += 1
                        elif act[0] == "PLANT":
                            tot[s]["plant"] += 1
                        elif act[0] == "HARVEST":
                            tot[s]["harvest"] += 1
                for o in a_.get("market") or []:
                    if o and o[0] == "BUY_SEED":
                        tot[s]["seeds"] += int(o[2])
                        tot[s]["seed_cost"] += K.CROPS[o[1]]["seed"] * int(o[2])
        # tiles still standing unharvested at the end
        for s in (0, 1):
            for y, row in enumerate(env.steps[-1][0]["observation"]["farms"][s]["tiles"]):
                for x, t in enumerate(row):
                    if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("yield_units", 0) > 0:
                        tot[s]["stranded"] += 1
    n = len(seeds)
    print(f"\n{n} games\n")
    print(f"{'metric':<22} {label_a:>10} {label_b:>10}")
    for k in ("seeds", "seed_cost", "plant", "harvest", "died", "died_seed_cost", "dig", "stranded"):
        print(f"{k:<22} {tot[0][k]/n:>10,.1f} {tot[1][k]/n:>10,.1f}")
    for s, lab in ((0, label_a), (1, label_b)):
        h = tot[s]["harvest"]
        p = tot[s]["plant"]
        print(f"\n{lab}: harvests per planting {h/max(1,p):.2f}   deaths per planting {tot[s]['died']/max(1,p):.3f}"
              f"   seed burned on dead tiles ${tot[s]['died_seed_cost']/n:,.0f}"
              f"   DIG turns {tot[s]['dig']/n:.0f}")
        cd = layout[s]["crop_d"]
        pd = layout[s]["pasture_d"]
        if cd:
            print(f"    mean shed-distance: crops {sum(cd)/len(cd):.2f}  pastures "
                  f"{(sum(pd)/len(pd) if pd else 0):.2f}")


if __name__ == "__main__":
    opp = sys.argv[1] if len(sys.argv) > 1 else "arena/agents/pub-ahmed-v38/main.py"
    mine = sys.argv[2] if len(sys.argv) > 2 else "arena/agents/champ-h8/main.py"
    audit(mine, opp, [31000, 31001, 31002])
