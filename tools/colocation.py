"""What does one-action-per-cell actually cost?

24.3% of unit-frames sit on a shared cell, so a GridNet head cannot give two
co-located units different jobs. That is only a real loss if top players
DO give them different jobs. Measured here on recorded play:

  - of co-located unit groups, how many emit >1 distinct action
  - where co-location happens (the shed is 2x2 at the centre and everything
    routes through it)
  - which ops the differing units are emitting

If co-located units almost always agree, accept the loss and keep the clean
architecture. If they systematically differ, the head needs a per-unit slot.
"""
import collections
import json
import multiprocessing as mp
import os
import sys

REPO = "/Users/gideonoba/WorkSpace/OpenSource/kaggriculture"
sys.path.insert(0, REPO)
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.price_forecast as pf

    c = collections.Counter()
    pairs = collections.Counter()
    try:
        s = json.load(open(path))
    except Exception:
        return c, pairs
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.reset(2)
    acts = s["actions"]
    step = 0
    while not env.done:
        o = env.state[0].observation
        n = len(o["farms"][0]["tiles"])
        for seat in (0, 1):
            f = o["farms"][seat]
            a = acts[seat][step] if step < len(acts[seat]) else None
            if not isinstance(a, dict):
                continue
            ops = [a.get("farmer")] + list(a.get("hands") or [])
            pos = [tuple(f["farmer"])] + [tuple(h) for h in (f.get("hands") or [])]
            if len(ops) != len(pos):
                continue
            groups = collections.defaultdict(list)
            for i, p in enumerate(pos):
                groups[p].append(i)
            for p, ids in groups.items():
                if len(ids) < 2:
                    c["solo"] += 1
                    continue
                c["groups"] += 1
                c["units_in_groups"] += len(ids)
                names = []
                for i in ids:
                    x = ops[i]
                    names.append(x[0] if x else "PASS")
                distinct = len(set(names))
                c["distinct_sum"] += distinct
                if distinct > 1:
                    c["groups_differing"] += 1
                    c["units_wasted"] += len(ids) - 1
                    for nm in sorted(set(names)):
                        pairs[nm] += 1
                shed = abs(p[0] - (n // 2 - 1)) <= 1 and \
                    abs(p[1] - (n // 2 - 1)) <= 1
                c["at_shed" if shed else "elsewhere"] += 1
                if distinct > 1:
                    c["diff_at_shed" if shed else "diff_elsewhere"] += 1
        env.step([pf._script(acts[0])(o), pf._script(acts[1])(
            env.state[1].observation)] if False else
            [acts[0][step] if step < len(acts[0]) and acts[0][step] else
             {"farmer": ["PASS"], "hands": [], "market": []},
             acts[1][step] if step < len(acts[1]) and acts[1][step] else
             {"farmer": ["PASS"], "hands": [], "market": []}])
        step += 1
    return c, pairs


if __name__ == "__main__":
    from tools.opp_coupling import load_index
    import random
    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(12).shuffle(paths)
    with mp.Pool(5, maxtasksperchild=30) as pool:
        out = pool.map(job, paths[:40])
    c = collections.Counter()
    pairs = collections.Counter()
    for a, b in out:
        c.update(a)
        pairs.update(b)

    g = c["groups"]
    print(f"\nCO-LOCATION over {len(out)} games, both seats\n")
    print(f"    unit-cells solo               {c['solo']:>10,}")
    print(f"    shared-cell GROUPS            {g:>10,}")
    print(f"    units inside those groups     {c['units_in_groups']:>10,}")
    print(f"    mean distinct ops per group   "
          f"{c['distinct_sum']/max(g,1):>10.2f}   (1.00 = always agree)")
    print(f"\n    groups where units DIFFER     {c['groups_differing']:>10,}"
          f"   = {100*c['groups_differing']/max(g,1):.1f}% of groups")
    print(f"    unit-decisions a per-cell head would get WRONG "
          f"{c['units_wasted']:>8,}")
    tot_units = c["solo"] + c["units_in_groups"]
    print(f"    as a share of ALL unit-decisions              "
          f"{100*c['units_wasted']/max(tot_units,1):>7.2f}%")

    print(f"\n    where co-location happens:")
    print(f"      at the shed (3x3 centre)    {c['at_shed']:>10,}"
          f"   differing {c['diff_at_shed']:>8,}")
    print(f"      elsewhere                   {c['elsewhere']:>10,}"
          f"   differing {c['diff_elsewhere']:>8,}")

    if pairs:
        print(f"\n    ops appearing in a DIFFERING group:")
        for nm, v in pairs.most_common(10):
            print(f"      {nm:<22}{v:>10,}")
