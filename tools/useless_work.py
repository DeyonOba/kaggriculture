"""How much emitted work changes nothing? And does anyone ever DIG?

The engine treats an illegal or redundant unit action as a SILENT no-op --
watering a wet tile, harvesting an unripe one, picking up from an animal
holding nothing. None of it errors, so none of it has ever been counted, and
a turn spent that way is indistinguishable from a turn spent well in every
metric we have.

This hooks `_apply_unit_action` and fingerprints the unit's tile plus its
inventory before and after. An action that moves neither did nothing.

It also prints the full op census. The engine accepts 23 ops and our tooling
had been working from a list of 12 -- PLACE, DROP, FERTILIZE, DIG,
BUILD_COOP, BUILD_PASTURE and COLLECT_FERTILIZER were being dropped from
every work count, including metric_harvest's.

DIG is the one to watch: it removes a plant, a weed, or an empty structure
(never a placed animal) and frees the tile. It is the only way to undo a
planting decision, and `kagent` never emits it.

    PYTHONPATH=. uv run python tools/useless_work.py --n 200
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def _fingerprint(farm, private, idx, pos):
    """Cheap state signature for the tile a unit is standing on, plus its bag."""
    if pos is None:
        return None
    fx, fy = pos[0], pos[1]
    t = farm["tiles"][fy][fx]
    if isinstance(t, dict):
        sig = (t.get("kind"), t.get("crop"), t.get("animal"),
               t.get("yield_units"), t.get("watered_today"),
               t.get("fed_today"), t.get("cared_today"),
               t.get("fertilized_until_day"), t.get("consecutive_unwatered"))
    else:
        sig = (t,)
    inv = (private.get("inventories") or private.get("inventory") or [{}])
    try:
        bag = tuple(sorted((inv[idx] or {}).items()))
    except Exception:
        bag = ()
    return sig + (bag, farm.get("money"))


def job(spec):
    path, agent = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _farmer_position)
    import tools.price_forecast as pf

    c = collections.Counter()
    orig = getattr(K, "_uw_unit", None) or K._apply_unit_action
    K._uw_unit = orig

    def unit(farm, private, idx, action, board_size, day, turns_per_day,
             shed_capacity=100):
        op = action[0] if isinstance(action, list) and action else None
        if op in MOVES or op == "PASS" or op is None:
            c["_MOVE" if op in MOVES else "_PASS"] += 1
            return orig(farm, private, idx, action, board_size, day,
                        turns_per_day, shed_capacity)
        pos = _farmer_position(farm, idx)
        before = _fingerprint(farm, private, idx, pos)
        r = orig(farm, private, idx, action, board_size, day, turns_per_day,
                 shed_capacity)
        after = _fingerprint(farm, private, idx, _farmer_position(farm, idx))
        c[f"op:{op}"] += 1
        c["_WORK"] += 1
        if before is not None and before == after:
            c[f"noop:{op}"] += 1
            c["_NOOP"] += 1
        return r

    K._apply_unit_action = unit
    try:
        if agent:
            g = {}
            exec(open(f"{REPO}/arena/agents/{agent}/main.py").read(), g)
            a0 = g["agent"]
            gb = {}
            exec(open(f"{REPO}/arena/agents/champ-h8/main.py").read(), gb)
            a1 = gb["agent"]
            env = make("kaggriculture", configuration={"seed": int(path)},
                       debug=False)
            env.reset(2)
            while not env.done:
                o = env.state[0].observation
                env.step([a0(o), a1(env.state[1].observation)])
        else:
            s = json.load(open(path))
            env = make("kaggriculture", configuration={"seed": s["seed"]},
                       debug=False)
            env.run([pf._script(s["actions"][0]), pf._script(s["actions"][1])])
    finally:
        K._apply_unit_action = orig
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--agent", default="",
                    help="score OUR agent on fresh seeds instead of the corpus")
    args = ap.parse_args()

    if args.agent:
        specs = [(str(300000 + i), args.agent) for i in range(args.n)]
        label = f"{args.agent} vs champ-h8, {args.n} fresh seeds"
    else:
        from tools.opp_coupling import load_index
        import random
        idx = load_index()
        paths = [v["path"] for v in idx.values()]
        random.Random(5).shuffle(paths)
        specs = [(p, "") for p in paths[: args.n]]
        label = f"corpus replays, {args.n} games (BOTH seats)"

    with mp.Pool(args.procs, maxtasksperchild=40) as pool:
        rows = pool.map(job, specs)
    c = collections.Counter()
    for r in rows:
        c.update(r)
    n = len(rows)

    work, noop = c["_WORK"], c["_NOOP"]
    print(f"\n{label}\n")
    print(f"    unit-actions per game   move {c['_MOVE']/n:>8.0f}"
          f"   pass {c['_PASS']/n:>7.0f}   work {work/n:>8.0f}")
    print(f"    of that work, USELESS   {noop/n:>8.0f} per game"
          f"   = {100*noop/max(work,1):>5.1f}% of all work actions\n")

    ops = sorted({k[3:] for k in c if k.startswith("op:")},
                 key=lambda o: -c[f"op:{o}"])
    print(f"    {'op':<20}{'per game':>10}{'useless':>10}{'% wasted':>10}")
    for o in ops:
        tot, bad = c[f"op:{o}"], c.get(f"noop:{o}", 0)
        print(f"    {o:<20}{tot/n:>10.1f}{bad/n:>10.1f}"
              f"{100*bad/max(tot,1):>9.1f}%")
    missing = [o for o in ("DIG", "DROP", "PLACE", "FERTILIZE",
                           "COLLECT_FERTILIZER", "BUILD_COOP",
                           "BUILD_PASTURE") if c[f"op:{o}"] == 0]
    if missing:
        print(f"\n    NEVER EMITTED: {', '.join(missing)}")


if __name__ == "__main__":
    main()
