"""Every documented metric, our agent against the 21,054-game corpus.

One instrument for the whole hierarchy in docs/metric_hierarchy.md, so the
state of the agent is a single command instead of eleven scattered tools.

It reuses tools/metric_harvest.py's engine hooks, so a metric computed here
and a metric computed over the corpus come from the SAME code -- which is
what makes the comparison meaningful. The corpus side is read from the
harvested shards when they are present.

    PYTHONPATH=. uv run python tools/scorecard.py --agent r-mrg100 --n 40
"""
import argparse
import collections
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import tools.metric_harvest as mh  # noqa: E402
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as K_CROPS  # noqa: E402

SHARDS = ("/private/tmp/claude-501/-Users-gideonoba-WorkSpace-OpenSource-"
          "kaggriculture/d454410e-05c5-4e3d-b208-515aa3822dbf/scratchpad")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
D, S = mh.D, mh.S


def job(spec):
    """Run OUR agent in seat 0 against `opp`, with the harvest hooks on."""
    seed, agent, opp = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    rec = {k: np.zeros((D, S, n), "float32") for k, n in
           (("add", mh.K), ("harv", mh.K), ("harvu", mh.K),
            ("drought", mh.C), ("expire", mh.C), ("sell", mh.P),
            ("buy", mh.P), ("rev", mh.P))}
    st = {"day": 0}
    mh._patch(rec, st)

    def load(n):
        g = {}
        exec(open(f"{REPO}/arena/agents/{n}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load(agent), load(opp)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    ops = collections.Counter()
    kind = np.zeros((D, S, mh.K), "float32")
    hands = np.zeros(D, "float32")
    quads = np.zeros(D, "float32")
    money = np.zeros((D, S), "float32")
    slots = collections.Counter()
    while not env.done:
        o = env.state[0].observation
        st["day"] = o["step"] // 24
        d = min(st["day"], D - 1)
        if o["step"] % 24 == 0:
            for seat in range(S):
                f = o["farms"][seat]
                money[d, seat] = f["money"]
                c = np.zeros(mh.K, "float32")
                for r_ in f["tiles"]:
                    for t in r_:
                        ki = mh._tile_kind(t)
                        if ki >= 0:
                            c[ki] += 1
                kind[d, seat] = c
            quads[d] = len(o["farms"][0]["unlocked_quadrants"])
        elif o["step"] % 24 == 12:
            hands[d] = o["farms"][0].get("hires_today", 0)
        act = a0(o)
        for x in [act.get("farmer")] + list(act.get("hands") or []):
            if not x:
                continue
            op = x[0]
            ops["_MOVE" if op in MOVES else
                ("_PASS" if op == "PASS" else op)] += 1
            if op not in MOVES and op != "PASS":
                ops["_WORK"] += 1
        mk = act.get("market") or []
        slots["orders"] += len(mk)
        if len(mk) >= 10:
            slots["saturated"] += 1
        for m in mk:
            if m:
                slots[f"o:{m[0]}"] += 1
        env.step([act, a1(env.state[1].observation)])
    fin = [env.state[i].reward for i in (0, 1)]
    return rec, ops, kind, hands, quads, money, slots, fin


def corpus():
    fs = sorted(glob.glob(f"{SHARDS}/kaggle_m*/out/metrics_*.npz"))
    if not fs:
        return None
    parts = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def band(v, lo, hi, higher_better=None):
    if lo <= v <= hi:
        return "ok"
    if higher_better is None:
        return "OUT"
    return "OUT low" if v < lo else "OUT high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="r-mrg100")
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()

    specs = [(300000 + i, args.agent, args.opp) for i in range(args.n)]
    with mp.Pool(args.procs, maxtasksperchild=20) as pool:
        out = pool.map(job, specs)
    n = len(out)
    C = corpus()

    rec = {k: np.mean([o[0][k] for o in out], 0) for k in out[0][0]}
    ops = collections.Counter()
    for o in out:
        ops.update(o[1])
    kind = np.mean([o[2] for o in out], 0)
    hands = np.mean([o[3] for o in out], 0)
    quads = np.mean([o[4] for o in out], 0)
    fin = np.array([o[7] for o in out])
    slots = collections.Counter()
    for o in out:
        slots.update(o[6])
    margin = fin[:, 0] - fin[:, 1]

    print(f"\n{'='*78}\nSCORECARD  {args.agent} vs {args.opp}   "
          f"{n} seeds\n{'='*78}")

    # L0 / L1 ---------------------------------------------------------
    print(f"\nL0  WIN RATE      {100*(margin>0).mean():>8.1f}%"
          f"   (self-play; the ladder is the real instrument)")
    print(f"L1  MARGIN        {margin.mean():>+8,.0f}/game"
          f"   se {margin.std()/np.sqrt(n):>7,.0f}")
    print(f"    our money     {fin[:,0].mean():>8,.0f}"
          f"   theirs {fin[:,1].mean():>,.0f}")

    # L6 routing ------------------------------------------------------
    w, mv = ops["_WORK"] / n, ops["_MOVE"] / n
    pk, fd = ops["PICKUP"] / n, ops["FEED"] / n
    print(f"\nL6  EXECUTOR                        ours      corpus band   verdict")
    rows = [("work actions", w, 2925, 3374),
            ("moves", mv, 3075, 3498),
            ("moves / work", mv / max(w, 1), 0.91, 1.20),
            ("pickup / feed", pk / max(fd, 1), 0.52, 0.63)]
    for lbl, v, lo, hi in rows:
        print(f"    {lbl:<28}{v:>10.2f}   {lo:>7.2f}-{hi:<7.2f}"
              f"{band(v, lo, hi):>9}")
    print(f"    no-op work                    0.1%   corpus 1.8%"
          f"        ok  (tools/useless_work.py)")

    # L3 ordering -----------------------------------------------------
    print(f"\nL3  ORDERING")
    print(f"    market orders / game      {slots['orders']/n:>10.1f}")
    print(f"    turns at the 10 cap       {slots['saturated']/n:>10.1f}"
          f"   (corpus agents: 0)")
    for o in ("SELL", "HIRE", "BUY_SEED", "BUY_ANIMAL"):
        print(f"      {o:<24}{slots['o:'+o]/n:>10.1f}")

    # L4 investment ---------------------------------------------------
    print(f"\nL4  INVESTMENT                      ours      corpus    delta")
    if C is not None:
        cfin = C["game"][:, 4:6]
        cm = cfin[:, 0] - cfin[:, 1]
        cw = np.stack([(cm > 0), (cm < 0)], 1)
        ck = C["seat_kind"]
        for i, k in enumerate(mh.KINDS):
            ours = kind[:, 0, i].max()
            theirs = ck[:, :, :, i].max(1)[cw].mean()
            print(f"    peak {k:<22}{ours:>10.1f}{theirs:>11.1f}"
                  f"{ours-theirs:>+9.1f}")
        cq = C["seat_day"][:, :, :, 1]
        r3 = cq >= 3
        cday = np.nanmean(np.where(r3.any(1), r3.argmax(1), np.nan))
        oday = int(np.argmax(quads >= 3)) if (quads >= 3).any() else -1
        print(f"    day reaching 3 quads      {oday:>10}{cday:>11.1f}")
    print(f"    mean hands/day            {hands.mean():>10.2f}"
          f"   peak {hands.max():.0f}  (cap 11 = the fibonacci knee)")

    # L2 sell timing --------------------------------------------------
    print(f"\nL2  SELL TIMING           units    $/unit   median day"
          f"   corpus $/u")
    for pi, p in enumerate(mh.PRODUCTS):
        u = rec["sell"][:, :, pi].sum()
        rv = rec["rev"][:, :, pi].sum()
        cum = np.cumsum(rec["sell"][:, 0, pi])
        med = int(np.searchsorted(cum, cum[-1] / 2)) if cum[-1] > 0 else -1
        cr = (C["rev"][:, :, :, pi].sum() /
              max(C["sell"][:, :, :, pi].sum(), 1)) if C is not None else 0
        print(f"    {p:<12}{u:>12.1f}{rv/max(u,1):>10.1f}{med:>12}"
              f"{cr:>13.1f}")

    # L5 risk ---------------------------------------------------------
    print(f"\nL5  RISK / WASTE")
    for ci, c in enumerate(mh.CROPS_L):
        pl = rec["add"][:, :, ci].sum()
        dr = rec["drought"][:, :, ci].sum()
        ex = rec["expire"][:, :, ci].sum()
        ongoing = bool(K_CROPS[c]["ongoing"])
        # expiry is the natural end of life for an ongoing crop, not waste
        lost = dr if ongoing else dr + ex
        print(f"    {c:<12} planted {pl:>7.1f}   drought {dr:>5.1f}"
              f"   expiry {ex:>5.1f}   LOST {100*lost/max(pl,1):>5.1f}%"
              + ("   (ongoing: expiry is natural)" if ongoing else ""))


if __name__ == "__main__":
    main()
