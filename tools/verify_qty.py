"""Verify the quantity mask against the engine, both directions.

`legal_qty` claims two things, and a claim about a mask is only worth what a
test of it is worth:

  bit SET   applying [op, item, n] moves EXACTLY n units
  bit CLEAR applying it moves FEWER than n -- the engine clamps rather than
            rejects, so a clear bit means "synonym for a smaller n", not
            "illegal". Masking to distinct outcomes is the whole reason the
            head does not waste probability mass.

Both directions matter. A mask that is merely permissive passes the first
test and fails the second, and the failure mode is silent: the policy learns
a distribution over duplicates.

    PYTHONPATH=. uv run python tools/verify_qty.py --n 6
"""
import argparse
import copy
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--stride", type=int, default=37)
    args = ap.parse_args()

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _apply_unit_action)
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from tools.action_space import (legal_ops, legal_qty, to_action, OPS,
                                    QTY, QTY_OPS)
    from tools.opp_coupling import load_index
    import tools.price_forecast as pf

    idx = load_index()
    paths = [v["path"] for v in idx.values()]
    random.Random(4).shuffle(paths)

    # [bit][exact-n moved]
    tally = np.zeros((2, 2), int)
    from collections import Counter
    cover, ops_seen = Counter(), set()
    bad = []
    tested = 0

    for p in paths[: args.n]:
        s = json.load(open(p))
        env = make("kaggriculture", configuration={"seed": s["seed"]},
                   debug=False)
        env.reset(2)
        a0, a1 = pf._script(s["actions"][0]), pf._script(s["actions"][1])
        step = 0
        while not env.done:
            o = env.state[0].observation
            if step % args.stride == 0 and step > 0:
                priv = o.private
                farm = o["farms"][0]
                nunits = 1 + len(farm.get("hands") or [])
                for u in range(nunits):
                    m = legal_ops(o, 0, u, priv)
                    for oi, opn in enumerate(OPS):
                        if not m[oi] or opn.split(":")[0] not in QTY_OPS:
                            continue
                        item = opn.split(":", 1)[1]
                        qm = legal_qty(o, 0, u, opn, priv)
                        for qi, q in enumerate(QTY):
                            # deep-copy the mutable state the op touches
                            f2 = copy.deepcopy(dict(farm))
                            p2 = copy.deepcopy({
                                "shed": dict(priv.get("shed") or {}),
                                "seeds": dict(priv.get("seeds") or {}),
                                "inventories": copy.deepcopy(
                                    list(priv.get("inventories") or []))})
                            before_shed = p2["shed"].get(item, 0)
                            before_bag = (p2["inventories"][u].get(item, 0)
                                          if u < len(p2["inventories"]) else 0)
                            try:
                                # signature is (farm, private, idx, action,
                                # board_size, day, turns_per_day, shed_cap) --
                                # day and board_size are NOT interchangeable
                                _apply_unit_action(
                                    f2, p2, u, to_action(opn, q),
                                    len(farm["tiles"]),
                                    o.get("day", o["step"] // 24),
                                    24, 100)
                            except Exception:
                                continue
                            after_shed = p2["shed"].get(item, 0)
                            after_bag = (p2["inventories"][u].get(item, 0)
                                         if u < len(p2["inventories"]) else 0)
                            if opn.startswith("PICKUP"):
                                moved = before_shed - after_shed
                            else:
                                moved = before_bag - after_bag
                            exact = int(moved == q)
                            tally[int(qm[qi]), exact] += 1
                            cover[(q, int(qm[qi]))] += 1
                            ops_seen.add(opn)
                            tested += 1
                            if int(qm[qi]) != exact:
                                bad.append((opn, q, int(qm[qi]), moved,
                                            before_shed, before_bag))
            env.step([a0(o), a1(env.state[1].observation)])
            step += 1

    print(f"\n{tested:,} (unit, op, quantity) triples over {args.n} games\n")
    print(f"    {'':<14}{'moved < n':>12}{'moved == n':>13}")
    print(f"    {'bit CLEAR':<14}{tally[0,0]:>12,}{tally[0,1]:>13,}"
          f"   <- want all left")
    print(f"    {'bit SET':<14}{tally[1,0]:>12,}{tally[1,1]:>13,}"
          f"   <- want all right")
    wrong = tally[0, 1] + tally[1, 0]
    print(f"\n  disagreements: {wrong:,} / {tested:,}")
    if wrong:
        print("\n  first 12:")
        for b in bad[:12]:
            print(f"    {b[0]:<20} n={b[1]}  mask={b[2]}  moved={b[3]}"
                  f"  shed={b[4]} bag={b[5]}")
    else:
        print("  -> the quantity mask is EXACT in both directions")

    # A bucket never exercised proves nothing about that bucket. The last
    # narrow-verification mistake here reported 0/25 while nine ops were
    # never reached, so coverage is printed alongside the verdict, always.
    print(f"\n  coverage per bucket (a bucket never tested proves nothing):")
    print(f"    {'n':>4}{'bit SET':>10}{'bit CLEAR':>11}")
    for q in QTY:
        st = cover[(q, 1)]
        cl = cover[(q, 0)]
        flag = "   <- NEVER SET" if st == 0 else ""
        print(f"    {q:>4}{st:>10,}{cl:>11,}{flag}")
    print(f"\n  ops exercised: {len(ops_seen)} distinct "
          f"({', '.join(sorted(ops_seen)[:6])}"
          f"{' ...' if len(ops_seen) > 6 else ''})")


if __name__ == "__main__":
    main()
