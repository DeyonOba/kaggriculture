"""Per-tile placement labels: did a crop planted HERE survive to yield?

The executor currently assigns crops to tiles by `i % len(wanted)` -- position
index modulo the quota list -- and plants whichever tiles the units reach
first. Tile quality has never entered the decision, and the board encoder's
spatial channels (distance-to-shed, clustering, occupancy) have never been
consumed by anything.

Label, extracted by differencing frames: for every tile that goes from empty to
planted, follow it until it either

    YIELDS   the plant reaches yield_units > 0 and is harvested (tile clears
             for a one-shot crop, or yield_units drops for an ongoing one)
    DIES     it becomes a WEED -- `_daily_refresh_plants` converts a plant with
             consecutive_unwatered >= 2 (kaggriculture.py:783)

This is observational: we only see tiles they DID plant, so a raw "planted
tiles survive" rate is confounded by their choosing good tiles. What makes it
usable is that the FEATURES are positional and available for empty tiles too,
so the scorer transfers to the ranking problem even though the label does not
come from a randomised experiment. The gate is the arbiter, as always.

    PYTHONPATH=. uv run python tools/place_label.py --n 150
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "rl/datasets/session_index.json")
OUT = os.path.join(REPO, "rl/datasets/place.npz")



def _rating_ok(v, floor):
    """Sessions from the DAILY DUMPS carry finals but no scores, so a strict
    `scores is not None` filter discards every harvested game. Accept them
    whenever no rating floor is requested."""
    sc = v.get("scores")
    if sc and all(x is not None for x in sc):
        return min(sc) >= floor
    return floor <= 0

def job(path):
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.board_tensor import board_pair_full
    import tools.price_forecast as pf

    s = json.load(open(path))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run([pf._script(s["actions"][0]), pf._script(s["actions"][1])])

    steps = [st[0].observation for st in env.steps]
    n = len(steps[0]["farms"][0]["tiles"])

    # follow every tile of seat 0 through the whole game
    fate = {}          # (x, y, planted_step) -> 1 yielded / 0 died
    live = {}          # (x, y) -> [planted_step, saw_yield]
    prev = None
    for t, o in enumerate(steps):
        tiles = o["farms"][0]["tiles"]
        for y in range(n):
            for x in range(n):
                c = tiles[y][x]
                was = prev[y][x] if prev else None
                is_plant = isinstance(c, dict) and c.get("kind") == "PLANT"
                was_plant = isinstance(was, dict) and was.get("kind") == "PLANT"
                if is_plant and not was_plant:
                    live[(x, y)] = [t, 0]
                elif was_plant and (x, y) in live:
                    if is_plant:
                        if c.get("yield_units", 0) > 0:
                            live[(x, y)][1] = 1
                    else:
                        died = isinstance(c, dict) and c.get("kind") == "WEED"
                        st0, saw = live.pop((x, y))
                        fate[(x, y, st0)] = 0 if died else max(saw, 1)
        prev = tiles
    for (x, y), (st0, saw) in live.items():
        fate[(x, y, st0)] = saw

    # one sample per DAWN: the board, plus a 10x10 mask of tiles planted that
    # day and a 10x10 label of how each turned out
    T, M, L, D = [], [], [], []
    for day in range(30):
        t0 = day * 24
        if t0 >= len(steps):
            break
        mask = np.zeros((n, n), "float32")
        lab = np.zeros((n, n), "float32")
        any_ = False
        for (x, y, st0), ok in fate.items():
            if t0 <= st0 < t0 + 24:
                mask[y, x] = 1.0
                lab[y, x] = float(ok)
                any_ = True
        if not any_:
            continue
        T.append(board_pair_full(steps[t0], 0))
        M.append(mask)
        L.append(lab)
        D.append(day)
    if not T:
        return None
    return (np.stack(T), np.stack(M), np.stack(L), np.asarray(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    idx = json.load(open(INDEX))
    paths = [v["path"] for v in idx.values()
             if _rating_ok(v, args.min_rating)]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.n]
    print(f"replaying {len(paths)} sessions ...", flush=True)
    with mp.Pool(args.procs) as pool:
        out = [o for o in pool.map(job, paths) if o]
    T = np.concatenate([o[0] for o in out])
    M = np.concatenate([o[1] for o in out])
    L = np.concatenate([o[2] for o in out])
    D = np.concatenate([o[3] for o in out])
    G = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    np.savez_compressed(OUT, T=T, M=M, L=L, D=D, G=G)
    planted = M.sum()
    died = (M * (1 - L)).sum()
    print(f"{len(T):,} day-samples, {int(planted):,} plantings, "
          f"{int(died):,} died ({100*died/max(1,planted):.1f}%) -> {OUT}\n")

    # Does POSITION predict survival at all? If a tile's coordinates and its
    # distance to the shed carry nothing, a placement model has no room.
    from tools.board_tensor import F_OURS
    dist = T[:, F_OURS[0] + 23]                 # static distance-to-shed plane
    sel = M > 0
    d = dist[sel]
    y = L[sel]
    qs = np.quantile(d, [0.25, 0.5, 0.75])
    print("  survival by distance-to-shed quartile (the one purely positional"
          " feature):")
    lo = -1e9
    for i, hi in enumerate(list(qs) + [1e9]):
        m = (d > lo) & (d <= hi)
        if m.sum():
            print(f"    Q{i+1}  n={int(m.sum()):>6}   survived {100*y[m].mean():>5.1f}%")
        lo = hi
    print(f"\n  overall survival {100*y.mean():.1f}%")


if __name__ == "__main__":
    main()
