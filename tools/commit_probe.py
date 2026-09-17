"""E1's structural kill test: does a worker commit to a target, or re-decide
every turn?

`pick` re-chooses each turn as other units claim tasks, so a unit can walk
toward a tile, lose it, turn around, and come back later. Zone assignment is
already sticky (`unit_index % n_zones`); the TASK is not, and no variant has
ever carried per-worker state. This measures the symptom before anything is
built, because the trajectory's kill test is structural: if moves/tile does not
fall below 2.2, stop.

Measured per unit per day, then averaged:

    moves/action    MOVE actions per work action -- the headline number
                    (ours 1.45 vs ahmed 0.84 is already on record)
    moves/tile      MOVE actions per DISTINCT tile worked
    revisit rate    fraction of work actions on a tile this unit ALREADY
                    worked earlier today after going somewhere else --
                    the direct signature of re-deciding
    runs/tile       consecutive-work "runs" per distinct tile; 1.0 means a
                    unit finishes a tile before leaving it, 2.0 means it
                    comes back once on average
    switch rate     fraction of consecutive work actions that change tile

Both seats are measured, so the opponent is the target rather than a guess.

    PYTHONPATH=. uv run python tools/commit_probe.py --n 4
"""

import argparse
import collections
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from kaggle_environments import make   # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def unit_ops(step, seat):
    """-> list of ops for every unit this turn, farmer first.

    Actions carry NO coordinates -- they are ['WATER'], ['PLANT','WHEAT'],
    ['WEST']. A work action applies to the tile the unit is standing on, so
    tile identity has to come from the unit's POSITION in the observation.
    """
    a = step[seat].get("action") or {}
    out = []
    for x in [a.get("farmer")] + list(a.get("hands") or []):
        out.append(x[0] if isinstance(x, list) and x else "PASS")
    return out


def unit_pos(step, seat):
    """-> list of (x, y) for every unit, farmer first, matching unit_ops order."""
    obs = step[seat].observation if hasattr(step[seat], "observation") \
        else step[seat]["observation"]
    farm = obs["farms"][seat]
    pos = [tuple(farm.get("farmer") or (-1, -1))]
    for h in farm.get("hands") or []:
        pos.append(tuple(h))
    return pos, obs["day"]


def measure(steps, seat):
    """Per (unit, day) stats, aggregated over the game."""
    days = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"moves": 0, "work": 0, "seq": []}))
    for t in range(1, len(steps)):
        pos, day = unit_pos(steps[t], seat)
        for i, op in enumerate(unit_ops(steps[t], seat)):
            if i >= len(pos):
                break            # a hand hired later this turn
            if op in MOVES:
                days[day][i]["moves"] += 1
            elif op != "PASS":
                days[day][i]["work"] += 1
                days[day][i]["seq"].append(pos[i])
    mv_act, mv_tile, revisit, runs_tile, switch, hop = [], [], [], [], [], []
    for day, units in days.items():
        for i, d in units.items():
            seq = [s for s in d["seq"] if s is not None]
            if not seq:
                continue
            # Manhattan distance between CONSECUTIVE distinct worked tiles.
            # If commitment is not the problem, this is: our worker's tile
            # ORDER is less spatially coherent, which is a different bug.
            steps_between = [abs(a[0] - b[0]) + abs(a[1] - b[1])
                             for a, b in zip(seq, seq[1:]) if a != b]
            if steps_between:
                hop.append(st.mean(steps_between))
            distinct = len(set(seq))
            mv_act.append(d["moves"] / max(1, d["work"]))
            mv_tile.append(d["moves"] / max(1, distinct))
            # runs of consecutive work on the same tile
            runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
            runs_tile.append(runs / distinct)
            switch.append(sum(1 for a, b in zip(seq, seq[1:]) if a != b) /
                          max(1, len(seq) - 1) if len(seq) > 1 else 0.0)
            # a revisit: working a tile that was worked earlier today, after
            # having worked a DIFFERENT tile in between
            seen, back = set(), 0
            prev = None
            for s in seq:
                if s in seen and s != prev:
                    back += 1
                seen.add(s)
                prev = s
            revisit.append(back / len(seq))
    return mv_act, mv_tile, revisit, runs_tile, switch, hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=71000)
    args = ap.parse_args()

    acc = {0: [[] for _ in range(6)], 1: [[] for _ in range(6)]}
    for k in range(args.n):
        env = make("kaggriculture", configuration={"seed": args.base_seed + k},
                   debug=False)
        env.run([os.path.join(REPO, "arena", "agents", args.agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", args.opp, "main.py")])
        for seat in (0, 1):
            for slot, vals in zip(acc[seat], measure(env.steps, seat)):
                slot.extend(vals)

    print(f"{args.n} games, per (unit, day) then averaged\n")
    print(f"  {'':<14}{'moves/action':>14}{'moves/tile':>12}{'revisit':>10}"
          f"{'runs/tile':>11}{'switch':>9}{'hop dist':>10}")
    for seat, name in ((0, args.agent), (1, args.opp)):
        a, b, c, d, e, f = acc[seat]
        if not a:
            continue
        print(f"  {name[:13]:<14}{st.mean(a):>14.2f}{st.mean(b):>12.2f}"
              f"{st.mean(c):>10.3f}{st.mean(d):>11.2f}{st.mean(e):>9.3f}"
              f"{st.mean(f):>10.2f}")
    a0, b0, c0, d0, _, f0 = acc[0]
    print(f"\n  E1 kill test: moves/tile must fall below 2.2 "
          f"(ours now {st.mean(b0):.2f}).")
    print(f"  revisit rate is the direct signature of re-deciding: a worker "
          f"that commits\n  to a target does not come back to a tile it "
          f"already left.")
    print(f"  hop dist = Manhattan steps between CONSECUTIVE worked tiles. If "
          f"runs/tile is\n  already ~1.0, the moves/tile gap is tile ORDER, "
          f"not commitment.")


if __name__ == "__main__":
    main()
