"""Joint assignment labels for the pointer network, from M & M & P & Q replays.

`tools/exec_bc.py` builds PER-UNIT samples: one candidate list per unit, one
label. A pointer network needs the JOINT assignment -- for a whole turn, which
unit took which task -- because its whole point is that a unit's choice depends
on what the other units take. So the row here is a TURN, not a unit.

Reuses exec_bc's label extraction verbatim (next work action that unit takes
that day) and its `featurize`, so the two models are trained on exactly the
same signal and differ only in architecture.

Unit features are deliberately minimal (position, load, zone, hour) because
everything about a task's desirability already lives in the task features,
which are computed RELATIVE to the unit.
"""

import collections
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
TEAM = "M & M & P & Q"
MAX_U, MAX_T = 12, 64
OUT = os.path.join(REPO, "rl/datasets/ptrnet.npz")


def build(n_files=12, replay_dir="replays/mmpq", every=3):
    from kagent.gamestate import View
    from kagent.planner import make_champion_plan
    from kagent import scheduler
    from tools.exec_bc import featurize, FEATS

    F_T = len(FEATS)
    U, T, UM, TM, Y, G, D = [], [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(REPO, replay_dir, "*.json")))[:n_files]
    print(f"{len(files)} replays in {replay_dir}", flush=True)
    for gi, p in enumerate(files):
        rep = json.load(open(p))
        names = (rep.get("info") or {}).get("TeamNames") or []
        if TEAM not in names:
            continue
        seat = names.index(TEAM)
        steps = rep["steps"]

        nxt, last = {}, {}
        for t in range(len(steps) - 1, 0, -1):
            o = steps[t - 1][seat].get("observation")
            if not o:
                continue
            day = o["day"]
            a = steps[t][seat].get("action") or {}
            f = o["farms"][seat]
            units = [f.get("farmer")] + list(f.get("hands") or [])
            acts = [a.get("farmer")] + list(a.get("hands") or [])
            for i, (pos, act) in enumerate(zip(units, acts)):
                op = act[0] if isinstance(act, list) and act else "PASS"
                if op not in MOVES and op != "PASS" and pos:
                    last[(day, i)] = (tuple(pos[:2]), op)
                nxt[(t - 1, i)] = last.get((day, i))

        for t in range(0, len(steps) - 1, every):
            o = steps[t][seat].get("observation")
            if not o:
                continue
            try:
                v = View(o)
                plan = make_champion_plan(v, None)
                tasks = scheduler.build_tasks(v, plan)
                zone_of, n_zones = scheduler.zone_map(v)
            except Exception:
                continue
            if not tasks or not v.me.units:
                continue
            tasks = tasks[:MAX_T]
            here = collections.Counter((k.x, k.y) for k in tasks)

            uu = np.zeros((MAX_U, 6), "float32")
            tt = np.zeros((MAX_T, F_T), "float32")
            um = np.zeros(MAX_U, bool)
            tm = np.zeros(MAX_T, bool)
            yy = np.full(MAX_U, -1, "int64")
            # Unit-task distance is the ranker's dominant feature
            # (weight -11.5). A single farmer-relative task encoding
            # throws it away, so carry it as an explicit pairwise
            # matrix and let the pointer layer bias on it.
            dd = np.zeros((MAX_U, MAX_T), "float32")

            # task features are relative to a unit, so use the FARMER as the
            # reference frame and give each unit its own offset in U
            ref = v.me.units[0]
            for j, k in enumerate(tasks):
                tt[j] = featurize(v, k, ref[1], ref[2], zone_of, n_zones,
                                  0, 0, here[(k.x, k.y)])
                tm[j] = True

            any_label = False
            for idx, ux, uy in v.me.units:
                if idx >= MAX_U:
                    break
                um[idx] = True
                for j, k in enumerate(tasks):
                    dd[idx, j] = (abs(ux - k.x) + abs(uy - k.y)) / 10.0
                uu[idx] = [ux / 10.0, uy / 10.0,
                           min(sum(v.unit_inv(idx).values()), 12) / 12.0,
                           (idx % max(1, n_zones)) / max(1, n_zones),
                           v.hour / 24.0, v.day / 29.0]
                tgt = nxt.get((t, idx))
                if not tgt:
                    continue
                (tx, ty), top = tgt
                for j, k in enumerate(tasks):
                    if k.x == tx and k.y == ty and k.op[0] == top:
                        yy[idx] = j
                        any_label = True
                        break
            if not any_label:
                continue
            U.append(uu); T.append(tt); UM.append(um); TM.append(tm)
            Y.append(yy); G.append(gi); D.append(dd)

    if not U:
        raise SystemExit("no labelled turns -- check the replay dir and team name")
    np.savez_compressed(OUT, U=np.stack(U), T=np.stack(T), UM=np.stack(UM),
                        TM=np.stack(TM), Y=np.stack(Y), G=np.asarray(G),
                        D=np.stack(D))
    lab = sum(int((y >= 0).sum()) for y in Y)
    print(f"{len(U):,} turns, {lab:,} labelled unit-assignments "
          f"({lab/max(1,len(U)):.1f} per turn) -> {OUT}")


if __name__ == "__main__":
    build()
