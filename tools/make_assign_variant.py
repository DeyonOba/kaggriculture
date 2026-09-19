"""Joint OPTIMAL assignment of units to tasks, with two competing cost functions.

`scheduler.assign` runs `for idx in sorted(free)` and each unit greedily takes
its own nearest task, so unit 0 claims a tile unit 5 is standing on. The
archive already tested GLOBAL-GREEDY matching (#9 `match-p1` -$1,119, #10
`match-all` null), but greedy is not optimal: it commits the best single pair
first and can strand every later unit.

This separates the two hypotheses that a "pointer network for routing" would
conflate:

    --cost dist     OPTIMAL matching, CURRENT cost (urgent, distance, pri).
                    If the defect is the MATCHING, this fixes it. If this is
                    null, matching is not the bottleneck and no assignment
                    architecture can help.

    --cost ptr      POINTER NETWORK: task self-attention, units glimpsing over
                  the task set, and a compatibility layer emitting per-unit
                  logits with a learned unit-task distance bias. Unlike the
                  two below it can represent "unit A should take this BECAUSE
                  unit B is taking that" -- the joint structure neither a
                  linear ranker nor Hungarian-on-a-fixed-cost can express.
                  Trained on M & M & P & Q joint assignments, 59.7% joint
                  top-1 (24.1% before the distance bias was added).

  --cost learned  OPTIMAL matching, LEARNED cost -- the exec_bc ranker
                    cloned from M & M & P & Q, which predicts their task
                    choice at 69.4% against our rule's 27.9%. If THIS beats
                    the above, the cost function was wrong, which is the only
                    thing a learned policy could have fixed.

Optimality comes from scipy's `linear_sum_assignment`, verified importable
from an isolated agent (statuses ['DONE','DONE'] with the repo stripped from
sys.path).

    PYTHONPATH=. uv run python tools/make_assign_variant.py --name x-hung --cost dist
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

RANKER = os.path.join(REPO, "rl/datasets/exec_bc/ranker.json")

ANCHOR = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)'''

REPL = '''    # Pass 1 — own zone, real work. OPTIMAL joint assignment: the greedy
    # per-unit loop let unit 0 claim a tile unit 5 was standing on.
    _pairs = _assign_optimal(view, sorted(free), zone_tasks, need_tasks,
                             units, claimed, n_zones, zone_of, late)
    for idx, task in _pairs:
        if idx in free and try_execute(idx, task):
            free.discard(idx)
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)'''

GLUE = '''

# --- optimal unit->task assignment ------------------------------------------
# Two costs, so the experiment separates "the matching was wrong" from "the
# cost was wrong". Only the second is something a learned policy could fix.
_ASSIGN_COST = "__COST__"
_BCW = __BCW__
_BCF = __BCF__


def _assign_feats(view, task, ux, uy, zone_of, n_zones, idx):
    d = abs(ux - task.x) + abs(uy - task.y)
    z = zone_of.get((task.x, task.y))
    inv = view.unit_inv(idx)
    f = {
        "d": d / 10.0, "d0": 1.0 if d == 0 else 0.0,
        "d1": 1.0 if d == 1 else 0.0, "dfar": 1.0 if d >= 4 else 0.0,
        "pri": task.pri / 5.0, "surv": 1.0 if task.pri <= 0 else 0.0,
        "filler": 1.0 if task.pri >= 5 else 0.0, "hour": view.hour / 24.0,
        "late": 1.0 if view.hour >= 16 else 0.0, "day_frac": view.day / 29.0,
        "need_ok": 1.0 if (not task.need or inv.get(task.need, 0) >= 1) else 0.0,
        "same_zone": 1.0 if (z is not None and z == idx % max(1, n_zones)) else 0.0,
        "load": min(sum(inv.values()), 12) / 12.0, "n_here": 0.0,
    }
    for k in _BCF:
        if k.startswith("op_"):
            f[k] = 1.0 if task.op[0] == k[3:] else 0.0
    return f


def _assign_optimal(view, idxs, zone_tasks, need_tasks, units, claimed,
                    n_zones, zone_of, late):
    """-> [(unit, task)] maximising total score over the whole free set."""
    if not idxs:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np
    except Exception:
        return []
    pool = []
    seen = set()
    for i in idxs:
        for t in list(need_tasks) + list(zone_tasks.get(i % n_zones, [])):
            k = (t.x, t.y, t.op[0])
            if k in claimed or k in seen:
                continue
            if not (P_SURVIVAL <= t.pri < P_FILLER):
                continue
            seen.add(k)
            pool.append(t)
    if not pool:
        return []
    BIG = 1e6
    C = _np.full((len(idxs), len(pool)), BIG, dtype="float64")
    for r, i in enumerate(idxs):
        ux, uy = units[i]
        inv = view.unit_inv(i)
        for c, t in enumerate(pool):
            if t.need and inv.get(t.need, 0) < 1:
                continue
            if _ASSIGN_COST == "ptr":
                continue            # filled in below, jointly
            if _ASSIGN_COST == "learned":
                f = _assign_feats(view, t, ux, uy, zone_of, n_zones, i)
                s = 0.0
                for nm, w in zip(_BCF, _BCW):
                    s += w * f.get(nm, 0.0)
                C[r, c] = -s                      # maximise the ranker score
            else:
                d = abs(ux - t.x) + abs(uy - t.y)
                urgent = 0 if (late and t.pri == P_SURVIVAL) else 1
                C[r, c] = urgent * 1000.0 + d * 10.0 + t.pri
    if _ASSIGN_COST == "ptr":
        try:
            nu, nt = len(idxs), len(pool)
            Uf = _np.zeros((nu, 6), "float32")
            Tf = _np.zeros((nt, len(_BCF)), "float32")
            Df = _np.zeros((nu, nt), "float32")
            rx, ry = units[idxs[0]]
            for c, t in enumerate(pool):
                f = _assign_feats(view, t, rx, ry, zone_of, n_zones, idxs[0])
                Tf[c] = [f.get(nm, 0.0) for nm in _BCF]
            for r, i in enumerate(idxs):
                ux, uy = units[i]
                inv = view.unit_inv(i)
                Uf[r] = [ux / 10.0, uy / 10.0,
                         min(sum(inv.values()), 12) / 12.0,
                         (i % max(1, n_zones)) / max(1, n_zones),
                         view.hour / 24.0, view.day / 29.0]
                for c, t in enumerate(pool):
                    Df[r, c] = (abs(ux - t.x) + abs(uy - t.y)) / 10.0
            lg = _ptr_logits(Uf, Tf, Df, _np.ones(nt, bool))
            for r, i in enumerate(idxs):
                inv = view.unit_inv(i)
                for c, t in enumerate(pool):
                    if t.need and inv.get(t.need, 0) < 1:
                        continue
                    C[r, c] = -float(lg[r, c])
        except Exception:
            return []
    rows, cols = linear_sum_assignment(C)
    return [(idxs[r], pool[c]) for r, c in zip(rows, cols) if C[r, c] < BIG]

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cost", choices=["dist", "learned", "ptr"], required=True)
    ap.add_argument("--ptr", default="", help="module from export_ptrnet.py")
    args = ap.parse_args()

    d = json.load(open(RANKER))
    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    ptr_src = open(args.ptr).read() if args.cost == "ptr" else ""
    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if ANCHOR not in text:
                raise SystemExit("pass-1 anchor not found in scheduler")
            text = (ptr_src + GLUE.replace("__COST__", args.cost)
                        .replace("__BCW__", repr(d["w"]))
                        .replace("__BCF__", repr(d["features"]))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no scheduler module in bundle")

    p = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(p, exist_ok=True)
    f = os.path.join(p, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(f, "w").write(code)
    compile(code, f, "exec")
    print(f"{args.name}: {args.base} + OPTIMAL pass-1 assignment, cost={args.cost}")


if __name__ == "__main__":
    main()
