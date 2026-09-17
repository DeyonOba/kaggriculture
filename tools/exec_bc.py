"""Learned executor: clone a 3,093-rated agent's TASK CHOICE, not its moves.

Sixteen hand-written attempts failed to close the execution gap, and the
task-set diff says why they were doomed: we already propose the right work
(only 6.6% of their actions are at a tile we propose nothing for) and we already
pick the same MIX (band shares within ~3 points, op shares within a few). The
deficit is entirely in which of ~60 proposed tasks a unit walks to:

                tiles/worker-day  actions  moves  actions/tile  moves/tile
    M&M&P&Q            6.26       12.48    8.95      2.10          1.43
    pub-ahmed-v38      6.01       11.50    9.70      2.02          1.61
    champ-h8           5.59        9.26   13.39      1.69          2.39

We under-chain (1.69 vs 2.10 actions per arrival) and over-travel (2.39 vs 1.43
moves per tile) in a neighbourhood of the same size (worker-day span 6.13 vs
6.32). `scheduler.pick` hard-codes the choice as `(urgent, distance, priority)`
and re-decides EVERY turn, so a unit zig-zags as other units claim its target.

So learn the ranker instead of writing it.

LABELLING. A unit that is moving has no visible target, but its next productive
action that day reveals where it was going. So for each unit-turn the label is
the (tile, op) of that unit's NEXT work action within the same day, matched
against the candidate tasks our own `build_tasks` proposes from their state.
Turns with no later work that day are dropped.

MODEL. A linear scorer over per-candidate features, trained with a softmax
cross-entropy over the candidate set (a ranker, not a classifier -- the
candidate set changes every turn). Inference is a dot product per candidate, so
the exported agent stays pure Python with the weights embedded, exactly like
rl/value_weights.json.

    PYTHONPATH=. uv run python tools/exec_bc.py build --n 10
    PYTHONPATH=. uv run python tools/exec_bc.py fit --epochs 30
    PYTHONPATH=. uv run python tools/exec_bc.py export --name bc-exec
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent.gamestate import View              # noqa: E402
from kagent.planner import make_champion_plan  # noqa: E402
from kagent import scheduler                   # noqa: E402

TEAM = "M & M & P & Q"
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
DATA = os.path.join(REPO, "rl", "datasets", "exec_bc")
OPS = ["WATER", "HARVEST", "COLLECT_FERTILIZER", "FEED", "CARE", "PLANT",
       "FERTILIZE", "DIG", "PLACE", "BUILD_COOP", "BUILD_PASTURE"]

# Feature names are the contract between fit and export; keep them in one place.
FEATS = (["d", "d0", "d1", "dfar", "pri", "surv", "filler", "hour", "late",
          "day_frac", "need_ok", "same_zone", "load", "n_here"]
         + [f"op_{o}" for o in OPS])


def featurize(view, task, ux, uy, zone_of, n_zones, idx, load, here_count):
    d = abs(ux - task.x) + abs(uy - task.y)
    z = zone_of.get((task.x, task.y))
    f = {
        "d": d / 10.0,
        "d0": 1.0 if d == 0 else 0.0,
        "d1": 1.0 if d == 1 else 0.0,
        "dfar": 1.0 if d >= 4 else 0.0,
        "pri": task.pri / 5.0,
        "surv": 1.0 if task.pri <= 0 else 0.0,
        "filler": 1.0 if task.pri >= 5 else 0.0,
        "hour": view.hour / 24.0,
        "late": 1.0 if view.hour >= 16 else 0.0,
        "day_frac": view.day / 29.0,
        "need_ok": 1.0 if (not task.need or view.unit_inv(idx).get(task.need, 0) >= 1) else 0.0,
        "same_zone": 1.0 if (z is not None and z == idx % max(1, n_zones)) else 0.0,
        "load": min(load, 12) / 12.0,
        "n_here": min(here_count, 4) / 4.0,
    }
    for o in OPS:
        f[f"op_{o}"] = 1.0 if task.op[0] == o else 0.0
    return [f[k] for k in FEATS]


def build(args):
    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "samples.jsonl"), "w")
    kept = dropped = 0
    for p in sorted(glob.glob(os.path.join(REPO, args.dir, "*.json")))[:args.n]:
        rep = json.load(open(p))
        names = (rep.get("info") or {}).get("TeamNames") or []
        if TEAM not in names:
            continue
        seat = names.index(TEAM)
        steps = rep["steps"]

        # For every (unit, turn): the next work action that unit takes that day.
        nxt = {}
        last = {}
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
                key = (day, i)
                if op not in MOVES and op != "PASS" and pos:
                    last[key] = (tuple(pos[:2]), op)
                nxt[(t - 1, i)] = last.get(key)

        for t in range(0, len(steps) - 1, args.every):
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
            if not tasks:
                continue
            here = collections.Counter((k.x, k.y) for k in tasks)
            for idx, ux, uy in v.me.units:
                tgt = nxt.get((t, idx))
                if not tgt:
                    dropped += 1
                    continue
                (tx, ty), top = tgt
                cand, label = [], -1
                for k in tasks:
                    if len(cand) >= args.max_cand:
                        break
                    if k.x == tx and k.y == ty and k.op[0] == top:
                        label = len(cand)
                    cand.append(featurize(v, k, ux, uy, zone_of, n_zones, idx,
                                          sum(v.unit_inv(idx).values()),
                                          here[(k.x, k.y)]))
                if label < 0:
                    dropped += 1
                    continue
                out.write(json.dumps({"x": cand, "y": label}) + "\n")
                kept += 1
    out.close()
    print(f"kept {kept:,} samples, dropped {dropped:,} "
          f"({kept/max(1,kept+dropped)*100:.1f}% labelled)")
    print(f"-> {os.path.join(DATA, 'samples.jsonl')}")


def _load():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "samples.jsonl"))]
    return rows


def fit(args):
    rows = _load()
    random.Random(0).shuffle(rows)
    cut = int(len(rows) * 0.8)
    tr, te = rows[:cut], rows[cut:]
    n = len(FEATS)
    w = [0.0] * n
    lr = args.lr

    def score(x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def evaluate(data):
        hit = 0
        for r in data:
            s = [score(x) for x in r["x"]]
            if s.index(max(s)) == r["y"]:
                hit += 1
        return hit / max(1, len(data))

    # Baseline: what the shipped rule gets, i.e. argmin (surv, d, pri).
    def baseline(data):
        i_d, i_pri, i_surv = FEATS.index("d"), FEATS.index("pri"), FEATS.index("surv")
        hit = 0
        for r in data:
            best = min(range(len(r["x"])),
                       key=lambda j: (-r["x"][j][i_surv], r["x"][j][i_d], r["x"][j][i_pri]))
            if best == r["y"]:
                hit += 1
        return hit / max(1, len(data))

    print(f"{len(tr):,} train / {len(te):,} held out, {n} features")
    print(f"shipped pick() rule, held out: {baseline(te)*100:.1f}%")
    for ep in range(args.epochs):
        random.Random(ep).shuffle(tr)
        for r in tr:
            s = [score(x) for x in r["x"]]
            m = max(s)
            ex = [math.exp(v - m) for v in s]
            z = sum(ex)
            p = [e / z for e in ex]
            for j, x in enumerate(r["x"]):
                g = p[j] - (1.0 if j == r["y"] else 0.0)
                if g:
                    for k2 in range(n):
                        if x[k2]:
                            w[k2] -= lr * g * x[k2]
        if ep % 5 == 4 or ep == 0:
            print(f"  epoch {ep+1:>3}  train {evaluate(tr[:2000])*100:>5.1f}%  "
                  f"held out {evaluate(te)*100:>5.1f}%")
    json.dump({"features": FEATS, "w": w},
              open(os.path.join(DATA, "ranker.json"), "w"), indent=1)
    print(f"\nlearned ranker, held out: {evaluate(te)*100:.1f}%  "
          f"(shipped rule {baseline(te)*100:.1f}%)")
    top = sorted(zip(FEATS, w), key=lambda kv: -abs(kv[1]))[:12]
    print("\nlargest weights:")
    for name, val in top:
        print(f"  {name:<24} {val:>+8.3f}")





# --- export ------------------------------------------------------------------
EXPORT_SRC = '''

# --- learned executor ranker (tools/exec_bc.py) ------------------------------
# Linear scorer over per-candidate features, cloned from M & M & P & Q (3,093,
# reactive). Predicts which task a unit walks to with __ACC__% held-out accuracy
# against __BASE__% for the (urgent, distance, priority) rule it replaces.
BC_FEATS = __FEATS__
BC_W = __W__
BC_OPS = __OPS__
# Chaining bonus. Measured: their agent performs 2.10 actions per tile visit to
# our 1.67. Dropping the band passes fixes travel (moves/tile 2.39 -> 2.00) but
# costs chaining (1.78 -> 1.47), because a unit that has just worked a tile is
# immediately re-scored against every task and sent away. The ranker's own d0
# weight is not enough on its own, so make staying put explicit.
BC_CHAIN = __CHAIN__
_BC_HERE = {}


def _bc_score(view, task, ux, uy, zone_of, n_zones, idx, load):
    d = abs(ux - task.x) + abs(uy - task.y)
    z = zone_of.get((task.x, task.y))
    inv = view.unit_inv(idx)
    v = {
        "d": d / 10.0,
        "d0": 1.0 if d == 0 else 0.0,
        "d1": 1.0 if d == 1 else 0.0,
        "dfar": 1.0 if d >= 4 else 0.0,
        "pri": task.pri / 5.0,
        "surv": 1.0 if task.pri <= 0 else 0.0,
        "filler": 1.0 if task.pri >= 5 else 0.0,
        "hour": view.hour / 24.0,
        "late": 1.0 if view.hour >= 16 else 0.0,
        "day_frac": view.day / 29.0,
        "need_ok": 1.0 if (not task.need or inv.get(task.need, 0) >= 1) else 0.0,
        "same_zone": 1.0 if (z is not None and z == idx % max(1, n_zones)) else 0.0,
        "load": min(load, 12) / 12.0,
        "n_here": min(_BC_HERE.get((task.x, task.y), 1), 4) / 4.0,
    }
    for o in BC_OPS:
        v["op_" + o] = 1.0 if task.op[0] == o else 0.0
    s = 0.0
    for name, wt in zip(BC_FEATS, BC_W):
        if wt:
            s += wt * v.get(name, 0.0)
    if d == 0:
        s += BC_CHAIN
    return s
'''

PICK_FIND = '''            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            score = (urgent, d, task.pri)
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None'''

PICK_REPL = '''            urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
            # Learned ranker in place of (urgent, distance, priority). Urgency
            # stays lexicographic so a dies-tonight task is never traded away.
            score = (urgent, -_bc_score(view, task, x, y, _BC_ZONE_OF,
                                        _BC_NZONES, idx, sum(inv.values())))
            if best is None or score < best[0]:
                best = (score, task)
        return best[1] if best else None'''

ZONE_FIND = '''    plant_requests = {}
    claimed = set()  # (x, y, op0) already assigned this turn'''

ZONE_REPL = '''    plant_requests = {}
    claimed = set()  # (x, y, op0) already assigned this turn
    # Per-turn context for the learned ranker: tasks-per-tile and the zone map.
    global _BC_ZONE_OF, _BC_NZONES
    _BC_ZONE_OF, _BC_NZONES = zone_of, n_zones
    _BC_HERE.clear()
    for _t in tasks:
        _BC_HERE[(_t.x, _t.y)] = _BC_HERE.get((_t.x, _t.y), 0) + 1'''


GLOBAL_FIND = """    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "zone")
    _seen = set(actions)

    # Pass 2 — real work anywhere: animal-heavy zones (pastures cluster near
    # the shed) get help before anyone waters a plant that is not due.
    for idx in sorted(free):
        task = pick(idx, all_tasks + global_tasks, P_SURVIVAL, P_FILLER, max_dist=HELP_RADIUS)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "idlehelp")
    _seen = set(actions)

    # Pass 3 — filler work: own zone, then anywhere.
    for idx in sorted(free):
        task = pick(idx, zone_tasks.get(idx % n_zones, []), P_FILLER, 99)
        if task is None:
            task = pick(idx, all_tasks + global_tasks, P_FILLER, 99)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "filler")
    _seen = set(actions)"""

GLOBAL_REPL = """    # ONE GLOBAL PASS. The three band passes are gone: the learned ranker was
    # trained to predict, from a unit's position, which of ALL proposed tasks a
    # 3,093-rated agent walks to -- so constraining it to a band and a zone
    # throws away most of what it learned. Bands survive only inside the ranker
    # (pri, surv and filler are features it weighs itself) and in the survival
    # override below, which still guarantees nothing dying is stranded.
    # Animal deliveries keep their hard precedence: a carried cow is placed now.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, all_tasks + global_tasks, 0, 99)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    _count_moves(actions, _seen, "ranker")
    _seen = set(actions)"""


def export(args):
    import ast
    model = json.load(open(os.path.join(DATA, "ranker.json")))
    base = os.path.join(REPO, "arena", "agents", args.base, "main.py")
    src = open(base).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, ok = [], 0
    for mod, text in mods:
        if PICK_FIND in text and ZONE_FIND in text:
            text = text.replace(PICK_FIND, PICK_REPL, 1)
            text = text.replace(ZONE_FIND, ZONE_REPL, 1)
            blk = (EXPORT_SRC.replace("__FEATS__", repr(model["features"]))
                             .replace("__W__", repr([round(x, 5) for x in model["w"]]))
                             .replace("__OPS__", repr(OPS))
                             .replace("__CHAIN__", repr(float(args.chain)))
                             .replace("__ACC__", str(args.acc))
                             .replace("__BASE__", str(args.base_acc)))
            if args.mode == "global":
                if GLOBAL_FIND not in text:
                    raise RuntimeError("three-pass block not found")
                text = text.replace(GLOBAL_FIND, GLOBAL_REPL, 1)
            k = text.index("def _dist(")
            text = text[:k] + "_BC_ZONE_OF = {}\n_BC_NZONES = 1\n" + blk + "\n\n" + text[k:]
            ok = 1
        out.append((mod, text))
    if not ok:
        raise RuntimeError("pick()/claimed anchors not found")
    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("build", "fit", "export"):
        p = sub.add_parser(nm)
        p.add_argument("--dir", default="replays/mmpq")
        p.add_argument("--n", type=int, default=10)
        p.add_argument("--every", type=int, default=2)
        p.add_argument("--max-cand", type=int, default=60)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--lr", type=float, default=0.05)
        p.add_argument("--name", default="bc-exec")
        p.add_argument("--base", default="champ-h8")
        p.add_argument("--acc", default="69.4")
        p.add_argument("--base-acc", default="27.9")
        p.add_argument("--chain", type=float, default=0.0,
                       help="extra score for a task on the unit's own tile")
        p.add_argument("--mode", choices=("banded", "global"), default="banded",
                       help="banded keeps the three passes; global replaces them "
                            "with one ranker pass over every task")
    a = ap.parse_args()
    {"build": build, "fit": fit, "export": export}[a.cmd](a)
