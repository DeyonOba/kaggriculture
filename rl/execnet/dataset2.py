"""Build the execnet v2 (destination) dataset, curating by the LEADERS'
EXECUTOR PROFILE in the same pass that extracts the samples.

A candidate seat is replayed once; while it replays we accumulate both the
tools/tile_metrics step format and the v2 samples. At the end the seat's
executor metrics are computed and the game is KEPT only when it plays like
the three leaders (M & M & P & Q, Boey, DSM read 1.9-2.2 chores per visit,
0.71-0.83 moves per productive op, 0.46-0.55 pickups per feed and 4.4-4.7
wheat units per harvest). A rating floor alone would mix in seats that win
some other way and blur the target distribution.

Per kept game one npz:
  planes u8 [S, 24+8, 100]  board channels + the scheduler's request map
  day/hour i8 [S]
  ux, uy  i8 [N]            the unit's position (its own plane is built at load)
  scal    f16[N, N_SCALAR]  the v1 scalar vector
  sidx    i16[N]            index into the step arrays
  dest    i16[N]            0..99 tile | 100 SHED | 101 STAY
  opc     i8 [N]            request class at the destination, -1 when none
  ours    i8 [N]            our scheduler's destination class: 0 STAY 1 SHED 2 MOVE
  inmask  u8 [N]            the destination was expressible (a pending request)
The legality mask is NOT stored: it is rebuilt at load time from the request
planes (tiles with a request, minus the unit's own tile, plus SHED unless
the unit stands at the shed, plus STAY).

    PYTHONPATH=. uv run python rl/execnet/dataset2.py --n 260 --procs 3 \
        --replays replays/mmpq --replays replays/boey --replays replays/dsm
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from kagent.gamestate import Memory, View  # noqa: E402
from rl.execnet import features as F  # noqa: E402
from rl.execnet import features2 as F2  # noqa: E402

# the three leaders' profile (docs/top3_executor_comparison.md); a seat is
# kept when it clears all four
PROFILE = {"chores_per_visit": (1.7, None), "moves_per_productive": (None, 0.95),
           "pickups_per_feed": (None, 0.7), "wheat_units_per_harvest": (4.0, None)}


def passes(m):
    def get(k):
        if k == "wheat_units_per_harvest":
            return (m.get("one_time_units_per_harvest") or {}).get("WHEAT")
        return m.get(k)
    for k, (lo, hi) in PROFILE.items():
        v = get(k)
        if v is None:
            return False, k
        if lo is not None and v < lo:
            return False, k
        if hi is not None and v > hi:
            return False, k
    return True, None


class Collector:
    """One seat of one game: v2 samples plus the tile_metrics step format."""

    def __init__(self):
        from kagent.policy import ChampionPlanProvider, HeuristicPolicy
        self.pol = HeuristicPolicy(ChampionPlanProvider())
        self.mem = Memory()
        self.planes, self.day, self.hour = [], [], []
        self.ux, self.uy, self.scal, self.sidx, self.ours, self.reqtiles = [], [], [], [], [], []
        self.unit_ops = {}        # unit index -> [op per step]
        self.unit_pos = {}        # unit index -> [(x, y) per step]
        self.n_steps = 0
        self.metric_steps = []

    def add(self, obs, action):
        from tools.game_trace import tile_code
        view = View(obs)
        self.mem.observe(view)
        k = self.n_steps
        self.n_steps += 1
        plan = None
        try:
            plan = self.pol.make_plan(view, self.mem) if (
                self.mem.day_plan is None or self.mem.day_plan_day != view.day) else self.mem.day_plan
            self.mem.day_plan, self.mem.day_plan_day = plan, view.day
            from kagent import scheduler
            tasks = scheduler.build_tasks(view, plan)
            plan.fert_reserve = sum(1 for t in tasks if t.op[0] == "FERTILIZE")
            our_actions = scheduler.assign(view, plan, tasks)
        except Exception:
            tasks, our_actions = [], {}
        req, req_tiles = F2.request_planes(tasks)
        board = F2.board_planes(view)
        self.planes.append(np.rint(np.asarray(board + req, dtype=np.float32) * 255).astype(np.uint8))
        self.day.append(view.day)
        self.hour.append(view.hour)
        self.reqtiles.append(req_tiles)
        counts = F.farm_counts(view)
        n = len(view.me.units)
        top_ops = F.unit_ops(action, n)
        for ui in range(n):
            idx, x, y = view.me.units[ui]
            _, s = F.unit_features(view, ui, counts)
            self.ux.append(x)
            self.uy.append(y)
            self.scal.append(np.asarray(s, dtype=np.float16))
            self.sidx.append(k)
            cls = F2.scheduler_destination(view, ui, our_actions.get(idx))
            self.ours.append({"STAY": 0, "SHED": 1, "MOVE": 2}[cls])
            self.unit_ops.setdefault(ui, []).append(top_ops[ui])
            self.unit_pos.setdefault(ui, []).append((x, y))
        # pad units that vanished this step (hands are hired mid-day)
        for ui in list(self.unit_ops):
            while len(self.unit_ops[ui]) < self.n_steps:
                self.unit_ops[ui].append(None)
                self.unit_pos[ui].append(self.unit_pos[ui][-1] if self.unit_pos[ui] else (4, 4))
        acts = [top_ops[0] or ["PASS"]] + [o or ["PASS"] for o in top_ops[1:]]
        self.metric_steps.append({
            "day": view.day, "hour": view.hour,
            "tiles": [[tile_code(c, view.day) for c in row] for row in view.me.tiles],
            "units": [(x, y) for _, x, y in view.me.units], "acts": acts,
            "market": list((action or {}).get("market") or []) if isinstance(action, dict) else [],
        })

    def arrays(self):
        if not self.sidx:
            return None
        dests, opcs = {}, {}
        for ui, ops in self.unit_ops.items():
            dests[ui], opcs[ui] = F2.destinations(ops, self.unit_pos[ui],
                                                  self.day, self.n_steps)
        # walk the per-unit label series back onto the flat sample list
        per_step_unit = {}
        for j, k in enumerate(self.sidx):
            per_step_unit.setdefault(k, []).append(j)
        dest = np.full(len(self.sidx), F2.DEST_STAY, dtype=np.int16)
        opc = np.full(len(self.sidx), -1, dtype=np.int8)
        inmask = np.zeros(len(self.sidx), dtype=np.uint8)
        for k, js in per_step_unit.items():
            for ui, j in enumerate(js):
                d = dests[ui][k] if ui in dests and k < len(dests[ui]) else F2.DEST_STAY
                c = opcs[ui][k] if ui in opcs and k < len(opcs[ui]) else -1
                x, y = self.ux[j], self.uy[j]
                if d == F2.DEST_SHED and (x, y) in F2.SHED_SET:
                    d = F2.DEST_STAY
                ok = (d in (F2.DEST_SHED, F2.DEST_STAY)
                      or (d % F2.BOARD, d // F2.BOARD) in self.reqtiles[k])
                dest[j], opc[j], inmask[j] = d, c, 1 if ok else 0
        return {
            "planes": np.stack(self.planes),
            "day": np.asarray(self.day, dtype=np.int8), "hour": np.asarray(self.hour, dtype=np.int8),
            "ux": np.asarray(self.ux, dtype=np.int8), "uy": np.asarray(self.uy, dtype=np.int8),
            "scal": np.stack(self.scal).astype(np.float16),
            "sidx": np.asarray(self.sidx, dtype=np.int16),
            "dest": dest, "opc": opc,
            "ours": np.asarray(self.ours, dtype=np.int8), "inmask": inmask,
        }


def _finish(col, out_path, final, keep_all=False):
    import tile_metrics
    if not col.metric_steps:
        return out_path, 0, None, "empty"
    m = tile_metrics.analyse(col.metric_steps, final)
    ok, why = passes(m)
    prof = {"chores_per_visit": m.get("chores_per_visit"),
            "moves_per_productive": m.get("moves_per_productive"),
            "pickups_per_feed": m.get("pickups_per_feed"),
            "wheat_units_per_harvest": (m.get("one_time_units_per_harvest") or {}).get("WHEAT")}
    if not ok and not keep_all:
        return out_path, 0, prof, why
    arr = col.arrays()
    if arr is None:
        return out_path, 0, prof, "empty"
    np.savez_compressed(out_path, **arr)
    return out_path, int(len(arr["dest"])), prof, None


def build_session(job):
    path, seat, out_path = job[:3]
    keep_all = len(job) > 3 and job[3]
    if os.path.exists(out_path):
        return out_path, -1, None, None
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from rl import rollout
    with open(path) as fh:
        session = json.load(fh)
    col = Collector()
    tape = rollout.make_script_agent(session["actions"][seat])
    other = rollout.make_script_agent(session["actions"][1 - seat])

    def top_agent(obs):
        a = tape(obs)
        col.add(obs, a)
        return a
    agents = [None, None]
    agents[seat] = top_agent
    agents[1 - seat] = other
    env = make("kaggriculture", configuration={"seed": session["seed"]}, debug=False)
    env.run(agents)
    fin = session.get("finals") or [0, 0]
    final = {"me": float(fin[seat]), "opp": float(fin[1 - seat]),
             "win": float(fin[seat]) > float(fin[1 - seat])}
    return _finish(col, out_path, final, keep_all=keep_all)


def build_replay(job):
    path, seat, out_path = job
    if os.path.exists(out_path):
        return out_path, -1, None, None
    with open(path) as fh:
        d = json.load(fh)
    steps = d["steps"]
    col = Collector()
    for k in range(len(steps) - 1):
        col.add(steps[k][seat]["observation"], steps[k + 1][seat].get("action"))
    fin = [float(steps[-1][s].get("reward") or 0.0) for s in range(2)]
    final = {"me": fin[seat], "opp": fin[1 - seat], "win": fin[seat] > fin[1 - seat]}
    return _finish(col, out_path, final, keep_all=True)   # the three leaders define the profile


def _team_names(info):
    tn = info.get("TeamNames")
    if isinstance(tn, str):
        tn = ast.literal_eval(tn)
    return list(tn or [])


EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]


def band_of(r):
    for i, (lo, hi) in enumerate(EDGES):
        if lo <= r < hi:
            return i
    return len(EDGES) - 1


def select_sessions(index_path, min_rating, n, per_band=0, seed=0):
    """Highest-rated seats first (the curated pass), or `per_band` seats drawn
    from each of the six standard rating bins (the conditioned pass, which
    needs the whole range including the untidy low bands)."""
    import random
    with open(index_path) as fh:
        idx = json.load(fh)
    rows = []
    for sid, v in idx.items():
        sc = v.get("scores") or []
        if len(sc) < 2 or any(s is None for s in sc) or int(sid) % 5 == 0:
            continue
        seat = 0 if sc[0] >= sc[1] else 1
        if not per_band and sc[seat] < min_rating:
            continue
        rows.append((sc[seat], sid, v["path"], seat))
    if not per_band:
        rows.sort(reverse=True)
        return rows[:n]
    rng = random.Random(seed)
    out = []
    for b in range(len(EDGES)):
        pool = [r for r in rows if band_of(r[0]) == b]
        rng.shuffle(pool)
        out.extend(pool[:per_band])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--per-band", type=int, default=0,
                    help="draw this many seats from EACH of the six rating bins (conditioned pass)")
    ap.add_argument("--keep-all", action="store_true",
                    help="store every seat, profile and all, instead of only the ones that pass")
    ap.add_argument("--min-rating", type=float, default=2900.0)
    ap.add_argument("--replays", action="append", default=[])
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "rl/datasets/execnet2"))
    ap.add_argument("--index", default=os.path.join(REPO, "rl/datasets/session_index.json"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    jobs_s, jobs_r = [], []
    for rating, sid, path, seat in select_sessions(args.index, args.min_rating, args.n,
                                                   args.per_band):
        p = path if os.path.isabs(path) else os.path.join(REPO, path)
        gid = f"session-{sid}"
        meta.setdefault(gid, {})
        meta[gid].update({"rating": rating, "band": band_of(rating), "seat": seat, "source": "session"})
        jobs_s.append((p, seat, os.path.join(args.out, gid + ".npz"), args.keep_all))
    for d in args.replays:
        tag = os.path.basename(d.rstrip("/")).lower()
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                rep = json.load(fh)
            names = _team_names(rep.get("info", {}))
            by_dir = [i for i, nm in enumerate(names)
                      if tag and tag in nm.lower().replace(" ", "").replace("&", "")]
            seat = by_dir[0] if by_dir else (0 if (rep["steps"][-1][0].get("reward") or 0) >= (rep["steps"][-1][1].get("reward") or 0) else 1)
            gid = "replay-" + os.path.splitext(os.path.basename(f))[0]
            meta.setdefault(gid, {})
            meta[gid].update({"player": names[seat] if names else "?", "seat": seat,
                              "source": tag, "band": len(EDGES) - 1})
            jobs_r.append((f, seat, os.path.join(args.out, gid + ".npz")))
    print(f"{len(jobs_s)} session candidates + {len(jobs_r)} replays -> {args.out}", flush=True)
    t0 = time.time()
    kept = dropped = 0
    with Pool(args.procs) as pool:
        for fn, jobs in ((build_replay, jobs_r), (build_session, jobs_s)):
            for out_path, n, prof, why in pool.imap_unordered(fn, jobs):
                gid = os.path.splitext(os.path.basename(out_path))[0]
                if n >= 0:
                    meta[gid].update({"n": n, "profile": prof, "kept": bool(n), "failed_on": why})
                    kept += 1 if n else 0
                    dropped += 0 if n else 1
                if (kept + dropped) % 20 == 0:
                    print(f"  {kept} kept / {dropped} dropped, {time.time() - t0:.0f}s", flush=True)
                    json.dump(meta, open(meta_path, "w"), indent=1)
    json.dump(meta, open(meta_path, "w"), indent=1)
    total = sum(m.get("n", 0) for m in meta.values())
    print(f"done: {kept} kept, {dropped} dropped, {total:,} samples, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
