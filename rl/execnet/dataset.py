"""Build the execnet dataset: per-(step, unit) samples of what a top player's
unit did, with the features and legality mask of rl/execnet/features.py, and
what OUR scheduler (today's kagent/, champion plan) would have done from the
same observation.

Sources:
  * recorded sessions rl/datasets/sessions/session-*.json whose better seat
    is rated >= --min-rating in rl/datasets/session_index.json, not held out
    (id % 5 == 0 is the eval's draw, rl/session_bc.is_held_out), highest
    ratings first, capped at --n; each is re-simulated with both seats
    scripted (tools/fast_env for speed) to recover the observations
  * Kaggle replay JSONs (replays/mmpq, replays/boey, replays/dsm ...): the
    observations are in the file, no simulation

Output: one npz per game under --out (resumable) with
  tiles  uint8  [N, N_TILE]     channel values x 255
  scal   f16    [N, N_SCALAR]
  mask   uint8  [N, len(OPS)]
  y      int8   [N]             top player's op (index into OPS), -1 dropped
  ours   int8   [N]             our scheduler's op on the same state, -1 none
  step   int16  [N]   unit int8 [N]
plus meta.json {game_id: {"player":..., "rating":..., "n":...}}.

    PYTHONPATH=. uv run python rl/execnet/dataset.py --n 120 --procs 2 \\
        --replays replays/mmpq --out rl/datasets/execnet
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

from kagent.gamestate import Memory, View  # noqa: E402
from rl.execnet import features as F  # noqa: E402

OUR_KEY = "ours"


def _our_policy():
    from kagent.policy import ChampionPlanProvider, HeuristicPolicy
    return HeuristicPolicy(ChampionPlanProvider()), Memory()


class Collector:
    """Accumulates samples for one seat of one game."""

    def __init__(self):
        self.tiles, self.scal, self.mask, self.y, self.ours, self.step, self.unit = [], [], [], [], [], [], []
        self.pol, self.mem = _our_policy()

    def add(self, obs, action):
        view = View(obs)
        self.mem.observe(view)
        try:
            our_action = self.pol.act(view, self.mem)
        except Exception:
            our_action = None
        counts = F.farm_counts(view)
        n = len(view.me.units)
        top_ops = F.unit_ops(action, n)
        our_ops = F.unit_ops(our_action, n) if our_action else [None] * n
        for ui in range(n):
            lab = F.label_of(top_ops[ui])
            if lab < 0:
                continue
            t, s = F.unit_features(view, ui, counts)
            self.tiles.append(np.asarray(t, dtype=np.float32))
            self.scal.append(np.asarray(s, dtype=np.float16))
            self.mask.append(np.asarray(F.legal_mask(view, ui), dtype=np.uint8))
            self.y.append(lab)
            self.ours.append(F.label_of(our_ops[ui]) if our_ops[ui] is not None else -1)
            self.step.append(view.step)
            self.unit.append(ui)

    def arrays(self):
        if not self.y:
            return None
        return {
            "tiles": np.rint(np.stack(self.tiles) * 255).astype(np.uint8),
            "scal": np.stack(self.scal).astype(np.float16),
            "mask": np.stack(self.mask).astype(np.uint8),
            "y": np.asarray(self.y, dtype=np.int8),
            OUR_KEY: np.asarray(self.ours, dtype=np.int8),
            "step": np.asarray(self.step, dtype=np.int16),
            "unit": np.asarray(self.unit, dtype=np.int8),
        }


def build_session(job):
    """job: (path, seat, out_path). Re-simulate with both seats scripted."""
    path, seat, out_path = job
    if os.path.exists(out_path):
        return out_path, -1
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from rl import rollout

    with open(path) as fh:
        session = json.load(fh)
    actions = session["actions"]
    col = Collector()
    tape = rollout.make_script_agent(actions[seat])
    other = rollout.make_script_agent(actions[1 - seat])

    def top_agent(obs):
        a = tape(obs)
        col.add(obs, a)
        return a

    agents = [None, None]
    agents[seat] = top_agent
    agents[1 - seat] = other
    env = make("kaggriculture", configuration={"seed": session["seed"]}, debug=False)
    env.run(agents)
    arr = col.arrays()
    if arr is None:
        return out_path, 0
    np.savez_compressed(out_path, **arr)
    return out_path, int(len(arr["y"]))


def build_replay(job):
    """job: (path, seat, out_path). Observations are in the replay."""
    path, seat, out_path = job
    if os.path.exists(out_path):
        return out_path, -1
    with open(path) as fh:
        d = json.load(fh)
    steps = d["steps"]
    col = Collector()
    for k in range(len(steps) - 1):
        obs = steps[k][seat]["observation"]
        act = steps[k + 1][seat].get("action")
        col.add(obs, act)
    arr = col.arrays()
    if arr is None:
        return out_path, 0
    np.savez_compressed(out_path, **arr)
    return out_path, int(len(arr["y"]))


def _team_names(info):
    tn = info.get("TeamNames")
    if isinstance(tn, str):
        tn = ast.literal_eval(tn)
    return list(tn or [])


def select_sessions(index_path, min_rating, n):
    with open(index_path) as fh:
        idx = json.load(fh)
    rows = []
    for sid, v in idx.items():
        sc = v.get("scores") or []
        if len(sc) < 2 or any(s is None for s in sc):
            continue
        if int(sid) % 5 == 0:
            continue                      # the eval's held-out draw
        seat = 0 if sc[0] >= sc[1] else 1
        if sc[seat] < min_rating:
            continue
        rows.append((sc[seat], sid, v["path"], seat))
    rows.sort(reverse=True)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=120, help="sessions (highest-rated seats first)")
    ap.add_argument("--min-rating", type=float, default=3000.0)
    ap.add_argument("--replays", action="append", default=[],
                    help="replay directory; the seat is the team named by --team, else the higher final")
    ap.add_argument("--team", default="M & M & P & Q")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(REPO, "rl/datasets/execnet"))
    ap.add_argument("--index", default=os.path.join(REPO, "rl/datasets/session_index.json"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    jobs_s, jobs_r = [], []
    for rating, sid, path, seat in select_sessions(args.index, args.min_rating, args.n):
        p = path if os.path.isabs(path) else os.path.join(REPO, path)
        with open(p) as fh:
            teams = json.load(fh).get("teams")
        if isinstance(teams, str):
            teams = ast.literal_eval(teams)
        gid = f"session-{sid}"
        meta[gid] = {"player": (teams or ["?", "?"])[seat], "rating": rating, "seat": seat, "source": "session"}
        jobs_s.append((p, seat, os.path.join(args.out, gid + ".npz")))
    for d in args.replays:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                rep = json.load(fh)
            names = _team_names(rep.get("info", {}))
            if args.team in names:
                seat = names.index(args.team)
            else:
                rewards = [(rep["steps"][-1][s].get("reward") or 0) for s in range(2)]
                seat = 0 if rewards[0] >= rewards[1] else 1
            gid = "replay-" + os.path.splitext(os.path.basename(f))[0]
            meta[gid] = {"player": names[seat] if names else "?", "rating": None, "seat": seat,
                         "source": os.path.basename(d.rstrip("/"))}
            jobs_r.append((f, seat, os.path.join(args.out, gid + ".npz")))
    print(f"{len(jobs_s)} sessions + {len(jobs_r)} replays -> {args.out}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(args.procs) as pool:
        for out_path, n in pool.imap_unordered(build_replay, jobs_r):
            gid = os.path.splitext(os.path.basename(out_path))[0]
            if n >= 0:
                meta[gid]["n"] = n
            done += 1
        for out_path, n in pool.imap_unordered(build_session, jobs_s):
            gid = os.path.splitext(os.path.basename(out_path))[0]
            if n >= 0:
                meta[gid]["n"] = n
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs_s) + len(jobs_r)} games, {time.time() - t0:.0f}s", flush=True)
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh, indent=1)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    total = sum(m.get("n", 0) for m in meta.values())
    print(f"done: {len(meta)} games, {total:,} samples, {time.time() - t0:.0f}s -> {meta_path}")


if __name__ == "__main__":
    main()
