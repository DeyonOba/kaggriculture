"""BC samples WITH board tensors, from the action-only session corpus.

`rl/replay_bc.py` reads full Kaggle replays, which carry every observation.
There are 223 of those. The corpus is 21,701 SESSIONS -- seed plus both
720-step action scripts, ~270 KB each instead of ~33 MB -- and nothing built
BC samples from them, so plumbing a board tensor through `replay_bc` would
have left the trunk on 8,310 samples against 42,192 parameters.

This replays a session exactly (recorded seed, both recorded scripts, no agent
in the loop), regenerates the observations, and emits at each dawn:

    x        168 features, from that seat's own view
    masks    the legal-action mask
    choices  the standing targets that seat's play IMPLIES, quantised onto
             our heads -- end-of-day crop counts, herd, hires, land
    board    (103, 10, 10) float16, arm T_crop_103

Two things it does differently from `replay_bc`, both deliberate.

BOARDS GO TO A MEMMAP, NOT THE JSONL. 103x10x10 float16 is 20 KB a sample;
at 150,000 samples that is 3 GB, which does not belong in a line-delimited
JSON file. The JSONL keeps `bi`, an index into a companion `.npy`.

SESSIONS ARE DRAWN PER RATING BAND. The corpus is 68% 2500+, so sampling it
as it lies trains the trunk on the top band and calls it general -- the
mistake `tools/net_ood.py` exists to catch, and the one the band protocol was
adopted to stop.

    PYTHONPATH=. uv run python rl/session_bc.py --n 5000 --procs 5 \\
        --out rl/datasets/session_bc.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TPD = 24


def is_held_out(path):
    """rl/train.py reserves session ids with id % 5 == 0 as its held-out
    field eval. Anything that FITS on outcomes -- the trunk, a value head --
    must not see them, or the honest eval scores memorised games. Both trunk
    builders drew from every session before this existed."""
    try:
        return int(os.path.basename(path).split("-")[-1][:-5]) % 5 == 0
    except ValueError:
        return False


def extract(spec):
    """Replay one session; return (samples, boards) for both seats.

    Each sample carries BOTH supervised targets, because they are not equally
    useful and the choice should be made at training time, not baked in here:

      choices  the champion's implied standing targets. This is what BC has
               always cloned, and cloning it gated NULL over 4,800 paired
               games at 69.4% action accuracy -- ten of thirteen heads are
               ALLOCATION, and allocation is measured flat.
      m, w     the seat's final margin and win. This is the target H-margin
               passed on: T_crop_103 beat E_full_83 by >= +0.01 R2 in 4 of 5
               rating bands, with win AUC up in 4 of 5.

    Pretraining a representation on a measured null teaches it the thing that
    does not pay. `m` is the one with evidence behind it.
    """
    path, arm, margin, win = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from kagent import features
    from kagent.gamestate import Memory, View
    from tools.board_tensor import board_pair_temporal_both, arm_channels
    from rl.replay_bc import _counts, _targets_to_choices

    try:
        with open(path) as fh:
            s = json.load(fh)
    except Exception:
        return None
    scripts = s.get("actions") or []
    if len(scripts) != 2 or not scripts[0] or not scripts[1]:
        return None

    def at(script, t):
        if t < len(script) and script[t]:
            return script[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    e = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    e.reset(2)
    dawn, mid, end = {}, {}, {}
    while not e.done:
        o0 = e.state[0].observation
        t = o0["step"]
        d, h = t // TPD, t % TPD
        if h == 0:
            dawn[d] = (o0, e.state[1].observation)
        elif h == 12:
            mid[d] = (o0, e.state[1].observation)
        end[d] = (o0, e.state[1].observation)          # last frame seen this day
        e.step([at(scripts[0], t), at(scripts[1], t)])

    cols = arm_channels(arm)
    out, boards = [], []
    for side in (0, 1):
        mem = Memory()
        hist = {d: fr[0] for d, fr in dawn.items()}     # public state is shared
        mids = {d: fr[0] for d, fr in mid.items()}
        for d in sorted(dawn):
            if d not in end:
                continue
            pub, own = dawn[d][0], dawn[d][side]
            obs = dict(pub)
            obs["private"] = own.get("private") or {}
            obs["player"] = side
            obs.setdefault("step", d * TPD)
            if "farms" not in obs:
                continue
            try:
                view = View(obs)
                x = features.encode(view, mem)
                m = features.masks(view)
                e_pub, e_own = end[d][0], end[d][side]
                farm_end = e_pub["farms"][side]
                crops, herd = _counts(farm_end["tiles"])
                shed = (e_own.get("private") or {}).get("shed") or {}
                bought = (len(farm_end.get("unlocked_quadrants", []))
                          > len(pub["farms"][side].get("unlocked_quadrants", [])))
                choices = _targets_to_choices(view, m, crops, herd, shed,
                                              farm_end.get("hires_today", 0), bought)
                past = [hist[k] for k in range(d - 1, -1, -1) if k in hist]
                t3 = board_pair_temporal_both(pub, side, prev_mid=mids.get(d - 1),
                                              hist=past)
            except Exception:
                continue
            out.append({"x": x, "masks": m, "choices": choices,
                        "m": margin[side], "w": win[side], "seat": side,
                        "day": d})
            boards.append(t3[cols].astype("float16"))
    if not out:
        return None
    return out, np.stack(boards)


def main() -> None:
    from multiprocessing import Pool
    from tools.ladder_panel import pick_sessions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default=os.environ.get("KAGG_BOARD_ARM", "T_crop_103"))
    ap.add_argument("--out", default=os.path.join(REPO, "rl", "datasets",
                                                  "session_bc.jsonl"))
    ap.add_argument("--include-held-out", action="store_true",
                    help="keep the id %% 5 == 0 sessions train.py holds out. "
                         "Only for a dataset that will never train anything "
                         "PPO is then evaluated on")
    args = ap.parse_args()

    paths = pick_sessions(args.n, True, args.seed)
    if not args.include_held_out:
        before = len(paths)
        paths = [p for p in paths if not is_held_out(p)]
        print(f"  excluded {before - len(paths)} PPO held-out sessions "
              f"(id % 5 == 0); {len(paths)} remain")
    # Outcomes come from the index, not from a second replay: `finals` is
    # already there, and margin is seat-relative.
    import json as _json
    with open(os.path.join(REPO, "rl", "datasets", "session_index.json")) as fh:
        _idx = _json.load(fh)
    _by = {v["path"]: v for v in _idx.values() if v.get("path")}

    def _outcome(p):
        rel = p[len(REPO) + 1:] if p.startswith(REPO) else p
        fin = ((_by.get(rel) or _by.get(p) or {}).get("finals")) or []
        if len(fin) != 2:
            return (float("nan"),) * 2, (float("nan"),) * 2
        a, b = float(fin[0]), float(fin[1])
        return (a - b, b - a), (1.0 if a > b else 0.0, 1.0 if b > a else 0.0)
    print(f"replaying {len(paths)} sessions on {args.procs} procs, arm {args.arm} ...",
          flush=True)
    board_path = args.out.replace(".jsonl", "_boards.npy")
    # Stream into a preallocated memmap. Holding every chunk and then
    # concatenating needs the whole array twice over in RAM -- at 5,000
    # sessions that is ~12 GB peak, which is the shape of the OOM that broke
    # the first band-stratified flow build.
    from tools.board_tensor import arm_channels
    n_ch = len(arm_channels(args.arm))
    cap = len(paths) * 62                      # 2 seats x 30 days, with slack
    B = np.lib.format.open_memmap(board_path, mode="w+", dtype="float16",
                                  shape=(cap, n_ch, 10, 10))
    t0, n_s, n_b = time.time(), 0, 0
    # The sessions ACTUALLY written, in order, with their sample counts. The
    # trunk pretrainer used to reconstruct this from pick_sessions(seed) and
    # bi // 60, which breaks the moment a session is filtered or a day is
    # dropped; now it reads this file.
    used = []
    with open(args.out, "w") as fh, Pool(args.procs) as pool:
        jobs = [(p, args.arm) + _outcome(p) for p in paths]
        for i, res in enumerate(pool.imap(extract, jobs), 1):   # ordered
            if res is None:
                continue
            samples, boards = res
            if n_b + len(boards) > cap:        # never write past the allocation
                print(f"    capacity {cap:,} reached at session {i}; stopping")
                break
            for k, smp in enumerate(samples):
                smp["bi"] = n_b + k
                fh.write(json.dumps(smp) + "\n")
            B[n_b:n_b + len(boards)] = boards
            p_i = paths[i - 1]
            used.append({"path": p_i[len(REPO) + 1:] if p_i.startswith(REPO) else p_i,
                         "n": len(samples)})
            n_s += len(samples); n_b += len(boards)
            if i % 200 == 0:
                print(f"    {i}/{len(paths)} sessions, {n_s:,} samples, "
                      f"{n_b * n_ch * 100 * 2 / 1e9:.1f} GB, "
                      f"{time.time()-t0:.0f}s", flush=True)
    if n_b == 0:
        raise SystemExit("nothing extracted")
    with open(args.out.replace(".jsonl", "_sessions.json"), "w") as sfh:
        json.dump({"held_out_excluded": not args.include_held_out,
                   "arm": args.arm, "seed": args.seed, "sessions": used}, sfh)
    B.flush(); del B
    # Truncate the memmap to what was actually written.
    full = np.load(board_path, mmap_mode="r")[:n_b]
    np.save(board_path + ".tmp.npy", np.asarray(full))
    del full
    os.replace(board_path + ".tmp.npy", board_path)
    B = np.load(board_path, mmap_mode="r")
    print(f"-> {args.out}  {n_s:,} samples")
    print(f"-> {board_path}  {B.shape} {B.dtype}  "
          f"{os.path.getsize(board_path)/1e9:.2f} GB")
    from rl.common import BoardTrunk
    tp = sum(p.numel() for p in BoardTrunk().parameters())
    print(f"   trunk params {tp:,} -> {len(B)/tp:.1f} samples/param")


if __name__ == "__main__":
    main()
