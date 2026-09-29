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
HORIZON = int(os.environ.get("KAGG_BC_HORIZON", "1"))   # label days ahead; 1 = same day


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
    from kagent.constants import ANIMALS
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
                m = features.masks(view, horizon=HORIZON)   # capacity AT the label horizon
                # LABEL HORIZON. The default labels dawn d with the stock at the
                # END of day d -- a same-day target. A 3-day planner needs the
                # stock at the end of day d+2: what the leader had standing
                # three dawns on, which is what a 3-day commitment must reach.
                # Clamped to the last recorded day so the endgame still labels.
                d_lab = min(d + HORIZON - 1, max(end))
                e_pub, e_own = end[d_lab][0], end[d_lab][side]
                farm_end = e_pub["farms"][side]
                crops, herd = _counts(farm_end["tiles"])
                shed = (e_own.get("private") or {}).get("shed") or {}
                bought = (len(farm_end.get("unlocked_quadrants", []))
                          > len(pub["farms"][side].get("unlocked_quadrants", [])))
                # FERTILIZE actions per day over the label window, from this
                # side's recorded script -- the v5 fert_budget label.
                n_fert = 0
                for _t in range(d * TPD, min((d_lab + 1) * TPD, len(scripts[side]))):
                    _a = at(scripts[side], _t)
                    n_fert += sum(1 for _u in [_a.get("farmer")] + list(_a.get("hands") or [])
                                  if _u and _u[0] == "FERTILIZE")
                fert_per_day = int(round(n_fert / max(1, d_lab - d + 1)))
                choices = _targets_to_choices(view, m, crops, herd, shed,
                                              farm_end.get("hires_today", 0), bought,
                                              fert=fert_per_day)
                past = [hist[k] for k in range(d - 1, -1, -1) if k in hist]
                t3 = board_pair_temporal_both(pub, side, prev_mid=mids.get(d - 1),
                                              hist=past)
            except Exception:
                continue
            # `choices` are head INDICES, quantised onto features.HEADS as it
            # stands today, so a change to a head's value list silently
            # invalidates them (2026-09-26: the goose head went from
            # [0,8,16,24,32] to [0,4,6,10,16] and every stored goose index
            # changed meaning). Keep the RAW standing targets beside them so
            # the next head change can re-quantise without replaying 16,000
            # games -- see rl/session_bc.requantise().
            out.append({"x": x, "masks": m, "choices": choices,
                        "raw": {"crops": crops, "herd": herd,
                                "shed": {k: v for k, v in shed.items() if k in ANIMALS},
                                "hires": farm_end.get("hires_today", 0),
                                "fert": fert_per_day,
                                "land": bool(bought)},
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
    ap.add_argument("--horizon", type=int, default=1,
                    help="label dawn d with the stock at the end of day "
                         "d+horizon-1. 1 = same day (the historical corpus); "
                         "3 = the 3-day planner's own commitment window. "
                         "Exported to KAGG_BC_HORIZON so spawn workers see it.")
    ap.add_argument("--heads-version", type=int,
                    default=int(os.environ.get("KAGG_HEADS_V", 0)) or None,
                    help="action space to QUANTISE THE LABELS ONTO. Without "
                         "this the corpus is pinned to whatever HEADS_VERSION "
                         "happens to be at import (1), no matter what the "
                         "trainer is later switched to -- and the label is "
                         "then wrong before training starts. Measured over "
                         "1,200 leader dawns, decoded back to absolute tiles: "
                         "v1 names the true count on 37.8%% of dawns and is "
                         "off by 2.16 tiles on average (wheat 6.1%%/6.11 "
                         "tiles, strawberry 9.1%%/6.78); v2 42.7%%/2.08; v3 "
                         "35.0%%/3.51; v4 98.7%%/0.04.")
    ap.add_argument("--out", default=os.path.join(REPO, "rl", "datasets",
                                                  "session_bc.jsonl"))
    ap.add_argument("--include-held-out", action="store_true",
                    help="keep the id %% 5 == 0 sessions train.py holds out. "
                         "Only for a dataset that will never train anything "
                         "PPO is then evaluated on")
    ap.add_argument("--sessions", default=None,
                    help="a glob of session files to use INSTEAD of the rating "
                         "index -- e.g. a fresh harvest converted with "
                         "tools.extract_episodes.to_session. --n/--min-rating "
                         "are ignored and the PPO held-out exclusion is skipped.")
    ap.add_argument("--min-rating", type=float, default=None,
                    help="TOP-PLAYER mode: every session with a seat rated at "
                         "least this (max of the two seats), instead of the "
                         "band-stratified draw; --n 0 takes all of them")
    args = ap.parse_args()

    from kagent import features        # module-scope import is worker-local here

    global HORIZON
    HORIZON = int(args.horizon)
    os.environ["KAGG_BC_HORIZON"] = str(HORIZON)     # spawn workers re-import
    print(f"label horizon: {HORIZON} day(s)")

    if args.heads_version:
        # BEFORE any worker is spawned and before any label is built.
        # `set_heads_version` exports KAGG_HEADS_V, which is what carries the
        # choice into a spawn worker -- a version set only in the parent
        # leaves every worker on v1 while the parent thinks it is on v4.
        features.set_heads_version(args.heads_version)
    print(f"heads v{features.HEADS_VERSION}: {len(features.HEADS)} heads, "
          f"{sum(len(v) for _n, v in features.HEADS)} logits")


    # Outcomes come from the index, not from a second replay: `finals` is
    # already there, and margin is seat-relative.
    import json as _json
    if args.sessions:
        import glob as _glob
        paths = sorted(os.path.abspath(p) for p in _glob.glob(args.sessions))
        print(f"  {len(paths)} sessions from --sessions {args.sessions}")
        args.include_held_out = True
    with open(os.path.join(REPO, "rl", "datasets", "session_index.json")) as fh:
        _idx = _json.load(fh)
    _by = {v["path"]: v for v in _idx.values() if v.get("path")}

    if args.sessions:
        pass                                   # explicit list, already set
    elif args.min_rating is not None:
        paths = []
        for v in _idx.values():
            sc = [x for x in (v.get("scores") or []) if x is not None]
            if v.get("path") and sc and max(sc) >= args.min_rating:
                paths.append(os.path.join(REPO, v["path"]))
        paths.sort()
        np.random.default_rng(args.seed).shuffle(paths)
        if args.n and 0 < args.n < len(paths):
            paths = paths[:args.n]
        print(f"  {len(paths)} sessions with a seat rated >= {args.min_rating:.0f}")
    else:
        paths = pick_sessions(args.n, True, args.seed)
    if not args.include_held_out:
        before = len(paths)
        paths = [p for p in paths if not is_held_out(p)]
        print(f"  excluded {before - len(paths)} PPO held-out sessions "
              f"(id % 5 == 0); {len(paths)} remain")

    # extract() answers None for an unreadable session and the consumer loop
    # skips it, so a corpus that never mounted prints "nothing extracted" and
    # exits 0 -- it cost two Kaggle runs before anyone read it as a missing
    # mount rather than an empty replay. The corpus is external input, so
    # check it here and say which path was expected.
    missing = sum(1 for p in paths if not os.path.exists(p))
    if missing == len(paths) and paths:
        raise SystemExit(
            f"no session file exists on disk (checked {len(paths)}); the corpus "
            f"is not mounted. First expected path: {paths[0]}")
    if missing:
        print(f"  WARNING: {missing}/{len(paths)} session files missing from disk")

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
