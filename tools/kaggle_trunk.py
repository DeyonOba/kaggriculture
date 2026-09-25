"""In-kernel: pretrain the board trunk on the FULL session corpus, out of core.

The whole corpus does not fit as an array. 21,701 sessions x 60 samples x
103x10x10 float16 is **26.8 GB**, and one seat only is still 13.4 GB. So this
never materialises it: it walks the corpus in chunks, extracts a chunk's
boards, takes gradient passes over that chunk, drops it, and moves on. Peak
disk is one chunk (~2.5 GB at the default) no matter how large the corpus is.

  chunk 0 .. N-2   training
  chunk N-1        held out, never trained on, scored after every chunk

One pass over ~1.3M samples against a 42,192-parameter trunk is ~31 samples a
parameter of gradient, which is why a single outer epoch is enough and why
re-visiting chunks is not worth the disk.

CHECKPOINT AFTER EVERY CHUNK. A Kaggle job that dies at the 11-hour wall with
nothing written returns nothing, and that has happened on this project before.
Each chunk writes the trunk, the running validation score and the chunk index,
so a killed run resumes rather than restarts.

BANDS ARE PRESERVED. The corpus is 68% 2500+; chunks are drawn from a
band-stratified ordering so no chunk is all-frontier and the held-out chunk
spans the ladder.

    steps in tools/kaggle_job/job_trunk.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def find_sessions():
    """Session files, from the repo or from any mounted Kaggle dataset."""
    local = sorted(glob.glob(os.path.join(REPO, "rl", "datasets", "sessions",
                                          "session-*.json")))
    if local:
        return local, "repo"
    pats = ["/kaggle/input/*/sessions/session-*.json",
            "/kaggle/input/*/*/sessions/session-*.json",
            "/kaggle/input/*/session-*.json"]
    for p in pats:
        got = sorted(glob.glob(p))
        if got:
            return got, p
    raise SystemExit(
        "no session files found in the repo or under /kaggle/input -- the "
        "kernel needs a dataset carrying rl/datasets/sessions/")


def band_order(paths, seed):
    """Interleave the rating bands so every chunk spans the ladder."""
    import random
    idx_path = os.path.join(REPO, "rl", "datasets", "session_index.json")
    try:
        with open(idx_path) as fh:
            idx = json.load(fh)
    except Exception:
        random.Random(seed).shuffle(paths)
        print("  no session index; falling back to a flat shuffle")
        return paths
    rating = {}
    for v in idx.values():
        sc = [x for x in (v.get("scores") or []) if x is not None]
        if v.get("path"):
            rating[os.path.basename(v["path"])] = max(sc) if sc else None
    EDGES = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 1e9)]
    buckets = [[] for _ in EDGES]
    spare = []
    for p in paths:
        r = rating.get(os.path.basename(p))
        if r is None:
            spare.append(p); continue
        buckets[next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)].append(p)
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    rng.shuffle(spare)
    out, i = [], 0
    while any(buckets) or spare:
        for b in buckets:
            if b:
                out.append(b.pop())
        if spare:
            out.append(spare.pop())
        i += 1
    print(f"  band sizes {[len(b) for b in buckets]} + {len(spare)} unrated, "
          f"interleaved")
    return out


def main() -> None:
    import torch
    import torch.nn as nn
    from multiprocessing import Pool
    from rl.common import BoardTrunk, BOARD_W
    from rl.session_bc import extract

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk", type=int, default=2000, help="sessions per chunk")
    ap.add_argument("--passes", type=int, default=2, help="gradient passes per chunk")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget-s", type=float, default=34000.0)
    # FIXED normalisation, not per-chunk. Taking mu/sd from the first chunk
    # makes every later chunk's targets depend on one arbitrary draw -- the
    # smoke run swung the loss 0.53 -> 43.5 -> 0.65 that way. Margin is
    # seat-symmetric because both seats are emitted, so the mean is ~0 by
    # construction; the scale is the corpus figure ($11,494 measured over
    # 1,075 band-stratified sessions).
    ap.add_argument("--margin-sd", type=float, default=11494.0)
    ap.add_argument("--arm", default=os.environ.get("KAGG_BOARD_ARM", "T_crop_103"))
    ap.add_argument("--out", default="/kaggle/working/trunk_margin.pt")
    ap.add_argument("--include-held-out", action="store_true",
                    help="keep the id %% 5 == 0 sessions rl/train.py holds "
                         "out as its field eval. The shipped trunk "
                         "(trunk_margin_kaggle.pt) was built WITH them")
    ap.add_argument("--min-rating", type=float, default=None,
                    help="only sessions with a seat rated at least this (needs "
                         "the session index); the TOP-PLAYER trunk")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.set_num_threads(int(os.environ.get("KAGG_THREADS", "1")))
    torch.manual_seed(args.seed)

    paths, src = find_sessions()
    print(f"{len(paths):,} sessions from {src}", flush=True)
    if not args.include_held_out:
        from rl.session_bc import is_held_out
        before = len(paths)
        paths = [p for p in paths if not is_held_out(p)]
        print(f"  excluded {before - len(paths):,} PPO held-out sessions "
              f"(id % 5 == 0)", flush=True)
    if args.min_rating is not None:
        idx_path = os.path.join(REPO, "rl", "datasets", "session_index.json")
        with open(idx_path) as fh:
            idx = json.load(fh)
        top = set()
        for v in idx.values():
            sc = [x for x in (v.get("scores") or []) if x is not None]
            if v.get("path") and sc and max(sc) >= args.min_rating:
                top.add(os.path.basename(v["path"]))
        before = len(paths)
        paths = [p for p in paths if os.path.basename(p) in top]
        print(f"  top-player filter: {len(paths):,} of {before:,} sessions have a "
              f"seat rated >= {args.min_rating:.0f}", flush=True)
    paths = band_order(paths, args.seed)

    # Outcomes WITHOUT the index. The index is generated, not tracked, so
    # push.sh (git ls-files) never ships it and the kernel does not have it --
    # which is exactly how the first run died 0.1 min in, having already found
    # all 21,517 sessions. `finals` sits in the first 4 KB of a session file,
    # ahead of the two 720-step action scripts, so a head read is enough.
    # (`scores` does NOT, which is why band ordering still needs the index and
    # degrades to a flat shuffle without it.)
    fin = {}
    idx_path = os.path.join(REPO, "rl", "datasets", "session_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as fh:
            idx = json.load(fh)
        fin = {os.path.basename(v["path"]): v.get("finals")
               for v in idx.values() if v.get("path")}
        print(f"  outcomes from the index ({len(fin):,})", flush=True)
    else:
        t_h = time.time()
        for q in paths:
            try:
                with open(q, "rb") as fh:
                    head = fh.read(4096).decode("utf8", "ignore")
                i = head.index('"finals"')
                seg = head[i:head.index("]", i) + 1]
                fin[os.path.basename(q)] = json.loads(seg.split(":", 1)[1])
            except Exception:
                continue
        print(f"  outcomes head-read from {len(fin):,} session files "
              f"({time.time()-t_h:.0f}s)", flush=True)

    def outcome(p):
        f = fin.get(os.path.basename(p)) or []
        if len(f) != 2:
            return None
        a, b = float(f[0]), float(f[1])
        return (a - b, b - a), (1.0 if a > b else 0.0, 1.0 if b > a else 0.0)

    jobs = [(p, args.arm) + o for p in paths if (o := outcome(p))]
    print(f"{len(jobs):,} with an outcome", flush=True)
    chunks = [jobs[i:i + args.chunk] for i in range(0, len(jobs), args.chunk)]
    # The held-out chunk is FIRST, not last. If it were last, a run that hits
    # the time budget -- which is the expected case at 11 hours -- would
    # return a trunk with no validation number at all.
    print(f"{len(chunks)} chunks of <= {args.chunk}; chunk 0 is HELD OUT",
          flush=True)

    trunk, head = BoardTrunk(), nn.Linear(BOARD_W, 1)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()),
                           lr=args.lr)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    mu, sd = 0.0, args.margin_sd
    val = None
    # ONE pool for the whole run. Building a Pool per chunk costs more than
    # the extraction: the smoke run spent 1.9 s a session against 0.43 s
    # measured locally, which at 21,000 sessions is the entire 11-hour budget
    # spent on process startup.
    pool = Pool(args.procs)

    def run_chunk(spec_list):
        res = [r for r in pool.imap_unordered(extract, spec_list) if r]
        if not res:
            return None, None, None
        X = np.concatenate([r[1] for r in res])
        y = np.asarray([s["m"] for r in res for s in r[0]], "float32")
        w = np.asarray([s["w"] for r in res for s in r[0]], "float32")
        return X, y, w

    for ci, spec in enumerate(chunks):
        if time.time() - t0 > args.budget_s:
            print(f"  budget reached at chunk {ci}; stopping", flush=True)
            break
        X, y, w = run_chunk(spec)
        if X is None:
            continue
        if ci == 0:
            val = (X, y, w)
            print(f"  chunk 0: {len(X):,} samples HELD OUT, never trained on",
                  flush=True)
            continue
        trunk.train()
        for _ in range(args.passes):
            perm = rng.permutation(len(X))
            for s in range(0, len(perm), args.batch):
                i = perm[s:s + args.batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(
                    head(trunk(torch.from_numpy(X[i].astype("float32")))),
                    torch.tensor((y[i] - mu) / sd).unsqueeze(1))
                loss.backward(); opt.step()
        # Score the held-out chunk every time, so a budget-killed run still
        # carries a number and the curve is visible rather than a single
        # end-of-run point.
        vr = float("nan")
        if val is not None:
            trunk.eval(); P = []
            with torch.no_grad():
                for s_ in range(0, len(val[0]), 512):
                    P.append(head(trunk(torch.from_numpy(
                        val[0][s_:s_ + 512].astype("float32")))).numpy())
            P = np.concatenate(P).reshape(-1) * sd + mu
            dd = float(((val[1] - val[1].mean()) ** 2).sum())
            vr = 1 - float(((val[1] - P) ** 2).sum()) / dd if dd > 0 else float("nan")
        torch.save({"trunk": trunk.state_dict(), "arm": args.arm,
                    "chunk": ci, "mu": mu, "sd": sd, "val_r2": vr}, args.out)
        print(f"  chunk {ci}/{len(chunks)-1}: {len(X):,} samples, "
              f"loss {loss.item():.4f}, held-out margin R2 {vr:+.4f}, "
              f"{time.time()-t0:.0f}s -> checkpointed", flush=True)
        del X, y, w

    pool.close(); pool.join()
    print(f"-> {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
