"""Reduce raw Kaggle episode replays (~32 MB each) to SESSION records
(~20 KB): seed, both sides' aligned action scripts, finals, shops, teams and
per-seat ratings. Runs inside a Kaggle kernel over mounted daily datasets
(kaggle/kaggriculture-episodes-YYYY-MM-DD) or locally over downloaded files.

    python tools/extract_episodes.py --inputs-glob '/kaggle/input/**/kaggriculture-episodes-20*' \
        --ratings rl/datasets/ratings/agents.csv --out /kaggle/working/out --procs 4

Output: one `sessions_<dataset-slug>.jsonl.gz` per input directory (one
session per line, the format rl/datasets/sessions/session-<id>.json uses) plus
`extract_stats.json`. Filters: engine module_version (default 1.32.7), both
finals >= --min-final (drops crashed/idle seats).
"""

import argparse
import csv
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

MIN_FINAL = 15000.0


def load_ratings(path):
    """agents.csv (georgymamarin/kaggriculture-episodes): episode_id, agent_index,
    submission_id, team_id, final_bank, rating_after."""
    ratings = {}
    if not path or not os.path.exists(path):
        return ratings
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ratings[(int(r["episode_id"]), int(r["agent_index"]))] = (
                    float(r["rating_after"]) if r.get("rating_after") else None,
                    int(r["submission_id"]) if r.get("submission_id") else None,
                )
            except (ValueError, KeyError):
                continue
    return ratings


def load_leaderboard(path):
    """Kaggle leaderboard CSV (TeamName, Score): current team rating, the
    fallback join when the community crawl has no row for an episode (the
    official daily datasets sample episodes the crawl never saw)."""
    lb = {}
    if not path or not os.path.exists(path):
        return lb
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = r.get("TeamName") or r.get("teamName")
            score = r.get("Score") or r.get("score")
            if name and score:
                try:
                    lb[name] = float(score)
                except ValueError:
                    pass
    return lb


def to_session(path, module_version, min_final):
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception as e:  # corrupt file: skip, report
        return None, f"load-error {e.__class__.__name__}"
    if module_version and str(r.get("module_version")) != module_version:
        return None, f"version {r.get('module_version')}"
    info = r.get("info") or {}
    seed = info.get("seed")
    steps = r.get("steps") or []
    if seed is None or len(steps) < 2:
        return None, "no-seed"
    finals = [(s.get("reward") or 0) for s in steps[-1]]
    if min(finals) < min_final:
        return None, "low-final"
    # steps[t].action produced state t; replay steps[t+1].action at local step t.
    actions = [[], []]
    for t in range(len(steps) - 1):
        for side in (0, 1):
            actions[side].append(steps[t + 1][side].get("action") or {})
    shops = ((steps[-1][0].get("observation") or {}).get("town") or {}).get("unlocked_shops", [])
    return {
        "episode": int(info.get("EpisodeId") or os.path.basename(path).split(".")[0]),
        "seed": int(seed),
        "teams": info.get("TeamNames"),
        "finals": finals,
        "shops": shops,
        "actions": actions,
        "module_version": r.get("module_version"),
    }, None


def _job(args):
    path, module_version, min_final = args
    s, why = to_session(path, module_version, min_final)
    return path, s, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-glob", required=True, help="directories of <episode>.json files")
    ap.add_argument("--ratings", default=None, help="agents.csv for per-seat ratings (rating at game time)")
    ap.add_argument("--leaderboard", default=None, help="leaderboard CSV: current team rating, fallback by team name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--module-version", default="1.32.7")
    ap.add_argument("--min-final", type=float, default=MIN_FINAL)
    ap.add_argument("--limit", type=int, default=None, help="episodes per input (smoke)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ratings = load_ratings(args.ratings)
    leaderboard = load_leaderboard(args.leaderboard)
    print(f"ratings loaded for {len(ratings)} seats; leaderboard teams {len(leaderboard)}", flush=True)
    dirs = sorted(d for d in glob.glob(args.inputs_glob, recursive=True) if os.path.isdir(d))
    stats = {}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if args.limit:
            files = files[: args.limit]
        slug = os.path.basename(d.rstrip("/"))
        out_path = os.path.join(args.out, f"sessions_{slug}.jsonl.gz")
        t0 = time.time()
        kept = 0
        reasons = {}
        with Pool(args.procs) as pool, gzip.open(out_path, "wt") as f:
            for path, s, why in pool.imap_unordered(_job, [(p, args.module_version, args.min_final) for p in files], chunksize=2):
                if s is None:
                    key = (why or "?").split()[0]
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                s["scores"] = [(ratings.get((s["episode"], i)) or (None, None))[0] for i in (0, 1)]
                s["submissions"] = [(ratings.get((s["episode"], i)) or (None, None))[1] for i in (0, 1)]
                s["score_source"] = ["crawl" if v is not None else None for v in s["scores"]]
                teams = s.get("teams") or [None, None]
                for i in (0, 1):
                    if s["scores"][i] is None and teams[i] in leaderboard:
                        s["scores"][i] = leaderboard[teams[i]]
                        s["score_source"][i] = "leaderboard"
                f.write(json.dumps(s) + "\n")
                kept += 1
        rated = kept  # (ratings may be None for episodes the community crawl missed)
        stats[slug] = {"files": len(files), "kept": kept, "skipped": reasons, "seconds": round(time.time() - t0)}
        print(f"{slug}: {kept}/{len(files)} sessions in {time.time() - t0:.0f}s skipped={reasons}", flush=True)
    with open(os.path.join(args.out, "extract_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
