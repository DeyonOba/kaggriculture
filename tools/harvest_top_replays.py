"""Climb the leaderboard's episode graph and harvest SESSIONS from the top tier.

Why this exists: the official daily-episode index (tools/episodes_pipeline.py)
lags -- on 2026-09-15 its newest day was 2026-09-13, already in the ledger -- so
it cannot deliver seats from the current frontier. Meanwhile the field top has
moved to 3,222 while the tape cluster we actually ship (tape-sel) is a 3,029
lineage. This walks the ListEpisodes graph from a seed submission, always
expanding the HIGHEST-rated submission it has seen, and turns the episodes it
finds into sessions.

Replays are ~30 MB each and are deleted immediately after extraction, so disk
stays bounded no matter how long it runs. `seen.txt` makes runs resumable.

Per-seat ratings come from the crawl's own `initialScore` (the rating at game
time), which is strictly better than the leaderboard-by-team-name fallback
tools/extract_episodes.py has to use inside the kernel.

    PYTHONPATH=. uv run python tools/harvest_top_replays.py \
        --seed-subs 56237601 --min-score 2600 --max-episodes 400

Then rebuild the tape library over the enlarged corpus and re-run the panel:
    PYTHONPATH=. uv run python tools/tape_agent.py build
    PYTHONPATH=. uv run python tools/data.py push -m "top-tier harvest"
"""

import argparse
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.harvest_stream import list_episodes            # noqa: E402
from tools.extract_episodes import to_session             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS_DIR = os.path.join(REPO, "rl", "datasets", "sessions")
STATE_DIR = os.path.join(REPO, "rl", "datasets", "harvest_top")


def fetch_replay(eid, tmp_dir):
    r = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "replay", str(eid), "-p", tmp_dir],
        capture_output=True, text=True, cwd=REPO,
    )
    for name in (f"episode-{eid}-replay.json", f"{eid}.json"):
        p = os.path.join(tmp_dir, name)
        if os.path.exists(p):
            return p
    hits = [f for f in os.listdir(tmp_dir) if str(eid) in f and f.endswith(".json")]
    if hits:
        return os.path.join(tmp_dir, hits[0])
    return None if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-subs", required=True,
                    help="comma-separated submission ids to start the climb from")
    ap.add_argument("--min-score", type=float, default=2600.0,
                    help="only harvest episodes with a seat at or above this rating")
    ap.add_argument("--expand-min", type=float, default=2000.0,
                    help="only follow submissions rated at or above this")
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--max-subs", type=int, default=250)
    ap.add_argument("--module-version", default="1.32.7")
    ap.add_argument("--min-final", type=float, default=15000.0)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(SESS_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    seen_path = os.path.join(STATE_DIR, "seen.txt")
    seen = {int(l) for l in open(seen_path)} if os.path.exists(seen_path) else set()
    seen_f = open(seen_path, "a")
    log_f = open(os.path.join(STATE_DIR, "harvest_log.jsonl"), "a")

    # Max-heap on rating: always expand the best submission we know about, so
    # the crawl climbs toward the frontier instead of wandering the middle.
    frontier, queued, visited = [], set(), set()
    for s in args.seed_subs.split(","):
        heapq.heappush(frontier, (-99999.0, int(s)))
        queued.add(int(s))

    tmp_dir = tempfile.mkdtemp(prefix="toprep-")
    kept = skipped = downloaded = have = 0
    best_seen = 0.0
    try:
        while frontier and len(visited) < args.max_subs and kept < args.max_episodes:
            negscore, sub = heapq.heappop(frontier)
            if sub in visited:
                continue
            visited.add(sub)
            try:
                eps = [e for e in list_episodes(sub) if e.get("state") == "COMPLETED"]
            except Exception as e:
                print(f"sub {sub}: listing failed ({e})", flush=True)
                continue

            for ep in eps:
                agents = ep.get("agents", [])
                scores = [a.get("initialScore") for a in agents]
                for a in agents:
                    sid, sc = a.get("submissionId"), a.get("initialScore") or 0
                    best_seen = max(best_seen, sc)
                    if sid and sid not in visited and sid not in queued and sc >= args.expand_min:
                        heapq.heappush(frontier, (-sc, sid))
                        queued.add(sid)

                if ep["id"] in seen or kept >= args.max_episodes:
                    continue
                if not any((s or 0) >= args.min_score for s in scores):
                    continue
                # The corpus already holds ~20.8k sessions from the official
                # pipeline, and the frontier's episodes are exactly the ones it
                # covers best -- the first run re-fetched 132 of 209 episodes it
                # already had. A 30 MB download to rewrite an identical file is
                # the whole cost of this tool, so skip on disk.
                if os.path.exists(os.path.join(SESS_DIR, f"session-{ep['id']}.json")):
                    have += 1
                    seen.add(ep["id"])
                    seen_f.write(f"{ep['id']}\n")
                    continue

                path = fetch_replay(ep["id"], tmp_dir)
                downloaded += 1
                seen.add(ep["id"])
                seen_f.write(f"{ep['id']}\n")
                seen_f.flush()
                if not path:
                    skipped += 1
                    continue
                s, why = to_session(path, args.module_version, args.min_final)
                os.remove(path)
                if s is None:
                    skipped += 1
                    log_f.write(json.dumps({"episode": ep["id"], "skip": why}) + "\n")
                    continue
                s["scores"] = [float(x) if x else None for x in scores]
                s["submissions"] = [a.get("submissionId") for a in agents]
                s["score_source"] = ["crawl" if x else None for x in scores]
                out = os.path.join(SESS_DIR, f"session-{s['episode']}.json")
                with open(out, "w") as f:
                    json.dump(s, f)
                kept += 1
                log_f.write(json.dumps({"episode": ep["id"], "scores": s["scores"],
                                        "finals": s["finals"]}) + "\n")
                log_f.flush()
                time.sleep(args.sleep)

            print(f"sub {sub} (rating {-negscore:,.0f}): kept {kept}, skipped {skipped}, "
                  f"downloaded {downloaded}, had {have}, frontier {len(frontier)}, best seen "
                  f"{best_seen:,.0f}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nDONE: {kept} new sessions, {have} already held, {skipped} skipped, "
              f"{downloaded} replays fetched, {len(visited)} submissions crawled, "
              f"best rating seen {best_seen:,.0f}")


if __name__ == "__main__":
    main()
