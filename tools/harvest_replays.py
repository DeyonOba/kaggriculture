"""Harvest top-agent episodes: snowball-crawl the leaderboard's upper tier via
the episode-listing endpoint, then download replays for study/imitation.

Each episode lists both agents' submissionId and rating, so starting from one
known top submission discovers the whole tier.

Usage:
    uv run python tools/harvest_replays.py --seed-subs 55829779 \
        --min-score 2600 --max-subs 8 --per-sub 12 --out replays/top
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def list_episodes(submission_id):
    body = json.dumps({"submissionId": submission_id}).encode()
    req = urllib.request.Request(
        LIST_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("episodes", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-subs", required=True, help="comma-separated known top submission ids")
    ap.add_argument("--min-score", type=float, default=2600.0)
    ap.add_argument("--max-subs", type=int, default=8)
    ap.add_argument("--per-sub", type=int, default=12)
    ap.add_argument("--out", default="replays/top")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    seen_eps = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            seen_eps = {json.loads(line)["episode_id"] for line in f if line.strip()}

    frontier = [int(s) for s in args.seed_subs.split(",")]
    visited, sub_scores = set(), {}
    manifest = open(manifest_path, "a")

    while frontier and len(visited) < args.max_subs:
        sub = frontier.pop(0)
        if sub in visited:
            continue
        visited.add(sub)
        try:
            episodes = list_episodes(sub)
        except Exception as e:
            print(f"sub {sub}: listing failed ({e})")
            continue
        episodes = [e for e in episodes if e.get("state") == "COMPLETED"]
        episodes.sort(key=lambda e: e.get("endTime", ""), reverse=True)
        print(f"sub {sub}: {len(episodes)} completed episodes")

        taken = 0
        for ep in episodes:
            agents = ep.get("agents", [])
            for a in agents:
                sid = a.get("submissionId")
                score = a.get("initialScore") or 0
                if sid:
                    sub_scores[sid] = max(sub_scores.get(sid, 0), score)
                if (
                    sid
                    and sid not in visited
                    and sid not in frontier
                    and score >= args.min_score
                ):
                    frontier.append(sid)
            if taken >= args.per_sub:
                continue
            eid = ep["id"]
            if eid in seen_eps:
                continue
            path = os.path.join(args.out, f"episode-{eid}-replay.json")
            r = subprocess.run(
                ["uv", "run", "kaggle", "competitions", "replay", str(eid), "-p", args.out],
                capture_output=True,
                text=True,
            )
            if os.path.exists(path):
                rec = {
                    "episode_id": eid,
                    "agents": [
                        {"submissionId": a.get("submissionId"),
                         "teamId": a.get("teamId"),
                         "score": a.get("initialScore"),
                         "reward": a.get("reward")}
                        for a in agents
                    ],
                }
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                seen_eps.add(eid)
                taken += 1
                print(f"  downloaded ep {eid} (rewards {[a.get('reward') for a in agents]})")
            else:
                print(f"  ep {eid} download failed: {r.stderr.strip()[:120]}")
            time.sleep(1.0)

    manifest.close()
    top = sorted(sub_scores.items(), key=lambda kv: -kv[1])[:12]
    print("\ndiscovered submissions by score:")
    for sid, sc in top:
        print(f"  {sid}: {sc:.0f}")


if __name__ == "__main__":
    main()
