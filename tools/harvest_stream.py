"""Streaming leaderboard harvest at scale: crawl the top tier's episodes,
extract BOTH sides of every game into outcome-labeled training samples plus a
per-side stats row, then DELETE the replay (~30MB each) so thousands of games
fit in bounded disk.

Every sample carries {win, margin, money, opp_money} so training can weight
winners by profit margin and keep losers as contrast, and the stats file
answers the dataset questions directly (win rate, winning/losing margins,
which plans win).

Usage:
    PYTHONPATH=. uv run python tools/harvest_stream.py \
        --seed-subs 56082749,55714252 --min-score 2200 --max-subs 40 \
        --per-sub 30 --out rl/datasets/field
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.replay_bc import extract_side  # noqa: E402

LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def list_episodes(submission_id):
    body = json.dumps({"submissionId": submission_id}).encode()
    req = urllib.request.Request(
        LIST_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("episodes", [])


def process_episode(eid, agents_meta, tmp_dir, samples_f, stats_f, min_reward):
    path = os.path.join(tmp_dir, f"episode-{eid}-replay.json")
    r = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "replay", str(eid), "-p", tmp_dir],
        capture_output=True, text=True,
    )
    if not os.path.exists(path):
        return 0
    try:
        replay = json.load(open(path))
        steps = replay["steps"]
        rewards = [s.get("reward") or 0 for s in steps[-1]]
        n = 0
        # Emit a session file too (seed + both action scripts): the raw
        # replay is deleted below, but sessions power replay training.
        seed = (replay.get("info") or {}).get("seed")
        if seed is not None and min(rewards) >= 15000:
            sess_dir = "rl/datasets/sessions"
            os.makedirs(sess_dir, exist_ok=True)
            sp = os.path.join(sess_dir, f"session-{eid}.json")
            if not os.path.exists(sp):
                actions = [[], []]
                for t in range(len(steps) - 1):
                    for side in (0, 1):
                        actions[side].append(steps[t + 1][side].get("action") or {})
                with open(sp, "w") as f:
                    json.dump({
                        "episode": int(eid), "seed": int(seed),
                        "teams": (replay.get("info") or {}).get("TeamNames"),
                        "scores": [a.get("initialScore") for a in agents_meta],
                        "finals": rewards,
                        "shops": steps[-1][0]["observation"]["town"]["unlocked_shops"],
                        "actions": actions,
                    }, f)
        for side in (0, 1):
            my_r, opp_r = rewards[side], rewards[1 - side]
            win = 1 if my_r > opp_r else 0
            meta = agents_meta[side] if side < len(agents_meta) else {}
            stats_f.write(json.dumps({
                "episode": eid, "side": side, "money": my_r, "opp_money": opp_r,
                "win": win, "margin": my_r - opp_r,
                "score": meta.get("initialScore"), "sub": meta.get("submissionId"),
            }) + "\n")
            samples = extract_side(replay, side, min_reward)
            for s in samples:
                s["win"] = win
                s["margin"] = my_r - opp_r
                s["money"] = my_r
                samples_f.write(json.dumps(s) + "\n")
                n += 1
        return n
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-subs", required=True)
    ap.add_argument("--min-score", type=float, default=2200.0)
    ap.add_argument("--max-score", type=float, default=99999.0, help="frontier band upper bound")
    ap.add_argument("--max-subs", type=int, default=40)
    ap.add_argument("--per-sub", type=int, default=30)
    ap.add_argument("--min-reward", type=float, default=15000.0)
    ap.add_argument("--out", default="rl/datasets/field")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp_dir = os.path.join(args.out, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    seen_path = os.path.join(args.out, "seen.txt")
    seen = set()
    if os.path.exists(seen_path):
        seen = {int(l) for l in open(seen_path) if l.strip()}

    samples_f = open(os.path.join(args.out, "samples.jsonl"), "a")
    stats_f = open(os.path.join(args.out, "stats.jsonl"), "a")
    seen_f = open(seen_path, "a")

    frontier = [int(s) for s in args.seed_subs.split(",")]
    visited = set()
    total_samples = games = 0

    while frontier and len(visited) < args.max_subs:
        sub = frontier.pop(0)
        if sub in visited:
            continue
        visited.add(sub)
        try:
            episodes = [e for e in list_episodes(sub) if e.get("state") == "COMPLETED"]
        except Exception as e:
            print(f"sub {sub}: listing failed ({e})", flush=True)
            continue
        episodes.sort(key=lambda e: e.get("endTime", ""), reverse=True)
        taken = 0
        for ep in episodes:
            for a in ep.get("agents", []):
                sid, score = a.get("submissionId"), a.get("initialScore") or 0
                if sid and sid not in visited and sid not in frontier and args.min_score <= score < args.max_score:
                    frontier.append(sid)
            if taken >= args.per_sub or ep["id"] in seen:
                continue
            n = process_episode(
                ep["id"], ep.get("agents", []), tmp_dir, samples_f, stats_f, args.min_reward
            )
            seen.add(ep["id"])
            seen_f.write(f"{ep['id']}\n")
            for f in (samples_f, stats_f, seen_f):
                f.flush()
            if n:
                games += 1
                total_samples += n
                taken += 1
            time.sleep(0.5)
        print(f"sub {sub}: +{taken} games (total {games} games, {total_samples} samples, "
              f"frontier {len(frontier)})", flush=True)

    print(f"\nDONE: {games} games, {total_samples} outcome-labeled samples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
