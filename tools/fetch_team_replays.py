"""Fetch a leaderboard team's recent replays by walking the episode graph.

    PYTHONPATH=. uv run python tools/fetch_team_replays.py --team Boey \
        --team-id 16915014 --n 12 --out replays/boey

Walks ListEpisodes from --seed-sub (any known top submission), expanding the
highest-scored submissions seen, until the team's public-leaderboard
submission id turns up in a response's `teams` list; then lists that
submission's episodes, keeps the most recent finished ones, and downloads
each with `kaggle competitions replay <id>` (~30 MB each). Existing files are
kept; the run is resumable.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def list_episodes(submission_id):
    body = json.dumps({"submissionId": int(submission_id)}).encode()
    req = urllib.request.Request(LIST_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def find_submission(team_id, seed_sub, max_hops=30, sleep=0.4):
    """-> (submission_id, team_name) for team_id, walking from seed_sub."""
    seen = set()
    frontier = [(0.0, int(seed_sub))]
    while frontier and len(seen) < max_hops:
        frontier.sort(reverse=True)
        _, sub = frontier.pop(0)
        if sub in seen:
            continue
        seen.add(sub)
        try:
            d = list_episodes(sub)
        except Exception as e:  # noqa: BLE001
            print(f"  list_episodes({sub}) failed: {e}", file=sys.stderr)
            continue
        for t in d.get("teams", []):
            if int(t.get("id", -1)) == int(team_id) and t.get("publicLeaderboardSubmissionId"):
                return int(t["publicLeaderboardSubmissionId"]), t.get("teamName")
        # expand: the best-scored submissions in these episodes
        for e in d.get("episodes", []):
            for a in e.get("agents", []):
                s = a.get("submissionId")
                if s and s not in seen:
                    frontier.append((float(a.get("updatedScore") or 0), int(s)))
        time.sleep(sleep)
    return None, None


def fetch_replay(eid, out_dir):
    subprocess.run(["uv", "run", "kaggle", "competitions", "replay", str(eid), "-p", out_dir],
                   capture_output=True, text=True, cwd=REPO)
    for name in (f"episode-{eid}-replay.json", f"{eid}.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True, help="team name as shown in replays' TeamNames")
    ap.add_argument("--team-id", type=int, required=True, help="leaderboard teamId")
    ap.add_argument("--seed-sub", type=int, default=56237601)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--submission", type=int, default=None, help="skip the walk: the team's submission id")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sub = args.submission
    if sub is None:
        sub, name = find_submission(args.team_id, args.seed_sub)
        print(f"{args.team}: submission {sub} ({name})")
        if sub is None:
            raise SystemExit("team not found in the episode graph; pass --submission")
    d = list_episodes(sub)
    # public ladder games only: the one EPISODE_TYPE_VALIDATION game is the
    # submission playing itself
    eps = [e for e in d.get("episodes", [])
           if str(e.get("state", "")) == "COMPLETED" and str(e.get("type", "")) == "EPISODE_TYPE_PUBLIC"]
    eps.sort(key=lambda e: e.get("endTime") or e.get("createTime") or "", reverse=True)
    print(f"{len(eps)} finished episodes for submission {sub}")
    got = 0
    for e in eps:
        if got >= args.n:
            break
        eid = e["id"]
        path = os.path.join(args.out, f"episode-{eid}-replay.json")
        if os.path.exists(path):
            got += 1
            continue
        p = fetch_replay(eid, args.out)
        if not p:
            print(f"  {eid}: no file", flush=True)
            continue
        try:
            with open(p) as fh:
                r = json.load(fh)
            names = (r.get("info") or {}).get("TeamNames")
            ok = args.team in (names or [])
        except Exception as ex:  # noqa: BLE001
            ok, names = False, str(ex)[:80]
        if not ok:
            print(f"  {eid}: team not in {names}; removed", flush=True)
            os.remove(p)
            continue
        got += 1
        print(f"  {eid}: {names} {os.path.getsize(p) / 1e6:.0f} MB", flush=True)
        time.sleep(0.3)
    print(f"{got} replays in {args.out}")


if __name__ == "__main__":
    main()
