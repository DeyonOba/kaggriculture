"""Backfill per-seat ratings AT GAME TIME over the whole session corpus.

The problem: 36,796 of 42,550 seats (86%) carry a rating obtained by matching
`TeamNames` against a leaderboard SNAPSHOT, and 4,508 carry no rating at all.
Only the 1,246 seats from tools/harvest_top_replays.py have the real thing --
`initialScore`, the rating each seat held when the game was played.

Measured against those 1,246 seats, the leaderboard-by-name label is unbiased
(mean -20, median +3) but noisy, sd 251: 9% of seats are off by more than 100
points and **12% land on the wrong side of the `--min-rating 3000` cut** that
selects the tape library. That measurement is also taken on the friendliest
possible sample -- recently harvested games, where "rating today" and "rating at
game time" nearly coincide -- so the July/August corpus is likely worse.

This matters twice over: the tape library is SELECTED by rating, and the rating
ladder (rl/session_bench.py) -- the instrument every agent is judged on -- BINS
opponents by rating. Both inherit the error.

The fix is cheap because `ListEpisodes` is unauthenticated metadata: one call
returns ~209 episode records, each carrying both seats' `initialScore`. No 30 MB
replay download is involved. It is however SLOW (~11.7 s/call measured), so
covering the corpus is hours of pure waiting -- a Kaggle kernel job, not a
laptop job (`tools/kaggle_job/job_ratings.json`).

    # on Kaggle (12h budget), or locally:
    PYTHONPATH=. uv run python tools/backfill_ratings.py crawl \
        --out rl/datasets/ratings/episode_ratings.json --budget-s 39000
    # then, locally:
    PYTHONPATH=. uv run python tools/backfill_ratings.py apply \
        --map rl/datasets/ratings/episode_ratings.json

`crawl` checkpoints continuously and resumes from its own output, so a killed
run loses nothing. `apply` is a dry run unless given --write.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.harvest_stream import list_episodes            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS_DIR = os.path.join(REPO, "rl", "datasets", "sessions")
DEFAULT_MAP = os.path.join(REPO, "rl", "datasets", "ratings", "episode_ratings.json")


def list_episodes_backoff(sub, tries=6, base=5.0, cap=300.0, state=None):
    """ListEpisodes with exponential backoff on HTTP 429.

    The endpoint rate-limits: a 254-submission crawl on 2026-09-15 got
    `HTTP Error 429` on nearly every call and salvaged 7 sessions. Without
    backoff the caller just burns its frontier at full speed collecting
    nothing, which on a 12h kernel means 12h of nothing. `state` carries a
    running inter-call delay so a run that meets sustained 429s slows down and
    STAYS slowed rather than re-probing the limit on every call.
    """
    state = state if state is not None else {}
    delay = state.get("delay", 0.0)
    for attempt in range(tries):
        if delay:
            time.sleep(delay)
        try:
            eps = list_episodes(sub)
            state["delay"] = max(0.0, delay * 0.9)   # ease off slowly on success
            state["consecutive_429"] = 0
            return eps
        except Exception as e:
            is_429 = "429" in str(e)
            if not is_429 and attempt >= 1:
                raise
            state["consecutive_429"] = state.get("consecutive_429", 0) + (1 if is_429 else 0)
            wait = min(cap, base * (2 ** attempt))
            # Raise the floor so subsequent calls are paced, not just this one.
            state["delay"] = min(cap, max(delay, base * (2 ** min(attempt, 4))))
            delay = state["delay"]
            if attempt == tries - 1:
                raise
            time.sleep(wait)
    return []


def corpus_episode_ids():
    out = set()
    for p in glob.glob(os.path.join(SESS_DIR, "session-*.json")):
        try:
            out.add(int(os.path.basename(p)[len("session-"):-len(".json")]))
        except ValueError:
            continue
    return out


def seed_submissions():
    """Submission ids we already know, from crawl-sourced sessions."""
    subs = set()
    for p in glob.glob(os.path.join(SESS_DIR, "session-*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for s in (d.get("submissions") or []):
            if s:
                subs.add(int(s))
    return subs


def load_state(path):
    if os.path.exists(path):
        d = json.load(open(path))
        return ({int(k): v for k, v in d.get("episodes", {}).items()},
                set(d.get("visited", [])))
    return {}, set()


def save_state(path, episodes, visited):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"episodes": episodes, "visited": sorted(visited)}, f)
    os.replace(tmp, path)


def cmd_crawl(args):
    episodes, visited = load_state(args.out)
    targets = corpus_episode_ids() if args.targets_from_corpus else set()
    print(f"resuming with {len(episodes):,} episode records, {len(visited):,} "
          f"submissions visited; corpus targets {len(targets):,}", flush=True)

    seeds = seed_submissions() if not args.seed_subs else \
        {int(s) for s in args.seed_subs.split(",")}
    frontier = deque(s for s in seeds if s not in visited)
    queued = set(frontier)
    if not frontier:
        # Nothing known: bootstrap from whatever episodes we already hold.
        for sc in list(episodes.values())[:500]:
            for sid in (sc.get("subs") or []):
                if sid and sid not in visited:
                    frontier.append(sid)
                    queued.add(sid)
    print(f"frontier seeded with {len(frontier):,} submissions", flush=True)

    t0 = time.time()
    calls = new_records = failures = 0
    rate = {"delay": args.pace_s, "consecutive_429": 0}
    while frontier:
        if args.budget_s and time.time() - t0 > args.budget_s:
            print("budget reached", flush=True)
            break
        if args.max_calls and calls >= args.max_calls:
            break
        sub = frontier.popleft()
        if sub in visited:
            continue
        try:
            eps = list_episodes_backoff(sub, state=rate)
        except Exception as e:
            # NOT marked visited: a submission we were rate-limited out of was
            # never actually read, and must be retried when the run resumes.
            failures += 1
            print(f"sub {sub}: {e} (delay now {rate.get('delay', 0):.0f}s)", flush=True)
            # Sustained rate limiting: nothing is being collected, so stop and
            # keep the checkpoint rather than burn the remaining budget.
            if failures >= args.max_failures:
                print(f"aborting: {failures} consecutive listing failures", flush=True)
                break
            frontier.append(sub)   # retry later; a 429 is transient, not a verdict
            continue
        visited.add(sub)
        failures = 0
        calls += 1
        for e in eps:
            agents = e.get("agents", [])
            scores = [a.get("initialScore") for a in agents]
            subs = [a.get("submissionId") for a in agents]
            if any(scores) and e["id"] not in episodes:
                episodes[e["id"]] = {"scores": scores, "subs": subs}
                new_records += 1
            for sid in subs:
                if sid and sid not in visited and sid not in queued:
                    frontier.append(sid)
                    queued.add(sid)

        if calls % args.checkpoint_every == 0:
            save_state(args.out, episodes, visited)
            cov = len(set(episodes) & targets) if targets else 0
            rate = (time.time() - t0) / max(calls, 1)
            print(f"{calls:,} calls ({rate:.1f}s each) | {len(episodes):,} records "
                  f"(+{new_records:,}) | corpus covered {cov:,}"
                  f"{f'/{len(targets):,} = {cov/len(targets)*100:.1f}%' if targets else ''}"
                  f" | frontier {len(frontier):,}", flush=True)

    save_state(args.out, episodes, visited)
    cov = len(set(episodes) & targets) if targets else 0
    print(f"\nDONE: {calls:,} calls, {len(episodes):,} episode records, "
          f"{len(visited):,} submissions visited, corpus covered {cov:,}"
          f"{f'/{len(targets):,} = {cov/len(targets)*100:.1f}%' if targets else ''}")


def cmd_apply(args):
    episodes, _ = load_state(args.map)
    print(f"rating map: {len(episodes):,} episodes")
    changed = added = unchanged = missing = 0
    deltas = []
    crossings = 0
    for p in glob.glob(os.path.join(SESS_DIR, "session-*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rec = episodes.get(int(d["episode"]))
        if not rec:
            missing += 1
            continue
        old = list(d.get("scores") or [None, None])
        oldsrc = list(d.get("score_source") or [None, None])
        new = [float(s) if s else None for s in rec["scores"]]
        touched = False
        for i in range(min(len(old), len(new))):
            if new[i] is None:
                continue
            if old[i] is None:
                added += 1
                touched = True
            elif abs(old[i] - new[i]) > 0.05:
                deltas.append(new[i] - old[i])
                if (old[i] >= 3000) != (new[i] >= 3000):
                    crossings += 1
                changed += 1
                touched = True
            else:
                unchanged += 1
        if touched:
            d["scores"] = [new[i] if i < len(new) and new[i] is not None else old[i]
                           for i in range(len(old))]
            d["score_source"] = ["crawl" if (i < len(new) and new[i] is not None)
                                 else oldsrc[i] for i in range(len(old))]
            d["submissions"] = rec["subs"]
            if args.write:
                with open(p, "w") as f:
                    json.dump(d, f)

    import statistics as st
    print(f"seats: {changed:,} corrected, {added:,} newly labelled, "
          f"{unchanged:,} already exact; {missing:,} sessions not in the map")
    if deltas:
        print(f"correction size: mean {st.mean(deltas):+,.0f}, median "
              f"{st.median(deltas):+,.0f}, sd {st.stdev(deltas):,.0f}, "
              f"|d|>300 {sum(1 for x in deltas if abs(x) > 300)/len(deltas)*100:.0f}%")
        print(f"seats that CROSS the --min-rating 3000 tape cut: {crossings:,}")
    print("\n(dry run -- pass --write to modify sessions)" if not args.write
          else "\nsessions updated.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl")
    c.add_argument("--out", default=DEFAULT_MAP)
    c.add_argument("--seed-subs", default=None)
    c.add_argument("--budget-s", type=float, default=39000.0)
    c.add_argument("--max-calls", type=int, default=0)
    c.add_argument("--checkpoint-every", type=int, default=10)
    c.add_argument("--pace-s", type=float, default=1.0,
                   help="baseline delay between calls; backoff raises it on 429")
    c.add_argument("--max-failures", type=int, default=25,
                   help="abort after this many consecutive listing failures")
    c.add_argument("--targets-from-corpus", action="store_true", default=True)
    c.set_defaults(func=cmd_crawl)

    a = sub.add_parser("apply")
    a.add_argument("--map", default=DEFAULT_MAP)
    a.add_argument("--write", action="store_true")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
