"""Powered, band-stratified, PAIRED evaluation of rl weights on held-out sessions.

WHY. The in-training SESSION-WR was 3 games a bin, 18 in all: 0.06 was one
win and 0.11 two, and six Stage B arms were ranked on it. Resolving a 0.05
win-rate difference needs on the order of 750 PAIRED games (memory: gate
size). This draws ONE set of held-out sessions and plays every arm on exactly
that set, so the comparison is paired by world and by opponent, and the test
is McNemar on the discordant games -- the project's standing protocol for
win-rate gates (tools/ladder_gate.py). Per band, because the 2500+ band is
the hardest and a rating-floor eval is a top-only eval.

    KAGG_BOARD=1 KAGG_BOARD_ARM=T_crop_103 PYTHONPATH=. uv run python rl/session_eval.py \\
        --arm control=rl/runs/stage_b2/control/eval.json \\
        --arm frozen=rl/runs/stage_b2/frozen/eval.json \\
        --arm trainable=rl/runs/stage_b2/trainable/eval.json \\
        --per-bin 125 --procs 5 --out rl/runs/stage_b2/session_eval.json

Weights files are `rl.common.to_layers` JSON (train.py writes eval.json and
w<it>.json). A file carrying a "board" trunk needs KAGG_BOARD=1 in the
environment before this script imports rl.rollout; it refuses otherwise. The
first --arm is the baseline every other arm is paired against.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Same bins as rl/train.py's curriculum and eval.
EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]


def _label(b):
    lo, hi = EDGES[b]
    return f"{lo}-{hi}" if hi < 99999 else f"{lo}+"


def held_out_entries(idx_path, seed, per_bin):
    """[(rel_path, script_side, bin, rating)] -- rated seats of held-out
    sessions (id % 5 == 0), `per_bin` from each rating bin, seeded."""
    with open(idx_path) as fh:
        idx = json.load(fh)
    bins = [[] for _ in EDGES]
    for v in idx.values():
        p = v.get("path")
        if not p:
            continue
        try:
            sid = int(os.path.basename(p).split("-")[-1][:-5])
        except ValueError:
            continue
        if sid % 5 != 0:
            continue
        for ss, r in enumerate(v.get("scores") or [None, None]):
            if r is None:
                continue
            b = next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)
            bins[b].append((p, ss, b, float(r)))
    rng = random.Random(seed)
    out = []
    for b in bins:
        b.sort()
        rng.shuffle(b)
        out += b[:per_bin]
    return out


def binom_two_sided(k, n):
    """Exact two-sided binomial p at p=0.5: McNemar's exact test."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def reuse_results(prior_rows, entries, label="reused arm"):
    """Accept a previous run's per-game rows as an arm, if and only if they
    were played on exactly this draw: same sessions, same seats, same order.
    Temperature-0 play is deterministic (B2's control replayed 47/785 with
    every game identical), so a matching prior run IS the replay."""
    if len(prior_rows) != len(entries):
        raise SystemExit(f"{label}: {len(prior_rows)} prior games vs {len(entries)} "
                         f"drawn -- different --per-bin or corpus; refusing")
    for i, (row, (p, ss, b, r)) in enumerate(zip(prior_rows, entries)):
        if row.get("path") != p or row.get("script_side") != ss:
            raise SystemExit(f"{label}: game {i} is {row.get('path')} seat "
                             f"{row.get('script_side')}, drawn {p} seat {ss} -- "
                             f"not the same draw (seed?); refusing")
    return prior_rows


def _wr(rows):
    if not rows:
        return float("nan"), float("nan")
    p = sum(r["win"] for r in rows) / len(rows)
    return p, math.sqrt(max(p * (1 - p), 0.0) / len(rows))


def paired(base, arm):
    """McNemar on discordant games, plus the paired delta and its se."""
    b = sum(1 for x, y in zip(base, arm) if x["win"] < 0.75 <= y["win"])   # arm wins, base does not
    c = sum(1 for x, y in zip(base, arm) if y["win"] < 0.75 <= x["win"])   # base wins, arm does not
    d = [y["win"] - x["win"] for x, y in zip(base, arm)]
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    var = sum((v - mean) ** 2 for v in d) / (n - 1) if n > 1 else float("nan")
    return {"b_arm_only": b, "c_base_only": c, "delta": mean,
            "se": math.sqrt(var / n) if n > 1 else float("nan"),
            "p_mcnemar": binom_two_sided(b, b + c), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path/to/weights.json; the first is the baseline")
    ap.add_argument("--per-bin", type=int, default=125)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON of every game, for re-analysis")
    ap.add_argument("--reuse", action="append", default=[],
                    help="name=prior_eval.json[:arm] -- take that arm's per-game "
                         "results from a previous --out file instead of replaying "
                         "(saves ~19 min an arm). Checked game by game against "
                         "this run's draw and refused on any mismatch")
    ap.add_argument("--baseline", default=None,
                    help="arm every other arm is paired against (a played or "
                         "reused name); default the first --arm")
    args = ap.parse_args()

    from rl import rollout

    arms = []          # (name, path, has_board, is_bundle)
    for a in args.arm:
        name, _, path = a.partition("=")
        if not path:
            raise SystemExit(f"--arm wants name=path or name=bundle:path, got {a!r}")
        if path.startswith("bundle:"):
            # A frozen main.py plays OUR side -- the deployment comparison.
            bpath = os.path.abspath(path[len("bundle:"):])
            if not os.path.exists(bpath):
                raise SystemExit(f"{name}: no bundle at {bpath}")
            arms.append((name, bpath, False, True))
            continue
        if path.startswith("dev:"):
            # The DEV TREE's scripted agent plays our side: dev:champion, or
            # dev:knobs:<rules.json> (a BANDIT_RULES dict lifted from a bundle).
            parts = path.split(":", 2)
            if parts[1] == "champion":
                spec = {"kind": "champion"}
            elif parts[1] in ("knobs", "bandit") and len(parts) == 3:
                with open(parts[2]) as fh:
                    spec = {"kind": parts[1], "rules": json.load(fh)}
            elif parts[1] == "execnet" and len(parts) == 3:
                # dev:execnet:<weights.pt>[:<planner>[:<conf>[:<mode>]]] -- the
                # learned executor (rl/execnet). <planner> is a rules JSON (the
                # knob-bandit plan, as before) or a to_layers weights JSON (a
                # neural day-planner: the HYBRID arm). Told apart by content.
                from rl.execnet.policy import parse_arm_fields
                spec = parse_arm_fields(parts[2].split(":"))
                if spec.get("planner") == "weights" and not rollout._BOARD:
                    with open(spec["planner_weights"]) as fh:
                        if json.load(fh)["policy"].get("board") is not None:
                            raise SystemExit(
                                f"{name}: the planner's weights carry a board trunk "
                                f"but KAGG_BOARD is not set -- export KAGG_BOARD=1")
            else:
                raise SystemExit(f"{name}: dev arm wants dev:champion, dev:knobs:<rules.json> "
                                 f"or dev:execnet:<weights.pt>[:<rules.json|planner-weights.json>[:<conf>[:<mode>]]]")
            arms.append((name, spec, False, "dev"))
            continue
        with open(path) as fh:
            pol = json.load(fh)["policy"]
        has_board = pol.get("board") is not None
        if has_board and not rollout._BOARD:
            raise SystemExit(f"{name}: weights carry a board trunk but KAGG_BOARD "
                             f"is not set -- export KAGG_BOARD=1 first")
        arms.append((name, os.path.abspath(path), has_board, False))

    entries = held_out_entries(os.path.join(REPO, "rl", "datasets", "session_index.json"),
                               args.seed, args.per_bin)
    per_bin = [sum(1 for e in entries if e[2] == b) for b in range(len(EDGES))]
    print(f"{len(entries)} held-out games per arm; per bin "
          + " ".join(f"{_label(b)}:{n}" for b, n in enumerate(per_bin)))
    print(f"arms: " + ", ".join(
        f"{n}{' (board)' if hb else ''}{' (bundle)' if ib is True else ''}{' (dev tree)' if ib == 'dev' else ''}"
        for n, _, hb, ib in arms))

    results = {}
    reused = {}
    for spec in args.reuse:
        name, _, rest = spec.partition("=")
        path, _, arm = rest.partition(":")
        if not path:
            raise SystemExit(f"--reuse wants name=path.json[:arm], got {spec!r}")
        with open(path) as fh:
            prior = json.load(fh)
        rows = prior["results"].get(arm or name)
        if rows is None:
            raise SystemExit(f"--reuse {spec}: no arm {arm or name!r} in {path}")
        results[name] = reuse_results(rows, entries, label=f"--reuse {name}")
        reused[name] = f"{path}:{arm or name}"
        p_, se = _wr(rows)
        print(f"  {name:<12} win {p_:.3f} +/- {se:.3f}   (reused from {reused[name]})")
    t0 = time.time()
    with Pool(args.procs) as pool:
        for name, path, _, is_bundle in arms:
            jobs = [{"weights_path": None if is_bundle else path,
                     "learner_bundle": path if is_bundle is True else None,
                     "learner_policy": path if is_bundle == "dev" else None,
                     "temperature": 0.0, "seed": 0,
                     "side": 1 - ss, "opponent": ("session", os.path.join(REPO, p), ss),
                     "mode": "rl"} for p, ss, b, r in entries]
            out = list(pool.imap(rollout.run_episode, jobs))          # ordered: paired
            rows = []
            for ep, (p, ss, b, r) in zip(out, entries):
                rows.append({"win": ep["win"], "money": ep["final_money"],
                             "opp": ep["opp_money"], "bin": b, "rating": r,
                             "path": p, "script_side": ss,
                             "bad": any(s != "DONE" for s in ep["statuses"])})
            results[name] = rows
            p_, se = _wr(rows)
            print(f"  {name:<12} win {p_:.3f} +/- {se:.3f}   money "
                  f"{sum(x['money'] for x in rows)/len(rows):>9,.0f}   bad "
                  f"{sum(x['bad'] for x in rows)}   ({time.time()-t0:.0f}s)", flush=True)

    # Baseline: --baseline, else the first --arm. Played arms print first.
    results = {**{n: results[n] for n, _, _, _ in arms}, **{n: results[n] for n in reused}}
    base_name = args.baseline or arms[0][0]
    if base_name not in results:
        raise SystemExit(f"--baseline {base_name!r} is not a played or reused arm "
                         f"({list(results)})")
    print(f"\n{'arm':<12}{'win':>8}{'se':>7}{'money':>10}  " +
          "".join(f"{_label(b):>11}" for b in range(len(EDGES))))
    for name, rows in results.items():
        p_, se = _wr(rows)
        cells = "".join(f"{_wr([x for x in rows if x['bin'] == b])[0]:>11.3f}"
                        for b in range(len(EDGES)))
        print(f"{name:<12}{p_:>8.3f}{se:>7.3f}{sum(x['money'] for x in rows)/len(rows):>10,.0f}  {cells}")

    if len(results) > 1:
        print(f"\nPAIRED vs {base_name} (same sessions, same seats), McNemar on discordant games")
        print(f"{'arm':<12}{'delta':>8}{'se':>7}{'z':>6}{'arm-only':>10}{'base-only':>11}{'p':>8}  "
              + "".join(f"{_label(b):>11}" for b in range(len(EDGES))))
        for name, rows in results.items():
            if name == base_name:
                continue
            pr = paired(results[base_name], rows)
            z = pr["delta"] / pr["se"] if pr["se"] and pr["se"] > 0 else float("nan")
            cells = ""
            for b in range(len(EDGES)):
                pb = paired([x for x in results[base_name] if x["bin"] == b],
                            [x for x in rows if x["bin"] == b])
                cells += f"{pb['delta']:>+11.3f}"
            print(f"{name:<12}{pr['delta']:>+8.3f}{pr['se']:>7.3f}{z:>6.2f}"
                  f"{pr['b_arm_only']:>10}{pr['c_base_only']:>11}{pr['p_mcnemar']:>8.3f}  {cells}")
        print("delta = arm win rate minus baseline's on identical games; the per-band "
              "columns are the same delta within each rating bin.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"seed": args.seed, "per_bin": args.per_bin,
                       "arms": {n: (p if isinstance(p, str) else p.get("kind")) for n, p, _, _ in arms},
                       "reused": reused,
                       "results": results}, fh)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
