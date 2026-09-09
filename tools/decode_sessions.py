"""Decode what rated opponents actually DO, from recorded sessions.

Replays each session on its true seed (fidelity is exact) and logs every
side's per-day state (money, hands, land, planted tiles by crop, animals,
shed) plus per-day action tallies (sells, buys, hires, land) from the
recorded scripts. Optionally seats our bundle against the same scripts on
the same worlds and logs it identically, so profiles compare like for like.

Usage:
    PYTHONPATH=. uv run python tools/decode_sessions.py --per-bin 12 --procs 10 \
        --ours arena/agents/rl-v9/main.py --out rl/datasets/decode
    PYTHONPATH=. uv run python tools/decode_sessions.py report --out rl/datasets/decode
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagent.gamestate import View  # noqa: E402
from rl.session_bench import _load_bundle, make_script_agent  # noqa: E402

SESS_DIR = "rl/datasets/sessions"
EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]
LABELS = ["<600", "600-1000", "1000-1500", "1500-2000", "2000-2500", "2500+"]
DAYS = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 22, 26, 29]


def rating_bin(r):
    return next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)


def _snapshot(view):
    plants = defaultdict(int)
    for _, _, t in view.me.plants():
        plants[t["crop"]] += 1
    animals = defaultdict(int)
    for _, _, t in view.me.animals():
        animals[t["animal"]] += 1
    empties = sum(1 for _ in view.me.empty_structures())
    return {
        "money": view.me.money,
        "hands": len(view.me.units) - 1,
        "land": len(view.me.unlocked_quadrants),
        "plants": dict(plants),
        "animals": dict(animals),
        "empty_structures": empties,
        "shed_total": view.shed_total,
        "shed": dict(view.shed),
    }


def _logger(inner, log):
    """Wrap an agent: record a snapshot at hour 0 of each day, plus market
    prices every turn (for sell-revenue attribution)."""

    def agent(obs):
        view = View(obs)
        # Mid-day: hires are live (hands are dismissed at end of day).
        if view.hour == 12 or view.step == 718:
            log["days"][view.day] = _snapshot(view)
        log["prices"][view.step] = dict(view.market_prices)
        held = dict(view.shed)
        for idx, _, _ in view.me.units:
            for k, n in view.unit_inv(idx).items():
                held[k] = held.get(k, 0) + n
        log["held"][view.step] = held
        log["money"][view.step] = view.me.money
        return inner(obs)

    return agent


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _tally_actions(actions, prices, held=None, money=None):
    held = held or {}
    sells = defaultdict(lambda: [0, 0.0])
    sells_by_day = defaultdict(lambda: defaultdict(float))
    buys = defaultdict(int)
    seeds = defaultdict(int)
    hires = defaultdict(int)
    land_days = []
    feed_bought = 0
    plants_by_day = defaultdict(lambda: defaultdict(int))
    for t, a in enumerate(actions):
        day = t // 24
        for m in a.get("market", []) or []:
            if not m:
                continue
            kind = m[0]
            if kind == "SELL":
                p = prices.get(t, {}).get(m[1], 0)
                # Scripts may issue dump orders (sell 99999); the engine
                # clips to holdings, so clip here too.
                qty = min(_int(m[2]), held.get(t, {}).get(m[1], 0))
                sells[m[1]][0] += qty
                sells[m[1]][1] += qty * p
                sells_by_day[day][m[1]] += qty * p
            elif kind == "BUY_ANIMAL":
                buys[m[1]] += _int(m[2])
            elif kind == "BUY_SEED":
                seeds[m[1]] += _int(m[2])
            elif kind == "HIRE":
                hires[day] += 1
            elif kind == "BUY_LAND":
                land_days.append(day)
            elif kind == "BUY_PRODUCT":
                feed_bought += _int(m[2])
        units = [a.get("farmer") or []] + list(a.get("hands", []) or [])
        for u in units:
            if u and u[0] == "PLANT" and len(u) > 1:
                plants_by_day[day][u[1]] += 1
    cash_in = defaultdict(float)
    cash_out = defaultdict(float)
    if money:
        steps = sorted(money)
        for a_, b_ in zip(steps, steps[1:]):
            d = money[b_] - money[a_]
            if d > 0:
                cash_in[a_ // 24] += d
            else:
                cash_out[a_ // 24] -= d
    return {
        "sells": {k: v for k, v in sells.items()},
        "sells_by_day": {d: dict(v) for d, v in sells_by_day.items()},
        "cash_in": dict(cash_in),
        "cash_out": dict(cash_out),
        "animal_buys": dict(buys),
        "seed_buys": dict(seeds),
        "hires_by_day": dict(hires),
        "land_days": land_days,
        "feed_bought": feed_bought,
        "plants_by_day": {d: dict(v) for d, v in plants_by_day.items()},
    }


def decode_job(job):
    from kaggle_environments import make

    path, ours_path, ours_seat = job
    s = json.load(open(path))
    logs = [{"days": {}, "prices": {}, "held": {}, "money": {}} for _ in (0, 1)]
    agents = []
    for seat in (0, 1):
        if ours_path and seat == ours_seat:
            inner = _load_bundle(ours_path)
        else:
            inner = make_script_agent(s["actions"][seat])
        agents.append(_logger(inner, logs[seat]))
    env = make("kaggriculture", configuration={"seed": s["seed"]}, debug=False)
    env.run(agents)
    finals = [st.reward or 0 for st in env.steps[-1]]
    rows = []
    for seat in (0, 1):
        if ours_path and seat == ours_seat:
            actions = [st[seat].action or {} for st in env.steps[1:]]
            kind, rating = "ours", None
            opp_rating = (s.get("scores") or [None, None])[1 - seat]
        else:
            actions = s["actions"][seat]
            # A script facing OUR agent plays a diverged world; keep it out
            # of the recorded-game profiles.
            kind = "script_vs_ours" if ours_path else "recorded"
            rating = (s.get("scores") or [None, None])[seat]
            opp_rating = (s.get("scores") or [None, None])[1 - seat]
        row = {
            "episode": s["episode"], "seat": seat, "kind": kind, "rating": rating,
            "opp_rating": opp_rating, "final": finals[seat], "opp_final": finals[1 - seat],
            "win": finals[seat] > finals[1 - seat],
            "days": {str(k): v for k, v in logs[seat]["days"].items()},
        }
        row.update(_tally_actions(actions, logs[seat]["prices"], logs[seat]["held"], logs[seat]["money"]))
        rows.append(row)
    return rows


def select_sessions(paths, per_bin, rng):
    need = [per_bin] * len(EDGES)
    chosen = []
    for p in paths:
        s = json.load(open(p))
        sc = s.get("scores") or [None, None]
        take = False
        for r in sc:
            if r is not None and need[rating_bin(r)] > 0:
                take = True
        if take:
            chosen.append((p, sc))
            for r in sc:
                if r is not None:
                    need[rating_bin(r)] -= 1
        if all(n <= 0 for n in need):
            break
    return chosen


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def report(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "profiles.jsonl"))]
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] == "ours":
            groups["OURS vs " + LABELS[rating_bin(r["opp_rating"] or 0)]].append(r)
        elif r["rating"] is not None:
            groups[LABELS[rating_bin(r["rating"])]].append(r)
    order = LABELS + sorted(k for k in groups if k.startswith("OURS"))
    for g in order:
        rs = groups.get(g)
        if not rs:
            continue
        print(f"\n=== {g}  (n={len(rs)} sides, final ${_mean([r['final'] for r in rs]):,.0f}, "
              f"win {_mean([r['win'] for r in rs]):.2f}) ===")
        print(f"{'day':>4} {'money':>8} {'hands':>5} {'land':>4} {'tiles':>5} {'anim':>4} "
              f"{'empty':>5} {'shed':>5}  plants                          animals")
        for d in DAYS:
            snaps = [r["days"].get(str(d)) for r in rs]
            snaps = [s for s in snaps if s]
            if not snaps:
                continue
            plants = defaultdict(float)
            animals = defaultdict(float)
            for s in snaps:
                for k, v in s["plants"].items():
                    plants[k] += v / len(snaps)
                for k, v in s["animals"].items():
                    animals[k] += v / len(snaps)
            pl = " ".join(f"{k[:3]}{v:.0f}" for k, v in sorted(plants.items(), key=lambda kv: -kv[1]) if v >= 0.5)
            an = " ".join(f"{k[:3]}{v:.0f}" for k, v in sorted(animals.items(), key=lambda kv: -kv[1]) if v >= 0.5)
            print(f"{d:>4} {_mean([s['money'] for s in snaps]):>8,.0f} {_mean([s['hands'] for s in snaps]):>5.1f} "
                  f"{_mean([s['land'] for s in snaps]):>4.1f} {_mean([sum(s['plants'].values()) for s in snaps]):>5.1f} "
                  f"{_mean([sum(s['animals'].values()) for s in snaps]):>4.1f} "
                  f"{_mean([s['empty_structures'] for s in snaps]):>5.1f} {_mean([s['shed_total'] for s in snaps]):>5.1f}  "
                  f"{pl:<31} {an}")
        print("  cash flow by day (in/out, top sell products):")
        for d in range(0, 30):
            ci = _mean([r.get("cash_in", {}).get(str(d), 0) for r in rs])
            co = _mean([r.get("cash_out", {}).get(str(d), 0) for r in rs])
            byp = defaultdict(float)
            for r in rs:
                for k, v in r.get("sells_by_day", {}).get(str(d), {}).items():
                    byp[k] += v / len(rs)
            top = " ".join(f"{k[:3]}${v:,.0f}" for k, v in sorted(byp.items(), key=lambda kv: -kv[1])[:4] if v >= 50)
            if d <= 14 or d % 4 == 0:
                print(f"    d{d:>2}: in ${ci:>6,.0f} out ${co:>6,.0f}  {top}")
        rev = defaultdict(float)
        qty = defaultdict(float)
        for r in rs:
            for k, (q, v) in r["sells"].items():
                rev[k] += v / len(rs)
                qty[k] += q / len(rs)
        total = sum(rev.values()) or 1
        print("  revenue mix: " + ", ".join(
            f"{k} ${v:,.0f} ({v/total:.0%}, {qty[k]:.0f}u)" for k, v in sorted(rev.items(), key=lambda kv: -kv[1]) if v / total >= 0.02))
        ab = defaultdict(float)
        for r in rs:
            for k, v in r["animal_buys"].items():
                ab[k] += v / len(rs)
        hires_total = _mean([sum(r["hires_by_day"].values()) for r in rs])
        hires_d0 = _mean([r["hires_by_day"].get("0", 0) for r in rs])
        land = [d for r in rs for d in r["land_days"]]
        land_n = _mean([len(r["land_days"]) for r in rs])
        print(f"  animal buys: {dict((k, round(v, 1)) for k, v in ab.items())}  hires/game {hires_total:.1f} "
              f"(day0 {hires_d0:.1f})  land buys/game {land_n:.2f} at days {sorted(land)[:12]}  "
              f"feed bought {_mean([r['feed_bought'] for r in rs]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="decode", choices=["decode", "report"])
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--ours", default=None, help="bundle to seat against the recorded scripts")
    ap.add_argument("--ours-bins", default="0,5", help="rating bins (indices) of opponents to face")
    ap.add_argument("--ours-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rl/datasets/decode")
    args = ap.parse_args()

    if args.mode == "report":
        report(args.out)
        return

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    paths = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    rng.shuffle(paths)
    chosen = select_sessions(paths, args.per_bin, rng)
    jobs = [(p, None, None) for p, _ in chosen]
    if args.ours:
        want = {int(b) for b in args.ours_bins.split(",")}
        count = defaultdict(int)
        for p, sc in chosen + [(p, json.load(open(p)).get("scores") or [None, None]) for p in paths]:
            for script_side in (0, 1):
                r = sc[script_side]
                if r is None:
                    continue
                b = rating_bin(r)
                if b in want and count[b] < args.ours_n:
                    jobs.append((p, args.ours, 1 - script_side))
                    count[b] += 1
    print(f"{len(chosen)} recorded sessions, {len(jobs) - len(chosen)} 'ours' games", flush=True)
    with Pool(args.procs) as pool:
        with open(os.path.join(args.out, "profiles.jsonl"), "w") as f:
            for i, rows in enumerate(pool.imap_unordered(decode_job, jobs)):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(jobs)} games decoded", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
