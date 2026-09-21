"""TDD ground truth: can our executor perform each README action, in a real state?

Two test approaches, both run here:

  MINE   find a state where a 2000+ rated seat performed the action and the
         README precondition held -- their doing it is the evidence it was
         right there, so the corpus defines correctness, not us.
  RUN    replay that episode from step 0 on its own seed (fidelity is exact),
         hand OUR executor the wheel at that step, judge it, PASS to the end.

Two verdicts per case, because they answer different questions:

  RIGID       our executor emits the op at EXACTLY step T on EXACTLY that tile
              -- agreement with the reference decision
  CAPABILITY  it emits the op on ANY tile where the same README precondition
              holds, any time from T until the opportunity closes (end of the
              in-game day) -- can it do the thing at all

Three tiers inside each verdict: emitted (T1), accepted by the engine (T2),
and the README rule's value actually materialised (T3). T3 is the bar.

    PYTHONPATH=. uv run python tools/rule_tests.py mine  --sessions 60
    PYTHONPATH=. uv run python tools/rule_tests.py run   --agent champ-h8
    PYTHONPATH=. uv run python tools/rule_tests.py report
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import rule_harness as H            # noqa: E402  (installs fast_env)
from tools.rule_catalogue import RULES, BY_NAME  # noqa: E402

TPD = 24
SESS_DIR = os.path.join(REPO, "rl", "datasets", "sessions")
CASES = os.path.join(REPO, "rl", "rule_cases.json")
RESULTS = os.path.join(REPO, "rl", "rule_results.json")


def results_path(agent):
    return os.path.join(REPO, "rl", f"rule_results_{agent}.json")
OUTDIR = os.path.join(REPO, "tests", "replays")
DAY_BANDS = [(0, 2), (3, 6), (7, 10), (11, 15), (16, 21), (22, 29)]
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def band_of(day):
    for i, (lo, hi) in enumerate(DAY_BANDS):
        if lo <= day <= hi:
            return i
    return len(DAY_BANDS) - 1


# ------------------------------------------------------------------- mine ---
def eligible_sessions(min_rating, limit, seed=0):
    out = []
    for p in sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json"))):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        sc = s.get("scores") or []
        seats = [i for i in (0, 1)
                 if len(sc) > i and sc[i] and sc[i] >= min_rating
                 and s.get("actions") and len(s["actions"]) == 2]
        for seat in seats:
            out.append((p, seat, sc[seat]))
    random.Random(seed).shuffle(out)
    return out[:limit]


def mine(args):
    picks = eligible_sessions(args.min_rating, args.sessions, args.seed)
    print(f"mining {len(picks)} reference seats rated >= {args.min_rating}\n")
    # rule -> band -> [case]
    found = collections.defaultdict(lambda: collections.defaultdict(list))
    t0 = time.time()
    for n, (path, seat, rating) in enumerate(picks, 1):
        s = H.load_session(path)
        try:
            out = H.replay_only(s, seat)
        except Exception as e:                                # pragma: no cover
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        rec = out["rec"]
        by_step_market = collections.defaultdict(list)
        for m in rec.market:
            by_step_market[m["step"]].append(m)
        for r in RULES:
            if r.kind == "unit":
                src = (x for x in rec.unit_ops if r.matches(x))
            else:
                src = ({"step": m["step"], "op": [m["op"]], "item": m["item"],
                        "price": m["price"], "ok": m["ok"],
                        "day": m["step"] // TPD, "x": -1, "y": -1}
                       for m in rec.market
                       if m["ok"] and m["op"] == r.op
                       and (r.arg is None or m["item"] == r.arg))
            for hit in src:
                try:
                    if not r.precond(hit):
                        continue
                except Exception:
                    continue
                day = hit["step"] // TPD
                found[r.name][band_of(day)].append({
                    "rule": r.name, "session": os.path.basename(path),
                    "seat": seat, "rating": rating, "step": hit["step"],
                    "day": day, "x": hit["x"], "y": hit["y"],
                    "op": hit["op"], "item": hit.get("item"),
                })
        if n % 10 == 0:
            print(f"  {n}/{len(picks)} sessions  ({time.time()-t0:.0f}s)",
                  flush=True)

    # even spread: up to `per_band` cases per rule per day band, distinct
    # sessions first so one game cannot define a whole rule
    rng = random.Random(args.seed)
    cases = []
    for r in RULES:
        chosen = []
        for b in range(len(DAY_BANDS)):
            pool = found[r.name][b]
            rng.shuffle(pool)
            seen = set()
            take = []
            for c in pool:
                key = (c["session"], c["seat"])
                if key in seen and len(take) < args.per_band:
                    continue
                seen.add(key)
                take.append(c)
                if len(take) >= args.per_band:
                    break
            chosen += take
        for i, c in enumerate(chosen):
            c["id"] = f"{r.name}#{i:02d}"
        cases += chosen
        bands = collections.Counter(band_of(c["day"]) for c in chosen)
        print(f"  {r.name:42s} {len(chosen):3d} cases  "
              f"bands {[bands.get(b, 0) for b in range(len(DAY_BANDS))]}")

    os.makedirs(os.path.dirname(CASES), exist_ok=True)
    json.dump(cases, open(CASES, "w"), indent=1)
    print(f"\nwrote {len(cases)} cases to {CASES}  ({time.time()-t0:.0f}s)")
    missing = [r.name for r in RULES if not found[r.name]]
    if missing:
        print("\nNO REFERENCE CASE FOUND (the corpus never does this, or the "
              "precondition never holds):")
        for m in missing:
            print(f"  {m}")


# -------------------------------------------------------------------- run ---
def build_ctx(rec, case):
    grids = rec.grid
    snap_steps = sorted(grids)

    def grid_at(step):
        return grids.get(step)

    def yield_at(day, x, y):
        """yield_units on the last snapshot of `day` before its refresh."""
        best = None
        for s in snap_steps:
            if s // TPD == day:
                t = grids[s][y][x]
                if isinstance(t, dict):
                    best = t.get("yield_units", 0)
        return best

    def shed_at_eod(day):
        return rec.eod_shed.get(day, {})

    def worked_units(day):
        return sum(1 for r in rec.unit_ops
                   if r["step"] // TPD == day
                   and r["op"][0] not in MOVES and r["op"][0] != "PASS")

    def new_quadrant_used(step):
        before = after = None
        for s in sorted(rec.priv):
            if s <= step:
                before = rec.priv[s]["quadrants"]
            elif after is None:
                after = rec.priv[s]["quadrants"]
        if not before or not after:
            return False
        new = [q for q in after if q not in before]
        if not new:
            return False
        box = {"NW": (0, 0), "NE": (5, 0), "SW": (0, 5), "SE": (5, 5)}
        for q in new:
            x0, y0 = box[q]
            for s in snap_steps:
                if s <= step:
                    continue
                g = grids[s]
                for yy in range(y0, y0 + 5):
                    for xx in range(x0, x0 + 5):
                        if isinstance(g[yy][xx], dict):
                            return True
        return False

    return {"grid_at": grid_at, "snap_steps": snap_steps, "yield_at": yield_at,
            "shed_at_eod": shed_at_eod, "worked_units": worked_units,
            "new_quadrant_used": new_quadrant_used, "eod_grid": rec.eod_grid,
            "our_ops": rec.unit_ops}


def score_one(r, case, rec, ctx):
    """-> (rigid_tier, capability_tier, evidence) with tiers in 0..3."""
    T = case["step"]
    win_end = (T // TPD + 1) * TPD - 1
    ev = {}

    if r.kind == "market":
        recs = [{"step": m["step"], "op": [m["op"]], "item": m["item"],
                 "price": m["price"], "ok": m["ok"], "day": m["step"] // TPD,
                 "x": -1, "y": -1}
                for m in rec.market
                if m["op"] == r.op and (r.arg is None or m["item"] == r.arg)]
        strict = [m for m in recs if m["step"] == T]
        widen = [m for m in recs if T <= m["step"] <= win_end]
    else:
        strict = [m for m in rec.unit_ops
                  if m["step"] == T and r.matches(m)
                  and (m["x"], m["y"]) == (case["x"], case["y"])]
        widen = []
        for m in rec.unit_ops:
            if not (T <= m["step"] <= win_end) or not r.matches(m):
                continue
            try:
                if r.precond(m):
                    widen.append(m)
            except Exception:
                pass

    def tier(pool):
        if not pool:
            return 0, None
        best, bestrec = 1, pool[0]
        for m in pool:
            t = 1
            try:
                if r.accepted(m):
                    t = 2
                    if r.paid(m, ctx):
                        t = 3
            except Exception:
                pass
            if t > best:
                best, bestrec = t, m
        return best, bestrec

    rt, rrec = tier(strict)
    ct, crec = tier(widen)
    ev["rigid_step"] = rrec["step"] if rrec else None
    ev["cap_step"] = crec["step"] if crec else None
    ev["cap_tile"] = [crec["x"], crec["y"]] if crec else None
    ev["window"] = [T, win_end]
    return rt, ct, ev


def run_case(case, agent_name, render_dir=None):
    r = BY_NAME[case["rule"]]
    s = H.load_session(os.path.join(SESS_DIR, case["session"]))
    T = case["step"]
    win_end = (T // TPD + 1) * TPD - 1
    settle_end = min(718, win_end + r.settle_days * TPD)
    snap_lo = max(0, T - TPD)
    out = H.play(s, case["seat"], H.load_agent(agent_name), T, settle_end,
                 watch=[(case["x"], case["y"])] if case["x"] >= 0 else (),
                 snap_range=(snap_lo, min(719, settle_end + TPD)),
                 keep_env=bool(render_dir))
    rec = out["rec"]
    ctx = build_ctx(rec, case)
    rigid, cap, ev = score_one(r, case, rec, ctx)
    res = dict(case)
    res.update({"agent": agent_name, "rigid": rigid, "capability": cap,
                "settle_end": settle_end, "us": out["us"], "them": out["them"],
                **ev})
    return res, out


# ----------------------------------------------------------------- render ---
VIEWER = "_viewer.js"
_SCRIPT_RE = re.compile(r'<script type="module" crossorigin>')


def externalise(html, outdir, rel=""):
    """Pull the 14.7 MB vite bundle out into one shared file per directory.

    The bundle is a plain IIFE (no top-level import/export/import.meta), so a
    classic `defer` script is equivalent -- and unlike `type=module` it loads
    from file:// without tripping CORS, so the pages open by double-click.
    `defer` also keeps it running after the inline window.kaggle payload,
    which `get_player` injects later in <head>.
    """
    m = _SCRIPT_RE.search(html)
    if not m:
        return html
    end = html.find("</script>", m.end())
    body = html[m.end():end]
    path = os.path.join(outdir, VIEWER)
    if not os.path.exists(path) or os.path.getsize(path) != len(body):
        with open(path, "w") as f:
            f.write(body)
    return (html[:m.start()] + f'<script defer src="{rel}{VIEWER}"></script>'
            + html[end + len("</script>"):])


LEAD_DAYS = 2      # in-game days of run-up before the commanded step
TAIL_TURNS = 10    # turns kept after the opportunity closes


def window_for(T, settle_end):
    """Step slice to render: LEAD_DAYS before the command, TAIL_TURNS after.

    The -1 matters. `env.steps[t]`'s observation is the state AFTER step t's
    actions, so a slice starting at `t` opens on turn t+1 and the run-up to
    the tested turn is off by one -- which is exactly what made a rendered
    window look like it began one turn late.
    """
    lo = max(0, T - LEAD_DAYS * TPD - 1)
    hi = min(719, settle_end + TAIL_TURNS)
    return lo, hi


def case_name(case):
    """Self-describing replay filename: what, where, which day/turn, whose seat.

    e.g. BUY_LAND.the_quadrant_gets_used-00-NE-d12-t06-p1
         WATER.bonus_window-03-MELON-x3y7-d12-t06-p0
    so a directory listing answers "which case is this" without the JSON.
    """
    op = case.get("op") or []
    bits = []
    if case.get("item"):                         # market ops: quadrant / item
        bits.append(str(case["item"]))
    elif len(op) > 1 and isinstance(op[1], str):  # PLANT WHEAT, PLACE COW, ...
        bits.append(op[1])
    if case.get("x", -1) >= 0:
        bits.append(f"x{case['x']}y{case['y']}")
    tag = "-".join(bits)
    nn = case["id"].split("#")[-1]
    day, hour = case["step"] // TPD, case["step"] % TPD
    stem = f"{case['rule']}-{nn}"
    if tag:
        stem += f"-{tag}"
    return f"{stem}-d{day:02d}-t{hour:02d}-p{case['seat']}".replace("/", "_")


def render(env, outdir, name, lo=None, hi=None, subdir=None):
    """One replay per case. `subdir` files it under its action, and the
    shared viewer stays at the agent root so each page loads `../_viewer.js`."""
    root = outdir
    if subdir:
        outdir = os.path.join(outdir, subdir)
    os.makedirs(outdir, exist_ok=True)
    allsteps = env.steps
    try:
        if lo is not None:
            env.steps = allsteps[max(0, lo):min(len(allsteps), hi + 1)]
        html = env.render(mode="html", width=1000, height=700)
    finally:
        env.steps = allsteps
    html = externalise(html, root, rel="../" if subdir else "")
    p = os.path.join(outdir, name)
    with open(p, "w") as f:
        f.write(html)
    return p, os.path.getsize(p)


def run(args):
    cases = json.load(open(CASES))
    if args.rule:
        cases = [c for c in cases if c["rule"] in args.rule]
    if args.limit:
        cases = cases[:args.limit]
    outdir = os.path.join(OUTDIR, args.agent)
    os.makedirs(outdir, exist_ok=True)
    results = []
    total_bytes = 0
    t0 = time.time()
    print(f"{len(cases)} cases, agent {args.agent}\n")
    print(f"{'case':46s}{'day':>4}{'rigid':>7}{'cap':>5}  replay")
    for i, c in enumerate(cases, 1):
        want_html = not args.no_html
        try:
            res, out = run_case(c, args.agent, render_dir=outdir if want_html else None)
        except Exception as e:                                # pragma: no cover
            print(f"  {c['id']:44s} ERROR {e}")
            continue
        link = ""
        if want_html and out["env"] is not None:
            T, S = c["step"], res["settle_end"]
            lo, hi = window_for(T, S)
            fname = case_name(c) + ".html"
            p, n = render(out["env"], outdir, fname, lo, hi, subdir=c["rule"])
            res["html"] = os.path.relpath(p, REPO)
            total_bytes += n
            link = f"{os.path.basename(p)} ({n/1e6:.1f} MB)"
            # a failing case gets the whole season too, so you can see what the
            # executor was doing instead
            if res["capability"] < 3 and not args.no_full:
                fp, fn = render(out["env"], outdir, fname[:-5] + "-full.html",
                                subdir=c["rule"])
                res["html_full"] = os.path.relpath(fp, REPO)
                total_bytes += fn
                link += f" +full ({fn/1e6:.1f} MB)"
        results.append(res)
        print(f"  {c['id']:44s}{c['day']:>4}{res['rigid']:>7}"
              f"{res['capability']:>5}  {link}", flush=True)
    out_path = results_path(args.agent)
    json.dump(results, open(out_path, "w"), indent=1)
    json.dump(results, open(RESULTS, "w"), indent=1)   # latest, for report
    print(f"\nwrote {len(results)} results to {out_path}  ({time.time()-t0:.0f}s)")
    if total_bytes:
        print(f"replays in {os.path.relpath(outdir, REPO)}/  "
              f"({total_bytes/1e9:.2f} GB + one shared {VIEWER})")
    report(args, results)


# ----------------------------------------------------------------- report ---
def report(args, results=None):
    if results is None:
        results = json.load(open(RESULTS))
    print(f"\n{'rule':44s}{'n':>3} | "
          f"{'RIGID  T1  T2  T3':>19} | {'CAPABILITY  T1  T2  T3':>24} | verdict")
    print("-" * 118)
    order = [r.name for r in RULES]
    for name in order:
        rows = [r for r in results if r["rule"] == name]
        if not rows:
            continue
        n = len(rows)

        def pct(key, t):
            return sum(1 for r in rows if r[key] >= t) / n

        rv = [pct("rigid", t) for t in (1, 2, 3)]
        cv = [pct("capability", t) for t in (1, 2, 3)]
        verdict = "PASS" if cv[2] == 1.0 else ("partial" if cv[2] > 0 else "FAIL")
        print(f"{name:44s}{n:>3} | "
              f"{rv[0]:>7.0%}{rv[1]:>5.0%}{rv[2]:>5.0%}  | "
              f"{cv[0]:>11.0%}{cv[1]:>5.0%}{cv[2]:>5.0%}   | {verdict}")
    n = len(results)
    if n:
        print("-" * 118)
        print(f"{'ALL':44s}{n:>3} | "
              f"{sum(r['rigid'] >= 1 for r in results)/n:>7.0%}"
              f"{sum(r['rigid'] >= 2 for r in results)/n:>5.0%}"
              f"{sum(r['rigid'] >= 3 for r in results)/n:>5.0%}  | "
              f"{sum(r['capability'] >= 1 for r in results)/n:>11.0%}"
              f"{sum(r['capability'] >= 2 for r in results)/n:>5.0%}"
              f"{sum(r['capability'] >= 3 for r in results)/n:>5.0%}")
        print("\nT1 emitted where the precondition held · T2 the engine did not "
              "no-op it · T3 the README rule's value materialised")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine")
    m.add_argument("--sessions", type=int, default=60)
    m.add_argument("--per-band", type=int, default=2)
    m.add_argument("--min-rating", type=float, default=2000)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=mine)

    r = sub.add_parser("run")
    r.add_argument("--agent", default="champ-h8")
    r.add_argument("--rule", nargs="*")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-html", action="store_true")
    r.add_argument("--no-full", action="store_true",
                   help="skip the whole-season render that failing cases get")
    r.set_defaults(fn=run)

    p = sub.add_parser("report")
    p.set_defaults(fn=lambda a: report(a))

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
