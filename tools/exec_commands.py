"""Does the executor OBEY? Command an intent, then check it was carried out.

`tools/rule_tests.py` asks "did the executor do the right thing unprompted",
and champ-h8 misses 21% of the time. That number cannot say WHOSE fault it is:
the planner may never have asked. The executor's actual contract is narrower --
the planner hands it intents (buy the 4th quadrant, build a coop, fertilise
this plant, fetch n of an item, dig that tile) and it must carry them out.

So: take the SAME mined states, inject the intent, and see if it lands.

    layer   what is injected                            what it exonerates
    -----   ------------------------------------------  ------------------
    TASK    a Task at P_SURVIVAL on a named tile         routing + execution
    DELIVER a Task whose `need` is only in the shed      provisioning + carry
    PLAN    a DayPlan field, re-asserted every turn      build_tasks + market

Read against `rule_tests.py`:

    unprompted FAIL + commanded PASS  -> the PLANNER never asked
    unprompted FAIL + commanded FAIL  -> the EXECUTOR cannot do it

Both are injected through the real machinery -- `build_tasks` gets the extra
Task and the live `plan` object, exactly as the planner would hand them over --
so nothing here bypasses assignment, provisioning or the market controller.

    PYTHONPATH=. uv run python tools/exec_commands.py run --agent champ-h8
    PYTHONPATH=. uv run python tools/exec_commands.py run --cmd task.DIG
"""
import argparse
import collections
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import rule_harness as H                      # noqa: E402
from tools.rule_tests import (CASES, OUTDIR, TPD, case_name,  # noqa: E402
                              render, window_for)

RESULTS = os.path.join(REPO, "rl", "exec_command_results.json")
LAND_ORDER = ["NE", "SW", "SE"]


# --------------------------------------------------------------- bundle ----
def load_bundle(name):
    """-> (agent callable, shared globals dict).

    Every module in a bundle execs into ONE dict and the module aliases are a
    SimpleNamespace over it, so patching a name means patching both.
    """
    path = os.path.join(REPO, "arena", "agents", name, "main.py")
    outer = {"__name__": f"bundle_{name}"}
    exec(compile(open(path).read(), path, "exec"), outer)
    return outer["agent"], outer["_G"]


def patch(G, name, fn):
    G[name] = fn
    for alias in ("scheduler", "planner", "policy", "market"):
        ns = G.get(alias)
        if ns is not None and hasattr(ns, name):
            setattr(ns, name, fn)


# ---------------------------------------------------------- the commands ---
def tile(view, x, y):
    try:
        return view.me.tile(x, y)
    except Exception:
        return None


def _is(t, kind):
    return isinstance(t, dict) and t.get("kind") == kind


def carried(view, item):
    return sum(view.unit_inv(i).get(item, 0) for i, _, _ in view.me.units)


class Command:
    """`apply` runs every turn from T; `done` says the intent has landed."""

    def __init__(self, name, layer, rules, apply, done, settle_days=0,
                 need=None):
        self.name = name
        self.layer = layer        # task | deliver | plan
        self.rules = rules        # which mined cases to use as targets
        self.apply = apply
        self.done = done
        self.settle_days = settle_days
        self.need = need          # item the deliver layer must fetch


COMMANDS = []


def task_cmd(name, op, rules, done, need=None, layer="task", settle_days=0):
    def apply(view, plan, tasks, Task, c):
        tasks.append(Task(0, c["x"], c["y"], list(op), need=need))
    COMMANDS.append(Command(name, layer, rules, apply, done, settle_days, need))


# -- TASK layer: the unit needs nothing it does not already have -------------
task_cmd("task.WATER", ["WATER"], ["WATER.bonus_window", "WATER.survival",
                                   "WATER.fertilised_production"],
         lambda view, c: bool(_is(tile(view, c["x"], c["y"]), "PLANT")
                              and tile(view, c["x"], c["y"]).get("watered_today")))

task_cmd("task.HARVEST", ["HARVEST"],
         ["HARVEST.one_time_ripe", "HARVEST.ongoing_before_clip",
          "HARVEST.animal_before_clip"],
         lambda view, c: (lambda t: t is None or not isinstance(t, dict)
                          or t.get("yield_units", 0) == 0)(
             tile(view, c["x"], c["y"])))

task_cmd("task.DIG", ["DIG"], ["DIG.clears_a_weed"],
         lambda view, c: not _is(tile(view, c["x"], c["y"]), "WEED"))

task_cmd("task.CARE", ["CARE"], ["CARE.banks_a_bonus"],
         lambda view, c: (lambda t: isinstance(t, dict) and t.get("cared_today"))(
             tile(view, c["x"], c["y"])))

task_cmd("task.COLLECT_FERTILIZER", ["COLLECT_FERTILIZER"],
         ["COLLECT_FERTILIZER.use_it_or_lose_it"],
         lambda view, c: not (isinstance(tile(view, c["x"], c["y"]), dict)
                              and tile(view, c["x"], c["y"]).get("fertilizer_available")))

task_cmd("task.BUILD_COOP", ["BUILD_COOP"], ["BUILD_COOP.for_a_goose_we_hold"],
         lambda view, c: _is(tile(view, c["x"], c["y"]), "COOP"))

task_cmd("task.BUILD_PASTURE", ["BUILD_PASTURE"],
         ["BUILD_PASTURE.for_stock_we_hold"],
         lambda view, c: _is(tile(view, c["x"], c["y"]), "PASTURE"))


# -- DELIVER layer: the item is in the shed, so it must be fetched first -----
task_cmd("deliver.FERTILIZE", ["FERTILIZE"],
         ["FERTILIZE.one_time", "FERTILIZE.ongoing"],
         lambda view, c: (lambda t: isinstance(t, dict)
                          and t.get("fertilized_until_day", -1) >= view.day)(
             tile(view, c["x"], c["y"])),
         need="FERTILIZER", layer="deliver")

task_cmd("deliver.FEED", ["FEED"], ["FEED.keeps_it_alive"],
         lambda view, c: (lambda t: isinstance(t, dict) and t.get("fed_today"))(
             tile(view, c["x"], c["y"])),
         need="WHEAT", layer="deliver")


def _place_done(view, c):
    t = tile(view, c["x"], c["y"])
    return isinstance(t, dict) and "animal" in t


def _place_apply(view, plan, tasks, Task, c):
    kind = (c.get("op") or [None, None])[1]
    if kind not in ("GOOSE", "COW", "SHEEP"):
        kind = "COW"
    tasks.append(Task(0, c["x"], c["y"], ["PLACE", kind], need=kind))


COMMANDS.append(Command("deliver.PLACE", "deliver",
                        ["PLACE.animal_on_structure"], _place_apply,
                        _place_done, 0, need="ANIMAL"))


# -- PLAN layer: a DayPlan field, re-asserted every turn ---------------------
def plan_cmd(name, rules, apply, done, settle_days=1):
    COMMANDS.append(Command(name, "plan", rules, apply, done, settle_days))


def _next_quadrant(view):
    for q in LAND_ORDER:
        if q not in view.me.unlocked_quadrants:
            return q
    return None


plan_cmd("plan.BUY_LAND", ["BUY_LAND.the_quadrant_gets_used"],
         lambda view, plan, tasks, Task, c: (
             plan.__setattr__("land_buys", [_next_quadrant(view)])
             if _next_quadrant(view) else None,
             plan.__setattr__("land_margin", 0))[0],
         lambda view, c: len(view.me.unlocked_quadrants) > c["_q0"],
         settle_days=2)


def _plant_apply(view, plan, tasks, Task, c):
    # a case mined from PLANT carries the crop in op[1]; one mined from
    # BUY_SEED is a market op with op == ["BUY_SEED"] and the crop in `item`
    op = c.get("op") or []
    crop = c.get("item") or (op[1] if len(op) > 1 else None) or "WHEAT"
    plan.plant_quota = dict(plan.plant_quota or {})
    plan.plant_quota[crop] = max(3, plan.plant_quota.get(crop, 0))


plan_cmd("plan.PLANT", ["PLANT.survives_its_first_night",
                        "BUY_SEED.gets_planted"],
         _plant_apply,
         lambda view, c: sum(
             1 for _, _, t in view.me.plants()
             if t.get("planted_day", -1) >= c["day"]) >= 1,
         settle_days=1)


def _coop_apply(view, plan, tasks, Task, c):
    plan.structure_builds = dict(plan.structure_builds or {})
    plan.structure_builds["COOP"] = max(1, plan.structure_builds.get("COOP", 0))


plan_cmd("plan.COOP", ["BUILD_COOP.for_a_goose_we_hold",
                       "BUY_ANIMAL.reaches_a_structure"],
         _coop_apply,
         lambda view, c: sum(1 for _, _, t in view.me.empty_structures()
                             if t["kind"] == "COOP") > c["_coop0"]
         or sum(1 for _, _, t in view.me.animals()
                if t["animal"] == "GOOSE") > c["_goose0"],
         settle_days=1)


def _animal_apply(view, plan, tasks, Task, c):
    kind = c.get("_kind", "GOOSE")
    plan.animal_buys = dict(plan.animal_buys or {})
    plan.animal_buys[kind] = max(1, plan.animal_buys.get(kind, 0))
    plan.structure_builds = dict(plan.structure_builds or {})
    st = {"GOOSE": "COOP"}.get(kind, "PASTURE")
    plan.structure_builds[st] = max(1, plan.structure_builds.get(st, 0))


plan_cmd("plan.BUY_ANIMAL", ["BUY_ANIMAL.reaches_a_structure",
                             "PLACE.animal_on_structure"],
         _animal_apply,
         lambda view, c: sum(1 for _ in view.me.animals()) > c["_herd0"],
         settle_days=2)


plan_cmd("plan.HIRE", ["HIRE.the_hand_does_work"],
         lambda view, plan, tasks, Task, c: plan.__setattr__(
             "hires_target", max(plan.hires_target, view.me.hires_today + 2)),
         lambda view, c: view.me.hires_today > c["_hires0"],
         settle_days=0)


BY_NAME = {c.name: c for c in COMMANDS}


# -------------------------------------------------------------------- run ---
def run_case(cmd, case, agent_name, keep_env=False):
    agent, G = load_bundle(agent_name)
    Task = G["Task"]
    s = H.load_session(os.path.join(REPO, "rl", "datasets", "sessions",
                                    case["session"]))
    T = case["step"]
    win_end = (T // TPD + 1) * TPD - 1
    settle_end = min(718, win_end + cmd.settle_days * TPD)

    state = {"done_step": None, "baseline": False, "had_need": None,
             "picked": False, "harness_error": None, "injected": 0}
    orig_bt = G["build_tasks"]

    def wrapper(view, plan):
        tasks = orig_bt(view, plan)
        # NOTHING in here may raise: build_tasks is called from inside
        # policy.act, and SafeAgent catches any exception and emits all-PASS.
        # A crash here therefore looks exactly like an executor that refused
        # the command. (It did: an IndexError on `op[1]` for argument-less ops
        # silently PASSed the whole of step T in every earlier run.)
        try:
            if view.step == T:
                # baseline values the `done` predicates compare against
                case["_q0"] = len(view.me.unlocked_quadrants)
                case["_coop0"] = sum(1 for _, _, t in view.me.empty_structures()
                                     if t["kind"] == "COOP")
                case["_goose0"] = sum(1 for _, _, t in view.me.animals()
                                      if t["animal"] == "GOOSE")
                case["_herd0"] = sum(1 for _ in view.me.animals())
                case["_hires0"] = view.me.hires_today
                _op = case.get("op") or []
                _arg = _op[1] if len(_op) > 1 else None
                case["_kind"] = _arg if _arg in ("GOOSE", "COW", "SHEEP") else "GOOSE"
                if cmd.layer == "deliver" and cmd.need:
                    item = cmd.need if cmd.need != "ANIMAL" else case["_kind"]
                    state["had_need"] = carried(view, item)
        except Exception as e:                                # pragma: no cover
            state["harness_error"] = f"{type(e).__name__}: {e}"
        # `done` is checked one step PAST the window: an action taken on the
        # last turn only shows up in the next turn's observation, so stopping
        # at settle_end scores every command obeyed on its final turn as a
        # failure. Injection still stops at settle_end.
        if T <= view.step <= settle_end + 1:
            if state["done_step"] is None:
                try:
                    if cmd.done(view, case):
                        state["done_step"] = min(view.step, settle_end)
                except Exception as e:
                    state["harness_error"] = f"done: {type(e).__name__}: {e}"
            if state["done_step"] is None and view.step <= settle_end:
                try:
                    cmd.apply(view, plan, tasks, Task, case)
                    state["injected"] = state.get("injected", 0) + 1
                except Exception as e:
                    state["harness_error"] = f"apply: {type(e).__name__}: {e}"
        return tasks

    patch(G, "build_tasks", wrapper)

    out = H.play(s, case["seat"], agent, T, settle_end,
                 watch=[(case["x"], case["y"])] if case.get("x", -1) >= 0 else (),
                 snap_range=(max(0, T - TPD), min(719, settle_end + TPD)),
                 keep_env=keep_env)

    rec = out["rec"]
    ops = [r for r in rec.unit_ops if T <= r["step"] <= settle_end]
    state["picked"] = any(
        r["op"][0] == "PICKUP" and len(r["op"]) > 1
        and r["op"][1] == (cmd.need if cmd.need != "ANIMAL"
                           else case.get("_kind", "GOOSE"))
        for r in ops) if cmd.need else None
    res = dict(case)
    for k in list(res):
        if k.startswith("_"):
            res.pop(k)
    res.update({
        "cmd": cmd.name, "layer": cmd.layer, "agent": agent_name,
        "obeyed": state["done_step"] is not None,
        "done_step": state["done_step"],
        "latency": (state["done_step"] - T) if state["done_step"] is not None else None,
        "window": [T, settle_end],
        "carried_at_T": state["had_need"],
        "fetched": state["picked"],
        "injected_turns": state.get("injected", 0),
        "harness_error": state.get("harness_error"),
    })
    return res, out


def run(args):
    cases = json.load(open(CASES))
    by_rule = collections.defaultdict(list)
    for c in cases:
        by_rule[c["rule"]].append(c)

    cmds = [c for c in COMMANDS if not args.cmd or c.name in args.cmd]
    only = None
    if args.failures_only:
        prev_path = os.path.join(REPO, "rl",
                                 f"exec_command_results_{args.agent}.json")
        if not os.path.exists(prev_path):
            raise SystemExit(f"no previous results at {prev_path}")
        prev = json.load(open(prev_path))
        only = {(r["cmd"], r["id"]) for r in prev if not r["obeyed"]}
        print(f"re-running {len(only)} previously failing cases only\n")
    outdir = os.path.join(OUTDIR, args.agent + "-commands")
    os.makedirs(outdir, exist_ok=True)
    results, total = [], 0
    t0 = time.time()
    print(f"agent {args.agent}, {len(cmds)} commands\n")
    print(f"{'command / case':66s}{'obey':>6}{'lat':>5}  replay")
    for cmd in cmds:
        targets = []
        for r in cmd.rules:
            targets += by_rule.get(r, [])
        if args.per_cmd:
            targets = targets[:args.per_cmd]
        for case in targets:
            if only is not None and (cmd.name, case["id"]) not in only:
                continue
            try:
                res, out = run_case(cmd, dict(case), args.agent,
                                    keep_env=not args.no_html)
            except Exception as e:                           # pragma: no cover
                print(f"  {cmd.name}/{case['id']:50s} ERROR {e}")
                continue
            link = ""
            if out["env"] is not None:
                lo, hi = window_for(case["step"], res["window"][1])
                fname = cmd.name + "--" + case_name(case) + ".html"
                p, n = render(out["env"], outdir, fname, lo, hi, subdir=cmd.name)
                res["html"] = os.path.relpath(p, REPO)
                total += n
                link = f"{os.path.basename(p)} ({n/1e6:.1f} MB)"
            results.append(res)
            lat = res["latency"]
            print(f"  {cmd.name + '/' + case['id']:64s}"
                  f"{'yes' if res['obeyed'] else 'NO':>6}"
                  f"{(lat if lat is not None else '-'):>5}  {link}", flush=True)
    out_path = os.path.join(REPO, "rl", f"exec_command_results_{args.agent}.json")
    # ALWAYS merge into whatever is on disk. A subset run (--cmd, --limit,
    # --failures-only) that overwrote the file would silently truncate the
    # agent's baseline, and every later comparison would read the subset as
    # the whole suite.
    subset = bool(args.cmd or getattr(args, "per_cmd", 0)
                  or args.failures_only)
    if subset and os.path.exists(out_path):
        merged = {(r["cmd"], r["id"]): r for r in json.load(open(out_path))}
        merged.update({(r["cmd"], r["id"]): r for r in results})
        results = [merged[k] for k in sorted(merged)]
    json.dump(results, open(out_path, "w"), indent=1)
    json.dump(results, open(RESULTS, "w"), indent=1)   # latest, for report
    bad = [r for r in results if r.get("harness_error")]
    if bad:
        print(f"\n!! {len(bad)} cases hit a HARNESS error -- their verdicts are "
              f"NOT about the executor. Fix before reading the table:")
        for r in bad[:5]:
            print(f"   {r['cmd']}/{r['id']}: {r['harness_error']}")
    print(f"\nwrote {len(results)} results to {out_path}  ({time.time()-t0:.0f}s)")
    if total:
        print(f"replays in {os.path.relpath(outdir, REPO)}/ ({total/1e9:.2f} GB)")
    report(args, results)


def report(args, results=None):
    if results is None:
        results = json.load(open(RESULTS))
    unprompted = {}
    p = os.path.join(REPO, "rl", "rule_results.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            unprompted[(r["rule"], r["id"])] = r["capability"] >= 3

    print(f"\n{'command':26s}{'layer':9s}{'n':>4}{'OBEYED':>8}"
          f"{'median lat':>12}{'unprompted':>12}   verdict")
    print("-" * 92)
    for cmd in COMMANDS:
        rows = [r for r in results if r["cmd"] == cmd.name]
        if not rows:
            continue
        n = len(rows)
        ob = sum(r["obeyed"] for r in rows) / n
        lats = sorted(r["latency"] for r in rows if r["latency"] is not None)
        med = lats[len(lats) // 2] if lats else None
        un = [unprompted.get((r["rule"], r["id"])) for r in rows]
        un = [u for u in un if u is not None]
        unp = sum(un) / len(un) if un else None
        if ob == 1.0:
            v = "OBEYS" + ("  (planner never asked)"
                           if unp is not None and unp < 1.0 else "")
        elif ob >= 0.5:
            v = "partial"
        else:
            v = "EXECUTOR CANNOT"
        print(f"{cmd.name:26s}{cmd.layer:9s}{n:>4}{ob:>8.0%}"
              f"{(med if med is not None else '-'):>12}"
              f"{(f'{unp:.0%}' if unp is not None else '-'):>12}   {v}")
    n = len(results)
    if n:
        print("-" * 92)
        print(f"{'ALL':26s}{'':9s}{n:>4}"
              f"{sum(r['obeyed'] for r in results)/n:>8.0%}")
        print("\nOBEYED = the commanded intent landed inside its window. "
              "lat = turns from command to completion.\nunprompted = the same "
              "states in tools/rule_tests.py, where nobody asked.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="c", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", default="champ-h8")
    r.add_argument("--cmd", nargs="*")
    r.add_argument("--per-cmd", type=int, default=0)
    r.add_argument("--no-html", action="store_true")
    r.add_argument("--failures-only", action="store_true",
                   help="re-run only the cases this agent failed last time, "
                        "so a failing set can be rendered without redoing all "
                        "217 (results are merged back into the agent's file)")
    r.set_defaults(fn=run)
    p = sub.add_parser("report")
    p.set_defaults(fn=lambda a: report(a))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
