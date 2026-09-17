"""Execute the LP's own plan and gate it — is this problem solvable by calculation?

`tools/lp_bound.py` has only ever been used as a BOUND. It reproduces the mined
champion build order from first principles (wheat 154 vs their median 148,
strawberries 29 vs 29, cows 5.9 vs 6-9) which is strong evidence the economics
are right, but nobody has ever asked the executor to actually follow it.

This converts an LP solution into the planner's own language and ships it as a
frozen bundle, so `tools/exec_gate.py` can answer the question directly.

FLOWS vs STANDING TARGETS. The LP is a flow model ("plant x tiles of c on day
d"); the planner is a standing-target model ("keep n tiles of c occupied, quota
= target - current"). Injecting flows is brittle -- one failed planting and the
schedule never recovers -- so this converts the LP's activity occupancy into a
per-day STANDING count, which is what the planner already self-corrects toward.

    PYTHONPATH=. uv run python tools/lp_controller.py extract --mirror 5
    PYTHONPATH=. uv run python tools/lp_controller.py export --name lp-plan
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.lp_bound as L  # noqa: E402

SCHED = os.path.join(REPO, "rl", "datasets", "lp_schedule.json")
BASE_AGENT = "arena/agents/champ-h8/main.py"
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("COW", "SHEEP", "GOOSE")


def _rebuild(label):
    """Regenerate the activity a solution label names, to recover its occupancy."""
    parts = label.split("/")
    kind = parts[0]
    day = int(parts[1][1:])
    fert = 1 if "F" in parts[2:] else 0
    care = 1 if "C" in parts[2:] else 0
    if kind in ANIMALS:
        return L.animal_activity(kind, day, care)
    age = next((int(p[1:]) for p in parts[2:] if p.startswith("a")), None)
    if age is None:
        return L.ongoing_activity(kind, day, fert)
    return L.one_time_activity(kind, day, age, fert)


def extract(args):
    sched, res = None, None
    for r in range(max(1, args.mirror)):
        res = L.build_and_solve(args.eff, None, opp_schedule=sched, verbose=False)
        cur = res["sell_schedule"]
        sched = cur if r == 0 else {p: [(sched[p][d] + cur[p][d]) / 2
                                        for d in range(L.DAYS)] for p in sched}
        print(f"  round {r}: ${res['final']:,.0f}", flush=True)

    # standing occupancy per crop per day, and herd size per day
    stand = defaultdict(lambda: defaultdict(float))
    herd = defaultdict(lambda: defaultdict(float))
    for key, rows in res["plant_days"].items():
        for _, qty, label in rows:
            act = _rebuild(label)
            if act is None:
                continue
            tgt = herd if key in ANIMALS else stand
            for d in act["occupancy"]:
                tgt[d][key] += qty

    out = {
        "standing": {str(d): {c: round(stand[d][c]) for c in CROPS if stand[d][c] > 0.5}
                     for d in range(L.DAYS)},
        "herd": {str(d): {a: round(herd[d][a]) for a in ANIMALS if herd[d][a] > 0.5}
                 for d in range(L.DAYS)},
        "hands": [max(2, round(h)) for h in res["hands"]],
        "land_day": {str(j): int(d) for j, (d, _) in sorted(res["land"].items())},
        "margin": res["final"],
    }
    os.makedirs(os.path.dirname(SCHED), exist_ok=True)
    json.dump(out, open(SCHED, "w"), indent=1)
    print(f"\nconverged margin ${res['final']:,.0f}")
    print(f"{'day':>4} {'hands':>6}  standing tiles / herd")
    for d in range(L.DAYS):
        s = " ".join(f"{c[:4]}{v}" for c, v in sorted(out['standing'].get(str(d), {}).items()))
        h = " ".join(f"{a[:4]}{v}" for a, v in sorted(out['herd'].get(str(d), {}).items()))
        print(f"{d:>4} {out['hands'][d]:>6}  {s}   {h}")
    print(f"land: {out['land_day']}\nwrote {SCHED}")


# --------------------------------------------------------------------------
# Bundle patch: replace the planner's quota / herd / hires / land decisions
# --------------------------------------------------------------------------

INJECT = '''
# --- LP controller schedule (tools/lp_controller.py) -------------------------
# Standing tile counts and herd sizes per day, straight from the LP optimum.
# The planner already works in standing targets, so these drop into the same
# "quota = target - current" shape it uses, and self-correct when a planting
# fails.
LP_STAND = __STAND__
LP_HERD = __HERD__
LP_LAND = __LAND__


def _lp_stand(day, crop):
    return LP_STAND.get(day, {}).get(crop, 0)


def _lp_herd(day, kind):
    return LP_HERD.get(day, {}).get(kind, 0)
'''

# The whole planting section is replaced, not just the opening: the strawberry
# and wheat blocks below it would otherwise overwrite the LP's quotas.
PLANT_START = "        if day <= 1:\n            for crop, n in CHAMP_OPENING.items():"
PLANT_END = '            quota["WHEAT"] = max(0, wheat_target - crops.get("WHEAT", 0))'

PLANT_REPL = """        for _c in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
            _t = _lp_stand(day, _c)
            if _t:
                quota[_c] = max(0, _t - crops.get(_c, 0))"""


def export(args):
    sch = json.load(open(SCHED))
    stand = {int(k): v for k, v in sch["standing"].items()}
    herd = {int(k): v for k, v in sch["herd"].items()}
    land = {int(j): d for j, d in sch["land_day"].items()}

    src = open(os.path.join(REPO, BASE_AGENT)).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out = []
    for mod, text in mods:
        if mod == "planner":
            a = text.index(PLANT_START)
            b = text.index(PLANT_END, a) + len(PLANT_END)
            # 1. crop quotas entirely from the LP standing schedule
            text = text[:a] + PLANT_REPL + text[b:]
            # 2. hires straight from the LP
            text, n = re.subn(r"CHAMP_HIRES = \[[0-9,\s\n]*?\]",
                              "CHAMP_HIRES = " + repr(list(sch["hands"])), text, count=1)
            if n != 1:
                raise RuntimeError("CHAMP_HIRES not found")
            # 3. land on the LP's days
            lf = "    if next_idx < CHAMP_LAND_BUYS and CHAMP_LAND_DAY <= day <= 22:"
            if lf not in text:
                raise RuntimeError("land anchor not found")
            text = text.replace(
                lf, "    _lcap = sum(1 for _j, _d in LP_LAND.items() if day >= _d)\n"
                    "    if next_idx < _lcap and day <= 22:", 1)
            # 4. herd targets come from _champ_herd_targets, not the buy loop
            hf = ("def _champ_herd_targets(day, knobs=None):\n"
                  "    knobs = knobs or KNOB_DEFAULTS")
            if hf not in text:
                raise RuntimeError("herd anchor not found")
            text = text.replace(
                hf, hf + "\n    if day in LP_HERD or any(d <= day for d in LP_HERD):\n"
                         "        return {k: _lp_herd(day, k) "
                         "for k in ('GOOSE', 'COW', 'SHEEP')}", 1)
            k = text.index("CHAMP_HIRES = ")
            text = (text[:k] + INJECT.replace("__STAND__", repr(stand))
                    .replace("__HERD__", repr(herd)).replace("__LAND__", repr(land))
                    + "\n" + text[k:])
        out.append((mod, text))

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--mirror", type=int, default=5)
    e.add_argument("--eff", type=float, default=L.EFF_OURS)
    e.set_defaults(func=extract)
    x = sub.add_parser("export")
    x.add_argument("--name", default="lp-plan")
    x.set_defaults(func=export)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
