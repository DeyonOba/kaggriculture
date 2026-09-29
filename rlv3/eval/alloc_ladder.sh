#!/bin/bash
# The rlv3 allocation dose ladder.
#
# READ FIRST: rl/runs/draw995/alloc.log. The CONSTANT version of this idea is
# running right now (rl/runs/draw995/run_alloc.sh) and its first arm is in:
#
#     ship2_hv   win 0.354 +/- 0.015      <- champ-h10, the reference
#     mix_mild   win 0.339 +/- 0.015      <- strawberry 26, carrot 8: -1.5 pts
#
# So a constant mix is already LOSING. If the constant mix loses and the
# adaptive one wins, the burden is to say which state variable the constant
# could not see, on the same worlds -- the water-fill trace this script prints
# is that evidence. If the adaptive one also loses, and loses MONOTONICALLY in
# lam, it is the LP allocation refutation again (a-lp33/67/100: -$9,937 /
# -$17,950 / -$31,728, win 0.319 -> 0.159) and the component dies at lam = 0
# cost, which is exactly why lam = 0 is bit-identical by construction.
#
#     bash rlv3/eval/alloc_ladder.sh            # build, identity-check, queue
#     bash rlv3/eval/alloc_ladder.sh --build    # build the bundles only
#     bash rlv3/eval/alloc_ladder.sh --check    # build + 2-game identity check
#
# NEVER raises --procs above 2 and NEVER starts training.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

OUT=rl/runs/draw995
BUILD=rlv3/eval/build
REF=arena/agents/champ-h10/main.py
LADDER="0 0.33 0.67 1.0"
MODE="${1:-all}"

mkdir -p "$BUILD"

# --- build one bundle per dose ------------------------------------------------
# rlv3 modules are exec'd into a SECOND namespace layered over kagent's, not
# into the same flat dict tools/build_submission.py uses. They have to be:
# rlv3.rent defines `sell_revenue` and rlv3.allocation defines `apply_to_plan`
# and `capacity`, every one of which collides with a kagent name that the flat
# bundle resolves by bare lookup. A copied dict keeps kagent's own globals
# intact while letting rlv3 see them.
uv run python - "$BUILD" $LADDER <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from tools.build_submission import MODULES, TEMPLATE, _strip_intra_imports  # read-only

build, doses = sys.argv[1], [float(x) for x in sys.argv[2:]]
REPO = os.getcwd()
RLV3 = ["rent", "land", "allocation"]        # dependency order; land before allocation

def src(path):
    with open(path) as fh:
        return _strip_intra_imports(fh.read().replace("from rlv3 import rent", "")
                                    .replace("from rlv3.land import LandDecider, observe_dawn", ""))

kagent = [(m, _strip_intra_imports(open(os.path.join(REPO, "kagent", m + ".py")).read()))
          for m in MODULES]
rlv3 = [(m, src(os.path.join(REPO, "rlv3", m + ".py"))) for m in RLV3]
params = json.load(open(os.path.join(REPO, "rlv3", "params", "allocation_v0.json")))
EXEC = ["lastwater", "h23", "harvest_visit"]   # champ-h10's EXEC_RULES
rules = {"kind": "knobs", "min_gain": 1500, "max_knobs": 5, "models": [], "conditions": [
    {"knob": "premium_scale", "value": 0.35},
    {"knob": "carrot_day", "value": 25},
    {"knob": "cows", "value": 6, "max_consumers": {"MILK": 0}},
    {"knob": "sheep", "value": 10, "day": 12, "min_price": {"WOOL": 1.0}}]}

TAIL = '''
# champ-h10 ships these in its patched kagent/weights.py; the dev tree's
# weights.py is empty, so a bundle built from kagent/ has the executor flags
# OFF and the v1 head set. Without both lines lam=0 is NOT champ-h10 -- the
# first 2-game identity check came out $11,638 apart on exactly this.
_G["set_exec"](_EXEC)
_G["set_mode"]("residual")
_R = dict(_G)
for _name, _src in _RLV3_SOURCES:
    exec(compile(_src, "rlv3/" + _name + ".py", "exec"), _R)
_RNS = _types.SimpleNamespace(**_R)
for _name, _ in _RLV3_SOURCES:
    _R[_name] = _RNS
_PROV = _R["AllocationPlanProvider"](knobs=_RULES, params=_PARAMS, lam=_LAM)
_A = _G["SafeAgent"](_G["HeuristicPolicy"](_PROV),
                     fallback=_G["HeuristicPolicy"](_G["ChampionPlanProvider"]()))


def agent(observation, configuration=None):
    return _A.act_safe(observation, configuration)
'''

for lam in doses:
    body = TEMPLATE.format(sources="[\n" + "".join(
        f"    ({n!r}, {t!r}),\n" for n, t in kagent) + "]")
    body = body.rsplit("_IMPL = ", 1)[0]
    head = ("_RLV3_SOURCES = [\n" + "".join(f"    ({n!r}, {t!r}),\n" for n, t in rlv3) + "]\n"
            f"_PARAMS = {params!r}\n_RULES = {rules!r}\n_LAM = {lam!r}\n"
            f"_EXEC = {EXEC!r}\n")
    tag = "lam%03d" % round(lam * 100)
    d = os.path.join(build, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "main.py"), "w") as fh:
        fh.write(body + head + TAIL)
    print(f"built {d}/main.py  lam={lam}")
PY
[ "$MODE" = "--build" ] && exit 0

# --- (a) IDENTITY FIRST, 2 games, zero risk -----------------------------------
# lam = 0 must be BIT-IDENTICAL to the baseline the eval actually pairs
# against. That baseline is ship2_hv = the DEV TREE with champ-h10's four
# knobs and its three executor flags -- NOT arena/agents/champ-h10/main.py,
# which is a frozen snapshot of an older kagent/ and finishes $11,638 apart
# from today's dev tree on world 770000. Comparing against the wrong one is
# how this check first came out FAIL.
#
# Exact final-money equality on one world in both seats is a far stronger
# statement than the h2h "margin 0", which any deterministic agent playing
# itself satisfies.
echo "[$(date +%H:%M)] identity check: lam0 vs dev ship2_hv, 2 games"
uv run python - <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from rl import rollout
REF = "arena/agents/champ-h10/main.py"
rules = json.load(open("tools/rules/rules_h8_ship2_hv.json"))
ok = True
for side in (0, 1):
    job = {"seed": 770000, "side": side, "temperature": 0.0, "opponent": ("bundle", REF)}
    base = rollout.run_episode(dict(job, learner_policy={"kind": "knobs", "rules": rules}))
    arm = rollout.run_episode(dict(job, learner_bundle="rlv3/eval/build/lam000/main.py"))
    same = (base["final_money"], base["opp_money"]) == (arm["final_money"], arm["opp_money"])
    ok = ok and same
    print(f"  seat {side}: ship2_hv {base['final_money']:.0f}/{base['opp_money']:.0f}  "
          f"lam0 {arm['final_money']:.0f}/{arm['opp_money']:.0f}  "
          f"{'IDENTICAL' if same else 'DIFFERENT'}")
print("IDENTITY:", "PASS" if ok else "FAIL -- fix this before trusting any dose")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && exit 1
[ "$MODE" = "--check" ] && exit 0

# --- the water-fill trace: the mechanism check --------------------------------
echo
echo "[$(date +%H:%M)] water-fill trace at lam=1.0 (which crop, at what marginal)"
uv run python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "rlv3", "tests"))
import test_allocation as T
from kagent import planner
for lbl, st in (
    ("day 6", dict(day=6, money=9000, quads=("NW", "NE"), crops={"MELON": 12, "WHEAT": 7},
                   animals={"COW": 4, "SHEEP": 2})),
    ("day 12", dict(day=12, money=30000, quads=("NW", "NE", "SW"),
                    crops={"STRAWBERRY": 33, "WHEAT": 15}, animals={"COW": 6, "SHEEP": 8},
                    opp_crops={"STRAWBERRY": 21, "WHEAT": 25})),
    ("day 20", dict(day=20, money=60000, quads=("NW", "NE", "SW"),
                    crops={"STRAWBERRY": 41, "WHEAT": 20, "CARROT": 1},
                    animals={"COW": 6, "SHEEP": 10})),
):
    ref, got, a, _v = T._pair(st, 1.0)
    print(f"\n{lbl}  stance={a.trace['stance']} lead={a.trace['lead']:.0f} "
          f"q={a.trace['quantile']} labour=${a.trace['labour_price']}/turn "
          f"cap={a.crop_capacity} herd_cap={a.herd_capacity}")
    print(f"  champ-h10 quota {ref.plant_quota}  buys {ref.animal_buys}  land {ref.land_buys}")
    print(f"  rlv3      quota {got.plant_quota}  buys {got.animal_buys}  land {got.land_buys}")
    print(f"  targets   {a.crop_targets}  herd {a.herd_targets}")
    print(f"  caps      {a.trace['crop_caps']}  dropped {a.trace['dropped']}")
    print(f"  fill      {a.trace['fill_crop'][:8]} ... pass2@{a.trace['fill_pass2_from']}")
    assert got.hires_target == ref.hires_target, "HIRING MOVED -- run is void"
PY

# --- (b) THE DOSE LADDER, one block, reusing the standing draw ----------------
echo
while pgrep -f 'rl/session_eval.py' >/dev/null; do
  echo "[$(date +%H:%M)] waiting on the running session_eval ..."; sleep 120
done
echo "[$(date +%H:%M)] rlv3 allocation dose ladder ..."
uv run python -u rl/session_eval.py \
  --arm lam33=bundle:$BUILD/lam033/main.py \
  --arm lam67=bundle:$BUILD/lam067/main.py \
  --arm lam100=bundle:$BUILD/lam100/main.py \
  --reuse ship2_hv="$OUT/batch2.json:ship2_hv" \
  --baseline ship2_hv --per-bin 175 --procs 2 \
  --out "$OUT/rlv3_alloc.json" 2>&1 | tee "$OUT/rlv3_alloc.log"

# --- the scorecard: the per-part judge ----------------------------------------
echo
echo "[$(date +%H:%M)] scorecard for the best dose"
uv run python tools/agent_scorecard.py \
  --agent "bundle:$BUILD/lam100/main.py" \
  --eval "$OUT/rlv3_alloc.json:lam100" \
  --reference "bundle:$REF" --reference-eval "$OUT/batch2.json:ship2_hv" \
  --games 6 --out "$OUT/rlv3_alloc_scorecard.json"
echo "[$(date +%H:%M)] done."
