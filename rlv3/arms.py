"""One single-lever arm per flag, in the format the two existing instruments
ALREADY accept. Nothing in rl/ or tools/ is patched.

An arm file is champ-h10's own BANDIT_RULES dict (tools/rules/
rules_h8_ship2_hv.json) with `"exec"` extended by one rlv3 flag. Both
instruments read exactly that key:

    rl/session_eval.py  -> rl/rollout.run_episode, which does
                           `flags = spec.get("exec") or spec["rules"]["exec"]`
                           then `reset_exec(); set_exec(flags)`
    tools/agent_scorecard.py -> parse_spec("knobs:<path>") -> trace_game,
                           the same two lines

So `dev:knobs:rlv3/arms/<flag>.json` and `knobs:rlv3/arms/<flag>.json`
consume these unchanged -- PROVIDED `rlv3.executor.register()` has already
run in that process, or `set_exec` raises on the unknown flag. That is
`rlv3/sitecustomize.py`'s job and the reason every run line below carries
`KAGG_RLV3=1 PYTHONPATH=.:rlv3`.

PARAMS DO NOT TRAVEL IN THE JSON. Neither instrument reads a key it does not
know, so `rlv3_params` is written into the file for provenance only and the
values that actually take effect come from `KAGG_RLV3_PARAMS`, which is an
environment variable and therefore inherited by spawned workers.
`run_line()` prints both halves together so the two cannot drift.

WHAT TO EXPECT. `pl_wheat_adj` must be gated head-to-head against the
SHIPPED `layout` flag, which already does nearly this and measured +1.6 pts
(z 1.1, p 0.30, 6/6 bands positive) while RAISING the opponent's bank
$1,304 -- so `layout` is in ARMS as the reference arm. An rlv3 result that
looks like `layout`'s is not a result. `pl_pasture_ring` and `mv_dest` are
negative controls and are expected to lose.
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# champ-h10's shipped executor (tools/rules/rules_h8_ship2_hv.json)
BASE_EXEC = ("lastwater", "h23", "harvest_visit")
BASE_RULES_PATH = "tools/rules/rules_h8_ship2_hv.json"

ARMS = {
    # positive claims
    "pl_wheat_adj": ("pl_wheat_adj",),
    "pl_crop_cluster": ("pl_crop_cluster",),
    "pl_water_first": ("pl_water_first",),
    "co_chain_tile": ("co_chain_tile",),
    "co_animal_first": ("co_animal_first",),
    "co_value": ("co_value",),
    # the reference an rlv3 placement arm must beat, not `nothing`
    "layout_ref": ("layout",),
    # negative controls, expected to lose
    "pl_pasture_ring": ("pl_pasture_ring",),
    "mv_dest": ("mv_dest",),
    "pl_net": ("pl_net",),
    # the base itself, as the calibration arm
    "base": (),
}


def base_rules():
    with open(os.path.join(REPO, BASE_RULES_PATH)) as fh:
        return json.load(fh)


def write_arm(name, flags, params=None, out_dir="rlv3/arms"):
    """-> the absolute path of the arm JSON."""
    rules = base_rules()
    seen, merged = set(), []
    for f in tuple(BASE_EXEC) + tuple(flags):
        if f not in seen:
            seen.add(f)
            merged.append(f)
    rules["exec"] = merged
    if params:
        rules["rlv3_params"] = dict(params)   # provenance only; see docstring
    d = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(rules, fh)
        fh.write("\n")
    return path


def write_all(out_dir="rlv3/arms", params=None):
    return {n: write_arm(n, f, params, out_dir) for n, f in ARMS.items()}


def run_line(name, instrument="scorecard", params=None, games=6):
    """The exact command for one arm, env and all. Never hand-assemble it:
    a missing KAGG_RLV3=1 gives a clean run of the SHIPPED executor."""
    env = "KAGG_RLV3=1 PYTHONPATH=.:rlv3"
    if params:
        env += ' KAGG_RLV3_PARAMS="' + ",".join(f"{k}={v}" for k, v in params.items()) + '"'
    path = f"rlv3/arms/{name}.json"
    if instrument == "scorecard":
        return (f"cd {REPO} && {env} uv run python tools/agent_scorecard.py "
                f"--agent knobs:{path} --eval rl/runs/draw995/batch2.json:ship2_hv "
                f"--games {games}")
    return (f"cd {REPO} && {env} uv run python rl/session_eval.py "
            f"--arms dev:knobs:{path} --per-bin 1 --procs 2")


if __name__ == "__main__":
    for n, p in sorted(write_all().items()):
        print(f"{n:<18} {os.path.relpath(p, REPO)}")
    print()
    print(run_line("pl_wheat_adj"))
