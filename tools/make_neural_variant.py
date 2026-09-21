"""Put an old trained planner head back on the FIXED executor.

Every neural planner this project trained was measured against the executor
that disobeys 11% of direct orders. `tools/exec_bc.py`'s ranker already
flipped from −$1,052 to +$1,233 when moved onto `h12-ufh`, so the same
question is open for the planner heads: `rl/runs/g10..g14e/best_weights.json`
were all scored on the broken dispatch and abandoned.

`rl/export.py` cannot be used for this -- it writes `kagent/weights.py` and
rebuilds from the dev tree, which is not the gated artifact. This patches the
frozen bundle's own `weights` module instead.

Flag combination matters. `policy.py` dispatches:

    if BANDIT_RULES:      -> KnobBanditPolicy over the champion plan
    elif CHAMPION_PLAN:   -> ChampionPlanProvider          <- never reaches the net
    elif _LAYERS:         -> NeuralPolicy(residual=RESIDUAL)

so a neural arm needs BANDIT_RULES None **and** CHAMPION_PLAN False. With
`--residual` the head emits offsets over `make_champion_plan` and the
SafeAgent fallback is the champion plan itself; without it the net plans from
scratch.

Because BANDIT_RULES has to be off, the honest baseline is `h13-weights`
(the same bundle with the rules off and no net), NOT `h12-ufh` -- otherwise
the arm is charged −$538 for losing the two conditional knob rules.

    PYTHONPATH=. uv run python tools/make_neural_variant.py \\
        --name n1-g14e --weights rl/runs/g14e/best_weights.json --residual
"""
import argparse
import ast
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def build(base, name, weights_json, residual, mpc):
    with open(weights_json) as f:
        data = json.load(f)
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    body = ('"""Neural planner weights (tools/make_neural_variant.py)."""\n\n'
            f"WEIGHTS_DATA = {data!r}\n"
            f"MPC_ENABLED = {bool(mpc)!r}\n"
            "CHAMPION_PLAN = False\n"      # else policy.py never reaches the net
            f"RESIDUAL = {bool(residual)!r}\n"
            "BANDIT_RULES = None\n")

    out, done = [], False
    for mod, text in mods:
        if mod == "weights":
            text, done = body, True
        out.append((mod, text))
    if not done:
        raise SystemExit(f"{base} has no weights module")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    open(os.path.join(dst_dir, "rules.json"), "w").write("{}")

    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)          # a bad blob raises HERE
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    layers = g["_G"].get("_LAYERS")
    if layers is None:
        raise SystemExit("weights decoded to None -- the net would never run")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  "
          f"residual={residual} mpc={mpc}  layers loaded OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="h13-weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--mpc", action="store_true")
    args = ap.parse_args()
    build(args.base, args.name, args.weights, args.residual, args.mpc)


if __name__ == "__main__":
    main()
