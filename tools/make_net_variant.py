"""S4: replace the arithmetic opponent-supply term with BoardNet's prediction.

`_rent_inventory` projects the opponent's future supply by running the engine's
yield rules over their visible board. S3 measured the ceiling on improving that
term -- perfect knowledge of their actual market flow takes k=5 forecast MAE
from 13.8 to 11.3 -- and BoardNet closes 67% of it.

This wires the model into that exact slot and nothing else, so the gate
measures the model and not a second change.

Three things make it safe rather than clever:

* The encoder is the TRAINING source inlined verbatim, fed `memory.last_raw_obs`
  (stashed by policy.act_safe), so there is no train/serve skew to introduce.
* The forward pass is numpy at 0.3 ms; 30 days costs 0.01 s of the 60 s season
  overage bank. Weights are float16 base64 literals because submit.sh forbids
  `open(`.
* Any exception falls back to the arithmetic projection, so the worst case is
  r-mrg100 exactly.

The model predicts 10 days and _RENT_HORIZON is 14, so days 11-14 keep the
arithmetic term.

    PYTHONPATH=. uv run python tools/make_net_variant.py \
        --name r-net-e --net /tmp/net_E.py
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)'''

REPL = '''    mine = _rent_board_supply(view.me.tiles, day, horizon)
    opp = _rent_board_supply(view.opp.tiles, day, horizon)
    _no = _net_supply()
    if _no is not None:
        # BoardNet for the days it was trained on; arithmetic beyond.
        for _k in range(1, min(horizon, len(_no) - 1) + 1):
            opp[_k] = _no[_k]'''

CALL_ANCHOR = "    knobs = _knobs(memory)\n"
CALL_REPL = ('    knobs = _knobs(memory)\n'
             '    _NET_OBS["o"] = getattr(memory, "last_raw_obs", None)\n')

GLUE = '''

# --- BoardNet wiring --------------------------------------------------------
# One forward per DAY, cached: _rent_pick may ask for the inventory path more
# than once inside a day and the board has not moved between those calls.
_NET_OBS = {"o": None}
_NET_CACHE = {"day": -2, "flow": None}


def _net_supply():
    o = _NET_OBS.get("o")
    if o is None:
        return None
    try:
        d = o.get("day", -1)
    except Exception:
        return None
    if _NET_CACHE["day"] != d:
        _NET_CACHE["day"] = d
        try:
            _NET_CACHE["flow"] = _net_opp_flow(o)
        except Exception:
            # Fall back to the arithmetic projection -- worst case is r-mrg100.
            _NET_CACHE["flow"] = None
    return _NET_CACHE["flow"]

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--net", required=True, help="module from export_boardnet.py")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    net_src = open(args.net).read()

    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text or CALL_ANCHOR not in text:
                raise SystemExit("rent anchors not found; is the base r-mrg100?")
            text = text.replace(ANCHOR, REPL, 1)
            text = text.replace(CALL_ANCHOR, CALL_REPL, 1)
            text = GLUE + text
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no planner module in bundle")
    # the net module must exec BEFORE the planner references its globals at
    # call time; order only matters for module-level code, but keep it early.
    out.insert(1, ("boardnet", net_src))

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + BoardNet opponent supply "
          f"({os.path.getsize(p)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
