"""Does a strategy vector REACH the board? One game, censused day by day.

Memory, twice over: "trace ONE game to confirm a behavioural switch works
BEFORE launching an arm", and "always check a behaviour is reachable in plan
space, action head AND executor before believing a refutation" -- the goose
line was refuted while three independent gates held it at zero, so the
refutation was of nothing. A search over `kagent.theta` is only a search over
the coordinates that actually move a tile, and this is what says which do.

    PYTHONPATH=. uv run python tools/theta_probe.py                  # champion
    PYTHONPATH=. uv run python tools/theta_probe.py --set geese=6 \\
        --set animal_visit=1 --set coop_room=1

Prints, per day, what stands on our farm. A coordinate whose census column is
identical to the champion's is a coordinate the search cannot see.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def census(view):
    me = view.me
    animals, crops = {}, {}
    for _x, _y, t in me.animals():
        animals[t["animal"]] = animals.get(t["animal"], 0) + 1
    empty = {}
    for _x, _y, t in me.empty_structures():
        empty[t["kind"]] = empty.get(t["kind"], 0) + 1
    for _x, _y, t in me.plants():
        crops[t["crop"]] = crops.get(t["crop"], 0) + 1
    return {"animals": animals, "empty": empty, "crops": crops,
            "money": float(me.money), "hands": len(me.units) - 1,
            "quads": len(me.unlocked_quadrants)}


def play(theta, session_path=None, seed=7, side=0, script_side=0):
    """One game with `theta` on our side. -> (rows, result)."""
    from kaggle_environments import make
    from kagent import scheduler as _sched
    from kagent import theta as _theta
    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy, ThetaPlanProvider
    from rl import rollout

    spec = _theta.spec(theta)
    prev = dict(_sched.EXEC)
    _sched.reset_exec()
    _sched.set_exec(spec["exec"])

    mem = Memory()
    pol = HeuristicPolicy(ThetaPlanProvider(spec["knobs"]))
    rows = []
    seen = set()

    def learner(obs):
        view = View(obs)
        mem.observe(view)
        # Census at MIDDAY, not hour 0: hands are hired each morning and the
        # day-0 board is empty by construction, so an hour-0 census reports
        # zero hands all season and hides the `hires_delta` coordinate.
        if view.day not in seen and view.step % 24 >= 12:
            seen.add(view.day)
            rows.append({"day": view.day, **census(view)})
        return pol.act(view, mem)

    if session_path:
        s = rollout.load_session(session_path)
        seed = s["seed"]
        opponent = rollout._make_opponent(("session", session_path, script_side))
        side = 1 - script_side
    else:
        opponent = rollout._make_opponent("heuristic")

    agents = [learner, opponent] if side == 0 else [opponent, learner]
    try:
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run(agents)
    finally:
        _sched.EXEC.update(prev)
    final = env.steps[-1]
    mine, theirs = final[side].reward or 0.0, final[1 - side].reward or 0.0
    return rows, {"money": mine, "opp": theirs, "win": mine > theirs,
                  "statuses": [s.status for s in final]}


def _coerce(name, raw):
    from kagent import theta as _theta
    _lo, _hi, kind = _theta._bounds(name)
    if kind == "bool":
        return raw not in ("0", "false", "False", "")
    if kind == "int":
        return int(float(raw))
    return float(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], metavar="name=value")
    ap.add_argument("--vector", default=None, help="a theta JSON to load instead")
    ap.add_argument("--session", default=None, help="replay path for the opponent")
    ap.add_argument("--script-side", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", default="0,2,4,6,8,10,14,18,22,26,29")
    args = ap.parse_args()

    from kagent import theta as _theta
    t = _theta.champion()
    if args.vector:
        with open(args.vector) as fh:
            t.update(json.load(fh).get("knobs", {}) or json.load(open(args.vector)))
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in _theta.NAMES:
            raise SystemExit(f"unknown coordinate {k!r}; known: {_theta.NAMES}")
        t[k] = _coerce(k, v)

    print(f"theta: {_theta.describe(t)}")
    rows, res = play(t, session_path=args.session, seed=args.seed,
                     script_side=args.script_side)
    want = {int(d) for d in args.days.split(",")}
    print(f"\n{'day':>4}{'money':>9}{'hd':>4}{'qd':>3}  "
          f"{'GOOSE':>6}{'COW':>5}{'SHEEP':>6}{'coop':>6}{'past':>6}   crops")
    for r in rows:
        if r["day"] not in want:
            continue
        a, e = r["animals"], r["empty"]
        crops = " ".join(f"{k[:4].lower()}{v}" for k, v in sorted(r["crops"].items()))
        print(f"{r['day']:>4}{r['money']:>9,.0f}{r['hands']:>4}{r['quads']:>3}  "
              f"{a.get('GOOSE',0):>6}{a.get('COW',0):>5}{a.get('SHEEP',0):>6}"
              f"{e.get('COOP',0):>6}{e.get('PASTURE',0):>6}   {crops}")
    print(f"\nfinal {res['money']:,.0f} vs {res['opp']:,.0f}  "
          f"{'WIN' if res['win'] else 'loss'}  statuses {res['statuses']}")


if __name__ == "__main__":
    main()
