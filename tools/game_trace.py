"""Trace ONE game step by step, for the HTML replay viewer (tools/game_viewer.py).

Plays our agent (a `rl.common.to_layers` weights file, or the heuristic when
none is given) against a recorded session tape on the tape's true world --
the same call the paired eval makes, temperature 0, so the game reproduces
the eval's row exactly -- and records, at every one of the 720 steps:

    both farms      every tile (compact codes), units, money, quadrants
    market          prices and inventory for the nine products, shops open
    actions         what each side did this step (farmer, hands, market)
    our private     shed, seeds
    signals         what the TRAINER sees: portfolio value, asset lead, money
                    lead, the phase potential Phi(s, day); and what the day
                    planner chose at dawn (head -> value)
    counts          plants / animals / weeds / empty / locked per farm

Tile codes (JSON arrays, ints):
    0                          locked
    1                          empty, unlocked
    2                          weed
    [3, s]                     empty structure, s: 0 coop, 1 pasture
    [4, c, age, yield, w, u]   plant: crop index, age, yield units, watered
                               today, consecutive unwatered
    [5, a, fed, cared, fert, yield]  animal: index, fed today, cared today,
                               fertilizer available, yield units

    PYTHONPATH=. uv run python tools/game_trace.py \\
        --weights rl/runs/stage_b2/control/eval.json \\
        --session rl/datasets/sessions/session-XXXX.json --script-side 0 \\
        --out /tmp/game.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS_L = ["GOOSE", "COW", "SHEEP"]
PRODUCTS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]


def tile_code(c, day):
    if c == "LOCKED":
        return 0
    if c is None:
        return 1
    if not isinstance(c, dict):
        return 1
    kind = c.get("kind")
    if kind == "WEED":
        return 2
    if kind == "PLANT":
        crop = c.get("crop")
        return [4, CROPS_L.index(crop) if crop in CROPS_L else -1,
                int(day - c.get("planted_day", day)), int(c.get("yield_units", 0)),
                1 if c.get("watered_today") else 0, int(c.get("consecutive_unwatered", 0))]
    if "animal" in c:
        a = c.get("animal")
        return [5, ANIMALS_L.index(a) if a in ANIMALS_L else -1,
                1 if c.get("fed_today") else 0, 1 if c.get("cared_today") else 0,
                1 if c.get("fertilizer_available") else 0, int(c.get("yield_units", 0))]
    return [3, 1 if str(kind).upper().startswith("PAST") else 0]


def farm_record(farm, day):
    tiles = [[tile_code(c, day) for c in row] for row in farm["tiles"]]
    flat = [t for row in tiles for t in row]
    counts = {
        "plants": sum(1 for t in flat if isinstance(t, list) and t[0] == 4),
        "animals": sum(1 for t in flat if isinstance(t, list) and t[0] == 5),
        "weeds": sum(1 for t in flat if t == 2),
        "empty": sum(1 for t in flat if t == 1),
        "locked": sum(1 for t in flat if t == 0),
        "structs": sum(1 for t in flat if isinstance(t, list) and t[0] == 3),
    }
    return {
        "tiles": tiles,
        "farmer": list(farm.get("farmer") or [0, 0]),
        "hands": [list(h) for h in (farm.get("hands") or [])],
        "money": float(farm.get("money", 0)),
        "quadrants": sorted(farm.get("unlocked_quadrants") or ["NW"]),
        "counts": counts,
    }


def _base_prices():
    from kagent.constants import MARKET_PARAMS
    return [float(MARKET_PARAMS[p]["base"]) for p in PRODUCTS_L]


def _plan_choices(choices):
    """head name -> chosen value, from the planner's choice indices."""
    from kagent import features
    out = {}
    for (name, vals), idx in zip(features.HEADS, choices):
        v = vals[idx] if 0 <= idx < len(vals) else None
        out[name] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=None, help="to_layers JSON; omit for the heuristic")
    ap.add_argument("--champion", action="store_true",
                    help="play the dev tree's CHAMPION plan (speculative=True), "
                         "the agent the bundler ships, instead of the plain heuristic")
    ap.add_argument("--rules", default=None,
                    help="knob-bandit rules JSON (a dev:knobs arm, e.g. tools/rules/rules_h8.json)")
    ap.add_argument("--exec", default=None,
                    help="executor rule flags, comma-separated (kagent/scheduler.EXEC_FLAGS)")
    ap.add_argument("--execnet", default=None,
                    help="learned executor: <weights.pt>[:<rules.json>[:<confidence>]] (rl/execnet)")
    ap.add_argument("--session", required=True)
    ap.add_argument("--script-side", type=int, required=True, help="the tape's seat")
    ap.add_argument("--label", default=None, help="name shown in the viewer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kaggle_environments import make

    from kagent import scheduler
    from kagent.gamestate import View
    from rl import rollout
    from rl.phase_potential import ingredients, phase_potential

    if args.exec:
        scheduler.set_exec(args.exec)

    session = rollout.load_session(args.session)
    seed = session["seed"]
    ss = args.script_side
    side = 1 - ss
    layers = rollout._load_layers(args.weights) if args.weights else None
    if layers and layers["policy"].get("board") is not None and not rollout._BOARD:
        raise SystemExit("weights carry a board trunk: set KAGG_BOARD=1")

    records = []
    if args.execnet:
        fields = args.execnet.split(":")
        spec = {"kind": "execnet", "weights": fields[0]}
        if len(fields) > 1 and fields[1]:
            with open(fields[1]) as fh:
                spec["rules"] = json.load(fh)
            if spec["rules"].get("exec") and not args.exec:
                scheduler.set_exec(spec["rules"]["exec"])
        if len(fields) > 2 and fields[2]:
            spec["conf"] = float(fields[2])
        learner, _ = rollout._make_dev_policy_agent(spec)
    elif args.rules:
        with open(args.rules) as fh:
            rules = json.load(fh)
        if rules.get("exec") and not args.exec:
            scheduler.set_exec(rules["exec"])      # the same flags the eval arm applies
        learner, _ = rollout._make_dev_policy_agent({"kind": "knobs", "rules": rules})
    elif args.champion:
        learner, _ = rollout._make_dev_policy_agent({"kind": "champion"})
    else:
        learner = rollout._make_learner_agent(layers, 0.0, None, records, "rl")
    tape = rollout.make_script_agent(session["actions"][ss])

    seen = {0: [], 1: []}          # (obs, action) per step per seat

    def wrap(fn, seat):
        def g(obs):
            a = fn(obs)
            seen[seat].append((obs, a))
            return a
        return g

    agents = [None, None]
    agents[side] = wrap(learner, side)
    agents[ss] = wrap(tape, ss)
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run(agents)
    final = env.steps[-1]
    my_money = final[side].reward or 0.0
    opp_money = final[ss].reward or 0.0

    # ratings from the index, when present
    rating = [None, None]
    try:
        with open(os.path.join(REPO, "rl", "datasets", "session_index.json")) as fh:
            idx = json.load(fh)
        rel = args.session[len(REPO) + 1:] if args.session.startswith(REPO) else args.session
        for v in idx.values():
            if v.get("path") in (rel, args.session):
                rating = v.get("scores") or [None, None]
                break
    except Exception:
        pass

    n = min(len(seen[0]), len(seen[1]))
    steps = []
    dawn_plans = {}
    for r in records:
        d = int(r["lead"]["day"]) if "lead" in r else None
        if d is not None:
            dawn_plans[d] = _plan_choices(r["choices"])
    for t in range(n):
        obs_me, act_me = seen[side][t]
        obs_opp, act_opp = seen[ss][t]
        day = int(obs_me.get("day", t // 24))
        hour = int(obs_me.get("hour", t % 24))
        view = View(obs_me)
        lead = ingredients(view)
        phi = phase_potential(lead["asset_lead"], lead["money_lead"], lead["day"])
        market = obs_me.get("market") or {}
        px = market.get("prices") or {}
        inv = market.get("inventory") or {}
        private = obs_me.get("private") or {}
        farms = obs_me["farms"]
        steps.append({
            "t": t, "day": day, "hour": hour,
            "me": farm_record(farms[side], day),
            "opp": farm_record(farms[ss], day),
            "market": {"prices": [round(float(px.get(p, 0)), 2) for p in PRODUCTS_L],
                       "inv": [int(inv.get(p, 0)) for p in PRODUCTS_L],
                       "shops": list((obs_me.get("town") or {}).get("unlocked_shops", []))},
            "shed": {k: int(v) for k, v in (private.get("shed") or {}).items() if v},
            "seeds": {k: int(v) for k, v in (private.get("seeds") or {}).items() if v},
            "act": {"me": act_me, "opp": act_opp},
            "sig": {"pv": round(rollout.portfolio_value(view)),
                    "asset_lead": round(lead["asset_lead"]),
                    "money_lead": round(lead["money_lead"]),
                    "phi": round(phi, 3)},
        })
    trace = {
        "label": args.label or os.path.basename(args.session),
        "session": os.path.basename(args.session), "seed": seed,
        "our_seat": side, "tape_seat": ss,
        "ratings": rating, "weights": args.weights,
        "final": {"me": my_money, "opp": opp_money,
                  "win": my_money > opp_money, "margin": my_money - opp_money},
        "products": PRODUCTS_L, "crops": CROPS_L, "animals": ANIMALS_L,
        "base_prices": _base_prices(),
        "plans": {str(d): p for d, p in sorted(dawn_plans.items())},
        "steps": steps,
    }
    with open(args.out, "w") as fh:
        json.dump(trace, fh, separators=(",", ":"))
    print(f"{len(steps)} steps -> {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)  "
          f"final me ${my_money:,.0f} vs opp ${opp_money:,.0f}  "
          f"{'WIN' if my_money > opp_money else 'loss'} by ${my_money - opp_money:+,.0f}")


if __name__ == "__main__":
    main()
