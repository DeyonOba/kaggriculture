"""Plan vs execution, day by day: what the dawn plan ASKED and what the
executor DELIVERED by the next dawn, on one tape world.

    PYTHONPATH=. uv run python tools/plan_ledger.py --agent champion --world 0 --days 0-8
    PYTHONPATH=. uv run python tools/plan_ledger.py --agent readout --world 0
    PYTHONPATH=. uv run python tools/plan_ledger.py --agent "readout:readout=rl/runs/headfix/readout_v5_board_e2e/bc.pt"

`--agent champion` is the dev-tree champion plan + the shipped rules (h11);
anything else is an rlv3 arm string (rlv3/arm.py grammar). The ASK is the
plan as the provider returned it at hour 0 (before the market builder and the
scheduler consume it); the DELIVERY is counted from the engine's own step
records for our seat: PLANT by crop, BUILD_*, PLACE (animals actually put on
a structure), hands at the next dawn, FERTILIZE. Money is at dawn.
"""
from __future__ import annotations
import argparse
import collections
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def snapshot(plan):
    return {"plant": dict(plan.plant_quota or {}), "buys": dict(plan.animal_buys or {}),
            "builds": dict(plan.structure_builds or {}), "hires": int(plan.hires_target or 0),
            "fert": plan.fert_budget, "liq": bool(plan.liquidation)}


def build_agent(spec_text):
    from kagent import weights as W
    from kagent.gamestate import Memory, View
    from kagent.policy import HeuristicPolicy, ChampionPlanProvider
    from rl import rollout
    asks = {}
    if spec_text == "champion" or spec_text.startswith("champion:"):
        from kagent.policy import ThetaPlanProvider
        rollout._set_exec(list(W.EXEC_RULES))
        knobs = {}
        for kv in spec_text.partition(":")[2].split(","):
            if "=" in kv:
                k, v = kv.split("="); knobs[k.strip()] = float(v) if "." in v else int(v)
        provider = ThetaPlanProvider(knobs) if knobs else ChampionPlanProvider()
        pol, mem = HeuristicPolicy(provider), Memory()
        inner = provider.make_plan

        def rec(view, memory):
            plan = inner(view, memory); asks[view.day] = snapshot(plan); return plan
        provider.make_plan = rec

        def agent_fn(obs):
            view = View(obs); mem.observe(view); return pol.act(view, mem)
        return agent_fn, asks
    from rlv3 import arm as ARM
    from rlv3.agent import make_dev_agent
    spec = ARM.parse(spec_text)
    rollout._set_exec(spec["exec"])
    agent_fn, _ = make_dev_agent(spec)
    provider = agent_fn.provider
    inner = provider.make_plan

    def rec(view, memory):
        plan = inner(view, memory); asks[view.day] = snapshot(plan); return plan
    provider.make_plan = rec
    return agent_fn, asks


def play(spec_text, world, seed=7, per_bin=24):
    from kaggle_environments import make
    from rl import rollout, session_eval as SE
    entries = SE.held_out_entries(os.path.join(REPO, "rl", "datasets", "session_index.json"), seed, per_bin)
    p, ss, _b, _r = entries[world]
    sess = rollout.load_session(os.path.join(REPO, p))
    agent_fn, asks = build_agent(spec_text)
    opp = rollout._make_opponent(("session", os.path.join(REPO, p), ss))
    side = 1 - ss
    env = make("kaggriculture", configuration={"seed": sess["seed"]}, debug=False)
    env.run([agent_fn, opp] if side == 0 else [opp, agent_fn])
    return env.steps, side, asks, os.path.basename(p)


def ledger(steps, side, asks, days):
    rows = []
    for d in days:
        dawn = steps[d * 24][side]["observation"]["farms"][side]
        nxt = steps[min((d + 1) * 24, len(steps) - 1)][side]["observation"]["farms"][side]
        done = collections.Counter()
        for t in range(d * 24, min((d + 1) * 24, len(steps))):
            a = steps[t][side].get("action") or {}
            for u in [a.get("farmer") or []] + list(a.get("hands") or []):
                if not u:
                    continue
                if u[0] == "PLANT" and len(u) > 1:
                    done["plant:" + str(u[1])] += 1
                elif u[0] in ("BUILD_PASTURE", "BUILD_COOP"):
                    done[u[0]] += 1
                elif u[0] == "PLACE" and len(u) > 1:
                    done["place:" + str(u[1])] += 1
                elif u[0] in ("FERTILIZE", "HARVEST", "WATER"):
                    done[u[0]] += 1
            for m in a.get("market") or []:
                if m and m[0] == "BUY_ANIMAL" and len(m) > 2:
                    done["buyreq:" + str(m[1])] += int(m[2])
                if m and m[0] == "HIRE" and len(m) > 1:
                    done["hirereq"] += int(m[1]) if isinstance(m[1], int) else 1
        ask = asks.get(d) or {"plant": {}, "buys": {}, "builds": {}, "hires": 0, "fert": None, "liq": False}
        # hands are day labourers hired at hour 0: count them at hour 1
        h1 = steps[min(d * 24 + 1, len(steps) - 1)][side]["observation"]["farms"][side]
        hands0, hands1 = len(dawn.get("hands") or []), len(h1.get("hands") or [])
        # what the ask would COST at dawn prices, and the ground it needs
        from kagent import econ
        from kagent.constants import CROPS, ANIMALS
        cost = (sum(CROPS[c]["seed"] * n for c, n in ask["plant"].items() if c in CROPS)
                + sum(ANIMALS[k]["cost"] * n for k, n in ask["buys"].items() if k in ANIMALS)
                + econ.hire_total_cost(ask["hires"]))
        # the engine encodes a bare tile as None (locked tiles are not in the farm's rows as None either: they carry a kind)
        bare = sum(1 for row in dawn["tiles"] for c in row if c is None) if dawn.get("tiles") else None
        rows.append({"day": d, "money": dawn.get("money", 0), "ask": ask, "done": done,
                     "hands0": hands0, "hands1": hands1, "cost": cost, "bare": bare})
    return rows


def pct(got, ask):
    return None if ask <= 0 else min(1.0, got / ask)


def report(label, world_name, rows):
    print(f"\n=== {label} on {world_name} ===")
    print(f"{'day':>3s} {'money':>7s} {'ask$':>6s} {'bare':>4s} | {'PLANT ask -> done':>34s} | {'ANIMALS ask -> placed':>26s} | {'BUILD ask -> done':>18s} | {'HANDS target->hired':>19s} | {'FERT':>10s} | {'done%':>5s}")
    tot_ask = tot_done = 0
    for r in rows:
        a, dn = r["ask"], r["done"]
        pa = ", ".join(f"{c[:3]}{n}->{dn['plant:' + c]}" for c, n in a["plant"].items()) or "-"
        an = ", ".join(f"{k[:2]}{n}->{dn['place:' + k]}" for k, n in a["buys"].items()) or "-"
        bd = ", ".join(f"{s[:3]}{n}->{dn['BUILD_' + s]}" for s, n in a["builds"].items() if n) or "-"
        hire_ask = a["hires"]; hire_got = r["hands1"]
        hd = f"{a['hires']}->{r['hands1']}"
        fa = a["fert"]; fd = f"{fa if fa is not None else 'gate'}->{dn['FERTILIZE']}"
        ask_n = sum(a["plant"].values()) + sum(a["buys"].values()) + sum(a["builds"].values()) + hire_ask
        done_n = sum(min(dn["plant:" + c], n) for c, n in a["plant"].items()) + sum(min(dn["place:" + k], n) for k, n in a["buys"].items()) \
            + sum(min(dn["BUILD_" + s], n) for s, n in a["builds"].items()) + min(hire_got, hire_ask)
        tot_ask += ask_n; tot_done += done_n
        pc = f"{100 * done_n / ask_n:5.0f}%" if ask_n else "  -  "
        bare = "-" if r["bare"] is None else str(r["bare"])
        print(f"{r['day']:3d} {r['money']:7,.0f} {r['cost']:6,.0f} {bare:>4s} | {pa:>34s} | {an:>26s} | {bd:>18s} | {hd:>19s} | {fd:>10s} | {pc}")
    print(f"  tasks asked {tot_ask}, delivered {tot_done} -> {100 * tot_done / max(1, tot_ask):.0f}% (plants + animals placed + builds + hands); "
          f"ask$ = seeds + animals + hire bill at list prices vs money at dawn; bare = empty tiles at dawn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", action="append", required=True, help="champion, or an rlv3 arm string; repeatable")
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--days", default="0-8")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.days.split("-"))
    for spec in args.agent:
        steps, side, asks, name = play(spec, args.world)
        report(spec, f"{name} seat {side}", ledger(steps, side, asks, range(lo, hi + 1)))


if __name__ == "__main__":
    main()
