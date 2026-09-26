"""Characterise a WHOLE agent, on every dimension we can measure, beside the
three leaderboard players and our own best agent.

Eleven rule variants were judged on a single number (paired win rate) on
2026-09-26 and almost all were refuted. A win rate says an arm lost; it does
not say which part of the agent is the limiting one. This prints four groups
-- outcome, planner, market/demand, executor -- for the agent under test, for
a reference agent (champ-h10 by default), and for M & M & P & Q / Boey / DSM,
and closes with a gap line per group saying how much of the distance from the
reference to the leaders the agent has closed.

    PYTHONPATH=. uv run python tools/agent_scorecard.py \\
        --agent bundle:arena/agents/champ-h10/main.py \\
        --eval rl/runs/draw995/batch2.json:ship2_hv --games 6

Agent specs: champion | knobs:<rules.json> | bundle:<main.py> |
weights:<to_layers.json> | execnet:<pt>[:<planner>[:<conf>[:<mode>]]].

Nothing here re-runs a paired eval: `--eval` attaches one that already
exists under rl/runs/draw995/. Traces are played fresh against recorded
sessions, exactly as tools/game_trace.py does, and fed to the existing
tools/tile_metrics.py and tools/sell_metrics.py analysers.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import tile_metrics as TM          # noqa: E402
import sell_metrics as SM          # noqa: E402
from game_trace import farm_record, CROPS_L, ANIMALS_L, PRODUCTS_L, _base_prices  # noqa: E402

LEADERS = (("mmpq", "M & M & P & Q", "replays/mmpq/*.json"),
           ("boey", "Boey", "replays/boey/*.json"),
           ("dsm", "DSM", "replays/dsm/*.json"))
TOP3_JSON = os.path.join(REPO, "docs", "results", "top3_tile_metrics.json")
BANDS = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]
CRASH_RATIO = 0.35        # a price below this share of base is a crashed market


# ------------------------------------------------------------------ specs
def parse_spec(text):
    """'champion' | 'knobs:<json>' | 'bundle:<path>' | 'weights:<path>' |
    'execnet:<pt>[:...]' -> a dict the trace runner understands."""
    if not text or text == "champion":
        return {"kind": "champion"}
    head, _, rest = text.partition(":")
    if head == "bundle":
        return {"kind": "bundle", "path": os.path.abspath(rest)}
    if head == "weights":
        return {"kind": "weights", "path": os.path.abspath(rest)}
    if head == "knobs":
        with open(rest) as fh:
            return {"kind": "knobs", "rules": json.load(fh)}
    if head == "execnet":
        from rl.execnet.policy import parse_arm_fields
        return parse_arm_fields(rest.split(":"))
    raise ValueError(f"unknown agent spec {text!r}")


def spec_label(spec):
    k = spec["kind"]
    if k == "bundle":
        return os.path.basename(os.path.dirname(spec["path"]))
    if k == "weights":
        return os.path.basename(os.path.dirname(spec["path"])) or "weights"
    if k == "execnet":
        return "execnet"
    return k


# ------------------------------------------------------------- trace play
def _build_agent(spec):
    """-> (agent_fn, restore_fn). Mirrors tools/game_trace.py's wiring so a
    scorecard trace is the same game the paired eval would play."""
    from rl import rollout
    if spec["kind"] == "bundle":
        return rollout._make_opponent(("bundle", spec["path"])), (lambda: None)
    if spec["kind"] == "weights":
        layers = rollout._load_layers(spec["path"])
        if layers["policy"].get("board") is not None and not rollout._BOARD:
            raise SystemExit("those weights carry a board trunk: set KAGG_BOARD=1")
        return rollout._make_learner_agent(layers, 0.0, None, [], "rl"), (lambda: None)
    fn, switched = rollout._make_dev_policy_agent(spec)

    def restore():
        if switched:
            from kagent import features
            features.set_mode("absolute")
    return fn, restore


def trace_game(spec, session_path, script_side, out_path):
    """Play `spec` against the recorded seat on that game's true world and
    write the trace tools/tile_metrics.py and tools/sell_metrics.py read."""
    from kaggle_environments import make
    from kagent import scheduler
    from kagent.gamestate import View
    from rl import rollout

    flags = spec.get("exec") or (spec.get("rules") or {}).get("exec")
    prev_exec = dict(scheduler.EXEC)
    if flags:
        scheduler.reset_exec()
        scheduler.set_exec(flags)
    learner, restore = _build_agent(spec)
    session = rollout.load_session(session_path)
    ss, side = script_side, 1 - script_side
    tape = rollout.make_script_agent(session["actions"][ss])
    seen = {0: [], 1: []}

    def wrap(fn, seat):
        def g(obs):
            a = fn(obs)
            seen[seat].append((obs, a))
            return a
        return g

    agents = [None, None]
    agents[side] = wrap(learner, side)
    agents[ss] = wrap(tape, ss)
    env = make("kaggriculture", configuration={"seed": session["seed"]}, debug=False)
    try:
        env.run(agents)
    finally:
        restore()
        scheduler.EXEC.update(prev_exec)
    last = env.steps[-1]
    me, opp = last[side].reward or 0.0, last[ss].reward or 0.0

    steps = []
    for t in range(min(len(seen[0]), len(seen[1]))):
        obs, act_me = seen[side][t]
        _, act_opp = seen[ss][t]
        day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
        market = obs.get("market") or {}
        px, inv = market.get("prices") or {}, market.get("inventory") or {}
        private = obs.get("private") or {}
        steps.append({
            "t": t, "day": day, "hour": hour,
            "me": farm_record(obs["farms"][side], day),
            "opp": farm_record(obs["farms"][ss], day),
            "market": {"prices": [round(float(px.get(p, 0)), 2) for p in PRODUCTS_L],
                       "inv": [int(inv.get(p, 0)) for p in PRODUCTS_L],
                       "shops": list((obs.get("town") or {}).get("unlocked_shops", []))},
            "shed": {k: int(v) for k, v in (private.get("shed") or {}).items() if v},
            "seeds": {k: int(v) for k, v in (private.get("seeds") or {}).items() if v},
            "act": {"me": act_me, "opp": act_opp},
        })
    trace = {"label": os.path.basename(out_path), "session": os.path.basename(session_path),
             "seed": session["seed"], "our_seat": side, "tape_seat": ss,
             "final": {"me": me, "opp": opp, "win": me > opp, "margin": me - opp},
             "products": PRODUCTS_L, "crops": CROPS_L, "animals": ANIMALS_L,
             "base_prices": _base_prices(), "plans": {}, "steps": steps}
    with open(out_path, "w") as fh:
        json.dump(trace, fh, separators=(",", ":"))
    return out_path


def pick_games(n, baselines=os.path.join(REPO, "rl", "runs", "draw995", "baselines.json")):
    """n recorded games spread across the rating bands, from the standing draw."""
    rows = json.load(open(baselines))["results"]["dev_knobs_h8"]
    by_band = collections.defaultdict(list)
    for r in rows:
        by_band[r["bin"]].append(r)
    out, b = [], 0
    while len(out) < n and any(by_band.values()):
        pool = by_band[b % len(BANDS)]
        if pool:
            out.append(pool.pop(len(pool) // 2))
        b += 1
    return [(os.path.join(REPO, r["path"]), r["script_side"], r["bin"]) for r in out[:n]]


# ------------------------------------------------------------ metric groups
def planner_metrics(traces):
    """Crop mix and herd at days 6/12/20, hires, land timing, idle ground."""
    snaps = {d: collections.defaultdict(list) for d in (6, 12, 20)}
    hires, quad_days, quads = [], collections.defaultdict(list), []
    for path in traces:
        T = json.load(open(path))
        prev_q = None
        peak_hands = 0
        for s in T["steps"]:
            me = s["me"]
            peak_hands = max(peak_hands, len(me["hands"]))
            q = sorted(me["quadrants"])
            if prev_q is not None and len(q) > len(prev_q):
                quad_days[len(q)].append(s["day"])
            prev_q = q
            if s["hour"] != 0 or s["day"] not in snaps:
                continue
            c = collections.Counter()
            for row in me["tiles"]:
                for t in row:
                    if isinstance(t, list) and t[0] == 4:
                        c[CROPS_L[t[1]] if 0 <= t[1] < len(CROPS_L) else "?"] += 1
                    elif isinstance(t, list) and t[0] == 5:
                        c[ANIMALS_L[t[1]] if 0 <= t[1] < len(ANIMALS_L) else "?"] += 1
            for k, v in c.items():
                snaps[s["day"]][k].append(v)
        hires.append(peak_hands)
        quads.append(len(prev_q or []))
    out = {"hands_peak": st.mean(hires) if hires else None,
           "quadrants_end": st.mean(quads) if quads else None}
    for d in snaps:
        for k, v in snaps[d].items():
            out[f"day{d}_{k.lower()}"] = st.mean(v)
    for n, days in quad_days.items():
        out[f"quadrant{n}_day"] = st.mean(days)
    return out


def market_metrics(traces):
    """Demand satisfied, realised price, and volume dumped into a crash.

    Demand is the town's REALISED drain: `econ.town_drain_per_day` evaluated
    against the shops actually open each day, summed. Volume is the units our
    SELL orders asked for -- the engine fills only as far as the shed allows,
    so this is an offer, not a fill; the leaders are measured the same way so
    the comparison holds."""
    from kagent import econ
    from kagent.constants import MARKET_PARAMS
    per = collections.defaultdict(list)
    for path in traces:
        T = json.load(open(path))
        prods = T["products"]
        demand = collections.Counter()
        offered = collections.Counter()
        crashed = collections.Counter()
        opp_offered = collections.Counter()
        seen_day = set()
        for s in T["steps"]:
            if s["day"] not in seen_day:
                seen_day.add(s["day"])
                for item, rate in econ.town_drain_per_day(s["market"]["shops"]).items():
                    demand[item] += rate
            px = dict(zip(prods, s["market"]["prices"]))
            for side, sink in (("me", offered), ("opp", opp_offered)):
                for o in (s["act"].get(side) or {}).get("market") or []:
                    if o and isinstance(o, (list, tuple)) and o[0] == "SELL" and len(o) >= 3:
                        item, n = o[1], int(o[2])
                        sink[item] += n
                        if side == "me" and px.get(item, 0) < CRASH_RATIO * MARKET_PARAMS[item]["base"]:
                            crashed[item] += n
        tot_off = sum(offered.values()) or 1
        per["demand_satisfied_share"].append(
            sum(min(offered[p], demand[p]) for p in demand) / max(1.0, sum(demand.values())))
        per["opp_demand_satisfied_share"].append(
            sum(min(opp_offered[p], demand[p]) for p in demand) / max(1.0, sum(demand.values())))
        per["units_offered"].append(sum(offered.values()))
        per["crashed_sale_share"].append(sum(crashed.values()) / tot_off)
        for p in ("WHEAT", "CARROT", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"):
            per[f"share_{p.lower()}"].append(offered[p] / max(1.0, demand[p]))
    return {k: st.mean(v) for k, v in per.items()}


def sell_metrics_for(traces):
    got = [SM.analyse(SM.load_trace(p)[0]) for p in traces]
    merged = SM._avg(got)
    out = {}
    for k in ("realized_over_mean", "evening_share", "units"):
        if merged.get("_all", {}).get(k) is not None:
            out[f"sell_{k}"] = merged["_all"][k]
    for p in ("STRAWBERRY", "MILK", "WOOL"):
        if merged.get(p, {}).get("impact") is not None:
            out[f"sell_impact_{p.lower()}"] = merged[p]["impact"]
    return out


def executor_metrics(traces):
    got = [TM.analyse(*TM.load_trace(p)) for p in traces]
    keys = ("moves_per_productive", "chores_per_visit", "chores_per_visit_animal",
            "chores_per_visit_crop", "pickups_per_feed", "feeds_from_hand_share",
            "shed_ops_per_unit_day", "harvest_ops", "plant_ops", "fertilize_ops",
            "h23_move_share", "one_time_units_total", "replant_within_1_step",
            "idle_share_mean", "weed_tile_days", "water_no_effect_share")
    out = {k: st.mean([g[k] for g in got if g.get(k) is not None]) for k in keys
           if any(g.get(k) is not None for g in got)}
    for c in ("WHEAT", "CARROT"):
        v = [g["one_time_units_per_harvest"].get(c) for g in got
             if g["one_time_units_per_harvest"].get(c) is not None]
        if v:
            out[f"units_per_harvest_{c.lower()}"] = st.mean(v)
    out["final_money"] = st.mean(g["final_money"] for g in got)
    out["opp_money"] = st.mean(g["opp_money"] for g in got)
    out["margin"] = out["final_money"] - out["opp_money"]
    out["trace_win_rate"] = st.mean(1.0 if g["win"] else 0.0 for g in got)
    return out


def outcome_metrics(eval_spec):
    """Win rate overall and per band from an eval that already exists."""
    if not eval_spec:
        return {}
    path, _, arm = eval_spec.partition(":")
    rows = json.load(open(path))["results"][arm]
    out = {"eval_games": len(rows),
           "win_rate": st.mean(1.0 if r["win"] >= 0.75 else 0.0 for r in rows),
           "eval_money": st.mean(r["money"] for r in rows),
           "eval_margin": st.mean(r["money"] - r["opp"] for r in rows)}
    for b in range(len(BANDS)):
        sub = [r for r in rows if r["bin"] == b]
        if sub:
            out[f"win_band{b}"] = st.mean(1.0 if r["win"] >= 0.75 else 0.0 for r in sub)
    return out


# ------------------------------------------------------------------ leaders
def leader_metrics():
    """Executor and selling for the three leaders. Executor comes from the
    cached docs/results/top3_tile_metrics.json when it is there (40 replays,
    already averaged); selling is recomputed, it is cheap."""
    import glob as _glob
    cached = json.load(open(TOP3_JSON)) if os.path.exists(TOP3_JSON) else {}
    out = {}
    if "ours" in cached:
        # the dev tree as measured this morning: the anchor a scorecard for
        # the reference agent itself is scored against, so the gap line still
        # says something ("how much of dev-tree -> leader has h10 closed")
        g = cached["ours"]
        dev = {k: g[k] for k in g if isinstance(g.get(k), (int, float))}
        for c in ("WHEAT", "CARROT"):
            v = (g.get("one_time_units_per_harvest") or {}).get(c)
            if v is not None:
                dev[f"units_per_harvest_{c.lower()}"] = v
        if "final_money" in dev and "opp_money" in dev:
            dev["margin"] = dev["final_money"] - dev["opp_money"]
        out["_dev_tree"] = dev
    for key, team, pattern in LEADERS:
        m = {}
        if key in cached:
            g = cached[key]
            for k in ("moves_per_productive", "chores_per_visit", "chores_per_visit_animal",
                      "chores_per_visit_crop", "pickups_per_feed", "feeds_from_hand_share",
                      "shed_ops_per_unit_day", "harvest_ops", "plant_ops", "fertilize_ops",
                      "h23_move_share", "one_time_units_total", "replant_within_1_step",
                      "idle_share_mean", "weed_tile_days", "water_no_effect_share",
                      "final_money", "opp_money"):
                if g.get(k) is not None:
                    m[k] = g[k]
            for c in ("WHEAT", "CARROT"):
                v = (g.get("one_time_units_per_harvest") or {}).get(c)
                if v is not None:
                    m[f"units_per_harvest_{c.lower()}"] = v
            if "final_money" in m and "opp_money" in m:
                m["margin"] = m["final_money"] - m["opp_money"]
        files = sorted(_glob.glob(os.path.join(REPO, pattern)))
        sells = []
        for f in files:
            try:
                sells.append(SM.analyse(SM.load_replay(f, team)[0]))
            except (ValueError, KeyError, IndexError):
                continue
        if sells:
            merged = SM._avg(sells)
            for k in ("realized_over_mean", "evening_share", "units"):
                if merged.get("_all", {}).get(k) is not None:
                    m[f"sell_{k}"] = merged["_all"][k]
            for p in ("STRAWBERRY", "MILK", "WOOL"):
                if merged.get(p, {}).get("impact") is not None:
                    m[f"sell_impact_{p.lower()}"] = merged[p]["impact"]
        m["_games"] = len(files)
        out[key] = m
    return out


PLANNER_CACHE = os.path.join(REPO, "docs", "results", "top3_planner.json")


def leader_planner(per_team=6):
    """Day 6/12/20 crop-and-herd snapshots, hands and land timing for the
    three leaders, read straight off the replay observations (tile_metrics
    drops quadrants and hands, which the planner group needs). Cached, since
    it is the only part that touches the raw replays."""
    import glob as _glob
    if os.path.exists(PLANNER_CACHE):
        return json.load(open(PLANNER_CACHE))
    from game_trace import tile_code
    out = {}
    for key, team, pattern in LEADERS:
        snaps = {d: collections.defaultdict(list) for d in (6, 12, 20)}
        hands, quad_days, quads = [], collections.defaultdict(list), []
        for f in sorted(_glob.glob(os.path.join(REPO, pattern)))[:per_team]:
            try:
                d = json.load(open(f))
                tn = d["info"].get("TeamNames")
                import ast as _ast
                tn = _ast.literal_eval(tn) if isinstance(tn, str) else tn
                seat = list(tn).index(team)
            except (ValueError, KeyError, IndexError):
                continue
            prev_q, peak = None, 0
            for raw in d["steps"]:
                obs = raw[0]["observation"]
                farm = obs["farms"][seat]
                day, hour = int(obs["day"]), int(obs["hour"])
                peak = max(peak, len(farm.get("hands") or []))
                q = sorted(farm.get("unlocked_quadrants") or ["NW"])
                if prev_q is not None and len(q) > len(prev_q):
                    quad_days[len(q)].append(day)
                prev_q = q
                if hour != 0 or day not in snaps:
                    continue
                c = collections.Counter()
                for row in farm["tiles"]:
                    for t in row:
                        code = tile_code(t, day)
                        if isinstance(code, list) and code[0] == 4:
                            c[CROPS_L[code[1]] if 0 <= code[1] < len(CROPS_L) else "?"] += 1
                        elif isinstance(code, list) and code[0] == 5:
                            c[ANIMALS_L[code[1]] if 0 <= code[1] < len(ANIMALS_L) else "?"] += 1
                for k, v in c.items():
                    snaps[day][k].append(v)
            hands.append(peak)
            quads.append(len(prev_q or []))
        m = {"hands_peak": st.mean(hands) if hands else None,
             "quadrants_end": st.mean(quads) if quads else None}
        for dd in snaps:
            for k, v in snaps[dd].items():
                m[f"day{dd}_{k.lower()}"] = st.mean(v)
        for n, days in quad_days.items():
            m[f"quadrant{n}_day"] = st.mean(days)
        out[key] = m
    try:
        with open(PLANNER_CACHE, "w") as fh:
            json.dump(out, fh, indent=1)
    except OSError:
        pass
    return out


# -------------------------------------------------------------------- gaps
# metric -> (+1 when higher is better, -1 when lower is better)
DIRECTION = {
    "moves_per_productive": -1, "chores_per_visit": +1, "chores_per_visit_animal": +1,
    "chores_per_visit_crop": +1, "pickups_per_feed": -1, "feeds_from_hand_share": +1,
    "shed_ops_per_unit_day": -1, "harvest_ops": +1, "plant_ops": +1, "fertilize_ops": +1,
    "h23_move_share": -1, "one_time_units_total": +1, "replant_within_1_step": +1,
    "idle_share_mean": -1, "weed_tile_days": -1, "water_no_effect_share": -1,
    "units_per_harvest_wheat": +1, "units_per_harvest_carrot": +1,
    "sell_realized_over_mean": +1, "sell_evening_share": +1, "sell_units": +1,
    "sell_impact_strawberry": +1, "sell_impact_milk": +1, "sell_impact_wool": +1,
    "demand_satisfied_share": +1, "crashed_sale_share": -1, "units_offered": +1,
}
# Metrics with no better/worse direction: the leaders' value is a TARGET to
# approach, not a number to beat. A crop mix is not "more is better" -- 41
# strawberry tiles against their 12-23 is a difference, and closing it means
# moving toward them from either side. The target is the leaders' mean.
TOWARD = {"hands_peak", "quadrants_end", "quadrant2_day", "quadrant3_day", "quadrant4_day",
          "share_wheat", "share_carrot", "share_strawberry", "share_melon",
          "share_egg", "share_milk", "share_wool"}


def _is_toward(k):
    return k in TOWARD or k.startswith(("day6_", "day12_", "day20_"))


GROUPS = {
    "OUTCOME": ["eval_games", "win_rate"] + [f"win_band{b}" for b in range(6)]
               + ["eval_money", "eval_margin", "trace_win_rate", "final_money", "opp_money", "margin"],
    "PLANNER": ["hands_peak", "quadrants_end", "quadrant2_day", "quadrant3_day", "quadrant4_day",
                "day6_wheat", "day6_strawberry", "day6_melon", "day6_cow", "day6_sheep", "day6_goose",
                "day12_wheat", "day12_carrot", "day12_strawberry", "day12_melon",
                "day12_cow", "day12_sheep", "day12_goose",
                "day20_wheat", "day20_carrot", "day20_strawberry",
                "day20_cow", "day20_sheep", "day20_goose",
                "idle_share_mean", "weed_tile_days"],
    "MARKET": ["units_offered", "demand_satisfied_share", "opp_demand_satisfied_share",
               "crashed_sale_share", "sell_realized_over_mean", "sell_evening_share",
               "sell_impact_strawberry", "sell_impact_milk", "sell_impact_wool",
               "share_wheat", "share_carrot", "share_strawberry", "share_melon",
               "share_egg", "share_milk", "share_wool"],
    "EXECUTOR": ["moves_per_productive", "chores_per_visit", "chores_per_visit_animal",
                 "chores_per_visit_crop", "pickups_per_feed", "feeds_from_hand_share",
                 "shed_ops_per_unit_day", "harvest_ops", "plant_ops", "fertilize_ops",
                 "one_time_units_total", "units_per_harvest_wheat", "units_per_harvest_carrot",
                 "replant_within_1_step", "h23_move_share", "water_no_effect_share"],
}


def furthest_from_leaders(agent, leaders, keys, n=4):
    """Where the agent sits furthest from the leaders, as a relative
    distance to their mean. Independent of any anchor, so it still says
    something when the reference row lacks the metric."""
    out = []
    for k in keys:
        tgt = [m[k] for m in leaders.values() if m.get(k) is not None]
        if not tgt or agent.get(k) is None:
            continue
        target = st.mean(tgt)
        scale = max(abs(target), 1e-6)
        out.append((k, agent[k], target, (agent[k] - target) / scale))
    out.sort(key=lambda r: -abs(r[3]))
    return out[:n]


def gap_closed(agent, ref, leaders, keys):
    """Share of the reference-to-leader distance the agent has closed, over
    every metric in the group that all three have and that has a direction.
    The leader target is the BEST of the three on that metric."""
    vals = []
    detail = []
    for k in keys:
        toward = _is_toward(k)
        d = DIRECTION.get(k)
        if (d is None and not toward) or k not in agent or k not in ref:
            continue
        tgt = [m[k] for m in leaders.values() if m.get(k) is not None]
        if not tgt:
            continue
        if toward:
            # approach the leaders' mean from either side
            target = st.mean(tgt)
            span = abs(target - ref[k])
            if span < 1e-9:
                continue
            moved = span - abs(target - agent[k])
            frac = max(-1.0, min(2.0, moved / span))
        else:
            target = max(tgt) if d > 0 else min(tgt)
            span = (target - ref[k]) * d
            if abs(span) < 1e-9:
                continue
            moved = (agent[k] - ref[k]) * d
            frac = max(-1.0, min(2.0, moved / span))
        vals.append(frac)
        detail.append((k, frac))
    return (st.mean(vals) if vals else None), detail


# ------------------------------------------------------------------ h2h
def head_to_head(spec, ref_spec, n_seeds, seed0=770_000):
    """Our agent against the reference agent directly, both seats per seed."""
    from rl import rollout
    if ref_spec["kind"] != "bundle":
        raise SystemExit("--h2h needs a bundle reference (the opponent slot takes a frozen main.py)")
    wins, margins = 0, []
    for i in range(n_seeds):
        for side in (0, 1):
            job = {"seed": seed0 + i, "side": side, "temperature": 0.0,
                   "opponent": ("bundle", ref_spec["path"])}
            if spec["kind"] == "bundle":
                job["learner_bundle"] = spec["path"]
            elif spec["kind"] == "weights":
                job["weights_path"] = spec["path"]
                job["mode"] = "rl"
            else:
                job["learner_policy"] = spec
            out = rollout.run_episode(job)
            wins += 1 if out["final_money"] > out["opp_money"] else 0
            margins.append(out["final_money"] - out["opp_money"])
    n = 2 * n_seeds
    return {"games": n, "wins": wins, "win_rate": wins / n if n else None,
            "mean_margin": st.mean(margins) if margins else None}


# ----------------------------------------------------------------- report
def collect(spec, games, workdir, tag):
    paths = []
    for i, (sess, side, band) in enumerate(games):
        out = os.path.join(workdir, f"{tag}_{i}_b{band}.json")
        trace_game(spec, sess, side, out)
        paths.append(out)
    return paths


def build_row(traces, eval_spec=None):
    m = {}
    m.update(executor_metrics(traces))
    m.update(sell_metrics_for(traces))
    m.update(market_metrics(traces))
    m.update(planner_metrics(traces))
    m.update(outcome_metrics(eval_spec))
    return m


FMT = {"win_rate": "{:.3f}", "trace_win_rate": "{:.2f}", "eval_money": "{:,.0f}",
       "eval_margin": "{:+,.0f}", "final_money": "{:,.0f}", "opp_money": "{:,.0f}",
       "margin": "{:+,.0f}", "weed_tile_days": "{:.0f}", "harvest_ops": "{:.0f}",
       "plant_ops": "{:.0f}", "fertilize_ops": "{:.0f}", "one_time_units_total": "{:.0f}",
       "units_offered": "{:.0f}", "sell_units": "{:.0f}", "eval_games": "{:.0f}"}


def print_table(rows, out=sys.stdout):
    labels = list(rows)
    w = max(12, max(len(l) for l in labels) + 2)
    for group, keys in GROUPS.items():
        present = [k for k in keys if any(rows[l].get(k) is not None for l in labels)]
        if not present:
            continue
        print(f"\n== {group} " + "=" * (34 + w * len(labels) - len(group) - 4), file=out)
        print(f"{'metric':<34}" + "".join(f"{l:>{w}}" for l in labels), file=out)
        for k in present:
            cells = []
            for l in labels:
                v = rows[l].get(k)
                if v is None:
                    cells.append("-")
                else:
                    cells.append(FMT.get(k, "{:.3f}" if abs(v) < 100 else "{:,.0f}").format(v))
            print(f"{k:<34}" + "".join(f"{c:>{w}}" for c in cells), file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", required=True, help="champion | knobs:<json> | bundle:<path> | weights:<path> | execnet:<...>")
    ap.add_argument("--reference", default="bundle:arena/agents/champ-h10/main.py")
    ap.add_argument("--eval", default=None, help="<paired eval json>:<arm>, attached as the OUTCOME row")
    ap.add_argument("--reference-eval", default="rl/runs/draw995/batch2.json:ship2_hv")
    ap.add_argument("--games", type=int, default=6, help="traces per agent, spread across bands")
    ap.add_argument("--h2h", type=int, default=0, help="seeds for a direct head-to-head (both seats each)")
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    ap.add_argument("--keep-traces", default=None, help="directory to keep the traces in")
    args = ap.parse_args()

    spec = parse_spec(args.agent)
    ref = parse_spec(args.reference)
    games = pick_games(args.games)
    print(f"{len(games)} games across bands {[b for _, _, b in games]}", file=sys.stderr)

    workdir = args.keep_traces or tempfile.mkdtemp(prefix="scorecard_")
    os.makedirs(workdir, exist_ok=True)
    rows = {}
    print(f"tracing {spec_label(spec)} ...", file=sys.stderr)
    rows[spec_label(spec)] = build_row(collect(spec, games, workdir, "agent"), args.eval)
    same = args.agent == args.reference
    if not same:
        print(f"tracing {spec_label(ref)} (reference) ...", file=sys.stderr)
        rows[spec_label(ref)] = build_row(collect(ref, games, workdir, "ref"), args.reference_eval)
    print("leaders ...", file=sys.stderr)
    leaders = leader_metrics()
    dev_tree = leaders.pop("_dev_tree", None)
    for k, m in leader_planner().items():
        if k in leaders:
            leaders[k].update({kk: vv for kk, vv in m.items() if vv is not None})
    for k, m in leaders.items():
        rows[k] = m

    print_table(rows)

    agent_row = rows[spec_label(spec)]
    if same:
        # scoring the reference against itself says nothing; anchor on the dev
        # tree instead and say so
        ref_row = dev_tree or agent_row
        anchor = "dev tree (cached)" if dev_tree else "itself"
    else:
        ref_row = rows[spec_label(ref)]
        anchor = spec_label(ref)
    print("\n== GAP TO THE LEADERS " + "=" * 40)
    print(f"share of the {anchor} -> best-leader distance this agent has closed")
    print("(0.00 = level with that anchor, 1.00 = level with the best leader)")
    gaps = {}
    for group, keys in GROUPS.items():
        frac, detail = gap_closed(agent_row, ref_row, leaders, keys)
        gaps[group] = frac
        if frac is None:
            print(f"  {group:<10} n/a")
            continue
        worst = sorted(detail, key=lambda kv: kv[1])[:3]
        print(f"  {group:<10} {frac:+.2f}   furthest behind: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in worst))
    print("\nfurthest from the leaders' mean, regardless of anchor "
          "(metric: ours vs theirs, relative):")
    for group, keys in GROUPS.items():
        far = furthest_from_leaders(agent_row, leaders, keys, n=3)
        if far:
            print(f"  {group:<10} " + ",  ".join(
                f"{k} {a:,.2f} vs {t:,.2f} ({rel:+.0%})" for k, a, t, rel in far))
    live = {g: v for g, v in gaps.items() if v is not None}
    if live:
        limiting = min(live, key=live.get)
        print(f"\nlimiting dimension: {limiting} ({live[limiting]:+.2f})")

    h2h = None
    if args.h2h:
        print(f"\n== HEAD TO HEAD vs {spec_label(ref)} ({2 * args.h2h} games) ==")
        h2h = head_to_head(spec, ref, args.h2h)
        print(f"  wins {h2h['wins']}/{h2h['games']} ({h2h['win_rate']:.3f})   "
              f"mean margin {h2h['mean_margin']:+,.0f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "gaps": gaps, "h2h": h2h,
                       "agent": args.agent, "reference": args.reference,
                       "games": [(os.path.basename(s), sd, b) for s, sd, b in games]}, fh, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
