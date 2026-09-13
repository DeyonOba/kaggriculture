"""Episode rollout worker (multiprocessing-safe, NO torch import).

Runs one full kaggriculture episode with the neural planner playing one side,
recording per-day decisions (features, masks, choices, logprob, value,
portfolio value) for PPO/BC training in the parent process.
"""

import json
import os
import random

from kagent import econ, planner
from kagent.constants import ANIMALS, CROPS
from kagent.gamestate import Memory, View
from kagent.nnpolicy import NeuralPolicy
from kagent.policy import HeuristicPolicy
from kagent import features

_WEIGHTS_CACHE = {}
KNOB_DAY = 6  # shops revealed on days 3 and 6
DECISION_DAYS = (6, 9, 12, 15)  # contexts recorded for conditional arms


def _load_layers(path):
    if path is None:
        return None
    key = (path, os.path.getmtime(path))
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE.clear() if len(_WEIGHTS_CACHE) > 8 else None
        with open(path) as f:
            _WEIGHTS_CACHE[key] = json.load(f)
    return _WEIGHTS_CACHE[key]


# Rough per-animal value of remaining production (used only for reward
# shaping during the curriculum; annealed to zero).
_ANIMAL_DAILY = {"GOOSE": 60, "COW": 100, "SHEEP": 70}


def stock_value(view):
    """Market value of everything harvested but not yet sold (shed + all unit
    inventories). Used for the carrying-cost penalty: held product is capital
    at risk of the end-of-season zero."""
    total = 0.0
    counts = dict(view.shed)
    for idx, _, _ in view.me.units:
        for item, n in view.unit_inv(idx).items():
            counts[item] = counts.get(item, 0) + n
    for item, n in counts.items():
        if n <= 0 or item in ANIMALS:
            continue
        total += n * view.market_prices.get(item, 10)
    return total


def portfolio_value(view):
    """Money + liquidation-ish value of shed, herd, and standing crops."""
    v = view.me.money
    for item, n in view.shed.items():
        if n <= 0:
            continue
        if item in ANIMALS:
            v += 0.5 * ANIMALS[item]["cost"] * n
            continue
        inv = view.market_inv.get(item)
        if inv is not None:
            v += econ.sell_revenue(item, inv, min(n, 40))
    days_left = max(0, 28 - view.day)
    for _, _, t in view.me.animals():
        v += _ANIMAL_DAILY.get(t.get("animal"), 50) * min(days_left, 8)
        v += 0.3 * ANIMALS.get(t.get("animal"), {}).get("cost", 300)
    for _, _, t in view.me.plants():
        crop = CROPS.get(t.get("crop"))
        if not crop:
            continue
        base = 0.6 * view.market_prices.get(t["crop"], 20)
        v += base * t.get("yield_units", 0)
        age = view.day - t.get("planted_day", view.day)
        if age <= crop["max_yield_day"]:
            v += 0.3 * base * max(0, crop["max_yield"] - t.get("yield_units", 0))
    return v


def _nearest_valid(vals, mask, target):
    valid = [i for i in range(len(vals)) if mask[i]]
    return min(valid, key=lambda i: abs(vals[i] - target))


def plan_to_choices(plan, view, head_masks):
    """Quantize a heuristic DayPlan into head choice indices (BC targets),
    expressed as ABSOLUTE standing targets (current count + planned delta)
    and projected onto the valid-action mask."""
    crop_counts = {}
    for _, _, t in view.me.plants():
        crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
    herd = {}
    for _, _, t in view.me.animals():
        herd[t["animal"]] = herd.get(t["animal"], 0) + 1

    anchor_vals = features.anchors(view)
    choices = []
    for (name, vals), mask in zip(features.HEADS, head_masks):
        if name in features.QUOTA_CROP:
            crop = features.QUOTA_CROP[name]
            target = crop_counts.get(crop, 0) + plan.plant_quota.get(crop, 0)
            if name in features.ANCHORED:
                target -= anchor_vals[name]
        elif name in features.BUY_KIND:
            kind = features.BUY_KIND[name]
            target = (
                herd.get(kind, 0)
                + view.shed.get(kind, 0)
                + plan.animal_buys.get(kind, 0)
            )
            if name in features.ANCHORED:
                target -= anchor_vals[name]
        elif name == "hires":
            target = plan.hires_target
        elif name == "land":
            target = 1 if plan.land_buys else 0
        elif name == "melon_floor":
            target = plan.sell_floors.get("MELON", 80)
        elif name.endswith("_floor_scale"):
            target = 1.0  # heuristic floors are the BC baseline
        else:
            target = vals[0]
        choices.append(_nearest_valid(vals, mask, target))
    return choices


def _residual():
    """Residual mode is selected by the parent via KAGG_RESIDUAL=1 (Pool
    workers inherit the environment); idempotent per process."""
    if os.environ.get("KAGG_RESIDUAL") == "1":
        if features.MODE != "residual":
            features.set_mode("residual")
        return True
    return False


def _make_learner_agent(layers, temperature, rng, records, mode, style="react", dev=None, extra=None):
    mem = Memory()
    residual = _residual()
    if mode == "knob":
        # Counterfactual bandit over SEASON-LONG knobs: the champion plan
        # with `dev` = {knob: value} overrides from decision day `_day`
        # (default KNOB_DAY: shops revealed); the state at that day is the
        # context. Contexts at every decision day are recorded so the
        # collector can choose price-/shop-conditioned arms from the twin.
        dev = dict(dev or {})
        apply_day = int(dev.pop("_day", KNOB_DAY))
        pol = HeuristicPolicy()

        def agent_fn(obs):
            view = View(obs)
            mem.observe(view)
            if mem.day_plan is None or mem.day_plan_day != view.day:
                if view.day in DECISION_DAYS and extra is not None:
                    extra.setdefault("ctx_days", {})[view.day] = features.context(view)
                if view.day == apply_day:
                    mem.knobs = dict(dev)
                    if extra is not None:
                        extra["dev_x"] = features.encode(view, mem)
                        extra["dev_ctx"] = features.context(view)
                        extra["dev_applied"] = True
                        extra["dev_noop"] = False
                mem.day_plan = planner.make_day_plan(view, mem, speculative=True)
                mem.day_plan_day = view.day
            return pol.act(view, mem)

        return agent_fn
    if mode == "deviate":
        # Counterfactual bandit: the champion plan every day except ONE
        # (day, head, offset) deviation; the state at that day is recorded so
        # the paired gain attributes to exactly one decision.
        dev_day, dev_head, dev_off = dev
        pol = HeuristicPolicy()

        def agent_fn(obs):
            view = View(obs)
            mem.observe(view)
            if mem.day_plan is None or mem.day_plan_day != view.day:
                plan = planner.make_day_plan(view, mem, speculative=True)
                if view.day == dev_day:
                    m = features.masks(view)
                    choices = features.residual_choices(plan, view, m)
                    choices[dev_head] = dev_off
                    before = (dict(plan.plant_quota), dict(plan.animal_buys), plan.hires_target,
                              list(plan.land_buys), dict(plan.sell_floors))
                    plan = features.apply_residual(plan, choices, view)
                    after = (dict(plan.plant_quota), dict(plan.animal_buys), plan.hires_target,
                             list(plan.land_buys), dict(plan.sell_floors))
                    if extra is not None:
                        extra["dev_x"] = features.encode(view, mem)
                        extra["dev_ctx"] = features.context(view)
                        extra["dev_applied"] = bool(m[dev_head][dev_off])
                        extra["dev_noop"] = before == after
                mem.day_plan = plan
                mem.day_plan_day = view.day
            return pol.act(view, mem)

        return agent_fn
    if mode == "bc":
        pol = HeuristicPolicy()
        speculative = style == "spec" or residual

        def agent_fn(obs):
            view = View(obs)
            mem.observe(view)
            if mem.day_plan is None or mem.day_plan_day != view.day:
                if view.day in DECISION_DAYS and extra is not None:
                    extra.setdefault("ctx_days", {})[view.day] = features.context(view)
                x = features.encode(view, mem)
                m = features.masks(view)
                plan = planner.make_day_plan(view, mem, speculative=speculative)
                choices = (
                    features.residual_choices(plan, view, m) if residual
                    else plan_to_choices(plan, view, m)
                )
                records.append(
                    {
                        "x": x,
                        "masks": m,
                        "choices": choices,
                        "pv": portfolio_value(view),
                    }
                )
                mem.day_plan = plan
                mem.day_plan_day = view.day
            return pol.act(view, mem)

        return agent_fn

    provider = NeuralPolicy(layers, temperature=temperature, rng=rng, residual=residual)
    pol = HeuristicPolicy(provider)

    last_view = {}

    def agent_fn(obs):
        view = View(obs)
        last_view["v"] = view
        mem.observe(view)
        provider.last_decision = None
        action = pol.act(view, mem)
        if provider.last_decision is not None:
            x, m, choices, logprob, value = provider.last_decision
            records.append(
                {
                    "x": x,
                    "masks": m,
                    "choices": choices,
                    "logprob": logprob,
                    "value": value,
                    "pv": portfolio_value(view),
                    "stock": stock_value(view),
                }
            )
        return action

    agent_fn.last_view = last_view
    return agent_fn


_BUNDLE_CACHE = {}
_SESSION_CACHE = {}


def load_session(path):
    key = (path, os.path.getmtime(path))
    if key not in _SESSION_CACHE:
        if len(_SESSION_CACHE) > 32:
            _SESSION_CACHE.clear()
        with open(path) as f:
            _SESSION_CACHE[key] = json.load(f)
    return _SESSION_CACHE[key]


def make_script_agent(actions):
    """Replays a recorded side's actions verbatim on its original world."""
    state = {"t": 0}

    def agent(obs):
        t = obs.get("step", state["t"])
        state["t"] = t + 1
        if t < len(actions) and actions[t]:
            return actions[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def _make_opponent(spec):
    if isinstance(spec, (list, tuple)) and spec[0] == "session":
        session = load_session(spec[1])
        return make_script_agent(session["actions"][spec[2]])
    if isinstance(spec, (list, tuple)) and spec[0] == "bundle":
        # A frozen self-contained main.py (e.g. arena/agents/rl-v1) exec'd the
        # way the Kaggle loader does it. Immune to feature/head version skew.
        path = spec[1]
        key = (path, os.path.getmtime(path))
        if key not in _BUNDLE_CACHE:
            env_dict = {}
            with open(path) as f:
                exec(f.read(), env_dict)
            _BUNDLE_CACHE.clear() if len(_BUNDLE_CACHE) > 4 else None
            _BUNDLE_CACHE[key] = [v for v in env_dict.values() if callable(v)][-1]
        return _BUNDLE_CACHE[key]
    if spec == "heuristic":
        mem = Memory()
        pol = HeuristicPolicy()

        def opp_fn(obs):
            view = View(obs)
            mem.observe(view)
            return pol.act(view, mem)

        return opp_fn
    if isinstance(spec, (list, tuple)) and spec[0] in ("snapshot", "league"):
        layers = _load_layers(spec[1])
        mem = Memory()
        # League members (past bests) play sharply; plain snapshots stay
        # noisy for exploration diversity.
        temp = 0.2 if spec[0] == "league" else 0.7
        provider = NeuralPolicy(layers, temperature=temp, rng=random.Random(0), residual=_residual())
        pol = HeuristicPolicy(provider)

        def opp_fn(obs):
            view = View(obs)
            mem.observe(view)
            return pol.act(view, mem)

        return opp_fn
    return spec  # builtin name string like "starter"


def run_episode(job):
    """job: dict(weights_path, temperature, seed, side, opponent, mode).

    Returns dict(records, final_money, opp_money, win, final_pv).
    """
    from kaggle_environments import make

    _residual()
    opp_spec = job.get("opponent", "heuristic")
    if isinstance(opp_spec, (list, tuple)) and opp_spec[0] == "session":
        # Session-replay: play on the recorded game's true world.
        job = dict(job, seed=load_session(opp_spec[1])["seed"])

    layers = _load_layers(job.get("weights_path"))
    rng = random.Random(job["seed"] * 7919 + job.get("side", 0))
    records = []
    mode = job.get("mode", "rl")
    temperature = job.get("temperature", 1.0)
    sample_rng = rng if (mode == "rl" and temperature > 0) else None

    extra = {}
    learner = _make_learner_agent(
        layers, temperature, sample_rng, records, mode, style=job.get("style", "react"),
        dev=job.get("dev"), extra=extra,
    )
    opponent = _make_opponent(job.get("opponent", "heuristic"))

    side = job.get("side", 0)
    agents = [learner, opponent] if side == 0 else [opponent, learner]

    import time as _time

    t0 = _time.time()
    env = make("kaggriculture", configuration={"seed": job["seed"]}, debug=False)
    env.run(agents)
    duration = _time.time() - t0

    if duration > 60:
        try:
            with open("rl/runs/slow_jobs.jsonl", "a") as f:
                f.write(json.dumps({"duration": round(duration, 1), "job": {
                    k: v for k, v in job.items()}}) + "\n")
        except OSError:
            pass

    final = env.steps[-1]
    my_r = final[side].reward or 0.0
    opp_r = final[1 - side].reward or 0.0

    # Season over: everything unsold is worthless. Measure it so training can
    # penalize dead capital explicitly (the "bad investment" signal).
    unsold = 0.0
    lv = getattr(learner, "last_view", {}).get("v")
    if lv is not None:
        unsold = stock_value(lv)

    return {
        "records": records,
        "final_money": my_r,
        "opp_money": opp_r,
        "win": 1.0 if my_r > opp_r else (0.5 if my_r == opp_r else 0.0),
        "statuses": [s.status for s in final],
        "duration": duration,
        "unsold_value": unsold,
        "is_session": isinstance(opp_spec, (list, tuple)) and opp_spec[0] == "session",
        "extra": extra,
    }
