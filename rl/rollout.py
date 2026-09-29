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

# LIVE board state for RL. The policy has only ever seen `kagent.features`
# day-level aggregates (168 numbers); `tools/board_tensor.py` was built for
# offline flow nets and never entered the training loop. With KAGG_BOARD=1
# every recorded decision also carries the board tensor for the CURRENT
# state, so a conv trunk can be trained end to end on states the policy
# actually visits instead of on a frozen corpus snapshot.
#
# float16 because it is a 91x10x10 of mostly zeros and ones: 18 KB a decision,
# ~35 MB an iteration at 64 episodes x 30 decisions, which survives the Pool.
_BOARD = os.environ.get("KAGG_BOARD", "") not in ("", "0")


# Which channel set the trunk eats. T_crop_103 is the arm that PASSED
# H-margin -- >= +0.01 R2 in 4 of 5 bands and win AUC up in 4 of 5 -- and it
# is 103 columns of the 123-channel TEMPORAL build, not of the 91-channel
# single frame. board_pair_full tops out at index 90, so the crop lag planes
# (103..122) simply do not exist in it; the flat board is the one encoding we
# have evidence does NOT help the outcome.
_BOARD_ARM = os.environ.get("KAGG_BOARD_ARM", "T_crop_103")


def _board(obs, seat, dawn_hist, mid_hist, day):
    """(len(arm), 10, 10) float16 for THIS seat, or None when the hook is off.

    Mirrors `tools/s3_flow.encode_day` so the RL encoder and the offline one
    cannot drift: past dawn observations newest-first, plus yesterday's
    midday frame. `prev_mid` and `hist` are empty on day 0, which zeroes the
    lag planes rather than failing -- the same thing the offline build does
    for the first day of a season.
    """
    if not _BOARD:
        return None
    from tools.board_tensor import board_pair_temporal_both, arm_channels
    past = [dawn_hist[d] for d in range(day - 1, -1, -1) if d in dawn_hist]
    t = board_pair_temporal_both(obs, seat, prev_mid=mid_hist.get(day - 1),
                                 hist=past)
    return t[arm_channels(_BOARD_ARM)].astype("float16")

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


def _lead(view):
    """Ingredients of the phase potential (rl/phase_potential.py): the asset
    and money LEADS over the opponent plus the day, recorded per decision so
    the trainer can shape with Phi(s, day) under --potential lead. One pass
    over both boards; no torch."""
    from rl.phase_potential import ingredients
    return ingredients(view)


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


def _decode():
    """How a v4/v5 net's choices become a plan, chosen by the parent via
    KAGG_DECODE=commit (Pool workers inherit the environment): see
    kagent.nnpolicy.NeuralPolicy. Default "plan" is the legacy path."""
    return os.environ.get("KAGG_DECODE", "plan")


def _set_exec(flags):
    """Executor rule flags for THIS episode: returns the previous table so
    run_episode can put the worker back for its next job. scheduler.EXEC is
    process-global, so the flags reach BOTH seats -- the heuristic opponent
    in this worker then plays the deployed champion, not the flagless one."""
    from kagent import scheduler as _sched
    prev = dict(_sched.EXEC)
    _sched.reset_exec()
    _sched.set_exec(flags)
    return prev


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
        # Same lag-plane history as the RL path: BC pretrains the trunk, so it
        # must see the identical channel set or the weights it hands to PPO
        # were fitted on a different encoder.
        bc_dawn, bc_mid = {}, {}

        def agent_fn(obs):
            view = View(obs)
            if _BOARD:
                if view.hour == 0:
                    bc_dawn[view.day] = obs
                elif view.hour == 12:
                    bc_mid[view.day] = obs
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
                        "board": _board(obs, view.player, bc_dawn, bc_mid,
                                        view.day),
                    }
                )
                mem.day_plan = plan
                mem.day_plan_day = view.day
            return pol.act(view, mem)

        return agent_fn

    provider = NeuralPolicy(layers, temperature=temperature, rng=rng, residual=residual,
                            decode=_decode())
    pol = HeuristicPolicy(provider)

    last_view = {}
    # Dawn and midday frames, kept for the temporal lag planes. Only the
    # observations are held, not tensors: ~60 Structs a season.
    dawn_hist, mid_hist = {}, {}

    def agent_fn(obs):
        view = View(obs)
        if _BOARD:
            if view.hour == 0:
                dawn_hist[view.day] = obs
            elif view.hour == 12:
                mid_hist[view.day] = obs
        last_view["v"] = view
        mem.observe(view)
        provider.last_decision = None
        # The board must reach the ACTING policy, not just the record. Until
        # this line the trunk was serialised nowhere and nnpolicy silently
        # dropped the 48 board columns of a 216-wide h1, so every board arm
        # played with a truncated policy while training on the full one.
        #
        # Built only when a plan will be made this step (HeuristicPolicy.act
        # rebuilds when day_plan_day != view.day, i.e. at hour 0): the tensor
        # is ~9 board_tensor passes and was being built at all 720 steps for
        # 30 decisions. On the other 690 steps the provider gets None, so a
        # plan made unexpectedly mid-day raises instead of reading a stale
        # board.
        need_plan = mem.day_plan is None or mem.day_plan_day != view.day
        brd = (_board(obs, view.player, dawn_hist, mid_hist, view.day)
               if need_plan else None)
        provider.board = brd
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
                    "board": brd,
                    "lead": _lead(view),
                }
            )
        return action

    agent_fn.last_view = last_view
    return agent_fn


def _make_dev_policy_agent(spec):
    """-> (agent_fn, switched_mode). spec: {"kind": "champion"} or
    {"kind": "knobs", "rules": {...}} (a BANDIT_RULES dict, e.g. champ-h8's).
    Mirrors kagent/policy.py's bundle wiring: CHAMPION_PLAN ->
    HeuristicPolicy(ChampionPlanProvider()); BANDIT_RULES -> residual heads +
    HeuristicPolicy(KnobBanditPolicy | BanditPolicy)."""
    # rlv3 (the adaptive package) dispatches FIRST and returns, so every other
    # kind still reaches the shipped implementation untouched. It lives here
    # rather than in a monkeypatch because multiprocessing starts `spawn` on
    # this machine: a patch applied in the parent is invisible in the worker,
    # which would import a clean rl.rollout and raise "unknown learner_policy
    # kind 'rlv3'". Putting the branch in the module means every worker gets
    # it on import, so rl/session_eval.py's Pool can run an rlv3 arm.
    if spec.get("kind") == "rlv3":
        from rlv3.agent import make_dev_agent
        return make_dev_agent(spec)

    from kagent.policy import ChampionPlanProvider, ThetaPlanProvider
    from kagent.banditpolicy import BanditPolicy, KnobBanditPolicy

    kind = spec.get("kind", "champion")
    switched = False
    needs_board = False
    if kind == "champion":
        pol = HeuristicPolicy(ChampionPlanProvider())
    elif kind == "theta":
        # A STRATEGY VECTOR: the champion plan with every knob fixed from day 0
        # (kagent/theta.py). The executor flags on spec["exec"] are applied by
        # run_episode, which also restores them, so a worker can play a
        # different candidate on its next job without leaking this one's rules.
        pol = HeuristicPolicy(ThetaPlanProvider(spec.get("knobs") or {}))
    elif kind == "reactive":
        # theta as a FUNCTION of the observed state (kagent/reactive.py): a base
        # vector from day 0 plus rules that fire on features.context at their
        # own decision day. With no rules this is exactly the theta arm, so a
        # reactive candidate is a pure addition to it and the gate attributes
        # to the rules.
        from kagent.reactive import ReactivePlanProvider
        pol = HeuristicPolicy(ReactivePlanProvider(spec.get("knobs") or {},
                                                   spec.get("rules") or []))
    elif kind in ("knobs", "bandit"):
        rules = spec["rules"]
        if features.MODE != "residual":
            features.set_mode("residual")
            switched = True
        cls = KnobBanditPolicy if rules.get("kind") == "knobs" else BanditPolicy
        pol = HeuristicPolicy(cls(rules))
    elif kind == "execnet":
        # The learned executor (rl/execnet) over an injectable DAY PLAN: the
        # champion plan (default), the knob bandit with a rules dict, or a
        # neural planner ("planner": "weights"). The hybrid arm is the
        # kernel-trained board net planning and the destination model doing
        # the chores and the walking.
        from rl.execnet.policy import load_policy, make_plan_provider
        rules = spec.get("rules")
        planner_kind = spec.get("planner") or ("knobs" if rules else "champion")
        provider, switched, needs_board = make_plan_provider(
            planner_kind, rules=rules, weights=spec.get("planner_weights"))
        pol = load_policy(spec["weights"], provider, conf=float(spec.get("conf", 0.0)),
                          mode=spec.get("mode", "chores"))
    else:
        raise ValueError(f"unknown learner_policy kind {kind!r}")
    mem = Memory()

    if needs_board:
        # A trunk-carrying planner must be handed the board tensor BEFORE it
        # plans, exactly as _make_learner_agent does for a plain weights arm;
        # without it nnpolicy silently drops the board columns and the arm
        # plays a truncated policy. Built only on the step a plan is made.
        dawn_hist, mid_hist = {}, {}

        def agent_fn(obs):
            view = View(obs)
            if view.hour == 0:
                dawn_hist[view.day] = obs
            elif view.hour == 12:
                mid_hist[view.day] = obs
            mem.observe(view)
            need_plan = mem.day_plan is None or mem.day_plan_day != view.day
            provider.board = (_board(obs, view.player, dawn_hist, mid_hist, view.day)
                              if need_plan else None)
            return pol.act(view, mem)

        return agent_fn, switched

    def agent_fn(obs):
        view = View(obs)
        mem.observe(view)
        return pol.act(view, mem)

    return agent_fn, switched


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
        provider = NeuralPolicy(layers, temperature=temp, rng=random.Random(0), residual=_residual(),
                                decode=_decode())
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
    switched_mode = False
    exec_prev = None
    if job.get("learner_policy"):
        # OUR side played by the DEV TREE's scripted agents: the champion plan,
        # or the knob bandit over it with a rules dict (lifted from a frozen
        # bundle). The same objects the bundler freezes into main.py, from
        # today's kagent/ -- the check that the dev tree still IS the champion.
        spec = job["learner_policy"]
        learner, switched_mode = _make_dev_policy_agent(spec)
        # Executor rule flags for THIS episode only (kagent/scheduler.EXEC):
        # "exec" on the spec or inside its rules JSON; restored afterwards so
        # the next job in the worker sees the shipped executor.
        flags = spec.get("exec") or (spec.get("rules") or {}).get("exec")
        if flags:
            exec_prev = _set_exec(flags)
    elif job.get("learner_bundle"):
        # OUR side played by a frozen bundle (a shipped main.py), exec'd the way
        # the Kaggle loader does it. No records: nothing to train on, this is
        # the deployment comparison -- the same games, the same seats, the
        # shipped agent instead of the net.
        learner = _make_opponent(("bundle", job["learner_bundle"]))
    else:
        learner = _make_learner_agent(
            layers, temperature, sample_rng, records, mode, style=job.get("style", "react"),
            dev=job.get("dev"), extra=extra,
        )
    # Flags on the JOB reach every kind of learner, the net included. Without
    # this a net trained and gated FLAGLESS while the champion it was compared
    # with carried the shipped four (rl/train.py --exec shipped sets it).
    if exec_prev is None and job.get("exec"):
        exec_prev = _set_exec(job["exec"])
    opponent = _make_opponent(job.get("opponent", "heuristic"))

    side = job.get("side", 0)
    agents = [learner, opponent] if side == 0 else [opponent, learner]

    import time as _time

    t0 = _time.time()
    env = make("kaggriculture", configuration={"seed": job["seed"]}, debug=False)
    try:
        env.run(agents)
    finally:
        if switched_mode:
            # the bandit needs residual heads; put the worker back for the
            # next job, which may be a neural net on absolute heads
            features.set_mode("absolute")
        if exec_prev is not None:
            from kagent import scheduler as _sched
            _sched.EXEC.update(exec_prev)
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
