"""Model-predictive day planning: at each day boundary, roll out candidate
day-plans ~2 days ahead in the real engine and keep the best.

"Optimize profit and investment per seeded environment" made literal: instead
of trusting one argmax plan, the agent simulates the plans against a proxy
opponent from the CURRENT observed world (market inventories, shop draw, both
farms) and picks by realized money + portfolio.

Budget: ~3 candidates x 48 engine steps ~ 0.3-0.8s once per game day, paid
from the 60s season overage bank; a guard skips MPC when the bank runs low.
Any exception falls back to the plain neural plan (SafeAgent guards the rest).
"""

import copy

from kagent import features, planner
from kagent.constants import CROPS, ANIMALS, TURNS_PER_DAY
from kagent.gamestate import Memory, View
from kagent.nnpolicy import NeuralPolicy

SIM_DAYS = 2
MIN_OVERAGE_LEFT = 15.0  # skip MPC below this remainingOverageTime


def _shift_state(farm, day0, step0):
    """Translate absolute day/step fields into the simulator's clock."""
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            for key in ("planted_day", "placed_day"):
                if key in t and t[key] is not None:
                    t[key] -= day0
            if t.get("fertilized_until_day", -1) >= 0:
                t["fertilized_until_day"] -= day0
            if t.get("max_lifespan_step", -1) >= 0:
                t["max_lifespan_step"] -= step0


class _PinnedPlanProvider:
    """Serves a fixed plan for the first sim day, heuristic afterward."""

    def __init__(self, pinned_plan):
        self.pinned = pinned_plan
        self.served = False

    def make_plan(self, view, memory):
        if not self.served:
            self.served = True
            return copy.deepcopy(self.pinned)
        return planner.make_day_plan(view, memory)


def _sim_agent(provider_factory, offset_day):
    from kagent.policy import HeuristicPolicy

    mem = Memory()
    pol = HeuristicPolicy(provider_factory())

    def agent(obs):
        obs = dict(obs)
        obs["day"] = obs.get("day", 0) + offset_day  # planner windows need real day
        view = View(obs)
        mem.observe(view)
        return pol.act(view, mem)

    return agent


def _score(obs, side):
    view = View(dict(obs, player=side))
    v = view.me.money
    for item, n in view.shed.items():
        if n > 0 and item not in ANIMALS:
            v += n * view.market_prices.get(item, 10) * 0.8
    for _, _, t in view.me.animals():
        v += 0.4 * ANIMALS.get(t.get("animal"), {}).get("cost", 300)
    for _, _, t in view.me.plants():
        crop = CROPS.get(t.get("crop"))
        if crop:
            v += 0.5 * t.get("yield_units", 0) * view.market_prices.get(t["crop"], 20)
    return v


def simulate_plan(obs, plan, day0):
    """Roll `plan` forward SIM_DAYS in a locally reconstructed engine."""
    from kaggle_environments import make

    player = obs["player"]
    env = make("kaggriculture", configuration={"episodeSteps": SIM_DAYS * TURNS_PER_DAY + 2}, debug=False)
    env.reset()
    step0 = day0 * TURNS_PER_DAY

    obs0 = env.state[0]["observation"]
    farms = copy.deepcopy(obs["farms"])
    for farm in farms:
        _shift_state(farm, day0, step0)
    obs0["farms"] = farms
    obs0["market"] = copy.deepcopy(obs["market"])
    obs0["town"] = copy.deepcopy(obs["town"])
    # Our private state is known; the opponent's shed is hidden — give the
    # proxy a plausible empty one (their public farm still drives behavior).
    env.state[player]["observation"]["private"] = copy.deepcopy(obs["private"])

    ours = _sim_agent(lambda: _PinnedPlanProvider(plan), day0)
    proxy = _sim_agent(lambda: None, day0)
    agents = [ours, proxy] if player == 0 else [proxy, ours]

    for _ in range(SIM_DAYS * TURNS_PER_DAY):
        if env.done:
            break
        actions = []
        for i, fn in enumerate(agents):
            try:
                actions.append(fn(_agent_obs(env, i)))
            except Exception:
                actions.append({"farmer": ["PASS"], "hands": [], "market": []})
        env.step(actions)
    return _score(_agent_obs(env, player), player)


def _agent_obs(env, i):
    shared = env.state[0]["observation"]
    own = env.state[i]["observation"]
    obs = dict(shared)
    obs["player"] = i
    obs["private"] = own.get("private", {})
    obs["step"] = len(env.steps) - 1
    return obs


def _perturb(choices, deltas):
    out = list(choices)
    for head_idx, d in deltas.items():
        size = features.HEAD_SIZES[head_idx]
        out[head_idx] = max(0, min(size - 1, out[head_idx] + d))
    return out


class MPCPlanProvider(NeuralPolicy):
    """NeuralPolicy whose day plan is chosen among engine-simulated candidates."""

    def make_plan(self, view, memory):
        base_plan = super().make_plan(view, memory)
        if self.layers is None or self.last_decision is None:
            return base_plan
        # Budget guard: only at day boundaries, only with overage to spare.
        overage = getattr(memory, "overage_left", 60.0)
        if view.hour != 0 or overage < MIN_OVERAGE_LEFT:
            return base_plan

        head_idx = {name: i for i, (name, _) in enumerate(features.HEADS)}
        x, m, choices, logprob, value = self.last_decision
        candidates = [("base", choices)]
        candidates.append(
            ("livestock+", _perturb(choices, {head_idx["cow_target"]: 1,
                                              head_idx["sheep_target"]: 1}))
        )
        candidates.append(
            ("crops+", _perturb(choices, {head_idx["straw_tiles"]: 1,
                                          head_idx["wheat_tiles"]: 1}))
        )

        best_plan, best_score = base_plan, None
        raw_obs = getattr(memory, "last_raw_obs", None)
        if raw_obs is None:
            return base_plan
        for name, cand in candidates:
            try:
                plan = planner.make_day_plan(view, memory)
                plan = features.apply_to_plan(plan, cand, view)
                score = simulate_plan(raw_obs, plan, view.day)
            except Exception:
                continue
            if best_score is None or score > best_score:
                best_score, best_plan = score, (
                    base_plan if name == "base"
                    else features.apply_to_plan(planner.make_day_plan(view, memory), cand, view)
                )
        return best_plan
