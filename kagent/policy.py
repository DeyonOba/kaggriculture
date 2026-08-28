"""Glue: HeuristicPolicy pipeline + SafeAgent wrapper + `agent` entrypoint.

The bundler guarantees `agent` is the LAST callable defined in the built
main.py (the Kaggle loader picks the last callable in the module).

If kagent/weights.py carries trained weights, the day plan's strategic fields
come from the neural planner (kagent/nnpolicy.py); execution stays scripted.
"""

import sys
import time
import traceback

from kagent.gamestate import Memory, View
from kagent import market, planner, scheduler
from kagent.nnpolicy import NeuralPolicy, decode_weights
from kagent.weights import WEIGHTS_DATA


class HeuristicPolicy:
    def __init__(self, plan_provider=None):
        self.plan_provider = plan_provider

    def make_plan(self, view, memory):
        if self.plan_provider is not None:
            return self.plan_provider.make_plan(view, memory)
        return planner.make_day_plan(view, memory)

    def act(self, view, memory):
        if memory.day_plan is None or memory.day_plan_day != view.day:
            memory.day_plan = self.make_plan(view, memory)
            memory.day_plan_day = view.day
        plan = memory.day_plan

        tasks = scheduler.build_tasks(view, plan)
        actions = scheduler.assign(view, plan, tasks)

        # Consume day quotas as ops are actually emitted.
        for op in actions.values():
            if not op:
                continue
            if op[0] == "PLANT" and plan.plant_quota.get(op[1], 0) > 0:
                plan.plant_quota[op[1]] -= 1
            elif op[0] == "BUILD_COOP":
                plan.structure_builds["COOP"] = max(0, plan.structure_builds["COOP"] - 1)
            elif op[0] == "BUILD_PASTURE":
                plan.structure_builds["PASTURE"] = max(0, plan.structure_builds["PASTURE"] - 1)

        orders = market.build_orders(view, plan, memory)

        n_hands = len(view.me.units) - 1
        return {
            "farmer": actions.get(0, ["PASS"]),
            "hands": [actions.get(i + 1, ["PASS"]) for i in range(n_hands)],
            "market": orders,
        }


class SafeAgent:
    """Never raise, never emit an invalid action, track step timing.

    Degradation ladder: neural planner -> pure heuristic -> PASS.
    """

    def __init__(self, policy, fallback=None):
        self.policy = policy
        self.fallback_policy = fallback
        self.memory = Memory()

    def _fallback(self, observation):
        try:
            farm = observation["farms"][observation["player"]]
            n_hands = len(farm.get("hands", []))
        except Exception:
            n_hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}

    def act_safe(self, observation, configuration=None):
        t0 = time.perf_counter()
        try:
            view = View(observation)
            self.memory.observe(view)
            action = self.policy.act(view, self.memory)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            if self.fallback_policy is not None:
                try:
                    self.memory.day_plan = None  # rebuild via heuristic
                    view = View(observation)
                    return self.fallback_policy.act(view, self.memory)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            return self._fallback(observation)
        ms = (time.perf_counter() - t0) * 1000.0
        if ms > self.memory.timing_max_ms:
            self.memory.timing_max_ms = ms
        return action


_LAYERS = decode_weights(WEIGHTS_DATA)
if _LAYERS is not None:
    _AGENT = SafeAgent(
        HeuristicPolicy(NeuralPolicy(_LAYERS)), fallback=HeuristicPolicy()
    )
else:
    _AGENT = SafeAgent(HeuristicPolicy())


def agent(observation, configuration=None):
    return _AGENT.act_safe(observation, configuration)
