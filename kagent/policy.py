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
from kagent import features, market, planner, scheduler
from kagent.banditpolicy import BanditPolicy, KnobBanditPolicy
from kagent.mpcpolicy import MPCPlanProvider
from kagent.nnpolicy import NeuralPolicy, decode_weights
from kagent.weights import WEIGHTS_DATA

# In a bundle, the weights module body shares this namespace, so an exported
# MPC flag lands in globals(); dev mode and legacy weights default to False.
MPC_ENABLED = globals().get("MPC_ENABLED", False)
# Champion-plan bundles run the decoded champion playbook (planner.
# make_champion_plan) as the primary policy, weights or not.
CHAMPION_PLAN = globals().get("CHAMPION_PLAN", False)
# Residual bundles: weights are offset heads over the champion plan.
RESIDUAL = globals().get("RESIDUAL", False)
# Bandit bundles: fitted per-(head, offset) gain models over the champion plan.
BANDIT_RULES = globals().get("BANDIT_RULES", None)
# Executor rules this bundle ships with (kagent/scheduler.EXEC_FLAGS), written
# into kagent/weights.py by `rl/export.py --exec`. Absent in dev mode, where
# the flags stay off and each eval arm sets its own via the rules JSON, so
# this is inert outside a bundle.
EXEC_RULES = globals().get("EXEC_RULES", None)
if EXEC_RULES:
    (globals().get("set_exec") or scheduler.set_exec)(EXEC_RULES)
if BANDIT_RULES:
    (globals().get("set_mode") or features.set_mode)("residual")
if RESIDUAL:
    # In a flat bundle the module aliases (features = namespace) are bound
    # only after every source has executed, so `features.set_mode` is a
    # NameError here; the bare function IS in the shared namespace, and
    # switching now (before the namespace snapshot) makes the residual head
    # set visible everywhere. Dev mode falls back to the package import.
    (globals().get("set_mode") or features.set_mode)("residual")


class ChampionPlanProvider:
    def make_plan(self, view, memory):
        return planner.make_day_plan(view, memory, speculative=True)


class HeuristicPolicy:
    def __init__(self, plan_provider=None):
        self.plan_provider = plan_provider

    def make_plan(self, view, memory):
        if self.plan_provider is not None:
            return self.plan_provider.make_plan(view, memory)
        return planner.make_day_plan(view, memory)

    def act(self, view, memory):
        if memory.day_plan is None or memory.day_plan_day != view.day:
            # Roll the labor-utilization ledger before the new day's plan so
            # the hires head sees yesterday's idle payroll.
            if memory.util_day >= 0 and memory.turns_accum > 0:
                memory.yesterday_idle = memory.idle_accum / memory.turns_accum
                memory.yesterday_hands = memory.hands_peak
            memory.idle_accum = 0
            memory.turns_accum = 0
            memory.hands_peak = 0
            memory.util_day = view.day
            memory.day_plan = self.make_plan(view, memory)
            memory.day_plan_day = view.day
        plan = memory.day_plan

        tasks = scheduler.build_tasks(view, plan)
        plan.fert_reserve = sum(1 for t in tasks if t.op[0] == "FERTILIZE")
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

        for op in actions.values():
            memory.turns_accum += 1
            if not op or op[0] == "PASS":
                memory.idle_accum += 1
        memory.hands_peak = max(memory.hands_peak, len(view.me.units) - 1)

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
            # MPC support: the raw observation and the remaining overage
            # budget let the day-planner simulate candidates safely.
            self.memory.last_raw_obs = observation
            self.memory.overage_left = (
                observation.get("remainingOverageTime", 60.0)
                if hasattr(observation, "get")
                else 60.0
            )
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
if BANDIT_RULES:
    _bandit_cls = KnobBanditPolicy if BANDIT_RULES.get("kind") == "knobs" else BanditPolicy
    _AGENT = SafeAgent(
        HeuristicPolicy(_bandit_cls(BANDIT_RULES)), fallback=HeuristicPolicy(ChampionPlanProvider())
    )
elif CHAMPION_PLAN:
    _AGENT = SafeAgent(HeuristicPolicy(ChampionPlanProvider()), fallback=HeuristicPolicy())
elif _LAYERS is not None:
    if MPC_ENABLED:
        _provider = MPCPlanProvider(_LAYERS)
    else:
        _provider = NeuralPolicy(_LAYERS, residual=RESIDUAL)
    _fallback = HeuristicPolicy(ChampionPlanProvider()) if RESIDUAL else HeuristicPolicy()
    _AGENT = SafeAgent(HeuristicPolicy(_provider), fallback=_fallback)
else:
    _AGENT = SafeAgent(HeuristicPolicy())


def agent(observation, configuration=None):
    return _AGENT.act_safe(observation, configuration)
