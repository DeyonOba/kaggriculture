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
from kagent import boardstate, features, market, planner, scheduler, executor2, sizing
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
# Theta bundles: a SEARCHED strategy vector (kagent/theta.py), every knob fixed
# from day 0. Written by `rl/export.py --theta <best_theta.json>`. It takes
# precedence over BANDIT_RULES because the two are alternatives, not layers --
# the bandit decides four knobs conditionally at days 6/9/12/15 and the vector
# fixes all twenty from the first plan, so running both would mean the gate
# could not say which chose what.
THETA_KNOBS = globals().get("THETA_KNOBS", None)
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


class ThetaPlanProvider:
    """The champion plan with a FIXED knob vector, set before the first plan.

    The knob bandit (`KnobBanditPolicy`) decides knobs at days 6/9/12/15 from
    fitted gain models, so day 0-5 always runs `KNOB_DEFAULTS` and the opening
    is not reachable through it. This provider sets the whole vector once, on
    the first plan of the game, which is what a SEARCH over the strategy needs:
    every coordinate live from the first decision, nothing conditional in the
    way, so a gate attributes to the vector and not to which rules fired.

    See kagent/theta.py for the vector and what is in it.
    """

    def __init__(self, knobs):
        self.knobs = dict(knobs or {})

    def make_plan(self, view, memory):
        if getattr(memory, "knobs", None) != self.knobs:
            memory.knobs = dict(self.knobs)
        return planner.make_day_plan(view, memory, speculative=True)


class HeuristicPolicy:
    def __init__(self, plan_provider=None):
        self.plan_provider = plan_provider

    def make_plan(self, view, memory):
        if self.plan_provider is not None:
            return self.plan_provider.make_plan(view, memory)
        return planner.make_day_plan(view, memory)

    def act(self, view, memory):
        # Board bundles only (`provider._bundle_arm`, set below when the
        # shipped weights carry a trunk). The dev/rollout paths set
        # provider.board themselves, so this must never run there.
        provider = self.plan_provider
        arm = getattr(provider, "_bundle_arm", None)
        if arm:
            obs = getattr(memory, "last_raw_obs", None)
            if obs is not None:
                boardstate.observe(memory, view, obs)
                # Built only on the step a plan is made -- the tensor is ~6
                # board passes and there are 30 plans in 720 steps. On the
                # other steps the provider gets None, so a plan made
                # unexpectedly mid-day raises instead of reading a stale board.
                need_plan = memory.day_plan is None or memory.day_plan_day != view.day
                provider.board = (boardstate.build(memory, view, obs, arm=arm)
                                  if need_plan else None)
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
            if scheduler.EXEC.get("replant"):
                memory.replant_added = scheduler.replant_quota(view, memory.day_plan)
            if any(scheduler.EXEC.get(f) for f in ("sized", "size_budget", "size_feed", "size_hires", "size_ground")):
                sizing.approve_plan(view, memory.day_plan, memory)
        plan = memory.day_plan
        if scheduler.EXEC.get("sized") or scheduler.EXEC.get("size_ground"):
            sizing.refill(view, plan, memory)

        tasks = scheduler.build_tasks(view, plan)
        plan.fert_reserve = sum(1 for t in tasks if t.op[0] == "FERTILIZE")
        plan.approved = bool(scheduler.EXEC.get("approve"))
        if scheduler.EXEC.get("executor2"):
            actions = executor2.assign2(view, plan, tasks, memory)
        else:
            actions = scheduler.assign(view, plan, tasks, memory)

        # Consume day quotas as ops are actually emitted.
        for idx, op in actions.items():
            if not op:
                continue
            if op[0] == "PLANT":
                ux, uy = next(((x, y) for i, x, y in view.me.units if i == idx), (0, 0))
                qd = scheduler._quota_for(plan, ux, uy)
                if qd.get(op[1], 0) > 0:
                    qd[op[1]] -= 1
            elif op[0] == "FERTILIZE" and getattr(plan, "fert_budget", None) is not None:
                plan.fert_budget = max(0, plan.fert_budget - 1)
            elif op[0] == "BUILD_COOP":
                plan.structure_builds["COOP"] = max(0, plan.structure_builds["COOP"] - 1)
            elif op[0] == "BUILD_PASTURE":
                plan.structure_builds["PASTURE"] = max(0, plan.structure_builds["PASTURE"] - 1)

        # carechain: remember each unit's op and tile this turn
        memory.last_ops = {idx: (op[0], (x, y)) for idx, x, y in view.me.units for op in [actions.get(idx) or ["PASS"]]}
        orders = market.build_orders(view, plan, memory)
        if getattr(plan, "approved", False):
            sizing.record_spend(view, memory, orders)

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
if THETA_KNOBS:
    _AGENT = SafeAgent(
        HeuristicPolicy(ThetaPlanProvider(THETA_KNOBS)),
        fallback=HeuristicPolicy(ChampionPlanProvider()),
    )
elif BANDIT_RULES:
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
        # A trunk-carrying planner must be handed the board tensor before it
        # plans. Until the export path shipped the trunk this was impossible
        # and board nets were refused at export; now the bundle builds the
        # tensor itself. The arm is read off the trunk so the width cannot
        # disagree with the weights.
        _board_layers = (_LAYERS.get("policy") or {}).get("board")
        if _board_layers is not None:
            # bare name, not `boardstate.`: in a flat bundle the module
            # aliases are bound only after every source has executed, so a
            # qualified reference is a NameError at import time (the same
            # reason set_mode/set_exec are looked up this way above).
            _arm_fn = globals().get("arm_for_trunk") or boardstate.arm_for_trunk
            _provider._bundle_arm = _arm_fn(_board_layers)
    _fallback = HeuristicPolicy(ChampionPlanProvider()) if RESIDUAL else HeuristicPolicy()
    _AGENT = SafeAgent(HeuristicPolicy(_provider), fallback=_fallback)
else:
    _AGENT = SafeAgent(HeuristicPolicy())


def agent(observation, configuration=None):
    return _AGENT.act_safe(observation, configuration)
