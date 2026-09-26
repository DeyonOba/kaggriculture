"""ExecNetPolicy: the planner's day plan and market orders exactly as
kagent/policy.HeuristicPolicy, with scheduler.assign's action for a unit
replaced by the model's masked argmax when the model is confident and the
op is one it may take over (moves, PASS, WATER, HARVEST, FEED, CARE,
FERTILIZE, DIG, COLLECT_FERTILIZER, DROP). Ops that need an item or a crop
choice (PLANT, PICKUP, PLACE, BUILD_*) stay with the scheduler: when the
model wants one of those and the scheduler's action is not that op, the
scheduler's action stands.
"""
from __future__ import annotations

import collections

import numpy as np
import torch

from kagent import market, scheduler
from kagent.policy import HeuristicPolicy
from rl.execnet import features as F
from rl.execnet.model import load

CHORES = {"PASS", "WATER", "HARVEST", "FEED", "CARE", "FERTILIZE", "DIG", "COLLECT_FERTILIZER", "DROP"}
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
OVERRIDE = CHORES | MOVE_OPS
# take-over modes: "chores" = the model decides what to do on the tile it
# stands on and never where to walk; "all" = chores and moves; "full" = all
# plus PLANT / PICKUP resolved with the plan's quota and the farm's needs
MODES = ("chores", "all", "full")
_MODEL_CACHE = {}


def load_model(path):
    if path not in _MODEL_CACHE:
        torch.set_num_threads(1)
        model, _ = load(path)
        _MODEL_CACHE[path] = model
    return _MODEL_CACHE[path]


class ExecNetPolicy(HeuristicPolicy):
    def __init__(self, model, plan_provider=None, conf=0.0, mode="chores"):
        super().__init__(plan_provider)
        if mode not in MODES:
            raise ValueError(f"execnet mode {mode!r}; known {MODES}")
        self.model = model
        self.conf = conf
        self.mode = mode
        self.stats = collections.Counter()
        self._reported = False

    def act(self, view, memory):
        if memory.day_plan is None or memory.day_plan_day != view.day:
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
        self._override(view, plan, actions)
        if view.step >= 717 and not self._reported:
            self._reported = True
            import sys
            print("execnet stats:", dict(self.stats), file=sys.stderr, flush=True)

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

    def _resolve_item_op(self, view, plan, idx, op, counts):
        """'full' mode: the model's PLANT / PICKUP given an item from the plan
        and the farm's needs; None when nothing sensible is available."""
        if op == "PLANT":
            quota = {c: n for c, n in plan.plant_quota.items() if n > 0 and view.seeds.get(c, 0) > 0}
            if not quota:
                return None
            crop = max(quota, key=quota.get)
            return ["PLANT", crop]
        if op == "PICKUP":
            unfed = counts["unfed"] > 0
            if unfed and view.shed.get("WHEAT", 0) > 0:
                return ["PICKUP", "WHEAT", min(5, view.shed["WHEAT"])]
            if view.shed.get("FERTILIZER", 0) > 0 and plan.fert_reserve > 0:
                return ["PICKUP", "FERTILIZER", min(3, view.shed["FERTILIZER"])]
            return None
        return None

    @torch.no_grad()
    def _override(self, view, plan, actions):
        units = view.me.units
        if not units:
            return
        counts = F.farm_counts(view)
        tiles, scal, masks = [], [], []
        for ui in range(len(units)):
            t, s = F.unit_features(view, ui, counts)
            tiles.append(t)
            scal.append(s)
            masks.append(F.legal_mask(view, ui))
        tiles = torch.tensor(np.asarray(tiles, dtype=np.float32))
        scal = torch.tensor(np.asarray(scal, dtype=np.float32))
        mask = torch.tensor(np.asarray(masks, dtype=np.bool_))
        probs = torch.softmax(self.model(tiles, scal, mask), dim=-1)
        best = probs.argmax(-1)
        plants_emitted = collections.Counter(op[1] for op in actions.values() if op and op[0] == "PLANT")
        for ui, (idx, _, _) in enumerate(units):
            k = int(best[ui])
            p = float(probs[ui, k])
            op = F.OPS[k]
            base = actions.get(idx)
            if p < self.conf or not masks[ui][k]:
                self.stats["kept_low_conf" if p < self.conf else "kept_illegal"] += 1
                continue
            if base and base[0] == op:
                self.stats["agree:" + op] += 1
                continue
            if op in MOVE_OPS and self.mode == "chores":
                self.stats["deferred_move"] += 1
                continue
            if op in CHORES or op in MOVE_OPS:
                actions[idx] = [op]
                self.stats["override:" + op] += 1
                continue
            if self.mode == "full" and op in ("PLANT", "PICKUP"):
                new = self._resolve_item_op(view, plan, idx, op, counts)
                if new is None:
                    self.stats["deferred_item_op"] += 1
                    continue
                if new[0] == "PLANT":
                    crop = new[1]
                    allowed = min(view.seeds.get(crop, 0), plan.plant_quota.get(crop, 0))
                    if plants_emitted[crop] + 1 > allowed:
                        self.stats["deferred_item_op"] += 1
                        continue
                    plants_emitted[crop] += 1
                actions[idx] = new
                self.stats["override:" + op] += 1
                continue
            self.stats["deferred_item_op"] += 1


def load_policy(weights, plan_provider=None, conf=0.0, mode="chores"):
    return ExecNetPolicy(load_model(weights), plan_provider, conf=conf, mode=mode)
