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
from rl.execnet import features2 as F2
from rl.execnet.model2 import is_v2, load as load2

CHORES = {"PASS", "WATER", "HARVEST", "FEED", "CARE", "FERTILIZE", "DIG", "COLLECT_FERTILIZER", "DROP"}
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
OVERRIDE = CHORES | MOVE_OPS
# take-over modes: "chores" = the model decides what to do on the tile it
# stands on and never where to walk; "all" = chores and moves; "full" = all
# plus PLANT / PICKUP resolved with the plan's quota and the farm's needs
# "dest" (v2): the model chooses a DESTINATION among the scheduler's pending
# requests (plus the shed, plus staying put); the scheduler keeps its
# logistics passes (shuttles, provisioning, survival) and does the walking.
MODES = ("chores", "all", "full", "dest")
# scheduler passes whose assignment the dest mode never overrides
KEEP_PASSES = {"shuttle", "provision", "morning", "survival"}
_MODEL_CACHE = {}


def load_model(path):
    """-> (model, cond) where cond is the elite conditioning vector of the
    'conditioned' v2 variant (the leaders' z-scored executor profile, which
    is what we clamp the model to at play time), or None."""
    if path not in _MODEL_CACHE:
        torch.set_num_threads(1)
        if is_v2(path):
            model, extra = load2(path)
            c = (extra or {}).get("cond")
            cond = np.asarray(c["elite_z"], dtype=np.float32) if c else None
        else:
            model, cond = load(path)[0], None
        _MODEL_CACHE[path] = (model, cond)
    return _MODEL_CACHE[path]


class ExecNetPolicy(HeuristicPolicy):
    def __init__(self, model, plan_provider=None, conf=0.0, mode="chores", cond=None):
        super().__init__(plan_provider)
        if mode not in MODES:
            raise ValueError(f"execnet mode {mode!r}; known {MODES}")
        self.model = model
        self.cond = cond
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
        if self.mode == "dest":
            prev_enabled = scheduler.MOVE_STATS_ENABLED
            scheduler.MOVE_STATS_ENABLED = True
            scheduler.ASSIGN_LABEL.clear()
            actions = scheduler.assign(view, plan, tasks)
            labels = dict(scheduler.ASSIGN_LABEL)
            scheduler.MOVE_STATS_ENABLED = prev_enabled
            self._override_dest(view, plan, tasks, actions, labels)
        else:
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


    def _task_op(self, view, plan, idx, task, emitted):
        """The op of a pending task, if this unit can execute it now."""
        op = list(task.op)
        inv = view.unit_inv(idx)
        if task.need and inv.get(task.need, 0) < 1:
            return None
        if op[0] == "PLANT":
            crop = op[1]
            allowed = min(view.seeds.get(crop, 0), plan.plant_quota.get(crop, 0))
            if emitted[crop] + 1 > allowed:
                return None
            emitted[crop] += 1
        return op

    @torch.no_grad()
    def _override_dest(self, view, plan, tasks, actions, labels):
        """v2: pick each free unit's DESTINATION among the pending requests."""
        units = view.me.units
        if not units:
            return
        by_tile = {}
        for t in tasks:
            by_tile.setdefault((t.x, t.y), []).append(t)
        for v in by_tile.values():
            v.sort(key=lambda t: t.pri)
        req_tiles = set(by_tile)
        free = [ui for ui, (idx, _, _) in enumerate(units)
                if labels.get(idx) not in KEEP_PASSES
                and not any(view.unit_inv(idx).get(a, 0) for a in ("GOOSE", "COW", "SHEEP"))]
        if not free:
            self.stats["dest_no_free_units"] += 1
            return
        board = F2.board_planes(view)
        req, _ = F2.request_planes(tasks)
        base = np.asarray(board + req, dtype=np.float32)
        planes, scal, masks = [], [], []
        counts = F.farm_counts(view)
        for ui in free:
            _, x, y = units[ui]
            up = np.zeros((1, F2.N_TILES), dtype=np.float32)
            up[0, F2.tile_index(x, y)] = 1.0
            planes.append(np.concatenate([base, up], axis=0).reshape(F2.N_PLANE, F2.BOARD, F2.BOARD))
            sv = F.unit_features(view, ui, counts)[1]
            if self.cond is not None:
                sv = list(sv) + list(self.cond)
            scal.append(sv)
            masks.append(F2.dest_mask(view, ui, req_tiles))
        mask = np.asarray(masks, dtype=bool)
        logits, op_logits = self.model(torch.tensor(np.asarray(planes, dtype=np.float32)),
                                       torch.tensor(np.asarray(scal, dtype=np.float32)),
                                       torch.tensor(mask))
        probs = torch.softmax(logits, dim=-1).numpy()
        emitted = collections.Counter(op[1] for op in actions.values() if op and op[0] == "PLANT")
        taken = set()
        order = sorted(range(len(free)), key=lambda i: -probs[i].max())
        for i in order:
            ui = free[i]
            idx, x, y = units[ui]
            p = probs[i].copy()
            for t in taken:
                p[t] = 0.0
            k = int(p.argmax())
            if p[k] < self.conf or not mask[i][k]:
                self.stats["kept_low_conf"] += 1
                continue
            if k == F2.DEST_STAY:
                # A leader's STAY means "you are already where the work is".
                # Our units are often standing on nothing, and honouring STAY
                # there just idles them ($5.7k vs $75.9k on a traced game), so
                # STAY is only obeyed when there IS work underfoot; otherwise
                # the scheduler's action stands.
                here = by_tile.get((x, y), [])
                op = next((o for o in (self._task_op(view, plan, idx, t, emitted) for t in here) if o), None)
                if op is None:
                    self.stats["stay_no_work_kept_scheduler"] += 1
                    continue
                new = op
            elif k == F2.DEST_SHED:
                base_op = actions.get(idx)
                if base_op and base_op[0] in ("PICKUP", "DROP"):
                    self.stats["agree:SHED"] += 1
                    continue
                if not sum(view.unit_inv(idx).values()) and view.shed_total == 0:
                    self.stats["shed_pointless_kept_scheduler"] += 1
                    continue
                tx, ty = scheduler._nearest_shed_tile(x, y)
                if (x, y) == (tx, ty):
                    new = ["DROP"] if sum(view.unit_inv(idx).values()) else ["PASS"]
                else:
                    step = scheduler._step_toward(x, y, tx, ty)
                    new = [step] if step else ["PASS"]
            else:
                tx, ty = k % F2.BOARD, k // F2.BOARD
                taken.add(k)
                if (x, y) == (tx, ty):
                    here = by_tile.get((tx, ty), [])
                    op = next((o for o in (self._task_op(view, plan, idx, t, emitted) for t in here) if o), None)
                    new = op or ["PASS"]
                else:
                    step = scheduler._step_toward(x, y, tx, ty)
                    new = [step] if step else ["PASS"]
            old = actions.get(idx)
            if old and old == new:
                self.stats["agree:" + new[0]] += 1
            else:
                self.stats["override:" + new[0]] += 1
            actions[idx] = new


PLANNERS = ("champion", "knobs", "weights")


def make_plan_provider(planner="champion", rules=None, weights=None):
    """The DAY PLAN half of an execnet agent -> (provider, switched, needs_board).

        "champion"  the dev tree's champion plan (the default, unchanged)
        "knobs"     the knob bandit over it; `rules` is a BANDIT_RULES dict
        "weights"   a neural day-planner; `weights` is an rl/common.to_layers
                    JSON. When those layers carry a board trunk the caller
                    MUST set `provider.board` before every plan is made -- see
                    rl/rollout._make_dev_policy_agent, which mirrors what
                    _make_learner_agent does for a plain weights arm. Without
                    it nnpolicy silently drops the board columns.

    `switched` is True when features was flipped to residual mode, so the
    caller can flip it back for the next job in the worker.
    """
    from kagent import features
    from kagent.banditpolicy import BanditPolicy, KnobBanditPolicy
    from kagent.nnpolicy import NeuralPolicy
    from kagent.policy import ChampionPlanProvider

    if planner not in PLANNERS:
        raise ValueError(f"execnet planner {planner!r}; known {PLANNERS}")
    switched = False
    if planner == "weights":
        import json as _json
        import os as _os
        with open(weights) as fh:
            layers = _json.load(fh)
        residual = _os.environ.get("KAGG_RESIDUAL") == "1"
        if residual and features.MODE != "residual":
            features.set_mode("residual")
            switched = True
        provider = NeuralPolicy(layers, temperature=0.0, rng=None, residual=residual)
        return provider, switched, layers["policy"].get("board") is not None
    if planner == "knobs" and rules:
        if features.MODE != "residual":
            features.set_mode("residual")
            switched = True
        cls = KnobBanditPolicy if rules.get("kind") == "knobs" else BanditPolicy
        return cls(rules), switched, False
    return ChampionPlanProvider(), switched, False


def parse_arm_fields(fields):
    """The `dev:execnet:` arm tail, shared by rl/session_eval.py,
    tools/game_trace.py and rl/rollout.py so the three cannot drift.

        <execnet.pt>[:<planner>[:<conf>[:<mode>]]]

    <planner> is a path: a rules JSON (the knob bandit, as before) or a
    to_layers weights JSON (a neural day-planner). They are told apart by
    content -- weights carry "policy", rules carry "kind"/"conditions" -- so
    every arm string that worked before parses to exactly what it did.
    Returns the `learner_policy` spec dict.
    """
    import json as _json
    import os as _os

    fields = list(fields)
    spec = {"kind": "execnet", "weights": _os.path.abspath(fields[0])}
    if len(fields) > 1 and fields[1]:
        with open(fields[1]) as fh:
            blob = _json.load(fh)
        if isinstance(blob, dict) and "policy" in blob:
            spec["planner"] = "weights"
            spec["planner_weights"] = _os.path.abspath(fields[1])
        else:
            spec["planner"] = "knobs"
            spec["rules"] = blob
    if len(fields) > 2 and fields[2]:
        spec["conf"] = float(fields[2])
    if len(fields) > 3 and fields[3]:
        spec["mode"] = fields[3]
    return spec


def load_policy(weights, plan_provider=None, conf=0.0, mode="chores"):
    model, cond = load_model(weights)
    return ExecNetPolicy(model, plan_provider, conf=conf, mode=mode, cond=cond)
