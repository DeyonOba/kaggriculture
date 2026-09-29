"""rlv3.agent -- the four components wired into ONE agent.

    objective   what "better" means, lexicographically (win, then profit,
                then loss margin), and the deterministic both-banks projection
                that makes it computable mid-season.
    allocation  what to plant, what to buy, how many hands, whether to buy
                land -- re-derived every dawn from the board rather than
                scheduled, with the opponent's rent subtracted.
    selling     when to sell, how much, and in which market-order slot.
    executor    Maia2: movement, chore order and placement, at the
                `kagent.scheduler.build_tasks / assign` seam.

SHAPE. `Rlv3PlanProvider` duck-types `kagent.policy.ChampionPlanProvider`, so
the whole wiring is `HeuristicPolicy(Rlv3PlanProvider(cfg))` -- the same shape
`rl.rollout._make_dev_policy_agent` already builds for `dev:champion` and
`dev:knobs`. `make_dev_agent(spec)` returns exactly what that function
returns, `(agent_fn, switched_mode)`, so adopting rlv3 upstream is one line
(see `rlv3/arm.py`).

ORDER OF OPERATIONS, and it is load-bearing.

    every step        ProjState.observe(view)      solve their market flow
    hour 0 only       ProjState.dawn(view)         residual -> two EWMAs
                      allocation.make_plan         land gates, then targets
                      objective.project(plan=...)  record/rank the outlook
                      mark memory for selling      the sell cfg for today
    every step        kagent.scheduler.build_tasks / assign   (executor seam)
    every step        kagent.market.build_orders   (selling seam)

`make_plan` runs at hour 0 and only at hour 0, because that is the only hour
`HeuristicPolicy.act` ever builds a day plan. Every planner decision in this
package is therefore a DAWN decision; the two seams are what act in between.

WHAT THE DEFAULT PRESET IS, and why it is not the maximal one. Each component
ran its own pre-gate and reported honestly:

    allocation  the CONSTANT version of this idea (mix_mild) is already
                -1.5 pts on the live draw; the dose ladder has not run.
    selling     the sizing half is margin-NEGATIVE on top of the slot move
                (+$891 vs +$1,281 on 2 traces); only the slot re-order is a
                transfer rather than a lift of one shared curve.
    executor    the movement budget optimised perfectly is worth about $0
                (kaggriculture-executor-bounded), and `mv_dest` and `pl_net`
                are already measured refuted and inert respectively.

So `PRESETS["base"]` is the IDENTITY arm -- every deviation at zero dose,
which must reproduce ship2_hv bit for bit, and is the thing to check first
whenever a result looks surprising. `PRESETS["recommended"]` adds only what
its own pre-gate supported. The doses live in `PRESETS` and on the spec, not
in the code.

STATE ON `Memory`, and the collision that was there. `kagent.gamestate.Memory`
has no `__slots__`, so components attach their own attributes:

    memory.rlv3            `rlv3.land.observe_dawn`'s measurement ledger, a
                           DICT, and `rlv3.allocation.chore_cost` reads
                           `memory.rlv3["chore"]` off it.
    memory.rlv3_sell       `rlv3.selling.SellState`
    memory.rlv3_sell_cfg   `rlv3.selling.SellConfig` for today
    memory.rlv3_proj       `rlv3.objective.ProjState`  <- THIS MODULE

`objective.ProjState`'s docstring said it lives at `memory.rlv3`. It does not
and it cannot: `land.observe_dawn` re-creates that attribute as a plain dict
every dawn whenever it is not one, so a ProjState parked there would be
silently destroyed on day 1 and `allocation.chore_cost` would raise on
`.get("chore")`. It lives at `memory.rlv3_proj`; the docstring is corrected.
"""

from __future__ import annotations

import collections
import copy
import json
import os

from kagent import features, planner, scheduler
from kagent.gamestate import Memory, View
from kagent.policy import HeuristicPolicy

from rlv3 import allocation as ALLOC
from rlv3 import objective as OBJ
from rlv3 import selling as SELL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The baseline rl/session_eval.py actually pairs against at 0.354: the dev
# tree with champ-h10's four knobs. NOT arena/agents/champ-h10/main.py, which
# is a snapshot of an OLDER kagent/ and finishes $11,638 away from today's
# tree on world 770000 (measured by the allocation component).
BASE_RULES = os.path.join(REPO, "tools", "rules", "rules_h8_ship2_hv.json")
BASE_EXEC = ("lastwater", "h23", "harvest_visit")

# The rlv3 executor levers this package will actually put on a gate. The
# other five registered flags ship off on purpose: `mv_dest` is measured
# refuted (-10.3/-8.0/-21.5 per-op, -19.6/-13.7/-8.0 per-destination),
# `pl_net` is measured inert (prior dispersion cv 0.018 over 25 candidates),
# `pl_pasture_ring` is the diagnostic negative reproducing -$11.3k, and
# `pl_crop_cluster` / `pl_water_first` / `co_animal_first` / `co_value` have
# no pre-gate of their own yet.
# TURNED OFF 2026-09-29: every preset now runs the BASE SCHEDULER.
# `kaggriculture-executor-bounded` puts the whole movement budget, optimised
# perfectly, at about $0, and `rlv3/docs` records the kill switches disabling
# every lever within 2-4 days of a 30-day season -- so `exec`/`full`/`select`
# were measuring a day-0-only intervention on a lever worth ~nothing. With
# this empty, `install_seams` never calls `EXEC.install`, so build_tasks and
# assign are the shipped functions and the executor is champ-h10's.
# MAIA2_EXEC is kept, not deleted: the plan is to reintroduce it LATER as a
# randomness source over board movement, to test whether the planner actually
# self-corrects when the executor stops being deterministic.
MAIA2_EXEC = ("pl_wheat_adj", "co_chain_tile")
FULL_EXEC = ()

SELL_MODES = {
    "off": None,                              # shipped sizer, no patch at all
    "slots": "RECOMMENDED",                   # slot re-order only: +$1,281/game
    "paced": dict(paced_only=True),           # schedule market.PACED only
    "full": dict(),                           # the whole withholding schedule
}

PRESETS = {
    # name           lam  sell     exec         select
    "base": dict(lam=0.0, sell="off", exec_flags=(), select=False),
    "recommended": dict(lam=0.0, sell="slots", exec_flags=(), select=False),
    "alloc": dict(lam=1.0, sell="off", exec_flags=(), select=False),
    "exec": dict(lam=0.0, sell="off", exec_flags=FULL_EXEC, select=False),
    "sell": dict(lam=0.0, sell="full", exec_flags=(), select=False),
    "full": dict(lam=1.0, sell="slots", exec_flags=FULL_EXEC, select=False),
    "select": dict(lam=1.0, sell="slots", exec_flags=FULL_EXEC, select=True),
    # The readout presets. `readout` is the S0 arm: champion base, the learned
    # 3-day commitment on top, pricing at NEUTRAL risk so agreement with the
    # top boards is measured before any posture is applied. `readout_risk`
    # is the same with the posture derived from the margin.
    "readout": dict(lam=0.0, sell="off", exec_flags=(), select=False,
                    readout="default", posture="neutral"),
    "readout_risk": dict(lam=0.0, sell="off", exec_flags=(), select=False,
                         readout="default", posture="auto"),
}

# Doses the "select" preset ranks with objective.better(). lam=0 is always in
# the grid, so selection can always fall back to the identity plan.
DEFAULT_LAM_GRID = (0.0, 0.5, 1.0)


# ------------------------------------------------------------------ counters
# Module-level and per-episode, because "the arm was dead, not null" is the
# failure this package exists to avoid and a silent fallback to the shipped
# heuristic is indistinguishable from a null result. Every caller that reads
# a number off this agent should read COUNTERS too.
COUNTERS = collections.Counter()


# The most recently built agent, so a driver that does not own the call site
# (rl.rollout.run_episode returns only the result dict) can still put the
# process back: `install_seams` patches PROCESS-GLOBAL attributes, and a
# worker that plays an rlv3 arm and then a baseline arm must not carry the
# patches across. `rlv3.arm.run_episode` restores in a finally.
_LAST = {"agent": None}


def last_agent():
    return _LAST["agent"]


def counters():
    return dict(COUNTERS)


def reset_counters():
    COUNTERS.clear()


# --------------------------------------------------------------------- config
class Rlv3Config:
    """One arm, fully described. Everything is a dose; nothing is a code path."""

    __slots__ = ("lam", "sell", "exec_flags", "rules_path", "rules", "params",
                 "band", "select", "lam_grid", "project", "label", "land_theta",
                 "readout", "posture")

    def __init__(self, lam=0.0, sell="off", exec_flags=(), rules_path=None,
                 rules=None, params=None, band=None, select=False, lam_grid=None,
                 project=True, label="", land_theta=None, readout=None,
                 posture="neutral"):
        self.lam = float(lam)
        self.sell = sell
        self.exec_flags = tuple(exec_flags or ())
        self.rules_path = rules_path or BASE_RULES
        # An inline BANDIT_RULES dict wins over the path, so a caller can hand
        # this a rules blob lifted from a bundle without writing a file.
        self.rules = dict(rules) if isinstance(rules, dict) else None
        self.params = params
        self.band = band
        self.select = bool(select)
        self.lam_grid = tuple(lam_grid or DEFAULT_LAM_GRID)
        self.project = bool(project)
        self.label = label or "rlv3"
        self.land_theta = land_theta
        # The learned readout as the plan source (rlv3/readout.py): a weights
        # path, or None for the champion allocation. `posture` is the risk
        # read the pricing ORDERS the commitment with -- "neutral" (alpha 1,
        # the S0 agreement setting) or "auto" (risk.alpha_for_lead from the
        # margin). Both default OFF so every existing preset is unchanged.
        self.readout = readout
        self.posture = posture

    def load(self):
        """The knob rules: the inline dict, else read off disk once, in the
        process that will play."""
        if self.rules is None:
            with open(self.rules_path) as fh:
                self.rules = json.load(fh)
        return self.rules

    def sell_cfg(self):
        mode = SELL_MODES.get(self.sell, "missing")
        if mode == "missing":
            raise ValueError(f"unknown sell mode {self.sell!r}; "
                             f"known: {sorted(SELL_MODES)}")
        if mode is None:
            return None
        if mode == "RECOMMENDED":
            return SELL.RECOMMENDED
        return SELL.SellConfig(**mode)

    def all_exec(self):
        """The full EXEC flag list for this arm: the shipped three, then ours.

        The shipped three are champ-h10's own (`lastwater`, `h23`,
        `harvest_visit`) and are part of the BASELINE, not part of rlv3 -- an
        arm that drops them is not measuring rlv3, it is measuring their loss.
        """
        return list(BASE_EXEC) + [f for f in self.exec_flags
                                  if f not in BASE_EXEC]

    def to_json(self):
        return {"lam": self.lam, "sell": self.sell,
                "exec_flags": list(self.exec_flags),
                "rules_path": os.path.relpath(self.rules_path, REPO),
                "band": self.band, "select": self.select,
                "lam_grid": list(self.lam_grid), "label": self.label,
                "objective": OBJ.fingerprint()}


def config_from_spec(spec):
    """spec: {"kind": "rlv3", "preset": "...", <overrides>} -> Rlv3Config."""
    spec = dict(spec or {})
    spec.pop("kind", None)
    preset = spec.pop("preset", "recommended")
    if preset not in PRESETS:
        raise ValueError(f"unknown rlv3 preset {preset!r}; known: {sorted(PRESETS)}")
    fields = dict(PRESETS[preset])
    fields["label"] = f"rlv3:{preset}"
    for k, v in spec.items():
        # `exec` is ROLLOUT's copy of all_exec(), put on the job spec so
        # rl.rollout.run_episode applies and restores scheduler.EXEC itself.
        # `objective` is the fingerprint to_json() stamps for provenance.
        if k in ("exec", "objective"):
            continue
        if k not in Rlv3Config.__slots__:
            raise ValueError(f"unknown rlv3 spec field {k!r}; known: "
                             f"{sorted(Rlv3Config.__slots__)} + preset/kind/exec")
        fields[k] = v
    return Rlv3Config(**fields)


# ------------------------------------------------------------- plan provider
class Rlv3PlanProvider:
    """Duck-types `kagent.policy.ChampionPlanProvider`.

    `.calls` is the liveness counter the smoke test asserts on. A spec that
    reports itself as rlv3 while `.calls == 0` has fallen back to the shipped
    heuristic, and its result is DEAD, not null.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if isinstance(cfg, Rlv3Config) else config_from_spec(cfg or {})
        self.sell = self.cfg.sell_cfg()
        rules = self.cfg.load()
        self.inner = ALLOC.AllocationPlanProvider(
            knobs=rules, params=self.cfg.params, lam=self.cfg.lam)
        self.calls = 0
        self.last_alloc = None
        self.last_outlook = None
        self.outlooks = []          # one per dawn, for a post-hoc read
        self.chosen_lam = []        # one per dawn when select is on
        self.board = None           # rollout sets this on board arms; unused
        # The readout, loaded once. "default" resolves to the current best
        # weights so a preset does not pin a path that a retrain moves.
        self.readout = None
        self.last_delta = None
        self.deltas = []
        if self.cfg.readout:
            from rlv3 import readout as RO
            w = RO.DEFAULT_WEIGHTS if self.cfg.readout == "default" else self.cfg.readout
            self.readout = RO.Readout(w)

    # -- the projection state ------------------------------------------
    def proj(self, memory):
        st = getattr(memory, "rlv3_proj", None)
        if not isinstance(st, OBJ.ProjState):
            st = OBJ.ProjState(band=self.cfg.band)
            memory.rlv3_proj = st
        return st

    def observe(self, view, memory):
        """Every step. The opponent's market flow is only solvable from the
        DELTA of a shared inventory, so a skipped step is a lost day."""
        self.proj(memory).observe(view)
        COUNTERS["observe"] += 1

    # -- the dawn decision ---------------------------------------------
    def make_plan(self, view, memory):
        self.calls += 1
        COUNTERS["make_plan"] += 1
        st = self.proj(memory)
        st.dawn(view)
        COUNTERS["dawn"] += 1

        if self.cfg.select and view.day >= 1:
            plan = self._select(view, memory, st)
        else:
            plan = self.inner.make_plan(view, memory)
            self.last_alloc = self.inner.last_alloc
            self.chosen_lam.append(self.cfg.lam)

        if self.readout is not None:
            # The 3-day commitment: readout sets the PACE (what should be
            # standing three dawns on), the pricing sets the ORDER at the
            # configured posture. Land stays with the planner's own gate.
            from rlv3 import portfolio as PF, readout as RO
            pred = self.readout.predict(view, memory)
            if self.cfg.posture == "auto":
                ranked = PF.rank(view)
            else:
                ranked = PF.rank(view, 1.0, False)
            order = [r["item"] for r in ranked["rows"]]
            plan, delta = RO.commit(plan, view, pred, order=order)
            delta["alpha"], delta["upper"] = ranked["alpha"], ranked["upper"]
            self.last_delta = delta
            self.deltas.append(delta)
            COUNTERS["readout"] = COUNTERS.get("readout", 0) + 1

        if self.cfg.project:
            o = OBJ.project(view, memory, plan=plan, state=st,
                            band=self.cfg.band, label=self.cfg.label)
            self.last_outlook = o
            self.outlooks.append(o)
            COUNTERS["project"] += 1
            COUNTERS["projects_win"] += 1 if o.projects_win else 0

        # Mark the memory for the selling seam. `None` means "this arm does
        # not want rlv3 selling", and the patched builder then routes an
        # unmarked/None memory straight to the shipped sizer -- which is what
        # keeps a baseline arm sharing this worker process uncontaminated.
        memory.rlv3_sell_cfg = self.sell
        if self.sell is not None:
            SELL.state_for(memory)
            COUNTERS["sell_marked"] += 1
        return plan

    def _select(self, view, memory, st):
        """Rank candidate DAY PLANS with the lexicographic comparator.

        OFF BY DEFAULT AND IT SHOULD STAY OFF UNTIL A PAIRED GATE SAYS
        OTHERWISE. V-ranking is CLOSED in this repo -- four heads ordered
        actions at chance out of sample (0.503/0.500) -- and this is a
        value-ranking device over candidate plans wearing different clothes.
        What is different: the ranking function here is an arithmetic
        projection with no fitted parameter except two bounded EWMA
        multipliers, and it retrodicts 39/39 finished arms. What is the same:
        nothing on disk shows it orders MID-GAME plans correctly, and that is
        the claim it needs.

        Each candidate is built on a COPY of the memory's knob state so a
        rejected dose cannot leave its knobs behind, and the plan that wins is
        rebuilt on the real memory so the accepted dose's own side effects
        (the champion builder mutates `memory.knobs`) land exactly once.
        """
        cands = []
        snap = (copy.deepcopy(getattr(memory, "knobs", None)),
                getattr(memory, "knobs_decided", False),
                memory.day_plan, memory.day_plan_day)
        for lam in self.cfg.lam_grid:
            plan = ALLOC.AllocationPlanProvider(
                knobs=self.cfg.load(), params=self.cfg.params, lam=lam
            ).make_plan(view, memory)
            o = OBJ.project(view, memory, plan=plan, state=st, band=self.cfg.band,
                            label=f"lam{lam:g}")
            cands.append((lam, o))
            # The knob bandit decides once per season and the champion builder
            # writes `memory.knobs` as a side effect; a REJECTED dose must not
            # leave either behind, so every trial runs from the same state.
            memory.knobs, memory.knobs_decided = copy.deepcopy(snap[0]), snap[1]
            memory.day_plan, memory.day_plan_day = snap[2], snap[3]
        order = OBJ.rank([o for _, o in cands])
        lam = cands[order[0]][0]
        COUNTERS[f"select_lam{lam:g}"] += 1
        self.chosen_lam.append(lam)
        prov = (self.inner if lam == self.cfg.lam else ALLOC.AllocationPlanProvider(
            knobs=self.cfg.load(), params=self.cfg.params, lam=lam))
        plan = prov.make_plan(view, memory)
        self.last_alloc = prov.last_alloc
        return plan


# ------------------------------------------------------------------ the agent
def install_seams(cfg):
    """Process-global patches this arm needs. Returns a restore callable.

    MUST RUN IN THE PROCESS THAT PLAYS. `multiprocessing` starts spawn here,
    so a patch applied in the parent is invisible to an eval worker: that is
    exactly how the Stage B board arms were read as null results for three
    runs while never training at all. `make_dev_agent` is called inside
    `rl.rollout.run_episode`, i.e. inside the worker, which is why the install
    is here and not at import time.
    """
    from rlv3 import executor as EXEC

    undo = []
    # The executor seam is registered ALWAYS, even at zero flags, because
    # `rl.rollout.run_episode` calls `scheduler.set_exec(spec["exec"])` AFTER
    # this function returns, and `set_exec` raises ValueError on a flag it has
    # never heard of. Registering is not installing: with every rlv3 flag
    # False the seam is a verified pass-through.
    EXEC.register(scheduler)
    if cfg.exec_flags:
        EXEC.install(scheduler)
        EXEC.reset_stats()
        undo.append(EXEC.uninstall)
    if cfg.sell_cfg() is not None:
        # cfg=None: route ONLY memories this provider has marked.
        undo.append(SELL.install(None))

    def restore():
        for fn in reversed(undo):
            fn()

    return restore


def make_dev_agent(spec):
    """-> (agent_fn, switched_mode). The signature `rl.rollout` expects.

    Mirrors `rl.rollout._make_dev_policy_agent` exactly, including the
    residual-heads switch the knob bandit needs and the `last_view` handle
    `run_episode` reads `unsold_value` off.

    The returned callable also carries `.provider`, `.cfg` and `.restore` so a
    caller can assert liveness (`agent_fn.provider.calls`) and put the process
    back the way it found it.
    """
    cfg = spec if isinstance(spec, Rlv3Config) else config_from_spec(spec)
    switched = False
    if features.MODE != "residual":
        features.set_mode("residual")
        switched = True
    restore = install_seams(cfg)

    provider = Rlv3PlanProvider(cfg)
    pol = HeuristicPolicy(provider)
    mem = Memory()
    last_view = {"v": None}
    COUNTERS["episodes"] += 1

    def agent_fn(obs):
        view = View(obs)
        mem.observe(view)
        last_view["v"] = view
        provider.observe(view, mem)
        return pol.act(view, mem)

    agent_fn.last_view = last_view
    agent_fn.provider = provider
    agent_fn.cfg = cfg
    agent_fn.restore = restore
    _LAST["agent"] = agent_fn
    return agent_fn, switched


def make_job(spec, session_path, script_side, seed=0, temperature=0.0):
    """One `rl.rollout.run_episode` job playing `spec` against a recorded seat.

    `exec` is put on the job spec so ROLLOUT applies it, with its own
    save/restore of `scheduler.EXEC` around the episode -- rather than this
    module reaching into the scheduler and leaking flags into the next job in
    the worker.
    """
    cfg = spec if isinstance(spec, Rlv3Config) else config_from_spec(spec)
    # The caller's own spec dict travels VERBATIM (it is what `config_from_spec`
    # validated), plus the two fields rollout reads: `kind`, which routes it,
    # and `exec`, which rollout applies to `scheduler.EXEC` and restores after
    # the episode. Round-tripping through `to_json()` here would silently drop
    # `params`, `rules` and `land_theta`, which is the kind of quiet
    # half-configuration that makes an arm unreproducible.
    js = dict(spec) if isinstance(spec, dict) else cfg.to_json()
    js["kind"] = "rlv3"
    js["exec"] = cfg.all_exec()
    js.pop("objective", None)
    return {"weights_path": None, "learner_bundle": None, "learner_policy": js,
            "temperature": temperature, "seed": seed, "mode": "rl",
            "side": 1 - script_side,
            "opponent": ("session", session_path, script_side)}


__all__ = ["PRESETS", "SELL_MODES", "BASE_EXEC", "FULL_EXEC", "BASE_RULES",
           "Rlv3Config", "Rlv3PlanProvider", "config_from_spec",
           "install_seams", "make_dev_agent", "make_job", "last_agent",
           "COUNTERS", "counters", "reset_counters"]
