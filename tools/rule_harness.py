"""Engine instrumentation + faithful mid-game takeover for the rule tests.

Two things live here because both halves of `tools/rule_tests.py` need them.

**Why a takeover and not a state splice.** Mid-game `env` resume is unfaithful
(see `kaggriculture-rollout-engine`), so a recorded state is only reachable by
replaying it. Sessions carry the seed and both 720-step action scripts and
`rl/session_bench.py fidelity` shows the reproduction is exact, so:

    steps 0..T-1   both seats run their recorded scripts  -> the real state
    steps T..S     OUR executor drives our seat           -> the thing measured
    steps S+1..end our seat PASSes                        -> a clean artifact

Our agent is *called* on every step from 0 and its output discarded before T,
so its `Memory` and its hour-0 day plans evolve along the recorded trajectory.
A cold call at hour 14 would make it plan off-cadence, which no real game does.

**What is recorded.** Every unit op our seat emits inside the window, with the
tile and inventory either side of the engine applying it, plus a per-step
snapshot of any tile we are watching so deferred payoffs (a FERTILIZE that
must still be watered, a CARE that must still be fed) can be scored.
"""
import copy
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

TURNS_PER_DAY = 24
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

_ORIG_UNIT = K._apply_unit_action
_ORIG_COMMIT = K._commit_unit
_ORIG_EOD = K._end_of_day
# HIRE and BUY_LAND never reach _commit_unit -- _process_market calls these two
# directly (kaggriculture.py:577/580), so they need their own hooks or the
# market log silently has no hires and no land in it.
_ORIG_HIRE = K._do_hire
_ORIG_LAND = K._do_buy_land


class Recorder:
    """Engine-level log for one seat of one episode.

    Player identification: the interpreter applies player 0's farmer (idx 0)
    then its hands, then player 1's. Counting idx==0 calls modulo the number
    of seats recovers the player without touching the interpreter.
    """

    def __init__(self, seat, watch=(), record_all=False, snap_range=None):
        self.seat = seat
        self.watch = set(watch)       # (x, y) tiles to snapshot every step
        self.record_all = record_all  # log ops outside the window too (mining)
        self.snap_range = snap_range  # (a, b): whole-grid snapshots in [a, b]
        self.active = False           # set by the runner while in-window
        self.unit_ops = []            # [{step, unit, op, x, y, ...}]
        self.market = []              # [{step, op, item, ok, price}]
        self.track = {}               # (x, y) -> {step: tile copy}
        self.grid = {}                # step -> whole tiles grid (deferred pay)
        self.eod_grid = {}            # day  -> tiles grid after the refresh
        self.eod_shed = {}            # day  -> shed after the midnight drop
        self.priv = {}                # step -> {shed, seeds, money, hands}
        self.day_snap = {}            # day -> farm/private snapshot at hour 0
        self._p = -1
        self._farm = [None, None]
        self._step = 0

    # -- engine hooks ------------------------------------------------------
    def unit_hook(self, farm, private, idx, action, board_size, day,
                  turns_per_day, shed_capacity=100):
        if idx == 0:
            self._p = (self._p + 1) % 2
            if self._p == self.seat:
                self._snapshot(farm, private)
        p = self._p
        self._farm[p] = farm
        if p != self.seat or not (self.active or self.record_all):
            return _ORIG_UNIT(farm, private, idx, action, board_size, day,
                              turns_per_day, shed_capacity)
        if not isinstance(action, list) or not action:
            return _ORIG_UNIT(farm, private, idx, action, board_size, day,
                              turns_per_day, shed_capacity)

        x, y = K._farmer_position(farm, idx)
        inv = private["inventories"][idx] if idx < len(private["inventories"]) else {}
        rec = {
            "step": self._step, "unit": idx, "op": list(action), "x": x, "y": y,
            "tile_before": copy.deepcopy(farm["tiles"][y][x]),
            "inv_before": dict(inv),
            "shed_before": dict(private["shed"]),
            "seeds_before": dict(private.get("seeds", {})),
            "day": day,
        }
        _ORIG_UNIT(farm, private, idx, action, board_size, day, turns_per_day,
                   shed_capacity)
        inv = private["inventories"][idx] if idx < len(private["inventories"]) else {}
        px, py = K._farmer_position(farm, idx)
        if rec["tile_before"] == "LOCKED":
            # How far inside unbought ground is this unit? Depth 1 means it is
            # standing on the border -- a hand spawned on a locked shed-access
            # tile, or a path clipping a corner. Deeper means it is cutting
            # across a quadrant where no tile action is legal.
            rec["locked_depth"] = min(
                abs(x - ux) + abs(y - uy)
                for uy, row in enumerate(farm["tiles"])
                for ux, t in enumerate(row) if t != "LOCKED")
        rec.update({
            "tile_after": copy.deepcopy(farm["tiles"][y][x]),
            "inv_after": dict(inv),
            "shed_after": dict(private["shed"]),
            "seeds_after": dict(private.get("seeds", {})),
            "pos_after": [px, py],
        })
        self.unit_ops.append(rec)

    def commit_hook(self, op, item, price, farm, private, market,
                    shed_capacity=100):
        mine = farm is self._farm[self.seat]
        ok = _ORIG_COMMIT(op, item, price, farm, private, market, shed_capacity)
        if mine and (self.active or self.record_all):
            self.market.append({"step": self._step, "op": op, "item": item,
                                "price": price, "ok": bool(ok)})
        return ok

    def hire_hook(self, farm, private, board_size, mult=1):
        mine = farm is self._farm[self.seat]
        before = len(farm.get("hands", []))
        cost = K._hire_cost(farm["hires_today"], mult)
        _ORIG_HIRE(farm, private, board_size, mult)
        if mine and (self.active or self.record_all):
            self.market.append({"step": self._step, "op": "HIRE",
                                "item": "HAND", "price": cost,
                                "ok": len(farm.get("hands", [])) > before})

    def land_hook(self, farm, board_size):
        mine = farm is self._farm[self.seat]
        before = list(farm["unlocked_quadrants"])
        _ORIG_LAND(farm, board_size)
        if mine and (self.active or self.record_all):
            new = [q for q in farm["unlocked_quadrants"] if q not in before]
            self.market.append({"step": self._step, "op": "BUY_LAND",
                                "item": new[0] if new else "NONE",
                                "price": 0, "ok": bool(new)})

    def eod_hook(self, state, env, day):
        # end-of-day mutates watched tiles (weeds, production, care banking),
        # so snapshot after it as well as before the next turn's actions
        out = _ORIG_EOD(state, env, day)
        farm = state[0].observation.farms[self.seat]
        for (x, y) in self.watch:
            self.track.setdefault((x, y), {})[f"eod{day}"] = \
                copy.deepcopy(farm["tiles"][y][x])
        if self._in_snap_range(self._step):
            self.eod_grid[day] = copy.deepcopy(farm["tiles"])
            priv = state[self.seat].observation.private
            self.eod_shed[day] = dict(priv["shed"])
        return out

    def _in_snap_range(self, step):
        return (self.snap_range is not None
                and self.snap_range[0] <= step <= self.snap_range[1])

    # -- bookkeeping -------------------------------------------------------
    def _snapshot(self, farm, private):
        for (x, y) in self.watch:
            self.track.setdefault((x, y), {})[self._step] = \
                copy.deepcopy(farm["tiles"][y][x])
        if self._in_snap_range(self._step):
            self.grid[self._step] = copy.deepcopy(farm["tiles"])
            self.priv[self._step] = {
                "shed": dict(private["shed"]),
                "seeds": dict(private.get("seeds", {})),
                "money": farm["money"],
                "hands": len(farm.get("hands", [])),
                "quadrants": list(farm["unlocked_quadrants"]),
            }
        if self._step % TURNS_PER_DAY == 0:
            self.day_snap[self._step // TURNS_PER_DAY] = {
                "money": farm["money"],
                "shed": dict(private["shed"]),
                "seeds": dict(private.get("seeds", {})),
                "quadrants": list(farm["unlocked_quadrants"]),
                "hands": len(farm.get("hands", [])),
            }

    def set_step(self, step):
        self._step = step
        self._p = -1

    def install(self):
        K._apply_unit_action = self.unit_hook
        K._commit_unit = self.commit_hook
        K._end_of_day = self.eod_hook
        K._do_hire = self.hire_hook
        K._do_buy_land = self.land_hook

    @staticmethod
    def uninstall():
        K._apply_unit_action = _ORIG_UNIT
        K._commit_unit = _ORIG_COMMIT
        K._end_of_day = _ORIG_EOD
        K._do_hire = _ORIG_HIRE
        K._do_buy_land = _ORIG_LAND


def load_agent(name):
    """Exec a frozen bundle and hand back a fresh `agent` callable."""
    path = os.path.join(REPO, "arena", "agents", name, "main.py")
    if not os.path.isfile(path):
        raise SystemExit(f"no agent bundle at {path}")
    g = {"__name__": f"bundle_{name}"}
    exec(compile(open(path).read(), path, "exec"), g)
    return g["agent"]


class Takeover:
    """Script -> our executor -> PASS, with the recorder's step kept in sync."""

    def __init__(self, script, ours, t_from, t_to, recorder):
        self.script = script
        self.ours = ours
        self.t_from = t_from
        self.t_to = t_to
        self.rec = recorder
        self.emitted = {}

    # kaggle_environments inspects the signature and passes (obs, config)
    # when it sees two parameters; a bound __call__ reads as two.
    def __call__(self, obs, configuration=None):
        t = obs.get("step", 0)
        self.rec.set_step(t)
        self.rec.active = self.t_from <= t <= self.t_to
        # Always call: Memory and the hour-0 day plan must evolve along the
        # recorded trajectory, or the executor plans off-cadence at takeover.
        try:
            mine = self.ours(obs)
        except Exception:                                   # pragma: no cover
            mine = dict(PASS_ACTION)
        if t < self.t_from:
            return self.script[t] if t < len(self.script) and self.script[t] \
                else dict(PASS_ACTION)
        if t <= self.t_to:
            self.emitted[t] = mine
            return mine
        n_hands = len(obs["farms"][obs["player"]].get("hands", []))
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}


class ScriptSeat:
    """The opponent seat: its recorded script, unchanged, all game."""

    def __init__(self, script):
        self.script = script

    def __call__(self, obs, configuration=None):
        t = obs.get("step", 0)
        if t < len(self.script) and self.script[t]:
            return self.script[t]
        return dict(PASS_ACTION)


def load_session(path):
    with open(path) as f:
        return json.load(f)


def play(session, seat, agent_fn, t_from, t_to, watch=(), record_all=False,
         keep_env=False, snap_range=None):
    """Replay `session`; our executor drives `seat` over [t_from, t_to].

    `t_from > last step` replays the recording untouched (used by mining).
    """
    from kaggle_environments import make

    rec = Recorder(seat, watch=watch, record_all=record_all,
                   snap_range=snap_range)
    rec.install()
    try:
        ours = Takeover(session["actions"][seat], agent_fn, t_from, t_to, rec)
        opp = ScriptSeat(session["actions"][1 - seat])
        agents = [ours, opp] if seat == 0 else [opp, ours]
        env = make("kaggriculture",
                   configuration={"seed": session["seed"], "episodeSteps": 720},
                   debug=False)
        env.run(agents)
    finally:
        rec.uninstall()
    finals = [st.reward or 0.0 for st in env.steps[-1]]
    return {
        "rec": rec,
        "emitted": ours.emitted,
        "finals": finals,
        "us": finals[seat],
        "them": finals[1 - seat],
        "env": env if keep_env else None,
    }


def replay_only(session, seat, watch=(), keep_env=False):
    """The recording itself, instrumented -- the reference trajectory."""

    def never(obs, configuration=None):
        return dict(PASS_ACTION)

    return play(session, seat, never, 10 ** 9, 10 ** 9, watch=watch,
                record_all=True, keep_env=keep_env)
