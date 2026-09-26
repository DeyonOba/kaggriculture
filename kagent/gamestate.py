"""Per-turn observation parsing (View) and episode-persistent state (Memory)."""

from kagent.constants import EPISODE_STEPS, TURNS_PER_DAY


class FarmView:
    __slots__ = ("money", "tiles", "units", "unlocked_quadrants", "hires_today", "board_size")

    def __init__(self, farm):
        self.money = farm["money"]
        self.tiles = farm["tiles"]  # tiles[y][x]
        self.board_size = len(self.tiles)
        fx, fy = farm["farmer"]
        self.units = [(0, fx, fy)] + [
            (i + 1, hx, hy) for i, (hx, hy) in enumerate(farm.get("hands", []))
        ]
        self.unlocked_quadrants = set(farm.get("unlocked_quadrants", ["NW"]))
        self.hires_today = farm.get("hires_today", 0)

    def tile(self, x, y):
        return self.tiles[y][x]

    def iter_tiles(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                yield x, y, self.tiles[y][x]

    def plants(self):
        for x, y, t in self.iter_tiles():
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                yield x, y, t

    def animals(self):
        """Occupied structures."""
        for x, y, t in self.iter_tiles():
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                yield x, y, t

    def empty_structures(self):
        for x, y, t in self.iter_tiles():
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and not t.get("animal"):
                yield x, y, t

    def weeds(self):
        for x, y, t in self.iter_tiles():
            if isinstance(t, dict) and t.get("kind") == "WEED":
                yield x, y, t

    def empty_tiles(self):
        for x, y, t in self.iter_tiles():
            if t is None:
                yield x, y


class View:
    def __init__(self, obs):
        self.player = obs["player"]
        self.step = obs.get("step", 0)
        self.day = obs.get("day", self.step // TURNS_PER_DAY)
        self.hour = obs.get("hour", self.step % TURNS_PER_DAY)
        self.hours_left_today = TURNS_PER_DAY - self.hour  # includes this turn
        # Last actable step is EPISODE_STEPS - 2 (718 by default).
        self.steps_left = (EPISODE_STEPS - 2) - self.step

        farms = obs["farms"]
        self.me = FarmView(farms[self.player])
        self.opp = FarmView(farms[1 - self.player])

        private = obs.get("private", {}) or {}
        self.shed = dict(private.get("shed", {}))
        self.seeds = dict(private.get("seeds", {}))
        self.unit_invs = [dict(inv) for inv in private.get("inventories", [{}])]
        while len(self.unit_invs) < len(self.me.units):
            self.unit_invs.append({})
        self.shed_total = sum(self.shed.values())

        market = obs.get("market", {}) or {}
        self.market_inv = dict(market.get("inventory", {}))
        self.market_prices = dict(market.get("prices", {}))
        self.unlocked_shops = list((obs.get("town", {}) or {}).get("unlocked_shops", []))

    def unit_inv(self, idx):
        return self.unit_invs[idx] if idx < len(self.unit_invs) else {}


class Memory:
    """Persists across steps within one episode (module globals survive).

    Reset whenever step goes backwards (new episode reusing the process).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_step = -1
        self.day_plan = None
        # Season-long champion-plan knob overrides (adaptive layer); None =
        # champ-h3 defaults. Cleared per episode (processes are reused).
        self.knobs = None
        self.knobs_decided = False
        self.day_plan_day = -1
        self.prev_market_inv = None
        self.my_sells_last_turn = {}
        self.opp_sell_ema = {}
        self.melons_sold = 0
        self.timing_max_ms = 0.0
        # Labor-utilization ledger: PASS-turns vs unit-turns, rolled daily.
        self.idle_accum = 0
        self.turns_accum = 0
        self.hands_peak = 0
        self.util_day = -1
        self.yesterday_idle = 0.0
        self.yesterday_hands = 0
        # Dawn and midday observations for the board trunk's lag planes
        # (kagent/boardstate.py). Only set in a board bundle; cleared here so
        # a reused process cannot carry one episode's frames into the next.
        self.bs_dawn = {}
        self.bs_mid = {}

    def observe(self, view):
        if view.step <= self.last_step:
            self.reset()
        self.last_step = view.step
