"""Self-correction state for the rlv3 executor seam: one module global,
reset when the step regresses.

Cross-turn state cannot live on the objects the seam is handed. `DayPlan`
has `__slots__` (not attachable) and `build_tasks` / `assign` never receive
`Memory`, so the ledger is a module global reset on step regression -- the
same idiom `kagent.gamestate.Memory` itself uses for a reused process.

WHAT IT IS FOR. Every rlv3 placement and chore-order lever is a claim about
a measurable executor statistic: pl_wheat_adj claims feeds from wheat in
hand rise, pl_crop_cluster claims the next chore lands next door,
pl_water_first claims fresh plants get their second water, co_chain_tile
claims chores per visit rise. The ledger measures exactly those four at
play time from the actions we emit, and `correct()` turns a claim that is
not paying into a within-episode kill switch. A lever that costs turns and
does not move its own metric is switched off for the rest of the game
instead of being paid for until day 29.

The counters are deliberately the SAME quantities tools/tile_metrics.py
computes off a finished trace, so a dawn number and a scorecard number mean
the same thing. They are not identical: the ledger sees only our own
emitted actions, never the engine's resolution, so a refused op still
counts as issued. Treat a ledger rate as the lever's own self-report and
the trace metric as the verdict.

A KILL IS A LIVENESS EVENT, so it is recorded rather than merely applied.
A lever switched off on day 2 of 30 means the arm played the SHIPPED
executor for the remaining 28 days, which is DEAD, not null -- exactly the
failure rlv3/README.md "Liveness, which is not optional" exists to stop, one
level below the plan provider. `_kill` therefore stamps `self.killed[flag]`
with the day and bumps `stats[f"{flag}.killed_day<d>"]` in the executor's
`_STATS`, so the kill reaches `exec_stats` (smoke, scorecard_hook) and
`result["extra"]["rlv3"]` (every gate row).

AND A KILL SWITCH MAY NOT BE SHORTER THAN ITS OWN LEVER'S CAUSAL HORIZON.
`pl_wheat_adj` seats wheat beside the animals so that feeds come from a
hand rather than a shed trip. Ground truth (README "Object Types"; engine
`CROPS["WHEAT"]["first_yield_day"] == 2`) says wheat carries no harvestable
yield until age 2, and the wheat must then be HARVESTed into a hand before a
FEED can come from it -- so no day earlier than `first wheat plant + 2` can
carry the lever's effect at all. Counting those days into the streak killed
the lever on day 2 of every game before it had ever acted.

AND A KILL IS A SUSPENSION, NOT A DEATH SENTENCE. `self.off` used to be
add-only, so the first two-day dip ended the lever for the season: measured
on the world `pick_games(1)` returns, `co_chain_tile` went off on day 3 and
`pl_wheat_adj` / `pl_crop_cluster` on day 4, and all three stayed off for the
remaining 26 days while their own metrics recovered (`feeds_from_hand_share`
0.20 on day 4, `next_chore_adjacent_share` back to 0.91 by day 19). Every
`rlv3/arms/*.json` single-lever arm was therefore measuring a four-day
intervention against a 30-day season. `_revive` is the inverse of `_kill` and
runs on exactly the branch that already resets the streak, and the
`cluster_radius` ladder climbs as well as falls.

TWO MORE THINGS A STREAK DAY MUST BE. A day only counts toward a kill when

  * the lever was LIVE on it. With the lever off the metric measures the
    SHIPPED executor, so it can neither confirm the kill nor extend it --
    `_handfeed_low` used to run to 24 on a lever that had been dead since
    day 4.
  * the rate had enough events behind it. `feeds_from_hand_share` off a
    single FEED is 0.0 or 1.0 and neither is evidence; see `MIN_FEEDS` /
    `MIN_VISITS`.
"""

from __future__ import annotations

MOVE_OPS = frozenset(("NORTH", "SOUTH", "EAST", "WEST"))
SHED_OPS = frozenset(("PICKUP", "DROP"))
TILE_OPS = frozenset((
    "WATER", "HARVEST", "PLANT", "FEED", "CARE", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "PLACE", "COLLECT_FERTILIZER",
))

# Regimes, in the lexicographic order of the objective. `salvage` is checked
# first because it is the most specific: it is the only state that serves
# objective 3 (minimise the loss margin) rather than objective 1 (win).
SALVAGE_LEAD = -20_000
SALVAGE_DAY = 24

# README "Object Types" / engine CROPS["WHEAT"]["first_yield_day"]: a wheat
# tile planted on day d has no harvestable yield before day d + 2, and the
# harvest has to reach a hand before a FEED can be served out of it. So
# `feeds_from_hand_share` cannot carry pl_wheat_adj's effect before then.
WHEAT_FIRST_YIELD_DAYS = 2

# A rate is evidence only if enough events went into it. `feeds_from_hand_
# share` off one FEED is 0.0 or 1.0 whatever the lever did, and
# `chores_per_visit` off three visits moves by a third on one extra chore.
# A rate row that does not state its denominator at all (a hand-built row in
# a unit test) is taken at face value: the guard refuses THIN evidence, not
# absent bookkeeping.
MIN_FEEDS = 4
MIN_VISITS = 8

# The levers `correct()` can switch off. `days_live` is counted for exactly
# these, so `rlv3.arm.ledger_report` states the DOSE the arm carried rather
# than inferring it from one kill day -- which stopped being the same number
# the moment `off` became revocable.
KILLABLE = ("pl_wheat_adj", "pl_crop_cluster", "co_chain_tile")


class ExecLedger:
    """Per-episode executor telemetry, rolled at dawn."""

    def __init__(self):
        self.reset_state()

    # ------------------------------------------------------------ lifecycle
    def reset_state(self):
        self.last_step = -1
        self.day = -1
        self.lead = 0.0
        self.dawn_lead = 0.0
        self.prev_lead_delta = 0.0
        # today's counters
        self.moves = 0
        self.productive = 0
        self.visits = 0
        self.chores = 0
        self.pickups = 0
        self.feeds = 0
        self.feeds_from_hand = 0
        self.adj_chores = 0
        self.adj_total = 0
        self.visit_chores = 0
        # per-unit scratch, cleared at dawn
        self._pos = {}
        self._visit_n = {}
        self._hand_wheat = {}
        self._last_chore = {}
        self._planted_today = set()
        self._planted_prev = set()
        # yesterday's realised rates and the streaks built off them
        self.rates = {}
        self.prev_rates = {}
        self.off = set()             # levers switched off for the episode
        self.killed = {}             # flag -> the day correct() switched it off
        self.first_wheat_day = None  # first day we emitted a PLANT WHEAT
        self.cluster_radius = None   # set on the first correct()
        self.conf_bump = 0.0
        self.water_tight = False
        self._handfeed_low = 0
        self._chain_flat = 0
        self._cluster_radius0 = None   # the configured base, for the climb
        self.days_live = {}          # flag -> days it was ON, counted at dawn
        self.days_rolled = 0
        self._corr_day = -2      # never equal to a real day, so the first
        self._corr = None        # correct() of an episode always computes

    def reset(self, step):
        """True when `step` starts a new episode (a reused process)."""
        if step <= self.last_step:
            self.reset_state()
            self.last_step = step
            return True
        self.last_step = step
        return False

    # ------------------------------------------------------------- observe
    def observe(self, view, actions):
        """Called at the tail of rlv3_assign, on the actions we emitted."""
        self.lead = float(view.me.money) - float(view.opp.money)
        pos_of = {idx: (x, y) for idx, x, y in view.me.units}
        for idx, op in actions.items():
            if not op:
                continue
            name = op[0]
            x, y = pos_of.get(idx, (None, None))
            if name in MOVE_OPS:
                self.moves += 1
                continue
            if name in TILE_OPS or name in SHED_OPS:
                self.productive += 1
            if name == "PICKUP":
                self.pickups += 1
                if len(op) > 1 and op[1] == "WHEAT":
                    # wheat from the shed is NOT wheat from the harvest
                    self._hand_wheat[idx] = False
            if name not in TILE_OPS:
                continue
            self.chores += 1
            # visit run: consecutive steps at one position with >= 1 chore
            if self._pos.get(idx) != (x, y):
                self._pos[idx] = (x, y)
                self._visit_n[idx] = 0
                self.visits += 1
            self._visit_n[idx] += 1
            self.visit_chores += 1
            # next-chore adjacency (the pl_crop_cluster claim)
            last = self._last_chore.get(idx)
            if last is not None and x is not None:
                self.adj_total += 1
                if abs(last[0] - x) + abs(last[1] - y) <= 1:
                    self.adj_chores += 1
            if x is not None:
                self._last_chore[idx] = (x, y)
            if name == "FEED":
                self.feeds += 1
                if self._hand_wheat.get(idx):
                    self.feeds_from_hand += 1
            elif name == "HARVEST" and x is not None:
                t = view.me.tile(x, y)
                if isinstance(t, dict) and t.get("crop") == "WHEAT":
                    self._hand_wheat[idx] = True
            elif name == "PLANT" and x is not None:
                self._planted_today.add((x, y))
                if (self.first_wheat_day is None
                        and len(op) > 1 and op[1] == "WHEAT"):
                    # pl_wheat_adj's causal clock starts here, not at day 0.
                    self.first_wheat_day = int(view.day)

    # ----------------------------------------------------------- dawn roll
    def dawn_roll(self, view):
        """Yesterday's realised rates. Idempotent within a day."""
        if view.day <= self.day:
            return self.rates
        first = self.day < 0
        # the fresh plants of yesterday that are already a dry night behind
        dry = 0
        for xy in self._planted_today:
            t = view.me.tile(*xy)
            if isinstance(t, dict) and t.get("consecutive_unwatered", 0) >= 1:
                dry += 1
        rates = {
            "feeds_from_hand_share": _r(self.feeds_from_hand, self.feeds),
            "next_chore_adjacent_share": _r(self.adj_chores, self.adj_total),
            "chores_per_visit": _r(self.visit_chores, self.visits),
            "moves_per_productive": _r(self.moves, self.productive),
            "pickups_per_feed": _r(self.pickups, self.feeds),
            "planted_dry_share": _r(dry, len(self._planted_today)),
            "n_planted": len(self._planted_today),
            # the denominators, so `correct()` can tell a thin day from a bad
            # one instead of killing a lever on a sample of one
            "n_feeds": self.feeds,
            "n_visits": self.visits,
            "lead_delta": self.lead - self.dawn_lead,
            "day": self.day,
        }
        self.prev_rates = self.rates
        self.rates = {} if first else rates
        self.prev_lead_delta = rates["lead_delta"]
        self._planted_prev = self._planted_today
        # reset the day
        self.day = view.day
        self.dawn_lead = self.lead
        self.moves = self.productive = self.visits = self.chores = 0
        self.pickups = self.feeds = self.feeds_from_hand = 0
        self.adj_chores = self.adj_total = self.visit_chores = 0
        self._pos = {}
        self._visit_n = {}
        self._hand_wheat = {}
        self._last_chore = {}
        self._planted_today = set()
        if not first:
            self.days_rolled += 1
        return self.rates

    # ---------------------------------------------------------- correction
    def _kill(self, flag, stats=None):
        """Switch `flag` off for the rest of the episode, ON THE RECORD.

        The day is kept on `self.killed` so `rlv3.arm.run_episode` can put
        "this lever was live for 2 of 30 days" on the result row, and the
        same fact is bumped into the executor's `_STATS` so it shows up in
        `exec_stats` -- the dict smoke.py and scorecard_hook.py already
        print. Idempotent: a rule that re-fires every day records once.
        """
        if flag in self.off:
            return
        self.off.add(flag)
        self.killed[flag] = self.day
        if stats is not None:
            key = f"{flag}.killed_day{self.day}"
            stats[key] = stats.get(key, 0) + 1   # Counter or plain dict

    def _revive(self, flag, stats=None):
        """Put `flag` back, ON THE RECORD. The exact inverse of `_kill`.

        A kill is a verdict on a MEASUREMENT and a measurement can come back.
        `self.off` was add-only, so the verdict outlived its evidence: on the
        traced world `feeds_from_hand_share` was 0.0 on days 2 and 3, which
        killed `pl_wheat_adj` at dawn on day 4, and 0.20 -- above the 0.15
        target -- on day 4 itself, which reset `_handfeed_low` to 0 and
        changed nothing else for 26 days.

        Recorded the same way `_kill` is, so `exec_stats` carries both events
        and `arm.ledger_report`'s `days_live` is not a fiction. Idempotent.
        """
        if flag not in self.off:
            return
        self.off.discard(flag)
        self.killed.pop(flag, None)
        if stats is not None:
            key = f"{flag}.revived_day{self.day}"
            stats[key] = stats.get(key, 0) + 1   # Counter or plain dict

    def wheat_effect_day(self):
        """First day on which `feeds_from_hand_share` can carry pl_wheat_adj.

        `None` until the lever has actually seated a wheat tile: a lever that
        has not acted cannot have failed, and killing it then is the
        unreachable-by-construction refutation this repo keeps re-learning.
        """
        if self.first_wheat_day is None:
            return None
        return self.first_wheat_day + WHEAT_FIRST_YIELD_DAYS

    def correct(self, params, stats=None):
        """Today's effective PARAMS, adjusted by yesterday's realised rates.

        Each rule is the kill switch for exactly one lever, and each reads
        the metric that lever exists to move. No rule ever turns a lever ON.

        IDEMPOTENT WITHIN A DAY, and it has to be: the streak counters and
        `conf_bump` accumulate, and `rlv3_assign` calls this again on every
        step where `mv_dest` is on. Without the guard a 24-step day would
        apply each correction 24 times and `mv_dest_conf` would saturate on
        the first losing day.
        """
        if self._corr_day == self.day and self._corr is not None:
            return dict(self._corr)
        out = dict(params)
        if self.cluster_radius is None:
            self.cluster_radius = float(params.get("cluster_radius", 2))
            self._cluster_radius0 = self.cluster_radius
        r = self.rates
        if r:
            # pl_wheat_adj -- two days below target and the adjacency claim
            # is not paying: stop seating wheat by the animals. Only days at
            # or past the lever's own causal horizon count: before the first
            # wheat it seated could be harvested and fed, the share is a
            # measurement of something else and two of them killed the lever
            # on day 2 of every game.
            share = r.get("feeds_from_hand_share")
            horizon = self.wheat_effect_day()
            measured = r.get("day")
            if (share is not None and horizon is not None
                    and measured is not None and measured >= horizon
                    and r.get("n_feeds", MIN_FEEDS) >= MIN_FEEDS):
                if share >= params.get("handfeed_target", 0.15):
                    # the claim is paying again: the streak resets AND so
                    # does the kill, or the first dip of the season ends the
                    # lever while its own metric recovers on the next day.
                    self._handfeed_low = 0
                    self._revive("pl_wheat_adj", stats)
                elif "pl_wheat_adj" not in self.off:
                    # only a LIVE lever can be judged: with the wheat no
                    # longer seated beside the animals this share measures
                    # the shipped executor, not the claim.
                    self._handfeed_low += 1
                    if self._handfeed_low >= 2:
                        self._kill("pl_wheat_adj", stats)

            # pl_crop_cluster -- the next chore is landing FARTHER away than
            # it did yesterday: shrink the radius. At 0 the lever is inert.
            cur = r.get("next_chore_adjacent_share")
            prev = (self.prev_rates or {}).get("next_chore_adjacent_share")
            if cur is not None and prev is not None:
                if cur < prev:
                    self.cluster_radius = max(0.0, self.cluster_radius - 1.0)
                elif cur > prev:
                    # the ladder climbs as well as falls, capped at the
                    # configured base. One-way it hit 0 on the second down
                    # day of the season and stayed there for 26 days while
                    # the share climbed back to 0.91.
                    self.cluster_radius = min(self._cluster_radius0,
                                              self.cluster_radius + 1.0)
            if self.cluster_radius <= 0:
                self._kill("pl_crop_cluster", stats)
            else:
                self._revive("pl_crop_cluster", stats)

            # pl_water_first -- fresh plants are going dry: tighten strictly
            # onto the pending-water cluster.
            dry = r.get("planted_dry_share")
            self.water_tight = bool(
                dry is not None and r.get("n_planted", 0) > 0
                and dry > params.get("dry_tol", 0.10)
            )

            # co_chain_tile -- chores per visit has not risen for two days:
            # the reorder is buying nothing, revert to the shipped order.
            cpv = r.get("chores_per_visit")
            pcpv = (self.prev_rates or {}).get("chores_per_visit")
            if (cpv is not None and pcpv is not None
                    and r.get("n_visits", MIN_VISITS) >= MIN_VISITS):
                if cpv > pcpv:
                    self._chain_flat = 0
                    self._revive("co_chain_tile", stats)
                elif "co_chain_tile" not in self.off:
                    self._chain_flat += 1
                    if self._chain_flat >= 2:
                        self._kill("co_chain_tile", stats)

            # mv_dest -- the record is monotone in "override less": on a day
            # the lead FELL, the model defers more. This runs WITH the
            # measured direction, not against it.
            if r.get("lead_delta", 0.0) < 0:
                self.conf_bump += params.get("conf_step", 0.05)

        # Once per day, after today's corrections: `correct()` is memoised on
        # `self.day`, so this is the dawn count and not a per-step one.
        for f in KILLABLE:
            if f not in self.off:
                self.days_live[f] = self.days_live.get(f, 0) + 1

        out["cluster_radius"] = self.cluster_radius
        out["mv_dest_conf"] = min(0.95, params.get("mv_dest_conf", 0.6) + self.conf_bump)
        out["water_tight"] = self.water_tight
        self._corr_day, self._corr = self.day, dict(out)
        return out

    # -------------------------------------------------------------- regime
    def regime(self, view, params):
        """Which arm of the lexicographic objective is live right now.

        `salvage`  losing badly and late  -> objective 3, cash per turn
        `contest`  behind at the contest day -> objective 1, reach the one
                   shared market before they do
        `ahead`    winning -> objective 2, finish what is underfoot and take
                   no new distant work
        """
        if self.lead < SALVAGE_LEAD and view.day >= SALVAGE_DAY:
            return "salvage"
        if self.lead < 0 and view.day >= params.get("contest_day", 18):
            return "contest"
        if self.lead > 0:
            return "ahead"
        return "even"


def _r(a, b):
    return (a / b) if b else None


LEDGER = ExecLedger()


def reset(step):
    return LEDGER.reset(step)
