"""Replay-tape agent over our own session corpus, with desync repair.

Architecture change, not a tuning round. Every parameter in the hand-written
planner has now been measured against real opponents and the knob space does not
contain the gap (see CHANGELOG). The top of the public leaderboard does not tune
either: it replays recorded 719-step action scripts from 2500+ seats and picks
between them with a small state-conditioned selector. We hold 20,793 such
sessions with ratings, so we can build that from our own corpus instead of
forking someone's notebook.

Replay is EXACT in a tape's own world (verified: episode 107273702 reproduces
$125,546 to the dollar), so the corpus is faithful. Moving a tape to a foreign
world costs money, and the cost is desync:

  * weeds spawn at random tiles per seed, so a PLANT or BUILD on a weed tile is
    silently dropped and every later action for that worker lands on the wrong
    tile;
  * a HIRE that the tape could afford may fail here, so the hands list changes
    length and every hand's action shifts by one seat.

`repair` fixes both: a blocked PLANT/BUILD becomes DIG and that worker's day is
replayed one turn late; the hands list is aligned to the live count.

    PYTHONPATH=. uv run python tools/tape_agent.py build --min-rating 3000 --n 400
    PYTHONPATH=. uv run python tools/tape_agent.py eval --n 6
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LIB = "rl/datasets/tapes/library.json"
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
BLOCKABLE = ("PLANT", "BUILD_COOP", "BUILD_PASTURE")
ECONOMIC_REPAIR = True
FORCE_SELL = False
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


# --------------------------------------------------------------------- build
def _shops_at(session, day):
    """Shops the town had revealed by `day` (unlock at end of days 2, 5, 8...)."""
    shops = session.get("shops") or []
    n = max(0, (day - 2) // 3 + 1) if day >= 2 else 0
    return tuple(shops[:n])


def build(args):
    paths = sorted(glob.glob("rl/datasets/sessions/session-*.json"))
    random.Random(args.seed).shuffle(paths)
    tapes = []
    for p in paths:
        try:
            s = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        scores = s.get("scores") or [None, None]
        for seat in (0, 1):
            r = scores[seat]
            if r is None or r < args.min_rating:
                continue
            if s["finals"][seat] < args.min_final:
                continue
            acts = s["actions"][seat]
            if len(acts) < 700:
                continue
            tapes.append({
                "episode": s["episode"], "seat": seat, "rating": r,
                "final": s["finals"][seat], "margin": s["finals"][seat] - s["finals"][1 - seat],
                "shops6": list(_shops_at(s, 6)), "shops_all": s.get("shops") or [],
                "actions": acts,
            })
        if len(tapes) >= args.n:
            break
    tapes.sort(key=lambda t: -t["margin"])
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    json.dump(tapes, open(LIB, "w"))
    mb = os.path.getsize(LIB) / 1e6
    print(f"{len(tapes)} tapes >= rating {args.min_rating} -> {LIB} ({mb:.1f} MB)")
    if tapes:
        print(f"  margin: best {tapes[0]['margin']:+,.0f}  median "
              f"{tapes[len(tapes)//2]['margin']:+,.0f}")
        by = defaultdict(int)
        for t in tapes:
            by[tuple(t["shops6"])] += 1
        print(f"  distinct day-6 shop pairs: {len(by)}; most common "
              f"{sorted(by.items(), key=lambda x: -x[1])[:3]}")


# ---------------------------------------------------------------- replay ----
class TapeRunner:
    """Replays one tape, repairing the two desync sources."""

    def __init__(self, actions):
        self.actions = actions
        self.delay = defaultdict(int)   # unit index -> turns this worker is behind
        self.last_day = -1

    def _tile(self, obs, seat, pos):
        try:
            return obs["farms"][seat]["tiles"][pos[1]][pos[0]]
        except Exception:  # noqa: BLE001
            return None

    def act(self, obs):
        step = obs.get("step", 0)
        seat = int(obs.get("player", 0))
        day = step // 24
        if day != self.last_day:
            # A worker's schedule is day-local; drop any unfinished shift so a
            # morning repair does not cascade through the whole season.
            self.delay.clear()
            self.last_day = day
        me = obs["farms"][seat]
        units = [me.get("farmer")] + list(me.get("hands") or [])
        out = {"farmer": ["PASS"], "hands": [], "market": []}

        def tape_at(t):
            return self.actions[t] if 0 <= t < len(self.actions) else None

        base = tape_at(step) or {}
        out["market"] = list(base.get("market") or [])
        for i, unit in enumerate(units):
            src = tape_at(step - self.delay[i]) or {}
            seq = [src.get("farmer") or ["PASS"]] + list(src.get("hands") or [])
            a = seq[i] if i < len(seq) else ["PASS"]
            if not isinstance(a, list) or not a:
                a = ["PASS"]
            # Weed repair: the tape wants to plant/build here, but this world
            # spawned a weed on the tile. Clear it and replay the intended
            # action next turn.
            if a[0] in BLOCKABLE and isinstance(unit, (list, tuple)) and len(unit) >= 2:
                tile = self._tile(obs, seat, (unit[0], unit[1]))
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    a = ["DIG"]
                    self.delay[i] += 1
            if i == 0:
                out["farmer"] = a
            else:
                out["hands"].append(a)

        if ECONOMIC_REPAIR:
            out["market"] = self._repair_market(obs, seat, out)
        return out

    def _repair_market(self, obs, seat, out):
        """Keep the tape's field plan executable when the market moved.

        The engine drops EVERY PLANT of a crop whose requests exceed the seeds
        in hand, so one unaffordable BUY_SEED silently voids a whole turn of
        planting; the tile then stays empty, its harvest never happens, income
        falls, and the next buy fails too. Measured on a 3125-rated tape moved
        to a foreign world: 134 seeds unplanted, 8 structures left empty, 2
        animals placed, final $35k against a recorded $116k. A tape is a plan
        for a cash trajectory it no longer has, so top up what its own field
        actions need and raise the cash by selling what is already in the shed.
        """
        market = list(out["market"])
        priv = obs.get("private") or {}
        seeds = dict(priv.get("seeds") or {})
        shed = dict(priv.get("shed") or {})
        money = float(obs["farms"][seat].get("money", 0))
        prices = (obs.get("market") or {}).get("prices") or {}

        want = defaultdict(int)
        for a in [out["farmer"]] + list(out["hands"]):
            if isinstance(a, list) and len(a) > 1 and a[0] == "PLANT":
                want[a[1]] += 1
        # seeds the tape's own orders are already buying this turn
        for o in market:
            if isinstance(o, list) and len(o) > 2 and o[0] == "BUY_SEED":
                seeds[o[1]] = seeds.get(o[1], 0) + int(o[2])
                money -= SEED_COST.get(o[1], 0) * int(o[2])

        need = [(c, n - seeds.get(c, 0)) for c, n in want.items() if n > seeds.get(c, 0)]
        if not need:
            return market
        cost = sum(SEED_COST.get(c, 0) * d for c, d in need)
        if cost > money and FORCE_SELL:
            # Raising cash by dumping the shed wrecks the tape's own selling
            # plan -- it sells into a market it has already crashed. Measured:
            # -$8k to -$70k a seed. Left here, off, as the recorded negative.
            for item in sorted(shed, key=lambda i: -prices.get(i, 0) * shed.get(i, 0)):
                if money >= cost or len(market) >= 10:
                    break
                n = shed.get(item, 0)
                p = prices.get(item, 0)
                if n <= 0 or p <= 1 or item in ("GOOSE", "COW", "SHEEP"):
                    continue
                market.append(["SELL", item, n])
                money += n * p
        for c, d in need:
            if len(market) >= 10:
                break
            if money >= SEED_COST.get(c, 0):
                market.append(["BUY_SEED", c, d])
                money -= SEED_COST.get(c, 0) * d
        return market[:10]


def make_tape_agent(actions):
    r = TapeRunner(actions)

    def agent(obs, config=None):
        return r.act(obs)
    return agent


def _load_lib():
    return json.load(open(LIB))


# ------------------------------------------------------------------ eval ----
def eval_cmd(args):
    from kaggle_environments import make
    from rl.rollout import make_script_agent
    lib = _load_lib()
    tape = lib[args.rank]
    print(f"tape ep {tape['episode']} seat {tape['seat']} rating {tape['rating']:.0f} "
          f"recorded ${tape['final']:,.0f} (margin {tape['margin']:+,.0f})\n")
    opps = [o.strip() for o in args.opps.split(",")]
    for opp in opps:
        raw_w = rep_w = 0
        raw_t = rep_t = 0
        for k in range(args.n):
            sd = args.base_seed + k
            for label in ("raw", "repaired"):
                env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd}, debug=False)
                me = (make_script_agent(tape["actions"]) if label == "raw"
                      else make_tape_agent(tape["actions"]))
                env.run([me, opp])
                f = [env.steps[-1][s]["reward"] for s in (0, 1)]
                if label == "raw":
                    raw_t += f[0]
                    raw_w += f[0] > f[1]
                else:
                    rep_t += f[0]
                    rep_w += f[0] > f[1]
        name = os.path.basename(os.path.dirname(opp))
        print(f"{name:<22} raw ${raw_t/args.n:>9,.0f} ({raw_w}/{args.n})   "
              f"repaired ${rep_t/args.n:>9,.0f} ({rep_w}/{args.n})   "
              f"gain ${(rep_t-raw_t)/args.n:>+8,.0f}")


EXPORT_TMPL = '''"""Replay-tape agent: {ep} seat {seat} (rating {rating:.0f}, recorded ${final:,.0f}).

Built by tools/tape_agent.py from our own session corpus. Follows a recorded
719-step script from a 3000+ seat, repairing the one desync that transfers:
a PLANT/BUILD onto a tile this world spawned a weed on becomes DIG, and that
worker's remaining day is replayed one turn late (schedules are day-local, so
the delay is dropped at dawn).

Screened empirically: a tape's recorded score does NOT predict how it transfers
(one $155k tape falls to $33k in foreign worlds, this $153k one holds $92k).
"""
import base64, json, zlib
from collections import defaultdict

_TAPE = json.loads(zlib.decompress(base64.b85decode("{blob}")).decode())
_BLOCKABLE = ("PLANT", "BUILD_COOP", "BUILD_PASTURE")
_S = {{"delay": defaultdict(int), "day": -1}}


def agent(observation, configuration=None):
    obs = observation
    step = obs.get("step", 0)
    seat = int(obs.get("player", 0))
    day = step // 24
    if day != _S["day"]:
        _S["delay"].clear()
        _S["day"] = day
    me = obs["farms"][seat]
    units = [me.get("farmer")] + list(me.get("hands") or [])
    tiles = me.get("tiles") or []
    base = _TAPE[step] if 0 <= step < len(_TAPE) else {{}}
    out = {{"farmer": ["PASS"], "hands": [], "market": list(base.get("market") or [])}}
    for i, unit in enumerate(units):
        t = step - _S["delay"][i]
        src = _TAPE[t] if 0 <= t < len(_TAPE) else {{}}
        seq = [src.get("farmer") or ["PASS"]] + list(src.get("hands") or [])
        a = seq[i] if i < len(seq) else ["PASS"]
        if not isinstance(a, list) or not a:
            a = ["PASS"]
        if a[0] in _BLOCKABLE and isinstance(unit, (list, tuple)) and len(unit) >= 2:
            try:
                tile = tiles[unit[1]][unit[0]]
            except Exception:
                tile = None
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                a = ["DIG"]
                _S["delay"][i] += 1
        if i == 0:
            out["farmer"] = a
        else:
            out["hands"].append(a)
    return out
'''


def export(args):
    import base64 as _b64
    import zlib as _z
    lib = _load_lib()
    t = lib[args.rank]
    blob = _b64.b85encode(_z.compress(json.dumps(t["actions"]).encode(), 9)).decode()
    code = EXPORT_TMPL.format(ep=t["episode"], seat=t["seat"], rating=t["rating"],
                              final=t["final"], blob=blob)
    d = os.path.join("arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"wrote {p} ({os.path.getsize(p)/1000:.0f} KB) from tape idx {args.rank} "
          f"ep {t['episode']} rating {t['rating']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval", "export"])
    ap.add_argument("--name", default="tape-v1")
    ap.add_argument("--min-rating", type=float, default=3000)
    ap.add_argument("--min-final", type=float, default=100000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=31000)
    ap.add_argument("--opps", default="arena/agents/pub-ahmed-v38/main.py,"
                                      "arena/agents/pub-boatlee-v16/main.py")
    args = ap.parse_args()
    {"build": build, "eval": eval_cmd, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
