"""The early-game metrics we measured, as a feature vector for the value head.

WHY THIS EXISTS. `value_spatial` fit on the 181-channel board tensor clears
the base rate in every day bin, but by far the LEAST in the early game:

    days 0-4  +4.7%      days 12-16 +19.7%
    days 4-8  +4.9%      days 16-20 +25.2%

Early boards look alike. Almost nothing has yielded, banks are close, and
the tensor is mostly empty tiles -- so the encoder has little to separate a
good opening from a bad one. But we have *measured* quantities that DO
separate them, found over this season's work and recorded one at a time.
This file is those findings, written as numbers a head can read.

EVERY FEATURE CITES A MEASUREMENT. Nothing here is invented, and where a
prior says a family of features has already failed for a different consumer,
the docstring says so rather than quietly hoping.

  investment vs money  the relative MONEY lead predicts nothing before
                       ~day 16 (r = -0.021 at day 10) while INVESTMENT does
                       (r = +0.304). The single most important early fact we
                       have, and the board tensor carries it only implicitly.
  days-to-yield        an action is worth $1,317 on day 3 and $103 on day 27
                       (z = 3.67); TIMING dominates selection.
  crash exposure       above_func splits products into uncrashable annuities
                       (WHEAT/EGG, log 0.2) and traps (MELON 3.6, WOOL 3.2,
                       STRAWBERRY/MILK 1.6). Premium lines hit the $1 floor
                       on modest gluts.
  labour ratio         we are labour-SHORT: hands/work 0.47-0.58 at days
                       11-15, and the 4th quadrant gated -$6,266/-$6,017
                       because +25 tiles buys idle ground.
  neglect              a plant unwatered twice is a weed that night, and a
                       new plant starts consecutive_unwatered = 1 with no
                       grace period.
  hauling              our deficit is pickup-per-feed 1.07 against 0.52-0.63
                       for stronger agents.
  quadrant capital     "do we have running capital to maintain quadrant 1
                       before buying quadrant 2" -- land is ORDERED behind
                       hiring, and LAND_PRICES [1000, 2000, 4000] against a
                       $3,000 opening bank.

STANDING PRIOR, stated up front: capacity-style channels were KILLED once
already -- +0.0187 flow R2 at p=0.0001 bought 0% forecast MAE and -44%
out-of-distribution. That was the MARKET-FLOW consumer, not the value head,
so it is a warning rather than a verdict. If these features do not move
held-out sign accuracy, they go.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

# above_target > 1 means a modest glut drives the price to the $1 floor
CRASHY = {p for p, m in K.MARKET_PARAMS.items() if m["above_target"] > 1.0}
SAFE = {p for p, m in K.MARKET_PARAMS.items() if m["above_func"] == "log"}
STRUCT_COST = {"COOP": 0, "PASTURE": 0}   # build actions cost a turn, not cash


def _sat(x, scale):
    """Monotone squash to [0,1) that NEVER pins.

    `min(x, C) / C` looks bounded and harmless and is neither: it discards
    every distinction above C. Measured on the first propagation run, 13 of
    48 features sat at their cap for >30% of samples and E:invested was
    pinned 71% of the time, driving its correlation to exactly 0.000 and
    making it impossible to test the recorded "invested r=+0.304 at day 10".
    tanh keeps the ordering everywhere and still bounds the range.
    """
    return float(np.tanh(max(0.0, float(x)) / float(scale)))


NAMES = [
    "invested", "invested_lead", "money", "money_lead", "invest_over_money",
    "occ_all", "occ_q0", "occ_q1", "occ_q2", "occ_q3",
    "quads", "money_over_next_land", "newest_quad_occ",
    "mean_days_to_yield", "frac_yielding", "season_left", "pending_yield",
    "crash_frac", "safe_frac", "melon_tiles", "livestock_tiles",
    "hands", "work_pending", "hands_per_work",
    "unwatered", "weeds", "unfed", "neglect_rate",
    "bag_units", "shed_frac", "seeds_held",
]
N_EARLY = len(NAMES)


def _quad(x, y, n):
    h = n // 2
    return (0 if y < h else 2) + (0 if x < h else 1)


def features(obs, seat, private=None):
    """-> (N_EARLY,) float32. Scale-free; every entry bounded to roughly [0,3]."""
    v = np.zeros(N_EARLY, "float32")
    f = obs["farms"][seat]
    opp = obs["farms"][1 - seat]
    if private is None:
        private = (obs.get("private") or {})
    tiles = f["tiles"]
    n = len(tiles)
    day = obs.get("day", obs["step"] // 24)
    ix = {k: i for i, k in enumerate(NAMES)}

    def invested_of(farm):
        """Capital standing ON the board -- seeds in the ground and animals
        placed. This is the quantity that predicts early (r = +0.304); the
        bank balance is the one that does not (r = -0.021)."""
        tot = 0.0
        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    tot += K.CROPS[t["crop"]]["seed"]
                elif t.get("animal"):
                    tot += K.ANIMALS[t["animal"]]["cost"]
        return tot

    inv_me, inv_op = invested_of(f), invested_of(opp)
    money, omoney = float(f["money"]), float(opp["money"])
    v[ix["invested"]] = _sat(inv_me, 6000)
    v[ix["invested_lead"]] = np.tanh((inv_me - inv_op) / 2000.0)
    v[ix["money"]] = _sat(money, 30000)
    v[ix["money_lead"]] = np.tanh((money - omoney) / 5000.0)
    v[ix["invest_over_money"]] = min(inv_me / max(money + inv_me, 1.0), 1.0)

    # ---- occupancy, per quadrant. The user's own proposed metric: how much
    # of the land we have bought is actually doing something.
    filled = np.zeros(4)
    unlocked = np.zeros(4)
    crashy = safe = melon = live = 0
    days_to = []
    yielding = pending = 0
    unwatered = weeds = 0
    work = 0
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            q = _quad(x, y, n)
            unlocked[q] += 1
            if t is None:
                continue
            filled[q] += 1
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "WEED":
                weeds += 1
                work += 1
            elif kind == "PLANT":
                crop = t["crop"]
                cd = K.CROPS[crop]
                age = day - t.get("planted_day", day)
                left = max(0, cd["first_yield_day"] - age)
                days_to.append(left)
                if t.get("yield_units", 0) > 0:
                    yielding += 1
                    pending += t.get("yield_units", 0)
                if not t.get("watered_today"):
                    unwatered += 1
                    work += 1
                if crop in CRASHY:
                    crashy += 1
                if crop in SAFE:
                    safe += 1
                if crop == "MELON":
                    melon += 1
            elif t.get("animal"):
                live += 1
                a = K.ANIMALS[t["animal"]]
                age = day - t.get("placed_day", day)
                days_to.append(max(0, a["first_yield_day"] - age))
                if t.get("yield_units", 0) > 0:
                    yielding += 1
                    pending += t.get("yield_units", 0)
                if not t.get("fed_today"):
                    work += 1
                if a["product"] in CRASHY:
                    crashy += 1
                if a["product"] in SAFE:
                    safe += 1
    tot_un = max(unlocked.sum(), 1)
    v[ix["occ_all"]] = filled.sum() / tot_un
    for q in range(4):
        v[ix[f"occ_q{q}"]] = filled[q] / max(unlocked[q], 1)

    nq = len(f.get("unlocked_quadrants") or ["NW"])
    v[ix["quads"]] = nq / 4.0
    nxt = K.LAND_PRICES[nq - 1] if nq - 1 < len(K.LAND_PRICES) else None
    # "can we afford the NEXT quadrant" -- land is ordered behind hiring and
    # the 4th quadrant gated -$6k twice, so this is a risk flag, not a goal
    v[ix["money_over_next_land"]] = (min(money / nxt, 3.0) if nxt else 0.0)
    newest = int(np.argmin(np.where(unlocked > 0, unlocked, 1e9)))
    v[ix["newest_quad_occ"]] = filled[newest] / max(unlocked[newest], 1)

    # ---- timing. An action is worth 13x more on day 3 than day 27.
    v[ix["mean_days_to_yield"]] = (_sat(float(np.mean(days_to)), 12)
                                   if days_to else 0.0)
    ncommit = max(len(days_to), 1)
    v[ix["frac_yielding"]] = yielding / ncommit
    v[ix["season_left"]] = max(0.0, (30 - day)) / 30.0
    v[ix["pending_yield"]] = _sat(pending, 40)

    v[ix["crash_frac"]] = crashy / ncommit
    v[ix["safe_frac"]] = safe / ncommit
    v[ix["melon_tiles"]] = _sat(melon, 25)
    v[ix["livestock_tiles"]] = _sat(live, 25)

    # ---- labour. We are labour-SHORT, so hands-per-work is the binding one.
    nh = len(f.get("hands") or [])
    v[ix["hands"]] = _sat(nh, 12)
    v[ix["work_pending"]] = _sat(work, 40)
    v[ix["hands_per_work"]] = _sat((nh + 1) / max(work, 1), 3)

    v[ix["unwatered"]] = _sat(unwatered, 25)
    v[ix["weeds"]] = _sat(weeds, 25)
    unfed = sum(1 for row in tiles for t in row
                if isinstance(t, dict) and t.get("animal")
                and not t.get("fed_today"))
    v[ix["unfed"]] = _sat(unfed, 12)
    v[ix["neglect_rate"]] = (unwatered + unfed) / ncommit

    shed = (private.get("shed") or {})
    invs = (private.get("inventories") or [])
    bag = sum(sum(x for x in (b or {}).values() if x > 0) for b in invs)
    v[ix["bag_units"]] = _sat(bag, 40)
    v[ix["shed_frac"]] = _sat(sum(shed.values()), 100)
    v[ix["seeds_held"]] = _sat(sum((private.get("seeds") or {}).values()), 30)
    return v


def main():
    """Sanity: the vector must be finite, bounded, and actually VARY."""
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load("champ-h8"), load("champ-h8")
    rows, days = [], []
    for seed in (500001, 500002, 500003):
        e = make("kaggriculture", configuration={"seed": seed}, debug=False)
        e.reset(2)
        while not e.done:
            o = e.state[0].observation
            if o["step"] % 24 == 0:
                rows.append(features(o, 0, o.private))
                days.append(o["step"] // 24)
            e.step([a0(o), a1(e.state[1].observation)])
    X = np.stack(rows)
    d = np.asarray(days)
    print(f"{len(X)} samples over 3 games, {X.shape[1]} features\n")
    print(f"  finite: {np.isfinite(X).all()}   "
          f"range [{X.min():.2f}, {X.max():.2f}]")
    dead = [NAMES[i] for i in range(X.shape[1]) if X[:, i].std() < 1e-9]
    print(f"  constant features: {dead or 'none'}\n")
    early = d < 8
    print(f"  {'feature':<22}{'day<8 mean':>12}{'day>=8 mean':>13}{'sd':>8}")
    for i, nm in enumerate(NAMES):
        print(f"  {nm:<22}{X[early, i].mean():>12.3f}"
              f"{X[~early, i].mean():>13.3f}{X[:, i].std():>8.3f}")


if __name__ == "__main__":
    main()
