"""Task valuation: what is our marginal worker-action actually spent on?

The LP's dual says a worker-action is worth **$13.00** and labour binds on
29/30 days. Measured, ours is worth **$1.75** -- `j-tile` raised work actions
per game 2,791 -> 3,057 and earned +$465 on the 240-game ladder. A 7.4x
shortfall. So the constraint is not how many actions we get, it is what the
marginal one buys.

This audits every work action against the ENGINE'S OWN GUARDS
(`kaggriculture.py:334-530`) and sorts it into:

  NOOP        the engine returns without changing anything -- the action is
              literally discarded. WATER on an already-watered tile, HARVEST
              with yield_units 0 or on an immature plant, FEED/CARE on an
              already-fed/cared animal, COLLECT_FERTILIZER when none is
              available, PLANT on an occupied tile, DIG on empty ground.
  DOOMED      legal, but on a crop that cannot produce again before day 29
  CARE_UNFED  a CARE that will not bank, because the engine banks the bonus
              only `if tile["cared_today"] and tile["fed_today"]` at end of day
  OK          everything else

ALIGNMENT, measured rather than assumed. The observation at `steps[t]` is the
state AFTER the action at `steps[t]` has been applied: over 188 farmer MOVEs,
`pos[t] + move[t+1] == pos[t+1]` held 96.8% of the time against 34.0% for
`pos[t] + move[t] == pos[t+1]`. So the tile state an action SAW is the one at
`steps[t-1]`, and reading `steps[t]` shows the action's own effect -- which is
why a first version of this tool reported WATER as 100% no-op (`watered_today`
was already True, set by the very action being judged).

A work action never moves its unit, so the unit's position at `steps[t]` is
still the tile it acted on; only the tile's STATE has to be taken from
`steps[t-1]`.

Inventory-dependent rejections (FEED with no wheat, PLANT with no seed) are
NOT counted as NOOP here -- the per-unit inventories are in `obs.private`,
which is only populated for our own seat, so the count would not be comparable
across seats. They can only make the true NOOP rate HIGHER.

    PYTHONPATH=. uv run python tools/action_audit.py --agent champ-h8 --n 4
"""

import argparse
import collections
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

fast_env.install()

from kaggle_environments import make          # noqa: E402
from kagent.constants import CROPS            # noqa: E402
from tools.commit_probe import unit_ops, unit_pos, MOVES   # noqa: E402

LAST_DAY = 29


def doomed(tile, day):
    c = tile.get("crop")
    if not c or c not in CROPS:
        return False
    cd, age = CROPS[c], day - tile["planted_day"]
    if not cd["ongoing"]:
        return age < cd["max_yield_day"] and day + (cd["max_yield_day"] - age) > LAST_DAY
    since = day - tile["planted_day"] - cd["first_yield_day"]
    ivl = max(1, cd["interval"])
    nxt = day + (ivl - since % ivl if since >= 0 else -since)
    made = (since // ivl + 1) if since >= 0 else 0
    return nxt > LAST_DAY or made >= cd["max_yield"]


def classify(op, tile, day):
    """-> 'NOOP' | 'DOOMED' | 'OK', following the engine's guards exactly."""
    d = isinstance(tile, dict)
    plant = d and tile.get("kind") == "PLANT"
    animal = d and "animal" in tile

    if op == "WATER":
        if not plant:
            return "NOOP"
        if tile.get("watered_today"):
            return "NOOP"
    elif op == "HARVEST":
        if not d or tile.get("yield_units", 0) <= 0:
            return "NOOP"
        if plant and day - tile["planted_day"] < CROPS[tile["crop"]]["first_yield_day"]:
            return "NOOP"
    elif op == "FEED":
        if not animal or tile.get("fed_today"):
            return "NOOP"
    elif op == "CARE":
        if not animal or tile.get("cared_today"):
            return "NOOP"
    elif op == "COLLECT_FERTILIZER":
        if not animal or not tile.get("fertilizer_available"):
            return "NOOP"
    elif op == "FERTILIZE":
        if not plant:
            return "NOOP"
    elif op == "PLANT":
        if tile is not None:
            return "NOOP"
    elif op == "DIG":
        if tile is None or animal:
            return "NOOP"
    elif op in ("BUILD_COOP", "BUILD_PASTURE"):
        if tile is not None:
            return "NOOP"

    if plant and doomed(tile, day):
        return "DOOMED"
    return "OK"


def audit(agent, opp, n, base_seed):
    tally = collections.Counter()
    per_op = collections.defaultdict(collections.Counter)
    care_unfed = 0
    for k in range(n):
        env = make("kaggriculture", configuration={"seed": base_seed + k}, debug=False)
        env.run([os.path.join(REPO, "arena", "agents", agent, "main.py"),
                 os.path.join(REPO, "arena", "agents", opp, "main.py")])
        # end-of-day fed/cared state, to catch CARE that never banks
        eod = {}
        for t in range(1, len(env.steps)):
            o = env.steps[t][0].observation
            if o["hour"] == 23:
                eod[o["day"]] = o["farms"][0]["tiles"]
        for t in range(1, len(env.steps)):
            # The tile state the action SAW is at t-1; steps[t] already carries
            # this action's own effect. Position is safe to read at t because a
            # work action never moves its unit.
            prev = env.steps[t - 1][0].observation
            day, tiles = prev["day"], prev["farms"][0]["tiles"]
            pos, _ = unit_pos(env.steps[t], 0)
            for i, op in enumerate(unit_ops(env.steps[t], 0)):
                if i >= len(pos) or op == "PASS":
                    continue
                if op in MOVES:
                    tally["MOVE"] += 1
                    continue
                x, y = pos[i]
                tile = (tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[0])
                        else None)
                cls = classify(op, tile, day)
                tally[cls] += 1
                per_op[op][cls] += 1
                if (op == "CARE" and cls == "OK" and day in eod
                        and 0 <= y < len(eod[day]) and 0 <= x < len(eod[day][0])):
                    et = eod[day][y][x]
                    if isinstance(et, dict) and "animal" in et and not et.get("fed_today"):
                        care_unfed += 1
    return tally, per_op, care_unfed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=71000)
    args = ap.parse_args()

    tally, per_op, care_unfed = audit(args.agent, args.opp, args.n, args.base_seed)
    work = tally["OK"] + tally["NOOP"] + tally["DOOMED"]
    g = max(1, args.n)
    print(f"{args.agent} vs {args.opp}, {args.n} games\n")
    print(f"  work actions per game   {work/g:>8,.0f}")
    print(f"  moves       per game    {tally['MOVE']/g:>8,.0f}\n")
    print(f"  {'class':<12}{'per game':>10}{'share':>9}")
    for cls in ("OK", "NOOP", "DOOMED"):
        print(f"  {cls:<12}{tally[cls]/g:>10,.0f}{100*tally[cls]/max(1,work):>8.1f}%")
    print(f"  {'CARE unfed':<12}{care_unfed/g:>10,.0f}"
          f"{100*care_unfed/max(1,work):>8.1f}%   (bonus never banks)")
    waste = tally["NOOP"] + tally["DOOMED"] + care_unfed
    print(f"\n  wasted (NOOP + DOOMED + unbanked CARE): {waste/g:,.0f}/game = "
          f"{100*waste/max(1,work):.1f}% of work actions")

    print(f"\n  {'op':<20}{'total':>8}{'NOOP':>8}{'noop %':>9}{'DOOMED':>9}")
    for op, c in sorted(per_op.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        if tot / g < 1:
            continue
        print(f"  {op:<20}{tot/g:>8,.0f}{c['NOOP']/g:>8,.0f}"
              f"{100*c['NOOP']/tot:>8.1f}%{c['DOOMED']/g:>9,.0f}")
    print("\n  NOOP = the engine's own guard rejects it; the action is discarded.")
    print("  Inventory-dependent rejections are NOT counted, so the true rate "
          "is higher.")


if __name__ == "__main__":
    main()
