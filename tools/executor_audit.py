"""Does our AGENT encode the rules the engine actually implements?

`tools/readme_audit.py` established that the README matches the engine:
286/286. That answers "what are the rules". It does NOT answer the question
that matters more, and that fourteen executor generations never asked:

    does the thing we SHIP know those rules?

The record is that it sometimes does not, and both examples surfaced only
when someone finally read the engine:

  - the enumerated action space restricted PICKUP to {WHEAT, FERTILIZER},
    which made PLACE:GOOSE permanently unreachable and severed
    BUILD_COOP -> GOOSE -> EGG entirely.
  - `to_action` emitted [op, item, 1]. n=1 is 27.5% of what real agents do
    (mean 3.32), so the vocabulary could express a quarter of the corpus.

Neither was a tuning problem. Both were the map not matching the territory,
and no amount of search over a wrong action space finds them.

This audits BEHAVIOUR against the verified rules. Every counter below is a
rule from readme_audit.py, turned into "how often do we get this wrong".

    PYTHONPATH=. uv run python tools/executor_audit.py --n 12
"""
import argparse
import collections
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

TPD = 24


def audit(agent="champ-h8", opp="champ-h8", n_games=12, base_seed=900000):
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]

    C = collections.Counter()
    orig_apply = K._apply_unit_action
    # SEAT IDENTIFICATION. _apply_unit_action runs inside e.step() for BOTH
    # players, so a flag set around the agent calls is 0 for everyone and
    # silently doubles every action count. The farm dict passed in IS the
    # engine's internal one, so identify the seat by matching its money
    # against the observation recorded just before the step.
    cur = {"day": 0, "seat0_id": None, "money0": None}

    def hooked(farm, private, idx, action, board_size, day, turns_per_day,
               shed_capacity=100):
        op = action[0] if action else "PASS"
        pos = K._farmer_position(farm, idx)
        tile = None
        if pos:
            fx, fy = pos
            if 0 <= fx < board_size and 0 <= fy < board_size:
                tile = farm["tiles"][fy][fx]
        if cur["seat0_id"] is None and cur["money0"] is not None \
                and abs(float(farm.get("money", -1)) - cur["money0"]) < 1e-9:
            cur["seat0_id"] = id(farm)
        me = (cur["seat0_id"] is not None and id(farm) == cur["seat0_id"])
        if me:
            C[f"act:{op}"] += 1
            # ---- WATER outside the bonus window on a ONE-TIME crop.
            # The bonus is applied in the WATER handler, ages
            # ceil(max_yield_day/2) .. max_yield_day. Outside it, watering
            # only resets the death clock -- necessary, but it buys no yield.
            if op == "WATER" and isinstance(tile, dict) \
                    and tile.get("kind") == "PLANT":
                cd = K.CROPS[tile["crop"]]
                if not cd["ongoing"]:
                    age = day - tile.get("planted_day", day)
                    ws = (cd["max_yield_day"] + 1) // 2
                    if age < ws:
                        C["water:before_window"] += 1
                    elif age > cd["max_yield_day"]:
                        C["water:after_window"] += 1
                    elif tile.get("yield_units", 0) >= cd["max_yield"]:
                        # already at cap: melon hits 6 at age 10 but its
                        # window runs to 12, so ages 11-12 are pure waste
                        C["water:at_cap_WASTED"] += 1
                    else:
                        C["water:in_window_ok"] += 1
            # ---- HARVEST of a one-time crop BEFORE peak yield
            if op == "HARVEST" and isinstance(tile, dict) \
                    and tile.get("kind") == "PLANT":
                cd = K.CROPS[tile["crop"]]
                if not cd["ongoing"]:
                    age = day - tile.get("planted_day", day)
                    y = tile.get("yield_units", 0)
                    if age < cd["max_yield_day"] and y < cd["max_yield"]:
                        C["harvest:before_peak"] += 1
                        C["harvest:units_left_on_plant"] += (
                            min(cd["max_yield"],
                                y + (cd["max_yield_day"] - age)) - y)
                    else:
                        C["harvest:at_peak_ok"] += 1
            # ---- PICKUP quantity: are we hauling one at a time?
            if op == "PICKUP" and len(action) >= 2:
                got = int(action[2]) if len(action) >= 3 else 1
                avail = (private.get("shed") or {}).get(action[1], 0)
                C["pickup:total"] += 1
                C["pickup:units"] += min(got, avail)
                if got == 1 and avail > 1:
                    C["pickup:ONE_when_more_available"] += 1
        return orig_apply(farm, private, idx, action, board_size, day,
                          turns_per_day, shed_capacity)

    K._apply_unit_action = hooked
    try:
        for i in range(n_games):
            a0, a1 = load(agent), load(opp)
            e = make("kaggriculture", configuration={"seed": base_seed + i},
                     debug=False)
            e.reset(2)
            prev_day = -1
            while not e.done:
                o = e.state[0].observation
                day = o["step"] // TPD
                cur["day"] = day
                # ---- end-of-day rule compliance, read at hour 23
                if o["step"] % TPD == TPD - 1:
                    pv = o.private
                    farm = o["farms"][0]
                    for row in farm["tiles"]:
                        for t in row:
                            if not isinstance(t, dict):
                                continue
                            if t.get("kind") == "WEED":
                                C["eod:weeds_standing"] += 1
                            elif t.get("kind") == "PLANT":
                                if not t.get("watered_today"):
                                    C["eod:plant_unwatered"] += 1
                                    if t.get("consecutive_unwatered", 0) >= 1:
                                        C["eod:plant_DIES_TONIGHT"] += 1
                            elif t.get("animal"):
                                if not t.get("fed_today"):
                                    C["eod:animal_unfed"] += 1
                                    if t.get("consecutive_unfed", 0) >= 1:
                                        C["eod:animal_ESCAPES_TONIGHT"] += 1
                                if t.get("fertilizer_available"):
                                    # use-it-or-lose-it: 1/animal/day, does
                                    # not accumulate, gone at the refresh
                                    C["eod:fertilizer_LOST"] += 1
                                if not t.get("cared_today") \
                                        and t.get("fed_today"):
                                    C["eod:care_skipped_while_fed"] += 1
                    shed_total = sum((pv.get("shed") or {}).values())
                    if shed_total >= 100:
                        C["eod:shed_FULL_blocks_buys"] += 1
                    C["eod:days"] += 1
                # money is distinct between seats almost always; on the rare
                # tie the seat stays unidentified for that step and nothing
                # is counted, which is safer than counting it twice
                m0 = float(o["farms"][0]["money"])
                m1 = float(o["farms"][1]["money"])
                cur["money0"] = m0 if abs(m0 - m1) > 1e-9 else None
                # the env re-structifies state each step, so the farm object
                # identity is NOT stable across steps -- re-identify each time
                cur["seat0_id"] = None
                act0 = a0(o)
                act1 = a1(e.state[1].observation)
                e.step([act0, act1])
                prev_day = day
    finally:
        K._apply_unit_action = orig_apply
    return C, n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--agent", default="champ-h8")
    args = ap.parse_args()
    C, ng = audit(agent=args.agent, n_games=args.n)

    def pct(a, b):
        return f"{100 * C[a] / max(C[b], 1):>5.1f}%"

    print(f"\n{args.agent}, {ng} games. Each line is a VERIFIED rule from "
          f"readme_audit.py.\n")

    print("=" * 70)
    print("  WATERING -- the one-time bonus applies ONLY in the window")
    print("=" * 70)
    tot = (C["water:before_window"] + C["water:after_window"]
           + C["water:at_cap_WASTED"] + C["water:in_window_ok"])
    print(f"  {'in the window, below cap (buys yield)':<46}"
          f"{C['water:in_window_ok']:>8,}{pct('water:in_window_ok', '_w'):>8}"
          .replace(pct('water:in_window_ok', '_w'),
                   f"{100*C['water:in_window_ok']/max(tot,1):>5.1f}%"))
    for k, lbl in (("water:before_window", "before the window opens"),
                   ("water:after_window", "after the window closes"),
                   ("water:at_cap_WASTED", "already AT CAP (pure waste)")):
        print(f"  {lbl:<46}{C[k]:>8,}"
              f"{100*C[k]/max(tot,1):>7.1f}%")
    print(f"\n  note: watering outside the window still resets the death")
    print(f"  clock, so it is not free waste -- but AT CAP it buys nothing")
    print(f"  at all, which is the melon age 11-12 case.")

    print()
    print("=" * 70)
    print("  HARVEST TIMING -- one-time crops")
    print("=" * 70)
    h = C["harvest:before_peak"] + C["harvest:at_peak_ok"]
    print(f"  {'harvested at/after peak':<46}{C['harvest:at_peak_ok']:>8,}"
          f"{100*C['harvest:at_peak_ok']/max(h,1):>7.1f}%")
    print(f"  {'harvested EARLY':<46}{C['harvest:before_peak']:>8,}"
          f"{100*C['harvest:before_peak']/max(h,1):>7.1f}%")
    print(f"  {'units left on the plant by early harvest':<46}"
          f"{C['harvest:units_left_on_plant']:>8,}"
          f"{C['harvest:units_left_on_plant']/max(ng,1):>7.0f}/game")

    print()
    print("=" * 70)
    print("  HAULING -- PICKUP takes an n; do we use it?")
    print("=" * 70)
    print(f"  {'PICKUP actions':<46}{C['pickup:total']:>8,}")
    print(f"  {'units moved':<46}{C['pickup:units']:>8,}")
    print(f"  {'units per PICKUP':<46}"
          f"{C['pickup:units']/max(C['pickup:total'],1):>8.2f}")
    print(f"  {'n=1 when MORE was in the shed':<46}"
          f"{C['pickup:ONE_when_more_available']:>8,}"
          f"{100*C['pickup:ONE_when_more_available']/max(C['pickup:total'],1):>7.1f}%")

    print()
    print("=" * 70)
    print("  WHAT THE ACTION BUDGET IS SPENT ON")
    print("=" * 70)
    acts = {k[4:]: v for k, v in C.items() if k.startswith("act:")}
    tot_a = sum(acts.values())
    for k, v in sorted(acts.items(), key=lambda kv: -kv[1]):
        if v * 200 < tot_a:
            continue
        print(f"  {k:<46}{v:>8,}{100*v/max(tot_a,1):>7.1f}%"
              f"{v/max(ng,1):>9.0f}/game")
    print(f"  {'TOTAL':<46}{tot_a:>8,}{'':>7}{tot_a/max(ng,1):>9.0f}/game")

    print()
    print("=" * 70)
    print("  END-OF-DAY COMPLIANCE  "
          f"({C['eod:days']} agent-days)")
    print("=" * 70)
    for k, lbl in (
            ("eod:plant_unwatered", "plants left unwatered"),
            ("eod:plant_DIES_TONIGHT", "  ... of which become WEEDS tonight"),
            ("eod:animal_unfed", "animals left unfed"),
            ("eod:animal_ESCAPES_TONIGHT", "  ... of which ESCAPE tonight"),
            ("eod:fertilizer_LOST", "fertilizer left uncollected (LOST)"),
            ("eod:care_skipped_while_fed", "CARE skipped on a fed animal"),
            ("eod:weeds_standing", "weeds left standing"),
            ("eod:shed_FULL_blocks_buys", "days with a FULL shed")):
        print(f"  {lbl:<48}{C[k]:>8,}{C[k]/max(ng,1):>9.1f}/game")


if __name__ == "__main__":
    main()
