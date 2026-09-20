"""WHY are CARE skipped and animals escaping? Classify before building a fix.

The executor audit found 97.9 skipped CAREs and 1.0 escaped animal per game.
The tempting read is "the rule is missing". It is not: scheduler.py emits
FEED for every animal every day at P_SURVIVAL when consecutive_unfed >= 1,
and emits CARE for every fed-but-uncared animal. So each miss has a CAUSE,
and the causes need different fixes -- or no fix at all:

  CARE
    a) day >= 28        deliberate: the bonus cannot pay before the bell
    b) price gate       deliberate "crash pivot": skip CARE when the product
                        is below 35% of base, because boosting output we
                        cannot sell is waste
    c) not fed          CARE banks nothing on an unfed day, so this is not a
                        CARE problem at all -- it is a FEED problem
    d) emitted, not done  labour-bound: the task existed and no unit got to it

  ESCAPE
    e) no WHEAT anywhere  FEED needs a wheat in a bag; if the farm is out,
                          priority is irrelevant
    f) wheat existed      genuinely a routing/priority failure

Only (d) and (f) are worth gating. (a) and (b) are deliberate policy, (c)
and (e) are supply problems wearing a scheduling costume.

    PYTHONPATH=. uv run python tools/care_diagnose.py --n 12
"""
import argparse
import collections
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

TPD = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--agent", default="champ-h8")
    args = ap.parse_args()

    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]

    C = collections.Counter()
    for i in range(args.n):
        a0, a1 = load(args.agent), load(args.agent)
        e = make("kaggriculture", configuration={"seed": 920000 + i},
                 debug=False)
        e.reset(2)
        while not e.done:
            o = e.state[0].observation
            if o["step"] % TPD == TPD - 1:          # end of day, pre-refresh
                day = o["step"] // TPD
                pv = o.private
                farm = o["farms"][0]
                prices = (o.get("market") or {}).get("prices") or {}
                wheat = ((pv.get("shed") or {}).get("WHEAT", 0)
                         + sum((b or {}).get("WHEAT", 0)
                               for b in (pv.get("inventories") or [])))
                for row in farm["tiles"]:
                    for t in row:
                        if not isinstance(t, dict) or not t.get("animal"):
                            continue
                        fed = bool(t.get("fed_today"))
                        cared = bool(t.get("cared_today"))
                        prod = K.ANIMALS[t["animal"]]["product"]
                        base = K.MARKET_PARAMS[prod]["base"]
                        px = float(prices.get(prod, base))
                        if not cared:
                            C["care_missed"] += 1
                            if day >= 28:
                                C["care:a_day>=28"] += 1
                            elif not fed:
                                C["care:c_not_fed"] += 1
                            elif px < 0.35 * base:
                                C["care:b_price_gate"] += 1
                            else:
                                C["care:d_LABOUR_BOUND"] += 1
                        else:
                            C["care_done"] += 1
                        # about to escape tonight?
                        if not fed and t.get("consecutive_unfed", 0) >= 1:
                            C["escape"] += 1
                            if wheat <= 0:
                                C["escape:e_NO_WHEAT"] += 1
                            else:
                                C["escape:f_wheat_existed"] += 1
                        if not fed:
                            C["unfed"] += 1
                            if wheat <= 0:
                                C["unfed:no_wheat"] += 1
            e.step([a0(o), a1(e.state[1].observation)])

    ng = args.n
    print(f"\n{args.agent}, {ng} games, seat 0, end-of-day\n")
    print("=" * 64)
    print("  WHY IS CARE SKIPPED?")
    print("=" * 64)
    tot = C["care_missed"]
    print(f"  {'CARE done':<34}{C['care_done']:>8,}"
          f"{C['care_done']/ng:>9.1f}/game")
    print(f"  {'CARE missed':<34}{tot:>8,}{tot/ng:>9.1f}/game\n")
    for k, lbl, verdict in (
            ("care:a_day>=28", "day >= 28", "deliberate"),
            ("care:b_price_gate", "product below 35% of base", "deliberate"),
            ("care:c_not_fed", "animal was NOT FED", "a FEED problem"),
            ("care:d_LABOUR_BOUND", "emitted, nobody got to it",
             "<-- the only gateable one")):
        print(f"  {lbl:<34}{C[k]:>8,}{100*C[k]/max(tot,1):>7.1f}%"
              f"   {verdict}")

    print()
    print("=" * 64)
    print("  WHY DO ANIMALS ESCAPE?")
    print("=" * 64)
    print(f"  {'unfed at end of day':<34}{C['unfed']:>8,}"
          f"{C['unfed']/ng:>9.1f}/game")
    print(f"  {'  ... with NO wheat on the farm':<34}"
          f"{C['unfed:no_wheat']:>8,}"
          f"{100*C['unfed:no_wheat']/max(C['unfed'],1):>7.1f}%")
    print(f"\n  {'ESCAPING tonight':<34}{C['escape']:>8,}"
          f"{C['escape']/ng:>9.1f}/game")
    for k, lbl, verdict in (
            ("escape:e_NO_WHEAT", "no wheat anywhere", "a SUPPLY problem"),
            ("escape:f_wheat_existed", "wheat existed",
             "<-- routing/priority, gateable")):
        print(f"  {lbl:<34}{C[k]:>8,}"
              f"{100*C[k]/max(C['escape'],1):>7.1f}%   {verdict}")


if __name__ == "__main__":
    main()
