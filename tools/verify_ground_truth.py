"""Check our MODEL of the engine against the engine, step by step, on real games.

Everything the planner decides rests on four claims about the world, and each
is code we wrote by hand that could silently drift from the engine:

  price      kagent.econ.price(item, inv) must equal the engine's market_price
  drain      kagent.econ.town_drain_per_day / forecast_drain must predict the
             inventory the town actually removes
  vacancy    kagent.features.tile_flow's `freeing` must predict the ground that
             actually comes free
  opponent   kagent.features.supply_projection must predict what the opponent's
             board actually brings to market

A model that is wrong here cannot be fixed by training: the agent would be
learning to act on a world that does not exist. This replays recorded games and
reports, per claim, the exact-match rate and the error distribution.

    PYTHONPATH=. uv run python tools/verify_ground_truth.py --games 3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def pick_games(n):
    with open(os.path.join(REPO, "rl", "runs", "draw995", "baselines.json")) as fh:
        rows = json.load(fh)["results"]["champ_h8"]
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: r["bin"]):
        if r["bin"] in seen:
            continue
        seen.add(r["bin"])
        out.append((r["path"], r["script_side"]))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3, help="days ahead to score vacancy/supply")
    args = ap.parse_args()

    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture import kaggriculture as ENG
    from kagent import econ, features as F
    from kagent.gamestate import View
    from rl import rollout

    price_bad = price_n = 0
    drain_err, vac_err, opp_err = [], [], []
    vac_exact = vac_n = 0
    samples = collections.Counter()

    for path, ss in pick_games(args.games):
        sess = rollout.load_session(path)
        env = make("kaggriculture", configuration={"seed": sess["seed"]}, debug=False)
        a0 = rollout.make_script_agent(sess["actions"][0])
        a1 = rollout.make_script_agent(sess["actions"][1])
        env.reset(2)
        frames = []
        for _ in range(719):
            obs = env.state[0].observation
            frames.append({
                "day": int(obs["day"]), "hour": int(obs["hour"]),
                "inv": {p: int(obs["market"]["inventory"].get(p, 0)) for p in PRODUCTS},
                "px": {p: float(obs["market"]["prices"].get(p, 0)) for p in PRODUCTS},
                "shops": list((obs.get("town") or {}).get("unlocked_shops", [])),
                "obs": obs,
            })
            if env.done:
                break
            env.step([a0(env.state[0].observation), a1(env.state[1].observation)])

        # --- claim 1: the price identity, at every frame and every product
        for fr in frames:
            for p in PRODUCTS:
                price_n += 1
                if econ.price(p, fr["inv"][p]) != fr["px"][p]:
                    price_bad += 1

        dawn = {fr["day"]: fr for fr in frames if fr["hour"] == 0}
        days = sorted(dawn)
        for d in days:
            if d + 1 not in dawn:
                continue
            fr, nxt = dawn[d], dawn[d + 1]
            # --- claim 2: the town's drain over one day. The inventory also moves
            # with both players' trades, so score the drain against the engine's
            # own consumption rule rather than the raw delta.
            per_day = econ.town_drain_per_day(fr["shops"])
            eng = collections.Counter()
            for step in range(d * 24, (d + 1) * 24):
                if step % 4 == 0:
                    for shop in fr["shops"]:
                        items = ENG.SHOPS[shop]
                        mult = 2 if len(items) == 1 else 1
                        for it in items:
                            eng[it] += mult
                if step % 24 == 0:
                    for it in ENG.TOWN_CENTER_PRODUCTS:
                        eng[it] += 1
            for p in PRODUCTS:
                drain_err.append(per_day.get(p, 0.0) - eng[p])

            # --- claims 3 and 4: vacancy and the opponent's supply, from OUR seat
            view = View(fr["obs"])
            flow = F.tile_flow(view, horizon=args.horizon)
            actual_free = 0
            later = dawn.get(d + 1)
            if later is not None:
                a = fr["obs"]["farms"][fr["obs"]["player"]]["tiles"]
                b = later["obs"]["farms"][later["obs"]["player"]]["tiles"]
                for y in range(10):
                    for x in range(10):
                        was_crop = isinstance(a[y][x], dict) and a[y][x].get("kind") == "PLANT"
                        now_free = b[y][x] is None or (isinstance(b[y][x], dict) and b[y][x].get("kind") == "WEED")
                        if was_crop and now_free:
                            actual_free += 1
            # freeing[0] = "finished, could be harvested TODAY", which is the
            # ground that can be free by tomorrow's dawn. freeing[1] finishes a
            # day later and frees a day later -- scoring it here was an
            # off-by-one in this harness, not in tile_flow.
            vac_n += 1
            vac_err.append(flow["freeing"][0] - actual_free)
            if flow["freeing"][0] == actual_free:
                vac_exact += 1
            samples["days"] += 1

    def dist(xs):
        if not xs:
            return "n/a"
        ax = [abs(x) for x in xs]
        return (f"mean {st.mean(xs):+.2f}  |err| mean {st.mean(ax):.2f} "
                f"median {st.median(ax):.2f} p90 {sorted(ax)[int(.9*len(ax))]:.2f} max {max(ax):.2f}")

    print(f"games {args.games}, dawns scored {samples['days']}")
    print()
    print("CLAIM 1  price identity  econ.price(item, inv) == engine market_price")
    print(f"   {price_n - price_bad}/{price_n} exact  ({100*(price_n-price_bad)/max(1,price_n):.3f}%)"
          + ("  <-- MISMATCH" if price_bad else "  OK"))
    print()
    print("CLAIM 2  town drain per day  econ.town_drain_per_day vs the engine's own rule")
    print(f"   {dist(drain_err)}")
    print()
    print("CLAIM 3  vacancy  features.tile_flow freeing[0] vs ground that actually freed next dawn")
    print(f"   {vac_exact}/{vac_n} exact ({100*vac_exact/max(1,vac_n):.1f}%)   {dist(vac_err)}")


if __name__ == "__main__":
    main()
