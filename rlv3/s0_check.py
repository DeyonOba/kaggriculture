"""S0: does the planner's 3-day commitment agree with the TOP BOARDS?

    PYTHONPATH=. uv run python rlv3/s0_check.py

No game is played and no executor is touched. The planner is put on each
leader's OWN board at each dawn -- the recorded observation -- and asked what
it would commit over the next three days. That is compared with what the
leader actually did over those three days. rlv3/PHASE1.md: agreement with
champ-h11 would only inherit its ceiling; the leaders are the reference.

THREE READS, in order:
  1. IDENTITY. The `readout` preset with the readout removed must produce the
     champion's plan on every dawn, field for field. If it does not, every
     number below is confounded by the wiring and none of it counts.
  2. TILES. Our emitted delta (plant_quota, animal_buys, hires) against the
     leader's realised delta from dawn d to dawn d+3, per asset.
  3. DOLLARS. Our commitment as seed/livestock/hands shares against the
     leader's forward 3-day spend, per day, per draw state, per stage --
     against the noise floor the leaders set against EACH OTHER.

Memory is rolled from the recorded actions the way rl/session_bc.py rolls it,
so the readout sees the feature vector it was trained on.
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kagent import econ, features                                    # noqa: E402
from kagent.constants import ANIMALS, CROPS                          # noqa: E402
from kagent.gamestate import Memory, View                            # noqa: E402
from rlv3 import agent as A, arm                                     # noqa: E402
from rlv3.trajectory import leader_paths, winner_seat                # noqa: E402

TURNS = 24
H = 3
FIELDS = ("plant_quota", "animal_buys", "structure_builds", "hires_target", "land_buys")
STAGES = (("opening", 0, 8), ("contest", 9, 26), ("consolidation", 27, 29))


def _standing(obs, seat):
    farm = obs["farms"][seat]
    crops, herd = collections.Counter(), collections.Counter()
    for row in farm["tiles"]:
        for c in row:
            if isinstance(c, dict):
                if c.get("kind") == "PLANT" and c.get("crop"):
                    crops[c["crop"]] += 1
                elif c.get("animal"):
                    herd[c["animal"]] += 1
    return crops, herd


def _day_stats(steps, seat, day):
    """(pass_frac, hands_peak, melons_sold) from the recorded actions of `day`."""
    ops = passes = 0
    peak = 0
    melons = 0
    for t in range(day * TURNS, min((day + 1) * TURNS, len(steps))):
        act = steps[t][seat].get("action") or {}
        units = [act.get("farmer")] + list(act.get("hands") or [])
        units = [u for u in units if u]
        ops += len(units)
        passes += sum(1 for u in units if u[0] == "PASS")
        peak = max(peak, len(act.get("hands") or []))
        for o in (act.get("market") or []):
            if not o:
                continue
            if o[0] == "SELL" and o[1] == "MELON":
                melons += int(o[2]) if len(o) > 2 else 1
    return (passes / ops if ops else 0.0), peak, melons


def _forward_plants(steps, seat, day):
    """PLANT actions per crop over days d..d+2: the leader's actual FLOW,
    replants included. A stock delta cannot see a replant; this can."""
    c = collections.Counter()
    for t in range(day * TURNS, min((day + H) * TURNS, len(steps))):
        act = steps[t][seat].get("action") or {}
        for u in [act.get("farmer")] + list(act.get("hands") or []):
            if u and u[0] == "PLANT" and len(u) > 1:
                c[u[1]] += 1
    return dict(c)


def _forward_bought(steps, seat, day):
    """Animals the leader actually BOUGHT on day d (BUY_ANIMAL orders)."""
    c = collections.Counter()
    for t in range(day * TURNS, min((day + 1) * TURNS, len(steps))):
        for o in ((steps[t][seat].get("action") or {}).get("market") or []):
            if o and o[0] == "BUY_ANIMAL":
                c[o[1]] += int(o[2]) if len(o) > 2 else 1
    return dict(c)


def _forward_fert(steps, seat, day):
    """Leader FERTILIZE actions per day, mean over d..d+2."""
    n = 0; days = 0
    for d in range(day, min(day + H, len(steps) // TURNS)):
        days += 1
        for t in range(d * TURNS, min((d + 1) * TURNS, len(steps))):
            act = steps[t][seat].get("action") or {}
            n += sum(1 for u in [act.get("farmer")] + list(act.get("hands") or [])
                     if u and u[0] == "FERTILIZE")
    return n / max(1, days)


def _forward_spend(steps, seat, day):
    """Leader's actual dollars into seed / livestock / hands over days d..d+2.
    BUY_PRODUCT is excluded on purpose: the plan does not emit feed."""
    spend = {"seed": 0.0, "livestock": 0.0, "hands": 0.0}
    for d in range(day, min(day + H, len(steps) // TURNS)):
        hired = 0
        for t in range(d * TURNS, min((d + 1) * TURNS, len(steps))):
            act = steps[t][seat].get("action") or {}
            for o in (act.get("market") or []):
                if not o:
                    continue
                q = int(o[2]) if len(o) > 2 else 1
                if o[0] == "BUY_SEED":
                    spend["seed"] += CROPS.get(o[1], {}).get("seed", 0) * q
                elif o[0] == "BUY_ANIMAL":
                    spend["livestock"] += ANIMALS.get(o[1], {}).get("cost", 0) * q
                elif o[0] == "HIRE":
                    for _ in range(q):
                        spend["hands"] += econ.hire_marginal_cost(hired); hired += 1
    return spend


def _our_spend(delta, plan):
    seed = sum(CROPS[c]["seed"] * n for c, n in delta["crops"].items() if c in CROPS)
    live = sum(ANIMALS[k]["cost"] * n for k, n in delta["herd"].items() if k in ANIMALS)
    # Hands are hired DAILY (hires_today resets each dawn), so a 3-day
    # commitment at this count carries H days of hiring. Charging one day
    # against a 3-day seed ask understated hands by ~3x and read as a seed
    # surplus in the contest stage (ours 0.45-0.75 seed vs theirs 0.33-0.51)
    # on the SAME hire count (9.39 vs 9.31).
    hands = econ.hire_total_cost(plan.hires_target) * H
    return {"seed": float(seed), "livestock": float(live), "hands": float(hands)}


def _shares(sp):
    tot = sum(sp.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in sp.items()}


def _tvd(a, b):
    return sum(abs(a[k] - b[k]) for k in a) / 2.0


def _fields(plan):
    return {f: (dict(getattr(plan, f)) if isinstance(getattr(plan, f), dict)
                else list(getattr(plan, f)) if isinstance(getattr(plan, f), list)
                else getattr(plan, f)) for f in FIELDS}


def run(out=None, preset="readout", leaders=None, weights=None):
    if leaders:
        os.environ["KAGG_LEADERS"] = leaders     # leader_paths() reads it
    base_cfg = A.config_from_spec(arm.parse("base"))
    spec = arm.parse(preset)
    if weights:
        spec = {**spec, "readout": weights}      # score a specific readout file
    ro_cfg = A.config_from_spec(spec)
    off_cfg = A.config_from_spec({**arm.parse(preset), "readout": None})
    print(f"preset: {preset}  (readout={ro_cfg.readout}, posture={ro_cfg.posture})")

    # A probe the live arm depends on: rl.rollout switches features to
    # residual MODE for knob arms; the readout was trained under the corpus
    # builder's mode. If encode's width changes with MODE, the readout raises
    # in play. Say so here rather than 144 games later.
    mode0 = features.MODE
    print(f"features.MODE at start: {mode0}; HEADS_VERSION {features.HEADS_VERSION}")

    rows = []           # one per (game, day)
    identity_bad = 0
    n_dawns = 0
    for tag, path in leader_paths():
        with open(path) as fh:
            steps = json.load(fh)["steps"]
        seat = winner_seat(path)
        n_days = len(steps) // TURNS
        prov = {k: A.Rlv3PlanProvider(c) for k, c in
                (("base", base_cfg), ("off", off_cfg), ("ro", ro_cfg))}
        mems = {k: Memory() for k in prov}
        stats = (0.0, 0, 0)
        for d in range(n_days):
            obs = dict(steps[d * TURNS][seat]["observation"])
            obs.setdefault("step", d * TURNS)
            if "farms" not in obs:
                continue
            plans = {}
            for k in prov:
                view = View(obs)
                m = mems[k]
                m.observe(view)
                m.yesterday_idle, m.yesterday_hands, m.melons_sold = stats
                plans[k] = prov[k].make_plan(view, m)
            n_dawns += 1
            if _fields(plans["base"]) != _fields(plans["off"]):
                identity_bad += 1
            delta = prov["ro"].last_delta
            # the leader's realised delta to dawn d+3 (clamped to the end)
            d3 = min(d + H, n_days - 1)
            c0, h0 = _standing(obs, seat)
            c3, h3 = _standing(steps[d3 * TURNS][seat]["observation"], seat)
            end = steps[min((d + 1) * TURNS - 1, len(steps) - 1)][seat]["observation"]
            lead_hires = end["farms"][seat].get("hires_today", 0)
            shops = tuple(sorted((obs.get("town") or {}).get("unlocked_shops") or []))
            ctx = features.context(View(obs))
            rows.append({
                "game": tag, "day": d, "n_shops": len(shops),
                "consumers": dict(ctx["consumers"]),
                "our_fert": delta.get("fert"), "lead_fert": _forward_fert(steps, seat, d),
                "alpha": delta.get("alpha"), "upper": delta.get("upper"),
                # the readout's PREDICTED stock and the pipeline commit() subtracted,
                # so an under-ask can be attributed: prediction vs pipeline vs cash
                "pred_herd": dict((delta.get("pred") or {}).get("herd", {})),
                "pred_crops": dict((delta.get("pred") or {}).get("crops", {})),
                "standing_herd": {k: int(v) for k, v in
                                  __import__("collections").Counter(t["animal"] for _x, _y, t in View(obs).me.animals()).items()},
                "shed": {k: int(v) for k, v in (View(obs).shed or {}).items() if k in ANIMALS},
                "money": float(obs["farms"][seat].get("money", 0)),
                "lead_bought": _forward_bought(steps, seat, d),
                "tilt": delta.get("tilt") or {},
                "margin": float(obs["farms"][seat].get("money", 0)) - float(obs["farms"][1 - seat].get("money", 0)),
                "our_crops": delta["crops"], "our_herd": delta["herd"],
                "our_hires": plans["ro"].hires_target, "lead_hires": lead_hires,
                "lead_dcrops": {c: max(0, c3.get(c, 0) - c0.get(c, 0)) for c in CROPS},
                "lead_dherd": {k: max(0, h3.get(k, 0) - h0.get(k, 0)) for k in ANIMALS},
                "lead_flow": _forward_plants(steps, seat, d),
                "our_spend": _our_spend(delta, plans["ro"]),
                "lead_spend": _forward_spend(steps, seat, d),
                "land_signal": delta["land_signal"],
                "lead_land": int(len(steps[d3 * TURNS][seat]["observation"]["farms"][seat]
                                     .get("unlocked_quadrants", []))
                                 > len(obs["farms"][seat].get("unlocked_quadrants", []))),
            })
            stats = _day_stats(steps, seat, d)

    # ---------------------------------------------------------------- 1
    print(f"\n[1] IDENTITY  {n_dawns} dawns over {len(leader_paths())} leader games: "
          f"readout preset minus readout == champion on {n_dawns - identity_bad}/{n_dawns}"
          + ("   OK -- the wiring is a pure addition" if identity_bad == 0
             else f"   ** {identity_bad} MISMATCHES: numbers below are confounded **"))

    # ---------------------------------------------------------------- 2
    print(f"\n[2] TILES  our 3-day commitment vs the leader over the same 3 days")
    print(f"{'asset':<12}{'ours':>7} | {'stock d':>8}{'mae':>6}{'sign':>6} | {'PLANT flow':>11}{'mae':>6}{'sign':>6}"
          f"   (mean per dawn; flow = their actual plants, replants included)")
    for c in list(CROPS) + list(ANIMALS):
        o = [r["our_crops"].get(c, 0) if c in CROPS else r["our_herd"].get(c, 0) for r in rows]
        t = [r["lead_dcrops"].get(c, 0) if c in CROPS else r["lead_dherd"].get(c, 0) for r in rows]
        mae = statistics.mean(abs(a - b) for a, b in zip(o, t))
        sign = statistics.mean(1.0 if (a > 0) == (b > 0) else 0.0 for a, b in zip(o, t))
        line = f"{c:<12}{statistics.mean(o):>7.2f} | {statistics.mean(t):>8.2f}{mae:>6.2f}{sign:>6.0%} | "
        if c in CROPS:
            f = [r["lead_flow"].get(c, 0) for r in rows]
            fm = statistics.mean(abs(a - b) for a, b in zip(o, f))
            fs = statistics.mean(1.0 if (a > 0) == (b > 0) else 0.0 for a, b in zip(o, f))
            line += f"{statistics.mean(f):>11.2f}{fm:>6.2f}{fs:>6.0%}"
        print(line)
    oh = [r["our_hires"] for r in rows]; lh = [r["lead_hires"] for r in rows]
    print(f"{'hires':<12}{statistics.mean(oh):>7.2f}{statistics.mean(lh):>8.2f}"
          f"{statistics.mean(abs(a-b) for a,b in zip(oh,lh)):>7.2f}{'':>11}")
    ls = [r["land_signal"] for r in rows]; ll = [r["lead_land"] for r in rows]
    print(f"{'land (adv.)':<12}{statistics.mean(ls):>7.2f}{statistics.mean(ll):>8.2f}"
          f"{'':>7}{statistics.mean(1.0 if a==b else 0.0 for a,b in zip(ls,ll)):>11.0%}"
          f"   readout signal vs leader buying within 3 days; NOT applied")

    # ---------------------------------------------------------------- 3
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    print(f"\n[3] DOLLARS  3-day commitment shares (seed/livestock/hands) vs the leaders' "
          f"forward 3-day spend")
    print(f"{'day':>4}{'shops':>6}{'n':>4}  {'ours: seed live hands':^24}  "
          f"{'theirs (median)':^24}  {'TVD':>6}{'floor':>7}{'in-band':>9}")
    day_tvd, day_floor, day_in = {}, {}, {}
    for d in sorted(by_day):
        rs = by_day[d]
        ours = [_shares(r["our_spend"]) for r in rs]
        theirs = [_shares(r["lead_spend"]) for r in rs]
        med_t = {k: statistics.median(x[k] for x in theirs) for k in ("seed", "livestock", "hands")}
        med_o = {k: statistics.median(x[k] for x in ours) for k in med_t}
        tvd = statistics.mean(_tvd(o, t) for o, t in zip(ours, theirs))   # paired per game
        # noise floor: leader vs leader, same day
        pairs = [_tvd(theirs[i], theirs[j]) for i in range(len(theirs)) for j in range(i + 1, len(theirs))]
        floor = statistics.mean(pairs) if pairs else float("nan")
        # in-band: each share of ours inside the leaders' IQR that day
        q = {}
        for k in med_t:
            v = sorted(x[k] for x in theirs)
            q[k] = (v[len(v) // 4], v[(3 * len(v)) // 4])
        inb = statistics.mean(1.0 if all(q[k][0] - 1e-9 <= o[k] <= q[k][1] + 1e-9 for k in q) else 0.0
                              for o in ours)
        day_tvd[d], day_floor[d], day_in[d] = tvd, floor, inb
        ns = statistics.mode(r["n_shops"] for r in rs)
        print(f"{d:>4}{ns:>6}{len(rs):>4}  "
              f"{med_o['seed']:>7.2f} {med_o['livestock']:>5.2f} {med_o['hands']:>5.2f}      "
              f"{med_t['seed']:>7.2f} {med_t['livestock']:>5.2f} {med_t['hands']:>5.2f}      "
              f"{tvd:>6.3f}{floor:>7.3f}{inb:>8.0%}")

    print(f"\n[3b] BY STAGE")
    print(f"{'stage':<14}{'days':>7}{'TVD to leader':>15}{'leader floor':>14}{'in-band':>9}   verdict")
    for name, lo, hi in STAGES:
        ds = [d for d in day_tvd if lo <= d <= hi]
        if not ds:
            continue
        t = statistics.mean(day_tvd[d] for d in ds)
        f = statistics.mean(day_floor[d] for d in ds)
        i = statistics.mean(day_in[d] for d in ds)
        verdict = "AT the floor" if t <= f * 1.15 else ("near" if t <= f * 1.5 else "outside")
        print(f"{name:<14}{f'{lo}-{hi}':>7}{t:>15.3f}{f:>14.3f}{i:>8.0%}   {verdict}")
    # ---------------------------------------------------------------- 4
    if any(r.get("our_fert") is not None for r in rows):
        print(f"\n[4] FERTILISER  FERTILIZE actions per day, ours (plan.fert_budget) vs leaders")
        print(f"{'stage':<14}{'ours':>7}{'leaders':>9}{'mae':>7}")
        for name, lo, hi in STAGES:
            rs = [r for r in rows if lo <= r["day"] <= hi and r.get("our_fert") is not None]
            if not rs:
                continue
            o = statistics.mean(r["our_fert"] for r in rs)
            t = statistics.mean(r["lead_fert"] for r in rs)
            m = statistics.mean(abs(r["our_fert"] - r["lead_fert"]) for r in rs)
            print(f"{name:<14}{o:>7.2f}{t:>9.2f}{m:>7.2f}")
        z = [r for r in rows if r["day"] <= 9 and r.get("our_fert") is not None]
        print(f"  days 0-9: ours {statistics.mean(r['our_fert'] for r in z):.2f}  "
              f"leaders {statistics.mean(r['lead_fert'] for r in z):.2f}   <- the phase gate")

    # ---------------------------------------------------------------- 5
    print(f"\n[5] DEMAND RESPONSE  does the ask RISE with the number of shops that want the product?")
    print(f"{'product->asset':<18}{'shops':>6}{'n':>6}{'ours':>8}{'leaders':>9}   (mean 3-day ask per dawn)")
    for prod, asset in (("WOOL", "SHEEP"), ("MILK", "COW"), ("EGG", "GOOSE"), ("WHEAT", "WHEAT")):
        for lo, hi, lab in ((0, 0, "0"), (1, 1, "1"), (2, 9, "2+")):
            rs = [r for r in rows if lo <= r["consumers"].get(prod, 0) <= hi]
            if not rs:
                continue
            if asset in CROPS:
                o = statistics.mean(r["our_crops"].get(asset, 0) for r in rs)
                t = statistics.mean(r["lead_flow"].get(asset, 0) for r in rs)
            else:
                o = statistics.mean(r["our_herd"].get(asset, 0) for r in rs)
                t = statistics.mean(r["lead_dherd"].get(asset, 0) for r in rs)
            print(f"{prod+'->'+asset:<18}{lab:>6}{len(rs):>6}{o:>8.2f}{t:>9.2f}")

    # ---------------------------------------------------------------- 6
    tilted = [r for r in rows if r.get("tilt")]
    if tilted:
        print(f"\n[6] POSTURE  risk.alpha_for_lead on the leader's own margin, and what the tilt did")
        behind = [r for r in rows if r.get("upper")]
        ahead = [r for r in rows if r.get("upper") is False and (r.get("alpha") or 1.0) < 0.999]
        level = [r for r in rows if (r.get("alpha") or 1.0) >= 0.999]
        print(f"  dawns: behind {len(behind)}  ahead {len(ahead)}  level {len(level)}   "
              f"mean alpha behind {statistics.mean(r['alpha'] for r in behind) if behind else float('nan'):.2f}  "
              f"ahead {statistics.mean(r['alpha'] for r in ahead) if ahead else float('nan'):.2f}")
        print(f"  mean tilt factor (1 = untouched):")
        print(f"  {'asset':<12}{'behind':>8}{'ahead':>8}")
        for a in ("WHEAT", "STRAWBERRY", "MELON", "CARROT", "GOOSE", "COW", "SHEEP"):
            fb = [r["tilt"][a] for r in behind if a in r["tilt"]]
            fa = [r["tilt"][a] for r in ahead if a in r["tilt"]]
            print(f"  {a:<12}{statistics.mean(fb) if fb else 1.0:>8.2f}{statistics.mean(fa) if fa else 1.0:>8.2f}")
        print(f"  PRE-DRAW LIVESTOCK (0 shops for the product), our ask vs leaders, by posture:")
        for prod, asset in (("WOOL", "SHEEP"), ("MILK", "COW"), ("EGG", "GOOSE")):
            rs0 = [r for r in rows if r["consumers"].get(prod, 0) == 0]
            for lab, sel in (("behind", [r for r in rs0 if r.get("upper")]),
                             ("ahead/level", [r for r in rs0 if not r.get("upper")])):
                if sel:
                    print(f"    {prod+'->'+asset:<14}{lab:<12} n={len(sel):>3}  ours "
                          f"{statistics.mean(r['our_herd'].get(asset, 0) for r in sel):.2f}  "
                          f"leaders {statistics.mean(r['lead_dherd'].get(asset, 0) for r in sel):.2f}")

    # ---------------------------------------------------------------- 7
    print(f"\n[7] OPENING LIVESTOCK, days 0-4  (mean per dawn over 40 games; pipeline = standing + shed)")
    print(f"{'day':>4}{'money':>8}   {'readout pred@d+3':^20}   {'pipeline':^16}   {'OUR buys':^16}   {'LEADER bought':^16}")
    for d in range(0, 5):
        rs = [r for r in rows if r["day"] == d]
        if not rs:
            continue
        def mean_of(key, k):
            return statistics.mean(float(r.get(key, {}).get(k, 0)) for r in rs)
        pipe = {k: mean_of("standing_herd", k) + mean_of("shed", k) for k in ("GOOSE", "COW", "SHEEP")}
        fmt = lambda f: " ".join(f"{k[0]}{f(k):.1f}" for k in ("GOOSE", "COW", "SHEEP"))
        print(f"{d:>4}{statistics.mean(r['money'] for r in rs):>8.0f}   "
              f"{fmt(lambda k: mean_of('pred_herd', k)):^20}   {fmt(lambda k: pipe[k]):^16}   "
              f"{fmt(lambda k: mean_of('our_herd', k)):^16}   {fmt(lambda k: mean_of('lead_bought', k)):^16}")

    print("\nTVD = paired total-variation distance between our shares and that leader's, "
          "same game, same day. floor = how far the leaders sit from EACH OTHER on that "
          "day: at or below it is full agreement, there is no credit beneath it.")

    out = out or os.path.join(REPO, "rl", "runs", "headfix", "s0_check.json")
    with open(out, "w") as fh:
        json.dump({"n_dawns": n_dawns, "identity_bad": identity_bad,
                   "day_tvd": day_tvd, "day_floor": day_floor, "day_in": day_in,
                   "rows": rows}, fh)
    print(f"-> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="readout", help="readout | readout_risk")
    ap.add_argument("--leaders", default=None, help="replay tree root, e.g. replays/now")
    ap.add_argument("--weights", default=None, help="readout bc.pt to score instead of the preset's")
    a = ap.parse_args()
    run(a.out, a.preset, a.leaders, a.weights)
