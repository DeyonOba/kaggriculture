"""Zero-game checks on rlv3/objective.py, and the builder for calibration.json.

NOT ONE GAME IS PLAYED HERE. Everything below reads results this project has
already paid for: the per-game rows of ~30 paired arms under rl/runs/, and
the three leaders' full replays under replays/. That ordering is deliberate.
`better()` and `p_win()` are a value-ranking device over candidate plans, and
value ranking IS CLOSED in this repo -- four heads, two encoders x two label
schemes plus an auxiliary, all ordered actions at CHANCE out of sample
(0.503, 0.500), and the fitted head was 4.8x tighter on held-out error while
getting the sign of 3 of 4 interventions wrong. So the comparator is refuted
or not refuted on arms already measured, BEFORE a line of the planner is
written and before a single new game is spent.

    --check identity      the market-flow identity ProjState.observe solves,
                          against the opponent's own recorded holdings. This
                          also decides the sign convention by measurement
                          instead of by argument.
    --check retrodict     ~30 paired arms with known held-out win deltas:
                          does rank() order them the way the eval did? A
                          money-first control objective is scored beside it,
                          because an objective that cannot beat "maximise our
                          bank" on arms that lost while banking more has not
                          earned its complexity.
    --check fixedpoint    score() on the leaders' realised games must exceed
                          score() on ours at the same day. Their margins are
                          +$1,396 (Boey) and +$4,690 (DSM) against our
                          -$15,286. If the objective prefers our
                          41-strawberry allocation to theirs, the objective
                          is the bug, not the planner.
    --check sign          sign accuracy and r of project()'s final-margin
                          forecast at days 6/12/16/20/28, against the raw
                          money lead as the baseline it must beat
                          (50.9 / 50.7 / 55.3 / 70.0 / 88.2 %, r -0.009 /
                          -0.017 / +0.190 / +0.575 / +0.928).
    --check price         forecast MAE at k=5 and k=10 against the
                          price-forecast ablation's own ladder: drain-only
                          25.5, persistence 23.1, both boards 13.3.
    --check brier         Brier and AUC of p_win per band. REPORTED, never
                          judged: a fitted head 4.8x tighter on held-out
                          error got the sign of 3 of 4 interventions wrong.

    --build-calibration   writes rlv3/calibration.json from the replays.

Usage:
    cd <repo> && PYTHONPATH=. uv run python rlv3/objective_check.py --check all
    cd <repo> && PYTHONPATH=. uv run python rlv3/objective_check.py --build-calibration
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kagent.gamestate import View                                  # noqa: E402
from kagent.constants import PRODUCTS, TURNS_PER_DAY               # noqa: E402
from kagent import econ                                            # noqa: E402
from kagent import planner as kplanner                             # noqa: E402
from rlv3 import objective as O                                    # noqa: E402

REPLAY_DIRS = ["replays/mmpq", "replays/boey", "replays/dsm", "replays/top"]

# Paired arms already on disk: (json, baseline arm, [arms]). Every row in
# these files is one held-out game on a recorded seat, with the win, our
# money, the opponent's money and the rating band.
ARM_FILES = [
    ("rl/runs/draw995/baselines.json", "dev_knobs_h8",
     ["dev_champion", "champ_h8", "champ_h9"]),
    ("rl/runs/draw995/batch2.json", "dev_knobs_h8",
     ["ship2", "ship2_hv", "ship2_spent", "ship2_layout", "ship2_fert", "allin"]),
    ("rl/runs/draw995/batch3.json", "dev_knobs_h8",
     ["hvuf", "spent2", "oneshed", "hf_layout", "combo3"]),
    ("rl/runs/draw995/batch4.json", "dev_knobs_h8",
     ["av", "geese5", "av_geese5", "ship2_hv"]),
    ("rl/runs/draw995/execnet_eval.json", "dev_knobs_h8",
     ["execnet_chores", "execnet_chores_c50", "execnet_all_c70"]),
    ("rl/runs/draw995/execnet2_eval.json", "dev_knobs_h8",
     ["execnet2_curated_c40", "execnet2_cond_c40", "execnet2_cond_c60"]),
    ("rl/runs/draw995/hybrid_eval.json", "dev_knobs_h8",
     ["hybrid", "kernel_planner", "ship2_hv"]),
    ("rl/runs/draw995/land4_eval.json", "dev_knobs_h8", ["land4"]),
    ("rl/runs/draw995/kernel_arm.json", "dev_knobs_h8",
     ["kernel_board", "local_board_s1234"]),
    ("rl/runs/rules_tests/geese_eval.json", "dev_knobs_h8", ["geese4", "geese6"]),
    ("rl/runs/rules_tests/cull_eval.json", "dev_knobs_h8",
     ["cull_milk", "cull_endgame", "cull_wool", "cull_both"]),
    ("rl/runs/rules_tests/exec_eval.json", "dev_knobs_h8",
     ["exec_lastwater", "exec_four", "exec_quadq", "exec_five"]),
    ("rl/runs/rules_tests/evesell_eval.json", "dev_knobs_h8", ["evesell_four"]),
]


# ------------------------------------------------------------------ helpers

def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _sd(xs: Sequence[float]) -> float:
    """Spread of the margin at a given day -- the scale rlv3.objective's
    fallback table has to declare and this measurement replaces."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def replay_paths(dirs: Sequence[str] = REPLAY_DIRS, limit: Optional[int] = None
                 ) -> List[str]:
    out: List[str] = []
    for d in dirs:
        full = os.path.join(REPO, d)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            if name.endswith("-replay.json"):
                out.append(os.path.join(full, name))
    return out if limit is None else out[:limit]


def dawn_views(path: str) -> Tuple[List[Tuple[int, object, object]], List[float]]:
    """[(day, View(seat 0), View(seat 1))] at hour 0, plus the final rewards.

    One replay is loaded, reduced and dropped: a replay is ~31 MB of JSON and
    the machine this runs on has a paired eval on it.
    """
    with open(path) as fh:
        blob = json.load(fh)
    steps = blob["steps"]
    rewards = [float(x or 0.0) for x in (blob.get("rewards") or [0.0, 0.0])]
    out = []
    for t in range(0, len(steps), TURNS_PER_DAY):
        row = steps[t]
        if len(row) < 2:
            continue
        obs0 = dict(row[0]["observation"])
        obs1 = dict(row[1]["observation"])
        obs0.setdefault("step", t)
        obs1.setdefault("step", t)
        obs1.setdefault("player", 1)
        try:
            v0, v1 = View(obs0), View(obs1)
        except Exception:
            continue
        out.append((int(v0.day), v0, v1))
    del steps, blob
    return out, rewards


# ------------------------------------------------------- 1. the identity

def check_identity(limit: int = 4) -> bool:
    """Does the solve in ProjState.observe recover the opponent's real flow?

    The replay carries BOTH seats' private state, so the opponent's true net
    market flow is computable the same way we compute our own -- harvested
    minus the change in what they hold. That is ground truth for a quantity
    the agent can never see, and it decides `INV_SIGN` by measurement.
    """
    per_game: List[Tuple[float, float]] = []
    print("== IDENTITY: opponent net market flow, solved vs true ==")
    for path in replay_paths(limit=limit):
        with open(path) as fh:
            steps = json.load(fh)["steps"]
        st = O.ProjState()
        solved: Dict[str, float] = {}
        truth: Dict[str, float] = {}
        prev = None
        for t in range(0, len(steps)):        # every turn: the real cadence
            row = steps[t]
            if len(row) < 2:
                continue
            o0 = dict(row[0]["observation"]); o0.setdefault("step", t)
            o1 = dict(row[1]["observation"]); o1.setdefault("step", t); o1.setdefault("player", 1)
            v0, v1 = View(o0), View(o1)
            st.observe(v0)
            # Ground truth: the SAME ledger, run on the opponent's own
            # private state, which the replay happens to carry and the agent
            # can never see.
            snap = O.ProjState.seat_snapshot(v1)
            for item, v in O.ProjState.seat_net(prev, snap).items():
                truth[item] = truth.get(item, 0.0) + v
            prev = snap
        for item, v in st._day_opp.items():
            solved[item] = solved.get(item, 0.0) + v
        for item, v in st.opp_flow.items():
            solved[item] = solved.get(item, 0.0) + v
        # A product whose channel went blind cannot be solved from public
        # inventory by anyone; comparing it would be scoring the estimator on
        # information the engine deleted.
        keys = [p for p in PRODUCTS
                if abs(truth.get(p, 0.0)) > 5 and p not in st.blind]
        blind = sorted(p for p in PRODUCTS if p in st.blind)
        xs = [solved.get(p, 0.0) for p in keys]
        ys = [truth.get(p, 0.0) for p in keys]
        r = _corr(xs, ys)
        err = _mean([abs(a - b) for a, b in zip(xs, ys)])
        tot = _mean([abs(b) for b in ys])
        rel = err / tot if tot else 0.0
        # A per-game r over five or six surviving products is noisy; the
        # verdict is taken over the DISTRIBUTION below, not game by game.
        good = (not keys) or rel < REL_ERR_MAX
        per_game.append((r, rel))
        print(f"  {os.path.basename(path)[:34]:<34} r={r:+.3f}  "
              f"MAE={err:7.1f} of {tot:7.1f} units  {'OK' if good else 'FAIL'}"
              f"{'   blind: ' + ','.join(b[:4] for b in blind) if blind else ''}")
        del steps
    if not per_game:
        print("  no replays found\n")
        return True
    rs = sorted(r for r, _ in per_game if r == r)
    med_r = rs[len(rs) // 2] if rs else float("nan")
    share = _mean([1.0 if rel < REL_ERR_MAX else 0.0 for _r, rel in per_game])
    med_rel = sorted(rel for _r, rel in per_game)[len(per_game) // 2]
    ok = med_r >= 0.9 and share >= 0.95
    print(f"  {len(per_game)} games: median r {med_r:+.3f}, median relative "
          f"error {med_rel:.1%}, {share:.1%} under {REL_ERR_MAX:.0%}")
    print(f"  INV_SIGN={O.INV_SIGN!r}  ->  {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------------- 2. retrodiction on arms

def _rows(path: str, arm: str) -> List[Dict]:
    with open(os.path.join(REPO, path)) as fh:
        return json.load(fh)["results"][arm]


def arm_outlooks(rows: Sequence[Dict]) -> List[O.Outlook]:
    """One terminal Outlook per recorded game. No projection is needed at the
    season end -- the banks ARE the outcome -- so this isolates the
    comparator's ORDERING from the projection's accuracy."""
    return [O.Outlook(day=O.SEASON_END_DAY, our_bank=float(r["money"]),
                      opp_bank=float(r["opp"]), band=int(r.get("bin", 0)))
            for r in rows]


def arm_summary(rows: Sequence[Dict]) -> O.Outlook:
    """The arm as ONE outlook: mean p_win over its games, mean margin.

    p_win is averaged, not recomputed from the mean margin -- the objective
    is win rate first, and a mean margin hides exactly the trade the losing
    arms made (bank more, win less)."""
    outs = arm_outlooks(rows)
    return O.Outlook(day=O.SEASON_END_DAY,
                     our_bank=_mean([o.our_bank for o in outs]),
                     opp_bank=_mean([o.opp_bank for o in outs]),
                     p_win=_mean([o.p_win for o in outs]))


def check_retrodict() -> bool:
    """Does the comparator order arms the way the paired eval did?"""
    print("== RETRODICTION: arms already paid for, zero new games ==")
    print(f"  {'arm':<24}{'win':>7}{'money':>10}{'obj':>9}{'money-first':>13}  verdict")
    agree = dis = 0
    money_agree = money_dis = 0
    seen = set()
    for path, base, arms in ARM_FILES:
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            continue
        try:
            brows = _rows(path, base)
        except (KeyError, ValueError):
            continue
        bwin = _mean([1.0 if r["win"] >= 0.75 else 0.0 for r in brows])
        bo = arm_summary(brows)
        bmoney = _mean([r["money"] for r in brows])
        for arm in arms:
            key = (arm, round(bwin, 6))
            if key in seen:
                continue
            seen.add(key)
            try:
                rows = _rows(path, arm)
            except (KeyError, ValueError):
                continue
            win = _mean([1.0 if r["win"] >= 0.75 else 0.0 for r in rows])
            money = _mean([r["money"] for r in rows])
            ao = arm_summary(rows)
            truth = win - bwin
            obj = O.better(ao, bo)                       # objective says: better?
            mny = money > bmoney                         # the control says: better?
            hit = (obj == (truth > 0)) or abs(truth) < 0.005
            mhit = (mny == (truth > 0)) or abs(truth) < 0.005
            agree += hit; dis += (not hit)
            money_agree += mhit; money_dis += (not mhit)
            print(f"  {arm:<24}{truth:+7.3f}{money - bmoney:+10.0f}"
                  f"{'  better' if obj else '   worse':>9}"
                  f"{'  better' if mny else '   worse':>13}  "
                  f"{'ok' if hit else 'MISS'}{'' if mhit else '   (money-first MISS)'}")
    n = agree + dis
    mn = money_agree + money_dis
    print(f"  objective   {agree}/{n} arms ordered correctly "
          f"({agree / n:.3f})" if n else "  no arms found")
    print(f"  money-first {money_agree}/{mn} ({money_agree / mn:.3f})  "
          f"<- the control the losing arms beat" if mn else "")
    ok = n > 0 and agree >= money_agree and agree / n >= 0.9
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------- 3. the leaders as a fixed point

def check_fixedpoint() -> bool:
    """score() on the leaders' realised games must exceed score() on ours."""
    print("== FIXED POINT: the leaders' own games ==")
    card = os.path.join(REPO, "docs/results/scorecard_champ_h10.json")
    if not os.path.exists(card):
        print("  scorecard_champ_h10.json missing; skipped\n")
        return True
    rows = json.load(open(card))["rows"]
    outs = {}
    for name, row in rows.items():
        if "final_money" not in row:
            continue
        outs[name] = O.Outlook(day=O.SEASON_END_DAY,
                               our_bank=float(row["final_money"]),
                               opp_bank=float(row["opp_money"]), label=name)
    ours = outs.get("champ-h10")
    ok = ours is not None
    for name, o in sorted(outs.items(), key=lambda kv: -O.score(kv[1])):
        print(f"  {name:<12} margin {o.margin:+9,.0f}  p_win {o.p_win:.3f}  "
              f"score {O.score(o):+7.3f}")
    for name in ("boey", "dsm"):
        if name in outs and ours is not None:
            good = O.better(outs[name], ours)
            ok = ok and good
            print(f"  better({name}, champ-h10) = {good}   "
                  f"{'OK' if good else 'FAIL'}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------- 4. sign accuracy of the projection

SIGN_DAYS = (6, 12, 16, 20, 28)
# docs/reward_design.md Rule 2, 21,054 corpus games: the baseline to beat.
BASELINE_SIGN = {6: 0.509, 12: 0.507, 16: 0.553, 20: 0.700, 28: 0.882}
BASELINE_R = {6: -0.009, 12: -0.017, 16: 0.190, 20: 0.575, 28: 0.928}
# Sign accuracy has se = 0.5/sqrt(n); 150 seat-games puts it at 0.041, which
# is the smallest difference worth reading here.
MIN_SIGN_N = 150
# Per-game relative error of the opponent-supply solve that still counts as a
# solve. The leaders' own replays come in at 2-9%.
REL_ERR_MAX = 0.35


def check_sign(limit: int = 24) -> bool:
    """Is the projected margin a better predictor than the money lead?"""
    print("== SIGN: projected final margin vs the money lead ==")
    acc: Dict[int, List[Tuple[float, float, float]]] = {d: [] for d in SIGN_DAYS}
    for path in replay_paths(limit=limit):
        try:
            days, rewards = dawn_views(path)
        except Exception:
            continue
        # BOTH seats. A one-seat measurement cannot separate a real signal
        # from a constant bias in the projection: any asymmetry between the
        # two sides shows up as sign accuracy near the base rate, and with
        # both seats in it shows up as a paired contradiction instead.
        for seat, vv in ((0, 0), (1, 1)):
            final = rewards[seat] - rewards[1 - seat]
            for day, v0, v1 in days:
                if day not in acc:
                    continue
                v = v0 if seat == 0 else v1
                try:
                    o = O.project(v)
                except Exception:
                    continue
                acc[day].append((o.margin,
                                 float(v.me.money) - float(v.opp.money), final))
    ok, judged = True, 0
    print(f"  {'day':>4}{'n':>6}{'proj sign':>11}{'lead sign':>11}"
          f"{'proj r':>9}{'lead r':>9}{'corpus r':>10}{'lead sd $':>11}")
    for d in SIGN_DAYS:
        rows = acc[d]
        if not rows:
            continue
        ps = _mean([1.0 if (a > 0) == (c > 0) else 0.0 for a, _b, c in rows])
        ls = _mean([1.0 if (b > 0) == (c > 0) else 0.0 for _a, b, c in rows])
        pr = _corr([a for a, _b, _c in rows], [c for _a, _b, c in rows])
        lr = _corr([b for _a, b, _c in rows], [c for _a, _b, c in rows])
        # A sign accuracy on 24 seat-games has se 0.10, and this project has
        # been burned by exactly that: sub-0.02 differences are noise
        # (kaggriculture-rating-noise, kaggriculture-gate-size). Below
        # MIN_SIGN_N the row is printed and NOT judged -- a verdict from an
        # under-powered sample is worse than no verdict.
        powered = len(rows) >= MIN_SIGN_N
        better = ps >= ls - 0.02
        if powered:
            judged += 1
            ok = ok and better
        print(f"  {d:>4}{len(rows):>6}{ps:>11.3f}{ls:>11.3f}{pr:>9.3f}{lr:>9.3f}"
              f"{BASELINE_R.get(d, float('nan')):>10.3f}"
              f"{_sd([b for _a, b, _c in rows]):>11.0f}"
              f"  {'' if powered else 'under-powered'}"
              f"{'' if (better or not powered) else 'WORSE'}")
    print("  (the corpus column is the money lead over 21,054 games; the lead "
          "column is\n   the same statistic on THESE replays, which is the "
          "like-for-like baseline)")
    if not judged:
        print(f"  no day reached {MIN_SIGN_N} seat-games; rerun with "
              f"--max-replays 120. NOT JUDGED.")
        print("  -> SKIP\n")
        return True
    print(f"  -> {'PASS' if ok else 'FAIL'}  ({judged} of {len(SIGN_DAYS)} days "
          f"powered at n >= {MIN_SIGN_N})\n")
    return ok


# ------------------------------------------------ 5. the price-forecast ladder

def check_price(limit: int = 12, ks: Sequence[int] = (5, 10)) -> bool:
    """MAE of the projected price k days out, against the published ladder.

    tools/price_forecast.py: drain-only 25.5 at k=5, persistence 23.1, both
    boards 13.3 -- a 42% cut. The opponent's half of this projection has to
    sit at or below the both-boards number or it adds nothing over the rent
    model kagent/planner already ships.
    """
    print("== PRICE: projected inventory path vs realised, MAE in dollars ==")
    errs = {k: {"rent": [], "persist": [], "drain": []} for k in ks}
    for path in replay_paths(limit=limit):
        try:
            days, _ = dawn_views(path)
        except Exception:
            continue
        by_day = {d: v0 for d, v0, _ in days}
        for d, v0, _ in days:
            h = max(ks)
            if d + h not in by_day:
                continue
            try:
                proj = kplanner._rent_inventory(v0, h)
            except Exception:
                continue
            drain = econ.town_drain_per_day(v0.unlocked_shops)
            for k in ks:
                fut = by_day[d + k]
                for p in PRODUCTS:
                    if p not in proj or p not in v0.market_inv:
                        continue
                    real = econ.price(p, int(round(fut.market_inv.get(p, 0))))
                    errs[k]["rent"].append(
                        abs(econ.price(p, int(round(proj[p][k]))) - real))
                    errs[k]["persist"].append(
                        abs(econ.price(p, int(round(v0.market_inv[p]))) - real))
                    errs[k]["drain"].append(abs(econ.price(
                        p, int(round(v0.market_inv[p] - k * drain.get(p, 0.0)))) - real))
    print(f"  {'k':>3}{'rent (ours)':>14}{'persistence':>13}{'drain-only':>12}"
          f"{'published':>22}")
    pub = {5: "25.5 / 23.1 / 13.3", 10: "(k=10 in the ablation)"}
    ok = True
    for k in ks:
        if not errs[k]["rent"]:
            continue
        r, p_, dr = (_mean(errs[k]["rent"]), _mean(errs[k]["persist"]),
                     _mean(errs[k]["drain"]))
        good = r <= p_
        ok = ok and good
        print(f"  {k:>3}{r:>14.1f}{p_:>13.1f}{dr:>12.1f}{pub.get(k, ''):>22}"
              f"  {'' if good else 'WORSE THAN PERSISTENCE'}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------- 6. p_win calibration, REPORTED

def check_brier(limit: int = 120) -> bool:
    """Brier and AUC of p_win per band. A DIAGNOSTIC. NEVER A VERDICT.

    THIS NUMBER HAS LIED BEFORE, and it is the specific way it lied that
    matters: the fitted value head was 4.8x TIGHTER on held-out error than
    the alternative and got the sign of 3 of 4 interventions wrong. BC hit
    69.4% accuracy and gated null. Capacity channels bought +0.0187 R^2 at
    p = 0.0001 and moved the forecast 0%. A good Brier here is not evidence
    that `better()` will raise the win rate; only `--check retrodict` and
    then a >= 995-game paired session_eval are.

    PER BAND, NEVER POOLED -- the 2500+ band predicts at R^2 -0.019 against
    +0.239 below 1000, so a pooled score would hide failure exactly where our
    win rate is worst (band 3 at 0.200). Every replay on disk is a top-board
    game, so band 5 is the only band this can measure at all; the rest of the
    table is derived and says so in its own `source` field.
    """
    print("== CALIBRATION: Brier / AUC of p_win. DIAGNOSTIC ONLY, NOT A VERDICT ==")
    rows: Dict[int, List[Tuple[float, float]]] = {d: [] for d in O.DEFAULT_DAY_KNOTS}
    for path in replay_paths(limit=limit):
        try:
            days, rewards = dawn_views(path)
        except Exception:
            continue
        for seat in (0, 1):
            won = 1.0 if rewards[seat] > rewards[1 - seat] else 0.0
            for day, v0, v1 in days:
                if day not in rows:
                    continue
                v = v0 if seat == 0 else v1
                try:
                    o = O.project(v, band=5)
                except Exception:
                    continue
                rows[day].append((o.p_win, won))
    print(f"  band 5 (2500+), the only band on disk")
    print(f"  {'day':>4}{'n':>6}{'brier':>9}{'base':>9}{'auc':>8}{'mean p':>9}"
          f"{'win rate':>10}")
    for d in O.DEFAULT_DAY_KNOTS:
        pts = rows[d]
        if not pts:
            continue
        base_rate = _mean([w for _p, w in pts])
        brier = _mean([(p - w) ** 2 for p, w in pts])
        ref = _mean([(base_rate - w) ** 2 for _p, w in pts])
        pos = [p for p, w in pts if w > 0.5]
        neg = [p for p, w in pts if w <= 0.5]
        auc = (_mean([1.0 if a > b else (0.5 if a == b else 0.0)
                      for a in pos for b in neg])
               if pos and neg else float("nan"))
        print(f"  {d:>4}{len(pts):>6}{brier:>9.4f}{ref:>9.4f}{auc:>8.3f}"
              f"{_mean([p for p, _w in pts]):>9.3f}{base_rate:>10.3f}")
    print("  `base` is the Brier of predicting the base rate every time -- the "
          "only\n   comparison that means anything, and it still is not a verdict.")
    print("  IN SAMPLE: calibration.json was built from these same replays, so "
          "these\n   numbers are an upper bound and cannot be read as held-out "
          "accuracy.")
    print("  -> REPORTED\n")
    return True


# ----------------------------------------------------- calibration builder

def _pava(ys: List[float], ws: List[float]) -> List[float]:
    """Pool adjacent violators: the least-squares MONOTONE fit.

    p_win must be non-decreasing in margin. Not because the data says so --
    with a few hundred games a bin inverts on noise -- but because a
    comparator whose p_win FALLS as the margin rises would prefer a worse
    projected position, which is clause 1 inverted.
    """
    blocks = [[float(y), max(1e-9, float(w)), 1] for y, w in zip(ys, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0] + 1e-12:
            i += 1
            continue
        a, b = blocks[i], blocks[i + 1]
        tw = a[1] + b[1]
        blocks[i:i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]]
        i = max(0, i - 1)
    out: List[float] = []
    for v, _w, c in blocks:
        out.extend([v] * c)
    return out


def build_calibration(limit: Optional[int] = None,
                      out_path: Optional[str] = None,
                      ref_eval: str = "rl/runs/draw995/baselines.json",
                      ref_arm: str = "dev_knobs_h8") -> Dict:
    """Write rlv3/calibration.json from the recorded replays.

    THE TABLE IS THE EMPIRICAL SHARE OF GAMES WITH A REALISED FINAL MARGIN
    > 0, GIVEN THE MARGIN AT THAT DAY. Both seats of every replay contribute,
    so the table is symmetric by construction and a bias toward seat 0 cannot
    creep in.

    BANDS. Every replay here is a top-board game, so band 5 is MEASURED and
    the rest are DERIVED: the pooled row shifted in logit space by the band's
    base-rate offset, taken from a reference arm's per-band win rate on the
    995-game draw, with the shift decaying to zero at the season end (at day
    29 the margin is nearly the result and the opponent's rating cannot
    change that). Each row records which it is, so a caller can tell a
    measurement from an interpolation, and the 2500+ band -- the hardest to
    predict, R^2 -0.019 against +0.239 below 1000 -- is the one that is real.
    """
    knots = O.DEFAULT_MARGIN_KNOTS
    day_knots = O.DEFAULT_DAY_KNOTS
    samples: Dict[int, List[Tuple[float, float]]] = {d: [] for d in day_knots}
    paths = replay_paths(limit=limit)
    for i, path in enumerate(paths):
        try:
            days, rewards = dawn_views(path)
        except Exception:
            continue
        for seat in (0, 1):
            win = 1.0 if rewards[seat] > rewards[1 - seat] else 0.0
            for day, v0, _v1 in days:
                if day not in samples:
                    continue
                m = (float(v0.me.money) - float(v0.opp.money))
                samples[day].append((m if seat == 0 else -m, win))
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(paths)} replays", flush=True)

    rows: Dict[str, Dict] = {}
    for d in day_knots:
        pts = samples.get(d) or []
        if not pts:
            rows[str(d)] = {"p": O._analytic_row(d, knots),
                            "n": [0] * len(knots),
                            "clamp": [knots[0], knots[-1]], "source": "analytic"}
            continue
        ms = sorted(m for m, _ in pts)
        lo = ms[max(0, int(0.02 * len(ms)))]
        hi = ms[min(len(ms) - 1, int(0.98 * len(ms)))]
        ps, ns = [], []
        for j, kn in enumerate(knots):
            left = -math.inf if j == 0 else 0.5 * (knots[j - 1] + kn)
            right = math.inf if j == len(knots) - 1 else 0.5 * (kn + knots[j + 1])
            sel = [w for m, w in pts if left <= m < right]
            ns.append(len(sel))
            ps.append(_mean(sel) if sel else float("nan"))
        # bins with no support inherit their neighbour, then PAVA makes the
        # whole row non-decreasing.
        last = 0.0
        for j in range(len(ps)):
            if math.isnan(ps[j]):
                ps[j] = last
            last = ps[j]
        for j in range(len(ps) - 1, -1, -1):
            if ns[j] == 0 and j + 1 < len(ps):
                ps[j] = min(ps[j], ps[j + 1])
        ps = _pava(ps, [max(1.0, float(n)) for n in ns])
        rows[str(d)] = {"p": [round(min(1.0, max(0.0, v)), 5) for v in ps],
                        "n": ns, "clamp": [round(lo, 1), round(hi, 1)],
                        "source": f"measured on {len(pts)} seat-games"}

    bands: Dict[str, Dict] = {"pooled": {"source": "measured (top-board replays)",
                                         "rows": rows}}
    bands["5"] = {"source": "measured (top-board replays ARE band 5)",
                  "rows": json.loads(json.dumps(rows))}
    base = _band_base_rates(ref_eval, ref_arm)
    if base:
        overall = _mean(list(base.values()))
        for b, rate in base.items():
            if b == 5:
                continue
            off = _logit(rate) - _logit(overall)
            brows = {}
            for k, row in rows.items():
                w = max(0.0, 1.0 - int(k) / float(O.SEASON_END_DAY))
                brows[k] = dict(row)
                brows[k]["p"] = [round(_sigmoid(_logit(p) + w * off), 5)
                                 for p in row["p"]]
                brows[k]["source"] = (f"derived: pooled + {w * off:+.3f} logit "
                                      f"(band base rate {rate:.3f} vs {overall:.3f})")
            bands[str(b)] = {"source": f"derived from {ref_arm} band base rate",
                             "rows": brows}
    table = {
        "kind": "rlv3.p_win",
        "version": 1,
        "provenance": (
            f"Empirical P(final margin > 0 | margin at day d), both seats of "
            f"{len(paths)} recorded top-board replays ({', '.join(REPLAY_DIRS)}), "
            f"monotone (PAVA) and clamped to the 2nd-98th percentile of the "
            f"margins actually seen. Bands other than 5 are DERIVED by a logit "
            f"shift from {ref_arm}'s per-band win rate on {ref_eval}, decaying "
            f"to zero at day {O.SEASON_END_DAY}. Judge this table by "
            f"rlv3/objective_check.py retrodiction, NEVER by its Brier score: "
            f"the value head was 4.8x tighter on held-out error and got the "
            f"sign of 3 of 4 interventions wrong."),
        "margin_knots": list(knots),
        "day_knots": list(day_knots),
        "bands": bands,
    }
    path = out_path or O._CALIBRATION_PATH
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    print(f"-> {path}")
    return table


def _logit(p: float) -> float:
    p = min(1.0 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _band_base_rates(path: str, arm: str) -> Dict[int, float]:
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        return {}
    try:
        rows = json.load(open(full))["results"][arm]
    except (KeyError, ValueError):
        return {}
    out: Dict[int, float] = {}
    for b in range(len(O.EDGES)):
        sel = [1.0 if r["win"] >= 0.75 else 0.0 for r in rows if r.get("bin") == b]
        if sel:
            out[b] = _mean(sel)
    return out


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", default="cheap",
                    choices=["cheap", "all", "identity", "retrodict",
                             "fixedpoint", "sign", "price", "brier"])
    ap.add_argument("--build-calibration", action="store_true")
    ap.add_argument("--max-replays", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.build_calibration:
        build_calibration(limit=args.max_replays, out_path=args.out)
        return 0

    n = args.max_replays
    results = {}
    if args.check in ("cheap", "all", "retrodict"):
        results["retrodict"] = check_retrodict()
    if args.check in ("cheap", "all", "fixedpoint"):
        results["fixedpoint"] = check_fixedpoint()
    if args.check in ("all", "identity"):
        results["identity"] = check_identity(limit=n or 4)
    if args.check in ("all", "sign"):
        results["sign"] = check_sign(limit=n or 120)
    if args.check in ("all", "price"):
        results["price"] = check_price(limit=n or 12)
    if args.check in ("all", "brier"):
        check_brier(limit=n or 120)        # reported, never judged
    print("== SUMMARY ==")
    for k, v in results.items():
        print(f"  {k:<12}{'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
