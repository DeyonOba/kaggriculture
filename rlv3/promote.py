"""Which checkpoint is "best" -- under the lexicographic objective, and on the
995-game paired gate, never on anything the trainer said about itself.

    1. WIN RATE.  A win is `final money > opponent final money`.
    2. When winning, maximise profit.
    3. When losing, minimise the loss margin.

Lexicographic, not a weighted sum. Rank 2 is reached only on a win-rate tie
inside noise and rank 3 only on a tie there as well. A candidate that banks
more money and wins fewer games is REFUSED -- that is the exact failure mode
of every learned component this project has shipped so far: our bank rises
while the opponent's rises faster, and the market is one shared inventory, so
the score is relative.

WHY NOT THE TRAINER'S OWN SELECTION. `rl/train.py` keeps `best.pt` by the
argmax of mean final money over its gates, and prints a SESSION-WR computed
from 3 games a bin, 18 in all -- where 0.06 is one win and 0.11 is two. Six
Stage B arms were ranked on that. `kaggriculture-selfplay-winrate-refuted`
and `kaggriculture-win-rate-not-margin` are the same lesson from the other
side. Selection happens here, on 995 paired held-out games, with McNemar on
the discordant ones, which is the project's standing protocol.

Frozen baseline rows: `rl/runs/draw995/batch2.json`
    ship2_hv  (= champ-h10)  win 0.354  won +$24,596  lost -$17,100  n=995
    champ_h8                 win 0.230  won +$24,283  lost -$23,351  n=995
"""

from __future__ import annotations

import json
import os
import shutil

from rlv3 import fingerprint as FP
from rlv3 import manifest as MF
from rlv3 import objective as OBJ

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TWO COMPARATORS, ONE OBJECTIVE, DIFFERENT DOMAINS -- and they must not be
# confused. `rlv3.objective.better` is the IN-GAME one: it ranks projected
# Outlooks (p_win, margin, day, band) mid-season, with dead bands sized to a
# projection that is only 50.7% sign-correct through day 12. This module is
# the OFFLINE one: it ranks FINISHED agents over 995 paired games, where the
# uncertainty is sampling error and the right instrument is McNemar, not a
# dead band. Same three clauses, in the same order; nothing else is shared.
#
# What IS shared is the definition of a win, taken from the sibling rather
# than restated: rl.rollout writes win in {0.0, 0.5, 1.0} (a draw is 0.5 and
# is NOT a win), rl.session_eval.paired thresholds at >= 0.75, and
# objective.win_of is the same convention. 0.75 is the threshold that maps it
# to a boolean, and `_assert_win_convention` below keeps the three in step.
WIN = 0.75


def _assert_win_convention():
    """A win is `final money > opponent final money` -- in both comparators."""
    assert OBJ.win_of(10.0, 5.0) >= WIN, "a strictly larger bank must be a win"
    assert OBJ.win_of(5.0, 5.0) < WIN, "a draw must not count as a win"
    assert OBJ.win_of(5.0, 10.0) < WIN, "a smaller bank must not be a win"
    assert list(OBJ.EDGES) == list(_edges()), "rating bands disagree"


def _edges():
    from rl import session_eval
    return session_eval.EDGES

BASELINE_ROWS = os.path.join(REPO, "rl", "runs", "draw995", "batch2.json")
BASELINE_ARM = "ship2_hv"        # champ-h10 on the 995 draw


def load_rows(eval_json, arm):
    """-> the per-game rows of one arm of a `rl/session_eval.py --out` file."""
    with open(eval_json) as fh:
        data = json.load(fh)
    rows = (data.get("results") or {}).get(arm)
    if rows is None:
        raise KeyError(f"{eval_json}: no arm {arm!r} "
                       f"(have {sorted((data.get('results') or {}).keys())})")
    return rows


def margin(row):
    """The only definition of margin that matches the win rule: our bank minus
    theirs. `kaggriculture-market-congestion` -- the opponent's supply costs
    us $52,652 a game through the one shared variable, so an absolute bank
    number is not a score."""
    return float(row["money"]) - float(row["opp"])


def objective(rows):
    """-> (win_rate, mean margin over WON games, mean margin over LOST games, n).

    NaN for a margin with no games on that side, so a 0-win arm cannot look
    like a 0-margin one.
    """
    n = len(rows)
    if not n:
        return (float("nan"),) * 3 + (0,)
    won = [r for r in rows if float(r["win"]) >= WIN]
    lost = [r for r in rows if float(r["win"]) < WIN]
    mw = sum(margin(r) for r in won) / len(won) if won else float("nan")
    ml = sum(margin(r) for r in lost) / len(lost) if lost else float("nan")
    return (len(won) / n, mw, ml, n)


def per_band(rows):
    """-> {band: (win_rate, n)}. Every promotion records all six.

    `kaggriculture-band-not-top-games`: the 2500+ band is the hardest to
    predict and every board arm we ran used a rating FLOOR, which made it a
    top-only eval. champ-h10's bands are 0.440 / 0.367 / 0.366 / 0.200 /
    0.406 / 0.349 -- band 3 (1500-2000) is the hole.
    """
    from rl import session_eval
    out = {}
    for b in range(len(session_eval.EDGES)):
        rs = [r for r in rows if r.get("bin") == b]
        out[b] = ((sum(1 for r in rs if float(r["win"]) >= WIN) / len(rs))
                  if rs else float("nan"), len(rs))
    return out


def lexicographic_better(cand_rows, base_rows, win_eps=0.0, p_max=0.10):
    """-> (better, why). The whole objective, in one place.

    rank 1  `rl.session_eval.paired(base, cand)`: delta > win_eps with McNemar
            p <= p_max promotes. A delta below -win_eps refuses OUTRIGHT, at
            any margin -- this is the clause that stops "more money, fewer
            wins" from ever being promoted.
    rank 2  reached only on a win-rate tie inside noise (|delta| <= win_eps,
            or a positive delta that McNemar cannot separate): higher mean
            margin on WON games.
    rank 3  reached only on a tie there too: less negative mean margin on LOST
            games.

    The rows must be the SAME DRAW in the same order -- paired by world and by
    opponent. `rl.session_eval.reuse_results` is the check for that and the
    caller should have run it; length is verified here.
    """
    from rl import session_eval

    if len(cand_rows) != len(base_rows):
        raise ValueError(f"unpaired: {len(cand_rows)} candidate games vs "
                         f"{len(base_rows)} baseline -- refusing to compare")
    pr = session_eval.paired(base_rows, cand_rows)
    delta, p = pr["delta"], pr["p_mcnemar"]
    c_win, c_mw, c_ml, _ = objective(cand_rows)
    b_win, b_mw, b_ml, _ = objective(base_rows)
    head = (f"win {c_win:.3f} vs {b_win:.3f} (delta {delta:+.3f}, "
            f"p {p:.3f}, arm-only {pr['b_arm_only']}, base-only "
            f"{pr['c_base_only']})")

    if delta < -win_eps:
        return False, f"rank 1 REFUSED: fewer wins -- {head}"
    if delta > win_eps and p <= p_max:
        return True, f"rank 1: more wins -- {head}"

    # rank 2 -- win-rate tie inside noise
    tie = f"rank 1 tie inside noise ({head})"
    if _gt(c_mw, b_mw):
        return True, f"rank 2: {tie}; won-game margin {c_mw:+,.0f} > {b_mw:+,.0f}"
    if _gt(b_mw, c_mw):
        return False, f"rank 2: {tie}; won-game margin {c_mw:+,.0f} < {b_mw:+,.0f}"

    # rank 3 -- tie there too
    if _gt(c_ml, b_ml):
        return True, (f"rank 3: {tie}; won-game margin tied; lost-game margin "
                      f"{c_ml:+,.0f} > {b_ml:+,.0f}")
    return False, (f"rank 3: {tie}; won-game margin tied; lost-game margin "
                   f"{c_ml:+,.0f} <= {b_ml:+,.0f}")


def _gt(a, b, eps=1e-9):
    """Strict >, with NaN never winning."""
    if a != a or b != b:
        return False
    return a > b + eps


# ---------------------------------------------------------------- promotion
def promote(run_dir, eval_json, arm, baseline_arm="champ_h10",
            baseline_json=None, ppo_run=None, fp=None, win_eps=0.0,
            p_max=0.10, man=None):
    """Copy the winning checkpoint into `<run>/best/`. -> path, or None.

    `baseline_json` defaults to the eval_json itself (the gate plays champ-h10
    as an arm on the same draw); it falls back to the frozen
    `rl/runs/draw995/batch2.json` rows when the gate did not carry a baseline
    arm, which is only valid because the draw is identical -- checked by
    length and by the first row's (path, seat).

    Refuses outright, before any comparison, when the candidate's eval.json
    carries a different head version from today's fingerprint: those stored
    `choices` indices mean different land, crop and livestock targets.
    """
    cand = load_rows(eval_json, arm)
    base_src = eval_json
    try:
        base = load_rows(eval_json, baseline_arm)
    except KeyError:
        base_src = baseline_json or BASELINE_ROWS
        base = load_rows(base_src, BASELINE_ARM if base_src == BASELINE_ROWS
                         else baseline_arm)
        _assert_same_draw(cand, base, eval_json, base_src)

    ppo_run = ppo_run or os.path.join(run_dir, "ppo")
    weights = os.path.join(ppo_run, "eval.json")
    if fp is not None and os.path.exists(weights):
        FP.verify_weights_head(weights, fp)     # raises HeadMismatch

    better, why = lexicographic_better(cand, base, win_eps=win_eps, p_max=p_max)
    obj = objective(cand)
    man = MF.load(run_dir) if man is None else man
    man["gate"] = {"eval_json": os.path.abspath(eval_json), "arm": arm,
                   "baseline": {"arm": baseline_arm, "from": os.path.abspath(base_src),
                                "objective": list(objective(base))},
                   "objective": list(obj),
                   "per_band": {str(k): list(v) for k, v in per_band(cand).items()},
                   "verdict": why, "promoted": bool(better)}
    if not better:
        MF.save(run_dir, man)
        return None

    best_dir = os.path.join(run_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    copied = []
    for src in (os.path.join(ppo_run, "last.pt"), weights,
                os.path.join(ppo_run, "best.pt")):
        if not os.path.exists(src):
            continue
        dst = os.path.join(best_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        sp = FP.stamp_path(src)
        if os.path.exists(sp):
            shutil.copy2(sp, FP.stamp_path(dst))
        elif fp is not None:
            FP.write_stamp(dst, fp, FP.ACTION_KEYS + FP.BOARD_KEYS + ("corpus",))
        copied.append(dst)
        MF.record_artifact(man, dst, FP.read_stamp(dst))
    shutil.copy2(eval_json, os.path.join(best_dir, "session_eval.json"))
    man["best"] = {
        "path": copied[0] if copied else None,
        "from_slice": len(man.get("slices") or []) - 1,
        "objective": list(obj),
        "per_band": {str(k): list(v) for k, v in per_band(cand).items()},
        "beat": baseline_arm, "why": why, "files": copied,
    }
    MF.save(run_dir, man)
    return man["best"]["path"]


def _assert_same_draw(cand, base, cand_src, base_src):
    if len(cand) != len(base):
        raise SystemExit(
            f"{cand_src} has {len(cand)} games and {base_src} has {len(base)} "
            f"-- different --per-bin or a grown corpus. The 995-game draw "
            f"already changed once (785 -> 995 on 26 September); refusing to "
            f"compare win columns from two different world sets "
            f"(kaggriculture-gate-cache-stale).")
    for i, (a, b) in enumerate(zip(cand, base)):
        if a.get("path") != b.get("path") or a.get("script_side") != b.get("script_side"):
            raise SystemExit(
                f"game {i}: {cand_src} plays {a.get('path')} seat "
                f"{a.get('script_side')}, {base_src} plays {b.get('path')} "
                f"seat {b.get('script_side')} -- not the same draw; refusing")


def report(rows, label="arm"):
    """One line plus the six bands, in the scorecard's own shape."""
    win, mw, ml, n = objective(rows)
    bands = per_band(rows)
    return (f"{label:<12} win {win:.3f}  won {mw:+9,.0f}  lost {ml:+9,.0f}  "
            f"n={n}  bands " + " ".join(f"{bands[b][0]:.3f}" for b in sorted(bands)))


__all__ = ["WIN", "objective", "per_band", "margin", "load_rows",
           "lexicographic_better", "promote", "report",
           "BASELINE_ROWS", "BASELINE_ARM"]
