"""Fit Phi, and -- the part that actually decides anything -- score it.

Policy invariance is free: `tools/shaping.py` proves any Phi is safe. USEFUL
is not free. Ng et al.'s own result says Phi = V* collapses the problem to a
one-step decision and Phi = anything else is merely harmless, so the entire
engineering question is how close Phi gets to V*. V* here is the expected
final margin from the current state, and that is directly measurable: we have
400 games x 30 days x 48 metrics with the final margin and the win attached
(`rl/datasets/propagation_perturb.npz`, built by `tools/propagation.py` with
randomised openings -- the randomisation is what gives the early days any
cross-game spread at all).

So this tool answers one question per day of the season:

    given the state at day d, how much of the final margin can Phi see?

and reports it against three references: the trivial Phi = 0, the cash-only
Phi the trainer effectively had, and the zero-fit `LeadPotential`.

METHODOLOGY, and why each choice is not optional here.

  GROUP SPLIT BY GAME. The 30 rows of one game share one label. Splitting
  rows at random puts day 3 of a game in train and day 4 in test, and the
  R^2 that comes back is a memorised outcome. Games are the unit; days are
  not independent samples.

  ALPHA CHOSEN INSIDE TRAIN. Ridge alpha is picked by grouped K-fold over
  the TRAINING games only, then applied once to a test set of games that no
  fit and no selection has touched. `docs/...train-on-deployment` records
  what happens when a selection step quietly sees the evaluation set.

  STANDARDISED ON TRAIN STATISTICS. Test rows are transformed with the
  training mean and scale, never their own.

  PER-DAY MODELS. A single stationary model cannot express a feature whose
  correlation with the outcome flips sign between day 8 and day 10, and
  `docs/metric_propagation.md` measured exactly that. Day-dependent
  potentials keep the invariance guarantee (Devlin & Kudenko, AAMAS 2012).

    PYTHONPATH=. uv run python tools/fit_potential.py --report
    PYTHONPATH=. uv run python tools/fit_potential.py --out rl/datasets/potential.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import shaping as S  # noqa: E402

DEFAULT_DATA = os.path.join(REPO, "rl", "datasets", "propagation_perturb.npz")
DEFAULT_OUT = os.path.join(REPO, "rl", "datasets", "potential.json")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
DAYS = 30

# Every recorded metric is tanh-squashed into roughly [-1, 1], so a training
# standard deviation below this is a constant column, not a weak feature.
# Clamping such a column instead of dropping it is what produced R^2 = -214
# on day 17 of the first run: `D:drain_cover` had train sd 8.5e-8, survived a
# 1e-9 clamp, and its de-standardised coefficient came back at ~1e7, whose
# intercept cancellation has no float64 precision left. Degenerate columns
# get DROPPED, and `_check_destandardised` fails the fit if any survivor
# still cannot be de-standardised exactly. Raised 1e-4 -> 1e-3 after the
# second run: `E:money_over_next_land` sat at train sd 4.5e-4 on day 17, took
# a coefficient of 5.2e5, and met a test row 3,677 sigmas out.
MIN_SD = 1e-3


# ------------------------------------------------------------------ data

class Panel:
    """Per-day metric rows with the game they came from and its outcome."""

    def __init__(self, path: str):
        blob = np.load(path, allow_pickle=True)
        self.X = np.asarray(blob["X"], dtype=np.float64)
        self.day = np.asarray(blob["D"], dtype=int)
        self.margin = np.asarray(blob["M"], dtype=np.float64)
        self.win = np.asarray(blob["W"], dtype=np.float64)
        self.names: List[str] = [str(s) for s in blob["names"]]
        n_days = len(set(self.day.tolist()))
        if "G" in blob.files:
            # Explicit groups. `tools/ladder_panel.py` emits BOTH seats of a
            # session, which share a world, so the split must keep them
            # together or the "held-out" games are half-seen.
            g = np.asarray(blob["G"], dtype=int)
            uniq = {v: i for i, v in enumerate(sorted(set(g.tolist())))}
            self.game = np.array([uniq[v] for v in g], dtype=int)
            self.seat = (np.asarray(blob["S"], dtype=int)
                         if "S" in blob.files else np.zeros(len(g), int))
            self.rating = (np.asarray(blob["R"], dtype=float)
                           if "R" in blob.files else
                           np.full(len(g), np.nan))
        else:
            if len(self.X) % n_days:
                raise ValueError(
                    f"{path}: {len(self.X)} rows is not a whole number of "
                    f"{n_days}-day games -- the game grouping would be wrong")
            self.game = np.arange(len(self.X)) // n_days
            self.seat = np.zeros(len(self.X), int)
            self.rating = np.full(len(self.X), np.nan)
            # M and W are recorded once per game and repeated down its rows.
            for arr, label in ((self.margin, "margin"), (self.win, "win")):
                per_game = arr.reshape(-1, n_days)
                if not np.allclose(per_game, per_game[:, :1]):
                    raise ValueError(f"{path}: {label} varies within a game; "
                                     f"the row->game mapping is wrong")
        self.n_days = n_days

    @property
    def n_games(self) -> int:
        return int(self.game[-1]) + 1

    def split(self, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Disjoint GAME ids: nothing from a test game is ever fitted on."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.n_games)
        cut = int(round(self.n_games * frac))
        return np.sort(order[:cut]), np.sort(order[cut:])


# ------------------------------------------------------------------ ridge

def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Closed-form ridge on centred data; the intercept is not penalised."""
    mu, ybar = X.mean(0), y.mean()
    Xc = X - mu
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - ybar))
    return w, float(ybar - mu @ w)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC of `score` against binary `y`. Ties get half credit."""
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def live_columns(X: np.ndarray) -> np.ndarray:
    """Indices of columns with real variation in THIS training slice."""
    return np.flatnonzero(X.std(0) >= MIN_SD)


def _zscore(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < MIN_SD, 1.0, sd)
    return ((train - mu) / sd,) + tuple((o - mu) / sd for o in others)


def _check_destandardised(X: np.ndarray, w_raw: np.ndarray, b_raw: float,
                          Xs: np.ndarray, w: np.ndarray, b: float,
                          day: int) -> None:
    """Folding the z-scoring into raw coefficients must be lossless.

    It is not, for a near-constant column: the intercept becomes a difference
    of two large numbers. Rather than trust the algebra, compare the two
    forms on the training rows and refuse the fit if they disagree.
    """
    a, c = X @ w_raw + b_raw, Xs @ w + b
    scale = max(1.0, float(np.abs(c).max()))
    err = float(np.abs(a - c).max()) / scale
    if err > 1e-6:
        raise FloatingPointError(
            f"day {day}: de-standardised coefficients lose precision "
            f"(rel err {err:.2e}); a near-constant column survived MIN_SD")


def pick_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               folds: int, seed: int) -> float:
    """Grouped K-fold over the TRAINING games only."""
    uniq = np.unique(groups)
    if len(uniq) < folds:
        return ALPHAS[len(ALPHAS) // 2]
    rng = np.random.default_rng(seed)
    assign = {g: i % folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([assign[g] for g in groups])
    scores = []
    for a in ALPHAS:
        oof = np.empty_like(y)
        for k in range(folds):
            tr, te = fold != k, fold == k
            if te.sum() == 0 or tr.sum() < 2:
                oof[te] = y.mean()
                continue
            xtr, xte = _zscore(X[tr], X[te])
            w, b = ridge(xtr, y[tr], a)
            oof[te] = xte @ w + b
        scores.append(r2(y, oof))
    return ALPHAS[int(np.argmax(scores))]


# ------------------------------------------------------------------ the fit

def allowed_columns(names: Sequence[str], day: int, rules: bool) -> List[int]:
    """Which metrics the potential may read on this day.

    With `rules`, the two measured findings in `docs/reward_design.md` are
    imposed as constraints rather than hoped for: no cash before day 14
    (Rule 2 -- r ~ 0.00, sign-correct 50%), and no absolute levels ever
    (Rule 1 -- differencing within the game is worth 2-6x). Whether that
    HELPS is hypothesis H8; the unconstrained fit is its control.
    """
    out = []
    for i, n in enumerate(names):
        if rules and n in S.LEVEL_FEATURES:
            continue
        if rules and day < S.CASH_EMBARGO_DAY and n in S.CASH_FEATURES:
            continue
        out.append(i)
    return out


def fit(panel: Panel, train_games: np.ndarray, folds: int, seed: int,
        target: str = "margin", rules: bool = False
        ) -> Tuple[S.LinearPotential, Dict[int, float]]:
    y_all = panel.margin if target == "margin" else panel.win
    in_train = np.isin(panel.game, train_games)
    coefs: Dict[int, Dict[str, float]] = {}
    inter: Dict[int, float] = {}
    alphas: Dict[int, float] = {}
    clip: Dict[int, Tuple[float, float]] = {}
    dropped: Dict[int, int] = {}
    for d in range(panel.n_days):
        m = in_train & (panel.day == d)
        Xfull, y, g = panel.X[m], y_all[m], panel.game[m]
        cols = sorted(set(allowed_columns(panel.names, d, rules))
                      & set(live_columns(Xfull).tolist()))
        if not cols:
            coefs[d], inter[d], alphas[d] = {}, float(y.mean()), ALPHAS[0]
            dropped[d] = len(panel.names)
            continue
        X = Xfull[:, cols]
        a = pick_alpha(X, y, g, folds, seed + d)
        Xs, = _zscore(X)
        mu, sd = X.mean(0), X.std(0)
        w, b = ridge(Xs, y, a)
        # fold the standardisation back in so the potential eats raw metrics
        w_raw = w / sd
        b_raw = float(b - float(mu @ w_raw))
        _check_destandardised(X, w_raw, b_raw, Xs, w, b, d)
        coefs[d] = {panel.names[c]: float(v)
                    for c, v in zip(cols, w_raw) if v != 0.0}
        inter[d] = b_raw
        alphas[d] = a
        # the range the fit actually saw, widened by half, as the clamp
        pred = X @ w_raw + b_raw
        lo, hi = float(pred.min()), float(pred.max())
        pad = 0.5 * (hi - lo)
        clip[d] = (lo - pad, hi + pad)
        dropped[d] = len(panel.names) - len(cols)
    note = (f"ridge on {len(train_games)} games, per-day, target={target}, "
            f"rules={'on' if rules else 'off'}; by tools/fit_potential.py")
    print(f"  {'rules' if rules else 'all 48'}: dropped "
          f"{min(dropped.values())}-{max(dropped.values())} columns/day "
          f"(constant in train, or blocked by reward_design)")
    return (S.LinearPotential(coefs=coefs, intercepts=inter, clip=clip,
                              note=note), alphas)


# ------------------------------------------------------------------ scoring

def row_feats(panel: Panel, i: int) -> Dict[str, float]:
    return {n: float(v) for n, v in zip(panel.names, panel.X[i])}


def score(panel: Panel, phi: S.Potential, games: np.ndarray
          ) -> Dict[int, Tuple[float, float]]:
    """(R^2 vs final margin, AUC vs win) per day, on held-out games."""
    keep = np.isin(panel.game, games)
    out = {}
    for d in range(panel.n_days):
        m = keep & (panel.day == d)
        idx = np.flatnonzero(m)
        pred = np.array([phi(row_feats(panel, i), d) for i in idx])
        out[d] = (r2(panel.margin[idx], pred), auc(panel.win[idx], pred))
    return out


class CashPotential(S.Potential):
    """What the trainer's portfolio delta effectively tracked: money, always."""

    def __call__(self, feats, day):
        return 5000.0 * float(feats.get("E:money_lead", 0.0))


# ------------------------------------------------------------------ report

def report(panel: Panel, rows: Dict[str, Dict[int, Tuple[float, float]]],
           alphas: Dict[int, float]) -> None:
    labels = list(rows)
    print(f"\nheld-out games only. R^2 against the final margin "
          f"(and AUC against the win)\n")
    print("  day  " + "  ".join(f"{k:>18}" for k in labels))
    for d in range(panel.n_days):
        cells = []
        for k in labels:
            r, a = rows[k][d]
            cells.append(f"{r:+7.3f} / {a:5.3f}")
        print(f"  {d:>3}  " + "  ".join(f"{c:>18}" for c in cells))
    print("\n  mean R^2   " + "  ".join(
        f"{k}={np.nanmean([rows[k][d][0] for d in range(panel.n_days)]):+.3f}"
        for k in labels))
    print("  mean AUC   " + "  ".join(
        f"{k}={np.nanmean([rows[k][d][1] for d in range(panel.n_days)]):5.3f}"
        for k in labels))
    print(f"\n  ridge alpha by day: "
          f"{sorted(set(alphas.values()))} "
          f"(mode {max(set(alphas.values()), key=list(alphas.values()).count)})")


def legacy_bias_report(panel: Panel, gamma: float) -> None:
    """How large the missing gamma was, in the trainer's own reward units.

    Phi is reconstructed as the trainer's `portfolio_value / 1000`: cash plus
    the board, both recorded tanh-squashed, so this is a scale proxy and is
    reported as such. What is exact is the SHAPE -- the bias is the discounted
    integral of Phi over the season, so it grows with how long value is held.
    """
    money = 30_000.0 * panel.X[:, panel.names.index("E:money")]
    board = 20_000.0 * panel.X[:, panel.names.index("M:board_value")]
    pv = (money + board) / 1000.0
    biases, spans = [], []
    for g in range(panel.n_games):
        traj = pv[panel.game == g]
        biases.append(S.legacy_bias(list(traj), gamma))
        spans.append(float(np.mean(traj)))
    b = float(np.mean(biases))
    print(f"\nH2 on real trajectories ({panel.n_games} games, gamma={gamma}):")
    print(f"  mean Phi over the season      {np.mean(spans):8.2f} reward units")
    print(f"  free return from the missing gamma  {b:8.2f} reward units/episode")
    print(f"  ... against WIN_COEF swing of      {2 * 6.0:8.2f} "
          f"(rl/train.py WIN_COEF = 6.0, reward is 2*win-1)")
    print(f"  ratio                              {b / 12.0:8.1%} of the win signal")
    print("  and it is paid for HOLDING, so it is a bribe against liquidating "
          "into\n  an engine that scores sold product only.")


def repeat_report(panel: Panel, args) -> None:
    """Mean and spread over independent GAME splits.

    A single 70/30 split of 400 games is one draw. The numbers below are what
    survives re-drawing it, which is the only version worth quoting.
    """
    acc: Dict[str, List[Tuple[float, float]]] = {}
    for rep in range(args.repeats):
        seed = args.seed + 1000 * rep
        tr, te = panel.split(args.train_frac, seed)
        arms = {
            "fit (all 48)": fit(panel, tr, args.folds, seed, rules=False)[0],
            "fit (rules)": fit(panel, tr, args.folds, seed, rules=True)[0],
            "lead (zero-fit)": S.LeadPotential(),
            "cash only": CashPotential(),
        }
        for k, phi in arms.items():
            per_day = score(panel, phi, te)
            r = float(np.nanmean([per_day[d][0] for d in range(panel.n_days)]))
            a = float(np.nanmean([per_day[d][1] for d in range(panel.n_days)]))
            acc.setdefault(k, []).append((r, a))
    print(f"\n{args.repeats} independent game splits, held-out means:\n")
    print(f"  {'potential':>18}   {'R^2 (margin)':>20}   {'AUC (win)':>18}")
    for k, vals in acc.items():
        r = np.array([v[0] for v in vals]); a = np.array([v[1] for v in vals])
        print(f"  {k:>18}   {r.mean():+7.3f} +/- {r.std(ddof=1):5.3f}      "
              f"{a.mean():6.3f} +/- {a.std(ddof=1):5.3f}")
    base = np.array([v[0] for v in acc["cash only"]])
    for k in ("fit (all 48)", "fit (rules)"):
        d = np.array([v[0] for v in acc[k]]) - base
        print(f"\n  {k} - cash only:  {d.mean():+.3f} +/- "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.3f}  (paired on the split)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default="", help="write the fitted potential here")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-split the games this many times and report the "
                         "spread; one split is an anecdote")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rules", action="store_true",
                    help="write the fit that obeys reward_design Rules 1 and 2 "
                         "(no absolute levels, no cash before day 14)")
    args = ap.parse_args()

    panel = Panel(args.data)
    if args.repeats > 1:
        repeat_report(panel, args)
        return
    tr, te = panel.split(args.train_frac, args.seed)
    print(f"{args.data}: {panel.n_games} games x {panel.n_days} days x "
          f"{len(panel.names)} metrics")
    print(f"  split by GAME: {len(tr)} train / {len(te)} test "
          f"(no game appears in both)")

    phi, alphas = fit(panel, tr, args.folds, args.seed, rules=False)
    phi_r, _ = fit(panel, tr, args.folds, args.seed, rules=True)
    problems = phi.validate()
    assert phi_r.validate() == [], "the rule-constrained fit broke its own rules"

    rows = {
        "fit (all 48)": score(panel, phi, te),
        "fit (rules)": score(panel, phi_r, te),
        "lead (zero-fit)": score(panel, S.LeadPotential(), te),
        "cash only": score(panel, CashPotential(), te),
    }
    report(panel, rows, alphas)

    if args.rules:
        phi = phi_r
        print("\n  writing the RULE-CONSTRAINED fit (--rules)")

    if problems:
        print(f"\n  reward-design rules broken by the fit ({len(problems)}):")
        for p in problems[:6]:
            print(f"    - {p}")
        if len(problems) > 6:
            print(f"    ... and {len(problems) - 6} more")
        print("  (these cost USEFULNESS, never correctness -- PBRS stays "
              "policy-invariant\n   whatever Phi reads)")

    if args.report:
        legacy_bias_report(panel, args.gamma)

    if args.out:
        phi.to_json(args.out)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
