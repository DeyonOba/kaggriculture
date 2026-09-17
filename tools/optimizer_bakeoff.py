"""Give four black-box optimisers the same game budget on the same space, and
see which one finds the best plan.

Why a bake-off and not a training run. Fourteen PPO generations are on record
here and none beat the hand-written heuristic; behaviour cloning reached 69.4%
held-out action accuracy and gated NULL over 4,800 paired games. The single
thing that ever converted compute into a replicated out-of-sample gain was a
ROLLOUT over a per-day schedule (hiring, +$3,287 +/- 1,528). The recorded
verdict was "the knob space was never the plan space".

So the question is not "which neural architecture" -- gradient learning of a
policy has been refuted here four times -- but "which search, over the plan
space, with our sample budget". Measured on this machine: a game is ~4 s and
model fitting is free (10 epochs x 20k samples = 1.3 s on CPU, and 1.4 s on
MPS, i.e. the GPU is slower than the CPU at this size). 100% of the cost is
game simulation. That makes SAMPLE EFFICIENCY UNDER NOISE the only axis that
matters, which is exactly what separates these four methods.

The space: each per-day schedule in `tools/rollout_plan.SCHEDULES` is
represented by `k` control points interpolated over its active days
(`tools/sched_bundle.curve`), so four schedules cost ~4k dimensions instead of
120. Smoothness is not a guess -- the one schedule known to pay is a smooth
ramp.

The fitness: LADDER money, sampled. The 3-opponent panel is a refuted
instrument (it inverted on tape-112: 70.8% local, rated 1330.6) and champ-h8
wins only 5.6% of panel games, which is too near the floor to rank candidates.
Ladder money correlates +0.94 with rating. Search uses a small ladder sample;
the winner is re-measured on a held-out one.

    PYTHONPATH=. uv run python tools/optimizer_bakeoff.py --budget 60 --procs 5
    PYTHONPATH=. uv run python tools/optimizer_bakeoff.py --method cem --budget 200
"""

import argparse
import glob
import json
import os
import random
import statistics as st
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl.session_bench import bench_job, SESS_DIR          # noqa: E402
from tools.rollout_plan import SCHEDULES                  # noqa: E402
from tools.sched_bundle import bundle_for_multi, curve    # noqa: E402

# Control points per schedule, and the range each point may take. Ranges are
# the base schedule's own min/max widened 2x, not guesses: a threshold whose
# scale was guessed once produced three bit-identical gates.
SPACE = {
    "hires":  {"k": 4, "lo": 0, "hi": 16, "days": (0, 29)},
    "straw":  {"k": 3, "lo": 0, "hi": 45, "days": (5, 13)},
    "wheat":  {"k": 3, "lo": 0, "hi": 48, "days": (2, 25)},
    "land":   {"k": 3, "lo": 0, "hi": 3,  "days": (4, 20)},
}
DIMS = [(n, i) for n in SPACE for i in range(SPACE[n]["k"])]


def base_point():
    """The base schedules, read back as control points -- the search's origin."""
    pt = []
    for name, i in DIMS:
        spec, s = SPACE[name], SCHEDULES[name]["base"]
        d0, d1 = spec["days"]
        d = d0 + round((d1 - d0) * i / max(1, spec["k"] - 1))
        pt.append(float(s[min(d, len(s) - 1)]))
    return pt


def clamp(pt):
    return [max(SPACE[n]["lo"], min(SPACE[n]["hi"], v)) for (n, _), v in zip(DIMS, pt)]


def to_schedules(pt):
    out, j = {}, 0
    for name in SPACE:
        spec = SPACE[name]
        pts = pt[j:j + spec["k"]]
        j += spec["k"]
        out[name] = curve(pts, 30, spec["days"], lo=spec["lo"], hi=spec["hi"])
    return out


# --- fitness -----------------------------------------------------------------

def ladder_jobs(n_per_bin, heldout, seed, exclude=()):
    edges = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]
    paths = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.json")))
    if heldout:
        paths = [p for p in paths if int(p.split("-")[-1][:-5]) % 5 == 0]
    random.Random(seed).shuffle(paths)
    per = {i: [] for i in range(len(edges))}
    for p in paths:
        if p in exclude:
            continue
        s = json.load(open(p))
        sc = s.get("scores") or [None, None]
        for side in (0, 1):
            if sc[side] is None:
                continue
            b = next(i for i, (lo, hi) in enumerate(edges) if lo <= sc[side] < hi)
            if len(per[b]) < n_per_bin:
                per[b].append((p, side))
    return [(p, side) for b in per for p, side in per[b]]


class Fitness:
    """Ladder money per game, on a FIXED world sample so candidates are paired.

    Why money and not win rate, when the rating is win-based: at ~48 worlds an
    evaluation, a win rate carries se = sqrt(.25/48) = 7.2 points, which is
    larger than any effect we are chasing, so it cannot rank candidates. Ladder
    money is the denser signal and correlates +0.94 with rating. Win rate is
    still reported on the held-out block, where n is large enough to mean
    something.

    `per` returns the per-world values so the held-out comparison against the
    base can be PAIRED on identical worlds, which is the only way a ~$3k effect
    is visible against a ~$40k per-game spread.
    """

    def __init__(self, jobs, pool, ref=None, trail=None):
        self.jobs, self.pool, self.games = jobs, pool, 0
        # Append every evaluation to a JSONL as it happens. Without this a run
        # that is killed mid-search leaves NOTHING -- the per-method JSON is
        # only written when a method finishes, and the first Kaggle attempt was
        # sized 2-3x too long to ever get there.
        self.trail = trail
        # `ref`: the base's per-world values on THESE worlds. When present the
        # fitness becomes a per-world PAIRED difference, which is the whole
        # ballgame. Per-world sd is $34,749, so an absolute mean over 36 worlds
        # carries se $5,792 -- and picking the best of 8 such evaluations
        # inflates the winner by $8,218 through luck alone, against a $3,000
        # effect. Differencing against the base on the SAME world cancels the
        # "this world is easy/hard" term, which is nearly all of that spread.
        self.ref = ref

    def __call__(self, pt, per=False):
        path = bundle_for_multi(to_schedules(clamp(pt)), tag="bakeoff")
        tasks = [(p, path, 1 - side) for p, side in self.jobs]
        res = [r for r in self.pool.map(bench_job, tasks) if r]
        self.games += len(res)
        if not res:
            return (-1e18, 0.0, {}) if per else (-1e18, 0.0)
        vals = {(r["episode"], r["seat"]): r["us"] - r["script"] for r in res}
        win = sum(r["win"] for r in res) / len(res)
        if self.ref:
            d = [v - self.ref[k] for k, v in vals.items() if k in self.ref]
            mean = st.mean(d) if d else -1e18
        else:
            mean = st.mean(vals.values())
        if self.trail:
            with open(self.trail, "a") as f:
                f.write(json.dumps({"games": self.games, "score": mean,
                                    "win": win, "n": len(res),
                                    "pt": [round(v, 4) for v in clamp(pt)]}) + "\n")
        return (mean, win, vals) if per else (mean, win)


def paired(a, b):
    """mean and se of (a - b) over the worlds both evaluated."""
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    if len(d) < 2:
        return 0.0, 0.0, len(d)
    return st.mean(d), st.stdev(d) / len(d) ** 0.5, len(d)


# --- the four methods ---------------------------------------------------------

def m_random(fit, budget, rng, log):
    best = None
    while fit.games < budget:
        pt = clamp([rng.uniform(SPACE[n]["lo"], SPACE[n]["hi"]) for n, _ in DIMS])
        v = fit(pt)
        if best is None or v[0] > best[0][0]:
            best = (v, pt)
        log(f"random  games={fit.games:>5}  this={v[0]:>+9,.0f}  best={best[0][0]:>+9,.0f}")
    return best


def m_hill(fit, budget, rng, log, sigma=None):
    """(1+1)-ES with the base plan as the origin -- the closest thing to the
    greedy rollout that actually worked, but over all schedules jointly.

    Step size is measured, not guessed: a sigma=0.25 perturbation costs
    -$18,944 paired while sigma=0.10 stays inside the basin, so the first
    round of searches was stepping clean out of the optimum every time.
    """
    sigma = SIGMA[0] if sigma is None else sigma
    cur = base_point()
    cv = fit(cur)
    best = (cv, list(cur))
    while fit.games < budget:
        step = [rng.gauss(0, sigma * (SPACE[n]["hi"] - SPACE[n]["lo"])) for n, _ in DIMS]
        cand = clamp([a + b for a, b in zip(cur, step)])
        v = fit(cand)
        if v[0] > cv[0]:
            cur, cv = cand, v
            if v[0] > best[0][0]:
                best = (v, list(cand))
        log(f"hill    games={fit.games:>5}  this={v[0]:>+9,.0f}  cur={cv[0]:>+9,.0f}  "
            f"best={best[0][0]:>+9,.0f}")
    return best


def m_cem(fit, budget, rng, log, pop=8, elite=3):
    """Cross-entropy method: robust to a noisy objective because it only uses
    the RANK of the elite set, never the fitness values themselves."""
    mu = base_point()
    sd = [0.35 * (SPACE[n]["hi"] - SPACE[n]["lo"]) for n, _ in DIMS]
    best = None
    while fit.games < budget:
        cands = [clamp([rng.gauss(m, s) for m, s in zip(mu, sd)]) for _ in range(pop)]
        scored = []
        for c in cands:
            if fit.games >= budget:
                break
            scored.append((fit(c), c))
        if not scored:
            break
        scored.sort(key=lambda x: -x[0][0])
        if best is None or scored[0][0][0] > best[0][0]:
            best = scored[0]
        top = [c for _, c in scored[:elite]]
        mu = [st.mean(c[i] for c in top) for i in range(len(mu))]
        sd = [max(0.05 * (SPACE[n]["hi"] - SPACE[n]["lo"]),
                  st.pstdev([c[i] for c in top]))
              for i, (n, _) in enumerate(DIMS)]
        log(f"cem     games={fit.games:>5}  gen best={scored[0][0][0]:>+9,.0f}  "
            f"best={best[0][0]:>+9,.0f}")
    return best


def m_halving(fit, budget, rng, log, arms=24):
    """Successive halving: many arms on few worlds, survivors on more. Spends
    the budget where the ranking is still uncertain, which is the right shape
    when a single game's noise (~$30k) dwarfs the effect (~$3k)."""
    full = fit.jobs
    pool_arms = [clamp([rng.uniform(SPACE[n]["lo"], SPACE[n]["hi"]) for n, _ in DIMS])
                 for _ in range(arms)]
    pool_arms[0] = base_point()
    frac, best = 0.25, None
    while len(pool_arms) > 1 and fit.games < budget:
        fit.jobs = full[:max(6, int(len(full) * frac))]
        scored = []
        for a in pool_arms:
            if fit.games >= budget:
                break
            scored.append((fit(a), a))
        if not scored:
            break
        scored.sort(key=lambda x: -x[0][0])
        if best is None or scored[0][0][0] > best[0][0]:
            best = scored[0]
        log(f"halving games={fit.games:>5}  arms={len(scored):>3} @ "
            f"{len(fit.jobs):>3} worlds  best={best[0][0]:>+9,.0f}")
        pool_arms = [a for _, a in scored[:max(1, len(scored) // 2)]]
        frac = min(1.0, frac * 2)
    fit.jobs = full
    return best


def m_bo(fit, budget, rng, log, n_init=5):
    """Bayesian optimisation, GP + expected improvement.

    The strongest match in the table in docs/training_algorithms.md for <=20
    expensive, noisy dimensions, and the only family with no local evidence
    either way. A GP is the one method here that models the OBSERVATION NOISE
    explicitly (the `noise` term below) rather than treating each evaluation as
    the truth -- which matters when per-game spread is 12x the effect.

    Hand-rolled on numpy: `cma`, `sklearn` and `optuna` are not installed and
    a submission must stay dependency-free anyway.
    """
    import numpy as np

    lo = np.array([SPACE[n]["lo"] for n, _ in DIMS], float)
    hi = np.array([SPACE[n]["hi"] for n, _ in DIMS], float)
    span = np.maximum(hi - lo, 1e-9)
    unit = lambda p: (np.asarray(p, float) - lo) / span
    real = lambda u: lo + np.asarray(u, float) * span

    X, y = [], []
    seeds = [unit(base_point())] + [
        np.array([rng.random() for _ in DIMS]) for _ in range(n_init - 1)]
    for u in seeds:
        if fit.games >= budget:
            break
        v = fit(real(u))
        X.append(u); y.append(v[0])
        log(f"bo      games={fit.games:>5}  init={v[0]:>+9,.0f}")

    def kern(A, B, ls, amp):
        d = ((A[:, None, :] - B[None, :, :]) ** 2 / ls ** 2).sum(-1)
        return amp * np.exp(-0.5 * d)

    best = (max(zip(y, X)) if y else None)
    while fit.games < budget and X:
        Xa = np.array(X); ya = np.array(y)
        mu_y, sd_y = ya.mean(), max(ya.std(), 1.0)
        z = (ya - mu_y) / sd_y
        ls = np.full(len(DIMS), 0.4)
        # Noise is large here on purpose: an evaluation is ~36 games against a
        # ~$40k per-game spread, so the GP must not interpolate its own noise.
        noise = 0.6
        K = kern(Xa, Xa, ls, 1.0) + noise * np.eye(len(Xa))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            break
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        cand = np.array([[rng.random() for _ in DIMS] for _ in range(2048)])
        Ks = kern(cand, Xa, ls, 1.0)
        m = Ks @ alpha
        v = np.maximum(1.0 - (np.linalg.solve(L, Ks.T) ** 2).sum(0), 1e-9)
        s = np.sqrt(v)
        from math import erf, sqrt as _sq
        best_z = z.max()
        g = (m - best_z) / s
        Phi = np.array([0.5 * (1 + erf(t / _sq(2))) for t in g])
        phi = np.exp(-0.5 * g ** 2) / _sq(2 * np.pi)
        ei = (m - best_z) * Phi + s * phi
        u = cand[int(np.argmax(ei))]
        val = fit(real(u))
        X.append(u); y.append(val[0])
        if best is None or val[0] > best[0]:
            best = (val[0], u)
        log(f"bo      games={fit.games:>5}  this={val[0]:>+9,.0f}  "
            f"best={best[0]:>+9,.0f}")
    if best is None:
        return None
    bu = best[1]
    i = int(np.argmax(np.array(y)))
    return ((y[i], 0.0), list(real(np.array(X[i]))))


SIGMA = [0.10]

METHODS = {"random": m_random, "hill": m_hill, "cem": m_cem,
           "halving": m_halving, "bo": m_bo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=["all"] + sorted(METHODS))
    ap.add_argument("--budget", type=int, default=1200,
                    help="games per method")
    ap.add_argument("--per-bin", type=int, default=6,
                    help="ladder sessions per rating bin per evaluation")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", type=int, default=25,
                    help="per-bin sessions for the held-out re-measurement")
    ap.add_argument("--sigma", type=float, default=0.10,
                    help="(1+1)-ES step, as a fraction of each dimension's "
                         "range. Measured: 0.25 costs -$18,944 paired, 0.10 "
                         "stays in the basin.")
    ap.add_argument("--paired", action="store_true",
                    help="fitness = per-world difference from the base on the "
                         "same worlds (common random numbers). Without this "
                         "the winner's curse at 36 worlds is $8,218 against a "
                         "$3,000 effect and no method can work.")
    args = ap.parse_args()
    SIGMA[0] = args.sigma

    search = ladder_jobs(args.per_bin, heldout=False, seed=args.seed)
    held = ladder_jobs(args.final, heldout=True, seed=args.seed + 1,
                       exclude={p for p, _ in search})
    print(f"search worlds {len(search)}  held-out worlds {len(held)}  "
          f"budget {args.budget} games/method  procs {args.procs}\n")

    names = sorted(METHODS) if args.method == "all" else [args.method]
    out = {}
    runs = os.path.join(REPO, "rl", "bakeoff_runs")
    os.makedirs(runs, exist_ok=True)
    cache = os.path.join(runs, f"base_held_{args.final}_{args.seed}.json")

    with Pool(args.procs) as pool:
        # The base's held-out evaluation is identical for every method, so
        # cache it: 150 games saved per run, which is what lets a method fit
        # inside one foreground call.
        if os.path.exists(cache):
            c = json.load(open(cache))
            bm, bw = c["mean"], c["win"]
            bper = {tuple(json.loads(k)): v for k, v in c["per"].items()}
            print(f"BASE plan on held-out: {bm:>+9,.0f}/game   win {bw:.3f}   "
                  f"{len(bper)} worlds   (cached)\n")
        else:
            bm, bw, bper = Fitness(held, pool)(base_point(), per=True)
            json.dump({"mean": bm, "win": bw,
                       "per": {json.dumps(list(k)): v for k, v in bper.items()}},
                      open(cache, "w"))
            print(f"BASE plan on held-out: {bm:>+9,.0f}/game   win {bw:.3f}   "
                  f"{len(bper)} worlds\n")

        # The base on the SEARCH worlds, for the paired fitness. Cached too --
        # it is the same for every method.
        sref = None
        if args.paired:
            scache = os.path.join(runs, f"base_search_{args.per_bin}_{args.seed}.json")
            if os.path.exists(scache):
                sref = {tuple(json.loads(k)): v
                        for k, v in json.load(open(scache)).items()}
            else:
                _, _, sref = Fitness(search, pool)(base_point(), per=True)
                json.dump({json.dumps(list(k)): v for k, v in sref.items()},
                          open(scache, "w"))
            sd_abs = st.stdev(sref.values())
            print(f"PAIRED fitness on: base scored on the {len(sref)} search "
                  f"worlds (per-world sd ${sd_abs:,.0f})\n")

        for name in names:
            rng = random.Random(args.seed + hash(name) % 1000)
            fit = Fitness(list(search), pool, ref=sref,
                          trail=os.path.join(runs, f"progress_{name}.jsonl"))
            t = time.time()
            res = METHODS[name](fit, args.budget, rng,
                                lambda s: print("  " + s, flush=True))
            if not res:
                continue
            (sv, sw), pt = res
            hv, hw, hper = Fitness(held, pool)(pt, per=True)
            dm, dse, dn = paired(hper, bper)
            out[name] = dict(search=sv, held=hv, held_win=hw, pt=clamp(pt),
                             diff=dm, se=dse, n=dn, win_delta=hw - bw,
                             base=bm, base_win=bw,
                             games=fit.games, mins=(time.time() - t) / 60)
            # Persist immediately -- a killed batch must not lose a finished
            # method, which has now happened twice.
            json.dump(out[name], open(os.path.join(runs, f"{name}.json"), "w"),
                      indent=1)
            print(f"  -> {name}: search {sv:>+9,.0f}  HELD-OUT {hv:>+9,.0f} "
                  f"(win {hw:.3f})  paired vs base {dm:>+9,.0f} +/-{dse:,.0f}  "
                  f"{fit.games} games  {out[name]['mins']:.1f} min\n")

    print(f"\n{'method':<10} {'search $/game':>14} {'HELD-OUT $/game':>16} "
          f"{'held win':>9} {'PAIRED vs base':>16} {'se':>8} {'z':>6}")
    for n, r in sorted(out.items(), key=lambda kv: -kv[1]["diff"]):
        z = r["diff"] / r["se"] if r["se"] else 0.0
        print(f"{n:<10} {r['search']:>+14,.0f} {r['held']:>+16,.0f} "
              f"{r['held_win']:>9.3f} {r['diff']:>+16,.0f} {r['se']:>8,.0f} {z:>6.2f}")
    print(f"{'BASE':<10} {'':>14} {bm:>+16,.0f} {bw:>9.3f} {0:>+16,.0f}")
    print("\n  search $/game is optimistic -- the winner of a noisy search is "
          "biased upward.\n  PAIRED vs base, on identical held-out worlds, is "
          "the number to believe;\n  |z| < 2 means the method found nothing.")
    json.dump(out, open(os.path.join(REPO, "rl", "bakeoff.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
