"""Which metric PROPAGATES at each phase -- i.e. carries forward to the bell.

THE PROBLEM. We have 48 measured quantities across three phases and no idea
which of them matter WHEN. We know exactly one data point: at day 10 the
relative money lead correlates r = -0.021 with the outcome while INVESTMENT
correlates +0.304. That single fact reshaped the opening. This does it for
every metric at every day.

DEFINITION. A metric propagates at day d if its value at day d predicts the
FINAL result. Measured two ways, because the project has both:

    r_margin   Pearson r with (final_0 - final_1)
    r_win      point-biserial with the binary win

They are not the same question and the record is explicit that rating is
win-based while every executor lever moves the size of a loss. Reporting
only one would hide exactly that gap.

WHY THIS IS SAFE TO USE, given observational/interventional. Correlation
with outcome is NOT evidence that forcing the metric up causes a win -- that
law has cost this project a dozen arms. But the intended CONSUMER here is a
shaping potential, and Ng, Harada & Russell (1999) prove potential-based
shaping F(s,s') = g*PHI(s') - PHI(s) leaves the optimal policy unchanged for
ANY PHI. So a merely-correlational PHI cannot corrupt the optimum; it can
only change how fast the value head learns. That is the one consumer for
which observational evidence is the right kind of evidence.

    PYTHONPATH=. uv run python tools/propagation.py --n 300
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT = os.environ.get("PROP_OUT",
                     os.path.join(REPO, "rl/datasets/propagation.npz"))
TPD = 24


def job(spec):
    """Play one game, sampling every metric each day.

    `perturb` randomises the OPENING. Without it this measurement cannot say
    anything about the early phase: champ-h8 self-play runs the same opening
    every game, so at day 2 only 4 of 48 features have any cross-game spread
    at all (mean sd 0.0019) and "nothing propagates early" is a statement
    about the agent, not the game.

    Randomising our own opening also upgrades the early answer from
    OBSERVATIONAL to INTERVENTIONAL: the metric differences are caused by an
    exogenous coin flip, so a correlation with the outcome is a causal
    effect, not a confound. That is the distinction that has cost this
    project a dozen arms.
    """
    seed, agent, opp, perturb = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    import tools.early_metrics as em
    import tools.phase_metrics as pm

    def load(nm):
        g = {}
        exec(open(f"{REPO}/arena/agents/{nm}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load(agent), load(opp)
    from tools.action_space import legal_ops, legal_qty, to_action, OPS, QTY
    rng = np.random.default_rng(seed)
    # which turns in days 0-8 get a random legal farmer action instead
    hit = set(rng.choice(np.arange(0, 9 * TPD), size=perturb, replace=False)
              .tolist()) if perturb else set()
    e = make("kaggriculture", configuration={"seed": seed}, debug=False)
    e.reset(2)
    rows, days = [], []
    while not e.done:
        o = e.state[0].observation
        # HOUR 12. At dawn hands and bags read 0 because units respawn empty
        # at the shed -- sampling at hour 0 is how an earlier pass concluded
        # we never hire.
        if o["step"] % TPD == 12:
            pv = o.private
            rows.append(np.concatenate([em.features(o, 0, pv),
                                        pm.mid_features(o, 0, pv),
                                        pm.end_features(o, 0, pv)]))
            days.append(o["step"] // TPD)
        act = a0(o)
        if o["step"] in hit:
            pv = o.private
            m = legal_ops(o, 0, 0, pv)
            cand = [i for i, ok in enumerate(m) if ok]
            if cand:
                opn = OPS[int(rng.choice(cand))]
                qm = legal_qty(o, 0, 0, opn, pv)
                ql = [QTY[i] for i, ok in enumerate(qm) if ok] or [1]
                act = dict(act)
                act["farmer"] = to_action(opn, int(rng.choice(ql)))
        e.step([act, a1(e.state[1].observation)])
    fin0, fin1 = e.state[0].reward, e.state[1].reward
    return (np.stack(rows).astype("float32"), np.asarray(days, "int16"),
            float(fin0 - fin1), float(fin0 > fin1))


def names():
    import tools.early_metrics as em
    import tools.phase_metrics as pm
    return (["E:" + n for n in em.NAMES]
            + ["M:" + n for n in pm.MID_NAMES]
            + ["D:" + n for n in pm.END_NAMES])


def build(args):
    specs = [(args.base_seed + i, args.agent, args.opp, args.perturb)
             for i in range(args.n)]
    print(f"{args.n} games, sampling {len(names())} metrics every day "
          f"at hour 12 ...", flush=True)
    X, D, M, W = [], [], [], []
    with mp.Pool(args.procs, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(job, specs)):
            if not r:
                continue
            x, d, m, w = r
            X.append(x); D.append(d)
            M.append(np.full(len(d), m, "float32"))
            W.append(np.full(len(d), w, "float32"))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)
    np.savez_compressed(OUT, X=np.concatenate(X), D=np.concatenate(D),
                        M=np.concatenate(M), W=np.concatenate(W),
                        names=np.array(names()))
    print(f"-> {OUT}  {np.concatenate(X).shape}")


def _r(x, y):
    """Pearson r, with zero-variance handled as NO ANSWER rather than 0.

    A metric that is constant at a fixed day (season_left is a function of
    the day; occ_q3 if the quadrant is never bought) has undefined
    correlation. Returning nan and FILTERING it is the honest move: an
    earlier version returned it into a `sorted(reverse=True)`, where nan
    comparisons are undefined, and constant features floated to the top of
    the ranking as though they were the best predictors.
    """
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else float("nan")


def report(args):
    d = np.load(OUT, allow_pickle=True)
    X, D, M, W, NM = d["X"], d["D"], d["M"], d["W"], list(d["names"])
    ng = len(np.unique(M))
    print(f"\n{len(X):,} samples, ~{ng} games, {X.shape[1]} metrics\n")

    PHASES = [("EARLY", 2, 9), ("MID", 9, 20), ("END", 20, 30)]
    prop = {}
    for pname, lo, hi in PHASES:
        m = (D >= lo) & (D < hi)
        rows = []
        for i, nm in enumerate(NM):
            rm, rw = _r(X[m, i], M[m]), _r(X[m, i], W[m])
            if not np.isfinite(rm):
                continue          # constant here: no answer, not a zero
            rows.append((abs(rm), rm, rw if np.isfinite(rw) else 0.0, nm))
        rows.sort(reverse=True)
        prop[pname] = rows
        n = int(m.sum())
        se = 1.0 / np.sqrt(max(n, 2))
        print(f"{'=' * 66}")
        print(f"  {pname}  (days {lo}-{hi - 1}, n={n:,}, se~{se:.3f})")
        print(f"{'=' * 66}")
        print(f"  {'metric':<28}{'r margin':>11}{'r win':>9}{'':>4}")
        for a, rm, rw, nm in rows[:10]:
            flag = "  <-- propagates" if abs(rm) > 6 * se else ""
            print(f"  {nm:<28}{rm:>11.3f}{rw:>9.3f}{flag}")
        print(f"  ...")
        for a, rm, rw, nm in rows[-3:]:
            print(f"  {nm:<28}{rm:>11.3f}{rw:>9.3f}")
        print()

    # the headline the project already has one data point for
    print(f"{'=' * 66}")
    print(f"  DOES THE KNOWN FACT REPRODUCE?")
    print(f"{'=' * 66}")
    for day in (5, 10, 15, 20, 25):
        m = D == day
        if m.sum() < 20:
            continue
        inv = _r(X[m, NM.index("E:invested")], M[m])
        mon = _r(X[m, NM.index("E:money_lead")], M[m])
        print(f"  day {day:>2}   invested r={inv:+.3f}    "
              f"money_lead r={mon:+.3f}")
    print(f"\n  recorded: at day 10 money_lead r=-0.021, invested r=+0.304")

    # which metric OVERTAKES which, day by day -- the propagation story
    print(f"\n{'=' * 66}")
    print(f"  TOP PROPAGATOR BY DAY (|r| with final margin)")
    print(f"{'=' * 66}")
    print(f"  {'day':>4}  {'best metric':<28}{'r':>8}{'   runner-up':<28}{'r':>8}")
    for day in range(2, 30, 2):
        m = D == day
        if m.sum() < 20:
            continue
        rs = sorted(((abs(_r(X[m, i], M[m])), _r(X[m, i], M[m]), nm)
                     for i, nm in enumerate(NM)
                     if np.isfinite(_r(X[m, i], M[m]))), reverse=True)
        if len(rs) < 2:
            continue
        print(f"  {day:>4}  {rs[0][2]:<28}{rs[0][1]:>8.3f}"
              f"   {rs[1][2]:<25}{rs[1][1]:>8.3f}")

    np.savez_compressed(os.path.join(REPO, "rl/datasets/propagation_rank.npz"),
                        **{k: np.array([(a, rm, rw, nm) for a, rm, rw, nm in v],
                                       dtype=object) for k, v in prop.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=810000)
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--perturb", type=int, default=0,
                    help="random legal farmer actions injected into days 0-8, "
                         "to create early variation to correlate against")
    args = ap.parse_args()
    (build if args.cmd == "build" else report)(args)


if __name__ == "__main__":
    main()
