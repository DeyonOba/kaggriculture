"""Derive the override threshold from the value head, instead of picking it.

In residual/gated mode the policy may override the heuristic. The obvious
knob is "override when confident above tau" -- and a picked tau is exactly
the kind of constant that failed five times this week.

The principled form drops tau as a free parameter. We have the EXACT
simulator, so the question is answerable by one-ply rollout rather than by a
confidence proxy:

    override iff   V(s_after_policy) - V(s_after_heuristic)  >  eps

where eps is the value head's OWN measured error, not a number. Never
override on a difference smaller than your own measurement noise.

But that is downstream of a prior question this project already has a bad
answer to. `tools/value_head.py` is 4.8x tighter than money as a statistic
and was WRONG ON SIGN for 3 of 4 arms, because it was fit on the recorded
corpus and learned what co-occurs with winning. If V cannot ORDER two
actions better than chance, no eps rescues it.

So this measures, on real branch points:

  1. SIGN ACCURACY -- does sign(V(a) - V(b)) match sign(outcome(a) - outcome(b))?
     Chance is 50%. Below that, residual mode is dead however tau is set.
  2. THE CURVE -- sign accuracy as a function of |V(a) - V(b)|.
     eps is read off it: the smallest gap where accuracy clears the target.

Both branches are played to the end from the SAME state with the SAME
downstream policy and the same seed, so the difference is attributable.

    PYTHONPATH=. uv run python tools/tau_derive.py --n 200 --target 0.60
"""
import argparse
import collections
import copy
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def job(spec):
    """One branch point: play action A and action B to the end, same world."""
    seed, day_at, agent, opp, head = spec
    sys.path.insert(0, REPO)
    import tools.fast_env as fast_env
    fast_env.install()
    from kaggle_environments import make
    from tools.action_space import legal_ops, to_action, OPS

    def load(n):
        g = {}
        exec(open(f"{REPO}/arena/agents/{n}/main.py").read(), g)
        return g["agent"]
    a0, a1 = load(agent), load(opp)

    # The two heads differ ONLY in what they can see. Same branch points, same
    # seeds, same downstream policy -- so blindness and sign accuracy are
    # directly comparable between them, which is the whole point of the refit.
    if head == "spatial":
        import tools.value_spatial as vs
        model = vs.load()

        import tools.board_tensor as bt

        def V(e):
            o = e.state[0].observation
            return vs.value(model, o, 0, o.private)

        def REP(e):
            o = e.state[0].observation
            return np.concatenate([
                bt.board_pair_units(o, 0, private=o.private).ravel(),
                bt.unit_state(o, 0, 0, o.private)])
    else:
        import tools.value_head as vh
        model = vh._load_model()

        def V(e):
            o = e.state[0].observation
            return vh.value(model, o, 0, o["step"] // 24)

        def REP(e):
            return np.asarray(vh.features(e.state[0].observation, 0), "float64")

    # run to the branch point
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    while not env.done and env.state[0].observation["step"] < day_at:
        o = env.state[0].observation
        env.step([a0(o), a1(env.state[1].observation)])
    if env.done:
        return None
    o = env.state[0].observation
    base_act = a0(o)
    priv = env.state[0].observation.private

    # alternative: a DIFFERENT legal op for the farmer
    m = legal_ops(o, 0, 0, priv)
    cur = (base_act.get("farmer") or ["PASS"])[0]
    alts = [i for i, ok in enumerate(m)
            if ok and OPS[i].split(":")[0] != cur]
    if not alts:
        return None
    rng = np.random.default_rng(seed + day_at)
    alt_i = int(rng.choice(alts))
    alt_act = dict(base_act)
    alt_act["farmer"] = to_action(OPS[alt_i])

    out = []
    for tag, act in (("base", base_act), ("alt", alt_act)):
        e2 = make("kaggriculture", configuration={"seed": seed}, debug=False)
        e2.reset(2)
        # replay to the branch, then force the chosen first action
        while e2.state[0].observation["step"] < day_at:
            oo = e2.state[0].observation
            e2.step([a0(oo), a1(e2.state[1].observation)])
        oo = e2.state[0].observation
        e2.step([act, a1(e2.state[1].observation)])
        v, rep = V(e2), REP(e2)
        while not e2.done:
            oo = e2.state[0].observation
            e2.step([a0(oo), a1(e2.state[1].observation)])
        margin = e2.state[0].reward - e2.state[1].reward
        out.append((v, margin, rep))
    (va, ma, ra), (vb, mb, rb) = out
    return {"day": day_at // 24, "v_diff": va - vb, "m_diff": ma - mb,
            "v_base": va, "op": OPS[alt_i],
            # the honest blindness test: did the head's INPUT differ at all?
            "in_same": bool(np.allclose(ra, rb))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="champ-h8")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.60,
                    help="sign accuracy eps must clear")
    ap.add_argument("--head", default="agg", choices=["agg", "spatial"],
                    help="agg = day-level value_head; spatial = the refit")
    args = ap.parse_args()

    rng = np.random.default_rng(5)
    specs = []
    for i in range(args.n):
        seed = 400_000 + i
        step = int(rng.integers(2 * 24, 26 * 24))
        specs.append((seed, step, args.agent, args.opp, args.head))
    with mp.Pool(args.procs, maxtasksperchild=10) as pool:
        rows = [r for r in pool.map(job, specs) if r]

    if not rows:
        print("no usable branch points")
        return
    vd = np.array([r["v_diff"] for r in rows])
    md = np.array([r["m_diff"] for r in rows])
    ok = np.abs(md) > 1e-9
    vd, md = vd[ok], md[ok]
    n = len(vd)
    # A TIE is not a wrong answer, it is NO answer -- and np.sign(0) == 0, so
    # counting ties as disagreements caps accuracy at (1 - tie rate) by
    # construction. V's features are day-level aggregates (money, crop and
    # herd counts, shed, prices) with no yield_units, no watered_today and no
    # unit inventory, so a single WATER / HARVEST / MOVE / FEED / CARE changes
    # NOTHING it can see. Blindness, not inversion, and they look identical if
    # ties are scored wrong.
    blind = np.array([r["in_same"] for r in rows])[ok]
    seen = ~blind
    agree = (np.sign(vd) == np.sign(md))
    # a head whose INPUT differs can still emit a tie; count that separately
    # rather than folding it into either bucket.
    out_tie = (np.abs(vd) < 1e-9) & seen

    print(f"\nhead = {args.head}   "
          f"({'spatial trunk, 181ch' if args.head == 'spatial' else 'day-level aggregates'})")
    print(f"\n{n} branch points where the outcome actually differed\n")
    print(f"  0. CAN V SEE THE ACTION AT ALL?\n")
    print(f"     INPUT identical after both actions: {blind.sum()}/{n}"
          f" = {100*blind.mean():.0f}%   <- V is BLIND to these,")
    print(f"       {'':<36}no threshold can rescue them")
    print(f"     input differs:                     {seen.sum()}/{n}"
          f"   (of which output still ties: {out_tie.sum()})\n")
    print(f"  1. WHERE V CAN SEE, DOES IT ORDER CORRECTLY?\n")
    if seen.sum() < 5:
        print("     too few non-tied branches to say anything")
        return
    agree = agree[seen]
    vd, md = vd[seen], md[seen]
    n = len(vd)
    p = agree.mean()
    se = np.sqrt(p * (1 - p) / max(n, 1))
    print(f"     sign accuracy {p:.3f} +/- {se:.3f}   "
          f"(chance 0.500, z = {(p-0.5)/max(se,1e-9):+.2f})")
    if p - 2 * se <= 0.5:
        print(f"     -> NOT above chance. Residual mode is dead at any tau,")
        print(f"        and this reproduces the recorded 'wrong sign on 3 of 4'.")

    print(f"\n  2. THE CURVE -- accuracy by |V(a) - V(b)|\n")
    print(f"     {'|v diff| >=':>14}{'n':>7}{'sign acc':>11}{'mean |margin|':>15}")
    eps = None
    for t in (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        m = np.abs(vd) >= t
        if m.sum() < 15:
            continue
        a = agree[m].mean()
        print(f"     {t:>14.2f}{m.sum():>7}{a:>11.3f}"
              f"{np.abs(md[m]).mean():>15,.0f}")
        if eps is None and a >= args.target:
            eps = t
    print()
    # by DAY. The suspected mechanism: V carries money as a strong feature,
    # and the relative money lead predicts NOTHING before ~day 16 (r = -0.021
    # at day 10) while investment does (+0.304). A head fit on co-occurrence
    # therefore penalises SPENDING exactly when spending is correct.
    print(f"  3. BY DAY -- is it worst early, where money is anti-predictive?\n")
    print(f"     {'days':>10}{'n':>7}{'sign acc':>11}")
    days = np.array([r["day"] for r in rows])[ok][seen]
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 30)):
        m = (days >= lo) & (days < hi)
        if m.sum() < 8:
            continue
        print(f"     {str(lo)+'-'+str(hi):>10}{m.sum():>7}{agree[m].mean():>11.3f}")
    print()
    if p + 2 * se < 0.5:
        print(f"  eps: UNDEFINED. Overall sign accuracy is {p:.3f}, significantly"
              f"\n       BELOW chance, so no threshold defines a safe override --"
              f"\n       a subset clearing the target is selection on noise."
              f"\n       Residual and gated modes are unavailable with this head.")
    elif eps is None:
        print(f"  eps: NONE -- accuracy never reaches {args.target:.2f}. The head"
              f"\n       cannot support an override rule; use pure or heuristic,"
              f"\n       and retrain V on SEARCH outcomes before revisiting.")
    else:
        print(f"  eps = {eps:.2f}   derived, not picked: the smallest value gap"
              f"\n        at which this head's sign accuracy clears "
              f"{args.target:.2f}.")
    print(f"\n  NOTE: BOTH heads are fit on the recorded corpus, so both learned"
          f"\n  co-occurrence. --head spatial changes the ENCODER only. The"
          f"\n  label fix (search outcomes) is a separate, later change, and"
          f"\n  sign accuracy is the statistic it would move.")


if __name__ == "__main__":
    main()
