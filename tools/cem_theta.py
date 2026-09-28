"""CEM over the strategy vector: search the champion's own constants.

    PYTHONPATH=. uv run python tools/cem_theta.py --gens 6 --pop 10 --games 144

WHAT THIS IS. `kagent/theta.py` makes the planner's constants a point in a
35-dimensional cube. This walks that cube with the cross-entropy method:
sample a population from N(mu, sigma^2), play each candidate against the SAME
held-out worlds the shipped gate uses, keep the best few, refit mu and sigma to
them, repeat. Nobody picks a lever; the gate does the picking.

WHY CEM AND NOT A GRADIENT OR A BANDIT. The objective is a win rate over a
discrete simulator -- no gradient exists, and every coordinate is coupled to
several others (a goose needs a coop, which needs ground, which the strawberry
target is holding, and an egg needs `animal_visit` to be collected at all), so
a per-coordinate bandit is searching the wrong space. That coupling is the
project's whole failure record: eleven single-lever arms, two of which won.
CEM is the cheapest thing that searches the JOINT.

THE MEASUREMENT IS THE EXPENSIVE PART AND IT IS NOISY.

  * PAIRING. Every candidate plays exactly the same worlds in the same seats
    against the same recorded opponents, and is scored on its PAIRED delta
    against the baseline's rows on those worlds. Memory: pairing on the same
    world raises the correlation to 0.95 and cuts the standard error threefold.
  * THE BASELINE IS PLAYED ONCE. Temperature-0 play is deterministic (B2's
    control replayed 785 games identically), so the baseline's rows are a
    constant of the world draw and are reused for the whole run.
  * OVERFITTING TO THE DRAW IS REAL. The search sees one world set for every
    generation, so a late winner has had many chances to fit it. A second,
    DISJOINT draw is kept back and never scored during the search; `--holdout`
    replays the final point on it. A gain that does not survive that is a gain
    to those worlds, not to the game.
  * A GENERATION IS NOT A SIGNIFICANCE TEST. At 144 games the paired standard
    error is about 4 points, and elite selection at that noise is partly
    luck. CEM tolerates it -- the elite MEAN is what moves, and an elite that
    got in on noise is one of several -- but no single generation's ranking
    should be read as a result. The run's output is the final point plus its
    holdout gate, not the leaderboard of any one generation.

Every candidate and its score is appended to `trials.jsonl` as it lands, so the
run is readable while it works and a crash costs one generation, not the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl.session_eval import EDGES, _label, _wr, held_out_entries, paired   # noqa: E402

IDX = os.path.join(REPO, "rl", "datasets", "session_index.json")


def _worker_init():
    """Every worker installs the fast harness.

    `tools/fast_env` removes the kaggle_environments deepcopy/structify/
    jsonschema plumbing, which profiling put at two thirds of a game, and is
    bit-identical -- it asserts equal final rewards on every seed
    (`uv run python tools/fast_env.py --verify N`). It must be installed INSIDE
    the worker: multiprocessing starts `spawn` here, so a patch applied in the
    parent is not in the child. 1.7x on a four-hour search is ninety minutes.
    """
    import tools.fast_env as fast_env
    fast_env.install()


# ----------------------------------------------------------------- playing
def _jobs_for(spec, entries):
    return [{"weights_path": None, "learner_bundle": None, "learner_policy": spec,
             "temperature": 0.0, "seed": 0, "side": 1 - ss,
             "opponent": ("session", os.path.join(REPO, p), ss), "mode": "rl"}
            for p, ss, _b, _r in entries]


def _rows(out, entries):
    return [{"win": ep["win"], "money": ep["final_money"], "opp": ep["opp_money"],
             "bin": b, "rating": r, "path": p, "script_side": ss,
             "bad": any(s != "DONE" for s in ep["statuses"])}
            for ep, (p, ss, b, r) in zip(out, entries)]


def theta_key(theta):
    """Stable hash of a vector, so rounding collisions are played once.

    from_unit rounds onto each coordinate's own type, so as sigma shrinks many
    distinct cube points name the SAME strategy. Without this the last
    generations would spend most of their budget replaying one candidate.
    """
    blob = json.dumps(theta, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def score(base_rows, rows):
    """-> dict. The objective is the paired win-rate delta; everything else is
    reported so a win can be read for its SHAPE (memory: the only two arms that
    ever shipped both raised our bank AND lowered the opponent's)."""
    pr = paired(base_rows, rows)
    wr, se = _wr(rows)
    return {"delta": pr["delta"], "delta_se": pr["se"], "p": pr["p_mcnemar"],
            "arm_only": pr["b_arm_only"], "base_only": pr["c_base_only"],
            "win": wr, "win_se": se,
            "money": sum(r["money"] for r in rows) / len(rows),
            "opp": sum(r["opp"] for r in rows) / len(rows),
            "margin": sum(r["money"] - r["opp"] for r in rows) / len(rows),
            "bad": sum(r["bad"] for r in rows),
            "bands": [paired([x for x in base_rows if x["bin"] == b],
                             [x for x in rows if x["bin"] == b])["delta"]
                      for b in range(len(EDGES))]}


# --------------------------------------------------------------------- CEM
def refit(elites, sigma0, shrink):
    """mu = the elite mean; sigma stays on its schedule.

    Under subset sampling most elites carry the CURRENT mean in most
    coordinates -- only the handful each one perturbed differ -- so the elite
    standard deviation measures how often a coordinate happened to be selected,
    not how uncertain its value is. Refitting sigma from it would collapse
    every coordinate nobody touched. So sigma follows the schedule and only mu
    is fitted, which is the update that carries the information: a coordinate
    moves a fraction of the way toward whatever perturbation got into the
    elites, and stays put when none did.
    """
    n = len(elites)
    dim = len(elites[0])
    mu = [sum(e[i] for e in elites) / n for i in range(dim)]
    return mu, [sigma0 * shrink] * dim


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="how far a candidate must beat the incumbent centre "
                         "to become an elite and move the mean")
    ap.add_argument("--games", type=int, default=144,
                    help="games per candidate; drawn per rating band")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="CEM sampling seed")
    ap.add_argument("--draw-seed", type=int, default=0, help="world-draw seed")
    ap.add_argument("--holdout-seed", type=int, default=101,
                    help="the DISJOINT draw the final point is re-gated on")
    ap.add_argument("--sigma0", type=float, default=0.25,
                    help="spread of a perturbation, in the unit cube")
    ap.add_argument("--block", type=int, default=4,
                    help="most coordinates one candidate may perturb; the rest "
                         "sit at the current mean")
    ap.add_argument("--init", default=None, help="starting theta JSON (default: champion)")
    ap.add_argument("--out", default=os.path.join(REPO, "rl/runs/cem_theta"))
    ap.add_argument("--holdout", action="store_true",
                    help="after the search, re-gate the best point on the held-back draw")
    ap.add_argument("--no-fast", action="store_true",
                    help="do not install tools/fast_env in the workers")
    ap.add_argument("--shipped", action="store_true",
                    help="also play the SHIPPED champ-h11 (conditional knob bandit) "
                         "so the offset between it and the theta baseline is known")
    args = ap.parse_args()

    from kagent import theta as T
    from rl import rollout                                              # noqa: F401

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.jsonl")
    per_bin = max(1, args.games // len(EDGES))
    entries = held_out_entries(IDX, args.draw_seed, per_bin)

    start = T.champion()
    if args.init:
        with open(args.init) as fh:
            d = json.load(fh)
        start.update(d.get("knobs", d))
        for n in T.EXEC_NAMES:
            if n in (d.get("exec") or []):
                start[n] = True

    print(f"CEM over {T.DIM} coordinates, up to {args.block} perturbed a candidate; "
          f"{len(entries)} paired games a candidate, "
          f"{args.pop} a generation, {args.gens} generations "
          f"= {args.pop * args.gens * len(entries):,} games")
    print(f"start: {T.describe(start)}")
    print(f"out:   {args.out}\n")

    seen = {}          # theta_key -> (theta, stats)
    t_run = time.time()

    with Pool(args.procs, initializer=None if args.no_fast else _worker_init) as pool:
        # --- the baseline, played once -----------------------------------
        t0 = time.time()
        base_rows = _rows(list(pool.imap(rollout.run_episode,
                                         _jobs_for(T.spec(start), entries))), entries)
        wr, se = _wr(base_rows)
        per_game = (time.time() - t0) / len(entries) * args.procs
        print(f"baseline (theta @ start)  win {wr:.3f} +/- {se:.3f}  "
              f"money {sum(r['money'] for r in base_rows)/len(base_rows):>9,.0f}   "
              f"({time.time()-t0:.0f}s, {per_game:.2f}s a game a core)")
        print(f"budget: ~{args.pop * args.gens * len(entries) * per_game / args.procs / 60:.0f} min\n")

        if args.shipped:
            from kagent.weights import BANDIT_RULES, EXEC_RULES
            sspec = {"kind": "knobs", "rules": {**BANDIT_RULES, "exec": EXEC_RULES}}
            srows = _rows(list(pool.imap(rollout.run_episode,
                                         _jobs_for(sspec, entries))), entries)
            st = score(base_rows, srows)
            print(f"shipped champ-h11         win {st['win']:.3f}   "
                  f"delta vs theta baseline {st['delta']:+.3f} +/- {st['delta_se']:.3f}\n")

        # --- the search ---------------------------------------------------
        rng = random.Random(args.seed)
        mu = T.to_unit(start)
        sigma = [args.sigma0] * T.DIM
        best_key = None

        for gen in range(args.gens):
            shrink = 1.0 - 0.6 * (gen / max(1, args.gens - 1))
            # The INCUMBENT is always a candidate: the point the mean
            # currently names, re-scored every generation. It anchors the
            # update (an elite set of pure perturbations would let the mean
            # drift on coordinates nobody actually improved) and it gives a
            # per-generation read on whether the centre itself is moving.
            centre_th = T.from_unit(mu)
            cand = [(T.to_unit(centre_th), centre_th, theta_key(centre_th))]
            tried = {cand[0][2]}
            guard = 0
            while len(cand) < args.pop and guard < args.pop * 40:
                guard += 1
                u = T.sample_block(rng, mu, sigma, args.block)
                th = T.from_unit(u)
                k = theta_key(th)
                if k in tried:
                    continue                 # duplicate WITHIN the generation
                tried.add(k)
                cand.append((T.to_unit(th), th, k))

            fresh = [(u, th, k) for u, th, k in cand if k not in seen]
            jobs, spans = [], []
            for _u, th, _k in fresh:
                spans.append((len(jobs), len(jobs) + len(entries)))
                jobs += _jobs_for(T.spec(th), entries)
            t0 = time.time()
            out = list(pool.imap(rollout.run_episode, jobs)) if jobs else []
            for (lo, hi), (_u, th, k) in zip(spans, fresh):
                st = score(base_rows, _rows(out[lo:hi], entries))
                seen[k] = (th, st)
                with open(trials_path, "a") as fh:
                    fh.write(json.dumps({"gen": gen, "key": k, "theta": th,
                                         "score": st,
                                         "diff": T.describe(th, start)}) + "\n")

            # ELITES MUST BEAT THE INCUMBENT BY A MARGIN. At 144 games the
            # paired standard error is about 0.045, so a candidate that
            # changed nothing of consequence still lands above the centre half
            # the time. Plain top-k would then move the mean every generation
            # on five coin flips, and after eight generations the vector has
            # drifted on coordinates nothing improved. Requiring `min_gain`
            # over the centre makes a neutral candidate elite about a third of
            # the time and a real one most of the time -- and when nothing
            # clears it, the elite set is the centre alone and the mean stays
            # exactly where it is. A search that declines to move is the right
            # answer when the champion is the best point on offer.
            centre_key = cand[0][2]
            centre_d = seen[centre_key][1]["delta"]
            scored = sorted(((seen[k][1]["delta"], seen[k][1]["margin"], k, u)
                             for u, _th, k in cand), reverse=True)
            winners = [x for x in scored
                       if x[2] != centre_key and x[0] >= centre_d + args.min_gain]
            elites = ([T.to_unit(seen[centre_key][0])]
                      + [u for _d, _m, _k, u in winners[:max(0, args.elite - 1)]])
            mu, sigma = refit(elites, args.sigma0, shrink)

            prev_best = seen[best_key][1]["delta"] if best_key else None
            best_key = max(seen, key=lambda k: (seen[k][1]["delta"], seen[k][1]["margin"]))
            print(f"gen {gen}  ({time.time()-t0:.0f}s, {len(fresh)} played, "
                  f"{len(cand)-len(fresh)} cached)   centre {centre_d:+.3f}   "
                  f"{len(winners)} of {len(cand)-1} cleared +{args.min_gain:.2f}   "
                  f"sigma {sum(sigma)/len(sigma):.3f}")
            for d, _m, k, _u in scored[:args.elite]:
                st = seen[k][1]
                mark = "*" if (prev_best is None or d > prev_best) else " "
                print(f"  {mark}{d:+.3f} +/-{st['delta_se']:.3f}  win {st['win']:.3f}  "
                      f"money {st['money']:>8,.0f}  opp {st['opp']:>8,.0f}  "
                      f"{T.describe(seen[k][0], start)[:96]}")
            # Checkpoint the distribution, not just the trials: a crash at
            # generation 6 of 8 otherwise costs the whole search, because mu
            # and sigma live only in this loop. --init takes the best vector
            # back in; state.json says where the distribution had got to.
            with open(os.path.join(args.out, "state.json"), "w") as fh:
                json.dump({"gen": gen, "mu": mu, "sigma": sigma,
                           "names": T.NAMES, "centre": T.from_unit(mu),
                           "best": seen[best_key][0],
                           "best_score": seen[best_key][1],
                           "elapsed_min": round((time.time() - t_run) / 60, 1)},
                          fh, indent=1)
            print(f"  best so far {seen[best_key][1]['delta']:+.3f}   "
                  f"[{(time.time()-t_run)/60:.0f} min]\n", flush=True)

        # --- the point the search landed on --------------------------------
        centre = T.from_unit(mu)
        ck = theta_key(centre)
        if ck not in seen:
            crows = _rows(list(pool.imap(rollout.run_episode,
                                         _jobs_for(T.spec(centre), entries))), entries)
            seen[ck] = (centre, score(base_rows, crows))
        cst = seen[ck][1]
        print(f"CEM centre  {cst['delta']:+.3f} +/-{cst['delta_se']:.3f}   "
              f"{T.describe(centre, start)}")
        # The CEM centre is the search's actual answer -- an average of elites,
        # so it carries less of whatever noise got one candidate elected. The
        # single best trial is taken only when it clearly beats the centre.
        bth, bst = seen[best_key]
        final, fst = ((centre, cst) if cst["delta"] >= bst["delta"] else (bth, bst))

        print(f"\nBEST      delta {fst['delta']:+.3f} +/- {fst['delta_se']:.3f}  "
              f"p {fst['p']:.3f}  win {fst['win']:.3f}  "
              f"money {fst['money']:,.0f}  opp {fst['opp']:,.0f}")
        print(f"bands     " + " ".join(f"{b:+.3f}" for b in fst["bands"]))
        print(f"vector    {T.describe(final, start)}")

        result = {"start": start, "best": final, "score": fst,
                  "draw_seed": args.draw_seed, "games": len(entries),
                  "gens": args.gens, "pop": args.pop, "trials": len(seen)}

        # --- the holdout: a draw the search has never been scored on --------
        if args.holdout:
            hentries = held_out_entries(IDX, args.holdout_seed, per_bin)
            overlap = len({(p, s) for p, s, _b, _r in entries}
                          & {(p, s) for p, s, _b, _r in hentries})
            print(f"\nHOLDOUT   {len(hentries)} games, seed {args.holdout_seed}, "
                  f"{overlap} shared with the search draw")
            hb = _rows(list(pool.imap(rollout.run_episode,
                                      _jobs_for(T.spec(start), hentries))), hentries)
            hf = _rows(list(pool.imap(rollout.run_episode,
                                      _jobs_for(T.spec(final), hentries))), hentries)
            hs = score(hb, hf)
            print(f"  baseline win {_wr(hb)[0]:.3f}   best win {hs['win']:.3f}")
            print(f"  delta {hs['delta']:+.3f} +/- {hs['delta_se']:.3f}  "
                  f"z {hs['delta']/hs['delta_se'] if hs['delta_se'] else float('nan'):.2f}  "
                  f"p {hs['p']:.3f}   arm-only {hs['arm_only']} base-only {hs['base_only']}")
            print(f"  bands " + " ".join(f"{b:+.3f}" for b in hs["bands"]))
            print(f"  money {hs['money']:,.0f} vs {_wr(hb)[0] and sum(r['money'] for r in hb)/len(hb):,.0f}"
                  f"   opp {hs['opp']:,.0f} vs {sum(r['opp'] for r in hb)/len(hb):,.0f}")
            from kagent.weights import BANDIT_RULES as _BR, EXEC_RULES as _ER
            hship = _rows(list(pool.imap(rollout.run_episode,
                                         _jobs_for({"kind": "knobs",
                                                    "rules": {**_BR, "exec": _ER}},
                                                   hentries))), hentries)
            vs = score(hship, hf)
            print(f"\n  vs SHIPPED champ-h11 on the same holdout draw:")
            print(f"  shipped win {_wr(hship)[0]:.3f}   best win {vs['win']:.3f}")
            print(f"  delta {vs['delta']:+.3f} +/- {vs['delta_se']:.3f}  "
                  f"z {vs['delta']/vs['delta_se'] if vs['delta_se'] else float('nan'):.2f}  "
                  f"p {vs['p']:.3f}   arm-only {vs['arm_only']} base-only {vs['base_only']}")
            print(f"  bands " + " ".join(f"{b:+.3f}" for b in vs["bands"]))
            result["holdout"] = {"seed": args.holdout_seed, "overlap": overlap,
                                 "vs_theta_baseline": hs, "vs_shipped": vs}
            if hs["delta"] <= 0 or vs["delta"] <= 0:
                print("  -> the search gain did NOT survive a fresh draw: it fit the "
                      "search worlds. Do not ship this point.")

    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    with open(os.path.join(args.out, "best_theta.json"), "w") as fh:
        json.dump(T.spec(final), fh, indent=1)
    print(f"\n-> {args.out}/best_theta.json   (play it: "
          f"--arm best=dev:theta:{args.out}/best_theta.json)")
    print(f"-> {trials_path}   ({len(seen)} distinct vectors)")


if __name__ == "__main__":
    main()
