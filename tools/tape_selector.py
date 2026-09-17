"""Fit and honestly evaluate a day-6 tape selector over the cluster panel.

The cluster's 58 tapes share days 0-5 byte for byte, so step 144 is a real
decision point with 58 arms. The panel records every arm's outcome on every
world plus the day-6 context, and the prize is large: the per-world ORACLE is
worth about +$27k a game against roughly +$6k for the best fixed tape.

What shipped (tape-sel) uses ONE hand-picked binary key -- a yarn store among
the first two shops. The obvious next step, a finer key, has already failed
once: keying on the exact day-6 shop pair scored only +$1,314 held out, because
34 pairs over 40 worlds means picking the argmax of 58 arms from about one
observation per cell. That is noise, not signal.

The fix is not a coarser key, it is SHRINKAGE. Tape quality is estimable --
it correlates +0.714 between halves of the panel -- so a cell with little data
should fall back toward the tape's global mean rather than trust its own
argmax:

    score(tape, cell) = (n * cell_mean + k * global_mean) / (n + k)

With k tuned by cross-validation this makes fine cells safe: a cell with plenty
of worlds keeps its own estimate, a cell with one world defers to global
quality. Every strategy here is scored the same way -- k-fold over WORLDS
(never rows), repeated, so a tape never sees a test world during fitting.

    PYTHONPATH=. uv run python tools/tape_selector.py --panel rl/datasets/tapes/panel.jsonl
    PYTHONPATH=. uv run python tools/tape_selector.py --report-best   # dump the fitted rule
"""

import argparse
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")


# --------------------------------------------------------------------------
# Panel loading
# --------------------------------------------------------------------------

def load_panel(path, prefix_tape=None):
    """-> (margins[tape][world], ctx_by_world, tapes, worlds)

    A world is (opponent, seed). Days 0-5 are byte-identical across the
    cluster, so the day-6 context is a property of the world, not the tape;
    we take the first row's copy and check the rest agree.
    """
    margins = defaultdict(dict)
    ctx_by_world = {}
    disagree = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        w = (r["opp"], r["seed"])
        margins[r["tape"]][w] = r["margin"]
        c = r.get("ctx") or {}
        if not c:
            continue
        # Against a REACTIVE opponent the day-6 context is not a pure property
        # of the world. pub-tschinkel-psr is a router: it branches on public
        # state, so our tape's economically-inert market differences change how
        # it plays, which changes its empty-tile count, which changes how much
        # weed RNG is consumed, which shifts the day-6 shop draw. Measured: 35
        # of 40 tschinkel worlds disagree on shops across arms, against 0 of 40
        # for ahmed and 0 of 40 for boatlee.
        #
        # A selector decides at step 144 having played the PREFIX, so the
        # context it actually observes is the prefix tape's. Prefer that row.
        if w not in ctx_by_world or r["tape"] == prefix_tape:
            if w in ctx_by_world and c.get("shops") != ctx_by_world[w].get("shops"):
                disagree += 1
            ctx_by_world[w] = c
        elif c.get("shops") != ctx_by_world[w].get("shops"):
            disagree += 1
    tapes = sorted(margins)
    all_worlds = sorted({w for t in margins for w in margins[t]})
    # A selector must be free to pick ANY arm, so a world is only usable when
    # every tape has played it. Scoring on partially covered worlds silently
    # grades each strategy on whichever subset its own pick happens to cover,
    # which flatters fitted rules and (with a half-written panel) can even push
    # the oracle below them.
    worlds = [w for w in all_worlds if all(w in margins[t] for t in tapes)]
    if disagree:
        print(f"WARNING: {disagree} rows disagree on day-6 shops for their world")
    if len(worlds) < len(all_worlds):
        print(f"note: {len(all_worlds) - len(worlds)} of {len(all_worlds)} worlds "
              f"dropped as partially played (panel still filling)")
    return margins, ctx_by_world, tapes, worlds


def features(ctx):
    """Candidate day-6 selector keys, each a hashable cell label."""
    shops = ctx.get("shops") or []
    cons = ctx.get("consumers") or {}
    price = ctx.get("price") or {}
    opp = ctx.get("opp") or {}
    me = ctx.get("me") or {}
    f = {
        "yarn": "YARN_STORE" in shops,                       # the shipped rule
        "shop_pair": tuple(sorted(shops)),
        "n_distinct_shops": len(set(shops)),
        "milk_consumer": cons.get("MILK", 0) >= 1,
        "straw_2plus": cons.get("STRAWBERRY", 0) >= 2,
        "carrot_consumer": cons.get("CARROT", 0) >= 1,
        "wheat_2plus": cons.get("WHEAT", 0) >= 2,
        "egg_consumer": cons.get("EGG", 0) >= 1,
        "tomato_consumer": cons.get("TOMATO", 0) >= 1,
        "wool_x_milk": (cons.get("WOOL", 0) >= 1, cons.get("MILK", 0) >= 1),
        "wool_x_straw": (cons.get("WOOL", 0) >= 1, cons.get("STRAWBERRY", 0) >= 2),
        "opp_sheep_hi": opp.get("SHEEP", 0) >= 4,
        "opp_cow_hi": opp.get("COW", 0) >= 3,
        "opp_straw_hi": opp.get("STRAWBERRY", 0) >= 6,
        "opp_ahead": ctx.get("opp_money", 0) > ctx.get("money", 0),
        "straw_price_hi": price.get("STRAWBERRY", 1.0) >= 1.3,
        "wheat_price_hi": price.get("WHEAT", 1.0) >= 1.3,
    }
    f["yarn_x_milk"] = (f["yarn"], f["milk_consumer"])
    f["yarn_x_oppsheep"] = (f["yarn"], f["opp_sheep_hi"])
    f["yarn_x_straw2"] = (f["yarn"], f["straw_2plus"])
    return f


# --------------------------------------------------------------------------
# Strategies. Each fits on train worlds and returns pick(world) -> tape.
# --------------------------------------------------------------------------

def fit_fixed(margins, tapes, train, ctx_by_world, **kw):
    best = max(tapes, key=lambda t: st.mean([margins[t][w] for w in train if w in margins[t]] or [0]))
    return lambda w: best


def _agg(by_opp, agg):
    """Combine per-opponent means into one score.

    'mean' pools every world equally, which is how the first fit was run and
    how it went wrong: tape 55 earns +$78,407 against pub-tschinkel-psr (a
    ROUTER, which it evidently drives into a bad branch) while losing $25,658
    to pub-ahmed-v38. Pooling let that one-opponent exploit win the global mean
    and become the default arm, and the fresh-block gate duly rejected it at
    -$2,852. 'min' scores a tape by its WORST opponent, which is the same
    lesson the league work learned in gen-2 when training against a single
    incumbent bred an exploiter.
    """
    vals = [v for v in by_opp.values() if v is not None]
    if not vals:
        return 0.0
    if agg == "min":
        return min(vals)
    if agg == "median":
        return st.median(vals)
    return st.mean(vals)


def _cell_stats(margins, tapes, train, ctx_by_world, key, agg="mean"):
    """global[tape] and cell[(label, tape)] -> (score, n), aggregated by `agg`."""
    glob, cell = {}, {}
    gsum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    csum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for t in tapes:
        for w in train:
            if w not in margins[t]:
                continue
            opp = w[0]
            g = gsum[t][opp]
            g[0] += margins[t][w]
            g[1] += 1
            if w not in ctx_by_world:
                continue
            lab = features(ctx_by_world[w])[key]
            c = csum[(lab, t)][opp]
            c[0] += margins[t][w]
            c[1] += 1
    for t in tapes:
        glob[t] = _agg({o: s / n for o, (s, n) in gsum[t].items() if n}, agg)
    for k, byo in csum.items():
        n = sum(n for _, n in byo.values())
        cell[k] = [_agg({o: s / c for o, (s, c) in byo.items() if c}, agg) * n, n]
    return glob, defaultdict(lambda: [0.0, 0], cell)


def fit_key(margins, tapes, train, ctx_by_world, key=None, k=0.0, min_n=0,
            top_k=0, agg="mean", **kw):
    """Per-cell pick with shrinkage k toward the tape's global mean.

    k = 0 is the plain per-cell argmax (what the exact-shop-pair attempt did);
    large k collapses to the best fixed tape. min_n refuses to trust a cell
    that has fewer than that many train worlds for the tape. top_k is a second,
    blunter regularizer: only tapes in the top k by GLOBAL quality are eligible,
    so a cell can re-rank good tapes but cannot promote a globally bad one on
    the strength of one lucky world.
    """
    glob, cell = _cell_stats(margins, tapes, train, ctx_by_world, key, agg)
    best_global = max(tapes, key=lambda t: glob[t])
    if top_k:
        tapes = sorted(tapes, key=lambda t: -glob[t])[:top_k]

    def score(t, lab):
        s, n = cell[(lab, t)]
        if n < min_n:
            return glob[t]
        return (s + k * glob[t]) / (n + k) if (n + k) else glob[t]

    cache = {}

    def pick(w):
        if w not in ctx_by_world:
            return best_global
        lab = features(ctx_by_world[w])[key]
        if lab not in cache:
            cache[lab] = max(tapes, key=lambda t: score(t, lab))
        return cache[lab]

    return pick


def fit_oracle(margins, tapes, train, ctx_by_world, **kw):
    return lambda w: max(tapes, key=lambda t: margins[t].get(w, -1e18))


def fit_random(margins, tapes, train, ctx_by_world, rng=None, **kw):
    rng = rng or random.Random(0)
    return lambda w: rng.choice(tapes)


# --------------------------------------------------------------------------
# Honest evaluation: k-fold over worlds, repeated
# --------------------------------------------------------------------------

def evaluate(margins, ctx_by_world, tapes, worlds, fit, folds=5, repeats=6,
             seed=0, **kw):
    per_world = []
    for rep in range(repeats):
        ws = list(worlds)
        random.Random(seed + rep).shuffle(ws)
        for f in range(folds):
            test = ws[f::folds]
            train = [w for w in ws if w not in set(test)]
            if not train:
                continue
            pick = fit(margins, tapes, train, ctx_by_world,
                       rng=random.Random(seed * 1000 + rep), **kw)
            for w in test:
                t = pick(w)
                # load_panel guarantees full coverage, so a miss is a bug, not
                # a row to quietly drop.
                per_world.append(margins[t][w])
    mean = st.mean(per_world) if per_world else 0.0
    se = st.stdev(per_world) / len(per_world) ** 0.5 if len(per_world) > 1 else 0.0
    return mean, se, len(per_world)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="rl/datasets/tapes/panel.jsonl")
    ap.add_argument("--prefix-tape", type=int, default=None,
                    help="tape index whose day-6 context the selector observes "
                         "(all arms share the prefix, so this is the honest ctx "
                         "against a reactive opponent)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--report-best", action="store_true",
                    help="refit the winning strategy on ALL worlds and print the rule")
    args = ap.parse_args()

    margins, ctx, tapes, worlds = load_panel(args.panel)
    print(f"panel: {len(tapes)} tapes x {len(worlds)} worlds "
          f"({sum(len(v) for v in margins.values()):,} rows)\n")

    def run(label, fit, **kw):
        m, se, n = evaluate(margins, ctx, tapes, worlds, fit,
                            folds=args.folds, repeats=args.repeats, **kw)
        return (m, se, n, label, fit, kw)

    results = [
        run("random tape", fit_random),
        run("best fixed on train", fit_fixed),
    ]
    keys = list(features(next(iter(ctx.values()))).keys())
    for key in keys:
        for k in (0.0, 2.0, 5.0, 10.0):
            for tk in (0, 8, 20):
                lab = f"key={key} shrink={k:g}" + (f" top{tk}" if tk else "")
                results.append(run(lab, fit_key, key=key, k=k, top_k=tk))

    results.sort(key=lambda r: -r[0])
    orc = evaluate(margins, ctx, tapes, worlds, fit_oracle,
                   folds=args.folds, repeats=args.repeats)

    shipped = next((r for r in results if r[3] == "key=yarn shrink=0"), None)
    print(f"{'strategy':<34} {'held-out margin':>16} {'se':>8}")
    print(f"{'ORACLE (per-world best)':<34} {orc[0]:>+16,.0f} {orc[1]:>8,.0f}")
    for m, se, n, label, _, _ in results[:18]:
        tag = ""
        if shipped and label == shipped[3]:
            tag = "   <- shipped rule (tape-sel)"
        print(f"{label:<34} {m:>+16,.0f} {se:>8,.0f}{tag}")
    if shipped and shipped not in results[:18]:
        print(f"{shipped[3]:<34} {shipped[0]:>+16,.0f} {shipped[1]:>8,.0f}"
              f"   <- shipped rule (tape-sel)")

    best = results[0]
    print(f"\nbest: {best[3]}  {best[0]:+,.0f} +/- {best[1]:,.0f}")
    if shipped:
        d = best[0] - shipped[0]
        print(f"vs shipped yarn rule: {d:+,.0f} "
              f"({'worth gating' if d > 2 * best[1] else 'NOT resolvable at this panel size'})")
    print(f"oracle headroom still unclaimed: {orc[0] - best[0]:+,.0f}")

    if args.report_best and best[4] is fit_key:
        pick = best[4](margins, tapes, worlds, ctx, **best[5])
        rule = {}
        for w in worlds:
            if w in ctx:
                rule[str(features(ctx[w])[best[5]["key"]])] = pick(w)
        print(f"\nfitted rule ({best[3]}), cell -> tape index:")
        for lab, t in sorted(rule.items()):
            print(f"  {lab:<44} tape {t}")


if __name__ == "__main__":
    main()


def nested_evaluate(margins, ctx_by_world, tapes, worlds, candidates,
                    folds=5, repeats=6, seed=0):
    """Honest estimate of the SELECTION PROCEDURE, not of its winner.

    evaluate() cross-validates one pre-specified strategy. Running it over ~200
    candidates and reporting the max measures the max of 200 noisy draws, which
    is inflated by roughly 2-3 se. Here the key is chosen INSIDE each training
    fold and only then scored on the held-out fold, so the number answers "what
    would this fitting procedure earn on worlds it has never seen".
    """
    per_world = []
    picked = defaultdict(int)
    for rep in range(repeats):
        ws = list(worlds)
        random.Random(seed + rep).shuffle(ws)
        for f in range(folds):
            test = set(ws[f::folds])
            train = [w for w in ws if w not in test]
            # inner CV over the training worlds only
            best, best_m = None, -1e18
            for label, fit, kw in candidates:
                m, _, _ = evaluate(margins, ctx_by_world, tapes, train, fit,
                                   folds=folds, repeats=1, seed=seed + 977, **kw)
                if m > best_m:
                    best, best_m = (label, fit, kw), m
            picked[best[0]] += 1
            pick = best[1](margins, tapes, train, ctx_by_world,
                           rng=random.Random(seed), **best[2])
            for w in test:
                per_world.append(margins[pick(w)][w])
    mean = st.mean(per_world)
    se = st.stdev(per_world) / len(per_world) ** 0.5
    return mean, se, picked
