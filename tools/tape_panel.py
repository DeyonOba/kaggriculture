"""Training panel for the day-6 tape selector.

All 58 tapes in the cluster share days 0-5 byte for byte, so at step 144 any of
them can be spliced in losslessly (verified: A->B reproduces B alone to the
dollar). That makes step 144 a genuine decision point with 58 arms, and the
oracle over it is worth +$15,826 a game above the best fixed tape.

This runs every arm on every world and records the day-6 context alongside the
outcome, so a selector can be fitted and held out.
"""
import json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402
import tools.tape_agent as T  # noqa: E402

OPPS = ["arena/agents/pub-ahmed-v38/main.py", "arena/agents/pub-boatlee-v16/main.py"]
OUT = "rl/datasets/tapes/panel.jsonl"
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def ctx_at(obs, seat=0):
    """Everything observable at step 144, day 6."""
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    market = obs.get("market") or {}
    prices = market.get("prices") or {}
    me, op = obs["farms"][seat], obs["farms"][1 - seat]

    def census(f):
        c = {}
        for row in f["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    if t.get("kind") == "PLANT":
                        c[t["crop"]] = c.get(t["crop"], 0) + 1
                    elif t.get("kind") in ("PASTURE", "COOP"):
                        a = t.get("animal")
                        a = a.get("type") if isinstance(a, dict) else a
                        if a:
                            c[a] = c.get(a, 0) + 1
        return c

    return {
        "shops": list(shops),
        "consumers": {p: sum(1 for s in shops if p in K.SHOPS.get(s, ())) for p in PRODUCTS},
        "price": {p: prices.get(p, 0) / K.MARKET_PARAMS[p]["base"] for p in PRODUCTS},
        "money": me["money"], "opp_money": op["money"],
        "opp": census(op), "me": census(me),
    }


def _job(a):
    idx, tape_actions, opp, seed = a
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    try:
        env.run([T.make_tape_agent(tape_actions), opp])
    except Exception:  # noqa: BLE001
        return None
    f = [env.steps[-1][s]["reward"] for s in (0, 1)]
    if f[0] is None or f[1] is None:
        return None
    ctx = ctx_at(env.steps[144][0]["observation"]) if len(env.steps) > 144 else {}
    return {"tape": idx, "opp": os.path.basename(os.path.dirname(opp)), "seed": seed,
            "final": float(f[0]), "margin": float(f[0] - f[1]), "ctx": ctx}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=90000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--opps", default=",".join(OPPS))
    ap.add_argument("--cluster", default="/tmp/cluster.json")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--prefix-tape", type=int, default=None,
                    help="index of the tape whose days 0-5 every arm is spliced "
                         "onto. The cluster is byte-identical in FIELD actions "
                         "only -- its 58 tapes carry 25 distinct MARKET scripts "
                         "over days 0-5, and market orders move money, plantings "
                         "and hence the empty-tile count that the weed RNG "
                         "consumes, which shifts the day-6 shop draw itself. So "
                         "measuring a tape played from step 0 is NOT what a "
                         "selector gets when it splices that tape in at 144. "
                         "With this set, the panel measures the spliced agent.")
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows and only run (tape, opp, seed) triples "
                         "not already present -- the world count is what limits how "
                         "fine a selector key can be fitted, so panels grow")
    args = ap.parse_args()
    lib = json.load(open("rl/datasets/tapes/library.json"))
    cl = json.load(open(args.cluster))
    opps = [o.strip() for o in args.opps.split(",") if o.strip()]
    done = set()
    if args.append and os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["tape"], r["opp"], r["seed"]))
    # WORLD-major: finish every tape on one world before starting the next.
    # A selector can only use a world where ALL arms have played, so tape-major
    # ordering leaves a killed run with a panel that is 100% useless until it is
    # ~100% done. This way a partial run is a smaller but complete panel.
    def arm(i):
        a = lib[i]["actions"]
        if args.prefix_tape is None:
            return a
        return lib[args.prefix_tape]["actions"][:144] + a[144:]

    jobs = [(i, arm(i), o, args.base_seed + k)
            for k in range(args.seeds) for o in opps for i in cl
            if (i, os.path.basename(os.path.dirname(o)), args.base_seed + k) not in done]
    print(f"{len(cl)} tapes x {len(opps)} opponents x {args.seeds} seeds; "
          f"{len(done)} rows already present -> {len(jobs)} games to run", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    OUT_MODE = "a" if args.append else "w"
    with open(args.out, OUT_MODE) as fh, Pool(args.procs) as p:
        for r in p.imap_unordered(_job, jobs, chunksize=4):
            if r:
                fh.write(json.dumps(r) + "\n")
                n += 1
                if n % 200 == 0:
                    print(f"  {n}/{len(jobs)}", flush=True)
    print(f"wrote {args.out}: {n} new rows ({len(done)+n} total)")


if __name__ == "__main__":
    main()
