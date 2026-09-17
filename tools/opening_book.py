"""An opening book for days 0-6, searched against a REAL opponent by margin.

Objective, settled by `tools/plan_search.py validate` over 90 games a side
separating the incumbent from a known-bad knob (+4 hands):

    final margin   |t| = 7.64      <- most sensitive by far
    win rate       |t| = 2.29
    V day 16       |t| = 1.86      <- the fitted P(win) head is the WORST
    V day 12       |t| = 0.12

So margin stays the objective. It is not a bad proxy in general -- it fails only
because we are ~$40k behind, so margin gains never cross the win threshold. The
flippable frontier says where it DOES convert:

    opponent      median loss   games within $20k
    ahmed             $43,845            0/48
    tschinkel         $37,712            0/48
    boatlee           $16,646           34/48

+$10k of margin against boatlee flips roughly 12 games (5/48 -> 17/48). The same
$10k against ahmed flips nothing. So search against the opponent we can actually
catch, and track the win conversion as the thing that matters.

Self-play was tried first and REFUTED: Thompson sampling found an opening at
446/464 = 96.1% against a champ-h8 twin that scored -$11,566 and 3/144 wins
against real agents. A symmetric pairing measures "beats a twin", and two copies
sharing one price ladder trade wins through relative selling timing alone.

Why win rate and why self-play, both of which are corrections to how this
project has measured things:

  * The leaderboard rating is WIN-BASED. We gated on paired margin for weeks and
    it does not transfer: champ-h8 loses 96.5% of its games at a mean margin of
    -$31,049, so every $1-5k lever moves the size of a LOSS. `match-all` gated
    +$1,381 +/- 641 over 360 games and took the head-to-head from 5/144 to
    4/144 wins.
  * Against the public agents a win rate of 3.5% cannot resolve anything -- we
    are 0/48 vs ahmed and 0/48 vs tschinkel, with ZERO losses inside $20k. Win
    rate is only a sensitive instrument where the baseline is 50%, and that is
    SELF-PLAY. (The old note that the twin arena "cannot gate" was about margin:
    two copies sharing a price ladder trade margin for free. They cannot trade
    WINS for free -- a symmetric pairing is 50% by construction, so any
    deviation is real.)

Search, not sweep. Every knob in this project has been measured one at a time
and none held the gap; `tools/plan_search.py` records why -- farm composition is
worth <0.01 win probability while cash is worth +0.099, so the gap lives in the
JOINT trajectory. This samples whole opening CONFIGURATIONS.

Sampling. Each configuration carries a Beta(wins+1, losses+1) posterior over its
win rate. Thompson sampling draws from those posteriors and spends the next
batch on whichever config the draw favours, so strong openings get deep
evaluation and weak ones are abandoned after a few games. The book persists, so
runs accumulate rather than restart.

    PYTHONPATH=. uv run python tools/opening_book.py seed --n 24
    PYTHONPATH=. uv run python tools/opening_book.py search --rounds 40 --games 8
    PYTHONPATH=. uv run python tools/opening_book.py report
"""

import argparse
import ast
import itertools
import json
import os
import random
import statistics as st
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")
BOOK = os.path.join(REPO, "rl", "datasets", "openings", "book.json")

# --- the opening decision space (everything settled by day 6) ----------------
SPACE = {
    "melon":       [8, 10, 12, 14, 16, 18],      # CHAMP_OPENING["MELON"], day-0
    "wheat":       [4, 7, 10, 14],               # CHAMP_OPENING["WHEAT"], day-0
    "straw_early": [12, 18, 24, 30],             # strawberries before day 10
    "land_day":    [0, 3, 5, 8],                 # first quadrant purchase
    "geese_start": [2, 4, 6, 99],                # 99 = no geese in the opening
    "hire_boost":  [-2, -1, 0, 1, 2, 3],         # added to CHAMP_HIRES days 0-6
}

PATCHES = [
    ("melon",       'CHAMP_OPENING = {"MELON": 12, "WHEAT": 7}', None),
    ("land_day",    "CHAMP_LAND_DAY = 5",            "CHAMP_LAND_DAY = {v}"),
    ("geese_start", "GEESE_BUY_START_DAY = 2",       "GEESE_BUY_START_DAY = {v}"),
    ("straw_early", "CHAMP_STRAW_EARLY = 18",        "CHAMP_STRAW_EARLY = {v}"),
]

HIRES_OLD = """CHAMP_HIRES = [5, 4, 4, 5, 4, 5, 7, 9, 9, 10,"""


def cfg_name(c):
    return ("o-m{melon}w{wheat}s{straw_early}l{land_day}"
            "g{geese_start}h{hire_boost}").format(**c).replace("-", "n")


def build(cfg, base="champ-h8"):
    """Write a frozen-bundle variant for one opening configuration."""
    name = cfg_name(cfg)
    path = os.path.join(AGENTS, name, "main.py")
    if os.path.exists(path):
        return name
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, hit = [], 0
    for mod, text in mods:
        if 'CHAMP_OPENING = {"MELON"' in text:
            text = text.replace(
                'CHAMP_OPENING = {"MELON": 12, "WHEAT": 7}',
                'CHAMP_OPENING = {"MELON": %d, "WHEAT": %d}' % (cfg["melon"], cfg["wheat"]), 1)
            hit += 1
            for key, old, new in PATCHES[1:]:
                if old in text:
                    text = text.replace(old, new.format(v=cfg[key]), 1)
                    hit += 1
            if HIRES_OLD in text:
                b = cfg["hire_boost"]
                first7 = [max(0, x + b) for x in (5, 4, 4, 5, 4, 5, 7)]
                text = text.replace(
                    HIRES_OLD,
                    "CHAMP_HIRES = [%s, 9, 9, 10," % ", ".join(str(x) for x in first7), 1)
                hit += 1
        out.append((mod, text))
    if hit < 5:
        raise RuntimeError(f"only {hit}/5 opening patch sites found")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(path).read(), path, "exec")
    return name


def play(name, opp, n, base_seed, procs):
    """Self-play `name` vs `opp`; -> (wins, games). Symmetric pairing = 50%."""
    res = os.path.join(REPO, "arena", "results", f"{name}_vs_{opp}.jsonl")
    if os.path.exists(res):
        os.remove(res)
    subprocess.run(
        ["uv", "run", "python", "arena/run_games.py", "--a", name, "--b", opp,
         "--n", str(n), "--base-seed", str(base_seed), "--procs", str(procs)],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO})
    if not os.path.exists(res):
        return 0, 0
    w = g = 0
    m_sum = m_sq = 0.0
    for line in open(res):
        r = json.loads(line)
        if name not in r["players"] or (r.get("statuses")
                                        and any(s != "DONE" for s in r["statuses"])):
            continue
        i = r["players"].index(name)
        a, b = r["rewards"][i], r["rewards"][1 - i]
        if a is None or b is None:
            continue
        g += 1
        w += a > b
        d = a - b
        m_sum += d
        m_sq += d * d
    return w, g, m_sum, m_sq


def load_book():
    if os.path.exists(BOOK):
        return json.load(open(BOOK))
    return {}


def save_book(b):
    os.makedirs(os.path.dirname(BOOK), exist_ok=True)
    json.dump(b, open(BOOK, "w"), indent=1, sort_keys=True)


def _post(rec):
    """Posterior mean margin and a Thompson draw.

    Margin is continuous and roughly normal across worlds, so the posterior on
    its mean is normal with se = sd/sqrt(n). An unplayed config draws from a
    wide prior so it gets looked at. (Beta-on-wins was the self-play design; at
    a 3.5% external win rate it has no resolution.)
    """
    g = rec["g"]
    if g == 0:
        return 0.0, random.gauss(0.0, 20_000.0)
    mean = rec["m_sum"] / g
    if g < 2:
        return mean, random.gauss(mean, 20_000.0)
    var = max(0.0, rec["m_sq"] / g - mean * mean)
    se = (var / g) ** 0.5
    return mean, random.gauss(mean, max(se, 1.0))


def seed_cmd(args):
    """Enumerate a spread of configurations and give each a small first look."""
    rng = random.Random(args.seed)
    keys = list(SPACE)
    combos = set()
    # the shipped opening, then a random spread
    combos.add((12, 7, 18, 5, 2, 0))
    while len(combos) < args.n:
        combos.add(tuple(rng.choice(SPACE[k]) for k in keys))
    book = load_book()
    for combo in sorted(combos):
        cfg = dict(zip(keys, combo))
        name = build(cfg)
        book.setdefault(name, {"cfg": cfg, "w": 0, "g": 0,
                               "m_sum": 0.0, "m_sq": 0.0})
    save_book(book)
    print(f"book now holds {len(book)} openings (built {len(combos)} this run)")


def search_cmd(args):
    book = load_book()
    if not book:
        sys.exit("empty book; run `seed` first")
    rng = random.Random(args.seed)
    for rnd in range(args.rounds):
        # Thompson: spend this batch on the config with the best sampled rate.
        pick = max(book, key=lambda k: _post(book[k])[1])
        rec = book[pick]
        base_seed = args.base_seed + rng.randrange(10_000) * 100
        w, g, ms, mq = play(pick, args.opp, args.games, base_seed, args.procs)
        rec["w"] += w
        rec["g"] += g
        rec["m_sum"] = rec.get("m_sum", 0.0) + ms
        rec["m_sq"] = rec.get("m_sq", 0.0) + mq
        save_book(book)
        mean, _ = _post(rec)
        print(f"[{rnd+1}/{args.rounds}] {pick:<26} +{w}/{g}  "
              f"total {rec['w']}/{rec['g']} wins  margin {mean:>+9,.0f}", flush=True)
    report_cmd(args)


def report_cmd(args):
    book = load_book()
    rows = []
    for name, rec in book.items():
        if rec["g"] < args.min_g:
            continue
        mean, _ = _post(rec)
        g = rec["g"]
        var = max(0.0, rec["m_sq"] / g - mean * mean) if g >= 2 else 0.0
        se = (var / g) ** 0.5 if g >= 2 else 0.0
        rows.append((mean, se, name, rec))
    rows.sort(reverse=True)
    print(f"\n{'opening':<26} {'n':>5} {'margin':>10} {'se':>8} {'wins':>8}  config")
    for mean, se, name, rec in rows[:args.top]:
        c = rec["cfg"]
        print(f"{name:<26} {rec['g']:>5} {mean:>+10,.0f} {se:>8,.0f} "
              f"{rec['w']:>3}/{rec['g']:<4}  "
              f"melon={c['melon']} wheat={c['wheat']} straw={c['straw_early']} "
              f"land={c['land_day']} geese={c['geese_start']} hire{c['hire_boost']:+d}")
    if rows:
        print(f"\n{len(book)} openings, {sum(r['g'] for r in book.values())} games")
        print("Margin is the search signal (|t|=7.64 vs 2.29 for win rate); WINS "
              "are the thing that matters. Against boatlee 34/48 games sit within "
              "$20k, so margin here can actually convert.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("seed", "search", "report"):
        p = sub.add_parser(nm)
        p.add_argument("--n", type=int, default=24, help="openings to seed")
        p.add_argument("--rounds", type=int, default=40)
        p.add_argument("--games", type=int, default=8, help="seeds per round (x2 swaps)")
        p.add_argument("--opp", default="champ-h8", help="self-play reference")
        p.add_argument("--base-seed", type=int, default=200_000)
        p.add_argument("--procs", type=int, default=6)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--min-g", type=int, default=8)
        p.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    {"seed": seed_cmd, "search": search_cmd, "report": report_cmd}[a.cmd](a)
