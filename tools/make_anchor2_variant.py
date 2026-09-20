"""Pooled anchors as a PURE ADDITION to the shipped zone passes.

The spec: at dawn each hand gets a start work coordinate. Score every square
by the investment standing on it and the work pending on it, normalise,
POOL over sub-regions, and hand each unit the densest region. LOCKED squares
are 0, empty unlocked squares are 1, every pending task layers on top:

    val = 0                    LOCKED
        = 1                    empty, unlocked
        = 1 + inv_n + work_n   occupied, normalised across the board

The board is 10x10 but usually only one or two 5x5 quadrants are unlocked,
so the pool runs over the whole 10x10 and the locked zeros never win a
window. Anchors are kept at least K apart -- that separation IS the
neighbour broadcast.

PURE ADDITION, because every previous attempt that replaced passes 1-3 was
confounded and said nothing about the ingredient under test. Order per turn:

  1. each free unit takes the best live task inside its anchor region
  2. every unit that pass did not consume goes through the ORIGINAL
     passes 1-3, untouched
  3. section 5 survival override still runs

ANCHOR_ON=False is bit-identical. Anchors are computed ONCE AT DAWN and held
for the day, which is what the spec asks for and also what the commitment
result argues for: local persistence is what paid, re-optimising every turn
is what lost.

EXPECTATION, set before running. The commitment arm scored +$3,233 at
z=+6.11 in self-play and gated at +$334 +/- $999 with the win rate going
DOWN. Self-play is not the instrument here. This goes to ladder_gate.

    PYTHONPATH=. uv run python tools/make_anchor2_variant.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena/agents")
sys.path.insert(0, REPO)

START = "    # Pass 1 — own zone, real work"
END = "    # ---- 5. Survival override"

HELPERS = '''ANCHOR_ON = __ON__
ANCHOR_K = __K__
ANCHOR_MAX = __MAX__
ANCHOR_R = __R__
ANCHOR_SCORE = {0: 12.0, 1: 6.0, 1.5: 5.0, 2: 3.0, 3: 2.0, 5: 1.0}

_ANCH = {}
_ANCH_DAY = [-1]


def _anch_score(t):
    return ANCHOR_SCORE.get(t.pri, 1.0)


def _anch_value_map(view, tasks):
    """LOCKED 0, empty 1, occupied 1 + investment + pending work."""
    tiles = view.me.tiles
    n = len(tiles)
    inv_g = [[0.0] * n for _ in range(n)]
    wrk_g = [[0.0] * n for _ in range(n)]
    opn = [[False] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            opn[y][x] = True
            if isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    inv_g[y][x] = float(CROPS[t["crop"]]["seed"])
                elif t.get("animal"):
                    inv_g[y][x] = float(ANIMALS[t["animal"]]["cost"])
    for t in tasks:
        if 0 <= t.x < n and 0 <= t.y < n:
            wrk_g[t.y][t.x] += _anch_score(t)
    mi = max([v for r in inv_g for v in r] or [0.0]) or 1.0
    mw = max([v for r in wrk_g for v in r] or [0.0]) or 1.0
    out = [[0.0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            if opn[y][x]:
                out[y][x] = 1.0 + inv_g[y][x] / mi + wrk_g[y][x] / mw
    return out


def _anch_pool(val, k, want):
    """Pool KxK windows, return up to `want` centres kept >= k apart."""
    n = len(val)
    cells = []
    for y in range(0, n - k + 1):
        for x in range(0, n - k + 1):
            vs = [val[y + dy][x + dx] for dy in range(k) for dx in range(k)]
            v = max(vs) if ANCHOR_MAX else sum(vs) / float(k * k)
            if v > 0:
                cells.append((v, x + k // 2, y + k // 2))
    cells.sort(reverse=True)
    out = []
    for v, cx, cy in cells:
        if any(_dist(cx, cy, ax, ay) < k for ax, ay in out):
            continue          # separation = the neighbour broadcast
        out.append((cx, cy))
        if len(out) >= want:
            break
    return out

'''

PRE = '''    # ---- 3b. POOLED ANCHORS (pure addition; original passes still run) ----
    if ANCHOR_ON:
        if _ANCH_DAY[0] != view.day or not _ANCH:
            # computed ONCE at dawn and held for the day, per the spec
            _vm = _anch_value_map(view, all_tasks + global_tasks)
            _regions = _anch_pool(_vm, ANCHOR_K, max(len(units), 1))
            _ANCH.clear()
            _ANCH_DAY[0] = view.day
            for _i, _idx in enumerate(sorted(units)):
                if _regions:
                    _ANCH[_idx] = _regions[_i % len(_regions)]
        for _idx in sorted(free):
            _a = _ANCH.get(_idx)
            if _a is None:
                continue
            _ax, _ay = _a
            _ux, _uy = units[_idx]
            _inv = view.unit_inv(_idx)
            _left = TURNS_PER_DAY - view.hour
            _best, _bt = None, None
            for _t in (all_tasks + global_tasks):
                if (_t.x, _t.y, _t.op[0]) in claimed:
                    continue
                if _dist(_t.x, _t.y, _ax, _ay) > ANCHOR_R:
                    continue          # outside this unit's region
                if _t.need and _inv.get(_t.need, 0) < 1:
                    continue
                _c = _dist(_ux, _uy, _t.x, _t.y) + 1
                if _c > _left:
                    continue
                _v = _anch_score(_t) / float(_c)
                if _best is None or _v > _best:
                    _best, _bt = _v, _t
            if _bt is not None and try_execute(_idx, _bt):
                free.discard(_idx)
        _count_moves(actions, _seen, "anchor")
        _seen = set(actions)

'''


def build(base, name, on=False, k=3, pmax=True, r=3):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, tail = src[:i], src[j:]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    out, hits = [], 0
    for mod, text in mods:
        if START in text and END in text:
            a = text.index(START)
            text = (HELPERS.replace("__ON__", repr(bool(on)))
                           .replace("__K__", repr(int(k)))
                           .replace("__MAX__", repr(bool(pmax)))
                           .replace("__R__", repr(int(r)))
                    + text[:a] + PRE + text[a:])
            hits += 1
        out.append((mod, text))
    if hits != 1:
        raise RuntimeError(f"matched {hits} modules, expected 1")
    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


ARMS = {
    "k-id":   dict(on=False),                        # MUST be identical
    "k-max3": dict(on=True, k=3, pmax=True,  r=3),   # 3x3 max pool
    "k-avg3": dict(on=True, k=3, pmax=False, r=3),   # 3x3 mean pool
    "k-max5": dict(on=True, k=5, pmax=True,  r=4),   # 5x5, quadrant-sized
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "champ-h8"
    for nm, kw in ARMS.items():
        print(f"  {nm:<8} {kw}  -> {build(base, nm, **kw)}")


if __name__ == "__main__":
    main()
