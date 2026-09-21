"""Route across unbought ground only at the border: locked depth <= PAD.

README Movement makes LOCKED tiles passable, and champ-h8 uses that -- 895
moves from a locked tile over 4 games. Most of it is unavoidable and correct:
a hired hand spawns on `(5,4)`, which is locked until NE is bought, and has
to walk out. But a Manhattan path can also cut through the *interior* of a
quadrant we do not own, where no tile action is legal and nothing can be
chained on the way.

The design constraint: a unit may stand on unbought ground only within `PAD`
tiles of owned land. PAD = 1 is the border ring, which is exactly what the
spawn-escape needs and nothing more.

Note what this can and cannot buy. Locked tiles are passable, not obstacles,
so a Manhattan path through them is never SHORTER than one around them -- the
detour costs nothing in distance. What it buys is exposure: the same number of
steps taken over workable ground instead of dead ground, which is the
`actions per tile arrival` term (ours 1.67 against the leaders' 2.10).

Measured before building: 96.5% of champ-h8's locked-ground unit-turns are
already at depth 1, and only 25 moves per 4 games go deeper. So this makes a
near-invariant into an invariant; it cannot move money at the ladder's
resolution and is not expected to.

    PYTHONPATH=. uv run python tools/make_pad_variant.py --name h10-pad1 --pad 1
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

HEAD = '''
# --- locked-ground padding (tools/make_pad_variant.py) ----------------------
# A unit may stand on unbought ground only within _PAD tiles of owned land.
# The depth grid is a multi-source BFS from every owned tile; it changes only
# when a quadrant is bought, so it is cached on the unlocked-quadrant key.
_PAD = __PAD__
_DEPTH = {"key": None, "grid": None}


def _pad_build(tiles):
    n = len(tiles)
    INF = 10 ** 6
    d = [[INF] * n for _ in range(n)]
    q = []
    for y in range(n):
        for x in range(n):
            if tiles[y][x] != "LOCKED":
                d[y][x] = 0
                q.append((x, y))
    head = 0
    while head < len(q):
        x, y = q[head]
        head += 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < n and 0 <= ny < n and d[ny][nx] == INF:
                d[ny][nx] = d[y][x] + 1
                q.append((nx, ny))
    return d


def _pad_set(view):
    key = tuple(sorted(view.me.unlocked_quadrants))
    if _DEPTH["key"] != key:
        _DEPTH["key"] = key
        _DEPTH["grid"] = _pad_build(view.me.tiles)


def _pad_ok(x, y):
    g = _DEPTH["grid"]
    if g is None:
        return True
    if not (0 <= y < len(g) and 0 <= x < len(g)):
        return False
    return g[y][x] <= _PAD

'''

STEP_FIND = '''def _step_toward(fx, fy, tx, ty):
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    if dx != 0:
        return "EAST" if dx > 0 else "WEST"
    return None'''

STEP_REPL = '''def _step_toward(fx, fy, tx, ty):
    """Manhattan step, preferring one that stays inside the padding.

    Candidate order is the original's branch order, so with the padding
    satisfied (or no grid set) this returns exactly what it always did.
    """
    dx, dy = tx - fx, ty - fy
    cands = []
    if abs(dx) >= abs(dy) and dx != 0:
        cands.append(("EAST" if dx > 0 else "WEST",
                      fx + (1 if dx > 0 else -1), fy))
    if dy != 0:
        cands.append(("SOUTH" if dy > 0 else "NORTH",
                      fx, fy + (1 if dy > 0 else -1)))
    if dx != 0:
        c = ("EAST" if dx > 0 else "WEST", fx + (1 if dx > 0 else -1), fy)
        if c not in cands:
            cands.append(c)
    if not cands:
        return None
    for name, nx, ny in cands:
        if _pad_ok(nx, ny):
            return name
    # Both axes would go deeper than the padding: take the original choice
    # rather than stall. A unit that spawned deep must still be able to leave.
    return cands[0][0]'''

SET_FIND = '''    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}'''

SET_REPL = '''    me = view.me
    _pad_set(view)
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}'''


def build(base, name, pad):
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            for find, repl in ((STEP_FIND, STEP_REPL), (SET_FIND, SET_REPL)):
                if text.count(find) != 1:
                    raise SystemExit(f"anchor not unique ({text.count(find)}):"
                                     f"\n{find[:80]}")
                text = text.replace(find, repl, 1)
            text = HEAD.replace("__PAD__", str(pad)) + text
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no scheduler module in the bundle")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    _rules = os.path.join(REPO, "arena", "agents", base, "rules.json")
    if os.path.isfile(_rules):        # keep the sidecar valid so
        import shutil                 # make_knob_variant can chain
        shutil.copyfile(_rules, os.path.join(dst_dir, "rules.json"))
    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  pad={pad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pad", type=int, default=1)
    args = ap.parse_args()
    build(args.base, args.name, args.pad)


if __name__ == "__main__":
    main()
