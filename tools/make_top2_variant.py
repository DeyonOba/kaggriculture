"""TOP dispatch, CLEANLY: replace only the task-matching rounds.

The first attempt (tools/make_top_variant.py) swapped assign() wholesale and
scored 0.517 work-per-travel against 0.736. That number was not a verdict on
Team Orienteering: replacing the whole function silently deleted three
things that live INSIDE assign() and do not come from build_tasks --

  section 1   liquidation / heavy-load shuttles   (DROP fell to ZERO)
  section 2   zone provisioning                    (PICKUP 370 -> 131)
  section 5   survival override

so it compared a 120-line dispatcher against a 270-line one.

This keeps all three and replaces ONLY passes 1-3, the rounds that match
free units to unclaimed tasks by `pick()`. Everything still goes through
try_execute(), so `claimed`, plant_requests and the action format are
untouched. The single changed variable is how work is matched.

Passes 1-3 are banded: pass 1 is own-zone, pass 2 is anywhere within
HELP_RADIUS, pass 3 is filler. The TOP solve is unbanded by design -- a
global optimal matching over every unclaimed task, priced by score over
reach cost, with the route lookahead. That IS the difference being tested.

    PYTHONPATH=. uv run python tools/make_top2_variant.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena/agents")
sys.path.insert(0, REPO)

START = "    # Pass 1 — own zone, real work"
END = "    # ---- 5. Survival override"

HELPERS = '''TURNS_PER_DAY = 24
TOP_ON = __TOP_ON__
TOP_LOOKAHEAD = __LOOKAHEAD__
TOP_POOL = __POOL__
TOP_SCORE = {0: 12.0, 1: 6.0, 1.5: 5.0, 2: 3.0, 3: 2.0, 5: 1.0}


def _top_score(t):
    return TOP_SCORE.get(t.pri, 1.0)


def _top_hungarian(M):
    """Maximise total value. scipy when available, greedy + 2-swap otherwise."""
    n_r = len(M)
    if not n_r:
        return []
    n_c = len(M[0])
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np
        A = _np.asarray(M, dtype=float)
        A = _np.where(_np.isfinite(A), A, -1e9)
        r, c = linear_sum_assignment(-A)
        return [(int(i), int(j)) for i, j in zip(r, c) if A[i][j] > -1e8]
    except Exception:
        pass
    used, out = set(), []
    for i in sorted(range(n_r),
                    key=lambda i: -max([v for v in M[i] if v > -1e8] or [-1e9])):
        best, bj = -1e9, -1
        for j in range(n_c):
            if j not in used and M[i][j] > best and M[i][j] > -1e8:
                best, bj = M[i][j], j
        if bj >= 0:
            used.add(bj)
            out.append((i, bj))
    changed = True
    while changed:
        changed = False
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                ia, ja = out[a]
                ib, jb = out[b]
                if M[ia][jb] + M[ib][ja] > M[ia][ja] + M[ib][jb] + 1e-9:
                    out[a], out[b] = (ia, jb), (ib, ja)
                    changed = True
    return out

'''

TOP_BLOCK = '''    # ---- 3b. TOP matching (replaces passes 1-3 when TOP_ON) --------------
    if TOP_ON:
        _pool = [t for t in (all_tasks + global_tasks)
                 if (t.x, t.y, t.op[0]) not in claimed][:TOP_POOL]
        _order = sorted(free)
        if _order and _pool:
            _M = []
            for _idx in _order:
                _ux, _uy = units[_idx]
                _inv = view.unit_inv(_idx)
                _row = []
                _left = TURNS_PER_DAY - view.hour
                for _t in _pool:
                    if _t.need and _inv.get(_t.need, 0) < 1:
                        _row.append(-1e9)
                        continue
                    _c = _dist(_ux, _uy, _t.x, _t.y) + 1
                    if _c > _left:
                        _row.append(-1e9)
                        continue
                    _v = _top_score(_t) / float(_c)
                    if TOP_LOOKAHEAD:
                        _rem = _left - _c
                        _b2 = 0.0
                        if _rem > 1:
                            for _t2 in _pool:
                                if _t2 is _t or _t2.need:
                                    continue
                                _c2 = _dist(_t.x, _t.y, _t2.x, _t2.y) + 1
                                if _c2 <= _rem:
                                    _s2 = _top_score(_t2) / float(_c2)
                                    if _s2 > _b2:
                                        _b2 = _s2
                        _v += TOP_LOOKAHEAD * _b2
                    _row.append(_v)
                _M.append(_row)
            for _i, _j in _top_hungarian(_M):
                _idx = _order[_i]
                if _idx in free and try_execute(_idx, _pool[_j]):
                    free.discard(_idx)
        _count_moves(actions, _seen, "top")
        _seen = set(actions)
    else:
__ORIGINAL__

'''


def build(base, name, top_on=False, lookahead=0.5, pool=80):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head = src[:i]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    tail = src[j:]

    out, hits = [], 0
    for mod, text in mods:
        if START in text and END in text:
            a = text.index(START)
            b = text.index(END)
            original = text[a:b].rstrip("\n")
            # indent the original passes by 4 so they sit under `else:`
            indented = "\n".join(("    " + ln) if ln.strip() else ln
                                 for ln in original.split("\n"))
            block = TOP_BLOCK.replace("__ORIGINAL__", indented)
            text = (HELPERS.replace("__TOP_ON__", repr(bool(top_on)))
                           .replace("__LOOKAHEAD__", repr(float(lookahead)))
                           .replace("__POOL__", repr(int(pool)))
                    + text[:a] + block + text[b:])
            hits += 1
        out.append((mod, text))
    if hits != 1:
        raise RuntimeError(f"pass 1-3 block matched {hits} modules, expected 1")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


ARMS = {
    "t2-id":   dict(top_on=False),                 # MUST be bit-identical
    "t2-myo":  dict(top_on=True, lookahead=0.0),   # optimal matching only
    "t2-look": dict(top_on=True, lookahead=0.5),   # + route lookahead
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "champ-h8"
    for nm, kw in ARMS.items():
        print(f"  {nm:<9} {kw}  -> {build(base, nm, **kw)}")


if __name__ == "__main__":
    main()
