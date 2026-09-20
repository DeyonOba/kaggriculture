"""A proper Team Orienteering dispatcher, re-solved every turn.

The dawn-snapshot oracle (tools/dispatch_oracle.py) could not answer the
question it was built for: a cheap anchored dispatch scored 0.59 work
actions per travel turn against the executor's 0.73, but it only ever saw
the DAWN task list, while the real set regenerates all day -- animals
produce, crops ripen, hauling appears. The executor logs 102.7 work actions
against 72.2 dawn tasks, so a static oracle structurally undercounts.

This replaces the ASSIGNMENT only. Task generation, market orders, hiring
and planting all stay exactly as shipped, so the single changed variable is
how units are matched to work.

THE SOLVE, once per turn, on the live task set:

  cost(u,t)   Manhattan u->t, plus a SHED DETOUR when the task needs a
              carried item the unit is not holding (FEED wants wheat,
              FERTILIZE wants fertilizer), plus 1 turn for the action and
              1 more for the PICKUP. Infeasible if it cannot finish before
              the day ends -- that is the orienteering budget.
  value(u,t)  score(t)/cost, PLUS a one-step lookahead: what is the best
              task reachable FROM t with the turns that remain after it.
              The lookahead is the difference between an assignment and a
              route; without it the solver is myopic and chases whatever is
              nearest.
  matching    optimal via Hungarian (scipy linear_sum_assignment), with a
              greedy + 2-swap fallback so the agent never hard-depends on
              scipy.

Re-solving every turn is what handles regeneration: a task created at hour
9 enters the matrix at hour 9.

    PYTHONPATH=. uv run python tools/make_top_variant.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena/agents")
sys.path.insert(0, REPO)

TOP_SRC = '''

# ---- Team Orienteering dispatcher -----------------------------------------
TOP_ON = __TOP_ON__
TOP_LOOKAHEAD = __LOOKAHEAD__
TOP_SCORE = {0: 12.0, 1: 6.0, 1.5: 5.0, 2: 3.0, 3: 2.0, 5: 1.0}


def _top_score(t):
    return TOP_SCORE.get(t.pri, 1.0)


def _top_hungarian(M):
    """Maximise total value. -> [(row, col)]. scipy when present, else a
    greedy + 2-swap that is optimal for the tiny sizes here (<=12 units)."""
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
    order = sorted(range(n_r),
                   key=lambda i: -max([v for v in M[i] if v > -1e8] or [-1e9]))
    for i in order:
        best, bj = -1e9, -1
        for j in range(n_c):
            if j in used or M[i][j] <= -1e8:
                continue
            if M[i][j] > best:
                best, bj = M[i][j], j
        if bj >= 0:
            used.add(bj)
            out.append((i, bj))
    improved = True
    while improved:                      # 2-swap polish
        improved = False
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                ia, ja = out[a]
                ib, jb = out[b]
                cur = M[ia][ja] + M[ib][jb]
                alt = M[ia][jb] + M[ib][ja]
                if alt > cur + 1e-9:
                    out[a], out[b] = (ia, jb), (ib, ja)
                    improved = True
    return out


def _top_cost(view, idx, ux, uy, t):
    """Turns to finish task t from (ux,uy). None if it cannot be finished."""
    need = t.need
    if need and view.unit_inv(idx).get(need, 0) < 1:
        sx, sy = _nearest_shed_tile(ux, uy)
        if view.shed.get(need, 0) < 1:
            return None                  # the item does not exist to fetch
        c = _dist(ux, uy, sx, sy) + 1 + _dist(sx, sy, t.x, t.y) + 1
    else:
        c = _dist(ux, uy, t.x, t.y) + 1
    left = TURNS_PER_DAY - view.hour
    return c if c <= left else None


def _top_assign(view, plan, tasks):
    """Per-turn TOP solve. Same contract as assign(): {unit_idx: op_list}."""
    me = view.me
    units = [(idx, x, y) for idx, x, y in me.units]
    if not units or not tasks:
        return {}
    cand = tasks[:TOP_POOL]
    M = []
    for idx, ux, uy in units:
        row = []
        for t in cand:
            c = _top_cost(view, idx, ux, uy, t)
            if c is None:
                row.append(-1e9)
                continue
            v = _top_score(t) / float(c)
            if TOP_LOOKAHEAD:
                # ROUTE, not assignment: credit the best follow-on task
                # reachable from t with what is left of the day.
                rem = (TURNS_PER_DAY - view.hour) - c
                best2 = 0.0
                if rem > 1:
                    for t2 in cand:
                        if t2 is t:
                            continue
                        c2 = _dist(t.x, t.y, t2.x, t2.y) + 1
                        if c2 <= rem and not t2.need:
                            s2 = _top_score(t2) / float(c2)
                            if s2 > best2:
                                best2 = s2
                v += TOP_LOOKAHEAD * best2
            row.append(v)
        M.append(row)

    out = {}
    for i, j in _top_hungarian(M):
        idx, ux, uy = units[i]
        t = cand[j]
        need = t.need
        if need and view.unit_inv(idx).get(need, 0) < 1:
            sx, sy = _nearest_shed_tile(ux, uy)
            if (ux, uy) == (sx, sy):
                want = 5 if need == "WHEAT" else 3
                n = min(want, view.shed.get(need, 0))
                out[idx] = ["PICKUP", need, n] if n > 0 else ["PASS"]
            else:
                st = _step_toward(ux, uy, sx, sy)
                out[idx] = [st] if st else ["PASS"]
            continue
        if (ux, uy) == (t.x, t.y):
            out[idx] = list(t.op)
        else:
            st = _step_toward(ux, uy, t.x, t.y)
            out[idx] = [st] if st else ["PASS"]
    for idx, ux, uy in units:
        out.setdefault(idx, ["PASS"])
    return out


_assign_zone = assign


def assign(view, plan, tasks):
    if TOP_ON:
        return _top_assign(view, plan, tasks)
    return _assign_zone(view, plan, tasks)
'''

HEAD = '''TURNS_PER_DAY = 24
TOP_POOL = __POOL__

'''


def build(base, name, top_on=False, lookahead=0.5, pool=60):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head = src[:i]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    tail = src[j:]

    out, hits = [], 0
    for mod, text in mods:
        if "def assign(view, plan, tasks):" in text:
            text = (HEAD.replace("__POOL__", repr(int(pool))) + text
                    + TOP_SRC.replace("__TOP_ON__", repr(bool(top_on)))
                             .replace("__LOOKAHEAD__", repr(float(lookahead))))
            hits += 1
        out.append((mod, text))
    if hits != 1:
        raise RuntimeError(f"assign() matched {hits} modules, expected 1")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


ARMS = {
    "top-id":   dict(top_on=False),                    # MUST be identical
    "top-myo":  dict(top_on=True, lookahead=0.0),      # assignment only
    "top-look": dict(top_on=True, lookahead=0.5),      # + route lookahead
    "top-deep": dict(top_on=True, lookahead=1.0),
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "champ-h8"
    for nm, kw in ARMS.items():
        print(f"  {nm:<10} {kw}  -> {build(base, nm, **kw)}")


if __name__ == "__main__":
    main()
