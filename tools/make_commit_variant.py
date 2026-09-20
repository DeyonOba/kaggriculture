"""Commitment as a PURE ADDITION to the shipped zone passes.

Three earlier attempts all REPLACED passes 1-3, and all three were therefore
confounded -- they compared a greedy dispatcher against the zone system and
could say nothing about the ingredient under test:

  make_top_variant    replaced all of assign(); DROP fell to 0
  make_top2_variant   replaced passes 1-3; clean, and optimal matching LOST
                      $3,636/game at z=-5.43 despite BETTER work-per-travel
  make_anchor_variant replaced passes 1-3 again; commitment looked bad, but
                      that was greedy+commitment against zones

This adds commitment WITHOUT removing anything. Order per turn:

  1. a unit with a standing target keeps walking to it
  2. every remaining unit goes through the ORIGINAL passes 1-3, untouched
  3. whatever a unit just started WALKING to becomes its standing target

COMMIT_ON=False is bit-identical; COMMIT_ON=True is exactly the shipped
executor plus persistence. One variable.

A commitment is dropped the moment it is completed, stolen, no longer a
live task, further than COMMIT_MAXD, or unreachable before the day ends.

    PYTHONPATH=. uv run python tools/make_commit_variant.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena/agents")
sys.path.insert(0, REPO)

START = "    # Pass 1 — own zone, real work"
END = "    # ---- 5. Survival override"

HELPERS = '''COMMIT_ON = __COMMIT__
COMMIT_MAXD = __MAXD__

# unit -> (x, y, op0) it is walking to. Cleared at dawn: hands are re-hired
# daily and the index is reused, so a stale target would send a brand new
# hand to yesterday's tile.
_COMMIT = {}
_COMMIT_DAY = [-1]

'''

PRE = '''    # ---- 3b. COMMITMENT (pure addition; the original passes still run) ----
    _claim_by = {}
    if COMMIT_ON:
        if _COMMIT_DAY[0] != view.day:
            _COMMIT.clear()
            _COMMIT_DAY[0] = view.day
        _live = {(t.x, t.y, t.op[0]): t for t in (all_tasks + global_tasks)}
        _turns_left = TURNS_PER_DAY - view.hour
        for _idx in sorted(free):
            _tgt = _COMMIT.get(_idx)
            if _tgt is None:
                continue
            _t = _live.get(_tgt)
            if _t is None or _tgt in claimed:
                _COMMIT.pop(_idx, None)
                continue
            _ux, _uy = units[_idx]
            _d = _dist(_ux, _uy, _t.x, _t.y)
            if _d + 1 > _turns_left or _d > COMMIT_MAXD:
                _COMMIT.pop(_idx, None)
                continue
            if try_execute(_idx, _t):
                free.discard(_idx)
                if (_ux, _uy) == (_t.x, _t.y):
                    _COMMIT.pop(_idx, None)
            else:
                _COMMIT.pop(_idx, None)
        _count_moves(actions, _seen, "commit")
        _seen = set(actions)

        _orig_try_execute = try_execute

        def try_execute(idx, task, _f=_orig_try_execute, _c=_claim_by):
            _ok = _f(idx, task)
            if _ok:
                _c[idx] = (task.x, task.y, task.op[0])
            return _ok

    _free_before = set(free)
'''

POST = '''
    if COMMIT_ON:
        # a unit that is WALKING (not executing) keeps that target
        for _idx in _free_before - free:
            _op = actions.get(_idx)
            if not _op or _op[0] not in ("NORTH", "SOUTH", "EAST", "WEST"):
                continue
            _tk = _claim_by.get(_idx)
            if _tk is not None:
                _COMMIT[_idx] = _tk

'''


def build(base, name, commit=False, maxd=12):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head = src[:i]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    tail = src[j:]
    out, hits = [], 0
    for mod, text in mods:
        if START in text and END in text:
            a, b = text.index(START), text.index(END)
            original = text[a:b].rstrip("\n")
            text = (HELPERS.replace("__COMMIT__", repr(bool(commit)))
                           .replace("__MAXD__", repr(int(maxd)))
                    + text[:a] + PRE + original + "\n" + POST + text[b:])
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
    "c-id":   dict(commit=False),           # MUST be bit-identical
    "c-hold": dict(commit=True, maxd=12),   # commit to anything reachable
    "c-near": dict(commit=True, maxd=4),    # commit only to nearby targets
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "champ-h8"
    for nm, kw in ARMS.items():
        print(f"  {nm:<8} {kw}  -> {build(base, nm, **kw)}")


if __name__ == "__main__":
    main()
