"""Cross-turn commitment, restricted to P_SURVIVAL tasks. Nothing else.

`tools/exec_commands.py` found champ-h8 disobeying 15% of direct orders, and
in 27 of 27 unobeyed tile commands a unit was standing ON the target and
emitted a move instead. `match-all` (global-greedy claim resolution) changes
none of it, so index-order claiming is not the cause. `a-commit` does move it
(+14pp on WATER, +17pp on FEED) -- but a-commit replaces the entire three-pass
assignment with a value-density scorer AND adds commitment, so its gain is
confounded and it cannot tell you which half did the work.

This is the isolated half: the three passes are untouched, and the only
change is that a unit walking toward a **P_SURVIVAL** task keeps that target
across turns instead of re-deciding from scratch. P_SURVIVAL only, because
those are the tasks where a dropped target costs a plant or an animal, and
because it is the band the commanded tests inject at.

    --shield  also exempt a committed unit from the shed shuttle, which runs
              before every work pass and `free.discard()`s whoever it grabs.
              Splits "re-decided by a later pass" from "hijacked by logistics".

    PYTHONPATH=. uv run python tools/make_survcommit_variant.py --name h10-cmt
    PYTHONPATH=. uv run python tools/make_survcommit_variant.py --name h10-cmts \
        --shield
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

HEAD = '''
# --- cross-turn commitment, P_SURVIVAL only (tools/make_survcommit_variant) --
# unit -> (x, y, op0) it is walking to. Cleared at dawn: hands are re-hired
# every day and the index is reused, so a stale entry would send a brand new
# hand to yesterday's tile.
_CMT = {}
_CMT_DAY = [-1]
_CMT_SHIELD = __SHIELD__

'''

DAWN_FIND = '''    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    seeds_left = dict(view.seeds)'''

DAWN_REPL = '''    me = view.me
    units = {idx: (x, y) for idx, x, y in me.units}
    free = set(units)
    actions = {}
    if _CMT_DAY[0] != view.day:
        _CMT.clear()
        _CMT_DAY[0] = view.day
    seeds_left = dict(view.seeds)'''

SHUTTLE_FIND = '''    fert_wanted = any(t.need == "FERTILIZER" for t in tasks)
    for idx, x, y in me.units:
        inv = view.unit_inv(idx)'''

SHUTTLE_REPL = '''    fert_wanted = any(t.need == "FERTILIZER" for t in tasks)
    for idx, x, y in me.units:
        if _CMT_SHIELD and idx in _CMT:
            continue        # already walking to something that dies tonight
        inv = view.unit_inv(idx)'''

EXEC_FIND = '''        if (x, y) == (task.x, task.y):
            if not _unlocked(view, task.x, task.y):
                return False
            actions[idx] = task.op
            if task.op[0] == "PLANT":
                plant_requests[task.op[1]] = plant_requests.get(task.op[1], 0) + 1
        else:
            step = _step_toward(x, y, task.x, task.y)
            if not step:
                return False
            actions[idx] = [step]
        claimed.add(key)
        return True'''

EXEC_REPL = '''        if (x, y) == (task.x, task.y):
            if not _unlocked(view, task.x, task.y):
                return False
            actions[idx] = task.op
            if task.op[0] == "PLANT":
                plant_requests[task.op[1]] = plant_requests.get(task.op[1], 0) + 1
            _CMT.pop(idx, None)                 # arrived and executed
        else:
            step = _step_toward(x, y, task.x, task.y)
            if not step:
                return False
            actions[idx] = [step]
            if task.pri == P_SURVIVAL:
                _CMT[idx] = (task.x, task.y, task.op[0])
        claimed.add(key)
        return True'''

PASS1_FIND = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):'''

PASS1_REPL = '''    # Pass 0 — honour a standing P_SURVIVAL target before anything re-decides.
    # Without this the unit walking to a dying plant is re-scored every turn
    # against a freshly claimed task list and drifts off; measured, a unit
    # reached the commanded tile in 27 of 27 failures and stepped off it.
    _live = {(t.x, t.y, t.op[0]): t for t in all_tasks + global_tasks}
    for _idx in sorted(free):
        _tgt = _CMT.get(_idx)
        if _tgt is None:
            continue
        _t = _live.get(_tgt)
        if _t is None or _tgt in claimed or _t.pri != P_SURVIVAL:
            _CMT.pop(_idx, None)                # done, stolen, or downgraded
            continue
        if try_execute(_idx, _t):
            free.discard(_idx)
        else:
            _CMT.pop(_idx, None)

    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):'''

PATCHES = [DAWN_FIND, SHUTTLE_FIND, EXEC_FIND, PASS1_FIND]
REPLS = [DAWN_REPL, SHUTTLE_REPL, EXEC_REPL, PASS1_REPL]


def build(base, name, shield):
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            for find, repl in zip(PATCHES, REPLS):
                if text.count(find) != 1:
                    raise SystemExit(
                        f"anchor not unique in scheduler ({text.count(find)} "
                        f"matches):\n{find[:80]}")
                text = text.replace(find, repl, 1)
            text = HEAD.replace("__SHIELD__", "True" if shield else "False") + text
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
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  "
          f"shield={shield}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--shield", action="store_true")
    args = ap.parse_args()
    build(args.base, args.name, args.shield)


if __name__ == "__main__":
    main()
