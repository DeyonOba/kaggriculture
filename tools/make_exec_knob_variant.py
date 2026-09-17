"""Execution knobs: screen candidate POLICY HEADS as constants before training any.

Why this exists. The RL plan's own recorded verdict was "training was not the
constraint; the economy the executor could express was", and the head audit
proves it literally: `RESIDUAL_HEADS` is ten allocation quantities plus three
sell floors. Allocation is measured flat (LP-optimal allocation, executed, tied
champ-h8 at +$828 +/- 2,317) and the floors are measured INERT (scaling the
staple floor over a 4x range changed 0 of 96 games, se = 0.00). Fourteen
generations plateaued at 810 because every parameter they could move was dead.

So the rule now is: screen a head as a CONSTANT before adding it. A knob that
does nothing as a constant cannot be learned as a head, and the floor sweep
found that out in a single run.

The knob here comes from the autopsy against pub-ahmed-v38, which scores
~$114.5k to our ~$75.0k on the SAME worlds with an essentially identical farm
(59 plants vs 60, 16 animals vs 17). The whole gap is revenue, and the turn
budget explains it:

                work          move          pass
    us     2,790 (40%)   4,033 (59%)      63 (1%)
    them   3,459 (50%)   2,920 (42%)     532 (8%)

They IDLE rather than walk, and convert it into 669 more productive actions and
~40% more harvested units. Our scheduler has the opposite bias: pass 2 helps
anywhere inside HELP_RADIUS and pass 3 takes filler work ANYWHERE on the farm,
so a free worker always walks.

    --idle-dist N   a worker whose nearest task is farther than N tiles PASSes
                    instead of walking to it
    --help-radius N overrides HELP_RADIUS (pass 2's reach)
    --no-far-filler pass 3 stops falling back to farm-wide filler work

Note this is NOT the five earlier zone relaxations. Those changed which tasks a
worker could reach; none of them ever let a worker decline work and stand
still. Idling is a different action, and it is the one the opponent uses 532
times a game.

    PYTHONPATH=. uv run python tools/make_exec_knob_variant.py --name x-idle6 --idle-dist 6
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

# pass 1 (own zone) and pass 3 (filler) are unbounded today; pass 2 uses
# HELP_RADIUS. Capping the distance a worker will WALK -- and leaving it free
# rather than sending it elsewhere -- is what produces an idle turn.
PICK_FIND = '''            d = _dist(x, y, task.x, task.y)
            if max_dist is not None and d > max_dist:
                continue'''

PICK_REPL = '''            d = _dist(x, y, task.x, task.y)
            if max_dist is not None and d > max_dist:
                continue
            if IDLE_DIST is not None and d > IDLE_DIST:
                # Decline the work and keep the position. Walking six tiles for
                # one action spends six turns to earn one; standing still keeps
                # the worker on top of tomorrow's local work.
                continue'''

FILLER_FIND = '''        task = pick(idx, zone_tasks.get(idx % n_zones, []), P_FILLER, 99)
        if task is None:
            task = pick(idx, all_tasks + global_tasks, P_FILLER, 99)'''

FILLER_NOFAR = '''        task = pick(idx, zone_tasks.get(idx % n_zones, []), P_FILLER, 99)'''


def build(base, name, idle_dist=None, help_radius=None, no_far_filler=False):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if PICK_FIND not in text:
                raise RuntimeError("pick() distance filter not found")
            text = text.replace(PICK_FIND, PICK_REPL, 1)
            text = ("IDLE_DIST = %r\n\n" % idle_dist) + text
            if help_radius is not None:
                k = text.index("HELP_RADIUS")
                eol = text.index("\n", k)
                text = text[:k] + f"HELP_RADIUS = {int(help_radius)}" + text[eol:]
            if no_far_filler:
                if FILLER_FIND not in text:
                    raise RuntimeError("filler fallback not found")
                text = text.replace(FILLER_FIND, FILLER_NOFAR, 1)
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("scheduler module not found")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--idle-dist", type=int, default=None)
    ap.add_argument("--help-radius", type=int, default=None)
    ap.add_argument("--no-far-filler", action="store_true")
    a = ap.parse_args()
    p = build(a.base, a.name, a.idle_dist, a.help_radius, a.no_far_filler)
    print(f"wrote {p}  (idle_dist={a.idle_dist} help_radius={a.help_radius} "
          f"no_far_filler={a.no_far_filler})")
