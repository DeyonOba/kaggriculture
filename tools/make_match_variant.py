"""Global assignment instead of first-come-first-served by unit index.

The defect is in the loop header, not the scoring. All three assignment passes
in `scheduler.assign` read

    for idx in sorted(free):
        task = pick(idx, pool, lo, hi)

so units claim tasks IN UNIT-INDEX ORDER. Unit 0 takes the task nearest to
*unit 0* even when unit 5 is standing on it, unit 5 then walks somewhere else,
and the claim is irrevocable (`claimed` is a set, `pick` skips claimed keys).
That is not greedy matching -- it is arbitrary-order greedy, whose cost can be
arbitrarily worse than the minimum-cost matching on the same candidate set.

This replaces each pass with global-greedy matching: build every (unit, task)
candidate pair that the pass would have considered, sort by the SAME cost the
old `pick` minimised -- (urgent, distance, priority), packed into one number --
and take pairs cheapest-first, skipping a pair whose unit or task is already
spoken for. Identical candidate set, identical filters, identical
`try_execute`; only the order in which claims are resolved changes.

Why this is not the previous eight routing attempts: five were relaxations of
the zone partition (walk caps, zone sizing, priority bands, idle-help scoring),
one changed what a task is worth (prize-collecting pick, much worse), one
changed pickup batching (null), one committed workers to dawn tours (worse).
Every one of them kept index-order claiming. None changed the ASSIGNMENT RULE.

Bounded bet, stated up front: the literature puts greedy at ~0.93x Hungarian on
multi-robot target tracking, and one agricultural field-layout study found
greedy actually BEAT Hungarian on batch completion time. So this is worth a few
percent, not a 30% swing -- but it is free at run time (12 units x ~60 tasks is
microseconds in pure Python) and it is the one mechanism still untried.

    PYTHONPATH=. uv run python tools/make_match_variant.py --name match-p1
    PYTHONPATH=. uv run python tools/make_match_variant.py --name match-all --passes all
    PYTHONPATH=. uv run python tools/make_match_variant.py --name match-x --cross-zone
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

MATCH_FN = '''

# --- global assignment (tools/make_match_variant.py) -------------------------
def _match_pairs(cands):
    """Global-greedy bipartite matching.

    cands: list of (cost, unit_idx, task). Returns {unit_idx: task}. Cheapest
    pair first, skipping any pair whose unit or task is already spoken for.
    Ties break on unit index so behaviour stays deterministic.
    """
    out = {}
    used_u = set()
    used_t = set()
    for cost, idx, task in sorted(cands, key=lambda c: (c[0], c[1])):
        key = (task.x, task.y, task.op[0])
        if idx in used_u or key in used_t:
            continue
        out[idx] = task
        used_u.add(idx)
        used_t.add(key)
    return out
'''

# Insert the pass-level matcher right after `all_tasks` is built, so it closes
# over units / view / claimed / late / try_execute / free exactly like pick().
HELPER_FIND = '''    all_tasks = sorted(
        (t for pool in zone_tasks.values() for t in pool),
        key=lambda t: (t.pri, t.y, t.x),
    )
'''

HELPER_REPL = '''    all_tasks = sorted(
        (t for pool in zone_tasks.values() for t in pool),
        key=lambda t: (t.pri, t.y, t.x),
    )

    def match_pass(lo, hi, pool_fn, max_dist=None):
        """One assignment pass, resolved as a global matching.

        Same candidate set and same cost as the pick() loop it replaces --
        (urgent, d, pri) packed into one number, priorities are small ints so
        the packing is order-preserving -- but claims are resolved
        cheapest-pair-first instead of in unit-index order.
        """
        cands = []
        for idx in sorted(free):
            ux, uy = units[idx]
            inv = view.unit_inv(idx)
            for task in pool_fn(idx):
                if not (lo <= task.pri < hi):
                    continue
                if (task.x, task.y, task.op[0]) in claimed:
                    continue
                if task.need and inv.get(task.need, 0) < 1:
                    continue
                d = _dist(ux, uy, task.x, task.y)
                if max_dist is not None and d > max_dist:
                    continue
                urgent = 0 if (late and task.pri == P_SURVIVAL) else 1
                cands.append((urgent * 10000 + d * 100 + task.pri, idx, task))
        chosen = _match_pairs(cands)
        for idx in sorted(chosen):
            if try_execute(idx, chosen[idx]):
                free.discard(idx)
'''

PASS1_FIND = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
'''

PASS1_REPL = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    # A unit already carrying an animal still places it immediately, anywhere;
    # only the zone round is resolved as a matching.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        if not deliveries:
            continue
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
    match_pass(P_SURVIVAL, P_FILLER, lambda idx: __POOL1__)
'''

PASS2_FIND = '''    for idx in sorted(free):
        task = pick(idx, all_tasks + global_tasks, P_SURVIVAL, P_FILLER, max_dist=HELP_RADIUS)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
'''

PASS2_REPL = '''    match_pass(P_SURVIVAL, P_FILLER, lambda idx: all_tasks + global_tasks,
               max_dist=HELP_RADIUS)
'''

PASS3_FIND = '''    for idx in sorted(free):
        task = pick(idx, zone_tasks.get(idx % n_zones, []), P_FILLER, 99)
        if task is None:
            task = pick(idx, all_tasks + global_tasks, P_FILLER, 99)
        if task is not None and try_execute(idx, task):
            free.discard(idx)
'''

PASS3_REPL = '''    match_pass(P_FILLER, 99, lambda idx: zone_tasks.get(idx % n_zones, []))
    match_pass(P_FILLER, 99, lambda idx: all_tasks + global_tasks)
'''

ZONE_POOL = "zone_tasks.get(idx % n_zones, [])"
CROSS_POOL = "all_tasks + global_tasks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--passes", choices=("1", "all"), default="1",
                    help="1 = zone round only (61%% of movement); all = every pass")
    ap.add_argument("--cross-zone", action="store_true",
                    help="pass 1 matches against every task, not just the unit's zone")
    args = ap.parse_args()

    src = open(os.path.join(AGENTS, args.base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, ok = [], False
    for mod, text in mods:
        if mod == "scheduler":
            for name, find in (("helper", HELPER_FIND), ("pass 1", PASS1_FIND)):
                if find not in text:
                    raise RuntimeError(f"{name} anchor not found")
            pool1 = CROSS_POOL if args.cross_zone else ZONE_POOL
            text = text.replace(HELPER_FIND, HELPER_REPL, 1)
            text = text.replace(PASS1_FIND, PASS1_REPL.replace("__POOL1__", pool1), 1)
            if args.passes == "all":
                for name, find, repl in (("pass 2", PASS2_FIND, PASS2_REPL),
                                         ("pass 3", PASS3_FIND, PASS3_REPL)):
                    if find not in text:
                        raise RuntimeError(f"{name} anchor not found")
                    text = text.replace(find, repl, 1)
            k = text.index("def _dist(")
            text = text[:k] + MATCH_FN + "\n\n" + text[k:]
            ok = True
        out.append((mod, text))
    if not ok:
        raise RuntimeError("scheduler module not found")

    d = os.path.join(AGENTS, args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    print(f"wrote {p}  (passes={args.passes}, "
          f"{'cross-zone' if args.cross_zone else 'within-zone'})")


if __name__ == "__main__":
    main()
