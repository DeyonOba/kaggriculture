"""Take the work underfoot, regardless of zone, below P_FILLER.

Measured cause of the 28 unobeyed tile commands (`docs/animal_chain_audit.md`
Part 7): pass 1 searches `zone_tasks[idx % n_zones]` -- the unit's OWN zone
only -- so a unit physically standing on a task tile is refused it when the
tile belongs to another worker's zone, and pass 1 hands it something back in
its own zone instead. `ASSIGN_LABEL` puts 68% of diversions in that stage and
the tile is in a foreign zone 65% of the time.

This inserts one check between the delivery branch and the own-zone branch of
pass 1: if there is an unclaimed task on the tile I am standing on, and it is
real work, do it.

Two things it deliberately does NOT do, both because `v-tile` -- "a same-tile
pass before the band passes" -- already gated **-$4,907**:

  * it does not run before the delivery branch, so a carrier holding an animal
    still delivers first (`deliver.PLACE` is at 100% and must stay there)
  * `--max-pri` bounds which bands qualify. The default P_FILLER (5) is "any
    real work", which is the narrowing v-tile lacked: v-tile grabbed FILLER
    too, and filler underfoot is exactly the low-value work that four
    throughput experiments have shown converts at zero.

    PYTHONPATH=. uv run python tools/make_underfoot_variant.py --name h11-uf
    PYTHONPATH=. uv run python tools/make_underfoot_variant.py --name h11-uf2 \
        --max-pri 2
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIND = '''    # Pass 1 — own zone, real work (below filler): animal deliveries first.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)'''

REPL = '''    # Work underfoot, regardless of whose zone the tile belongs to. `pick`
    # below only searches zone_tasks[idx % n_zones], and measured, the unit
    # standing on an unserved task tile is in a FOREIGN zone 65% of the time,
    # so it is refused the task and walks back to its own zone.
    _uf = {}
    for _t in all_tasks + global_tasks:
        if _t.pri < __MAXPRI__:
            _uf.setdefault((_t.x, _t.y), []).append(_t)

    def _underfoot(idx):
        _x, _y = units[idx]
        _inv = view.unit_inv(idx)
        _best = None
        for _t in _uf.get((_x, _y), ()):
            if (_t.x, _t.y, _t.op[0]) in claimed:
                continue
            if _t.need and _inv.get(_t.need, 0) < 1:
                continue
            if _best is None or _t.pri < _best.pri:
                _best = _t
        return _best

__HOIST__    # Pass 1 — deliveries, then the tile underfoot, then the unit's own zone.
    for idx in sorted(free):
        inv = view.unit_inv(idx)
        deliveries = [t for t in need_tasks if inv.get(t.need, 0) >= 1]
        task = pick(idx, deliveries, P_SURVIVAL, P_FILLER) if deliveries else None
        if task is None:
            task = _underfoot(idx)
        if task is None:
            task = pick(idx, zone_tasks.get(idx % n_zones, []), P_SURVIVAL, P_FILLER)
        if task is not None and try_execute(idx, task):
            free.discard(idx)'''


HOIST = """    # Pass 0 — the tile underfoot, ALL units, before anyone picks by zone.
    # Doing this inside pass 1's per-unit loop is not enough: `sorted(free)`
    # lets a DISTANT unit reach the task first through its own-zone pick,
    # claim it (`claimed` is set on the step TOWARD a task, not on arrival),
    # and the unit already standing on the tile is then refused. Measured on
    # the DIG residue: a unit stood on the weed at T, walked off, and the weed
    # was dug 14-37 turns later. Carriers are skipped so an animal in a pocket
    # still reaches its structure first.
    _hold = {i for i, _, _ in me.units
             if any(view.unit_inv(i).get(_a, 0)
                    for _a in ("GOOSE", "COW", "SHEEP"))}
    for idx in sorted(free - _hold):
        _t = _underfoot(idx)
        if _t is not None and try_execute(idx, _t):
            free.discard(idx)

"""


def build(base, name, max_pri, hoist=False):
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if text.count(FIND) != 1:
                raise SystemExit(f"anchor not unique ({text.count(FIND)})")
            repl = (REPL.replace("__MAXPRI__", repr(max_pri))
                        .replace("__HOIST__", HOIST if hoist else ""))
            text = text.replace(FIND, repl, 1)
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
          f"max_pri={max_pri} hoist={hoist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hoist", action="store_true",
                    help="run the underfoot check as its own pass before "
                         "pass 1, so a unit ON a task always claims it first")
    ap.add_argument("--max-pri", type=float, default=5.0,
                    help="a task qualifies if pri < this (P_FILLER = 5)")
    args = ap.parse_args()
    build(args.base, args.name, args.max_pri, args.hoist)


if __name__ == "__main__":
    main()
