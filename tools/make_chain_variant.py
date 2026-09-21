"""Correct three README-contradicting assumptions in the animal chain.

The chain the README specifies is BUY_ANIMAL (-> shed, counts against
shedCapacity, refused when the shed is full) -> PICKUP n (-> a unit's pocket)
-> walk -> PLACE on a matching EMPTY structure (takes exactly ONE, IGNORES n).
champ-h8's executor gets three things about it wrong.

FA-1/FA-2  `planner.make_champion_plan`
    pipeline = plan.animal_buys[kind] + view.shed[kind]
  `animal_buys` is a REQUEST the market controller has not placed yet. README
  Market Action + engine `_commit_unit`: BUY_ANIMAL is refused with no money
  AND when the shed is at shedCapacity. So a coop gets committed for a goose
  we never buy -- and once any animal lands on a structure, README Terrain
  says DIG can no longer remove it, so the tile is spent. The same expression
  omits animals already in a carrier's pocket, which `scheduler.build_tasks`
  DOES count in `available`, so a pocketed animal can be handed a PLACE task
  with no structure left for it.
    measured, g-goose6 vs pub-ahmed-v38, 3 seeds: 20 of 24 GOOSE orders
    filled; 390 idle-coop-turns per game (an empty coop, no goose anywhere).

FA-3  `scheduler.assign`, global provisioning
    batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
  README Animals: "PLACE ... places one animal from inventory onto the tile.
  The `n` argument is ignored." So animal #2 in a pocket is dead weight until
  its carrier reaches a SECOND free structure, and README Shed says the
  midnight inventory drop returns it to the shed -- or discards it outright
  when the shed is full. The comment four lines above already says "any unit
  may carry ONE to its pasture"; only the constant disagrees.
    measured, g-goose6, 3 seeds: 11 double-carries, 7 animals picked up and
    never placed (3 COW / 2 GOOSE / 2 SHEEP bounced back at midnight).

FA-4  `scheduler.assign`, shed-overflow drop
    item = max(inv, key=inv.get)   ->   ["PLACE", item, n]
  The same block builds `total` by EXCLUDING GOOSE/COW/SHEEP, then picks the
  item over the whole inventory, so it can name an animal. README: PLACE
  resolves the animal-placement branch FIRST when the unit stands on a
  matching empty structure -- one animal onto the tile, nothing into the shed,
  and the following `shed_room -= n` is then wrong. Reachable because the
  build queue sorts empties by distance to the shed and the four shed-access
  tiles tie for the minimum (measured: 9 of 46 structures built on them).
  Latent in the games measured -- shipped as a guard, expected null.

    PYTHONPATH=. uv run python tools/make_chain_variant.py --name h9-chain
    PYTHONPATH=. uv run python tools/make_chain_variant.py --name h9-carry \
        --fixes carry
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ALL_FIXES = ("build", "carry", "place")

# --- FA-1 + FA-2: build structures for animals we OWN -----------------------
# Anchored inside make_champion_plan (the champion-plan path champ-h8 runs);
# the identical expression in the unused reactive make_day_plan is left alone.
BUILD_FIND = '''    plan.animal_buys = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    if not plan.liquidation:
        # Full deficits, not gated on hour-0 cash: the market controller buys
        # animals before seeds the turn money lands (wool/milk arrive mid-day),
        # so the herd wins the intra-day cash race the way champions' does.
        targets = _champ_herd_targets(day, knobs)
        for kind in ("SHEEP", "COW", "GOOSE"):  # wool lands first (day 6) and pays for land
            want = targets[kind] - herd.get(kind, 0) - view.shed.get(kind, 0)
            n = max(0, min(want, 4))
            if n > 0:
                plan.animal_buys[kind] = n
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = plan.animal_buys.get(kind, 0) + view.shed.get(kind, 0)'''

BUILD_REPL = '''    plan.animal_buys = {}
    plan.structure_builds = {"COOP": 0, "PASTURE": 0}
    # An animal in a carrier's pocket is bought, alive and homeless, and both
    # lines below used to look straight past it -- they read `herd + shed`.
    # It is counted on the other side of the chain (`scheduler.build_tasks`
    # puts shed + pockets in `available` and hands it a PLACE task), so the
    # two halves disagree: the buy target treats the head as still missing and
    # orders a replacement, and the structure pipeline never queues it a
    # coop. README Terrain: DIG cannot recover a structure once an animal is
    # on it, so the tile commitment has to be right the first time.
    _pocket = {}
    for _i, _px, _py in me.units:
        for _it, _n in view.unit_inv(_i).items():
            _pocket[_it] = _pocket.get(_it, 0) + _n
    if not plan.liquidation:
        # Full deficits, not gated on hour-0 cash: the market controller buys
        # animals before seeds the turn money lands (wool/milk arrive mid-day),
        # so the herd wins the intra-day cash race the way champions' does.
        targets = _champ_herd_targets(day, knobs)
        for kind in ("SHEEP", "COW", "GOOSE"):  # wool lands first (day 6) and pays for land
            want = (targets[kind] - herd.get(kind, 0) - view.shed.get(kind, 0)
                    - _pocket.get(kind, 0))
            n = max(0, min(want, 4))
            if n > 0:
                plan.animal_buys[kind] = n
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = (plan.animal_buys.get(kind, 0) + view.shed.get(kind, 0)
                    + _pocket.get(kind, 0))'''

# --- FA-3: one animal per carrier -------------------------------------------
CARRY_FIND = '''        batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 2)
        while shortage > 0 and free:
            idx = min(
                free,
                key=lambda i: _dist(units[i][0], units[i][1],
                                    *_nearest_shed_tile(units[i][0], units[i][1])),
            )'''

CARRY_REPL = '''        # README Animals: PLACE onto a matching structure takes exactly ONE
        # and IGNORES n, so a second animal is dead weight until its carrier
        # reaches a second free structure -- and README Shed's midnight drop
        # returns it to the shed, or discards it when the shed is full. One
        # head per carrier: batch 1, and never hand a second animal to a unit
        # that is already holding one (the dispatcher picks the unit nearest
        # the shed, which is exactly the one that just picked up).
        batch = 5 if item == "WHEAT" else (3 if item == "FERTILIZER" else 1)
        pool = free
        if item in ("GOOSE", "COW", "SHEEP"):
            pool = {i for i in free
                    if not any(view.unit_inv(i).get(a, 0)
                               for a in ("GOOSE", "COW", "SHEEP"))}
        while shortage > 0 and pool:
            idx = min(
                pool,
                key=lambda i: _dist(units[i][0], units[i][1],
                                    *_nearest_shed_tile(units[i][0], units[i][1])),
            )
            pool = pool - {idx} if pool is not free else pool'''

# --- FA-4: never name an animal in the shed-overflow drop -------------------
PLACE_FIND = '''            else:
                item = max(inv, key=inv.get)
                n = min(inv[item], shed_room)
                if n <= 0:
                    continue'''

PLACE_REPL = '''            else:
                # README: PLACE resolves the animal-placement branch FIRST
                # when the unit stands on a matching empty structure, so
                # naming an animal here puts ONE animal on the tile and
                # nothing in the shed. Ship only what `total` counted as
                # cargo -- animals were deliberately excluded from it.
                cargo = {k: v for k, v in inv.items()
                         if k not in ("GOOSE", "COW", "SHEEP")}
                if not cargo:
                    continue
                item = max(cargo, key=cargo.get)
                n = min(cargo[item], shed_room)
                if n <= 0:
                    continue'''

PATCHES = {
    "build": ("planner", BUILD_FIND, BUILD_REPL),
    "carry": ("scheduler", CARRY_FIND, CARRY_REPL),
    "place": ("scheduler", PLACE_FIND, PLACE_REPL),
}


def build(base, name, fixes):
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head = src[:i]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    tail = src[j:]

    applied = []
    out = []
    for mod, text in mods:
        for fix in fixes:
            target, find, repl = PATCHES[fix]
            if mod != target:
                continue
            if find not in text:
                raise SystemExit(
                    f"anchor for fix '{fix}' not found in module '{mod}' of {base}"
                )
            if text.count(find) != 1:
                raise SystemExit(
                    f"anchor for fix '{fix}' is ambiguous in '{mod}' "
                    f"({text.count(find)} matches)"
                )
            text = text.replace(find, repl, 1)
            applied.append(fix)
        out.append((mod, text))

    if sorted(applied) != sorted(fixes):
        raise SystemExit(f"applied {applied}, wanted {fixes}")

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
    print(f"wrote {dst}  ({os.path.getsize(dst) / 1024:.0f} KiB)  fixes={','.join(fixes)}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fixes", default=",".join(ALL_FIXES),
                    help=f"comma-separated subset of {ALL_FIXES}")
    args = ap.parse_args()
    fixes = [f.strip() for f in args.fixes.split(",") if f.strip()]
    bad = [f for f in fixes if f not in PATCHES]
    if bad:
        raise SystemExit(f"unknown fixes {bad}; choose from {ALL_FIXES}")
    build(args.base, args.name, fixes)


if __name__ == "__main__":
    main()
