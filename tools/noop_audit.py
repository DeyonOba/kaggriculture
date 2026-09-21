"""How many of our emitted ops does the engine silently throw away, and why?

README Actions: "Invalid actions are silent no-ops." The engine never
complains -- `_apply_unit_action` returns early and the turn is simply gone.
A unit-turn spent on a no-op is indistinguishable from a PASS in every metric
we have, so this is the one class of waste none of the other instruments can
see.

Locked ground is the case worth stating precisely (kaggriculture.py:325-345,
README Movement):

  movement          allowed onto and across LOCKED tiles -- blocking it would
                    strand a hand spawned on a locked shed-access tile
  PICKUP/DROP/PLACE resolve BEFORE the locked guard; they use the tile only as
                    a standing position, so they work from a locked shed tile
  everything else   `if tile == "LOCKED": return` -- silent no-op, consumes
                    the turn, changes nothing

So "a non-movement task on a locked quadrant is ignored" is exactly right,
and it costs the unit-turn.

Two things are invisible to this audit by construction and are counted
separately from the action dict instead:
  - atomic PLANT: if a turn requests more PLANTs of a crop than seeds held,
    the interpreter rewrites ALL of them to PASS before the engine sees them
  - market orders past `maxMarketOrdersPerTurn` are dropped before parsing

    PYTHONPATH=. uv run python tools/noop_audit.py --agent champ-h8 --seeds 4
"""
import argparse
import collections
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import rule_harness as H                     # noqa: E402
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
SHED_OPS = ("PICKUP", "DROP", "PLACE")
ANIMALS = ("GOOSE", "COW", "SHEEP")


def changed(r):
    return (r["tile_before"] != r["tile_after"]
            or r["inv_before"] != r["inv_after"]
            or r["shed_before"] != r["shed_after"]
            or r["seeds_before"] != r["seeds_after"]
            or r["pos_after"] != [r["x"], r["y"]])


def why(r):
    """Cause of a no-op, read off the pre-state the engine saw."""
    op, t = r["op"][0], r["tile_before"]
    inv, shed = r["inv_before"], r["shed_before"]
    locked = t == "LOCKED"
    isd = isinstance(t, dict)
    if op in MOVES:
        return "off_board"
    if op in SHED_OPS:
        # shed ops resolve before the locked guard, so LOCKED is never the cause
        adj = (r["x"], r["y"]) in {(4, 4), (5, 4), (4, 5), (5, 5)}
        if not adj and not (op == "PLACE" and isd):
            return "not_shed_adjacent"
        if op == "PICKUP":
            item = r["op"][1] if len(r["op"]) > 1 else None
            return "shed_lacks_item" if not shed.get(item) else "unknown"
        if op == "DROP":
            if not sum(inv.values()):
                return "empty_inventory"
            return "shed_full"
        item = r["op"][1] if len(r["op"]) > 1 else None
        if not inv.get(item):
            return "item_not_in_inventory"
        return "shed_full"
    if locked:
        return "LOCKED_TILE"
    if op == "PLANT":
        crop = r["op"][1] if len(r["op"]) > 1 else None
        if t is not None:
            return "tile_occupied"
        return "no_seed" if not r["seeds_before"].get(crop) else "unknown"
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return "tile_occupied" if t is not None else "unknown"
    if op == "DIG":
        if t is None:
            return "tile_already_empty"
        return "structure_has_animal" if isd and "animal" in t else "unknown"
    plant = isd and t.get("kind") == "PLANT"
    animal = isd and "animal" in t
    if op == "WATER":
        if not plant:
            return "not_a_plant"
        return "already_watered" if t.get("watered_today") else "unknown"
    if op == "FERTILIZE":
        if not plant:
            return "not_a_plant"
        return "no_fertilizer" if not inv.get("FERTILIZER") else "unknown"
    if op == "HARVEST":
        if not isd:
            return "nothing_here"
        return "zero_yield" if t.get("yield_units", 0) <= 0 else "immature"
    if op == "FEED":
        if not animal:
            return "not_an_animal"
        if t.get("fed_today"):
            return "already_fed"
        return "no_wheat" if not inv.get("WHEAT") else "unknown"
    if op == "CARE":
        if not animal:
            return "not_an_animal"
        return "already_cared" if t.get("cared_today") else "unknown"
    if op == "COLLECT_FERTILIZER":
        if not animal:
            return "not_an_animal"
        return "none_available" if not t.get("fertilizer_available") else "unknown"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="champ-h8")
    ap.add_argument("--opp", default="pub-ahmed-v38")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=71000)
    args = ap.parse_args()

    from kaggle_environments import make
    a = os.path.join(REPO, "arena", "agents", args.agent, "main.py")
    b = os.path.join(REPO, "arena", "agents", args.opp, "main.py")

    tot = collections.Counter()
    noop_by_op = collections.Counter()
    cause = collections.Counter()
    over_plant = [0]
    dropped_orders = [0]
    depth = collections.Counter()
    depth_move = collections.Counter()

    for k in range(args.seeds):
        seed = args.base_seed + k
        rec = H.Recorder(0, record_all=True)
        rec.install()
        try:
            env = make("kaggriculture",
                       configuration={"seed": seed, "episodeSteps": 720},
                       debug=False)
            env.run([a, b])
        finally:
            rec.uninstall()
        grid_locked = set()
        for r in rec.unit_ops:
            op = r["op"][0]
            tot[op] += 1
            if r["tile_before"] == "LOCKED":
                # README Movement: locked tiles are passable but every tile
                # action no-ops on them. Standing on one is only legitimate
                # for a move, a PASS, or a shed op from a shed-access tile.
                key = ("move_from_locked" if op in MOVES
                       else "shed_op_from_locked" if op in SHED_OPS
                       else "PASS_on_locked" if op == "PASS"
                       else "TILE_OP_ON_LOCKED")
                grid_locked.add(key)
                cause[f"standing_on_locked:{key}"] += 1
                depth[r.get("locked_depth", 0)] += 1
                if op in MOVES:
                    depth_move[r.get("locked_depth", 0)] += 1
            if op == "PASS":
                continue
            if not changed(r):
                noop_by_op[op] += 1
                cause[f"{op}:{why(r)}"] += 1
        # Action-level checks the engine hides. Alignment matters and is easy
        # to get wrong (tools/build_sessions.py documents it): steps[t]'s
        # action is the decision taken while observing state t-1, and
        # steps[t]'s observation is the state AFTER it. So the seeds the agent
        # actually held are steps[t-1]'s, not steps[t]'s.
        for t in range(1, len(env.steps)):
            act = env.steps[t][0].get("action") or {}
            if not isinstance(act, dict):
                continue
            ops = [act.get("farmer")] + list(act.get("hands") or [])
            want = collections.Counter(
                o[1] for o in ops
                if isinstance(o, list) and len(o) > 1 and o[0] == "PLANT")
            priv = env.steps[t - 1][0].observation.private
            seeds = priv.get("seeds", {}) if hasattr(priv, "get") else {}
            for crop, n in want.items():
                if n > seeds.get(crop, 0):
                    over_plant[0] += n
            dropped_orders[0] += max(0, len(act.get("market") or []) - 10)

    acted = sum(v for kk, v in tot.items() if kk != "PASS")
    noops = sum(noop_by_op.values())
    print(f"{args.agent} vs {args.opp}, {args.seeds} games\n")
    print(f"unit-turns            {sum(tot.values()):>8,}")
    print(f"  PASS                {tot['PASS']:>8,}  "
          f"({tot['PASS']/max(1,sum(tot.values())):.1%})")
    print(f"  ops emitted         {acted:>8,}")
    print(f"  of those, NO-OPS    {noops:>8,}  ({noops/max(1,acted):.1%} of ops, "
          f"{noops/max(1,sum(tot.values())):.1%} of the turn budget)\n")
    print(f"{'op':22s}{'emitted':>9}{'no-op':>8}{'rate':>8}")
    for op in sorted(tot, key=lambda o: -noop_by_op[o]):
        if op == "PASS":
            continue
        print(f"{op:22s}{tot[op]:>9,}{noop_by_op[op]:>8,}"
              f"{noop_by_op[op]/max(1,tot[op]):>8.1%}")
    print(f"\n{'cause':44s}{'count':>8}")
    for c, n in cause.most_common():
        print(f"{c:44s}{n:>8,}")
    if depth:
        tot_d = sum(depth.values())
        print(f"\nunit-turns spent standing on LOCKED ground, by depth "
              f"(1 = on the border):")
        print(f"{'depth':>7}{'all ops':>10}{'of those moves':>16}{'share':>8}")
        for d in sorted(depth):
            print(f"{d:>7}{depth[d]:>10,}{depth_move[d]:>16,}"
                  f"{depth[d]/tot_d:>8.1%}")
        over = sum(n for d, n in depth.items() if d > 1)
        print(f"  deeper than 1 tile in: {over:,} ({over/tot_d:.1%})")
    print(f"\nhidden by the interpreter (counted from the action dict):")
    print(f"  PLANT requests voided by the atomic-seed rule   {over_plant[0]:>6,}")
    print(f"  market orders dropped past maxMarketOrdersPerTurn {dropped_orders[0]:>4,}")


if __name__ == "__main__":
    main()
