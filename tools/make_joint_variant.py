"""E2: change the PLANNER and the EXECUTOR together, not one at a time.

Every measurement in this project is single-lever, and the commit that shelved
the demand book said so outright: "Closing the last $30-40k needs a jointly
different plan, not another knob."

The evidence that the two are coupled: placed on a foreign farm at foreign unit
positions, champ-h8's work share collapses 40.5% -> 12.4% -- and that farm is
NOT task-poor (our own build_tasks finds MORE tasks there, 98 vs 66 at day 10).
The executor is co-adapted to the layout our planner produces, which is the
mechanical reason every executor change in isolation gates flat.

PLANNER HALF -- the crops demand book (`BOOK_CROPS`). Chosen because it is the
only book that is not already negative on its own: crops +$231 +/- 3,186, against
herd -$6,637 and all-books -$6,909. Starting from a neutral planner change means
an executor gain does not have to pay off a deficit first. It sizes strawberry,
carrot and tomato targets to the day-6 shop draw, and mechanically it works --
carrots 37 -> 126 units, tomatoes 0 -> 42 -- but wheat falls 181 -> 100 because
it REALLOCATES fixed capacity instead of adding volume.

EXECUTOR HALF -- same-tile-first (`--exec tile`). The hypothesis with a
mechanism: `v-tile` failed alone (-$4,907) because the work it won underfoot was
low-value P_FILLER. The crops book replaces wheat with CARROTS, which run a
~2-day cycle, so a carrot tile repeatedly wants harvest -> plant -> water. Under
the book the underfoot work should be high-value, which is exactly the condition
v-tile needed and did not have.

    PYTHONPATH=. uv run python tools/make_joint_variant.py --name j-crops
    PYTHONPATH=. uv run python tools/make_joint_variant.py --name j-crops-tile --exec tile
"""

import argparse
import ast
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")

# champ-h8's frozen bundle PREDATES the demand book -- BOOK_CROPS, DEMAND_BOOK
# and _apply_demand_book are all absent, though the carrot_standing /
# tomato_standing knobs it sets are present. Rebuilding from the dev tree would
# change other things and break the controlled comparison, so inject just the
# book. SHOPS resolves from the shared bundle globals (constants defines it).
BOOK_FIND = """def _knobs(memory):
    k = dict(KNOB_DEFAULTS)
    k.update(getattr(memory, "knobs", None) or {})
    return k"""

BOOK_REPL = """# --- crops demand book (tools/make_joint_variant.py) ---
# Production targets sized to the day-6 shop draw, mined from 1,857 seats rated
# 2900+. Index = consumers of that product among the open shops, capped at two.
DEMAND_BOOK_CROPS = {
    "straw_target": (21, 29, 35),
    "carrot_standing": (4, 10, 15),
    "tomato_standing": (4, 9, 14),
}
BOOK_PRODUCT_CROPS = {
    "straw_target": "STRAWBERRY",
    "carrot_standing": "CARROT",
    "tomato_standing": "TOMATO",
}


def _consumer_count(view, item):
    return sum(1 for s in view.unlocked_shops if item in SHOPS.get(s, ()))


def _knobs(memory, view=None):
    k = dict(KNOB_DEFAULTS)
    # Knobs are decided once the shops are revealed at day 6.
    if view is not None and view.day >= 6:
        for key, row in DEMAND_BOOK_CROPS.items():
            k[key] = row[min(2, _consumer_count(view, BOOK_PRODUCT_CROPS[key]))]
    # Explicit bandit rules still win over the book.
    k.update(getattr(memory, "knobs", None) or {})
    return k"""

BOOK_CALL_FIND = "    knobs = _knobs(memory)"
BOOK_CALL_REPL = "    knobs = _knobs(memory, view)"

# Same-tile pass. Anchored on Pass 1 -- `claimed` and `try_execute` are defined
# just above it, and anchoring earlier (on the "Zone rounds" comment) raises
# NameError every turn for a silent $3,000 agent.
TILE_FIND = """    # Pass 1 — own zone, real work (below filler): animal deliveries first."""

TILE_REPL = """    # Pass 0 — finish the tile you are standing on, whatever band it is in.
    # assign() is band-major and walks workers off underfoot work on 73.6% of
    # their moves. Alone this LOST money (-$4,907) because the work it won was
    # P_FILLER; paired with the crops book the underfoot work is carrot
    # harvest/replant on a ~2-day cycle, which is the condition it needed.
    for idx in sorted(free):
        _ux, _uy = units[idx]
        _inv = view.unit_inv(idx)
        _best = None
        for _t in all_tasks + global_tasks:
            if _t.x != _ux or _t.y != _uy:
                continue
            if (_t.x, _t.y, _t.op[0]) in claimed:
                continue
            if _t.need and _inv.get(_t.need, 0) < 1:
                continue
            if _best is None or _t.pri < _best.pri:
                _best = _t
        if _best is not None and try_execute(idx, _best):
            free.discard(idx)
    _count_moves(actions, _seen, "sametile")
    _seen = set(actions)

    # Pass 1 — own zone, real work (below filler): animal deliveries first."""


def build(base, name, exec_mode="none", book=True):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    out, done = [], set()
    for mod, text in mods:
        if book and BOOK_FIND in text and BOOK_CALL_FIND in text:
            text = text.replace(BOOK_FIND, BOOK_REPL, 1)
            text = text.replace(BOOK_CALL_FIND, BOOK_CALL_REPL, 1)
            done.add("book")
        if exec_mode == "tile" and TILE_FIND in text:
            text = text.replace(TILE_FIND, TILE_REPL, 1)
            done.add("exec")
        out.append((mod, text))
    need = ({"book"} if book else set()) | ({"exec"} if exec_mode != "none" else set())
    if done != need:
        raise RuntimeError(f"anchors missing: {sorted(need - done)}")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    # Pre-flight: a bundle that raises on import compiles fine and then scores
    # exactly $3,000 while looking healthy.
    g = {"__name__": "preflight"}
    exec(compile(open(p).read(), p, "exec"), g)
    if not callable(g.get("agent")):
        raise RuntimeError("bundle exports no agent()")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--exec", dest="exec_mode", choices=("none", "tile"), default="none")
    ap.add_argument("--no-book", action="store_true")
    a = ap.parse_args()
    print(f"wrote {build(a.base, a.name, a.exec_mode, not a.no_book)}  "
          f"(book={not a.no_book}, exec={a.exec_mode})")
