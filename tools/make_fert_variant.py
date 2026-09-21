"""Fertilise melons. The planner has no clause for the most valuable crop.

`tools/rule_tests.py` scores `FERTILIZE.one_time` at **20% capability / 0%
rigid** on every executor built so far -- the worst rule in the suite and the
only one no executor change touches. Reading the ten mined cases says why:

    5 x MELON   ages 6, 6, 7, 8, 6   <- `want_fert` has NO melon clause
    4 x WHEAT   age 2 each           <- clause exists, blocked by fert_price < 40
    1 x CARROT  age 2                <- clause exists, fired (day 24, fert cheap)

So half the gap is a missing branch, not a mis-tuned one.

WHAT THE RULES SAY IT IS WORTH. README Harvest Yields: a fertilised one-time
crop adds **2** units per watered day instead of 1. Melon's bonus window is
ages 6-12 and its cap is 6, so unfertilised it needs **5** watered days
(ages 6-10) and fertilised only **3** (1 + 2x3 = 7, capped at 6, by age 8).

It does NOT free the tile early: `first_yield_day` is 10 and the engine
refuses HARVEST before then (kaggriculture.py:453), so reaching the cap at
age 8 buys nothing in time. What it buys is **insurance**: our in-window
watering is 92%, so all five days land only ~0.92^5 = 66% of the time, and a
melon short of the cap loses a unit worth up to $250. Three days needed
instead of five turns that into ~0.92^3 = 78%.

That is a real but modest edge, which is why it is gated and not assumed --
and why the WHEAT guard is deliberately left alone. A fertilised wheat gains
+2 units at a $25 base, i.e. ~$50 of wheat for a $100 asset; the reference
seats doing it anyway is exactly the observational evidence this project has
been burned by four times.

    PYTHONPATH=. uv run python tools/make_fert_variant.py --name p1-mel100 \\
        --base h12-ufh --price 100
"""
import argparse
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIND = '''                (t["crop"] == "STRAWBERRY" and 9 <= age <= 15 and fert_price < 150)'''

REPL = '''                # README Harvest Yields: a fertilised one-time crop adds 2
                # units per watered day, so melon reaches its cap of 6 in 3
                # watered days instead of 5. first_yield_day 10 means that
                # does not free the tile sooner -- it insures the cap against
                # a missed watering, on the crop with the highest base price.
                (t["crop"] == "MELON" and 6 <= age <= 8 and fert_price < __PRICE__)
                or (t["crop"] == "STRAWBERRY" and 9 <= age <= 15 and fert_price < 150)'''


def build(base, name, price):
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
            text = text.replace(FIND, REPL.replace("__PRICE__", str(price)), 1)
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no scheduler module in the bundle")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    rules = os.path.join(REPO, "arena", "agents", base, "rules.json")
    if os.path.isfile(rules):
        import shutil
        shutil.copyfile(rules, os.path.join(dst_dir, "rules.json"))
    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  "
          f"melon fert at fert_price < {price}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="h12-ufh")
    ap.add_argument("--name", required=True)
    ap.add_argument("--price", type=int, default=100)
    args = ap.parse_args()
    build(args.base, args.name, args.price)


if __name__ == "__main__":
    main()
