"""Refuse a shed trip that carries too little to be worth the walk.

Measured, champ-h8 over 6 games:

    WHEAT       246 pickups   n=1:64  n=2:90  n=3:28  n=4:20  n=5:42
    FERTILIZER   91 pickups   n=1:47  n=2:29  n=3:14
    -> 618 wheat picked up to perform 329 FEEDs, and 111 of the 353 pickups
       carry a SINGLE item.

The corpus says every rated band hauls about half as much as we do --
pickup-per-feed 0.52-0.63 against our 1.07 -- while FEEDing the same amount
(316-338 vs our 329). So it is not that they feed less; it is that they make
fewer trips.

Batch SIZE is not the lever and is already implemented (`batch = 5 if item ==
"WHEAT"`, `n = min(need, shed_left, batch)`). Raising the cap cannot reduce the
trip COUNT. Singleton trips can: a pickup of one item costs the same two
moves as a pickup of five.

So this sets a floor -- skip the trip unless it carries `--min-batch` items --
with two escapes that keep it safe:

  * late in the day, when there is no time left to accumulate a bigger need
  * when the shed simply has no more of the item, so waiting cannot help

`tools/haul_probe.py` must show pickup-per-feed actually moving. Archive
attempt #7 ("bigger shed pickup batches -- null") and #11 ("mechanism does not
fire") are the warning: a gate on a mechanism that never fired says nothing.

    PYTHONPATH=. uv run python tools/make_haul_variant.py --name x-haul2 --min-batch 2
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = '''            n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)
            if n <= 0:
                continue'''

REPL = '''            n = min(need, shed_left[item], 5 if item == "WHEAT" else 3)
            if n <= 0:
                continue
            # A one-item trip costs the same two walks as a five-item trip.
            # Hold out for a bigger load unless the day is ending or the shed
            # is empty of this item, in which case waiting cannot help.
            if (n < _HAUL_MIN
                    and view.hours_left_today > _HAUL_LATE
                    and shed_left[item] > n):
                continue'''

GLUE = '''

# --- haul floor -------------------------------------------------------------
# champ-h8 makes 353 pickups a game against the corpus's 164-215, and 111 of
# them carry a single item. Trips, not batch size, are the cost.
_HAUL_MIN = __MIN__
_HAUL_LATE = __LATE__

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--min-batch", type=int, required=True,
                    help="skip a shed trip carrying fewer than this")
    ap.add_argument("--late", type=int, default=6,
                    help="hours left below which the floor is waived")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "scheduler":
            if ANCHOR not in text:
                raise SystemExit("zone-provisioning anchor not found in scheduler")
            text = (GLUE.replace("__MIN__", str(args.min_batch))
                        .replace("__LATE__", str(args.late))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no scheduler module in bundle")

    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base} + haul floor "
          f"(min batch {args.min_batch}, waived under {args.late}h left)")


if __name__ == "__main__":
    main()
