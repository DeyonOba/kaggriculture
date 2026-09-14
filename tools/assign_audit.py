"""How do the top-rated agents assign workers? Structure, not totals.

We already know their aggregate numbers (46% productive turns to our 42%). This
looks at the SHAPE of the assignment: does a worker hold a contiguous patch for
the day the way our zone executor does, does it specialise on animals or crops,
how far does it roam, and how much of its day is one long route versus many
short hops.
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments import make  # noqa: E402
from rl.rollout import make_script_agent  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
ANIMAL_OPS = {"FEED", "CARE", "COLLECT_FERTILIZER"}
CROP_OPS = {"WATER", "PLANT", "FERTILIZE", "DIG"}


def analyse(steps, seat, label, acc):
    """Per worker per day: tiles touched, spread, specialisation, route length."""
    day_units = defaultdict(lambda: defaultdict(list))   # day -> unit -> [(x,y,op)]
    for i, st in enumerate(steps):
        obs = st[seat]["observation"]
        if "farms" not in obs:
            continue
        farm = obs["farms"][seat]
        units = [farm.get("farmer")] + list(farm.get("hands") or [])
        a = st[seat].get("action") or {}
        ops = [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])
        for u, pos in enumerate(units):
            if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
                continue
            op = ops[u][0] if u < len(ops) and isinstance(ops[u], list) and ops[u] else "PASS"
            day_units[i // 24][u].append((pos[0], pos[1], op))
    for day, us in day_units.items():
        if day < 8 or day > 24:
            continue
        for u, seq in us.items():
            work = [(x, y, o) for x, y, o in seq if o not in MOVES and o != "PASS"]
            if not work:
                continue
            tiles = {(x, y) for x, y, _ in work}
            xs = [x for x, _, _ in work]
            ys = [y for _, y, _ in work]
            spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
            an = sum(1 for _, _, o in work if o in ANIMAL_OPS)
            cr = sum(1 for _, _, o in work if o in CROP_OPS)
            acc[label]["worker_days"] += 1
            acc[label]["actions"] += len(work)
            acc[label]["tiles"] += len(tiles)
            acc[label]["spread"] += spread
            acc[label]["moves"] += sum(1 for _, _, o in seq if o in MOVES)
            if an + cr:
                acc[label]["spec"] += max(an, cr) / (an + cr)
            acc[label]["animal_only"] += 1 if cr == 0 and an > 0 else 0
            acc[label]["crop_only"] += 1 if an == 0 and cr > 0 else 0


def main():
    top = json.load(open("/tmp/top10.json"))
    acc = defaultdict(lambda: defaultdict(float))
    for path, seat, rating in top:
        s = json.load(open(path))
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s["seed"]}, debug=False)
        env.run([make_script_agent(s["actions"][0]), make_script_agent(s["actions"][1])])
        analyse(env.steps, seat, "top10", acc)
    for k in range(4):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 31000 + k}, debug=False)
        env.run(["arena/agents/champ-h8/main.py", "arena/agents/pub-ahmed-v38/main.py"])
        analyse(env.steps, 0, "champ-h8", acc)
    print(f"{'':<12} {'wkr-days':>9} {'actions/day':>12} {'tiles/day':>10} {'spread':>7} "
          f"{'moves/day':>10} {'specialisation':>14} {'animal-only':>12} {'crop-only':>10}")
    for lab in ("top10", "champ-h8"):
        d = acc[lab]
        n = max(1.0, d["worker_days"])
        print(f"{lab:<12} {d['worker_days']:>9,.0f} {d['actions']/n:>12.2f} {d['tiles']/n:>10.2f} "
              f"{d['spread']/n:>7.2f} {d['moves']/n:>10.2f} {d['spec']/n:>14.2f} "
              f"{100*d['animal_only']/n:>11.0f}% {100*d['crop_only']/n:>9.0f}%")
    print("\nspecialisation = share of a worker-day's actions in its dominant class (0.5 mixed, 1.0 pure)")
    print("spread = bounding-box size of the tiles one worker touched that day")


if __name__ == "__main__":
    main()
