"""Convert harvested Kaggle replays of top agents into BC training samples.

Each replay stores BOTH agents' full observations (including private shed and
seeds) and actions at every step, so for every episode we extract TWO
imitation streams: at each day boundary, the champion's exact feature vector
(from their own observation) paired with the standing targets their play
implies (end-of-day crop counts, herd, hires, land) quantized onto our action
heads.

Usage:
    PYTHONPATH=. uv run python rl/replay_bc.py --replays replays/top \
        --min-reward 40000 --out rl/datasets/champions.jsonl
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagent import features  # noqa: E402
from kagent.gamestate import Memory, View  # noqa: E402

TURNS = 24


def _nearest_valid(vals, mask, target):
    valid = [i for i in range(len(vals)) if mask[i]]
    return min(valid, key=lambda i: abs(vals[i] - target))


def _counts(farm_tiles):
    crops, herd = {}, {}
    for row in farm_tiles:
        for t in row:
            if isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    crops[t["crop"]] = crops.get(t["crop"], 0) + 1
                elif t.get("animal"):
                    herd[t["animal"]] = herd.get(t["animal"], 0) + 1
    return crops, herd


def _targets_to_choices(view, masks, crops, herd, shed, hires, bought_land, fert=0):
    anchor_vals = features.anchors(view)
    choices = []
    for (name, vals), mask in zip(features.HEADS, masks):
        anchored = False
        if name in features.QUOTA_CROP:
            target = crops.get(features.QUOTA_CROP[name], 0)
            anchored = name in features.ANCHORED
        elif name in features.BUY_KIND:
            kind = features.BUY_KIND[name]
            target = herd.get(kind, 0) + shed.get(kind, 0)
            anchored = name in features.ANCHORED
        elif name == "hires":
            target = hires
        elif name == "land":
            target = 1 if bought_land else 0
        elif name == "fert_budget":
            target = fert                  # v5: FERTILIZE actions per day, from the replay
        elif name == "melon_floor":
            target = 80  # unobservable; neutral default
        elif name.endswith("_floor_scale"):
            target = 1.0
        else:
            target = vals[0]
        # PARAMETERISE THE TARGET THE WAY THIS HEAD SET EXPECTS IT. `target` is
        # an absolute count above; the grid it is quantised onto is not always
        # in those units, and a mismatch is silent -- at v3 the grid is
        # [0, .25, .5, .75, 1, 1.5] and an absolute target of 2 or more landed
        # on 1.5 for every crop and animal, i.e. the corpus taught "always ask
        # for 1.5x the anchor".
        if features.HEADS_VERSION >= 4:
            pass                       # the grid IS the integers: exact label
        elif features.HEADS_VERSION == 3 and name in features.FRACTION_HEADS:
            a = anchor_vals.get(name) or 0
            target = (float(target) / a) if a else 0.0
        elif anchored:
            target -= anchor_vals[name]           # v1/v2 residual encoding
        choices.append(_nearest_valid(vals, mask, target))
    return choices


def extract_side(replay, side, min_reward):
    steps = replay["steps"]
    final_reward = steps[-1][side].get("reward") or 0
    if final_reward < min_reward:
        return []

    samples = []
    memory = Memory()
    melons_sold = 0
    prev_pass_frac = 0.0
    prev_hands_peak = 0

    n_days = min(30, (len(steps) + TURNS - 1) // TURNS)
    for d in range(n_days):
        t0 = d * TURNS
        t_end = min(d * TURNS + TURNS - 1, len(steps) - 1)
        if t0 >= len(steps):
            break
        obs = dict(steps[t0][side].get("observation") or {})
        if "farms" not in obs:
            continue
        obs.setdefault("step", t0)  # agent 1's stored obs omits `step`

        # Feed the memory the utilisation/sales signals reconstructed from
        # the champion's own recorded actions of the PREVIOUS day.
        memory.yesterday_idle = prev_pass_frac
        memory.yesterday_hands = prev_hands_peak
        memory.melons_sold = melons_sold

        view = View(obs)
        x = features.encode(view, memory)
        m = features.masks(view)

        end_obs = steps[t_end][side].get("observation") or {}
        if "farms" not in end_obs:
            continue
        my_farm_end = end_obs["farms"][side]
        crops, herd = _counts(my_farm_end["tiles"])
        shed = (end_obs.get("private") or {}).get("shed") or {}
        hires = my_farm_end.get("hires_today", 0)
        bought_land = len(my_farm_end.get("unlocked_quadrants", [])) > len(
            obs["farms"][side].get("unlocked_quadrants", [])
        )
        choices = _targets_to_choices(view, m, crops, herd, shed, hires, bought_land)
        samples.append({"x": x, "masks": m, "choices": choices})

        # Roll the day's action stats for tomorrow's memory features.
        ops = passes = 0
        hands_peak = 0
        for t in range(t0, t_end + 1):
            act = steps[t][side].get("action") or {}
            day_ops = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
            hands_peak = max(hands_peak, len(act.get("hands") or []))
            for op in day_ops:
                ops += 1
                if not op or op[0] == "PASS":
                    passes += 1
            for order in act.get("market") or []:
                if order and order[0] == "SELL" and len(order) > 1 and order[1] == "MELON":
                    melons_sold += int(order[2]) if len(order) > 2 else 1
        prev_pass_frac = passes / ops if ops else 0.0
        prev_hands_peak = hands_peak
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="replays/top")
    ap.add_argument("--min-reward", type=float, default=40000)
    ap.add_argument("--out", default="rl/datasets/champions.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total, sides_used = 0, 0
    with open(args.out, "w") as out:
        for path in sorted(glob.glob(os.path.join(args.replays, "episode-*-replay.json"))):
            try:
                replay = json.load(open(path))
            except Exception as e:
                print(f"skip {path}: {e}")
                continue
            for side in (0, 1):
                samples = extract_side(replay, side, args.min_reward)
                if samples:
                    sides_used += 1
                    for s in samples:
                        out.write(json.dumps(s) + "\n")
                    total += len(samples)
            rewards = [s.get("reward") for s in replay["steps"][-1]]
            print(f"{os.path.basename(path)}: rewards={rewards} -> cumulative {total} samples")
    print(f"\nwrote {total} champion day-decisions from {sides_used} agent-sides -> {args.out}")


if __name__ == "__main__":
    main()
