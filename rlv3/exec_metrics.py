"""The four placement metrics the scorecard computes but never prints.

`tools/tile_metrics.py` already derives `layout_day15` -- animal_shed_dist,
plant_shed_dist, wheat_adjacent_to_animal_share -- and `visits_ge2_share`,
and `tools/agent_scorecard.py` never puts them on a row. They are the four
numbers every rlv3 placement lever is judged on before an evaluation is
spent, so this module prints them beside the leaders.

Leader values are from docs/results/top3_tile_metrics.txt. READ THE SIGNS:

  wheat_adjacent_to_animal_share  ours 0.09 vs 0.46 / 0.36 / 0.42 -- a real
      4-7x gap, and the ONLY layout metric where we are genuinely off all
      three leaders. This is what pl_wheat_adj targets.
  visits_ge2_share                ours 0.32 vs 0.60 / 0.52 / 0.54 -- the
      partner metric to chores_per_visit, which co_chain_tile targets.
  animal_shed_dist                ours 2.45 vs 3.53 / 2.07 / 2.10. Leader
      MEAN 2.57, and TWO OF THREE leaders sit CLOSER to the shed than we
      do. There is NO GAP here. It is printed so the next person can see
      that for themselves rather than re-deriving "move the animals away
      from the shed" from the one leader who does.
  plant_shed_dist                 ours 4.38 vs 4.17 / 4.72 / 4.73 -- level.
"""

from __future__ import annotations

import statistics as st
import sys

from tools.tile_metrics import analyse, load_trace

# docs/results/top3_tile_metrics.txt
LEADERS = {
    "mmpq": {
        "wheat_adjacent_to_animal_share": 0.46, "animal_shed_dist": 3.53,
        "plant_shed_dist": 4.17, "visits_ge2_share": 0.60,
        "chores_per_visit": 2.076, "chores_per_visit_animal": 2.77,
        "chores_per_visit_crop": 1.76, "moves_per_productive": 0.71,
        "replant_within_1_step": 0.78, "one_time_units_total": 888.5,
        "plant_ops": 266.1, "idle_share_mean": 0.065, "weed_tile_days": 61.2,
    },
    "boey": {
        "wheat_adjacent_to_animal_share": 0.36, "animal_shed_dist": 2.07,
        "plant_shed_dist": 4.72, "visits_ge2_share": 0.52,
        "chores_per_visit": 1.898, "chores_per_visit_animal": 1.82,
        "chores_per_visit_crop": 1.95, "moves_per_productive": 0.83,
        "replant_within_1_step": 0.86, "one_time_units_total": 1043.1,
        "plant_ops": 275.6, "idle_share_mean": 0.040, "weed_tile_days": 3.4,
    },
    "dsm": {
        "wheat_adjacent_to_animal_share": 0.42, "animal_shed_dist": 2.10,
        "plant_shed_dist": 4.73, "visits_ge2_share": 0.54,
        "chores_per_visit": 1.913, "chores_per_visit_animal": 1.98,
        "chores_per_visit_crop": 1.85, "moves_per_productive": 0.81,
        "replant_within_1_step": 0.88, "one_time_units_total": 1076.7,
        "plant_ops": 270.7, "idle_share_mean": 0.037, "weed_tile_days": 2.9,
    },
}
# champ-h10, same file / docs/results/scorecard_champ_h10.txt
OURS = {
    "wheat_adjacent_to_animal_share": 0.09, "animal_shed_dist": 2.45,
    "plant_shed_dist": 4.38, "visits_ge2_share": 0.32,
    "chores_per_visit": 1.453, "chores_per_visit_animal": 1.617,
    "chores_per_visit_crop": 1.267, "moves_per_productive": 1.233,
    "replant_within_1_step": 0.548, "one_time_units_total": 388.0,
    "plant_ops": 150.0, "idle_share_mean": 0.124, "weed_tile_days": 45.0,
}

# name, key, format, target: "up" / "down" / None (approach the leader mean)
ROWS = [
    ("-- placement (never printed by the scorecard) --", None, None, None),
    ("wheat beside an animal (day 15)", ("layout_day15", "wheat_adjacent_to_animal_share"), "{:.3f}", "up"),
    ("animal shed distance (day 15)", ("layout_day15", "animal_shed_dist"), "{:.2f}", None),
    ("plant shed distance (day 15)", ("layout_day15", "plant_shed_dist"), "{:.2f}", None),
    ("visits with >= 2 chores", "visits_ge2_share", "{:.3f}", "up"),
    ("-- executor --", None, None, None),
    ("chores per visit", "chores_per_visit", "{:.3f}", "up"),
    ("  on animal tiles", "chores_per_visit_animal", "{:.3f}", "up"),
    ("  on crop tiles", "chores_per_visit_crop", "{:.3f}", "up"),
    ("moves per productive op", "moves_per_productive", "{:.3f}", "down"),
    ("replant within 1 step", "replant_within_1_step", "{:.3f}", "up"),
    ("-- tripwires (must not move the wrong way) --", None, None, None),
    ("one-time units total", "one_time_units_total", "{:,.0f}", "up"),
    ("plant ops", "plant_ops", "{:,.0f}", "up"),
    ("idle share at dawn", "idle_share_mean", "{:.3f}", "down"),
    ("weed tile-days", "weed_tile_days", "{:,.0f}", "down"),
    ("final money", "final_money", "{:,.0f}", "up"),
    ("OPPONENT money", "opp_money", "{:,.0f}", "down"),
]


def _get(m, key):
    if isinstance(key, tuple):
        sub = m.get(key[0]) or {}
        return sub.get(key[1])
    return m.get(key)


def placement_row(trace_paths):
    """-> the mean metric dict over `trace_paths` (tools/tile_metrics traces)."""
    ms = [analyse(*load_trace(p)) for p in trace_paths]
    if not ms:
        return {}
    out = {"n_games": len(ms)}
    for _, key, _, _ in ROWS:
        if key is None:
            continue
        vals = [v for v in (_get(m, key) for m in ms) if v is not None]
        name = key if isinstance(key, str) else key[1]
        out[name] = st.mean(vals) if vals else None
    out["layout_day15"] = {
        k: out.get(k) for k in
        ("wheat_adjacent_to_animal_share", "animal_shed_dist", "plant_shed_dist")
    }
    return out


def print_table(rows, out=sys.stdout):
    """`rows` is {label: metric dict} -- e.g. {"arm": ..., "champ-h10": ...}."""
    names = list(rows)
    cols = names + ["mmpq", "boey", "dsm"]
    w = max([len(r[0]) for r in ROWS] + [10])
    print(f"{'metric':<{w}}" + "".join(f"{c:>12}" for c in cols), file=out)
    for label, key, fmt, direction in ROWS:
        if key is None:
            print(f"{label:<{w}}", file=out)
            continue
        cells = []
        for c in cols:
            m = LEADERS[c] if c in LEADERS else rows[c]
            name = key if isinstance(key, str) else key[1]
            v = _get(m, key)
            if v is None:
                v = m.get(name)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        mark = {"up": " ^", "down": " v"}.get(direction, "  ")
        print(f"{label:<{w}}" + "".join(f"{c:>12}" for c in cells) + mark, file=out)
    print("\n^ higher is better, v lower is better, blank = approach the leader MEAN.",
          file=out)
    print("animal_shed_dist has NO gap: leader mean 2.57 against our 2.45, and Boey "
          "(2.07)\nand DSM (2.10) both sit CLOSER to the shed than we do.", file=out)
