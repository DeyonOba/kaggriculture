"""execnet: a learned per-unit action model of the executor, cloned from the
top players' recorded games (rl/execnet/dataset.py -> train.py -> policy.py).
The day plan and the market orders stay with the planner; the model only
replaces scheduler.assign's action for a unit. See docs/action_modeling_research.md.
"""
