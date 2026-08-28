"""Scheduler invariants over a full simulated episode.

The invariant that matters most: PLANT ops emitted in one turn never exceed
the seeds held (the engine's atomic-PLANT rule voids ALL of that crop's
plants otherwise).
"""

from kaggle_environments import make

from kagent.gamestate import Memory, View
from kagent.policy import HeuristicPolicy


def test_full_episode_invariants():
    mem = Memory()
    pol = HeuristicPolicy()
    violations = []

    def agent(obs):
        view = View(obs)
        mem.observe(view)
        action = pol.act(view, mem)
        ops = [action["farmer"]] + action["hands"]
        plants = {}
        for op in ops:
            if op and op[0] == "PLANT":
                plants[op[1]] = plants.get(op[1], 0) + 1
        for crop, n in plants.items():
            if n > view.seeds.get(crop, 0):
                violations.append((view.step, crop, n, view.seeds.get(crop, 0)))
        if len(action["hands"]) != len(view.me.units) - 1:
            violations.append((view.step, "hands-size", len(action["hands"])))
        if len(action["market"]) > 10:
            violations.append((view.step, "market-cap", len(action["market"])))
        return action

    env = make("kaggriculture", configuration={"seed": 11}, debug=False)
    env.run([agent, "starter"])
    assert env.steps[-1][0].status == "DONE"
    assert not violations, violations[:10]
