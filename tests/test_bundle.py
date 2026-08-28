"""The built main.py must behave exactly like the Kaggle loader expects."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO, "main.py")


def _build():
    subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "build_submission.py")],
        check=True,
        capture_output=True,
    )


def _load_last_callable():
    # Mirrors kaggle_environments.agent get_last_callable: exec the file and
    # take the LAST callable defined in the module namespace.
    env = {}
    with open(MAIN) as f:
        exec(f.read(), env)
    callables = [v for v in env.values() if callable(v)]
    return callables[-1]


def test_last_callable_is_agent():
    _build()
    fn = _load_last_callable()
    assert fn.__name__ == "agent"


def test_two_bundles_are_isolated():
    _build()
    a, b = _load_last_callable(), _load_last_callable()
    obs = {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            {
                "money": 3000.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            }
        ]
        * 2,
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    before = {m for m in sys.modules if m.startswith("kagent")}
    ra, rb = a(obs), b(obs)
    assert set(ra) == {"farmer", "hands", "market"}
    assert set(rb) == {"farmer", "hands", "market"}
    # The bundle must not add anything to shared sys.modules.
    after = {m for m in sys.modules if m.startswith("kagent")}
    assert after == before


def test_full_episode_no_timeouts():
    _build()
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"seed": 5}, debug=False)
    env.run([MAIN, MAIN])
    for s in env.steps[-1]:
        assert s.status == "DONE"
        assert s.reward is not None
        assert s.reward > 3000  # both sides must actually profit in a mirror
