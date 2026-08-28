"""SafeAgent must never raise and always return a schema-valid action."""

from kagent.policy import SafeAgent, HeuristicPolicy


def test_garbage_observation_returns_pass():
    agent = SafeAgent(HeuristicPolicy())
    action = agent.act_safe({"player": 0})
    assert action == {"farmer": ["PASS"], "hands": [], "market": []}


def test_none_observation_returns_pass():
    agent = SafeAgent(HeuristicPolicy())
    action = agent.act_safe(None)
    assert action["farmer"] == ["PASS"]
    assert action["market"] == []


def test_hands_sized_from_farm():
    agent = SafeAgent(HeuristicPolicy())
    obs = {
        "player": 0,
        "farms": [
            {"money": 0, "tiles": "garbage", "farmer": [0, 0], "hands": [[1, 1], [2, 2]]}
        ],
    }
    action = agent.act_safe(obs)
    assert len(action["hands"]) == 2
