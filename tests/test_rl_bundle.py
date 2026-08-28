"""A bundle carrying neural weights must load them and still play a clean
episode (with the heuristic as SafeAgent fallback)."""

import base64
import os
import random
import struct
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PY = os.path.join(REPO, "kagent", "weights.py")
MAIN = os.path.join(REPO, "main.py")


def _random_weights_data(n_features):
    from kagent import features

    rng = random.Random(0)
    sizes = [
        [n_features, 8],
        [8, 8],
        [8, features.N_ACTION_LOGITS],
        [8, 1],
    ]
    flat = []
    for n_in, n_out in sizes:
        flat.extend(rng.uniform(-0.1, 0.1) for _ in range((n_in + 1) * n_out))
    raw = struct.pack("<%df" % len(flat), *flat)
    return {"sizes": sizes, "b85": base64.b85encode(raw).decode("ascii")}


def test_bundle_with_weights_plays_clean_episode():
    from kagent import features
    from kagent.gamestate import Memory, View
    from rl import rollout

    probe = rollout.run_episode(
        {"weights_path": None, "seed": 2, "side": 0, "opponent": "starter", "mode": "bc"}
    )
    n_features = len(probe["records"][0]["x"])

    with open(WEIGHTS_PY) as f:
        original = f.read()
    try:
        data = _random_weights_data(n_features)
        with open(WEIGHTS_PY, "w") as f:
            f.write(f"WEIGHTS_DATA = {data!r}\n")
        subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "build_submission.py")],
            check=True,
            capture_output=True,
        )
        env = {}
        with open(MAIN) as f:
            exec(f.read(), env)
        agent = [v for v in env.values() if callable(v)][-1]
        assert agent.__name__ == "agent"

        from kaggle_environments import make

        game = make("kaggriculture", configuration={"seed": 2}, debug=False)
        game.run([MAIN, "starter"])
        final = game.steps[-1]
        assert final[0].status == "DONE"
        assert final[0].reward is not None
    finally:
        with open(WEIGHTS_PY, "w") as f:
            f.write(original)
        subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "build_submission.py")],
            check=True,
            capture_output=True,
        )
