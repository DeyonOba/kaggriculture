"""Registry of runnable agents for the arena.

Values are either a kaggle-environments builtin name ("starter", "random",
"pass") or a path to an agent .py file (resolved relative to the repo root).
Frozen versions live in arena/agents/<version>/main.py.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGENTS = {
    "starter": "starter",
    "random": "random",
    "pass": "pass",
    "demo": os.path.join(REPO_ROOT, "arena", "agents", "demo", "main.py"),
    "dev": os.path.join(REPO_ROOT, "main.py"),
}


def _discover_frozen():
    agents_dir = os.path.join(REPO_ROOT, "arena", "agents")
    if not os.path.isdir(agents_dir):
        return
    for name in sorted(os.listdir(agents_dir)):
        path = os.path.join(agents_dir, name, "main.py")
        if os.path.isfile(path):
            AGENTS.setdefault(name, path)


_discover_frozen()


def resolve(name):
    if name in AGENTS:
        return AGENTS[name]
    if os.path.isfile(name):
        return os.path.abspath(name)
    raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
