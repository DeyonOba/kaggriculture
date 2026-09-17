"""Make local games 2-3x faster by removing kaggle_environments plumbing.

The compiled-language question has a measured answer, and it is "no". A
cProfile of one champ-h8 vs ahmed game (24.85 s under the profiler):

    copy.deepcopy                       13.13 s   53%   <- harness
    structify                            7.36 s   30%   <- harness (nested)
    jsonschema validate/check_schema      1.27 s    5%   <- harness
    both agents' own act()               ~2.5  s   10%
    kaggriculture interpreter itself     small

Over two thirds of a game is `kaggle_environments` cloning and validating
state; the game logic and our agent are a minority of it. Rewriting the
scheduler in C or Rust would therefore buy almost nothing, and cannot ship
anyway -- Kaggle submissions are a Python `main.py`. The real compute lever is
the harness, and it is pure Python.

Three overheads, all skippable for a LOCAL run where we trust both agents:

1. `Environment.__get_shared_state` full-`deepcopy`s the state per agent per
   step. `deepcopy` pays reduce/memo/dispatch cost on every node; the state is
   plain JSON-shaped dicts/lists/scalars, so a direct recursive clone is much
   cheaper and gives the same isolation.
2. `process_schema` runs `jsonschema.validate` on every action of every step,
   and `check_schema` re-validates the SCHEMA ITSELF 1,448 times a game. The
   schema does not change.
3. `structify` rebuilds the whole state into `Struct` objects twice per step;
   `Struct.__init__` writes every entry into both `dict` and `__dict__`.

What this does NOT change: the interpreter, the RNG, the agents, or any
observation content. `verify` asserts that -- identical final rewards on every
seed, else the patch is rejected. Nothing here touches a submitted bundle.

    PYTHONPATH=. uv run python tools/fast_env.py --verify 8
    # in a script:  import tools.fast_env as fast_env; fast_env.install()
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_INSTALLED = False


def install(skip_validation=True):
    """Patch kaggle_environments in place. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from kaggle_environments import core as kcore
    from kaggle_environments import utils as kutils
    Struct = kutils.Struct

    # --- 1. fast JSON-shaped clone, replacing copy.deepcopy ------------------
    def clone(o):
        t = type(o)
        if t is Struct:
            # Struct.__init__ drops "items" to keep dict.items() working; keep
            # that behaviour identical rather than constructing Structs by hand.
            return Struct(**{k: clone(v) for k, v in o.items()})
        if t is dict:
            return {k: clone(v) for k, v in o.items()}
        if t is list:
            return [clone(v) for v in o]
        return o

    def get_shared_state(self, position):
        # Same update_props walk as upstream, over a cheap clone instead of a
        # deepcopy. Hidden fields are still deleted, shared fields still point
        # at seat 0 -- so agents see exactly what they saw before.
        schema_props = self._Environment__state_schema.properties
        shared = self.state[0]

        def update_props(shared_state, state, props):
            for k, prop in props.items():
                if kutils.get(prop, bool, path=["hidden"], fallback=False):
                    if k in state:
                        del state[k]
                elif kutils.get(prop, bool, path=["shared"], fallback=False):
                    state[k] = shared_state[k]
                elif kutils.has(prop, dict, path=["properties"]):
                    update_props(shared_state[k], state[k], prop["properties"])
            return state

        return update_props(shared, clone(self.state[position]), schema_props)

    kcore.Environment._Environment__get_shared_state = get_shared_state

    # --- 2. structify without the double write ------------------------------
    _orig_structify = kutils.structify

    def structify(o):
        t = type(o)
        if t is list:
            return [structify(v) for v in o]
        if t is dict or t is Struct:
            return Struct(**{k: structify(v) for k, v in o.items()})
        return o

    kutils.structify = structify
    kcore.structify = structify

    # --- 2b. don't re-clone the state the interpreter just returned ---------
    # `__loop_through_interpreter` structifies TWICE per step:
    #
    #     args = [structify(state), self, logs]                  # clone A
    #     new_state = structify(self.interpreter(*args[:n]))     # clone B
    #
    # Clone A is load-bearing -- `env.steps` keeps the previous state, so the
    # interpreter must not mutate it. Clone B is not: the interpreter mutates
    # and returns clone A, which is already a Struct tree and is about to
    # become `self.state` anyway.
    #
    # The one thing clone B really does is resync attribute access. `Struct`
    # mirrors entries into `__dict__`, but only `__setattr__` keeps the two
    # sides in step -- an interpreter writing `farm["money"] = x` updates the
    # dict and leaves `farm.money` stale, and upstream only gets away with it
    # because structify rebuilds every Struct from its dict side. So instead of
    # rebuilding, walk the tree, promote any plain dict/list the interpreter
    # introduced, and refresh `__dict__` in place.
    def upgrade(o):
        t = type(o)
        if t is Struct:
            for k, v in list(o.items()):
                nv = upgrade(v)
                if nv is not v:
                    dict.__setitem__(o, k, nv)
            d = o.__dict__
            if "items" in o:
                # Struct.__init__ drops "items" so dict.items() keeps working.
                d.update({k: v for k, v in o.items() if k != "items"})
            else:
                d.update(o)
            return o
        if t is list:
            for i, v in enumerate(o):
                nv = upgrade(v)
                if nv is not v:
                    o[i] = nv
            return o
        if t is dict:
            return Struct(**{k: upgrade(v) for k, v in o.items()})
        return o

    def loop_through_interpreter(self, state, logs):
        args = [structify(state), self, logs]
        new_state = upgrade(
            self.interpreter(*args[: self.interpreter.__code__.co_argcount]))
        new_state[0].observation.step = 0 if self.done else len(self.steps)
        status_enum = self._Environment__state_schema.properties.status.enum
        for index, agent in enumerate(new_state):
            if index < len(logs) and "duration" in logs[index]:
                overage = max(0, logs[index]["duration"] - self.configuration.actTimeout)
                agent.observation.remainingOverageTime -= overage
            if agent.status not in status_enum:
                self.debug_print(f"Invalid Action: {agent.status}")
                agent.status = "INVALID"
            if agent.status in ["ERROR", "INVALID", "TIMEOUT"]:
                agent.reward = None
        return new_state

    kcore.Environment._Environment__loop_through_interpreter = loop_through_interpreter

    # --- 3. skip per-action jsonschema validation ---------------------------
    if skip_validation:
        _orig_process = kutils.process_schema

        def process_schema(schema, data, use_default=True):
            # Keep default-filling (the interpreter relies on it); drop the
            # validate() call and the schema re-check that goes with it.
            if use_default is True:
                data = kutils.default_schema(schema, clone(data))
            return None, data

        kutils.process_schema = process_schema
        kcore.process_schema = process_schema

    _INSTALLED = True


def _play(agent, opp, seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    env.run([agent, opp])
    last = env.steps[-1]
    return last[0]["reward"], last[1]["reward"]


def verify(n, agent, opp):
    """Run each seed patched and unpatched IN SEPARATE PROCESSES, compare.

    Separate processes because `install` is a one-way monkeypatch; comparing
    inside one process would measure a half-patched harness.
    """
    import json
    import subprocess

    seeds = [61000 + k for k in range(n)]
    out = {}
    for tag, flag in (("stock", "0"), ("fast", "1")):
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, flag, agent, opp, json.dumps(seeds)],
            cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": REPO},
        )
        if r.returncode != 0:
            sys.exit(f"{tag} run failed:\n{(r.stderr or r.stdout)[-2000:]}")
        out[tag] = (json.loads(r.stdout.strip().splitlines()[-1]), time.time() - t0)

    (a, ta), (b, tb) = out["stock"], out["fast"]
    print(f"{'seed':>8} {'stock (me, opp)':>26} {'fast (me, opp)':>26}   match")
    allok = True
    for i, sd in enumerate(seeds):
        ok = a[i] == b[i]
        allok &= ok
        print(f"{sd:>8} {str(tuple(a[i])):>26} {str(tuple(b[i])):>26}   "
              f"{'yes' if ok else 'NO'}")
    print(f"\nstock {ta:6.1f}s   fast {tb:6.1f}s   speedup {ta / tb:.2f}x "
          f"({ta / len(seeds):.2f} -> {tb / len(seeds):.2f} s/game)")
    print(f"\nverdict: {'IDENTICAL — safe to use' if allok else 'MISMATCH — do not use'}")
    return allok


_CHILD = """
import json, sys, os
sys.path.insert(0, os.getcwd())
flag, agent, opp, seeds = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
if flag == "1":
    import tools.fast_env as fe
    fe.install()
from kaggle_environments import make
out = []
for sd in seeds:
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd}, debug=False)
    env.run([agent, opp])
    out.append([env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]])
print(json.dumps(out))
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=8, help="seeds to cross-check")
    ap.add_argument("--agent", default="arena/agents/champ-h8/main.py")
    ap.add_argument("--opp", default="arena/agents/pub-ahmed-v38/main.py")
    args = ap.parse_args()
    sys.exit(0 if verify(args.verify, args.agent, args.opp) else 1)
