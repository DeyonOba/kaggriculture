"""Bundle kagent/ into a single-file main.py submission.

Isolation matters: two agents built this way can be exec'd in the SAME
process (local mirror matches; Kaggle's self-play validation episode), so the
bundle must not touch sys.modules or any shared global state. Instead, all
module sources are exec'd into one private dict, with intra-package imports
stripped and module names (econ, planner, ...) aliased to a SimpleNamespace
over that dict — every `econ.price`-style reference resolves at call time.

Usage:
    python tools/build_submission.py                # -> main.py
    python tools/build_submission.py --freeze v1    # -> arena/agents/v1/main.py too
"""

import argparse
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = [
    "constants",
    "econ",
    "gamestate",
    "planner",
    "scheduler",
    "market",
    "features",
    "nnpolicy",
    "boardstate",
    "mpcpolicy",
    "banditpolicy",
    "weights",
    "executor2",
    "sizing",
    "opp_profile",
    "selltiming",
    "policy",
]

# policy.py re-declares the weights module's four names on purpose (the
# "set visible everywhere" fallback); every other top-level name must be
# unique, because the bundle execs all modules into ONE dict and a later
# module's assignment silently rebinds an earlier one's (executor2.ANIMALS
# rebound constants.ANIMALS to a tuple and every plan crashed, 2026-09-30).
SHARED_NAMES = {"BANDIT_RULES", "CHAMPION_PLAN", "EXEC_RULES", "MPC_ENABLED"}
INTRA_IMPORT = re.compile(r"^\s*(from kagent[.\w]* import|import kagent)")

TEMPLATE = '''\
"""Kaggriculture agent (auto-generated bundle -- edit kagent/, not this file)."""

import types as _types

_SOURCES = {sources}

_G = {{"__name__": "kagent_bundle"}}
for _name, _src in _SOURCES:
    exec(compile(_src, "kagent/" + _name + ".py", "exec"), _G)
_NS = _types.SimpleNamespace(**_G)
for _name, _ in _SOURCES:
    _G[_name] = _NS
_IMPL = _G["agent"]


def agent(observation, configuration=None):
    return _IMPL(observation, configuration)
'''


def _strip_intra_imports(text):
    out = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if skipping:
            if ")" in line:
                skipping = False
            continue
        if INTRA_IMPORT.match(line):
            if "(" in line and ")" not in line:
                skipping = True
            continue
        out.append(line)
    return "".join(out)


def build(out=None, exec_rules=None, knobs=None, no_bandit=False):
    """`exec_rules`: replace the weights module's EXEC_RULES literal for THIS
    bundle only (the tree's kagent/weights.py is untouched), so a version
    with an extra executor flag is built the way it will be played -- the
    bundle applies EXEC_RULES at load (kagent/policy.py). `knobs`: {name:
    value} overrides written into the inlined planner's KNOB_DEFAULTS for
    THIS bundle only (an opening knob such as lead_opening must hold from
    day 0, before the day-6 knob layer decides anything). `no_bandit`: write
    BANDIT_RULES = None so the bundle plays the plain champion plan with the
    (patched) KNOB_DEFAULTS throughout -- the composition a `dev:theta` gate
    arm measures. `out`: where to write; default is the repo's main.py."""
    sources = []
    seen = set()
    for mod in MODULES:
        with open(os.path.join(REPO, "kagent", f"{mod}.py")) as f:
            text = _strip_intra_imports(f.read())
        if mod == "planner" and knobs:
            # KNOB_DEFAULTS is a dict literal; a key may share its line with
            # others ("cows": 8, "sheep": 8, "geese": 0,), so match the key
            # anywhere and replace only its value
            for k, v in knobs.items():
                text, n = re.subn(r'("%s"\s*:\s*)[^,}\n]+' % re.escape(k), lambda m: '%s%r' % (m.group(1), v),
                                  text, count=1)
                if n != 1:
                    raise SystemExit(f"knob {k!r} not found in kagent/planner.py KNOB_DEFAULTS")
        if mod == "weights" and no_bandit:
            text, n = re.subn(r"^BANDIT_RULES\s*=\s*\{.*$", "BANDIT_RULES = None", text, count=1, flags=re.M)
            if n != 1:
                raise SystemExit("BANDIT_RULES literal not found in kagent/weights.py")
        if mod == "weights" and exec_rules is not None:
            from kagent import scheduler as _sched
            bad = [r for r in exec_rules if r not in _sched.EXEC_FLAGS]
            if bad:
                raise SystemExit(f"unknown executor flag(s) {bad}; known: {_sched.EXEC_FLAGS}")
            text, n = re.subn(r"^EXEC_RULES\s*=\s*\[[^\]]*\]", "EXEC_RULES = " + repr(list(exec_rules)),
                              text, count=1, flags=re.M)
            if n != 1:
                raise SystemExit("EXEC_RULES literal not found in kagent/weights.py")
        for m in re.finditer(r"^(?:def |class |)(\w+)(?:\s*[(:=]|\s*$)", text, re.M):
            name = m.group(1)
            if name.startswith("_") or name in ("import", "from", "if", "for", "while", "try", "with", "else", "elif", "return", "pass", "raise", "except", "finally", "assert", "del", "global", "yield"):
                continue
            if name in seen and name not in SHARED_NAMES:
                raise SystemExit(f"top-level name collision across modules: {name} (in {mod})")
            seen.add(name)
        sources.append((mod, text))

    src_literal = "[\n" + "".join(
        f"    ({name!r}, {text!r}),\n" for name, text in sources
    ) + "]"
    text = TEMPLATE.format(sources=src_literal)

    out = out or os.path.join(REPO, "main.py")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out} ({len(text)} bytes)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default=None, help="also copy to arena/agents/<name>/main.py")
    ap.add_argument("--out", default=None, help="write the bundle here instead of the repo's main.py")
    ap.add_argument("--exec", default=None,
                    help="comma-separated EXEC_RULES for this bundle (default: kagent/weights.py's)")
    ap.add_argument("--no-bandit", action="store_true",
                    help="BANDIT_RULES = None for this bundle: the plain champion plan with KNOB_DEFAULTS, as a dev:theta arm plays")
    ap.add_argument("--knobs", default=None,
                    help="comma-separated name=value KNOB_DEFAULTS overrides for this bundle (e.g. lead_opening=1)")
    args = ap.parse_args()
    rules = [r for r in args.exec.split(",") if r.strip()] if args.exec else None
    knobs = None
    if args.knobs:
        import json as _json
        knobs = {}
        for kv in args.knobs.split(","):
            k, _, v = kv.partition("=")
            knobs[k.strip()] = _json.loads(v) if v.strip() else 1
    out = build(out=args.out, exec_rules=rules, knobs=knobs, no_bandit=args.no_bandit)
    if args.freeze:
        dst_dir = os.path.join(REPO, "arena", "agents", args.freeze)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(out, os.path.join(dst_dir, "main.py"))
        print(f"froze {args.freeze}")


if __name__ == "__main__":
    main()
